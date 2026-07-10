"""Fill in the missing Llama-3.2-1B evaluations only (OS-data perplexity is already done).

Distributed across all available GPUs with accelerate: each example (BLiMP/M-BLiMP/SIB-200/
MuBench row, perplexity text) is independent of every other one, so the dataset for each eval
is sharded across processes/GPUs and only the final correct-counts / NLL sums are reduced
across ranks -- one evaluation, split across N GPUs, not N separate jobs.

Currently missing per evaluation.md:
  - English : Test perplexity, BLiMP (67 tasks), SIB-200, MuBench
  - Hindi   : Test perplexity, M-BLiMP, SIB-200, MuBench
  - Telugu  : Test perplexity, SIB-200, MuBench (no M-BLiMP -- not available for Telugu)

Usage:
    accelerate launch --num_processes 4 eval_llama_missing.py
    accelerate launch --num_processes 4 eval_llama_missing.py --langs en hi
    accelerate launch --num_processes 4 eval_llama_missing.py --evals perplexity sib200
    accelerate launch --num_processes 4 eval_llama_missing.py --output results_llama_missing.json

    # still works single-GPU, no accelerate launch needed:
    python eval_llama_missing.py
"""

import argparse
import json
import math
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

MODEL_ID = "meta-llama/Llama-3.2-1B"
MAX_LEN = 1024
MAX_SEQ_LEN = 128
BATCH_SIZE = 32

LANGS = ["en", "hi", "te"]

SIB200_CONFIG = {"en": "eng_Latn", "hi": "hin_Deva", "te": "tel_Telu"}
SIB200_LABELS = [
    "science/technology", "travel", "politics", "sports",
    "health", "entertainment", "geography",
]

MUBENCH_DATASET_ID = "aialt/MuBench"
MUBENCH_LANG_SUFFIX = {"en": "_en", "hi": "_hi", "te": "_te"}
MUBENCH_TASKS_BASE = [
    "ARCChallengeDataset_local_template",
    "ARCEasyDataset_local_template",
    "BMLAMADataset_local_template",
    "GPQADataset_local_template",
    "HellaswagDataset_local_template",
    "MMLUDataset_local_template",
    "MMLUProDataset_local_template",
    "MNLIDataset_local_template",
    "SNLIDataset_local_template",
    "StoryClozeDataset_local_template",
    "TruthfulQADataset_local_template",
    "WinoGrandeDataset_local_template",
]

BLIMP_TASKS = [
    "adjunct_island", "anaphor_gender_agreement", "anaphor_number_agreement",
    "animate_subject_passive", "animate_subject_trans", "causative",
    "complex_NP_island",
    "coordinate_structure_constraint_complex_left_branch",
    "coordinate_structure_constraint_object_extraction",
    "determiner_noun_agreement_1", "determiner_noun_agreement_2",
    "determiner_noun_agreement_irregular_1", "determiner_noun_agreement_irregular_2",
    "determiner_noun_agreement_with_adj_2",
    "determiner_noun_agreement_with_adj_irregular_1",
    "determiner_noun_agreement_with_adj_irregular_2",
    "determiner_noun_agreement_with_adjective_1",
    "distractor_agreement_relational_noun", "distractor_agreement_relative_clause",
    "drop_argument", "ellipsis_n_bar_1", "ellipsis_n_bar_2",
    "existential_there_object_raising",
    "existential_there_quantifiers_1", "existential_there_quantifiers_2",
    "existential_there_subject_raising", "expletive_it_object_raising",
    "inchoative", "intransitive",
    "irregular_past_participle_adjectives", "irregular_past_participle_verbs",
    "irregular_plural_subject_verb_agreement_1",
    "irregular_plural_subject_verb_agreement_2",
    "left_branch_island_echo_question", "left_branch_island_simple_question",
    "matrix_question_npi_licensor_present",
    "npi_present_1", "npi_present_2",
    "only_npi_licensor_present", "only_npi_scope",
    "passive_1", "passive_2",
    "principle_A_c_command", "principle_A_case_1", "principle_A_case_2",
    "principle_A_domain_1", "principle_A_domain_2", "principle_A_domain_3",
    "principle_A_reconstruction",
    "regular_plural_subject_verb_agreement_1", "regular_plural_subject_verb_agreement_2",
    "sentential_negation_npi_licensor_present", "sentential_negation_npi_scope",
    "sentential_subject_island",
    "superlative_quantifiers_1", "superlative_quantifiers_2",
    "tough_vs_raising_1", "tough_vs_raising_2", "transitive", "wh_island",
    "wh_questions_object_gap", "wh_questions_subject_gap",
    "wh_questions_subject_gap_long_distance",
    "wh_vs_that_no_gap", "wh_vs_that_no_gap_long_distance",
    "wh_vs_that_with_gap", "wh_vs_that_with_gap_long_distance",
]

TEST_SOURCE = {
    "en": ("text", {"data_files": {"test": "hf://datasets/BabyLM-community/BabyLM-Test/*.test"}, "split": "test"}),
    "hi": ("pulipakav-1/translated-babylm-hindi", {"split": "test", "streaming": True}),
    "te": ("pulipakav-1/translated-babylm-telugu", {"split": "test", "streaming": True}),
}

# English-only "Filtered" test set: the 4 subsections used for the bilingual eng-hin/eng-tel
# GPT-BERT/GPT-2 Perplexity tables (bnc_spoken/open_subtitles/simple_wiki/switchboard). No
# equivalent exists for Hindi/Telugu -- translated-babylm-{hi,te} has no per-subsection files.
EN_FILTERED_SOURCES = ["bnc_spoken", "open_subtitles", "simple_wiki", "switchboard"]

# Task-level sharding (BLiMP/MuBench) is uneven -- some tasks (e.g. MuBench's GPQA) take far
# longer than others, so a fast rank can sit waiting at a collective well past NCCL's default
# 10-minute timeout while a slow rank finishes its heavier task. Use a generous timeout instead.
accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=4))])
DEVICE = accelerator.device
RANK = accelerator.process_index
WORLD_SIZE = accelerator.num_processes
IS_MAIN = accelerator.is_main_process


# ── Sharding / reduction helpers ────────────────────────────────────────────────

def shard(items):
    """Split a list so each rank gets a disjoint ~1/WORLD_SIZE slice."""
    items = list(items)
    return items[RANK::WORLD_SIZE]


def reduce_sum_pair(a, b):
    """All-reduce-sum two running totals (e.g. correct/total, or nll/token_count) across ranks."""
    if WORLD_SIZE > 1:
        t = torch.tensor([a, b], dtype=torch.float64, device=DEVICE)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        a, b = t.tolist()
    return a, b


def gather_merge_dicts(d):
    """Gather a {key: value} dict from every rank and merge into one (every rank gets the result)."""
    if WORLD_SIZE > 1:
        gathered = [None] * WORLD_SIZE
        dist.all_gather_object(gathered, d)
        merged = {}
        for gd in gathered:
            merged.update(gd)
        return merged
    return d


def load_dataset_retry(*args, retries=5, base_delay=5, **kwargs):
    """load_dataset wrapper with retry/backoff -- the Hub connection is flaky under concurrent
    load from multiple ranks, and an unhandled ConnectionError otherwise kills the whole job."""
    last_exc = None
    for attempt in range(retries):
        try:
            return load_dataset(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if attempt < retries - 1:
                wait = base_delay * (attempt + 1)
                print(f"  [rank {RANK}] load_dataset{args} failed ({e}); retrying in {wait}s "
                      f"({attempt + 1}/{retries})", flush=True)
                time.sleep(wait)
    raise last_exc


# ── Data loading ─────────────────────────────────────────────────────────────────

def load_test_texts(lang):
    name, kwargs = TEST_SOURCE[lang]
    ds = load_dataset_retry(name, **kwargs)
    if kwargs.get("streaming"):
        return [row["text"] for row in ds if row.get("text", "").strip()]
    return [row["text"] for row in ds if row.get("text")]


def load_english_filtered_test_texts():
    data_files = [f"hf://datasets/BabyLM-community/BabyLM-Test/{name}.test" for name in EN_FILTERED_SOURCES]
    ds = load_dataset_retry("text", data_files={"test": data_files}, split="test")
    return [row["text"] for row in ds if row.get("text")]


# ── Scoring ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def score_causal(model, tokenizer, text):
    ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
    ids = ids[:, -MAX_LEN:]
    if ids.size(1) < 2:
        return float("-inf")
    logits = model(ids).logits
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    token_ll = log_probs[0].gather(-1, ids[0, 1:].unsqueeze(-1)).squeeze(-1)
    return token_ll.sum().item()


@torch.no_grad()
def score_causal_normalized(model, tokenizer, text):
    ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
    ids = ids[:, -MAX_LEN:]
    n = ids.size(1) - 1
    if n < 1:
        return float("-inf")
    logits = model(ids).logits
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    token_ll = log_probs[0].gather(-1, ids[0, 1:].unsqueeze(-1)).squeeze(-1)
    return token_ll.sum().item() / n


def compute_perplexity(model, tokenizer, texts):
    my_texts = shard(texts)
    total_nll, total_tokens = 0.0, 0
    for i in tqdm(range(0, len(my_texts), BATCH_SIZE), desc="  Perplexity", leave=False, disable=not IS_MAIN):
        batch = my_texts[i:i + BATCH_SIZE]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_SEQ_LEN)
        input_ids = enc["input_ids"].to(DEVICE)
        attn_mask = enc["attention_mask"].to(DEVICE)
        with torch.no_grad():
            logits = model(input_ids, attention_mask=attn_mask).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attn_mask[:, 1:].contiguous().float()
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_ll = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        total_nll += -(token_ll * shift_mask).sum().item()
        total_tokens += shift_mask.sum().item()
    total_nll, total_tokens = reduce_sum_pair(total_nll, total_tokens)
    return math.exp(total_nll / total_tokens) if total_tokens > 0 else float("inf")


def eval_blimp(model, tokenizer):
    my_tasks = shard(BLIMP_TASKS)
    local_results = {}
    for task in tqdm(my_tasks, desc="  BLiMP", disable=not IS_MAIN):
        ds = load_dataset_retry("BabyLM-community/BabyLM-BLIMP-Filtered", task, split="train")
        correct = 0
        for row in ds:
            good = score_causal(model, tokenizer, row["sentence_good"])
            bad = score_causal(model, tokenizer, row["sentence_bad"])
            if good > bad:
                correct += 1
        local_results[task] = round(correct / len(ds), 4)
    results = gather_merge_dicts(local_results)
    if results:
        results["macro_avg"] = round(sum(results.values()) / len(results), 4)
    return results


def eval_mblimp(model, tokenizer):
    ds = load_dataset_retry("jumelet/multiblimp", "hin", split="train")
    my_rows = shard(list(ds))
    correct, total = 0, 0
    for row in tqdm(my_rows, desc="  M-BLiMP", leave=False, disable=not IS_MAIN):
        good = score_causal(model, tokenizer, row["sen"])
        bad = score_causal(model, tokenizer, row["wrong_sen"])
        if good > bad:
            correct += 1
        total += 1
    correct, total = reduce_sum_pair(correct, total)
    return {"accuracy": round(correct / total, 4) if total > 0 else None}


def eval_sib200(model, tokenizer, lang):
    ds = load_dataset_retry("Davlan/sib200", SIB200_CONFIG[lang], split="test")
    my_rows = shard(list(ds))
    correct, total = 0, 0
    for row in tqdm(my_rows, desc="  SIB-200", leave=False, disable=not IS_MAIN):
        text = row["text"]
        best_score, best_label = float("-inf"), None
        for label in SIB200_LABELS:
            score = score_causal_normalized(model, tokenizer, f"{text}\nTopic: {label}")
            if score > best_score:
                best_score, best_label = score, label
        if best_label == row["category"]:
            correct += 1
        total += 1
    correct, total = reduce_sum_pair(correct, total)
    return {"accuracy": round(correct / total, 4) if total > 0 else None}


def eval_mubench(model, tokenizer, lang):
    suffix = MUBENCH_LANG_SUFFIX[lang]
    my_tasks = shard(MUBENCH_TASKS_BASE)
    local_results = {}
    for task_base in tqdm(my_tasks, desc="  MuBench", disable=not IS_MAIN):
        task_config = task_base + suffix
        task_key = task_base.replace("Dataset_local_template", "")
        try:
            ds = load_dataset_retry(MUBENCH_DATASET_ID, task_config, split="test")
        except Exception as e:
            tqdm.write(f"    SKIP {task_config}: {e}")
            continue
        correct = 0
        for row in ds:
            prompt, choices, label = row["prompt"], row["choices"], row["label"]
            scores = []
            for choice in choices:
                text = prompt + choice
                n = max(len(tokenizer.encode(choice, add_special_tokens=False)), 1)
                scores.append(score_causal(model, tokenizer, text) / n)
            if scores.index(max(scores)) == label:
                correct += 1
        local_results[task_key] = round(correct / len(ds), 4)
    results = gather_merge_dicts(local_results)
    if results:
        results["avg"] = round(sum(results.values()) / len(results), 4)
    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+", choices=LANGS, default=LANGS)
    parser.add_argument("--evals", nargs="+",
                        choices=["perplexity", "blimp", "mblimp", "sib200", "mubench"],
                        default=["perplexity", "blimp", "mblimp", "sib200", "mubench"])
    parser.add_argument("--output", default="results_llama_missing.json")
    args = parser.parse_args()

    out_path = Path(args.output)
    all_results = json.loads(out_path.read_text()) if out_path.exists() and IS_MAIN else {}

    if IS_MAIN:
        print(f"Loading {MODEL_ID} on {WORLD_SIZE} process(es) ...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).eval().to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for lang in args.langs:
        if IS_MAIN:
            print(f"\n-- {lang.upper()} --")
        all_results.setdefault(lang, {})

        if "perplexity" in args.evals:
            test_texts = load_test_texts(lang)
            perplexity = {"test": round(compute_perplexity(model, tokenizer, test_texts), 4)}
            if lang == "en":
                filtered_texts = load_english_filtered_test_texts()
                perplexity["test_filtered"] = round(compute_perplexity(model, tokenizer, filtered_texts), 4)
            all_results[lang]["perplexity"] = perplexity
            if IS_MAIN:
                print(f"  perplexity: {all_results[lang]['perplexity']}")

        if lang == "en" and "blimp" in args.evals:
            all_results[lang]["blimp"] = eval_blimp(model, tokenizer)
            if IS_MAIN:
                print(f"  blimp macro_avg: {all_results[lang]['blimp']['macro_avg']}")

        if lang == "hi" and "mblimp" in args.evals:
            all_results[lang]["mblimp"] = eval_mblimp(model, tokenizer)
            if IS_MAIN:
                print(f"  mblimp: {all_results[lang]['mblimp']}")

        if "sib200" in args.evals:
            all_results[lang]["sib200"] = eval_sib200(model, tokenizer, lang)
            if IS_MAIN:
                print(f"  sib200: {all_results[lang]['sib200']}")

        if "mubench" in args.evals:
            all_results[lang]["mubench"] = eval_mubench(model, tokenizer, lang)
            if IS_MAIN:
                print(f"  mubench avg: {all_results[lang]['mubench'].get('avg')}")

        if IS_MAIN:
            out_path.write_text(json.dumps(all_results, indent=2))

    if IS_MAIN:
        print(f"\nDone. Results in {out_path}")


if __name__ == "__main__":
    main()
