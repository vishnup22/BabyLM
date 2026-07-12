"""Unified GPT-BERT evaluation: monolingual (en/hi/te) + bilingual (en-hi, en-tel; seed1/seed2).

Distributed across all available GPUs with accelerate: every example (BLiMP/M-BLiMP/SIB-200/
MuBench row, perplexity text) is independent of every other one, so each eval's dataset is
sharded across processes/GPUs and only the final correct-counts / NLL sums are reduced across
ranks -- one evaluation per model, split across GPUs, not N separate jobs.

For every model, evaluates on every language it was trained on:
  - Perplexity : causal (correct, comparable to GPT-2/Sarvam/Llama) on Test + OS-data.
  - BLiMP      : English only, 67 tasks (BabyLM-community/BabyLM-BLIMP-Filtered).
  - M-BLiMP    : Hindi only (jumelet/multiblimp, config=hin).
  - SIB-200    : Davlan/sib200, per-language config.
  - MuBench    : aialt/MuBench, per-language task suffix.

Monolingual models load from the pulipakav-1/*-gptbert_babylm2026 HF repos.
Bilingual models load from the pulipakav-1/en-hi-gptbert-checkpoints and
pulipakav-1/en-tel-gptbert-checkpoints HF repos (config + tokenizer + seed{1,2}_ema.bin).

Usage:
    accelerate launch --num_processes 4 eval_gptbert_all.py
    accelerate launch --num_processes 4 eval_gptbert_all.py --models mono_en en_hi_seed1
    accelerate launch --num_processes 4 eval_gptbert_all.py --evals perplexity sib200
    accelerate launch --num_processes 4 eval_gptbert_all.py --output results_gptbert_all.json
    python eval_gptbert_all.py --csv-only results_gptbert_all.json   # just re-export an existing JSON to CSV

    # still works single-GPU, no accelerate launch needed:
    python eval_gptbert_all.py
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT / "gpt-bert"))
from configuration_gptbert import GptBertConfig
from modeling_gptbert import GptBertForMaskedLM

MAX_SEQ_LEN = 128
BATCH_SIZE  = 32

# Task-level sharding (BLiMP/MuBench) is uneven -- some tasks (e.g. MuBench's GPQA) take far
# longer than others, so a fast rank can sit waiting at a collective well past NCCL's default
# 10-minute timeout while a slow rank finishes its heavier task. Use a generous timeout instead.
accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=4))])
DEVICE = accelerator.device
RANK = accelerator.process_index
WORLD_SIZE = accelerator.num_processes
IS_MAIN = accelerator.is_main_process

# ── Model registry ───────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "mono_en":     {"kind": "hf", "repo": "pulipakav-1/english-gptbert_babylm2026", "langs": ["en"]},
    "mono_hi":     {"kind": "hf", "repo": "pulipakav-1/hindi-gptbert_babylm2026",   "langs": ["hi"]},
    "mono_te":     {"kind": "hf", "repo": "pulipakav-1/telugu-gptbert_babylm2026",  "langs": ["te"]},
    "en_hi_seed1": {"kind": "local", "pair": "hi",  "seed": 1, "langs": ["en", "hi"]},
    "en_hi_seed2": {"kind": "local", "pair": "hi",  "seed": 2, "langs": ["en", "hi"]},
    "en_tel_seed1": {"kind": "local", "pair": "tel", "seed": 1, "langs": ["en", "te"]},
    "en_tel_seed2": {"kind": "local", "pair": "tel", "seed": 2, "langs": ["en", "te"]},
}

PAIR_HF_REPO = {
    "hi":  "pulipakav-1/en-hi-gptbert-checkpoints",
    "tel": "pulipakav-1/en-tel-gptbert-checkpoints",
}
NAME_PREFIX = {"hi": "en-hi-gptbert-seed", "tel": "en-tel-gptbert-seed"}
TOKENIZER_NAME = {"hi": "tokenizer_en_hi_vs32768.json", "tel": "tokenizer_en_tel_vs32768.json"}

# ── Per-language eval config ──────────────────────────────────────────────────────

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
OS_OPUS_CONFIG = {"en": ("en", "hi", "en"), "hi": ("en", "hi", "hi"), "te": ("en", "te", "te")}

# English-only "Filtered" test set: just the 4 subsections a bilingual model's English half was
# actually trained on (childes/gutenberg were replaced by Hindi/Telugu translations in bilingual
# training). No equivalent exists for the Hindi/Telugu side -- translated-babylm-{hi,te}'s `test`
# split has no per-subsection files to filter by.
EN_FILTERED_SOURCES = ["bnc_spoken", "open_subtitles", "simple_wiki", "switchboard"]


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


SYNC_DIR = REPO_ROOT / ".dist_sync" / os.environ.get("SLURM_JOB_ID", str(os.getpid()))


def gather_merge_dicts(d, tag):
    """Merge a {key: value} dict from every rank into one, via the filesystem instead of an
    NCCL collective. BLiMP/MuBench shard whole tasks across ranks -- each rank's dict is
    already self-contained, no cross-rank compute needed -- so a plain file wait/read avoids
    the NCCL/CUDA collective hang this cluster's old kernel is prone to on hours-long jobs
    (dist.all_gather_object hung for 4h+ on a completed MuBench run with no error raised).
    `tag` must be unique per (model, lang, eval) call within a job to avoid collisions."""
    if WORLD_SIZE == 1:
        return d
    SYNC_DIR.mkdir(parents=True, exist_ok=True)
    (SYNC_DIR / f"{tag}_rank{RANK}.json").write_text(json.dumps(d))
    expected = [SYNC_DIR / f"{tag}_rank{r}.json" for r in range(WORLD_SIZE)]
    waited = 0
    while not all(p.exists() for p in expected):
        time.sleep(5)
        waited += 5
        if IS_MAIN and waited % 300 == 0:
            missing = [p.name for p in expected if not p.exists()]
            print(f"  [sync] waiting on {missing} ({waited}s elapsed)", flush=True)
    time.sleep(2)  # guard against a reader racing a writer still flushing its file
    merged = {}
    for p in expected:
        merged.update(json.loads(p.read_text()))
    return merged


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


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model_and_tokenizer(spec):
    if spec["kind"] == "hf":
        model = GptBertForMaskedLM.from_pretrained(spec["repo"]).eval().to(DEVICE)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(spec["repo"])
    else:
        hf_repo = PAIR_HF_REPO[spec["pair"]]
        cfg_path = hf_hub_download(repo_id=hf_repo, filename="multilingual.json")
        with open(cfg_path) as f:
            cfg_dict = json.load(f)
        config = GptBertConfig(**cfg_dict)

        ckpt_name = f'{NAME_PREFIX[spec["pair"]]}{spec["seed"]}_ema.bin'
        ckpt_path = hf_hub_download(repo_id=hf_repo, filename=ckpt_name)
        state_dict = torch.load(ckpt_path, map_location="cpu")

        model = GptBertForMaskedLM(config)
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        if missing and IS_MAIN:
            print(f"  WARNING - missing keys: {missing}")
        if unexpected and IS_MAIN:
            print(f"  WARNING - unexpected keys: {unexpected}")
        model = model.eval().to(DEVICE)

        tok_path = hf_hub_download(repo_id=hf_repo, filename=TOKENIZER_NAME[spec["pair"]])
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tok_path))

    cls_id  = tokenizer.convert_tokens_to_ids("<s>")
    mask_id = tokenizer.convert_tokens_to_ids("<mask>")
    pad_id  = tokenizer.convert_tokens_to_ids("<pad>")
    if pad_id is None or pad_id < 0:
        pad_id = 0
    return model, tokenizer, cls_id, mask_id, pad_id


# ── Data loading (cached per language) ──────────────────────────────────────────

_test_cache = {}
_os_cache = {}


def load_test_texts(lang):
    if lang in _test_cache:
        return _test_cache[lang]
    name, kwargs = TEST_SOURCE[lang]
    ds = load_dataset_retry(name, **kwargs)
    if kwargs.get("streaming"):
        texts = [row["text"] for row in ds if row.get("text", "").strip()]
    else:
        texts = [row["text"] for row in ds if row.get("text")]
    _test_cache[lang] = texts
    return texts


_en_filtered_cache = None


def load_english_filtered_test_texts():
    global _en_filtered_cache
    if _en_filtered_cache is not None:
        return _en_filtered_cache
    data_files = [
        f"hf://datasets/BabyLM-community/BabyLM-Test/{name}.test" for name in EN_FILTERED_SOURCES
    ]
    ds = load_dataset_retry("text", data_files={"test": data_files}, split="test")
    _en_filtered_cache = [row["text"] for row in ds if row.get("text")]
    return _en_filtered_cache


def load_os_texts(lang):
    if lang in _os_cache:
        return _os_cache[lang]
    lang1, lang2, side = OS_OPUS_CONFIG[lang]
    ds = load_dataset_retry("Helsinki-NLP/opus-100", f"{lang1}-{lang2}", split="train", streaming=True)
    texts = []
    for row in ds:
        text = row["translation"][side].strip()
        if text:
            texts.append(text)
    _os_cache[lang] = texts
    return texts


# ── Scoring ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def score_gptbert_pll(model, token_ids, cls_id, mask_id):
    full_ids = [cls_id] + token_ids
    seq_len  = len(full_ids)
    n        = seq_len - 1
    if n == 0:
        return float("-inf")
    base     = torch.tensor(full_ids, dtype=torch.long, device=DEVICE)
    input_t  = base.unsqueeze(1).expand(-1, n).clone()
    labels_t = torch.full((seq_len, n), -100, dtype=torch.long, device=DEVICE)
    for j in range(n):
        pos = j + 1
        input_t[pos, j]  = mask_id
        labels_t[pos, j] = base[pos]
    attn = torch.zeros(n, 1, seq_len, seq_len, dtype=torch.bool, device=DEVICE)
    static_emb, rel_emb = model.embedding(input_t)
    hidden               = model.transformer(static_emb, attn, rel_emb)
    logits               = model.classifier(hidden, labels_t)
    log_probs = F.log_softmax(logits, dim=-1)
    gold      = base[1:].to(DEVICE)
    return log_probs[torch.arange(n, device=DEVICE), gold].sum().item()


def compute_causal_perplexity(model, tokenizer, texts, cls_id, pad_id,
                               max_seq_len=MAX_SEQ_LEN, batch_size=BATCH_SIZE):
    my_texts = shard(texts)
    total_nll, total_tokens = 0.0, 0
    for i in tqdm(range(0, len(my_texts), batch_size), desc="  Causal perplexity", leave=False, disable=not IS_MAIN):
        batch_texts = my_texts[i:i + batch_size]
        token_lists = [tokenizer.encode(t, add_special_tokens=False)[:max_seq_len - 1] for t in batch_texts]
        token_lists = [t for t in token_lists if len(t) > 0]
        if not token_lists:
            continue
        B = len(token_lists)
        L = max(len(t) for t in token_lists) + 1

        input_ids = torch.full((L, B), pad_id, dtype=torch.long)
        labels    = torch.full((L, B), -100,   dtype=torch.long)
        valid_lens = torch.zeros(B, dtype=torch.long)
        for b, tok in enumerate(token_lists):
            n = len(tok)
            input_ids[:n + 1, b] = torch.tensor([cls_id] + tok, dtype=torch.long)
            labels[:n + 1, b]    = torch.tensor(tok + [-100], dtype=torch.long)
            valid_lens[b] = n + 1

        input_ids = input_ids.to(DEVICE)
        labels    = labels.to(DEVICE)
        valid_lens = valid_lens.to(DEVICE)

        causal_block = torch.triu(torch.ones(L, L, dtype=torch.bool, device=DEVICE), diagonal=1)
        pad_block = torch.arange(L, device=DEVICE).unsqueeze(0) >= valid_lens.unsqueeze(1)
        pad_block = pad_block.unsqueeze(1).expand(-1, L, -1)
        mask_out = (causal_block.unsqueeze(0) | pad_block).unsqueeze(1)

        with torch.no_grad():
            static_emb, rel_emb = model.embedding(input_ids)
            hidden = model.transformer(static_emb, mask_out, rel_emb)
            logits = model.classifier(hidden, labels)

        gold = labels.flatten()
        gold = gold[gold != -100]
        if gold.numel() == 0:
            continue
        nll_sum = F.cross_entropy(logits, gold, reduction="sum").item()
        total_nll    += nll_sum
        total_tokens += gold.numel()

    total_nll, total_tokens = reduce_sum_pair(total_nll, total_tokens)
    return math.exp(total_nll / total_tokens) if total_tokens > 0 else float("inf")


def eval_perplexity(model, tokenizer, cls_id, pad_id, lang):
    test_texts = load_test_texts(lang)
    os_texts   = load_os_texts(lang)
    result = {
        "causal_test": round(compute_causal_perplexity(model, tokenizer, test_texts, cls_id, pad_id), 4),
        "causal_os":   round(compute_causal_perplexity(model, tokenizer, os_texts, cls_id, pad_id), 4),
    }
    if lang == "en":
        filtered_texts = load_english_filtered_test_texts()
        result["causal_test_filtered"] = round(
            compute_causal_perplexity(model, tokenizer, filtered_texts, cls_id, pad_id), 4)
    return result


def eval_blimp(model, tokenizer, cls_id, mask_id, model_key):
    my_tasks = shard(BLIMP_TASKS)
    local_results = {}
    for task in tqdm(my_tasks, desc="  BLiMP", disable=not IS_MAIN):
        ds = load_dataset_retry("BabyLM-community/BabyLM-BLIMP-Filtered", task, split="train")
        correct = 0
        for row in ds:
            good_ids = tokenizer.encode(row["sentence_good"], add_special_tokens=False)[:MAX_SEQ_LEN]
            bad_ids  = tokenizer.encode(row["sentence_bad"], add_special_tokens=False)[:MAX_SEQ_LEN]
            good = score_gptbert_pll(model, good_ids, cls_id, mask_id)
            bad  = score_gptbert_pll(model, bad_ids, cls_id, mask_id)
            if good > bad:
                correct += 1
        local_results[task] = round(correct / len(ds), 4)
    results = gather_merge_dicts(local_results, tag=f"{model_key}_blimp")
    if results:
        results["macro_avg"] = round(sum(results.values()) / len(results), 4)
    return results


def eval_mblimp(model, tokenizer, cls_id, mask_id):
    ds = load_dataset_retry("jumelet/multiblimp", "hin", split="train")
    my_rows = shard(list(ds))
    correct, total = 0, 0
    for row in tqdm(my_rows, desc="  M-BLiMP", leave=False, disable=not IS_MAIN):
        good_ids = tokenizer.encode(row["sen"], add_special_tokens=False)[:MAX_SEQ_LEN]
        bad_ids  = tokenizer.encode(row["wrong_sen"], add_special_tokens=False)[:MAX_SEQ_LEN]
        good = score_gptbert_pll(model, good_ids, cls_id, mask_id)
        bad  = score_gptbert_pll(model, bad_ids, cls_id, mask_id)
        if good > bad:
            correct += 1
        total += 1
    correct, total = reduce_sum_pair(correct, total)
    return {"accuracy": round(correct / total, 4) if total > 0 else None}


def eval_sib200(model, tokenizer, cls_id, mask_id, lang):
    ds = load_dataset_retry("Davlan/sib200", SIB200_CONFIG[lang], split="test")
    my_rows = shard(list(ds))
    correct, total = 0, 0
    for row in tqdm(my_rows, desc="  SIB-200", leave=False, disable=not IS_MAIN):
        text = row["text"]
        best_score, best_label = float("-inf"), None
        for label in SIB200_LABELS:
            ids = tokenizer.encode(f"{text}\nTopic: {label}", add_special_tokens=False)[:MAX_SEQ_LEN]
            score = score_gptbert_pll(model, ids, cls_id, mask_id) / max(len(ids), 1)
            if score > best_score:
                best_score, best_label = score, label
        if best_label == row["category"]:
            correct += 1
        total += 1
    correct, total = reduce_sum_pair(correct, total)
    return {"accuracy": round(correct / total, 4) if total > 0 else None}


def eval_mubench(model, tokenizer, cls_id, mask_id, lang, model_key):
    suffix  = MUBENCH_LANG_SUFFIX[lang]
    my_tasks = shard(MUBENCH_TASKS_BASE)
    local_results = {}
    for task_base in tqdm(my_tasks, desc="  MuBench", disable=not IS_MAIN):
        task_config = task_base + suffix
        task_key    = task_base.replace("Dataset_local_template", "")
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
                ids = tokenizer.encode(prompt + choice, add_special_tokens=False)[:MAX_SEQ_LEN]
                scores.append(score_gptbert_pll(model, ids, cls_id, mask_id) / max(len(ids), 1))
            if scores.index(max(scores)) == label:
                correct += 1
        local_results[task_key] = round(correct / len(ds), 4)
    results = gather_merge_dicts(local_results, tag=f"{model_key}_{lang}_mubench")
    if results:
        results["avg"] = round(sum(results.values()) / len(results), 4)
    return results


# ── CSV export ─────────────────────────────────────────────────────────────────

def flatten_results(all_results):
    """model -> lang -> category -> metric -> value, into flat (model, lang, category, metric, value) rows."""
    rows = []
    for model_key, langs in all_results.items():
        for lang, categories in langs.items():
            for category, metrics in categories.items():
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        rows.append((model_key, lang, category, metric, value))
                else:
                    rows.append((model_key, lang, category, "value", metrics))
    return rows


def write_csv(all_results, csv_path):
    rows = flatten_results(all_results)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "lang", "category", "metric", "value"])
        writer.writerows(rows)
    print(f"CSV written to {csv_path}  ({len(rows)} rows)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=list(MODEL_REGISTRY), default=list(MODEL_REGISTRY))
    parser.add_argument("--evals",  nargs="+",
                        choices=["perplexity", "blimp", "mblimp", "sib200", "mubench"],
                        default=["perplexity", "blimp", "mblimp", "sib200", "mubench"])
    parser.add_argument("--langs", nargs="+", choices=["en", "hi", "te"], default=None,
                        help="Restrict to these languages only (default: each model's full lang list)")
    parser.add_argument("--output", default="results_gptbert_all.json")
    parser.add_argument("--csv-only", metavar="JSON_PATH", default=None,
                        help="Skip evaluation entirely; just convert an existing results JSON to CSV")
    args = parser.parse_args()

    if args.csv_only:
        if IS_MAIN:
            json_path = Path(args.csv_only)
            all_results = json.loads(json_path.read_text())
            write_csv(all_results, json_path.with_suffix(".csv"))
        return

    out_path = Path(args.output)
    all_results = json.loads(out_path.read_text()) if out_path.exists() and IS_MAIN else {}

    for model_key in args.models:
        spec = MODEL_REGISTRY[model_key]
        if IS_MAIN:
            print(f"\n{'='*70}\nModel: {model_key}  ({spec})  [{WORLD_SIZE} GPU(s)]\n{'='*70}")

        model, tokenizer, cls_id, mask_id, pad_id = load_model_and_tokenizer(spec)
        all_results.setdefault(model_key, {})

        langs = [l for l in (args.langs or spec["langs"]) if l in spec["langs"]]
        for lang in langs:
            if IS_MAIN:
                print(f"\n  -- {lang.upper()} --")
            all_results[model_key].setdefault(lang, {})

            if "perplexity" in args.evals:
                all_results[model_key][lang]["perplexity"] = eval_perplexity(
                    model, tokenizer, cls_id, pad_id, lang)
                if IS_MAIN:
                    print(f"    perplexity: {all_results[model_key][lang]['perplexity']}")

            if lang == "en" and "blimp" in args.evals:
                all_results[model_key][lang]["blimp"] = eval_blimp(model, tokenizer, cls_id, mask_id, model_key)
                if IS_MAIN:
                    print(f"    blimp macro_avg: {all_results[model_key][lang]['blimp']['macro_avg']}")

            if lang == "hi" and "mblimp" in args.evals:
                all_results[model_key][lang]["mblimp"] = eval_mblimp(model, tokenizer, cls_id, mask_id)
                if IS_MAIN:
                    print(f"    mblimp: {all_results[model_key][lang]['mblimp']}")

            if "sib200" in args.evals:
                all_results[model_key][lang]["sib200"] = eval_sib200(model, tokenizer, cls_id, mask_id, lang)
                if IS_MAIN:
                    print(f"    sib200: {all_results[model_key][lang]['sib200']}")

            if "mubench" in args.evals:
                all_results[model_key][lang]["mubench"] = eval_mubench(model, tokenizer, cls_id, mask_id, lang, model_key)
                if IS_MAIN:
                    print(f"    mubench avg: {all_results[model_key][lang]['mubench'].get('avg')}")

            if IS_MAIN:
                out_path.write_text(json.dumps(all_results, indent=2))

        del model
        torch.cuda.empty_cache()

    if IS_MAIN:
        write_csv(all_results, out_path.with_suffix(".csv"))
        print(f"\nDone. Results in {out_path}")


if __name__ == "__main__":
    main()
