"""
Baseline evaluation: Sarvam-2B and Llama 3.2 1B across English, Hindi, Telugu.

Evaluations per language:
  English  — Perplexity, BLiMP (67 tasks), SIB-200, MuBench
  Hindi    — Perplexity, M-BLiMP (jumelet/multiblimp hin), SIB-200, MuBench
  Telugu   — Perplexity, SIB-200, MuBench

Usage:
    python baselines.py                                  # all models, all langs, all evals
    python baselines.py --models sarvam2b                # single model
    python baselines.py --langs en hi                    # subset of languages
    python baselines.py --evals perplexity blimp         # subset of evals
    python baselines.py --mubench_dataset <HF_REPO_ID>   # required for MuBench
    python baselines.py --output results_baselines.json
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ── Models ─────────────────────────────────────────────────────────────────────

MODELS = {
    "sarvam2b": "sarvamai/sarvam-2b-v0.5",
    "llama32_1b": "meta-llama/Llama-3.2-1B",
}

# ── Dataset configs ────────────────────────────────────────────────────────────

PERPLEXITY_DATASETS = {
    "en": "BabyLM-community/BabyLM-Test",
    "hi": ("pulipakav-1/translated-babylm-hindi", "test"),
    "te": ("pulipakav-1/translated-babylm-telugu", "test"),
}

SIB200_CONFIGS = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "te": "tel_Telu",
}

SIB200_TOPIC_PREFIX = {
    "en": "Topic: ",
    "hi": "विषय: ",
    "te": "విషయం: ",
}

SIB200_LABELS = [
    "science/technology", "travel", "politics", "sports",
    "health", "entertainment", "geography",
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

# ── Scoring ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def log_likelihood(model, tokenizer, text: str, device) -> float:
    ids = tokenizer.encode(text, return_tensors="pt").to(device)
    if ids.size(1) < 2:
        return float("-inf")
    logits    = model(ids).logits
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    return log_probs[0].gather(-1, ids[0, 1:].unsqueeze(-1)).squeeze(-1).sum().item()


@torch.no_grad()
def log_likelihood_normalized(model, tokenizer, text: str, device) -> float:
    ids = tokenizer.encode(text, return_tensors="pt").to(device)
    n = ids.size(1) - 1
    if n < 1:
        return float("-inf")
    logits    = model(ids).logits
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    return log_probs[0].gather(-1, ids[0, 1:].unsqueeze(-1)).squeeze(-1).sum().item() / n


# ── Perplexity ─────────────────────────────────────────────────────────────────

def eval_perplexity(model, tokenizer, lang: str, device) -> dict:
    cfg = PERPLEXITY_DATASETS[lang]
    if isinstance(cfg, tuple):
        repo, split = cfg
        splits = {split: load_dataset(repo, split=split)}
    else:
        ds = load_dataset(cfg)
        splits = dict(ds.items()) if hasattr(ds, "items") else {"data": ds}

    results = {}
    for split_name, dataset in splits.items():
        total_nll, total_tokens = 0.0, 0
        for row in tqdm(dataset, desc=f"  Perplexity [{lang}/{split_name}]", leave=False):
            text = row.get("text") or row.get("sentence") or ""
            if not text:
                continue
            ids = tokenizer.encode(text, return_tensors="pt").to(device)
            if ids.size(1) < 2:
                continue
            with torch.no_grad():
                logits    = model(ids).logits
                log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
                nll       = -log_probs[0].gather(-1, ids[0, 1:].unsqueeze(-1)).squeeze(-1).sum().item()
            total_nll    += nll
            total_tokens += ids.size(1) - 1
        ppl = math.exp(total_nll / total_tokens) if total_tokens > 0 else float("inf")
        results[split_name] = round(ppl, 4)
        print(f"    Perplexity [{lang}/{split_name}]: {ppl:.4f}")
    return results


# ── BLiMP (English only) ───────────────────────────────────────────────────────

def eval_blimp(model, tokenizer, device) -> dict:
    results = {}
    for task in tqdm(BLIMP_TASKS, desc="  BLiMP"):
        ds      = load_dataset("BabyLM-community/BabyLM-BLIMP-Filtered", task, split="train")
        correct = sum(
            log_likelihood(model, tokenizer, row["sentence_good"], device) >
            log_likelihood(model, tokenizer, row["sentence_bad"],  device)
            for row in ds
        )
        acc = correct / len(ds)
        results[task] = round(acc, 4)
        tqdm.write(f"    {task:<60s} {acc:.4f}")
    macro = sum(results.values()) / len(results)
    results["macro_avg"] = round(macro, 4)
    print(f"    BLiMP macro avg: {macro:.4f}")
    return results


# ── M-BLiMP (Hindi only) ───────────────────────────────────────────────────────

def eval_mblimp(model, tokenizer, device) -> dict:
    ds      = load_dataset("jumelet/multiblimp", "hin", split="train")
    correct = sum(
        log_likelihood(model, tokenizer, row["sen"],       device) >
        log_likelihood(model, tokenizer, row["wrong_sen"], device)
        for row in tqdm(ds, desc="  M-BLiMP", leave=False)
    )
    acc = correct / len(ds)
    print(f"    M-BLiMP [hin]: {acc:.4f}")
    return {"accuracy": round(acc, 4)}


# ── SIB-200 ────────────────────────────────────────────────────────────────────

def eval_sib200(model, tokenizer, lang: str, device) -> dict:
    config = SIB200_CONFIGS[lang]
    prefix = SIB200_TOPIC_PREFIX[lang]
    ds     = load_dataset("Davlan/sib200", config, split="test")
    correct = 0
    for row in tqdm(ds, desc=f"  SIB-200 [{lang}]", leave=False):
        text = row["text"]
        best_score, best_label = float("-inf"), None
        for label in SIB200_LABELS:
            score = log_likelihood_normalized(model, tokenizer, f"{text}\n{prefix}{label}", device)
            if score > best_score:
                best_score, best_label = score, label
        if best_label == row["category"]:
            correct += 1
    acc = correct / len(ds)
    print(f"    SIB-200 [{config}]: {acc:.4f}")
    return {"accuracy": round(acc, 4)}


# ── MuBench ────────────────────────────────────────────────────────────────────

def eval_mubench(model, tokenizer, lang: str, device, mubench_dataset_id: str) -> dict:
    suffix  = MUBENCH_LANG_SUFFIX[lang]
    results = {}
    tasks   = [t + suffix for t in MUBENCH_TASKS_BASE]
    for task_config in tqdm(tasks, desc=f"  MuBench [{lang}]"):
        task_key = task_config.replace(f"Dataset_local_template{suffix}", "")
        try:
            ds = load_dataset(mubench_dataset_id, task_config, split="test")
        except Exception as e:
            tqdm.write(f"    SKIP {task_config}: {e}")
            continue
        correct = 0
        for row in ds:
            prompt  = row["prompt"]
            choices = row["choices"]
            label   = row["label"]
            scores  = [
                log_likelihood(model, tokenizer, prompt + c, device) /
                max(len(tokenizer.encode(c, add_special_tokens=False)), 1)
                for c in choices
            ]
            if scores.index(max(scores)) == label:
                correct += 1
        acc = correct / len(ds)
        results[task_key] = round(acc, 4)
        tqdm.write(f"    {task_key:<30s} {acc:.4f}")
    if results:
        results["avg"] = round(sum(results.values()) / len(results), 4)
    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Baseline evaluation: Sarvam-2B and Llama 3.2 1B")
    parser.add_argument("--models",  nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--langs",   nargs="+", choices=["en", "hi", "te"], default=["en", "hi", "te"])
    parser.add_argument("--evals",   nargs="+",
                        choices=["perplexity", "blimp", "mblimp", "sib200", "mubench"],
                        default=["perplexity", "blimp", "mblimp", "sib200", "mubench"])
    parser.add_argument("--mubench_dataset", default="aialt/MuBench",
                        help="HuggingFace dataset ID for MuBench")
    parser.add_argument("--output",  default="results_baselines.json")
    args = parser.parse_args()

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_results = {}

    for model_name, model_id in MODELS.items():
        if model_name not in args.models:
            continue

        print(f"\n{'='*60}")
        print(f"Model : {model_name}  ({model_id})")
        print(f"{'='*60}")

        model = AutoModelForCausalLM.from_pretrained(model_id).eval().to(device)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        except ValueError:
            from transformers import PreTrainedTokenizerFast
            tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id)
        results = {}

        for lang in args.langs:
            results[lang] = {}
            print(f"\n  -- {lang.upper()} --")

            if "perplexity" in args.evals:
                results[lang]["perplexity"] = eval_perplexity(model, tokenizer, lang, device)

            if lang == "en" and "blimp" in args.evals:
                results[lang]["blimp"] = eval_blimp(model, tokenizer, device)

            if lang == "hi" and "mblimp" in args.evals:
                results[lang]["mblimp"] = eval_mblimp(model, tokenizer, device)

            if "sib200" in args.evals:
                results[lang]["sib200"] = eval_sib200(model, tokenizer, lang, device)

            if "mubench" in args.evals:
                results[lang]["mubench"] = eval_mubench(model, tokenizer, lang, device,
                                                         args.mubench_dataset)

        all_results[model_name] = results
        out = Path(args.output)
        out.write_text(json.dumps(all_results, indent=2))
        print(f"  Saved → {out}")
        del model
        torch.cuda.empty_cache()

    print(f"\nFinal results → {Path(args.output)}")


if __name__ == "__main__":
    main()
