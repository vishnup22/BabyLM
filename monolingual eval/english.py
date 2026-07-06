"""
Monolingual English evaluation.

Evaluations:
  - Perplexity   : BabyLM-community/BabyLM-Test
  - BLiMP        : BabyLM-community/BabyLM-BLIMP-Filtered (67 tasks)
  - SIB-200      : Davlan/sib200  (config eng_Latn)
  - MuBench      : per-task configs, lang suffix _en

Usage:
    python english.py                          # all models, all evals
    python english.py --models gpt2_mono       # single model
    python english.py --evals perplexity blimp # subset of evals
    python english.py --output results_en.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
from tqdm import tqdm

# ── Model registry ─────────────────────────────────────────────────────────────
# Fill in model_id once you have the HuggingFace repo links.
# arch: "causal" for GPT-2 / Llama / Sarvam  |  "gptbert" for GPT-BERT

MODELS = {
    "gpt2_mono": {
        "model_id": "pulipakav-1/gpt2-english-babylm2026",
        "arch": "causal",
    },
    "gptbert_mono": {
        "model_id": "pulipakav-1/english-gptbert_babylm2026",
        "arch": "gptbert",
    },
    "sarvam2b": {
        "model_id": "sarvamai/sarvam-2b-v0.5",
        "arch": "causal",
    },
    "llama32_1b": {
        "model_id": "meta-llama/Llama-3.2-1B",
        "arch": "causal",
    },
}

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

MUBENCH_TASKS = [
    "ARCChallengeDataset_local_template_en",
    "ARCEasyDataset_local_template_en",
    "BMLAMADataset_local_template_en",
    "GPQADataset_local_template_en",
    "HellaswagDataset_local_template_en",
    "MMLUDataset_local_template_en",
    "MMLUProDataset_local_template_en",
    "MNLIDataset_local_template_en",
    "SNLIDataset_local_template_en",
    "StoryClozeDataset_local_template_en",
    "TruthfulQADataset_local_template_en",
    "WinoGrandeDataset_local_template_en",
]

SIB200_CONFIG = "eng_Latn"

# ── Scoring helpers ────────────────────────────────────────────────────────────

@torch.no_grad()
def score_causal(model, tokenizer, text: str, device) -> float:
    ids = tokenizer.encode(text, return_tensors="pt").to(device)
    if ids.size(1) < 2:
        return float("-inf")
    logits    = model(ids).logits
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    token_ll  = log_probs[0].gather(-1, ids[0, 1:].unsqueeze(-1)).squeeze(-1)
    return token_ll.sum().item()


@torch.no_grad()
def score_causal_normalized(model, tokenizer, text: str, device) -> float:
    ids = tokenizer.encode(text, return_tensors="pt").to(device)
    n = ids.size(1) - 1
    if n < 1:
        return float("-inf")
    logits    = model(ids).logits
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    token_ll  = log_probs[0].gather(-1, ids[0, 1:].unsqueeze(-1)).squeeze(-1)
    return token_ll.sum().item() / n


@torch.no_grad()
def score_gptbert_pll(model, token_ids: list, cls_id: int, mask_id: int, device) -> float:
    full_ids = [cls_id] + token_ids
    seq_len  = len(full_ids)
    n        = seq_len - 1
    if n == 0:
        return float("-inf")
    base     = torch.tensor(full_ids, dtype=torch.long, device=device)
    input_t  = base.unsqueeze(1).expand(-1, n).clone()
    labels_t = torch.full((seq_len, n), -100, dtype=torch.long, device=device)
    for j in range(n):
        pos = j + 1
        input_t[pos, j]  = mask_id
        labels_t[pos, j] = base[pos]
    attn = torch.zeros(n, 1, seq_len, seq_len, dtype=torch.bool, device=device)
    static_emb, rel_emb = model.embedding(input_t)
    hidden               = model.transformer(static_emb, attn, rel_emb)
    logits               = model.classifier(hidden, labels_t)
    log_probs = F.log_softmax(logits, dim=-1)
    gold      = base[1:].to(device)
    return log_probs[torch.arange(n, device=device), gold].sum().item()


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(model_id: str, arch: str, device):
    if arch == "causal":
        model     = AutoModelForCausalLM.from_pretrained(model_id).eval().to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        return model, tokenizer, {}
    elif arch == "gptbert":
        _local = Path(__file__).parents[1] / "gpt-bert"
        sys.path.insert(0, str(_local))
        from modeling_gptbert import GptBertForMaskedLM
        model     = GptBertForMaskedLM.from_pretrained(model_id).eval().to(device)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id)
        extras    = {
            "cls_id":  tokenizer.convert_tokens_to_ids("<s>"),
            "mask_id": tokenizer.convert_tokens_to_ids("<mask>"),
        }
        return model, tokenizer, extras
    else:
        raise ValueError(f"Unknown arch: {arch}")


# ── Perplexity ─────────────────────────────────────────────────────────────────

def compute_perplexity(model, tokenizer, arch, texts: list[str], device, extras: dict) -> float:
    total_nll, total_tokens = 0.0, 0
    for text in tqdm(texts, desc="  Perplexity", leave=False):
        if arch == "causal":
            ids = tokenizer.encode(text, return_tensors="pt").to(device)
            if ids.size(1) < 2:
                continue
            with torch.no_grad():
                logits    = model(ids).logits
                log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
                nll       = -log_probs[0].gather(-1, ids[0, 1:].unsqueeze(-1)).squeeze(-1).sum().item()
            total_nll    += nll
            total_tokens += ids.size(1) - 1
        elif arch == "gptbert":
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if not token_ids:
                continue
            pll = score_gptbert_pll(model, token_ids, extras["cls_id"], extras["mask_id"], device)
            total_nll    += -pll
            total_tokens += len(token_ids)
    return math.exp(total_nll / total_tokens) if total_tokens > 0 else float("inf")


def eval_perplexity(model, tokenizer, arch, device, extras) -> dict:
    results = {}
    test_ds = load_dataset("BabyLM-community/BabyLM-Test")
    for split_name, split_ds in test_ds.items():
        texts = [row["text"] for row in split_ds if row.get("text")]
        ppl   = compute_perplexity(model, tokenizer, arch, texts, device, extras)
        results[split_name] = round(ppl, 4)
        print(f"    Perplexity [{split_name}]: {ppl:.4f}")
    return results


# ── BLiMP ──────────────────────────────────────────────────────────────────────

def eval_blimp(model, tokenizer, arch, device, extras) -> dict:
    results = {}
    for task in tqdm(BLIMP_TASKS, desc="  BLiMP"):
        ds      = load_dataset("BabyLM-community/BabyLM-BLIMP-Filtered", task, split="train")
        correct = 0
        for row in ds:
            if arch == "causal":
                good = score_causal(model, tokenizer, row["sentence_good"], device)
                bad  = score_causal(model, tokenizer, row["sentence_bad"],  device)
            else:
                good_ids = tokenizer.encode(row["sentence_good"], add_special_tokens=False)
                bad_ids  = tokenizer.encode(row["sentence_bad"],  add_special_tokens=False)
                good = score_gptbert_pll(model, good_ids, extras["cls_id"], extras["mask_id"], device)
                bad  = score_gptbert_pll(model, bad_ids,  extras["cls_id"], extras["mask_id"], device)
            if good > bad:
                correct += 1
        acc = correct / len(ds)
        results[task] = round(acc, 4)
        tqdm.write(f"    {task:<60s} {acc:.4f}")
    macro = sum(results.values()) / len(results)
    results["macro_avg"] = round(macro, 4)
    print(f"    BLiMP macro avg: {macro:.4f}")
    return results


# ── SIB-200 ────────────────────────────────────────────────────────────────────

SIB200_LABELS = [
    "science/technology", "travel", "politics", "sports",
    "health", "entertainment", "geography",
]

def eval_sib200(model, tokenizer, arch, device, extras) -> dict:
    ds      = load_dataset("Davlan/sib200", SIB200_CONFIG, split="test")
    correct = 0
    for row in tqdm(ds, desc="  SIB-200", leave=False):
        text = row["text"]
        best_score, best_label = float("-inf"), None
        for label in SIB200_LABELS:
            prompt = f"{text}\nTopic: {label}"
            if arch == "causal":
                score = score_causal_normalized(model, tokenizer, prompt, device)
            else:
                ids   = tokenizer.encode(prompt, add_special_tokens=False)
                score = score_gptbert_pll(model, ids, extras["cls_id"], extras["mask_id"], device) / max(len(ids), 1)
            if score > best_score:
                best_score, best_label = score, label
        if best_label == row["category"]:
            correct += 1
    acc = correct / len(ds)
    print(f"    SIB-200 [{SIB200_CONFIG}]: {acc:.4f}")
    return {"accuracy": round(acc, 4)}


# ── MuBench ────────────────────────────────────────────────────────────────────

def eval_mubench(model, tokenizer, arch, device, extras, mubench_dataset_id: str) -> dict:
    results = {}
    for task_config in tqdm(MUBENCH_TASKS, desc="  MuBench"):
        task_key = task_config.replace("Dataset_local_template_en", "").replace("Dataset_local_template_", "")
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
            scores  = []
            for choice in choices:
                text = prompt + choice
                n    = max(len(tokenizer.encode(choice, add_special_tokens=False)), 1)
                if arch == "causal":
                    s = score_causal(model, tokenizer, text, device) / n
                else:
                    ids = tokenizer.encode(text, add_special_tokens=False)
                    s   = score_gptbert_pll(model, ids, extras["cls_id"], extras["mask_id"], device) / max(len(ids), 1)
                scores.append(s)
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
    parser = argparse.ArgumentParser(description="English monolingual evaluation")
    parser.add_argument("--models",  nargs="+", choices=list(MODELS), default=list(MODELS),
                        help="Models to evaluate (default: all)")
    parser.add_argument("--evals",   nargs="+",
                        choices=["perplexity", "blimp", "sib200", "mubench"],
                        default=["perplexity", "blimp", "sib200", "mubench"],
                        help="Evaluations to run (default: all)")
    parser.add_argument("--mubench_dataset", default="aialt/MuBench",
                        help="HuggingFace dataset ID for MuBench")
    parser.add_argument("--output",  default="results_english.json")
    args = parser.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_results = {}

    for name in args.models:
        cfg      = MODELS[name]
        model_id = cfg["model_id"]
        arch     = cfg["arch"]

        if model_id == "TODO":
            print(f"\n[SKIP] {name} — model_id not set yet")
            continue

        print(f"\n{'='*60}")
        print(f"Model : {name}  ({model_id})  [{arch}]")
        print(f"{'='*60}")

        model, tokenizer, extras = load_model(model_id, arch, device)
        results = {}

        if "perplexity" in args.evals:
            results["perplexity"] = eval_perplexity(model, tokenizer, arch, device, extras)

        if "blimp" in args.evals:
            results["blimp"] = eval_blimp(model, tokenizer, arch, device, extras)

        if "sib200" in args.evals:
            results["sib200"] = eval_sib200(model, tokenizer, arch, device, extras)

        if "mubench" in args.evals:
            results["mubench"] = eval_mubench(model, tokenizer, arch, device, extras,
                                               args.mubench_dataset)

        all_results[name] = results
        out = Path(args.output)
        out.write_text(json.dumps(all_results, indent=2))
        print(f"  Saved → {out}")
        del model
        torch.cuda.empty_cache()

    print(f"\nFinal results → {Path(args.output)}")


if __name__ == "__main__":
    main()
