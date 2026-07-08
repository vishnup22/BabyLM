"""
Bilingual eng-tel evaluation.

Models:
  gpt2_bilingual : pulipakav-1/gpt2-tel-eng_babylm2026

Evaluations (English):
  - en_perplexity : BabyLM-community/BabyLM-Test
  - en_blimp      : BabyLM-community/BabyLM-BLIMP-Filtered (67 tasks)
  - en_sib200     : Davlan/sib200  (config eng_Latn)
  - en_mubench    : per-task configs, lang suffix _en

Evaluations (Telugu):
  - te_perplexity : pulipakav-1/translated-babylm-telugu  (split=test)
  - te_sib200     : Davlan/sib200  (config=tel_Telu)
  - te_mubench    : per-task configs, lang suffix _te

Note: No M-BLiMP for Telugu — Telugu is not in jumelet/multiblimp.

Usage:
    python eng_tel.py --cleaned_dir /path/to/cleaned
    python eng_tel.py --evals en_perplexity en_blimp te_perplexity te_sib200
    python eng_tel.py --output results_engtel.json
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

torch.set_num_threads(24)

# ── Model registry ─────────────────────────────────────────────────────────────

MODELS = {
    "gpt2_bilingual": {
        "model_id": "pulipakav-1/gpt2-tel-eng_babylm2026",
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

MUBENCH_TASKS_EN = [
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

MUBENCH_TASKS_TE = [
    "ARCChallengeDataset_local_template_te",
    "ARCEasyDataset_local_template_te",
    "BMLAMADataset_local_template_te",
    "GPQADataset_local_template_te",
    "HellaswagDataset_local_template_te",
    "MMLUDataset_local_template_te",
    "MMLUProDataset_local_template_te",
    "MNLIDataset_local_template_te",
    "SNLIDataset_local_template_te",
    "StoryClozeDataset_local_template_te",
    "TruthfulQADataset_local_template_te",
    "WinoGrandeDataset_local_template_te",
]

SIB200_LABELS = [
    "science/technology", "travel", "politics", "sports",
    "health", "entertainment", "geography",
]

# ── Scoring helpers ────────────────────────────────────────────────────────────

MAX_LEN = 1024

@torch.no_grad()
def score_causal(model, tokenizer, text: str, device) -> float:
    ids = tokenizer.encode(text, return_tensors="pt").to(device)
    ids = ids[:, -MAX_LEN:]
    if ids.size(1) < 2:
        return float("-inf")
    logits    = model(ids).logits
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    token_ll  = log_probs[0].gather(-1, ids[0, 1:].unsqueeze(-1)).squeeze(-1)
    return token_ll.sum().item()


@torch.no_grad()
def score_causal_normalized(model, tokenizer, text: str, device) -> float:
    ids = tokenizer.encode(text, return_tensors="pt").to(device)
    ids = ids[:, -MAX_LEN:]
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
        model = AutoModelForCausalLM.from_pretrained(model_id).eval().to(device)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        except ValueError:
            tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
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

def compute_perplexity(model, tokenizer, arch, texts: list, device, extras: dict,
                       max_seq_len: int = 128, batch_size: int = 32) -> float:
    total_nll, total_tokens = 0.0, 0

    if arch == "causal":
        for i in tqdm(range(0, len(texts), batch_size), desc="  Perplexity", leave=False):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_seq_len)
            input_ids = enc["input_ids"].to(device)
            attn_mask = enc["attention_mask"].to(device)
            with torch.no_grad():
                logits = model(input_ids, attention_mask=attn_mask).logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask   = attn_mask[:, 1:].contiguous().float()
            log_probs    = F.log_softmax(shift_logits, dim=-1)
            token_ll     = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
            total_nll    += -(token_ll * shift_mask).sum().item()
            total_tokens += shift_mask.sum().item()

    elif arch == "gptbert":
        for text in tqdm(texts, desc="  Perplexity", leave=False):
            token_ids = tokenizer.encode(text, add_special_tokens=False)[:max_seq_len]
            if not token_ids:
                continue
            pll = score_gptbert_pll(model, token_ids, extras["cls_id"], extras["mask_id"], device)
            total_nll    += -pll
            total_tokens += len(token_ids)

    return math.exp(total_nll / total_tokens) if total_tokens > 0 else float("inf")


# English sources matching bilingual training (excludes childes + gutenberg)
EN_BILINGUAL_SOURCES = [
    "hf://datasets/BabyLM-community/BabyLM-Test/bnc_spoken.test",
    "hf://datasets/BabyLM-community/BabyLM-Test/open_subtitles.test",
    "hf://datasets/BabyLM-community/BabyLM-Test/simple_wiki.test",
    "hf://datasets/BabyLM-community/BabyLM-Test/switchboard.test",
]

TE_BILINGUAL_SOURCES = [
    "hf://datasets/pulipakav-1/translated-babylm-telugu/test/childes.test.txt.train.te.txt",
    "hf://datasets/pulipakav-1/translated-babylm-telugu/test/gutenberg.test.txt.train.te.txt",
]

def eval_en_perplexity_filtered(model, tokenizer, arch, device, extras, max_samples=None, max_seq_len=128) -> dict:
    ds    = load_dataset("text", data_files={"test": EN_BILINGUAL_SOURCES}, split="test")
    texts = [row["text"] for row in ds if row.get("text", "").strip()]
    if max_samples and len(texts) > max_samples:
        import random; random.seed(42)
        texts = random.sample(texts, max_samples)
    ppl = compute_perplexity(model, tokenizer, arch, texts, device, extras, max_seq_len)
    print(f"    [EN] Perplexity [bilingual-filtered]: {ppl:.4f}")
    return {"bilingual_filtered": round(ppl, 4)}


def eval_te_perplexity_filtered(model, tokenizer, arch, device, extras, max_samples=None, max_seq_len=128) -> dict:
    ds    = load_dataset("text", data_files={"test": TE_BILINGUAL_SOURCES}, split="test")
    texts = [row["text"] for row in ds if row.get("text", "").strip()]
    if max_samples and len(texts) > max_samples:
        import random; random.seed(42)
        texts = random.sample(texts, max_samples)
    ppl = compute_perplexity(model, tokenizer, arch, texts, device, extras, max_seq_len)
    print(f"    [TE] Perplexity [bilingual-filtered]: {ppl:.4f}")
    return {"bilingual_filtered": round(ppl, 4)}


def eval_en_perplexity(model, tokenizer, arch, device, extras, cleaned_dir=None, max_samples=None, max_seq_len=128) -> dict:
    if cleaned_dir:
        cleaned_file = Path(cleaned_dir) / "english_test.txt"
        texts = cleaned_file.read_text(encoding="utf-8").splitlines()
        texts = [t for t in texts if t.strip()]
    else:
        ds    = load_dataset("text",
                             data_files={"test": "hf://datasets/BabyLM-community/BabyLM-Test/*.test"},
                             split="test")
        texts = [row["text"] for row in ds if row.get("text")]
    if max_samples and len(texts) > max_samples:
        import random; random.seed(42)
        texts = random.sample(texts, max_samples)
    ppl = compute_perplexity(model, tokenizer, arch, texts, device, extras, max_seq_len)
    print(f"    [EN] Perplexity [test]: {ppl:.4f}")
    return {"test": round(ppl, 4)}


def eval_te_perplexity(model, tokenizer, arch, device, extras, cleaned_dir=None, max_samples=None, max_seq_len=128) -> dict:
    if cleaned_dir:
        cleaned_file = Path(cleaned_dir) / "telugu_test.txt"
        texts = cleaned_file.read_text(encoding="utf-8").splitlines()
        texts = [t for t in texts if t.strip()]
        if max_samples and len(texts) > max_samples:
            import random; random.seed(42)
            texts = random.sample(texts, max_samples)
    else:
        ds = load_dataset("pulipakav-1/translated-babylm-telugu", split="test", streaming=True)
        texts = []
        for row in ds:
            t = row.get("text", "")
            if t.strip():
                texts.append(t)
            if max_samples and len(texts) >= max_samples:
                break
    ppl = compute_perplexity(model, tokenizer, arch, texts, device, extras, max_seq_len)
    print(f"    [TE] Perplexity [test]: {ppl:.4f}")
    return {"test": round(ppl, 4)}


# ── BLiMP (English) ────────────────────────────────────────────────────────────

def eval_en_blimp(model, tokenizer, arch, device, extras) -> dict:
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
    print(f"    [EN] BLiMP macro avg: {macro:.4f}")
    return results


# ── SIB-200 ────────────────────────────────────────────────────────────────────

def eval_sib200(model, tokenizer, arch, device, extras, config: str, prefix: str) -> dict:
    ds      = load_dataset("Davlan/sib200", config, split="test")
    correct = 0
    for row in tqdm(ds, desc=f"  SIB-200 [{config}]", leave=False):
        text = row["text"]
        best_score, best_label = float("-inf"), None
        for label in SIB200_LABELS:
            prompt = f"{text}\n{prefix}{label}"
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
    print(f"    SIB-200 [{config}]: {acc:.4f}")
    return {"accuracy": round(acc, 4)}


# ── MuBench ────────────────────────────────────────────────────────────────────

def eval_mubench(model, tokenizer, arch, device, extras, mubench_dataset_id: str,
                 tasks: list, lang_suffix: str) -> dict:
    results = {}
    for task_config in tqdm(tasks, desc=f"  MuBench [{lang_suffix}]"):
        task_key = task_config.replace(f"Dataset_local_template_{lang_suffix}", "")
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

ALL_EVALS = [
    "en_perplexity", "en_perplexity_filtered", "en_blimp", "en_sib200", "en_mubench",
    "te_perplexity", "te_perplexity_filtered", "te_sib200", "te_mubench",
]

def main():
    parser = argparse.ArgumentParser(description="Bilingual eng-tel evaluation")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--evals",  nargs="+", choices=ALL_EVALS, default=ALL_EVALS)
    parser.add_argument("--mubench_dataset", default="aialt/MuBench")
    parser.add_argument("--cleaned_dir", default=None,
                        help="Directory with pre-cleaned .txt files (english_test.txt, telugu_test.txt)")
    parser.add_argument("--max_ppl_samples", type=int, default=0,
                        help="Max sentences for perplexity (0 = all)")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--output", default="results_engtel.json")
    args = parser.parse_args()

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_stem = Path(args.output).stem
    out_dir  = Path(args.output).parent
    eval_results = {e: {} for e in args.evals}

    def save_eval(eval_name, model_name, data):
        eval_results[eval_name][model_name] = data
        path = out_dir / f"{out_stem}_{eval_name}.json"
        path.write_text(json.dumps(eval_results[eval_name], indent=2))
        print(f"  Saved → {path}")

    max_samples = args.max_ppl_samples or None

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

        # ── English evals ──────────────────────────────────────────────────────
        if "en_perplexity" in args.evals:
            save_eval("en_perplexity", name,
                      eval_en_perplexity(model, tokenizer, arch, device, extras,
                                         args.cleaned_dir, max_samples, args.max_seq_len))

        if "en_perplexity_filtered" in args.evals:
            save_eval("en_perplexity_filtered", name,
                      eval_en_perplexity_filtered(model, tokenizer, arch, device, extras,
                                                  max_samples, args.max_seq_len))

        if "en_blimp" in args.evals:
            save_eval("en_blimp", name,
                      eval_en_blimp(model, tokenizer, arch, device, extras))

        if "en_sib200" in args.evals:
            save_eval("en_sib200", name,
                      eval_sib200(model, tokenizer, arch, device, extras,
                                  "eng_Latn", "Topic: "))

        if "en_mubench" in args.evals:
            save_eval("en_mubench", name,
                      eval_mubench(model, tokenizer, arch, device, extras,
                                   args.mubench_dataset, MUBENCH_TASKS_EN, "en"))

        # ── Telugu evals ───────────────────────────────────────────────────────
        if "te_perplexity" in args.evals:
            save_eval("te_perplexity", name,
                      eval_te_perplexity(model, tokenizer, arch, device, extras,
                                         args.cleaned_dir, max_samples, args.max_seq_len))

        if "te_perplexity_filtered" in args.evals:
            save_eval("te_perplexity_filtered", name,
                      eval_te_perplexity_filtered(model, tokenizer, arch, device, extras,
                                                  max_samples, args.max_seq_len))

        if "te_sib200" in args.evals:
            save_eval("te_sib200", name,
                      eval_sib200(model, tokenizer, arch, device, extras,
                                  "tel_Telu", "విషయం: "))

        if "te_mubench" in args.evals:
            save_eval("te_mubench", name,
                      eval_mubench(model, tokenizer, arch, device, extras,
                                   args.mubench_dataset, MUBENCH_TASKS_TE, "te"))

        del model
        torch.cuda.empty_cache()

    print(f"\nDone. Results in {out_dir}/{out_stem}_*.json")


if __name__ == "__main__":
    main()
