"""
Monolingual Telugu evaluation.

Evaluations:
  - Perplexity   : pulipakav-1/translated-babylm-telugu  (split=test)
  - SIB-200      : Davlan/sib200  (config=tel_Telu)
  - MuBench      : per-task configs, lang suffix _te

Note: No M-BLiMP for Telugu — Telugu is not available in jumelet/multiblimp.

Usage:
    python telugu.py                          # all models, all evals
    python telugu.py --models gpt2_mono       # single model
    python telugu.py --evals perplexity sib200
    python telugu.py --output results_te.json
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
    "gpt2_mono_seed1": {
        "model_id": "pulipakav-1/gpt-2-telugu1",
        "arch": "causal",
    },
    "gpt2_mono_seed2": {
        "model_id": "pulipakav-1/gpt-2-telugu2",
        "arch": "causal",
    },
    "gptbert_mono": {
        "model_id": "pulipakav-1/telugu-gptbert_babylm2026",
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

MUBENCH_TASKS = [
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

SIB200_CONFIG = "tel_Telu"

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

def compute_perplexity(model, tokenizer, arch, texts: list[str], device, extras: dict,
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


def eval_perplexity(model, tokenizer, arch, device, extras, cleaned_dir=None, max_samples=None, max_seq_len=128) -> dict:
    if cleaned_dir:
        cleaned_file = Path(cleaned_dir) / "telugu_test.txt"
        texts = cleaned_file.read_text(encoding="utf-8").splitlines()
        texts = [t for t in texts if t.strip()]
        if max_samples and len(texts) > max_samples:
            import random; random.seed(42)
            texts = random.sample(texts, max_samples)
    else:
        # stream to avoid loading all 1.1M test rows
        ds = load_dataset("pulipakav-1/translated-babylm-telugu", split="test", streaming=True)
        texts = []
        for row in ds:
            t = row.get("text", "")
            if t.strip():
                texts.append(t)
            if max_samples and len(texts) >= max_samples:
                break
    ppl = compute_perplexity(model, tokenizer, arch, texts, device, extras, max_seq_len)
    print(f"    Perplexity [test]: {ppl:.4f}")
    return {"test": round(ppl, 4)}


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
            prompt = f"{text}\nవిషయం: {label}"
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
        task_key = task_config.replace("Dataset_local_template_te", "")
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
    parser = argparse.ArgumentParser(description="Telugu monolingual evaluation")
    parser.add_argument("--models",  nargs="+", choices=list(MODELS), default=list(MODELS),
                        help="Models: gpt2_mono_seed1, gpt2_mono_seed2, gptbert_mono, sarvam2b, llama32_1b")
    parser.add_argument("--evals",   nargs="+",
                        choices=["perplexity", "sib200", "mubench"],
                        default=["perplexity", "sib200", "mubench"])
    parser.add_argument("--mubench_dataset", default="aialt/MuBench",
                        help="HuggingFace dataset ID for MuBench")
    parser.add_argument("--cleaned_dir", default=None,
                        help="Directory with pre-cleaned .txt files from clean_dataset.py (e.g. cleaned/)")
    parser.add_argument("--max_ppl_samples", type=int, default=0,
                        help="Max sentences for perplexity (default 0 = all)")
    parser.add_argument("--max_seq_len", type=int, default=128,
                        help="Truncate sequences to this many tokens for perplexity (default 128)")
    parser.add_argument("--output",  default="results_telugu.json")
    args = parser.parse_args()

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_stem   = Path(args.output).stem
    out_dir    = Path(args.output).parent
    eval_results = {e: {} for e in args.evals}

    def save_eval(eval_name, model_name, data):
        eval_results[eval_name][model_name] = data
        path = out_dir / f"{out_stem}_{eval_name}.json"
        path.write_text(json.dumps(eval_results[eval_name], indent=2))
        print(f"  Saved → {path}")

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

        if "perplexity" in args.evals:
            save_eval("perplexity", name,
                      eval_perplexity(model, tokenizer, arch, device, extras,
                                      args.cleaned_dir,
                                      args.max_ppl_samples or None,
                                      args.max_seq_len))

        if "sib200" in args.evals:
            save_eval("sib200", name,
                      eval_sib200(model, tokenizer, arch, device, extras))

        if "mubench" in args.evals:
            save_eval("mubench", name,
                      eval_mubench(model, tokenizer, arch, device, extras,
                                   args.mubench_dataset))

        del model
        torch.cuda.empty_cache()

    print(f"\nDone. Results in {out_dir}/{out_stem}_*.json")


if __name__ == "__main__":
    main()
