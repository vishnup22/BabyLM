"""
Small model (GPT-Wee, 28M) Telugu evaluation.
Models: pulipakav-1/babylm_telugu_2026 — random_seed{1,2,3} and curriculum_seed{1,2,3}

Evaluations:
  - Perplexity : pulipakav-1/translated-babylm-telugu  (split=test)
  - SIB-200    : Davlan/sib200  (config=tel_Telu)
  - MuBench    : per-task configs, lang suffix _te

Note: No M-BLiMP for Telugu — Telugu is not available in jumelet/multiblimp.

Usage:
    python telugu.py --cleaned_dir cleaned
    python telugu.py --models random_seed1 curriculum_seed1
    python telugu.py --evals perplexity sib200
    python telugu.py --output results_small_telugu.json
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

REPO = "pulipakav-1/babylm_telugu_2026"

MODELS = {
    "random_seed1":     {"model_id": REPO, "subfolder": "random_seed1",     "arch": "causal"},
    "random_seed2":     {"model_id": REPO, "subfolder": "random_seed2",     "arch": "causal"},
    "random_seed3":     {"model_id": REPO, "subfolder": "random_seed3",     "arch": "causal"},
    "curriculum_seed1": {"model_id": REPO, "subfolder": "curriculum_seed1", "arch": "causal"},
    "curriculum_seed2": {"model_id": REPO, "subfolder": "curriculum_seed2", "arch": "causal"},
    "curriculum_seed3": {"model_id": REPO, "subfolder": "curriculum_seed3", "arch": "causal"},
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


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(model_id: str, arch: str, device, subfolder: str = None):
    kwargs = {"subfolder": subfolder} if subfolder else {}
    if arch == "causal":
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval().to(device)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
        except ValueError:
            tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id, **kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer, {}
    else:
        raise ValueError(f"Unknown arch: {arch}")


# ── Perplexity ─────────────────────────────────────────────────────────────────

def compute_perplexity(model, tokenizer, texts: list, device,
                       max_seq_len: int = 128, batch_size: int = 32) -> float:
    total_nll, total_tokens = 0.0, 0
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
    return math.exp(total_nll / total_tokens) if total_tokens > 0 else float("inf")


def eval_perplexity(model, tokenizer, device, cleaned_dir=None, max_samples=None, max_seq_len=128) -> dict:
    if cleaned_dir:
        texts = (Path(cleaned_dir) / "telugu_test.txt").read_text(encoding="utf-8").splitlines()
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
    ppl = compute_perplexity(model, tokenizer, texts, device, max_seq_len)
    print(f"    Perplexity [test]: {ppl:.4f}")
    return {"test": round(ppl, 4)}


# ── SIB-200 ────────────────────────────────────────────────────────────────────

SIB200_LABELS = [
    "science/technology", "travel", "politics", "sports",
    "health", "entertainment", "geography",
]

def eval_sib200(model, tokenizer, device) -> dict:
    ds      = load_dataset("Davlan/sib200", SIB200_CONFIG, split="test")
    correct = 0
    for row in tqdm(ds, desc="  SIB-200", leave=False):
        text = row["text"]
        best_score, best_label = float("-inf"), None
        for label in SIB200_LABELS:
            score = score_causal_normalized(model, tokenizer, f"{text}\nవిషయం: {label}", device)
            if score > best_score:
                best_score, best_label = score, label
        if best_label == row["category"]:
            correct += 1
    acc = correct / len(ds)
    print(f"    SIB-200 [{SIB200_CONFIG}]: {acc:.4f}")
    return {"accuracy": round(acc, 4)}


# ── MuBench ────────────────────────────────────────────────────────────────────

def eval_mubench(model, tokenizer, device, mubench_dataset_id: str) -> dict:
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
                n = max(len(tokenizer.encode(choice, add_special_tokens=False)), 1)
                s = score_causal(model, tokenizer, prompt + choice, device) / n
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
    parser = argparse.ArgumentParser(description="Small model Telugu evaluation")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--evals",  nargs="+",
                        choices=["perplexity", "sib200", "mubench"],
                        default=["perplexity", "sib200", "mubench"])
    parser.add_argument("--mubench_dataset", default="aialt/MuBench")
    parser.add_argument("--cleaned_dir", default=None)
    parser.add_argument("--max_ppl_samples", type=int, default=0)
    parser.add_argument("--max_seq_len",     type=int, default=128)
    parser.add_argument("--output", default="results_small_telugu.json")
    args = parser.parse_args()

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_stem = Path(args.output).stem
    out_dir  = Path(args.output).parent
    eval_results = {e: {} for e in args.evals}

    def save_eval(eval_name, model_name, data):
        path = out_dir / f"{out_stem}_{eval_name}.json"
        merged = json.loads(path.read_text()) if path.exists() else {}
        merged[model_name] = data
        eval_results[eval_name] = merged
        path.write_text(json.dumps(merged, indent=2))
        print(f"  Saved → {path}")

    for name in args.models:
        cfg       = MODELS[name]
        model_id  = cfg["model_id"]
        subfolder = cfg["subfolder"]
        arch      = cfg["arch"]

        print(f"\n{'='*60}")
        print(f"Model : {name}  ({model_id}/{subfolder})  [{arch}]")
        print(f"{'='*60}")

        model, tokenizer, _ = load_model(model_id, arch, device, subfolder)

        if "perplexity" in args.evals:
            save_eval("perplexity", name,
                      eval_perplexity(model, tokenizer, device,
                                      args.cleaned_dir, args.max_ppl_samples or None,
                                      args.max_seq_len))
        if "sib200" in args.evals:
            save_eval("sib200", name, eval_sib200(model, tokenizer, device))

        if "mubench" in args.evals:
            save_eval("mubench", name,
                      eval_mubench(model, tokenizer, device, args.mubench_dataset))

        del model
        torch.cuda.empty_cache()

    print(f"\nDone. Results in {out_dir}/{out_stem}_*.json")


if __name__ == "__main__":
    main()
