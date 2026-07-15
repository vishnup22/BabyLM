"""English-only clean (overlap-filtered) OS-data perplexity.

Filters OPUS-100 en-hi's English side down to lines that do NOT appear in BabyLM's
own open_subtitles.train.txt (the ~25% overlap that inflated the original English
OS-data perplexity numbers), then recomputes perplexity on the clean subset for
every model that has an English OS-data number in evaluation.md.

Run locally (single GPU, no accelerate launch needed):
    python eval_english_clean_ppl.py
"""

import json
import math
import os
import re
import sys
import time
from pathlib import Path

import requests
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    REPO_ROOT = Path(__file__).parent
except NameError:
    # __file__ isn't defined when this is pasted/run inside a notebook cell
    # (Colab) instead of executed as a .py script -- fall back to cwd.
    REPO_ROOT = Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

MAX_SEQ_LEN = 128
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_PATH = REPO_ROOT / "results_english_clean_ppl.json"

env_path = REPO_ROOT / ".env"
hf_token = None
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if line.startswith("HF_TOKEN="):
            hf_token = line.split("=", 1)[1].strip()
hf_token = hf_token or os.environ.get("HF_TOKEN", "")
os.environ["HF_TOKEN"] = hf_token
AUTH_HEADERS = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}


def normalize(line):
    line = line.strip().lower()
    return re.sub(r"\s+", " ", line)


def build_clean_english_texts():
    print("Downloading BabyLM open_subtitles.train.txt (English) ...")
    url = "https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict/resolve/main/open_subtitles.train.txt"
    resp = requests.get(url, headers=AUTH_HEADERS, timeout=300)
    resp.raise_for_status()
    babylm_en = {normalize(l) for l in resp.text.splitlines() if normalize(l)}
    print(f"  {len(babylm_en):,} unique normalized BabyLM English lines")

    print("Streaming OPUS-100 en-hi (English side only) ...")
    ds = load_dataset("Helsinki-NLP/opus-100", "en-hi", split="train", streaming=True)
    clean, total = [], 0
    for row in ds:
        en_text = row["translation"]["en"].strip()
        if not en_text:
            continue
        total += 1
        if normalize(en_text) not in babylm_en:
            clean.append(en_text)
    print(f"  Full: {total:,} | Clean: {len(clean):,} ({100*len(clean)/max(total,1):.2f}% retained, "
          f"{100*(1 - len(clean)/max(total,1)):.2f}% overlap)")
    return clean, total


def compute_causal_perplexity_hf(model, tokenizer, texts):
    total_nll, total_tokens = 0.0, 0
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="  perplexity"):
        batch = texts[i:i + BATCH_SIZE]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LEN)
        input_ids = enc["input_ids"].to(DEVICE)
        attn_mask = enc["attention_mask"].to(DEVICE)
        with torch.no_grad():
            logits = model(input_ids, attention_mask=attn_mask).logits
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attn_mask[:, 1:].contiguous().float()
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_ll = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        total_nll += -(token_ll * shift_mask).sum().item()
        total_tokens += shift_mask.sum().item()
    return math.exp(total_nll / total_tokens) if total_tokens > 0 else float("inf")


# smallest/fastest models first so partial results land early
HF_MODEL_SPECS = [
    ("GPT-Wee mono-en", "pulipakav-1/gpt2-english-babylm2026", False),
    ("GPT-Wee eng-hin", "pulipakav-1/gpt2-hin-eng_babylm2026", False),
    ("GPT-Wee eng-tel", "pulipakav-1/gpt2-tel-eng_babylm2026", False),
    ("Llama 3.2 1B", "meta-llama/Llama-3.2-1B", True),
    ("Sarvam-2B", "sarvamai/sarvam-2b-v0.5", True),
]

GPTBERT_MODEL_SPECS = [
    ("GPT-BERT mono-en", "mono_en"),
    ("GPT-BERT eng-hin", "en_hi_seed1"),
    ("GPT-BERT eng-tel", "en_tel_seed2"),
]


def save_results(results, clean_n, total_n):
    RESULTS_PATH.write_text(json.dumps({
        "clean_lines": clean_n, "full_lines": total_n,
        "overlap_pct": round(100 * (1 - clean_n / max(total_n, 1)), 4),
        "results": results,
    }, indent=2))


def main():
    t0 = time.time()
    clean_texts, total_n = build_clean_english_texts()
    clean_n = len(clean_texts)
    results = {}

    for label, repo, use_fp16 in HF_MODEL_SPECS:
        print(f"\n=== {label} ({repo}) ===", flush=True)
        m0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(repo, token=hf_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        kwargs = {"torch_dtype": torch.float16} if use_fp16 else {}
        model = AutoModelForCausalLM.from_pretrained(repo, token=hf_token, **kwargs).eval().to(DEVICE)
        ppl = compute_causal_perplexity_hf(model, tokenizer, clean_texts)
        print(f"  clean English OS-data perplexity: {ppl:.4f}  ({time.time()-m0:.1f}s)", flush=True)
        results[label] = ppl
        del model
        torch.cuda.empty_cache()
        save_results(results, clean_n, total_n)

    # GPT-BERT models
    from eval_gptbert_all import load_model_and_tokenizer, compute_causal_perplexity, MODEL_REGISTRY
    for label, model_key in GPTBERT_MODEL_SPECS:
        print(f"\n=== {label} ({model_key}) ===", flush=True)
        m0 = time.time()
        spec = MODEL_REGISTRY[model_key]
        model, tokenizer, cls_id, mask_id, pad_id = load_model_and_tokenizer(spec)
        ppl = compute_causal_perplexity(model, tokenizer, clean_texts, cls_id, pad_id)
        print(f"  clean English OS-data perplexity: {ppl:.4f}  ({time.time()-m0:.1f}s)", flush=True)
        results[label] = ppl
        del model
        torch.cuda.empty_cache()
        save_results(results, clean_n, total_n)

    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
    print("\n=== Summary (clean English OS-data perplexity) ===")
    for label, ppl in results.items():
        print(f"  {label:20s}: {ppl:.4f}")
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
