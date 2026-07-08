"""OpenSubtitles perplexity evaluation for general-purpose baselines (Sarvam-2B, Llama-3.2-1B)."""

import math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm.auto import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

# ── Config ────────────────────────────────────────────────────────────────────
MAX_SAMPLES = None  # None = use full dataset split
MAX_SEQ_LEN = 128
BATCH_SIZE  = 64
# ──────────────────────────────────────────────────────────────────────────────

# Models to evaluate
MODELS = {
    'sarvam2b':   'sarvamai/sarvam-2b-v0.5',
    'llama32_1b': 'meta-llama/Llama-3.2-1B',
}

# Which OS languages to evaluate for each model
EVAL_LANGS = {
    'sarvam2b':   ['en', 'hi', 'te'],
    'llama32_1b': ['en', 'hi', 'te'],
}


def load_os_texts(lang1, lang2, side):
    """Load OPUS-100 parallel text for one language side."""
    config = f"{lang1}-{lang2}"
    print(f'Loading Helsinki-NLP/opus-100 ({config}), extracting {side} side ...')
    ds = load_dataset(
        'Helsinki-NLP/opus-100',
        config,
        split='train',
        streaming=True,
    )
    texts = []
    for row in ds:
        text = row['translation'][side].strip()
        if text:
            texts.append(text)
        if MAX_SAMPLES is not None and len(texts) >= MAX_SAMPLES:
            break
    print(f'  Loaded {len(texts):,} lines')
    return texts


def compute_perplexity(model, tokenizer, texts):
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for i in tqdm(range(0, len(texts), BATCH_SIZE), leave=False):
        batch = texts[i:i + BATCH_SIZE]
        enc = tokenizer(
            batch, return_tensors='pt', padding=True,
            truncation=True, max_length=MAX_SEQ_LEN,
        )
        input_ids = enc['input_ids'].to(DEVICE)
        attn_mask = enc['attention_mask'].to(DEVICE)
        with torch.no_grad():
            logits = model(input_ids, attention_mask=attn_mask).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask   = attn_mask[:, 1:].contiguous().float()
        log_probs    = F.log_softmax(shift_logits, dim=-1)
        token_ll     = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        total_nll    += -(token_ll * shift_mask).sum().item()
        total_tokens += shift_mask.sum().item()
    return math.exp(total_nll / total_tokens) if total_tokens > 0 else float('inf')


def main():
    # Load OS data for each language once
    os_data = {}
    os_data['en'] = load_os_texts('en', 'hi', 'en')   # English side from EN-HI
    os_data['hi'] = load_os_texts('en', 'hi', 'hi')   # Hindi side from EN-HI
    os_data['te'] = load_os_texts('en', 'te', 'te')   # Telugu side from EN-TE

    results = {}

    for model_key, repo in MODELS.items():
        print(f'\n=== {model_key} ({repo}) ===')
        tokenizer = AutoTokenizer.from_pretrained(repo)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(repo).to(DEVICE)
        results[model_key] = {}

        for lang in EVAL_LANGS[model_key]:
            ppl = compute_perplexity(model, tokenizer, os_data[lang])
            results[model_key][lang] = round(ppl, 2)
            print(f'  OS-{lang}: {ppl:.2f}')

        del model
        torch.cuda.empty_cache()

    print('\n=== OpenSubtitles Perplexity Results ===')
    print(f'{"Model":<12} {"OS-EN":>10} {"OS-HI":>10} {"OS-TE":>10}')
    print('-' * 45)
    for model_key in MODELS:
        r = results[model_key]
        en_val = f"{r.get('en', '—'):>10}" if 'en' in r else f"{'—':>10}"
        hi_val = f"{r.get('hi', '—'):>10}" if 'hi' in r else f"{'—':>10}"
        te_val = f"{r.get('te', '—'):>10}" if 'te' in r else f"{'—':>10}"
        print(f'{model_key:<12} {en_val} {hi_val} {te_val}')


if __name__ == '__main__':
    main()
