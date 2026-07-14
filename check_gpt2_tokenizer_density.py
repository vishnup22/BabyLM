"""Measure GPT-Wee (GPT-2) tokenizer chars/token density for English/Hindi/Telugu, using the
same matched test sentences already used to measure GPT-BERT/Llama/Sarvam density -- so results
are directly comparable. Run this on the cluster, where the actual tokenizer files live (not
committed to git).

Usage:
    python check_gpt2_tokenizer_density.py
"""

import glob
from pathlib import Path

from transformers import PreTrainedTokenizerFast

REPO_ROOT = Path(__file__).parent

# Same sentences used for the GPT-BERT/Llama/Sarvam density comparison earlier.
SENTENCES = {
    "en": "The quick brown fox jumps over the lazy dog while the children play happily in the park.",
    "hi": "तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर से कूदती है जबकि बच्चे पार्क में खुशी से खेलते हैं।",
    "te": "వేగవంతమైన గోధుమరంగు నక్క సోమరి కుక్క మీదుగా దూకుతుంది, పిల్లలు పార్కులో సంతోషంగా ఆడుకుంటారు.",
}

# Candidate tokenizer locations to search, per model. Adjust/add paths here if your layout
# differs -- this globs broadly so it should find the right file regardless of exact naming.
SEARCH_ROOTS = {
    "GPT-Wee mono-en": ("Babylm2026/gpt-2/english", "en"),
    "GPT-Wee mono-hi": ("Babylm2026/gpt-2/hindi", "hi"),
    "GPT-Wee mono-te": ("Babylm2026/gpt-2/telugu", "te"),
    "GPT-Wee eng-hin (en)": ("Babylm2026/eng-hin/gpt2 multi", "en"),
    "GPT-Wee eng-hin (hi)": ("Babylm2026/eng-hin/gpt2 multi", "hi"),
    "GPT-Wee eng-tel (en)": ("Babylm2026/eng-tel/gpt2 multi", "en"),
    "GPT-Wee eng-tel (te)": ("Babylm2026/eng-tel/gpt2 multi", "te"),
}


def find_tokenizer_file(search_dir):
    candidates = glob.glob(str(REPO_ROOT / search_dir / "**" / "tokenizer*.json"), recursive=True)
    # Prefer files that aren't just a tokenizer_config.json (that's config, not the vocab/merges file)
    real = [c for c in candidates if Path(c).name != "tokenizer_config.json"]
    return real[0] if real else (candidates[0] if candidates else None)


def density(tok, text):
    ids = tok.encode(text, add_special_tokens=False)
    return len(text), len(ids), len(text) / max(len(ids), 1)


print(f"{'Model':28s} {'lang':5s} {'chars':>6s} {'tokens':>7s} {'chars/token':>12s}")
print("-" * 64)
for label, (search_dir, lang) in SEARCH_ROOTS.items():
    tok_path = find_tokenizer_file(search_dir)
    if tok_path is None:
        print(f"{label:28s} {lang:5s}  NOT FOUND under {search_dir}")
        continue
    try:
        tok = PreTrainedTokenizerFast(tokenizer_file=tok_path)
        chars, n_tok, cpt = density(tok, SENTENCES[lang])
        print(f"{label:28s} {lang:5s} {chars:6d} {n_tok:7d} {cpt:12.3f}   ({tok_path})")
    except Exception as e:
        print(f"{label:28s} {lang:5s}  ERROR loading {tok_path}: {e}")
