"""
Download and clean the three monolingual evaluation datasets from HuggingFace.

Applies the same cleaning pipeline used during training:
  - Unicode NFC normalization
  - Zero-width / BOM character removal
  - Control character replacement (ASCII < 32, DEL → space)
  - Non-printable character removal
  - Whitespace normalization
  - Empty line removal

Usage:
    python clean_dataset.py                        # clean all three datasets
    python clean_dataset.py --lang en              # English only
    python clean_dataset.py --lang hi              # Hindi only
    python clean_dataset.py --lang te              # Telugu only
    python clean_dataset.py --output_dir cleaned   # custom output dir
"""

import argparse
import re
import unicodedata
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


ZERO_WIDTH_AND_BOM = ["﻿", "​", "‌", "‍", "⁠"]

DATASETS = {
    "en": {
        "repo": "BabyLM-community/BabyLM-Test",
        "split": None,
        "label": "english",
    },
    "hi": {
        "repo": "pulipakav-1/translated-babylm-hindi",
        "split": "test",
        "label": "hindi",
    },
    "te": {
        "repo": "pulipakav-1/translated-babylm-telugu",
        "split": "test",
        "label": "telugu",
    },
}


def clean_preserve_punctuation(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text))
    for bad in ZERO_WIDTH_AND_BOM:
        text = text.replace(bad, "")
    cleaned_chars = []
    for ch in text:
        code = ord(ch)
        if code < 32 or code == 127:
            cleaned_chars.append(" ")
        elif ch.isprintable():
            cleaned_chars.append(ch)
    cleaned = "".join(cleaned_chars)
    return re.sub(r"\s+", " ", cleaned).strip()


def clean_lang(lang: str, output_dir: Path) -> None:
    cfg = DATASETS[lang]
    repo  = cfg["repo"]
    split = cfg["split"]
    label = cfg["label"]

    print(f"\n[{lang.upper()}] Loading {repo}" + (f" (split={split})" if split else "") + " ...")
    if lang == "en":
        # .test files are not auto-detected; load explicitly as text
        ds = load_dataset("text",
                          data_files={"test": f"hf://datasets/{repo}/*.test"},
                          split="test")
        splits = {"test": ds}
    elif split:
        ds = load_dataset(repo, split=split)
        splits = {split: ds}
    else:
        ds = load_dataset(repo)
        splits = dict(ds.items()) if hasattr(ds, "items") else {"data": ds}

    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, dataset in splits.items():
        original_count = len(dataset)
        cleaned_lines = []

        for row in tqdm(dataset, desc=f"  Cleaning {label}/{split_name}"):
            text = row.get("text") or row.get("sentence") or ""
            cleaned = clean_preserve_punctuation(text)
            if cleaned:
                cleaned_lines.append(cleaned)

        dropped = original_count - len(cleaned_lines)
        out_path = output_dir / f"{label}_{split_name}.txt"
        out_path.write_text("\n".join(cleaned_lines), encoding="utf-8")

        print(f"  [{lang.upper()}] {split_name}: {original_count:,} rows → "
              f"{len(cleaned_lines):,} kept, {dropped:,} dropped → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Clean monolingual eval datasets from HuggingFace.")
    parser.add_argument(
        "--lang", nargs="+", choices=["en", "hi", "te"], default=["en", "hi", "te"],
        help="Languages to process (default: all three)"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("cleaned"),
        help="Directory to write cleaned .txt files (default: cleaned/)"
    )
    args = parser.parse_args()

    for lang in args.lang:
        clean_lang(lang, args.output_dir)

    print(f"\nDone. Cleaned files written to {args.output_dir}/")


if __name__ == "__main__":
    main()
