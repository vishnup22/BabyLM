from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


ZERO_WIDTH_AND_BOM = ["﻿", "​", "‌", "‍", "⁠"]


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


def count_txt(path: Path) -> tuple[int, int]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    before = sum(len(l.split()) for l in lines)
    after = sum(len(c.split()) for l in lines if (c := clean_preserve_punctuation(l)))
    return before, after


def count_parquet(path: Path) -> tuple[int, int]:
    texts = pd.read_parquet(path)["text"].astype(str).tolist()
    before = sum(len(t.split()) for t in texts)
    after = sum(len(clean_preserve_punctuation(t).split()) for t in texts)
    return before, after


def main() -> None:
    p = argparse.ArgumentParser(description="Count words before and after cleaning.")
    p.add_argument("dataset", help="Dataset folder name under --data-root")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("word_counts.json"))
    args = p.parse_args()

    input_dir = args.data_root / args.dataset
    results: dict = {}
    total_before = total_after = 0

    for pf in sorted(input_dir.glob("*.train.parquet")):
        b, a = count_parquet(pf)
        results[pf.name] = {"before": b, "after": a, "removed": b - a}
        total_before += b
        total_after += a
        print(f"  {pf.name}: {b:,} -> {a:,} words")

    for tf in sorted(input_dir.glob("*.train*.txt")):
        b, a = count_txt(tf)
        results[tf.name] = {"before": b, "after": a, "removed": b - a}
        total_before += b
        total_after += a
        print(f"  {tf.name}: {b:,} -> {a:,} words")

    results["__total__"] = {
        "before": total_before,
        "after": total_after,
        "removed": total_before - total_after,
    }

    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nTotal: {total_before:,} -> {total_after:,} words ({total_before - total_after:,} removed)")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
