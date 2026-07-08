from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# Maps language name -> dataset folder name under {language}/data/
DATASET_NAMES = {
    "english": "BabyLM-2026-Strict",
    "hindi":   "translated-babylm-hindi",
    "telugu":  "translated-babylm-telugu",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create curriculum-ordered train files for all languages.")
    p.add_argument("--languages", nargs="+", default=["english", "hindi", "telugu"])
    return p.parse_args()


def load_lines_from_data_dir(data_dir: Path) -> list[str]:
    """Read all .train.txt and .train.parquet files and return nonempty lines."""
    lines = []

    for f in sorted(data_dir.glob("*.train*.txt")):
        for ln in f.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln:
                lines.append(ln)

    for f in sorted(data_dir.glob("*.train*.parquet")):
        df = pd.read_parquet(f)
        for text in df["text"].tolist():
            for ln in str(text).splitlines():
                ln = ln.strip()
                if ln:
                    lines.append(ln)

    return lines


def resolve_data_dir(language_dir: Path, dataset_name: str) -> Path:
    base = language_dir / "data" / dataset_name
    train_sub = base / "train"
    if any(train_sub.glob("*.train*.txt")) or any(train_sub.glob("*.train*.parquet")):
        return train_sub
    return base


def curriculum_order(lines: list[str]) -> pd.DataFrame:
    df = pd.DataFrame({"text": lines})
    df["words"] = df["text"].str.split()
    df["utterance_length"] = df["words"].str.len()
    df["mean_word_length"] = df["words"].apply(
        lambda ws: float(np.mean([len(w) for w in ws])) if ws else 0.0
    )

    sentence_freq = Counter(df["text"].tolist())
    df["frame_freq"] = df["text"].map(sentence_freq)

    word_freq = Counter()
    for ws in df["words"]:
        word_freq.update(ws)
    df["mean_word_freq"] = df["words"].apply(
        lambda ws: float(np.mean([word_freq[w] for w in ws])) if ws else 0.0
    )

    df["rank_frame_freq"] = df["frame_freq"].rank(method="average", ascending=False)
    df["rank_utter_len"] = df["utterance_length"].rank(method="average", ascending=True)
    df["rank_mean_word_len"] = df["mean_word_length"].rank(method="average", ascending=True)
    df["rank_mean_word_freq"] = df["mean_word_freq"].rank(method="average", ascending=False)
    df["final_rank"] = (
        df["rank_frame_freq"]
        + df["rank_utter_len"]
        + df["rank_mean_word_len"]
        + df["rank_mean_word_freq"]
    )
    return df.sort_values("final_rank", ascending=True).reset_index(drop=True)


def build_language_files(language: str) -> None:
    dataset_name = DATASET_NAMES[language]
    language_dir = Path(language)
    data_dir = resolve_data_dir(language_dir, dataset_name)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            f"Run scripts/download_dataset.sh from {language_dir}/ first."
        )

    print(f"[{language}] Reading from {data_dir} ...")
    lines = load_lines_from_data_dir(data_dir)
    print(f"[{language}] {len(lines):,} lines loaded — ordering ...")

    ordered_df = curriculum_order(lines)

    curriculum_out = language_dir / "curriculum_data"
    curriculum_out.mkdir(parents=True, exist_ok=True)

    (curriculum_out / f"{language}.txt").write_text(
        "\n".join(ordered_df["text"].tolist()), encoding="utf-8"
    )
    ordered_df[
        ["text", "frame_freq", "utterance_length", "mean_word_length", "mean_word_freq", "final_rank"]
    ].to_csv(curriculum_out / f"{language}_scores.csv", index=False, encoding="utf-8")

    print(f"[{language}] Done -> {curriculum_out / f'{language}.txt'}")


def main() -> None:
    args = parse_args()
    for lang in args.languages:
        build_language_files(lang)


if __name__ == "__main__":
    main()
