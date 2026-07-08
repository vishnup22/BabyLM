from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create curriculum-ordered train files for all languages.")
    p.add_argument("--clean-root", type=Path, default=Path("clean_data"))
    p.add_argument("--train-out", type=Path, default=Path("train_data"))
    p.add_argument("--dev-out", type=Path, default=Path("dev_data"))
    p.add_argument("--test-out", type=Path, default=Path("test_data"))
    p.add_argument("--curriculum-out", type=Path, default=Path("curriculum_data"))
    p.add_argument("--languages", nargs="+", default=["english", "hindi", "telugu"])
    return p.parse_args()


def load_nonempty_lines(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


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


def pick_train_file(clean_root: Path, language: str) -> Path:
    p = clean_root / language / "train" / f"{language}_train_cleaned_final.txt"
    if p.exists():
        return p
    fallback = clean_root / language / f"{language}_cleaned_final.txt"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Missing cleaned train file for {language}")


def build_language_files(
    clean_root: Path,
    train_out: Path,
    dev_out: Path,
    test_out: Path,
    curriculum_out: Path,
    language: str,
) -> None:
    train_file = pick_train_file(clean_root, language)
    val_file = clean_root / language / "val" / f"{language}_val_cleaned_final.txt"
    test_file = clean_root / language / "test" / f"{language}_test_cleaned_final.txt"
    if not val_file.exists() or not test_file.exists():
        raise FileNotFoundError(f"Missing val/test cleaned files for {language}")

    train_lines = load_nonempty_lines(train_file)
    val_lines = load_nonempty_lines(val_file)
    test_lines = load_nonempty_lines(test_file)
    ordered_df = curriculum_order(train_lines)

    train_out.mkdir(parents=True, exist_ok=True)
    dev_out.mkdir(parents=True, exist_ok=True)
    test_out.mkdir(parents=True, exist_ok=True)
    curriculum_out.mkdir(parents=True, exist_ok=True)

    (train_out / f"{language}.txt").write_text("\n".join(train_lines), encoding="utf-8")
    (dev_out / f"{language}.txt").write_text("\n".join(val_lines), encoding="utf-8")
    (test_out / f"{language}.txt").write_text("\n".join(test_lines), encoding="utf-8")
    (curriculum_out / f"{language}.txt").write_text("\n".join(ordered_df["text"].tolist()), encoding="utf-8")
    ordered_df[
        ["text", "frame_freq", "utterance_length", "mean_word_length", "mean_word_freq", "final_rank"]
    ].to_csv(curriculum_out / f"{language}_scores.csv", index=False, encoding="utf-8")

    print(f"[curriculum] {language}:")
    print(f"  train -> {train_out / f'{language}.txt'}")
    print(f"  dev   -> {dev_out / f'{language}.txt'}")
    print(f"  test  -> {test_out / f'{language}.txt'}")
    print(f"  curr  -> {curriculum_out / f'{language}.txt'}")


def main() -> None:
    args = parse_args()
    for lang in args.languages:
        build_language_files(
            args.clean_root,
            args.train_out,
            args.dev_out,
            args.test_out,
            args.curriculum_out,
            lang,
        )


if __name__ == "__main__":
    main()
