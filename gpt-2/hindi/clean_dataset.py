from __future__ import annotations

import argparse
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


def source_name_from_path(path: Path) -> str:
    name = path.name
    if '.train.' in name:
        return name.split('.train.')[0]
    if '.train' in name:
        return name.split('.train')[0]
    return path.stem


def resolve_dataset_dir(dataset_name: str, data_root: Path = Path('data')) -> Path:
    base_dir = data_root / dataset_name
    train_dir = base_dir / 'train'
    base_has_files = any(base_dir.glob('*.train*.txt')) or any(base_dir.glob('*.train*.parquet'))
    train_has_files = any(train_dir.glob('*.train*.txt')) or any(train_dir.glob('*.train*.parquet'))
    if base_has_files:
        return base_dir
    if train_has_files:
        return train_dir
    return base_dir


def clean_parquet_files(input_dir, output_dir, files):
    for pf in files:
        df = pd.read_parquet(pf)
        original_chars = df['text'].str.len().sum()
        df['text'] = df['text'].apply(clean_preserve_punctuation)
        cleaned_chars = df['text'].str.len().sum()
        out_path = output_dir / pf.name
        df.to_parquet(out_path, index=False)
        print(f'  Cleaned {pf.name}: {original_chars:,} -> {cleaned_chars:,} chars ({len(df)} rows)')


def clean_text_files(input_dir, output_dir, files):
    for tf in files:
        text = tf.read_text(encoding='utf-8', errors='replace')
        original_len = len(text)
        cleaned_lines = [c for l in text.splitlines() if (c := clean_preserve_punctuation(l))]
        cleaned_text = '\n'.join(cleaned_lines)
        out_path = output_dir / tf.name
        out_path.write_text(cleaned_text, encoding='utf-8')
        print(f'  Cleaned {tf.name}: {original_len:,} -> {len(cleaned_text):,} chars')


def clean_dataset(dataset_name: str, data_root: Path = Path('data')):
    input_dir = resolve_dataset_dir(dataset_name, data_root)
    parquet_files = sorted(input_dir.glob('*.train*.parquet'))
    text_files = sorted(input_dir.glob('*.train*.txt'))
    if parquet_files:
        clean_parquet_files(input_dir, input_dir, parquet_files)
    elif text_files:
        clean_text_files(input_dir, input_dir, text_files)
    else:
        raise FileNotFoundError(f'No .train.parquet or .train.txt files found in {input_dir}')
    print(f'Done. Files overwritten in {input_dir}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clean a BabyLM dataset.')
    parser.add_argument('dataset', help='Dataset folder name under data/ (e.g. babylm-nld)')
    args = parser.parse_args()
    clean_dataset(args.dataset)
