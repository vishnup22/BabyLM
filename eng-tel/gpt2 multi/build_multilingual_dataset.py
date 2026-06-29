"""
Build a multilingual BabyLM dataset by sampling from per-language sources.

Usage:
    python build_multilingual_dataset.py --languages en nld zho \
        --words-per-lang 50000000 --output data/multilingual-50M

Each language is sampled to just under (words_per_lang * byte_premium) words,
preserving the proportion of words from each data source within that language.

- English (.txt): preserves continuity by taking contiguous text from the start.
- NLD/ZHO (.parquet): keeps whole rows when sampling.

Output files are prefixed with the language code to avoid name conflicts.
"""

import argparse
from pathlib import Path

import pandas as pd


BYTE_PREMIUMS = {
    'en':  1.0,
    'hi':  2.5990,    # measured: 11.6M aligned pairs, std dev 0.0008
    'te':  2.7672,    # measured: 11.6M aligned pairs, std dev 0.0009
    'nld': 1.0516,
    'zho': 0.935966,
}

LANGUAGE_DATASETS = {
    'en': 'BabyLM-2026-Strict',
    'te': 'translated-babylm-telugu/train',
    'nld': 'babylm-nld',
    'zho': 'babylm-zho',
}

# Only these sources are used per language; None means use all available.
INCLUDED_SOURCES = {
    'en': {'bnc_spoken', 'open_subtitles', 'simple_wiki', 'switchboard'},
    'te': {'childes', 'gutenberg'},
}

# Glob pattern and suffix to strip when extracting source names from filenames.
TEXT_FILE_PATTERN = {
    'en': ('*.train.txt',    '.train.txt'),
    'te': ('*.train.te.txt', '.train.te.txt'),
}


def count_words_text(text: str) -> int:
    return len(text.split())


def load_text_sources(data_dir: Path, lang: str) -> dict[str, str]:
    """Load .txt sources for a language, filtered to INCLUDED_SOURCES[lang]."""
    allowed = INCLUDED_SOURCES.get(lang)
    glob_pattern, suffix = TEXT_FILE_PATTERN.get(lang, ('*.train.txt', '.train.txt'))
    sources = {}
    for f in sorted(data_dir.glob(glob_pattern)):
        source_name = f.name.replace(suffix, '')
        if allowed is not None and source_name not in allowed:
            print(f'  [{lang}] Skipping source not in inclusion list: {source_name}')
            continue
        sources[source_name] = f.read_text(encoding='utf-8')
    return sources


def load_parquet_sources(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load parquet sources, returning {source_name: dataframe}."""
    sources = {}
    for f in sorted(data_dir.glob('*.train.parquet')):
        source_name = f.name.replace('.train.parquet', '')
        sources[source_name] = pd.read_parquet(f)
    return sources


def sample_english_source(text: str, target_words: int) -> str:
    """Take a contiguous chunk from the start, truncating at a word boundary
    to land just under target_words."""
    words = text.split()
    if len(words) <= target_words:
        return text
    # Rejoin the first target_words words
    # Find the byte offset of the end of the target_words-th word
    # to preserve original whitespace/newlines
    count = 0
    end_pos = 0
    for match in __import__('re').finditer(r'\S+', text):
        count += 1
        if count >= target_words:
            end_pos = match.end()
            break
    return text[:end_pos]


def sample_parquet_source(df: pd.DataFrame, target_words: int) -> pd.DataFrame:
    """Take whole rows from the start until just under target_words (using num-tokens)."""
    cumulative = df['num-tokens'].cumsum()
    # Find the last row where cumulative sum is still under target
    mask = cumulative <= target_words
    if not mask.any():
        # Even the first row exceeds target — take nothing
        return df.iloc[:0]
    last_idx = mask.values.nonzero()[0][-1]
    return df.iloc[:last_idx + 1]


def build_language(lang: str, target_words: int, data_root: Path, output_dir: Path,
                   raw: bool = False):
    adjusted_target = target_words if raw else int(target_words * BYTE_PREMIUMS[lang])
    dataset_dir = data_root / LANGUAGE_DATASETS[lang]
    uses_text_files = lang in {'en', 'te'}

    if raw:
        print(f'\n  [{lang}] target: {target_words:,} words (raw, no byte premium)')
    else:
        print(f'\n  [{lang}] target: {target_words:,} words, '
              f'adjusted (x{BYTE_PREMIUMS[lang]}): {adjusted_target:,} words')

    if uses_text_files:
        sources = load_text_sources(dataset_dir, lang)
        # Count words per source
        source_word_counts = {name: count_words_text(text) for name, text in sources.items()}
    else:
        sources = load_parquet_sources(dataset_dir)
        # Use num-tokens column for word counts
        source_word_counts = {name: int(df['num-tokens'].sum()) for name, df in sources.items()}

    total_words = sum(source_word_counts.values())
    print(f'  [{lang}] total available: {total_words:,} words across {len(sources)} sources')

    if adjusted_target >= total_words:
        print(f'  [{lang}] WARNING: target ({adjusted_target:,}) >= available ({total_words:,}), '
              f'using all data')

    # Calculate per-source targets preserving proportions
    source_targets = {}
    for name, wc in source_word_counts.items():
        proportion = wc / total_words
        source_targets[name] = int(adjusted_target * proportion)

    # Sample and save each source
    total_sampled = 0
    for name in sorted(sources.keys()):
        target = source_targets[name]
        out_name_prefix = f'{lang}_{name}'

        if uses_text_files:
            sampled_text = sample_english_source(sources[name], target)
            sampled_words = count_words_text(sampled_text)
            out_path = output_dir / f'{out_name_prefix}.train.txt'
            out_path.write_text(sampled_text)
        else:
            sampled_df = sample_parquet_source(sources[name], target)
            sampled_words = int(sampled_df['num-tokens'].sum()) if len(sampled_df) > 0 else 0
            out_path = output_dir / f'{out_name_prefix}.train.parquet'
            sampled_df.to_parquet(out_path, index=False)

        total_sampled += sampled_words
        available = source_word_counts[name]
        print(f'    {name}: {sampled_words:,} / {available:,} words '
              f'(target {target:,}) -> {out_path.name}')

    print(f'  [{lang}] total sampled: {total_sampled:,} / adjusted target {adjusted_target:,}')


def count_available_words(lang: str, data_root: Path) -> int:
    """Count total words available for a language across its included sources."""
    dataset_dir = data_root / LANGUAGE_DATASETS[lang]
    if lang in {'en', 'te'}:
        sources = load_text_sources(dataset_dir, lang)
        return sum(count_words_text(text) for text in sources.values())
    else:
        sources = load_parquet_sources(dataset_dir)
        return sum(int(df['num-tokens'].sum()) for df in sources.values())


def main():
    parser = argparse.ArgumentParser(
        description='Build a multilingual BabyLM dataset.')
    parser.add_argument('--languages', nargs='+', required=True,
                        choices=list(LANGUAGE_DATASETS.keys()),
                        help='Languages to include (en, hi, nld, zho)')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--words-per-lang', type=int,
                       help='Target words per language (before byte premium)')
    group.add_argument('--total-words', type=int,
                       help='Total target words across all languages. '
                            'Non-English languages take all available; English fills the rest.')
    group.add_argument('--lang-words', nargs='+', metavar='LANG=N',
                       help='Exact raw word targets per language, e.g. en=46000000 hi=54000000')
    parser.add_argument('--output', type=str,
                        help='Output directory (e.g. data/en_tel_equal). Not required with --count-only.')
    parser.add_argument('--data-root', type=str, default='data',
                        help='Root data directory (default: data)')
    parser.add_argument('--count-only', action='store_true',
                        help='Just count available words per language and exit without building.')
    args = parser.parse_args()

    data_root = Path(args.data_root)

    if args.count_only:
        print(f'Available words per language (data-root: {data_root}):')
        total = 0
        for lang in args.languages:
            n = count_available_words(lang, data_root)
            total += n
            print(f'  [{lang}] {n:,} words')
        print(f'  [total] {total:,} words')
        return

    if not args.output:
        parser.error('--output is required unless --count-only is set.')

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.lang_words:
        per_lang = {}
        for item in args.lang_words:
            lang, n = item.split('=')
            per_lang[lang] = int(n)
        total = sum(per_lang.values())
        print(f'Building multilingual dataset: {args.languages}')
        print(f'Per-language raw targets (total {total:,}) -> {args.output}')
        for lang in args.languages:
            build_language(lang, per_lang[lang], data_root, output_dir, raw=True)

    elif args.total_words:
        # Count all non-English languages first, English fills the remainder.
        non_en_langs = [l for l in args.languages if l != 'en']
        non_en_words = 0
        per_lang_targets = {}

        print(f'Building multilingual dataset: {args.languages}')
        print(f'Total target: {args.total_words:,} words -> {args.output}')

        for lang in non_en_langs:
            available = count_available_words(lang, data_root)
            per_lang_targets[lang] = available
            non_en_words += available
            print(f'  [{lang}] available: {available:,} words (taking all)')

        en_target = args.total_words - non_en_words
        if 'en' in args.languages:
            per_lang_targets['en'] = en_target
            print(f'  [en] target: {en_target:,} words ({args.total_words:,} - {non_en_words:,} non-en)')

        for lang in args.languages:
            build_language(lang, per_lang_targets[lang], data_root, output_dir)
    else:
        print(f'Building multilingual dataset: {args.languages}')
        print(f'Target: {args.words_per_lang:,} words/lang -> {args.output}')
        for lang in args.languages:
            build_language(lang, args.words_per_lang, data_root, output_dir)

    print(f'\nDone. Output written to {output_dir}/')


if __name__ == '__main__':
    main()
