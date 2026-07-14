"""Download and byte-trim native (non-translated) Hindi and Telugu text from
ai4bharat/IndicCorpV2 (CC-0 licensed) for training monolingual GPT-BERT models on
naturally-written data, as a comparison point against the project's GPT-5-mini
translated-data models.

IndicCorpV2 is far larger than needed (~80GB Hindi, ~15.76GB Telugu), so this streams
through each language's split and stops the moment it collects TARGET_BYTES of raw UTF-8
text -- no need to download the full files. Hindi and Telugu are trimmed to the exact same
byte count (matching this project's existing translated-data scale by default: ~1.38GB for
Hindi, ~1.47GB for Telugu in the current translated corpora -- TARGET_BYTES below defaults
to the smaller of the two, 1,381,881,024 bytes, so both native corpora end up truly
byte-matched to each other).

Usage:
    python prepare_native_indic_data.py
    python prepare_native_indic_data.py --target-bytes 1000000000   # 1GB each, for a quicker run
    python prepare_native_indic_data.py --out-dir native_indic_data
"""

import argparse
from pathlib import Path

from datasets import load_dataset

REPO_ID = "ai4bharat/IndicCorpV2"
CONFIG_NAME = "indiccorp_v2"
SPLITS = {"hi": "hin_Deva", "te": "tel_Telu"}

# Defaults to this project's existing Hindi/Telugu *translated* corpus byte counts, so the
# native-data models end up trained on the same amount of text as the translated-data ones --
# the smaller of the two (Hindi's 1,381,881,024 bytes) is used for both, so Hindi and Telugu
# stay byte-matched to each other as well.
DEFAULT_TARGET_BYTES = 1_381_881_024


def collect_native_text(lang_code, split_name, target_bytes, out_path):
    print(f"[{lang_code}] streaming {REPO_ID} split={split_name} until {target_bytes:,} bytes...")
    ds = load_dataset(REPO_ID, CONFIG_NAME, split=split_name, streaming=True)

    collected_bytes = 0
    n_lines = 0
    n_words = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            text = row.get("text", "")
            if not text:
                continue
            line = text.strip()
            if not line:
                continue
            line_bytes = len(line.encode("utf-8")) + 1  # +1 for the newline we write
            if collected_bytes + line_bytes > target_bytes:
                break
            f.write(line + "\n")
            collected_bytes += line_bytes
            n_lines += 1
            n_words += len(line.split())
            if n_lines % 200_000 == 0:
                print(f"  [{lang_code}] {collected_bytes:,}/{target_bytes:,} bytes "
                      f"({n_lines:,} lines) so far...", flush=True)

    print(f"[{lang_code}] done: {collected_bytes:,} bytes, {n_lines:,} lines, "
          f"{n_words:,} words -> {out_path}")
    return collected_bytes, n_lines, n_words


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-bytes", type=int, default=DEFAULT_TARGET_BYTES,
                        help="Byte count each language is trimmed to (default: matches "
                             "this project's existing Hindi translated-corpus size)")
    parser.add_argument("--out-dir", default="native_indic_data")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    stats = {}
    for lang_code, split_name in SPLITS.items():
        out_path = out_dir / f"native_{lang_code}.txt"
        stats[lang_code] = collect_native_text(lang_code, split_name, args.target_bytes, out_path)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for lang_code, (b, lines, words) in stats.items():
        print(f"  {lang_code}: {b:,} bytes | {lines:,} lines | {words:,} words | "
              f"{words / max(b, 1):.5f} words/byte (1:{b / max(words, 1):.2f} word:byte)")


if __name__ == "__main__":
    main()
