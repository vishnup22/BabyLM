"""Download and byte-trim native (non-translated) Hindi and Telugu text from
ai4bharat/IndicCorpV2 (CC-0 licensed) for training monolingual GPT-BERT models on
naturally-written data, as a comparison point against the project's GPT-5-mini
translated-data models.

Fetches a raw HTTP Range request directly against each file's `resolve/main/` URL instead of
going through huggingface_hub's Python client / datasets streaming path -- the latter routes
through HF's newer Xet CDN bridge, which was returning persistent 403/SignatureError failures
for this repo's large files at the time this was written. A plain Range request hits HF's
classic resolve/CDN path instead, which is separate infrastructure.

IndicCorpV2 is far larger than needed (~80GB Hindi across 3 shards, ~15.76GB Telugu in one
file), so this only requests TARGET_BYTES (plus a small overfetch buffer) from the *first*
shard of each language -- no need to touch the rest. Hindi and Telugu are trimmed to the same
byte count (matching this project's existing translated-data scale by default: Hindi's
1,381,881,024 bytes, so both native corpora end up byte-matched to each other too).

Usage:
    python prepare_native_indic_data.py
    python prepare_native_indic_data.py --target-bytes 1000000000   # 1GB each, for a quicker run
    python prepare_native_indic_data.py --out-dir native_indic_data
"""

import argparse
import os
from pathlib import Path

import requests

REPO_ID = "ai4bharat/IndicCorpV2"
# First shard of each language is more than enough on its own (each is multi-GB) for any
# reasonable target size, so we never need to touch hi-2.txt/hi-3.txt.
FILE_PATHS = {"hi": "data/hi-1.txt", "te": "data/te.txt"}
RESOLVE_URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{{path}}"

# Defaults to this project's existing Hindi translated-corpus byte count, so both native
# corpora end up byte-matched to each other and to the scale of the existing translated data.
DEFAULT_TARGET_BYTES = 1_381_881_024
OVERFETCH_BYTES = 2_000_000  # extra bytes requested so we can trim back to a clean line boundary


def fetch_range(path, n_bytes):
    url = RESOLVE_URL.format(path=path)
    headers = {"Range": f"bytes=0-{n_bytes - 1}"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.get(url, headers=headers, stream=True, timeout=120)
    resp.raise_for_status()
    chunks = []
    got = 0
    for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
        chunks.append(chunk)
        got += len(chunk)
        if got % (200 * 1024 * 1024) < len(chunk):
            print(f"    ...{got:,} bytes fetched", flush=True)
    return b"".join(chunks)


def collect_native_text(lang_code, path, target_bytes, out_path):
    print(f"[{lang_code}] fetching bytes 0-{target_bytes + OVERFETCH_BYTES:,} of {path}...")
    raw = fetch_range(path, target_bytes + OVERFETCH_BYTES)

    # Trim to target_bytes, then back up to the last full line so we don't keep a
    # truncated/partial line (or a byte range that split a multi-byte UTF-8 character).
    trimmed = raw[:target_bytes]
    last_newline = trimmed.rfind(b"\n")
    if last_newline != -1:
        trimmed = trimmed[:last_newline + 1]

    text = trimmed.decode("utf-8", errors="ignore")
    out_path.write_text(text, encoding="utf-8")

    n_bytes = len(text.encode("utf-8"))
    n_words = len(text.split())
    n_lines = text.count("\n")
    print(f"[{lang_code}] done: {n_bytes:,} bytes, {n_lines:,} lines, "
          f"{n_words:,} words -> {out_path}")
    return n_bytes, n_lines, n_words


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-bytes", type=int, default=DEFAULT_TARGET_BYTES,
                        help="Byte count each language is trimmed to (default: matches "
                             "this project's existing Hindi translated-corpus size)")
    parser.add_argument("--out-dir", default="native_indic_data")
    parser.add_argument("--langs", nargs="+", choices=["hi", "te"], default=["hi", "te"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    stats = {}
    for lang_code in args.langs:
        out_path = out_dir / f"native_{lang_code}.txt"
        stats[lang_code] = collect_native_text(
            lang_code, FILE_PATHS[lang_code], args.target_bytes, out_path)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for lang_code, (b, lines, words) in stats.items():
        print(f"  {lang_code}: {b:,} bytes | {lines:,} lines | {words:,} words | "
              f"{words / max(b, 1):.5f} words/byte (1:{b / max(words, 1):.2f} word:byte)")


if __name__ == "__main__":
    main()
