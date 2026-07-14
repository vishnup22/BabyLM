"""Download and byte-trim native (non-translated) Hindi and Telugu text from CC-100 for
training monolingual GPT-BERT models on naturally-written data, as a comparison point against
the project's GPT-5-mini translated-data models.

Originally targeted ai4bharat/IndicCorpV2 on Hugging Face, but that repo's large files were
persistently failing with 403/SignatureError from HF's Xet CDN bridge (a server-side issue --
confirmed independent of client config: tried datasets streaming, hf_xet removal, and a raw
HTTP Range request against the resolve/main/ URL, all redirected to the same broken endpoint).
CC-100 is hosted directly by the original team at data.statmt.org, entirely independent of
Hugging Face, and is a standard, widely-used Common-Crawl-derived monolingual corpus.

Streams and decompresses each language's .xz file on the fly, stopping the moment it has
collected TARGET_BYTES of decompressed text -- no need to download the full compressed files
(Hindi is 2.59GB compressed, Telugu 562MB compressed; decompressed output needed here is only
~1.3GB per language, so this only pulls a fraction of either file).

License note: CC-100 is distributed under the Common Crawl Foundation's Terms of Use (not as
permissive as IndicCorpV2's CC-0, but standard for this widely-used research corpus).

Usage:
    python prepare_native_indic_data.py
    python prepare_native_indic_data.py --target-bytes 1000000000   # 1GB each, for a quicker run
    python prepare_native_indic_data.py --out-dir native_indic_data
"""

import argparse
import lzma
from pathlib import Path

import requests

CC100_URLS = {
    "hi": "https://data.statmt.org/cc-100/hi.txt.xz",
    "te": "https://data.statmt.org/cc-100/te.txt.xz",
}

# Matches this project's existing Hindi translated-corpus byte count, so both native corpora
# end up byte-matched to each other and to the scale of the existing translated data.
DEFAULT_TARGET_BYTES = 1_381_881_024
CHUNK_SIZE = 4 * 1024 * 1024


def collect_native_text(lang_code, url, target_bytes, out_path):
    print(f"[{lang_code}] streaming + decompressing {url} until {target_bytes:,} bytes...")
    decompressor = lzma.LZMADecompressor()
    collected = bytearray()
    compressed_read = 0

    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    try:
        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
            compressed_read += len(chunk)
            try:
                out = decompressor.decompress(chunk)
            except lzma.LZMAError as e:
                print(f"  [{lang_code}] decompression stopped ({e}) -- using what we have")
                break
            collected.extend(out)
            if len(collected) % (200 * 1024 * 1024) < len(out):
                print(f"  [{lang_code}] {len(collected):,}/{target_bytes:,} decompressed bytes "
                      f"({compressed_read:,} compressed bytes read)...", flush=True)
            if len(collected) >= target_bytes:
                break
    finally:
        resp.close()

    # Trim to target_bytes, then back up to the last full line.
    trimmed = bytes(collected[:target_bytes])
    last_newline = trimmed.rfind(b"\n")
    if last_newline != -1:
        trimmed = trimmed[:last_newline + 1]

    text = trimmed.decode("utf-8", errors="ignore")
    out_path.write_text(text, encoding="utf-8")

    n_bytes = len(text.encode("utf-8"))
    n_words = len(text.split())
    n_lines = text.count("\n")
    print(f"[{lang_code}] done: {n_bytes:,} bytes, {n_lines:,} lines, "
          f"{n_words:,} words -> {out_path}  ({compressed_read:,} compressed bytes fetched)")
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
            lang_code, CC100_URLS[lang_code], args.target_bytes, out_path)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for lang_code, (b, lines, words) in stats.items():
        print(f"  {lang_code}: {b:,} bytes | {lines:,} lines | {words:,} words | "
              f"{words / max(b, 1):.5f} words/byte (1:{b / max(words, 1):.2f} word:byte)")


if __name__ == "__main__":
    main()
