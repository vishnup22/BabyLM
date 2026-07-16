#!/usr/bin/env bash
set -euo pipefail

DATASET_NAME="native-babylm-telugu"
OUT_DIR="data/$DATASET_NAME"
mkdir -p "$OUT_DIR"

echo "Downloading native (non-translated, CC-100-sourced) Telugu text from pulipakav-1/hi-te ..."
python - <<'PY'
import os
import urllib.request
from pathlib import Path

out_path = Path("data/native-babylm-telugu/native.train.te.txt")
url = "https://huggingface.co/datasets/pulipakav-1/hi-te/resolve/main/telugu.txt"
token = os.environ.get("HF_TOKEN", "")

req = urllib.request.Request(url)
if token:
    req.add_header("Authorization", f"Bearer {token}")

with urllib.request.urlopen(req, timeout=600) as resp, open(out_path, "wb") as f:
    while True:
        chunk = resp.read(8 * 1024 * 1024)
        if not chunk:
            break
        f.write(chunk)
print(f"Downloaded {out_path} ({out_path.stat().st_size:,} bytes)")
PY

echo "Done."
