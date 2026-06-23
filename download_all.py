from huggingface_hub import snapshot_download
from pathlib import Path

DATASETS = [
    ("BabyLM-community/BabyLM-2026-Strict",      "gpt-2/english/data/BabyLM-2026-Strict"),
    ("pulipakav-1/translated-babylm-hindi",        "gpt-2/hindi/data/translated-babylm-hindi"),
    ("pulipakav-1/translated-babylm-telugu",       "gpt-2/telugu/data/translated-babylm-telugu"),
]

for repo_id, local_dir in DATASETS:
    print(f"\nDownloading {repo_id} -> {local_dir}")
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=local_dir)
    print(f"Done: {local_dir}")
