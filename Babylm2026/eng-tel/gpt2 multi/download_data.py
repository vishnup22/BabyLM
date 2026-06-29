from huggingface_hub import snapshot_download
from pathlib import Path

ROOT = Path(__file__).parent / "data"

DATASETS = [
    ("BabyLM-community/BabyLM-2026-Strict",   ROOT / "BabyLM-2026-Strict"),
    ("pulipakav-1/translated-babylm-telugu",     ROOT / "translated-babylm-telugu"),
]

for repo_id, local_dir in DATASETS:
    print(f"\nDownloading {repo_id} -> {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(local_dir))
    print(f"Done: {local_dir}")
