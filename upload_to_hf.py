"""
Upload a large folder to Hugging Face in batches.
Usage: python upload_to_hf.py --folder /path/to/folder --repo pulipakav-1/english-gpt2-seed1
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi

BATCH_SIZE = 5  # number of files per batch


def get_all_files(folder: Path):
    files = []
    for root, _, filenames in os.walk(folder):
        for fname in filenames:
            full_path = Path(root) / fname
            rel_path = full_path.relative_to(folder)
            files.append((full_path, str(rel_path)))
    return files


def upload_in_batches(folder: str, repo_id: str, repo_type: str = "model", batch_size: int = BATCH_SIZE):
    api = HfApi()
    folder = Path(folder)

    print(f"Creating repo '{repo_id}' (skipped if already exists)...")
    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)

    all_files = get_all_files(folder)
    total = len(all_files)
    print(f"Found {total} files to upload.")

    failed = []

    for i in range(0, total, batch_size):
        batch = all_files[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size
        print(f"\nBatch {batch_num}/{total_batches} — uploading {len(batch)} files...")

        for full_path, rel_path in batch:
            try:
                api.upload_file(
                    path_or_fileobj=str(full_path),
                    path_in_repo=rel_path,
                    repo_id=repo_id,
                    repo_type=repo_type,
                )
                print(f"  ✓ {rel_path}")
            except Exception as e:
                print(f"  ✗ {rel_path} — {e}")
                failed.append((full_path, rel_path))

    if failed:
        print(f"\n{len(failed)} files failed. Retrying...")
        for full_path, rel_path in failed:
            try:
                api.upload_file(
                    path_or_fileobj=str(full_path),
                    path_in_repo=rel_path,
                    repo_id=repo_id,
                    repo_type=repo_type,
                )
                print(f"  ✓ {rel_path} (retry success)")
            except Exception as e:
                print(f"  ✗ {rel_path} — gave up: {e}")
    else:
        print("\nAll files uploaded successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="Local path to the folder to upload")
    parser.add_argument("--repo", required=True, help="HF repo id, e.g. pulipakav-1/english-gpt2-seed1")
    parser.add_argument("--repo-type", default="model", help="model or dataset (default: model)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Files per batch (default: 20)")
    args = parser.parse_args()

    upload_in_batches(
        folder=args.folder,
        repo_id=args.repo,
        repo_type=args.repo_type,
        batch_size=args.batch_size,
    )
