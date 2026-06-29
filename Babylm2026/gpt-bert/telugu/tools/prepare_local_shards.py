import argparse
import shutil
from pathlib import Path

import torch
from tokenizers import Tokenizer


def resolve_input_dir(dataset_name: str, data_root: Path) -> Path:
    base_dir = data_root / dataset_name
    train_dir = base_dir / "train"

    base_has_files = any(base_dir.glob("*.train*.txt"))
    train_has_files = any(train_dir.glob("*.train*.txt"))

    if base_has_files:
        return base_dir
    if train_has_files:
        return train_dir
    return base_dir


def read_all_tokens(input_dir: Path, tokenizer: Tokenizer) -> list[int]:
    all_ids: list[int] = []
    for path in sorted(input_dir.glob("*.train*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        ids = tokenizer.encode(text).ids
        all_ids.extend(ids)
        print(f"Encoded {path.name}: {len(ids):,} tokens")
    return all_ids


def write_shards(token_ids: list[int], output_dir: Path, shard_size_bytes: int) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not token_ids:
        raise ValueError(f"No token ids to write for {output_dir}")

    element_size = torch.tensor([0], dtype=torch.long).element_size()
    elems_per_shard = max(1, shard_size_bytes // element_size)
    shard_idx = 0

    for start in range(0, len(token_ids), elems_per_shard):
        end = min(start + elems_per_shard, len(token_ids))
        shard_tensor = torch.tensor(token_ids[start:end], dtype=torch.long)
        shard_path = output_dir / f"shard_{shard_idx:03d}.bin"
        torch.save(shard_tensor, shard_path)
        shard_idx += 1

    print(f"Wrote {shard_idx} shard(s) to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_root", type=Path, default=Path("data/raw"))
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output_base", type=Path, default=Path("data/processed"))
    parser.add_argument("--valid_fraction", type=float, default=0.0)
    parser.add_argument("--shard_size_bytes", type=int, default=100_000_000)
    args = parser.parse_args()

    input_dir = resolve_input_dir(args.dataset, args.data_root)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    all_ids = read_all_tokens(input_dir, tokenizer)

    train_dir = args.output_base / "train"
    valid_dir = args.output_base / "valid"

    if args.valid_fraction <= 0.0:
        if valid_dir.exists():
            shutil.rmtree(valid_dir)
        print(f"Total tokens: {len(all_ids):,}; train={len(all_ids):,}; valid=0")
        write_shards(all_ids, train_dir, args.shard_size_bytes)
        return

    cutoff = int(len(all_ids) * (1.0 - args.valid_fraction))
    train_ids = all_ids[:cutoff]
    valid_ids = all_ids[cutoff:]
    if not valid_ids:
        raise ValueError("Validation split is empty; dataset too small or valid_fraction too low")

    print(f"Total tokens: {len(all_ids):,}; train={len(train_ids):,}; valid={len(valid_ids):,}")
    write_shards(train_ids, train_dir, args.shard_size_bytes)
    write_shards(valid_ids, valid_dir, args.shard_size_bytes)


if __name__ == "__main__":
    main()
