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


class ShardWriter:
    """Writes token ids to disk in fixed-size shards, keeping only one shard's worth of
    tokens in memory at a time -- avoids ever holding the whole corpus's tokens in RAM,
    which can exceed available memory (and get OOM-killed) on large corpora."""

    def __init__(self, output_dir: Path, shard_size_bytes: int):
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        element_size = torch.tensor([0], dtype=torch.long).element_size()
        self.elems_per_shard = max(1, shard_size_bytes // element_size)
        self.buffer: list[int] = []
        self.shard_idx = 0
        self.total_written = 0

    def add(self, ids) -> None:
        self.buffer.extend(ids)
        while len(self.buffer) >= self.elems_per_shard:
            self._flush(self.elems_per_shard)

    def _flush(self, n: int) -> None:
        chunk, self.buffer = self.buffer[:n], self.buffer[n:]
        if not chunk:
            return
        shard_tensor = torch.tensor(chunk, dtype=torch.long)
        shard_path = self.output_dir / f"shard_{self.shard_idx:03d}.bin"
        torch.save(shard_tensor, shard_path)
        self.shard_idx += 1
        self.total_written += len(chunk)

    def close(self) -> int:
        if self.buffer:
            self._flush(len(self.buffer))
        print(f"Wrote {self.shard_idx} shard(s), {self.total_written:,} tokens, to {self.output_dir}")
        return self.total_written


def write_shards(token_ids: list[int], output_dir: Path, shard_size_bytes: int) -> None:
    """Legacy whole-list writer, kept for the valid_fraction > 0 path (not currently used
    by the training notebook, which passes valid_fraction=0)."""
    if not token_ids:
        raise ValueError(f"No token ids to write for {output_dir}")
    writer = ShardWriter(output_dir, shard_size_bytes)
    writer.add(token_ids)
    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_root", type=Path, default=Path("data/raw"))
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output_base", type=Path, default=Path("data/processed"))
    parser.add_argument("--valid_fraction", type=float, default=0.0)
    parser.add_argument("--shard_size_bytes", type=int, default=100_000_000)
    parser.add_argument("--read_chunk_bytes", type=int, default=20_000_000,
                        help="How many bytes of raw text to read and encode at a time "
                             "(streaming path, used when valid_fraction=0).")
    args = parser.parse_args()

    input_dir = resolve_input_dir(args.dataset, args.data_root)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    files = sorted(input_dir.glob("*.train*.txt"))
    if not files:
        raise FileNotFoundError(f"No .train*.txt files found in {input_dir}")

    train_dir = args.output_base / "train"
    valid_dir = args.output_base / "valid"

    if args.valid_fraction <= 0.0:
        # Streaming path: reads and encodes the corpus in chunks, flushing completed
        # shards to disk as it goes, so peak memory stays bounded by shard_size_bytes
        # regardless of total corpus size.
        if valid_dir.exists():
            shutil.rmtree(valid_dir)
        writer = ShardWriter(train_dir, args.shard_size_bytes)
        total_tokens = 0
        for path in files:
            print(f"Encoding {path.name} ({path.stat().st_size:,} bytes) in chunks...")
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                while True:
                    text = f.read(args.read_chunk_bytes)
                    if not text:
                        break
                    text += f.readline()  # avoid cutting a line in half at the chunk boundary
                    ids = tokenizer.encode(text).ids
                    writer.add(ids)
                    total_tokens += len(ids)
                    print(f"  ...{total_tokens:,} tokens so far", flush=True)
        train_total = writer.close()
        print(f"Total tokens: {total_tokens:,}; train={train_total:,}; valid=0")
        return

    # valid_fraction > 0: whole-corpus-in-memory path (not currently used by the training
    # notebook, kept for compatibility)
    all_ids: list[int] = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        ids = tokenizer.encode(text).ids
        all_ids.extend(ids)
        print(f"Encoded {path.name}: {len(ids):,} tokens")

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
