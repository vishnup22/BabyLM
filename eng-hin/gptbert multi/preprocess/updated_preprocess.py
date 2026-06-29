import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

import pandas as pd
import torch
from tokenizers import Tokenizer
from tqdm import tqdm


def load_text_documents(dataset_dir: Path):
    documents = []

    for txt_file in sorted(dataset_dir.glob("*.train.txt")):
        text = txt_file.read_text(encoding="utf-8", errors="replace")
        documents.extend(line.strip() for line in text.splitlines() if line.strip())

    for parquet_file in sorted(dataset_dir.glob("*.train.parquet")):
        df = pd.read_parquet(parquet_file)
        if "text" not in df.columns:
            continue
        documents.extend(text.strip() for text in df["text"].astype(str) if text.strip())

    if not documents:
        raise FileNotFoundError(
            f"No training documents found in {dataset_dir} (.train.txt or .train.parquet)."
        )

    return documents


def split_documents(documents, dev_fraction=0.05):
    split_idx = max(1, int(len(documents) * (1.0 - dev_fraction)))
    train_docs = documents[:split_idx]
    valid_docs = documents[split_idx:]
    if not valid_docs:
        valid_docs = train_docs[-1:]
        train_docs = train_docs[:-1]
    return {"train": train_docs, "validation": valid_docs}


def encode_text_list(tokenizer_path, text_list):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    return [tokenizer.encode(text).ids for text in text_list]


def stream_encode_to_shards_mp(
    documents,
    tokenizer_path,
    output_dir,
    seq_length=128,
    shard_size_bytes=100_000_000,
    batch_size=1000,
    max_workers=None,
):
    os.makedirs(output_dir, exist_ok=True)
    buffer = []
    shard_count = 0
    element_size = torch.tensor([0], dtype=torch.long).element_size()
    num_elements_per_shard = max(seq_length, shard_size_bytes // element_size)

    max_workers = max_workers or cpu_count()
    print(f"Encoding {len(documents):,} docs with {max_workers} processes -> {output_dir}")

    batches = [documents[i : i + batch_size] for i in range(0, len(documents), batch_size)]
    total_batches = len(batches)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(encode_text_list, tokenizer_path, batch): idx
            for idx, batch in enumerate(batches)
        }

        with tqdm(total=total_batches, desc=f"Encoding {Path(output_dir).name}") as pbar:
            for future in as_completed(futures):
                enc_batch_ids = future.result()
                for ids in enc_batch_ids:
                    buffer.extend(ids)
                    while len(buffer) >= num_elements_per_shard:
                        shard_tensor = torch.tensor(
                            buffer[:num_elements_per_shard], dtype=torch.long
                        )
                        shard_file = os.path.join(output_dir, f"shard_{shard_count:03d}.bin")
                        torch.save(shard_tensor, shard_file)
                        shard_count += 1
                        buffer = buffer[num_elements_per_shard:]
                pbar.update(1)

    if buffer:
        shard_tensor = torch.tensor(buffer, dtype=torch.long)
        shard_file = os.path.join(output_dir, f"shard_{shard_count:03d}.bin")
        torch.save(shard_tensor, shard_file)
        print(f"Saved final shard {shard_count} with {shard_tensor.numel():,} tokens")

    print(f"Completed sharding for {output_dir}")


def main():
    repo_root = Path(__file__).resolve().parents[2]
    default_dataset_dir = repo_root / "gpt2 multi" / "data" / "en_hi_equal"
    default_base_dir = Path(__file__).resolve().parents[1] / "data" / "EN_HI_EQUAL"
    default_tokenizer = (
        Path(__file__).resolve().parents[1] / "tokenizers" / "tokenizer_en_hi_vs32768.json"
    )

    parser = argparse.ArgumentParser(
        description="Encode the local English+Hindi dataset into GPT-BERT shards."
    )
    parser.add_argument("--dataset_dir", type=Path, default=default_dataset_dir)
    parser.add_argument("--tokenizer", type=str, default=str(default_tokenizer))
    parser.add_argument("--seq_length", type=int, default=128)
    parser.add_argument("--shard_size_bytes", type=int, default=100_000_000)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--base_dir", type=Path, default=default_base_dir)
    parser.add_argument("--dev_fraction", type=float, default=0.05)
    args = parser.parse_args()

    documents = load_text_documents(args.dataset_dir)
    splits = split_documents(documents, dev_fraction=args.dev_fraction)

    train_dir = args.base_dir / "train"
    valid_dir = args.base_dir / "valid"

    stream_encode_to_shards_mp(
        splits["train"],
        args.tokenizer,
        str(train_dir),
        seq_length=args.seq_length,
        shard_size_bytes=args.shard_size_bytes,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
    )
    stream_encode_to_shards_mp(
        splits["validation"],
        args.tokenizer,
        str(valid_dir),
        seq_length=args.seq_length,
        shard_size_bytes=args.shard_size_bytes,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
