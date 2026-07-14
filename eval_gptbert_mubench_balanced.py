"""MuBench-only eval with even GPU load balancing: unlike eval_gptbert_all.py's eval_mubench
(which assigns whole tasks to each GPU -- 3 tasks/GPU with 4 GPUs -- so one GPU can get stuck
alone on a heavy task like GPQA/MMLU-Pro while the other 3 sit idle), this splits each task's
ITEMS across all 4 GPUs. All GPUs work on the same task together and move to the next task in
lockstep, so total time approaches (total work) / num_gpus instead of being bottlenecked by
whichever GPU drew the worst combination of whole tasks.

Reuses model loading / scoring / dataset-loading from eval_gptbert_all.py unchanged -- only the
sharding granularity for MuBench is different here.

Usage (same CLI shape as eval_gptbert_all.py, but mubench-only):
    accelerate launch --num_processes 4 eval_gptbert_mubench_balanced.py --models en_hi_seed1 --langs hi
    python eval_gptbert_mubench_balanced.py --models mono_te   # single-GPU, no accelerate needed
"""

import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm

from eval_gptbert_all import (
    DEVICE, IS_MAIN, MODEL_REGISTRY, MUBENCH_DATASET_ID, MUBENCH_LANG_SUFFIX,
    MUBENCH_TASKS_BASE, RANK, SYNC_DIR, WORLD_SIZE, load_dataset_retry,
    load_model_and_tokenizer, score_gptbert_pll, write_csv, MAX_SEQ_LEN,
)


def shard_items(items):
    """Split a list of *items within one task* evenly across ranks (item-level, not task-level)."""
    items = list(items)
    return items[RANK::WORLD_SIZE]


def gather_sum_pair(a, b, tag):
    """File-based sum of two numbers across all ranks -- same rationale as
    eval_gptbert_all.gather_merge_dicts: avoids the NCCL/CUDA collective hang this cluster's old
    kernel is prone to. Since every rank now finishes its even item-split of the *same* task at
    roughly the same time, this sync should resolve in seconds, not hours."""
    if WORLD_SIZE == 1:
        return a, b
    SYNC_DIR.mkdir(parents=True, exist_ok=True)
    (SYNC_DIR / f"{tag}_rank{RANK}.json").write_text(json.dumps([a, b]))
    expected = [SYNC_DIR / f"{tag}_rank{r}.json" for r in range(WORLD_SIZE)]
    waited = 0
    while not all(p.exists() for p in expected):
        time.sleep(5)
        waited += 5
        if IS_MAIN and waited % 300 == 0:
            missing = [p.name for p in expected if not p.exists()]
            print(f"  [sync] waiting on {missing} ({waited}s elapsed)", flush=True)
    time.sleep(2)  # guard against a reader racing a writer still flushing its file
    total_a, total_b = 0, 0
    for p in expected:
        va, vb = json.loads(p.read_text())
        total_a += va
        total_b += vb
    return total_a, total_b


def eval_mubench_balanced(model, tokenizer, cls_id, mask_id, lang, model_key):
    suffix = MUBENCH_LANG_SUFFIX[lang]
    results = {}
    for task_base in tqdm(MUBENCH_TASKS_BASE, desc="  MuBench", disable=not IS_MAIN):
        task_config = task_base + suffix
        task_key    = task_base.replace("Dataset_local_template", "")
        try:
            ds = load_dataset_retry(MUBENCH_DATASET_ID, task_config, split="test")
        except Exception as e:
            if IS_MAIN:
                tqdm.write(f"    SKIP {task_config}: {e}")
            continue

        my_rows = shard_items(ds)
        correct = 0
        for row in my_rows:
            prompt, choices, label = row["prompt"], row["choices"], row["label"]
            scores = []
            for choice in choices:
                ids = tokenizer.encode(prompt + choice, add_special_tokens=False)[:MAX_SEQ_LEN]
                scores.append(score_gptbert_pll(model, ids, cls_id, mask_id) / max(len(ids), 1))
            if scores.index(max(scores)) == label:
                correct += 1

        total_correct, total_n = gather_sum_pair(
            correct, len(my_rows), tag=f"{model_key}_{lang}_mubench_{task_key}")
        if total_n > 0:
            results[task_key] = round(total_correct / total_n, 4)

    if results:
        results["avg"] = round(sum(results.values()) / len(results), 4)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=list(MODEL_REGISTRY), default=list(MODEL_REGISTRY))
    parser.add_argument("--langs", nargs="+", choices=["en", "hi", "te"], default=None,
                        help="Restrict to these languages only (default: each model's full lang list)")
    parser.add_argument("--output", default="results_gptbert_mubench_balanced.json")
    args = parser.parse_args()

    out_path = Path(args.output)
    all_results = json.loads(out_path.read_text()) if out_path.exists() and IS_MAIN else {}

    for model_key in args.models:
        spec = MODEL_REGISTRY[model_key]
        if IS_MAIN:
            print(f"\n{'='*70}\nModel: {model_key}  ({spec})  [{WORLD_SIZE} GPU(s)]\n{'='*70}")

        model, tokenizer, cls_id, mask_id, pad_id = load_model_and_tokenizer(spec)
        all_results.setdefault(model_key, {})

        langs = [l for l in (args.langs or spec["langs"]) if l in spec["langs"]]
        for lang in langs:
            if IS_MAIN:
                print(f"\n  -- {lang.upper()} --")
            all_results[model_key].setdefault(lang, {})
            all_results[model_key][lang]["mubench"] = eval_mubench_balanced(
                model, tokenizer, cls_id, mask_id, lang, model_key)
            if IS_MAIN:
                print(f"    mubench avg: {all_results[model_key][lang]['mubench'].get('avg')}")
                out_path.write_text(json.dumps(all_results, indent=2))

        import torch
        del model
        torch.cuda.empty_cache()

    if IS_MAIN:
        write_csv(all_results, out_path.with_suffix(".csv"))
        print(f"\nDone. Results in {out_path}")


if __name__ == "__main__":
    main()
