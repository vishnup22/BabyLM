# Telugu BabyLM GPT-BERT Base (native, non-translated data)

This folder is a self-contained GPT-BERT training bundle based on the `babybabellm-gptbert` code and the `base` config.

Unlike `../telugu/` (which trains on IndicTrans2-translated BabyLM data), this bundle trains a monolingual GPT-BERT on **native Telugu text sourced from CC-100** -- deduplicated and byte-matched to the translated corpus's size so the two are comparable.

- Hugging Face dataset: `pulipakav-1/hi-te` (`telugu.txt`)
- Downloaded fresh by `scripts/download_dataset.sh` into `data/raw/native-babylm-telugu/native.train.te.txt`

This setup keeps the original GPT-BERT DDP trainer instead of rewriting it around Accelerate.

## Workflow

```bash
bash scripts/download_dataset.sh
bash scripts/clean_dataset.sh
bash scripts/train_tokenizer.sh
bash scripts/prepare_shards.sh
bash scripts/train_model.sh
```

Or all at once (used by `telugu_native.sh`'s SLURM job):

```bash
bash scripts/run_all.sh
```

`scripts/download_dataset.sh` and the final HF push both need `HF_TOKEN` set in the environment.

## Outputs

Running `scripts/run_all.sh` will create, and automatically push the final step:

- `data/raw/native-babylm-telugu/native.train.te.txt`
- `tokenizers/tokenizer_base_16384.json`
- `data/processed/train/`
- `model_checkpoints/`
- `pulipakav-1/telugu-native-gptbert-base-seed1` on Hugging Face (pushed automatically at the end of `run_all.sh` -- no manual step needed once the job finishes)

## Model

- config: `configs/base.json` (identical to `../telugu/` -- only the data source differs)
- tokenizer vocab: `16384` (trained fresh on the native corpus, not reused from `../telugu/`)
- architecture: GPT-BERT base

## Notes

- `prepare_shards.sh` is run with `--valid_fraction 0` (no validation split), matching `train_model.sh`'s `--no_validation`.
- `train_model.sh` uses `torchrun` with 4 GPUs by default.
- W&B is disabled by default in this bundle.
- `scripts/run_all.sh` pushes the trained EMA checkpoint to HF automatically as its last step (`../convert_gptbert_to_hf.py --lang telugu_native --seed 1 --repo pulipakav-1/telugu-native-gptbert-base-seed1`) -- no manual conversion step required.
- `NAME` in `train_model.sh` is deliberately set to `telugu_native-gptbert-base-seed $SEED` (with a space before the seed number) to exactly match `convert_gptbert_to_hf.py`'s expected checkpoint filename pattern (`{lang}-gptbert-base-seed {seed}_ema.bin`) -- don't "clean up" that space, it's load-bearing.
