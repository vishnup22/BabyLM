# English BabyLM Strict GPT-2

This folder is a self-contained copy of the `strict-gpt2` training path for the BabyLM 2026 English 100M Strict dataset.

It is set up to train a GPT-2 model from scratch on:

- Hugging Face dataset: `BabyLM-community/BabyLM-2026-Strict`
- local dataset folder: `data/BabyLM-2026-Strict`

## Included files

- `training.py`, `models.py`, `data_utils.py`, `utils.py`
- `train_tokenizer.py`
- `clean_dataset.py`
- `config.yaml`
- `requirements.txt`
- `scripts/` for English-only download, cleaning, tokenizer training, and training

## Expected generated directories

Running the scripts will create:

- `data/BabyLM-2026-Strict/`
- `tokenizers/BabyLM-2026-Strict/`
- `configs/BabyLM-2026-Strict/`
- `experiments/english-strict-100m/`

## Typical workflow

```bash
bash scripts/download_dataset.sh
bash scripts/clean_dataset.sh
bash scripts/train_tokenizer.sh
bash scripts/train_model.sh
```

Or run everything with:

```bash
bash scripts/run_all.sh
```

## Notes

- `train_tokenizer.py` writes both the tokenizer files and the matching GPT-2 config used by `training.py`.
- `training.py` expects the tokenizer under `./tokenizers/BabyLM-2026-Strict` and the model config under `./configs/BabyLM-2026-Strict`.
- Default training settings come from `config.yaml`, with the launch script overriding the experiment name and enabling Weights & Biases.
- The included launcher uses Hugging Face Accelerate for 4 A100 GPUs with bf16.
- The launcher sets `--batch_size 4` so the effective global batch stays at 16 across 4 GPUs, matching the original single-GPU setup more closely.
