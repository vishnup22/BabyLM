# Hindi BabyLM Strict GPT-2

This folder is a self-contained copy of the `strict-gpt2` training path adapted for the translated Hindi BabyLM dataset.

It is set up to train a GPT-2 model from scratch on:

- Hugging Face dataset: `pulipakav-1/translated-babylm-hindi`
- local dataset folder: `data/translated-babylm-hindi`

## Included files

- `training.py`, `models.py`, `data_utils.py`, `utils.py`
- `train_tokenizer.py`
- `clean_dataset.py`
- `config.yaml`
- `requirements.txt`
- `scripts/` for English-only download, cleaning, tokenizer training, and training

## Expected generated directories

Running the scripts will create:

- `data/translated-babylm-hindi/`
- `tokenizers/translated-babylm-hindi/`
- `configs/translated-babylm-hindi/`
- `experiments/hindi-strict-100m/`

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
- `training.py` expects the tokenizer under `./tokenizers/translated-babylm-hindi` and the model config under `./configs/translated-babylm-hindi`.
- Default training settings come from `config.yaml`, with the launch script overriding the experiment name and enabling Weights & Biases.
- The included launcher uses Hugging Face Accelerate for 4 A100 GPUs with bf16.
- The launcher sets `--batch_size 4` so the effective global batch stays at 16 across 4 GPUs, matching the original single-GPU setup more closely.
