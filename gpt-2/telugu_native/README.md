# Telugu BabyLM Strict GPT-2 (native, non-translated data)

This folder is a self-contained copy of the `strict-gpt2` training path, recovered from git history (the original translated-data version was removed as "legacy") and adapted to train on **native Telugu text sourced from CC-100** instead -- deduplicated and byte-matched to the translated corpus's size so the two are comparable. Same training method as `../telugu/` (translated-data GPT-2), only the data source differs -- mirrors the same native/translated split already set up for GPT-BERT (`../../gpt-bert/telugu_native/`).

It is set up to train a GPT-2 model from scratch on:

- Hugging Face dataset: `pulipakav-1/hi-te` (`telugu.txt`)
- Downloaded fresh by `scripts/download_dataset.sh` into `data/native-babylm-telugu/native.train.te.txt`

## Included files

- `training.py`, `models.py`, `data_utils.py`, `utils.py`
- `train_tokenizer.py`
- `clean_dataset.py`
- `push_to_hf.py`
- `config.yaml`
- `requirements.txt`
- `scripts/` for download, cleaning, tokenizer training, and training

## Expected generated directories

Running `scripts/run_all.sh` will create, and automatically push the final step:

- `data/native-babylm-telugu/`
- `tokenizers/native-babylm-telugu/`
- `configs/native-babylm-telugu/`
- `experiments/telugu-native-strict-100m/`
- `pulipakav-1/telugu-native-gpt2-base` on Hugging Face (pushed automatically at the end of `run_all.sh` -- no manual step needed once the job finishes)

## Typical workflow

```bash
bash scripts/download_dataset.sh
bash scripts/clean_dataset.sh
bash scripts/train_tokenizer.sh
bash scripts/train_model.sh
python push_to_hf.py --dataset native-babylm-telugu --experiment telugu-native-strict-100m --repo pulipakav-1/telugu-native-gpt2-base
```

Or run everything with (used by `telugu_native.sh`'s SLURM job):

```bash
bash scripts/run_all.sh
```

`scripts/download_dataset.sh` and `push_to_hf.py` both need `HF_TOKEN` set in the environment.

## Notes

- `train_tokenizer.py` writes both the tokenizer files and the matching GPT-2 config used by `training.py`.
- `training.py` expects the tokenizer under `./tokenizers/native-babylm-telugu` and the model config under `./configs/native-babylm-telugu`.
- Default training settings come from `config.yaml`; the launch script overrides `--dataset`, `--words_per_epoch`, `--batch_size`, and `--experiment_name`.
- The included launcher uses Hugging Face Accelerate for 4 GPUs with bf16 (`accelerate_4xa100_bf16.yaml`, `num_processes: 4` -- matches this bundle's `--gres=gpu:4` request, no separate `N_GPUS` env var needed unlike the GPT-BERT bundles).
- The launcher sets `--batch_size 4` so the effective global batch stays at 16 across 4 GPUs, matching the original single-GPU setup more closely.
- Trained with a single seed (no `--seed` override) rather than the original telugu launcher's seed-1/seed-2 pattern -- kept consistent with the single-seed choice already made for the native GPT-BERT bundles, given the 5-day time budget.
- `push_to_hf.py` auto-detects the highest-numbered `epoch_*` checkpoint under `experiments/<experiment>/checkpoints/` -- no need to know the final epoch number in advance.
