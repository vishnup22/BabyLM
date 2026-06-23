# Telugu BabyLM GPT-BERT Base

This folder is a self-contained GPT-BERT training bundle based on the `babybabellm-gptbert` code and the `base` config.

It trains a monolingual GPT-BERT on:

- Hugging Face dataset: `pulipakav-1/translated-babylm-telugu`
- shared cleaned text from: `../../gpt-2/telugu/data/translated-babylm-telugu`

This setup keeps the original GPT-BERT DDP trainer instead of rewriting it around Accelerate.

## Workflow

```bash
bash scripts/train_tokenizer.sh
bash scripts/prepare_shards.sh
bash scripts/train_model.sh
```

## Outputs

Running the pipeline will create:

- `tokenizers/tokenizer_base_16384.json`
- `data/processed/train/`
- `data/processed/valid/`
- `model_checkpoints/`

## Notes

- This bundle reuses the cleaned text prepared by `gpt-2/telugu` to save disk space.
- Make sure `gpt-2/telugu` has already finished download + cleaning before running tokenizer/shard prep here.

## Model

- config: `configs/base.json`
- tokenizer vocab: `16384`
- architecture: GPT-BERT base

## Notes

- `prepare_shards.sh` creates a simple 95/5 token split into train/valid shards from the local text files.
- `train_model.sh` uses `torchrun` with 4 GPUs by default.
- W&B is disabled by default in this bundle.
