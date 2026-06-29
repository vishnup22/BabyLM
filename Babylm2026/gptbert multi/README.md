English+Hindi Multilingual GPT-BERT
===================================

This folder is a focused GPT-BERT setup for the English+Hindi bilingual track.

It is wired to the dataset built by:

- `gpt2 multi/data/en_hi_equal`

Included here are the relevant files to:

- train a `32768`-vocab tokenizer on `en_hi_equal`
- preprocess that dataset into GPT-BERT shards
- launch multilingual GPT-BERT training

Suggested workflow
------------------

```bash
cd "gptbert multi"
python tokenizers/tokenizer.py
python preprocess/updated_preprocess.py
bash scripts/run_train_multigpu.sh
```

Notes
-----

- The default tokenizer output is `tokenizers/tokenizer_en_hi_vs32768.json`.
- The default shard output is `data/EN_HI_EQUAL/{train,valid}`.
- The launcher auto-builds the tokenizer and shards if they are missing.
