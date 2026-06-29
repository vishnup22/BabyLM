English+Hindi Multilingual GPT-2
================================

This folder is a focused copy of the multilingual GPT-2 pipeline from
`strict-gpt2`, kept separate for the English+Hindi setup.

Included here are the files needed to:

- build the `en_hi_equal` bilingual dataset
- train the bilingual tokenizer
- train the multilingual GPT-2 model
- upload checkpoints if needed

Suggested workflow
------------------

```bash
bash experiment_scripts/build_all_multilingual.sh
bash experiment_scripts/train_all_tokenizers.sh
bash experiment_scripts/train_multilingual.sh
```

Notes
-----

- The bilingual dataset is `en_hi_equal`.
- The `en_hi_equal` tokenizer is trained with vocab size `32768`.
- The original `strict-gpt2` folder is left untouched as the broader source copy.
