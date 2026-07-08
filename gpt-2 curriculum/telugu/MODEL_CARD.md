---
language:
- te
license: mit
tags:
- gpt2
- causal-lm
- telugu
- babylm
- babylm-2026
- indic
datasets:
- pulipakav-1/translated-babylm-telugu
metrics:
- perplexity
- accuracy
base_model: []
model-index:
- name: gpt-2-telugu2
  results:
  - task:
      type: text-generation
      name: Causal Language Modeling
    dataset:
      name: pulipakav-1/translated-babylm-telugu
      type: pulipakav-1/translated-babylm-telugu
    metrics:
    - type: perplexity
      value: 134.25
      name: Test Perplexity
      verified: false
  - task:
      type: text-generation
      name: Causal Language Modeling
    dataset:
      name: Davlan/sib200
      type: Davlan/sib200
      config: tel_Telu
    metrics:
    - type: accuracy
      value: 0.2108
      name: SIB-200 Accuracy (tel_Telu)
      verified: false
  - task:
      type: text-generation
      name: Causal Language Modeling
    dataset:
      name: aialt/MuBench
      type: aialt/MuBench
    metrics:
    - type: accuracy
      value: 0.2832
      name: MuBench Avg (12 tasks, Telugu)
      verified: false
---

# BabyLM Telugu GPT-2

A GPT-2 language model trained from scratch on Telugu, using a machine-translated version of the BabyLM 2026 Strict corpus. Trained as part of the [BabyLM Challenge 2026](https://babylm.github.io/) under the strict (100M word) data budget.

Note: No BLiMP-style benchmark exists for Telugu — Telugu is not available in `jumelet/multiblimp`.

## Model Details

### Architecture

| Parameter | Value |
|---|---|
| Model type | GPT-2 (causal LM) |
| Vocabulary size | 16,384 |
| Context length | 512 tokens |
| Hidden size | 768 |
| Layers | 12 |
| Attention heads | 12 |
| FFN size | 3,072 |
| Activation | GELU (new) |
| Dropout (resid / embd / attn) | 0.1 |
| Parameters | ~98M |

The tokenizer is a BPE tokenizer trained from scratch on the Telugu training split (vocab size 16,384).

### Training Data

Trained on [`pulipakav-1/translated-babylm-telugu`](https://huggingface.co/datasets/pulipakav-1/translated-babylm-telugu) — a Telugu translation of the BabyLM 2026 Strict corpus produced using [IndicTrans2](https://huggingface.co/ai4bharat/indictrans2-en-indic-1B) (AI4Bharat).

Source corpora (translated): BNC Spoken, CHILDES, Project Gutenberg, Open Subtitles, Simple Wikipedia, Switchboard.

### Training Procedure

| Field | Value |
|---|---|
| Framework | 🤗 Transformers |
| Optimizer | AdamW (β₁=0.9, β₂=0.999, ε=1e-8) |
| Weight decay | 0 (LayerNorm & bias params always exempt) |
| Learning rate | 5e-5 peak |
| LR schedule | Cosine decay with linear warmup (1% of total steps) |
| Gradient clipping | Norm 1.0 |
| Per-device batch size | 4 sequences |
| GPUs | 4 |
| Effective batch size | 16 sequences × 512 tokens = **8,192 tokens/step** |
| Epochs | 10 |
| Words per epoch | 100M |
| Total words seen | ~1B |
| Precision | bfloat16 mixed precision (Accelerate) |
| Checkpoint dtype | float32 |
| Hardware | 4× NVIDIA A100 (single node) |

Training used [Hugging Face Accelerate](https://huggingface.co/docs/accelerate) for multi-GPU distribution with bf16 mixed precision.

## Evaluation

### Perplexity

| Dataset | Perplexity |
|---|---|
| translated-babylm-telugu (test) | 134.25 |

### SIB-200

Zero-shot topic classification on [`Davlan/sib200`](https://huggingface.co/datasets/Davlan/sib200) (7 categories, length-normalised log-likelihood).

| Config | Accuracy |
|---|---|
| tel_Telu | 21.08% |

### MuBench (Telugu, zero-shot)

Zero-shot length-normalised log-likelihood on [`aialt/MuBench`](https://huggingface.co/datasets/aialt/MuBench).

| ARC-C | ARC-E | BMLAMA | GPQA | HellaSwag | MMLU | MMLU-Pro | MNLI | SNLI | StoryCloze | TruthfulQA | WinoGrande | **Avg** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 26.98 | 25.65 | 11.00 | 26.34 | 24.72 | 24.41 | 11.17 | 31.67 | 32.89 | 51.78 | 24.49 | 48.72 | **28.32** |

## Intended Use

- Studying language acquisition and low-resource learning under BabyLM challenge conditions
- Cross-lingual transfer and translation-based data augmentation research for low-resource Indic languages
- Benchmarking Telugu LMs under comparable training conditions to English and Hindi

## Limitations

- All training data is machine-translated from English and may contain translation artifacts, mistranslations, or unnatural phrasing
- Telugu is a morphologically rich, agglutinative language; translation quality from English may be lower than for Hindi
- No grammaticality benchmark (M-BLiMP) is available for Telugu
- The model does not represent naturally occurring Telugu text or native speaker intuitions
- Vocabulary and tokenization optimized for the translated corpus; may not generalise to formal Telugu domains

## Citation

If you use this model, please cite the BabyLM challenge and the IndicTrans2 translation system:

```bibtex
@inproceedings{babylm2026,
  title = {The BabyLM Challenge 2026},
  year  = {2026},
}

@article{gala2023indictrans2,
  title   = {IndicTrans2: Towards High-Quality and Accessible Machine Translation Models for all 22 Scheduled Indian Languages},
  author  = {Gala, Jay and others},
  journal = {arXiv preprint arXiv:2305.16307},
  year    = {2023}
}
```

## Model Card Authors

pulipakav-1
