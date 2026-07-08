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
- name: gpt-2-telugu1
  results:
  - task:
      type: text-generation
      name: Causal Language Modeling
    dataset:
      name: pulipakav-1/translated-babylm-telugu
      type: pulipakav-1/translated-babylm-telugu
    metrics:
    - type: perplexity
      value: 136.98
      name: Test Perplexity (seed 1)
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
      value: 0.2255
      name: SIB-200 Accuracy (tel_Telu, seed 1)
      verified: false
  - task:
      type: text-generation
      name: Causal Language Modeling
    dataset:
      name: aialt/MuBench
      type: aialt/MuBench
    metrics:
    - type: accuracy
      value: 0.2747
      name: MuBench Avg (12 tasks, Telugu, seed 1)
      verified: false
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
      name: Test Perplexity (seed 2)
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
      name: SIB-200 Accuracy (tel_Telu, seed 2)
      verified: false
---

# BabyLM Telugu GPT-2

GPT-2 language models trained from scratch on Telugu, using a machine-translated version of the BabyLM 2026 Strict corpus. Two seeds are provided (`gpt-2-telugu1`, `gpt-2-telugu2`) to measure training variance. Trained as part of the [BabyLM Challenge 2026](https://babylm.github.io/) under the strict (100M word) data budget.

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

Two runs with different random seeds are provided to assess training stability.

Training used [Hugging Face Accelerate](https://huggingface.co/docs/accelerate) for multi-GPU distribution with bf16 mixed precision.

## Evaluation

### Perplexity

| Model | Dataset | Perplexity |
|---|---|---|
| gpt-2-telugu1 (seed 1) | translated-babylm-telugu (test) | 136.98 |
| gpt-2-telugu2 (seed 2) | translated-babylm-telugu (test) | 134.25 |

### SIB-200

Zero-shot topic classification on [`Davlan/sib200`](https://huggingface.co/datasets/Davlan/sib200) (7 categories, length-normalised log-likelihood).

| Model | Config | Accuracy |
|---|---|---|
| gpt-2-telugu1 (seed 1) | tel_Telu | 22.55% |
| gpt-2-telugu2 (seed 2) | tel_Telu | 21.08% |

### MuBench (Telugu, zero-shot)

Zero-shot length-normalised log-likelihood on [`aialt/MuBench`](https://huggingface.co/datasets/aialt/MuBench). Results for seed 1 (`gpt-2-telugu1`):

| ARC-C | ARC-E | BMLAMA | GPQA | HellaSwag | MMLU | MMLU-Pro | MNLI | SNLI | StoryCloze | TruthfulQA | WinoGrande | **Avg** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 25.69 | 24.04 | 10.67 | 23.88 | 25.19 | 26.12 | 11.25 | 31.67 | 32.89 | 46.59 | 21.43 | 50.21 | **27.47** |

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
