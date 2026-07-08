---
language:
- hi
license: mit
tags:
- gpt2
- causal-lm
- hindi
- babylm
- babylm-2026
- indic
datasets:
- pulipakav-1/translated-babylm-hindi
metrics:
- perplexity
- accuracy
base_model: []
model-index:
- name: gpt2-hindi-babylm2026
  results:
  - task:
      type: text-generation
      name: Causal Language Modeling
    dataset:
      name: pulipakav-1/translated-babylm-hindi
      type: pulipakav-1/translated-babylm-hindi
    metrics:
    - type: perplexity
      value: 85.41
      name: Test Perplexity
      verified: false
  - task:
      type: text-generation
      name: Causal Language Modeling
    dataset:
      name: jumelet/multiblimp
      type: jumelet/multiblimp
      config: hin
    metrics:
    - type: accuracy
      value: 0.9261
      name: Hindi M-BLiMP Accuracy
      verified: false
  - task:
      type: text-generation
      name: Causal Language Modeling
    dataset:
      name: Davlan/sib200
      type: Davlan/sib200
      config: hin_Deva
    metrics:
    - type: accuracy
      value: 0.2353
      name: SIB-200 Accuracy (hin_Deva)
      verified: false
  - task:
      type: text-generation
      name: Causal Language Modeling
    dataset:
      name: aialt/MuBench
      type: aialt/MuBench
    metrics:
    - type: accuracy
      value: 0.2750
      name: MuBench Avg (12 tasks, Hindi)
      verified: false
---

# BabyLM Hindi GPT-2

A GPT-2 language model trained from scratch on Hindi, using a machine-translated version of the BabyLM 2026 Strict corpus. Trained as part of the [BabyLM Challenge 2026](https://babylm.github.io/) under the strict (100M word) data budget.

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

The tokenizer is a BPE tokenizer trained from scratch on the Hindi training split (vocab size 16,384).

### Training Data

Trained on [`pulipakav-1/translated-babylm-hindi`](https://huggingface.co/datasets/pulipakav-1/translated-babylm-hindi) — a Hindi translation of the BabyLM 2026 Strict corpus produced using [IndicTrans2](https://huggingface.co/ai4bharat/indictrans2-en-indic-1B) (AI4Bharat).

| Split | Sentences | Words |
|---|---|---|
| Train | 11,579,880 | 118,309,059 |
| Val | 1,153,113 | 11,882,662 |
| Test | 1,097,453 | 11,106,875 |
| **Total** | **13,830,446** | **141,298,596** |

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
| translated-babylm-hindi (test) | 85.41 |

### M-BLiMP (Hindi)

Evaluated on [`jumelet/multiblimp`](https://huggingface.co/datasets/jumelet/multiblimp) config `hin` — a Hindi minimal-pair grammaticality benchmark covering syntactic and morphological phenomena.

| Benchmark | Pairs | Accuracy |
|---|---|---|
| Hindi M-BLiMP (`jumelet/multiblimp`, `hin`) | 1,447 | **92.61%** |

### SIB-200

Zero-shot topic classification on [`Davlan/sib200`](https://huggingface.co/datasets/Davlan/sib200) (7 categories, length-normalised log-likelihood).

| Config | Accuracy |
|---|---|
| hin_Deva | 23.53% |

### MuBench (Hindi, zero-shot)

Zero-shot length-normalised log-likelihood on [`aialt/MuBench`](https://huggingface.co/datasets/aialt/MuBench).

| ARC-C | ARC-E | BMLAMA | GPQA | HellaSwag | MMLU | MMLU-Pro | MNLI | SNLI | StoryCloze | TruthfulQA | WinoGrande | **Avg** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 26.55 | 26.32 | 10.89 | 24.33 | 25.93 | 25.32 | 10.68 | 32.44 | 31.45 | 47.06 | 18.54 | 50.45 | **27.50** |

## Intended Use

- Studying language acquisition and low-resource learning under BabyLM challenge conditions
- Cross-lingual transfer and translation-based data augmentation research
- Benchmarking Hindi LMs against English BabyLM baselines

## Limitations

- All training data is machine-translated from English and may contain translation artifacts, mistranslations, or unnatural phrasing
- Source corpora with informal/fragmented speech (CHILDES, Switchboard) may have lower translation fidelity
- The model does not represent naturally occurring Hindi text or native speaker intuitions
- Vocabulary and tokenization optimized for the translated corpus; may not generalise to formal Hindi domains

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
