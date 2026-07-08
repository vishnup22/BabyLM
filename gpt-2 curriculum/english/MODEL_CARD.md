---
language:
- en
license: mit
tags:
- gpt2
- causal-lm
- english
- babylm
- babylm-2026
datasets:
- BabyLM-community/BabyLM-2026-Strict
metrics:
- perplexity
- accuracy
base_model: []
model-index:
- name: gpt2-english-babylm2026
  results:
  - task:
      type: text-generation
      name: Causal Language Modeling
    dataset:
      name: BabyLM-community/BabyLM-Test
      type: BabyLM-community/BabyLM-Test
    metrics:
    - type: perplexity
      value: 140.50
      name: Test Perplexity
      verified: false
  - task:
      type: text-generation
      name: Causal Language Modeling
    dataset:
      name: BabyLM-community/BabyLM-BLIMP-Filtered
      type: BabyLM-community/BabyLM-BLIMP-Filtered
    metrics:
    - type: accuracy
      value: 0.7469
      name: BLiMP Macro Avg (67 tasks)
      verified: false
  - task:
      type: text-generation
      name: Causal Language Modeling
    dataset:
      name: Davlan/sib200
      type: Davlan/sib200
      config: eng_Latn
    metrics:
    - type: accuracy
      value: 0.3284
      name: SIB-200 Accuracy (eng_Latn)
      verified: false
  - task:
      type: text-generation
      name: Causal Language Modeling
    dataset:
      name: aialt/MuBench
      type: aialt/MuBench
    metrics:
    - type: accuracy
      value: 0.2762
      name: MuBench Avg (12 tasks, English)
      verified: false
---

# BabyLM English GPT-2

A GPT-2 language model trained from scratch on English, using the BabyLM 2026 Strict corpus. Trained as part of the [BabyLM Challenge 2026](https://babylm.github.io/) under the strict (100M word) data budget.

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

The tokenizer is a BPE tokenizer trained from scratch on the English training split (vocab size 16,384).

### Training Data

Trained on [`BabyLM-community/BabyLM-2026-Strict`](https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict) — the strict-track English corpus for BabyLM 2026 (~100M words).

Source corpora: BNC Spoken, CHILDES, Project Gutenberg, Open Subtitles, Simple Wikipedia, Switchboard.

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

### BLiMP (67 tasks)

Evaluated on [`BabyLM-community/BabyLM-BLIMP-Filtered`](https://huggingface.co/datasets/BabyLM-community/BabyLM-BLIMP-Filtered) — English minimal-pair grammaticality benchmark.

| Benchmark | Macro Avg |
|---|---|
| BLiMP (67 tasks) | **74.69%** |

### Perplexity

| Dataset | Perplexity |
|---|---|
| BabyLM-Test | 140.50 |

### SIB-200

Zero-shot topic classification on [`Davlan/sib200`](https://huggingface.co/datasets/Davlan/sib200) (7 categories, length-normalised log-likelihood).

| Config | Accuracy |
|---|---|
| eng_Latn | 32.84% |

### MuBench (English, zero-shot)

Zero-shot length-normalised log-likelihood on [`aialt/MuBench`](https://huggingface.co/datasets/aialt/MuBench).

| ARC-C | ARC-E | BMLAMA | GPQA | HellaSwag | MMLU | MMLU-Pro | MNLI | SNLI | StoryCloze | TruthfulQA | WinoGrande | **Avg** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 23.11 | 25.48 | 10.37 | 23.66 | 25.08 | 22.73 | 11.13 | 32.42 | 31.45 | 52.48 | 23.30 | 50.21 | **27.62** |

## Intended Use

- Studying language acquisition and low-resource learning under BabyLM challenge conditions
- Benchmarking small English causal LMs trained from scratch
- Probing syntactic and linguistic knowledge in models trained on child-directed and naturalistic language

## Limitations

- Trained on a limited 100M word budget — performance is well below large-scale language models
- Context window of 512 tokens limits handling of long documents
- Tokenizer trained on BabyLM data only; may handle out-of-domain vocabulary poorly

## Citation

If you use this model, please cite the BabyLM challenge:

```bibtex
@inproceedings{babylm2026,
  title = {The BabyLM Challenge 2026},
  year  = {2026},
}
```

## Model Card Authors

pulipakav-1
