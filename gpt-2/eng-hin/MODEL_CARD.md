---
language:
- en
- hi
license: mit
tags:
- gpt2
- causal-lm
- english
- hindi
- bilingual
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
- name: gpt2-hin-eng-babylm2026
  results:
  - task:
      type: text-generation
      name: Causal Language Modeling
    dataset:
      name: pulipakav-1/translated-babylm-english
      type: pulipakav-1/translated-babylm-english
    metrics:
    - type: perplexity
      value: 381.98
      name: English Test Perplexity
      verified: false
  - task:
      type: text-generation
      name: Causal Language Modeling
    dataset:
      name: pulipakav-1/translated-babylm-hindi
      type: pulipakav-1/translated-babylm-hindi
    metrics:
    - type: perplexity
      value: 131.05
      name: Hindi Test Perplexity
      verified: false
  - task:
      type: text-generation
      name: Causal Language Modeling
    dataset:
      name: BabyLM-community/BabyLM-BLIMP-Filtered
      type: BabyLM-community/BabyLM-BLIMP-Filtered
    metrics:
    - type: accuracy
      value: 0.7070
      name: English BLiMP Macro-Avg
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
      value: 0.9150
      name: Hindi M-BLiMP Accuracy
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
      value: 0.2647
      name: SIB-200 Accuracy (eng_Latn)
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
      value: 0.1324
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
      value: 0.2827
      name: MuBench Avg (12 tasks, English)
      verified: false
  - task:
      type: text-generation
      name: Causal Language Modeling
    dataset:
      name: aialt/MuBench
      type: aialt/MuBench
    metrics:
    - type: accuracy
      value: 0.2729
      name: MuBench Avg (12 tasks, Hindi)
      verified: false
---

# BabyLM English–Hindi Bilingual GPT-2

A bilingual GPT-2 language model trained from scratch on English and Hindi, as part of the [BabyLM Challenge 2026](https://babylm.github.io/). The model is trained under the strict (100M word) data budget using a naturalistic bilingual split that mirrors how English and Hindi co-occur in educational settings in India.

## Model Details

### Architecture

| Parameter | Value |
|---|---|
| Model type | GPT-2 (causal LM) |
| Context length | 1,024 tokens |
| Parameters | ~98M |

### Training Data

The bilingual training data combines English and Hindi using a naturalistic split:

- **Hindi side**: CHILDES and Project Gutenberg (translated) — representing home/early language exposure
- **English side**: all remaining sources (BNC Spoken, OpenSubtitles, Simple Wikipedia, Switchboard) — representing educational/formal English exposure

This split mirrors the documented pattern of English–Hindi bilingualism in India, where Indic languages are dominant at home and English is used in formal/educational settings.

| Language | Sources | Notes |
|---|---|---|
| Hindi | CHILDES + Gutenberg | Machine-translated from English via GPT |
| English | BNC, OpenSubtitles, Wikipedia, Switchboard | Original English (no translation) |

### Training Procedure

| Field | Value |
|---|---|
| Framework | 🤗 Transformers |
| Optimizer | AdamW |
| Hardware | NVIDIA A100 |

## Evaluation

### Perplexity

**Full monolingual test sets** (includes sources outside the bilingual training distribution):

| Language | Dataset | Perplexity |
|---|---|---|
| English | pulipakav-1/translated-babylm-english (test) | 381.98 |
| Hindi | pulipakav-1/translated-babylm-hindi (test) | 131.05 |

**Filtered — training-matched test sets** (only sources seen during bilingual training):

| Language | Dataset / Sources | Perplexity |
|---|---|---|
| English | [BabyLM-community/BabyLM-Test](https://huggingface.co/datasets/BabyLM-community/BabyLM-Test): [bnc_spoken.test](https://huggingface.co/datasets/BabyLM-community/BabyLM-Test/blob/main/bnc_spoken.test) + [open_subtitles.test](https://huggingface.co/datasets/BabyLM-community/BabyLM-Test/blob/main/open_subtitles.test) + [simple_wiki.test](https://huggingface.co/datasets/BabyLM-community/BabyLM-Test/blob/main/simple_wiki.test) + [switchboard.test](https://huggingface.co/datasets/BabyLM-community/BabyLM-Test/blob/main/switchboard.test) | 233.69 |
| Hindi | [pulipakav-1/translated-babylm-hindi](https://huggingface.co/datasets/pulipakav-1/translated-babylm-hindi): [childes.test.txt.train.hi.txt](https://huggingface.co/datasets/pulipakav-1/translated-babylm-hindi/blob/main/test/childes.test.txt.train.hi.txt) + [gutenberg.test.txt.train.hi.txt](https://huggingface.co/datasets/pulipakav-1/translated-babylm-hindi/blob/main/test/gutenberg.test.txt.train.hi.txt) | 93.05 |

### BLiMP (English)

Evaluated on [`BabyLM-community/BabyLM-BLIMP-Filtered`](https://huggingface.co/datasets/BabyLM-community/BabyLM-BLIMP-Filtered) — 67 English grammaticality tasks.

| Benchmark | Tasks | Macro-Avg Accuracy |
|---|---|---|
| BLiMP | 67 | **70.70%** |

### M-BLiMP (Hindi)

Evaluated on [`jumelet/multiblimp`](https://huggingface.co/datasets/jumelet/multiblimp) config `hin`.

| Benchmark | Pairs | Accuracy |
|---|---|---|
| Hindi M-BLiMP | 1,447 | **91.50%** |

### SIB-200

Zero-shot topic classification on [`Davlan/sib200`](https://huggingface.co/datasets/Davlan/sib200) (7 categories).

| Language | Config | Accuracy |
|---|---|---|
| English | eng_Latn | 26.47% |
| Hindi | hin_Deva | 13.24% |

### MuBench (zero-shot)

Zero-shot length-normalised log-likelihood on [`aialt/MuBench`](https://huggingface.co/datasets/aialt/MuBench).

**English:**

| ARC-C | ARC-E | BMLAMA | GPQA | HellaSwag | MMLU | MMLU-Pro | MNLI | SNLI | StoryCloze | TruthfulQA | WinoGrande | **Avg** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 25.77 | 24.88 | 10.47 | 22.32 | 25.63 | 24.92 | 11.18 | 35.90 | 35.67 | 52.40 | 20.24 | 49.88 | **28.27** |

**Hindi:**

| ARC-C | ARC-E | BMLAMA | GPQA | HellaSwag | MMLU | MMLU-Pro | MNLI | SNLI | StoryCloze | TruthfulQA | WinoGrande | **Avg** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 24.31 | 23.57 | 10.64 | 24.55 | 24.90 | 26.24 | 11.08 | 31.67 | 32.89 | 46.28 | 21.26 | 50.12 | **27.29** |

## Intended Use

- Studying computational bilingualism in South Asian language contexts
- Modeling English–Hindi code-switching and language co-existence
- Benchmarking low-resource bilingual LMs under BabyLM challenge conditions

## Limitations

- Hindi training data is machine-translated and may contain translation artifacts
- Hindi training is restricted to Childes and Gutenberg; generalisation to other Hindi domains is limited
- The model does not represent naturally occurring Hindi text or native speaker intuitions
- Full-test perplexity includes Childes/Gutenberg sources not seen during bilingual training; use the filtered perplexity for a fair in-distribution comparison

## Citation

```bibtex
@inproceedings{babylm2026,
  title = {The BabyLM Challenge 2026},
  year  = {2026},
}
```

## Model Card Authors

pulipakav-1
