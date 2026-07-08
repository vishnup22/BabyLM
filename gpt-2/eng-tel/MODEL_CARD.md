---
language:
- en
- te
license: mit
tags:
- gpt2
- causal-lm
- english
- telugu
- bilingual
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
- name: gpt2-tel-eng-babylm2026
  results:
  - task:
      type: text-generation
      name: Causal Language Modeling
    dataset:
      name: pulipakav-1/translated-babylm-english
      type: pulipakav-1/translated-babylm-english
    metrics:
    - type: perplexity
      value: 327.79
      name: English Test Perplexity
      verified: false
  - task:
      type: text-generation
      name: Causal Language Modeling
    dataset:
      name: pulipakav-1/translated-babylm-telugu
      type: pulipakav-1/translated-babylm-telugu
    metrics:
    - type: perplexity
      value: 258.52
      name: Telugu Test Perplexity
      verified: false
  - task:
      type: text-generation
      name: Causal Language Modeling
    dataset:
      name: BabyLM-community/BabyLM-BLIMP-Filtered
      type: BabyLM-community/BabyLM-BLIMP-Filtered
    metrics:
    - type: accuracy
      value: 0.7136
      name: English BLiMP Macro-Avg
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
      value: 0.2794
      name: SIB-200 Accuracy (eng_Latn)
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
      value: 0.2304
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
      value: 0.2803
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
      value: 0.2765
      name: MuBench Avg (12 tasks, Telugu)
      verified: false
---

# BabyLM English–Telugu Bilingual GPT-2

A bilingual GPT-2 language model trained from scratch on English and Telugu, as part of the [BabyLM Challenge 2026](https://babylm.github.io/). The model is trained under the strict (100M word) data budget using a naturalistic bilingual split that mirrors how English and Telugu co-occur in educational settings in India.

## Model Details

### Architecture

| Parameter | Value |
|---|---|
| Model type | GPT-2 (causal LM) |
| Context length | 1,024 tokens |
| Parameters | ~98M |

### Training Data

The bilingual training data combines English and Telugu using a naturalistic split:

- **Telugu side**: CHILDES and Project Gutenberg (translated) — representing home/early language exposure
- **English side**: all remaining sources (BNC Spoken, OpenSubtitles, Simple Wikipedia, Switchboard) — representing educational/formal English exposure

This split mirrors the documented pattern of English–Telugu bilingualism in India, where Indic languages are dominant at home and English is used in formal/educational settings.

| Language | Sources | Notes |
|---|---|---|
| Telugu | CHILDES + Gutenberg | Machine-translated from English via GPT |
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
| English | pulipakav-1/translated-babylm-english (test) | 327.79 |
| Telugu | pulipakav-1/translated-babylm-telugu (test) | 258.52 |

**Filtered — training-matched test sets** (only sources seen during bilingual training):

| Language | Dataset / Sources | Perplexity |
|---|---|---|
| English | [BabyLM-community/BabyLM-Test](https://huggingface.co/datasets/BabyLM-community/BabyLM-Test): [bnc_spoken.test](https://huggingface.co/datasets/BabyLM-community/BabyLM-Test/blob/main/bnc_spoken.test) + [open_subtitles.test](https://huggingface.co/datasets/BabyLM-community/BabyLM-Test/blob/main/open_subtitles.test) + [simple_wiki.test](https://huggingface.co/datasets/BabyLM-community/BabyLM-Test/blob/main/simple_wiki.test) + [switchboard.test](https://huggingface.co/datasets/BabyLM-community/BabyLM-Test/blob/main/switchboard.test) | 199.26 |
| Telugu | [pulipakav-1/translated-babylm-telugu](https://huggingface.co/datasets/pulipakav-1/translated-babylm-telugu): [childes.test.txt.train.te.txt](https://huggingface.co/datasets/pulipakav-1/translated-babylm-telugu/blob/main/test/childes.test.txt.train.te.txt) + [gutenberg.test.txt.train.te.txt](https://huggingface.co/datasets/pulipakav-1/translated-babylm-telugu/blob/main/test/gutenberg.test.txt.train.te.txt) | 270.56 |

### BLiMP (English)

Evaluated on [`BabyLM-community/BabyLM-BLIMP-Filtered`](https://huggingface.co/datasets/BabyLM-community/BabyLM-BLIMP-Filtered) — 67 English grammaticality tasks.

| Benchmark | Tasks | Macro-Avg Accuracy |
|---|---|---|
| BLiMP | 67 | **71.36%** |

### SIB-200

Zero-shot topic classification on [`Davlan/sib200`](https://huggingface.co/datasets/Davlan/sib200) (7 categories).

| Language | Config | Accuracy |
|---|---|---|
| English | eng_Latn | 27.94% |
| Telugu | tel_Telu | 23.04% |

### MuBench (zero-shot)

Zero-shot length-normalised log-likelihood on [`aialt/MuBench`](https://huggingface.co/datasets/aialt/MuBench).

**English:**

| ARC-C | ARC-E | BMLAMA | GPQA | HellaSwag | MMLU | MMLU-Pro | MNLI | SNLI | StoryCloze | TruthfulQA | WinoGrande | **Avg** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 24.48 | 24.88 | 10.32 | 25.45 | 24.37 | 24.43 | 10.81 | 32.46 | 31.45 | 53.33 | 24.49 | 49.88 | **28.03** |

**Telugu:**

| ARC-C | ARC-E | BMLAMA | GPQA | HellaSwag | MMLU | MMLU-Pro | MNLI | SNLI | StoryCloze | TruthfulQA | WinoGrande | **Avg** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 23.37 | 24.80 | 10.04 | 25.22 | 25.49 | 24.46 | 10.63 | 31.67 | 32.89 | 47.99 | 24.66 | 50.54 | **27.65** |

## Intended Use

- Studying computational bilingualism in South Asian language contexts
- Modeling English–Telugu code-switching and language co-existence
- Benchmarking low-resource bilingual LMs under BabyLM challenge conditions

## Limitations

- Telugu training data is machine-translated and may contain translation artifacts; Telugu is lower-resource than Hindi and translation quality may be lower
- Telugu training is restricted to Childes and Gutenberg; generalisation to other Telugu domains is limited
- The model does not represent naturally occurring Telugu text or native speaker intuitions
- Full-test perplexity includes Childes/Gutenberg sources not seen during bilingual training; use the filtered perplexity for a fair in-distribution comparison
- No M-BLiMP evaluation for Telugu (not available in `jumelet/multiblimp`)

## Citation

```bibtex
@inproceedings{babylm2026,
  title = {The BabyLM Challenge 2026},
  year  = {2026},
}
```

## Model Card Authors

pulipakav-1
