# BabyLM 2026 — Evaluation Results

> **Status:** Partial — GPT-BERT, Sarvam-2B, Llama 3.2 1B, and multilingual models pending.

---

## Monolingual Models

### English

**Models evaluated on BabyLM-Test (perplexity), BabyLM-BLIMP-Filtered (BLiMP), Davlan/sib200 `eng_Latn` (SIB-200), aialt/MuBench `_en` (MuBench).**

#### Perplexity

| Model | Test |
|-------|------|
| GPT-2 (mono) | 140.50 |
| GPT-BERT (mono) | — |
| Sarvam-2B | — |
| Llama 3.2 1B | — |

#### BLiMP (macro avg, 67 tasks)

| Model | Score |
|-------|-------|
| GPT-2 (mono) | 0.7469 |
| GPT-BERT (mono) | — |
| Sarvam-2B | — |
| Llama 3.2 1B | — |

#### SIB-200 (`eng_Latn`)

| Model | Accuracy |
|-------|----------|
| GPT-2 (mono) | 0.3284 |
| GPT-BERT (mono) | — |
| Sarvam-2B | — |
| Llama 3.2 1B | — |

#### MuBench — English (zero-shot, length-normalised log-likelihood)

| Model | ARC-C | ARC-E | BMLAMA | GPQA | HellaSwag | MMLU | MMLU-Pro | MNLI | SNLI | StoryCloze | TruthfulQA | WinoGrande | Avg |
|-------|-------|-------|--------|------|-----------|------|----------|------|------|------------|------------|------------|-----|
| GPT-2 (mono) | 0.2311 | 0.2548 | 0.1037 | 0.2366 | 0.2508 | 0.2273 | 0.1113 | 0.3242 | 0.3145 | 0.5248 | 0.2330 | 0.5021 | 0.2762 |
| GPT-BERT (mono) | — | — | — | — | — | — | — | — | — | — | — | — | — |
| Sarvam-2B | — | — | — | — | — | — | — | — | — | — | — | — | — |
| Llama 3.2 1B | — | — | — | — | — | — | — | — | — | — | — | — | — |

---

### Hindi

**Models evaluated on pulipakav-1/translated-babylm-hindi (perplexity), jumelet/multiblimp `hin` (M-BLiMP), Davlan/sib200 `hin_Deva` (SIB-200), aialt/MuBench `_hi` (MuBench).**

#### Perplexity

| Model | Test |
|-------|------|
| GPT-2 (mono) | 85.41 |
| GPT-BERT (mono) | — |
| Sarvam-2B | — |
| Llama 3.2 1B | — |

#### M-BLiMP (`hin`, 1,447 pairs)

| Model | Accuracy |
|-------|----------|
| GPT-2 (mono) | 0.9261 |
| GPT-BERT (mono) | — |
| Sarvam-2B | — |
| Llama 3.2 1B | — |

#### SIB-200 (`hin_Deva`)

| Model | Accuracy |
|-------|----------|
| GPT-2 (mono) | 0.2353 |
| GPT-BERT (mono) | — |
| Sarvam-2B | — |
| Llama 3.2 1B | — |

#### MuBench — Hindi (zero-shot, length-normalised log-likelihood)

| Model | ARC-C | ARC-E | BMLAMA | GPQA | HellaSwag | MMLU | MMLU-Pro | MNLI | SNLI | StoryCloze | TruthfulQA | WinoGrande | Avg |
|-------|-------|-------|--------|------|-----------|------|----------|------|------|------------|------------|------------|-----|
| GPT-2 (mono) | 0.2655 | 0.2632 | 0.1089 | 0.2433 | 0.2593 | 0.2532 | 0.1068 | 0.3244 | 0.3145 | 0.4706 | 0.1854 | 0.5045 | 0.2750 |
| GPT-BERT (mono) | — | — | — | — | — | — | — | — | — | — | — | — | — |
| Sarvam-2B | — | — | — | — | — | — | — | — | — | — | — | — | — |
| Llama 3.2 1B | — | — | — | — | — | — | — | — | — | — | — | — | — |

---

### Telugu

**Models evaluated on pulipakav-1/translated-babylm-telugu (perplexity), Davlan/sib200 `tel_Telu` (SIB-200), aialt/MuBench `_te` (MuBench). No M-BLiMP — Telugu not available in jumelet/multiblimp.**

#### Perplexity

| Model | Test |
|-------|------|
| GPT-2 (mono) | 134.25 |
| GPT-BERT (mono) | — |
| Sarvam-2B | — |
| Llama 3.2 1B | — |

#### SIB-200 (`tel_Telu`)

| Model | Accuracy |
|-------|----------|
| GPT-2 (mono) | 0.2108 |
| GPT-BERT (mono) | — |
| Sarvam-2B | — |
| Llama 3.2 1B | — |

#### MuBench — Telugu (zero-shot, length-normalised log-likelihood)

| Model | ARC-C | ARC-E | BMLAMA | GPQA | HellaSwag | MMLU | MMLU-Pro | MNLI | SNLI | StoryCloze | TruthfulQA | WinoGrande | Avg |
|-------|-------|-------|--------|------|-----------|------|----------|------|------|------------|------------|------------|-----|
| GPT-2 (mono) | 0.2698 | 0.2565 | 0.1100 | 0.2634 | 0.2472 | 0.2441 | 0.1117 | 0.3167 | 0.3289 | 0.5178 | 0.2449 | 0.4872 | 0.2832 |
| GPT-BERT (mono) | — | — | — | — | — | — | — | — | — | — | — | — | — |
| Sarvam-2B | — | — | — | — | — | — | — | — | — | — | — | — | — |
| Llama 3.2 1B | — | — | — | — | — | — | — | — | — | — | — | — | — |

---

## Multilingual (Bilingual) Models

### English evals (all bilingual models)

#### Perplexity

| Model | Test |
|-------|------|
| GPT-2 (eng-hin) | — |
| GPT-2 (eng-tel) | — |

#### BLiMP (macro avg, 67 tasks)

| Model | Score |
|-------|-------|
| GPT-2 (eng-hin) | — |
| GPT-2 (eng-tel) | — |

#### SIB-200 (`eng_Latn`)

| Model | Accuracy |
|-------|----------|
| GPT-2 (eng-hin) | — |
| GPT-2 (eng-tel) | — |

#### MuBench — English

| Model | Avg |
|-------|-----|
| GPT-2 (eng-hin) | — |
| GPT-2 (eng-tel) | — |

---

### Hindi evals (eng-hin models only)

#### Perplexity

| Model | Test |
|-------|------|
| GPT-2 (eng-hin) | — |

#### M-BLiMP (`hin`)

| Model | Accuracy |
|-------|----------|
| GPT-2 (eng-hin) | — |

#### SIB-200 (`hin_Deva`)

| Model | Accuracy |
|-------|----------|
| GPT-2 (eng-hin) | — |

#### MuBench — Hindi

| Model | Avg |
|-------|-----|
| GPT-2 (eng-hin) | — |

---

### Telugu evals (eng-tel models only)

#### Perplexity

| Model | Test |
|-------|------|
| GPT-2 (eng-tel) | — |

#### SIB-200 (`tel_Telu`)

| Model | Accuracy |
|-------|----------|
| GPT-2 (eng-tel) | — |

#### MuBench — Telugu

| Model | Avg |
|-------|-----|
| GPT-2 (eng-tel) | — |
