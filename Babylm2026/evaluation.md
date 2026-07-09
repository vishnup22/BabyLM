# BabyLM 2026 — Evaluation Results

> **Status:** Partial — GPT-BERT, Sarvam-2B, and Llama 3.2 1B pending. Bilingual GPT-2 (eng-hin, eng-tel) complete.

---

## Monolingual Models

### English

**Models evaluated on BabyLM-Test (perplexity), BabyLM-BLIMP-Filtered (BLiMP), Davlan/sib200 `eng_Latn` (SIB-200), aialt/MuBench `_en` (MuBench).**

#### Perplexity

| Model | Test | OS-data |
|-------|------|---------|
| GPT-2 (mono) | 140.50 | 306.06 |
| GPT-BERT (mono) | 4.8953 | — |
| Sarvam-2B | 186.58 | 89.50 |
| Llama 3.2 1B | — | 52.18 |

> OS-data: perplexity on the English side of `Helsinki-NLP/opus-100` (`en-hi` config, full split, streamed).

#### BLiMP (macro avg, 67 tasks)

| Model | Score |
|-------|-------|
| GPT-2 (mono) | 0.7469 |
| GPT-BERT (mono) | 0.7978 |
| Sarvam-2B | 0.8006 |
| Llama 3.2 1B | — |

#### SIB-200 (`eng_Latn`)

| Model | Accuracy |
|-------|----------|
| GPT-2 (mono) | 0.3284 |
| GPT-BERT (mono) | 0.2451 |
| Sarvam-2B | 0.4706 |
| Llama 3.2 1B | — |

#### MuBench — English (zero-shot, length-normalised log-likelihood)

| Model | ARC-C | ARC-E | BMLAMA | GPQA | HellaSwag | MMLU | MMLU-Pro | MNLI | SNLI | StoryCloze | TruthfulQA | WinoGrande | Avg |
|-------|-------|-------|--------|------|-----------|------|----------|------|------|------------|------------|------------|-----|
| GPT-2 (mono) | 0.2311 | 0.2548 | 0.1037 | 0.2366 | 0.2508 | 0.2273 | 0.1113 | 0.3242 | 0.3145 | 0.5248 | 0.2330 | 0.5021 | 0.2762 |
| GPT-BERT (mono) | — | — | — | — | — | — | — | — | — | — | — | — | — |
| Sarvam-2B | 0.2242 | 0.2726 | 0.1125 | 0.2299 | 0.2517 | 0.2288 | 0.1149 | 0.3590 | 0.3567 | 0.5433 | 0.2024 | 0.5029 | 0.2833 |
| Llama 3.2 1B | — | — | — | — | — | — | — | — | — | — | — | — | — |

---

### Hindi

**Models evaluated on pulipakav-1/translated-babylm-hindi (perplexity), jumelet/multiblimp `hin` (M-BLiMP), Davlan/sib200 `hin_Deva` (SIB-200), aialt/MuBench `_hi` (MuBench).**

#### Perplexity

| Model | Test | OS-data |
|-------|------|---------|
| GPT-2 (mono) | 85.41 | 707.13 |
| GPT-BERT (mono) | 1.3261 | — |
| Sarvam-2B | 210.51 | 229.16 |
| Llama 3.2 1B | — | 21.07 |

> OS-data: perplexity on the Hindi side of `Helsinki-NLP/opus-100` (`en-hi` config, full split, streamed).

#### M-BLiMP (`hin`, 1,447 pairs)

| Model | Accuracy |
|-------|----------|
| GPT-2 (mono) | 0.9261 |
| GPT-BERT (mono) | 0.9399 |
| Sarvam-2B | 0.9772 |
| Llama 3.2 1B | — |

#### SIB-200 (`hin_Deva`)

| Model | Accuracy |
|-------|----------|
| GPT-2 (mono) | 0.2353 |
| GPT-BERT (mono) | 0.1961 |
| Sarvam-2B | 0.4510 |
| Llama 3.2 1B | — |

#### MuBench — Hindi (zero-shot, length-normalised log-likelihood)

| Model | ARC-C | ARC-E | BMLAMA | GPQA | HellaSwag | MMLU | MMLU-Pro | MNLI | SNLI | StoryCloze | TruthfulQA | WinoGrande | Avg |
|-------|-------|-------|--------|------|-----------|------|----------|------|------|------------|------------|------------|-----|
| GPT-2 (mono) | 0.2655 | 0.2632 | 0.1089 | 0.2433 | 0.2593 | 0.2532 | 0.1068 | 0.3244 | 0.3145 | 0.4706 | 0.1854 | 0.5045 | 0.2750 |
| GPT-BERT (mono) | — | — | — | — | — | — | — | — | — | — | — | — | — |
| Sarvam-2B | 0.2569 | 0.2467 | 0.1114 | 0.2567 | 0.2521 | 0.2377 | 0.1169 | 0.3525 | 0.3290 | 0.5364 | 0.2194 | 0.5087 | 0.2854 |
| Llama 3.2 1B | — | — | — | — | — | — | — | — | — | — | — | — | — |

---

### Telugu

**Models evaluated on pulipakav-1/translated-babylm-telugu (perplexity), Davlan/sib200 `tel_Telu` (SIB-200), aialt/MuBench `_te` (MuBench). No M-BLiMP — Telugu not available in jumelet/multiblimp.**

#### Perplexity

| Model | Test | OS-data |
|-------|------|---------|
| GPT-2 (mono) | 134.25 | 1343.92 |
| GPT-BERT (mono) | — | — |
| Sarvam-2B | 205.81 | 2350.40 |
| Llama 3.2 1B | — | 4.41 |

> OS-data: perplexity on the Telugu side of `Helsinki-NLP/opus-100` (`en-te` config, full split, streamed).

#### SIB-200 (`tel_Telu`)

| Model | Accuracy |
|-------|----------|
| GPT-2 (mono) | 0.2108 |
| GPT-BERT (mono) | — |
| Sarvam-2B | 0.3824 |
| Llama 3.2 1B | — |

#### MuBench — Telugu (zero-shot, length-normalised log-likelihood)

| Model | ARC-C | ARC-E | BMLAMA | GPQA | HellaSwag | MMLU | MMLU-Pro | MNLI | SNLI | StoryCloze | TruthfulQA | WinoGrande | Avg |
|-------|-------|-------|--------|------|-----------|------|----------|------|------|------------|------------|------------|-----|
| GPT-2 (mono) | 0.2698 | 0.2565 | 0.1100 | 0.2634 | 0.2472 | 0.2441 | 0.1117 | 0.3167 | 0.3289 | 0.5178 | 0.2449 | 0.4872 | 0.2832 |
| GPT-BERT (mono) | — | — | — | — | — | — | — | — | — | — | — | — | — |
| Sarvam-2B | 0.2268 | 0.2573 | 0.1074 | 0.2277 | 0.2472 | 0.2356 | 0.1179 | 0.3167 | 0.3289 | 0.5333 | 0.2245 | 0.4979 | 0.2768 |
| Llama 3.2 1B | — | — | — | — | — | — | — | — | — | — | — | — | — |



## Multilingual (Bilingual) Models

### English evals (all bilingual models)

#### Perplexity

| Model | Full test | Filtered (training sources only) | OS-data |
|-------|-----------|----------------------------------|---------|
| GPT-2 (eng-hin) | 381.98 | 233.69 | 459.80 |
| GPT-2 (eng-tel) | 327.79 | 199.26 | 377.26 |

> Full test: `pulipakav-1/translated-babylm-english` (all sources).
> Filtered: [`BabyLM-community/BabyLM-Test`](https://huggingface.co/datasets/BabyLM-community/BabyLM-Test) — `bnc_spoken.test`, `open_subtitles.test`, `simple_wiki.test`, `switchboard.test` (matches bilingual training distribution).
> OS-data: perplexity on the English side of `Helsinki-NLP/opus-100` (full split, streamed) — `en-hi` config for eng-hin, `en-te` config for eng-tel.

#### BLiMP (macro avg, 67 tasks)

| Model | Score |
|-------|-------|
| GPT-2 (eng-hin) | 0.7070 |
| GPT-2 (eng-tel) | 0.7136 |
| GPT-BERT (eng-hin) | 0.7868 |

#### SIB-200 (`eng_Latn`)

| Model | Accuracy |
|-------|----------|
| GPT-2 (eng-hin) | 0.2647 |
| GPT-2 (eng-tel) | 0.2794 |

#### MuBench — English (zero-shot, length-normalised log-likelihood)

| Model | ARC-C | ARC-E | BMLAMA | GPQA | HellaSwag | MMLU | MMLU-Pro | MNLI | SNLI | StoryCloze | TruthfulQA | WinoGrande | Avg |
|-------|-------|-------|--------|------|-----------|------|----------|------|------|------------|------------|------------|-----|
| GPT-2 (eng-hin) | 0.2577 | 0.2488 | 0.1047 | 0.2232 | 0.2563 | 0.2492 | 0.1118 | 0.3590 | 0.3567 | 0.5240 | 0.2024 | 0.4988 | 0.2827 |
| GPT-2 (eng-tel) | 0.2448 | 0.2488 | 0.1032 | 0.2545 | 0.2437 | 0.2443 | 0.1081 | 0.3246 | 0.3145 | 0.5333 | 0.2449 | 0.4988 | 0.2803 |

---

### Hindi evals (eng-hin models only)

#### Perplexity

| Model | Full test | Filtered (training sources only) | OS-data |
|-------|-----------|----------------------------------|---------|
| GPT-2 (eng-hin) | 131.05 | 93.05 | 879.17 |

> Full test: `pulipakav-1/translated-babylm-hindi` (all sources).
> Filtered: `pulipakav-1/translated-babylm-hindi` — `childes.test.txt.train.hi.txt` + `gutenberg.test.txt.train.hi.txt`.
> OS-data: perplexity on the Hindi side of `Helsinki-NLP/opus-100` (`en-hi` config, full split, streamed).

#### M-BLiMP (`hin`, 1,447 pairs)

| Model | Accuracy |
|-------|----------|
| GPT-2 (eng-hin) | 0.9150 |
| GPT-BERT (eng-hin) | 0.9136 |

#### SIB-200 (`hin_Deva`)

| Model | Accuracy |
|-------|----------|
| GPT-2 (eng-hin) | 0.1324 |

#### MuBench — Hindi (zero-shot, length-normalised log-likelihood)

| Model | ARC-C | ARC-E | BMLAMA | GPQA | HellaSwag | MMLU | MMLU-Pro | MNLI | SNLI | StoryCloze | TruthfulQA | WinoGrande | Avg |
|-------|-------|-------|--------|------|-----------|------|----------|------|------|------------|------------|------------|-----|
| GPT-2 (eng-hin) | 0.2431 | 0.2357 | 0.1064 | 0.2455 | 0.2490 | 0.2624 | 0.1108 | 0.3167 | 0.3289 | 0.4628 | 0.2126 | 0.5012 | 0.2729 |

---

### Telugu evals (eng-tel models only)

#### Perplexity

| Model | Full test | Filtered (training sources only) | OS-data |
|-------|-----------|----------------------------------|---------|
| GPT-2 (eng-tel) | 258.52 | 270.56 | 3937.21 |

> Full test: `pulipakav-1/translated-babylm-telugu` (all sources).
> Filtered: `pulipakav-1/translated-babylm-telugu` — `childes.test.txt.train.te.txt` + `gutenberg.test.txt.train.te.txt`.
> OS-data: perplexity on the Telugu side of `Helsinki-NLP/opus-100` (`en-te` config, full split, streamed).

#### SIB-200 (`tel_Telu`)

| Model | Accuracy |
|-------|----------|
| GPT-2 (eng-tel) | 0.2304 |

#### MuBench — Telugu (zero-shot, length-normalised log-likelihood)

| Model | ARC-C | ARC-E | BMLAMA | GPQA | HellaSwag | MMLU | MMLU-Pro | MNLI | SNLI | StoryCloze | TruthfulQA | WinoGrande | Avg |
|-------|-------|-------|--------|------|-----------|------|----------|------|------|------------|------------|------------|-----|
| GPT-2 (eng-tel) | 0.2337 | 0.2480 | 0.1004 | 0.2522 | 0.2549 | 0.2446 | 0.1063 | 0.3167 | 0.3289 | 0.4799 | 0.2466 | 0.5054 | 0.2765 |

---

## OpenSubtitles Corpus Statistics

Word/byte counts for the target-language side of `Helsinki-NLP/opus-100` (full `train` split, streamed, UTF-8), used for the word-to-byte ratio comparison against the GPT-5-mini translated curriculum data in the paper.

| Language | Words | Bytes | Ratio (word:byte) |
|----------|-------|-------|--------------------|
| Hindi (`en-hi`) | 7,295,887 | 87,313,577 | 1:11.97 |
| Telugu (`en-te`) | 263,812 | 4,533,526 | 1:17.18 |

> Translated curriculum data (Hindi/Telugu) word/byte counts — pending, needs to be computed on the cluster from `curriculum_data/hindi.txt` and `curriculum_data/telugu.txt`.
