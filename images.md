# Paper Figures

## Figure 1 — Training Curves

**What:** Loss (and/or perplexity) over training steps for each model, one panel per language/architecture.

**Why:** Demonstrates convergence and justifies the choice of training steps/checkpoints.

**Panels:**
- English GPT-2 (mono)
- English GPT-BERT (mono)
- Hindi GPT-2 (mono)
- Hindi GPT-BERT (mono)
- Telugu GPT-2 (mono)
- Telugu GPT-BERT (mono)
- eng-hin GPT-2 (bilingual)
- eng-hin GPT-BERT (bilingual)
- eng-tel GPT-2 (bilingual)
- eng-tel GPT-BERT (bilingual)

**Source:** Training logs / W&B

**Status:** [ ] todo

---

## Figure 2 — Translation Examples Table

**What:** Side-by-side English / Hindi / Telugu sentence triples from different corpus sources, showing translation quality. Include at least one "good" example and one "bad"/artifact-y example.

**Why:** Directly supports the translation-quality caveat in Methods and strengthens the Limitations section.

**Sources to sample from:** CHILDES, Gutenberg, BNC Spoken

**Columns:** Source | English | Hindi | Telugu | Notes

| Source | English | Hindi | Telugu | Notes |
|--------|---------|-------|--------|-------|
| CHILDES | | | | good example |
| Gutenberg | | | | good example |
| BNC Spoken | | | | artifact / bad example |

**Status:** [ ] todo

---

## Figure 3 — Byte / Token Length Distributions

**What:** Histogram or violin plot comparing sentence length in bytes or tokens across English / Hindi / Telugu, supporting the byte-ratio numbers cited in Methods.

**Why:** Visualises the byte premium (Hindi ≈ 2.60×, Telugu ≈ 2.77× over English) at the sentence level.

**Variants to show:**
- Raw byte length distribution (English vs Hindi vs Telugu)
- Token length distribution after tokenisation (per-model tokeniser)

**Reference numbers:** Hindi byte premium = 2.5990 · Telugu byte premium = 2.7672 (measured over 11.6M aligned pairs, std dev < 0.001)

**Status:** [ ] todo

---

## Figure 4 — Per-Model Score Breakdown

**What:** Full per-seed or per-checkpoint score breakdown (e.g., boxplots across multiple training seeds) for all evaluation metrics, supplementing the averaged main-text tables.

**Why:** Adds rigour and shows variance across seeds without cluttering the main tables.

**Metrics to include:**
- BLiMP (English)
- M-BLiMP (Hindi)
- MuBench sub-tasks
- Perplexity (test / OS-data / essay)

**Format:** Boxplot or bar chart with error bars, one panel per metric, grouped by model family (GPT-2 vs GPT-BERT, mono vs bilingual)

**Status:** [ ] todo
