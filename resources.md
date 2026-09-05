# Resources Catalog

**Phase:** `resource_finder` (complete) | **Date:** 2026-08-26

## Summary

| Resource | Count | Location |
|---|---|---|
| Papers (PDF) | **45** (44 arXiv + 1 journal) | `papers/` |
| Datasets (raw) | **8** directories | `datasets/` |
| Datasets (derived, analysis-ready) | **3** | `datasets/derived/` |
| Code repositories cloned | **4** | `code/` |
| Models staged & verified | **4** | `models/` |

Everything below has been **downloaded and validated on disk**, not merely located.

---

## Papers

45 PDFs, 139 MB. Full annotated list with abstracts: **`papers/README.md`**.
Synthesis and gap analysis: **`literature_review.md`**.

Deep-read (all chunks, via `pdf_chunker.py`; chunk PDFs in `papers/pages/`):

| Paper | File | Why deep-read |
|---|---|---|
| Ding et al. 2025, *L2C-TabPFN* | `2508.17649_*.pdf` | **User-specified reference.** Supplies the exact baseline protocol (L2C eqs. 1–7, MAUC/BCA/MAE, Frog/XGBoost comparison). |
| Kearney & Yang et al. 2026, *TAP-GPT* | `2603.17191_*.pdf` | Closest prior work; has public code; explicitly *disables* reasoning modes → defines our gap. |
| Lanham et al. 2023, *Measuring Faithfulness in CoT* | `2307.13702_*.pdf` | Supplies the four-test faithfulness battery used as the interpretability metric. |

Screened by abstract: 42 further papers spanning brain-age modelling, ATN biomarkers, tabular
foundation models, CoT faithfulness, multi-agent debate, clinical LLM reasoning, and concept
bottlenecks.

Grouping: see the five sections of `papers/README.md` (A–E).

---

## Datasets

Full documentation, download commands, schemas and gotchas: **`datasets/README.md`**.
Rebuild: `bash datasets/download.sh && python code/build_derived_datasets.py`

### Derived (use these)

| Name | Shape | Task | Location | Notes |
|---|---|---|---|---|
| **allen_act_atn** | 377 × 59 | amyloid/tau status; ATN reasoning | `datasets/derived/allen_act_atn.csv` | **Real Aβ40/Aβ42/tau/ptau** + Braak/CERAD/NIA-Reagan. 107 donors × 4 regions. Split by `donor_id`. Age binned >89. |
| **oasis_brainage** | 809 × 18 | brain-age regression; dementia status | `datasets/derived/oasis_brainage.csv` | OASIS-1 + OASIS-2 pooled. 323 controls, ages 18–98. Fit normative model on controls only. |
| **oasis_l2c** | 223 × 50 | longitudinal progression | `datasets/derived/oasis_l2c.csv` | OASIS-2 under Ding et al.'s L2C transform. 150 subjects, 47.1% positive. |

### Raw

| Dir | Source | Size | Note |
|---|---|---|---|
| `allen_aging_dementia_tbi/` | aging.brain-map.org (open API) | 256 KB | 3 CSVs; the ATN source of truth |
| `kaggle_mri-and-alzheimers/` | Kaggle `jboysen/...` | 12 KB | OASIS-1 (436) + OASIS-2 (373) |
| `kaggle_alzheimers-disease-dataset/` | Kaggle `rabieelkharoua/...` | 267 KB | 2,149 × 35 — **synthetic**, scale-up only |
| `kaggle_comprehensive-health-and-brain-imaging-dataset/` | Kaggle `snmahsa/...` | 28 KB | 1,830 × 19 real SVD cohort, heavy missingness |
| `kaggle_alzheimer-allftd-t-test/` | Kaggle `imtkaggleteam/...` | 72 KB | FTD-GRN proteomics summary; contextual |
| `kaggle_alzheimer-features/`, `kaggle_dementia-prediction-dataset/` | Kaggle | 4 + 6 KB | **duplicates of OASIS-2**; provenance only |

Samples (committed): `datasets/samples/*.json`.

---

## Code Repositories

Details: **`code/README.md`**.

| Name | URL | Purpose | Runnable here? |
|---|---|---|---|
| **TAP-GPT** | github.com/sophie-kearney/TAP-GPT | Closest prior work; ICL/prompt construction, interpretable prompting, MAS module | ✗ needs gated ADNI QT-PAD — **design reference** |
| **TabPFN** | github.com/PriorLabs/TabPFN | Tabular foundation model baseline | ✓ verified on GPU (with local ckpt) |
| **tabpfn-extensions** | github.com/PriorLabs/tabpfn-extensions | SHAP-style post-hoc attribution for TabPFN | ✓ |
| **TADPOLE** | github.com/noxtoby/TADPOLE | Official MAUC / BCA metric implementations | ✓ (metrics only) |

Not found: **no public repo for Ding et al. 2025** — the paper says code will be released "upon
publication". Its L2C transform has therefore been **reimplemented from the paper's equations**
in `code/build_derived_datasets.py`.

---

## Models

Details and usage snippets: **`models/README.md`**.

| Path | What | Verified |
|---|---|---|
| `models/hub/…Qwen2.5-7B-Instruct` | primary CoT generator | ✓ produced a valid A/T/N reasoning chain on real Allen ACT hippocampus data |
| `models/hub/…Qwen2.5-1.5B-Instruct` | small-model faithfulness arm | ✓ downloaded |
| `models/tabpfn/tabpfn-v2-classifier.ckpt` | TabPFN v2 clf | ✓ GPU smoke test |
| `models/tabpfn/tabpfn-v2-regressor.ckpt` | TabPFN v2 reg | ✓ GPU smoke test |

---

## Resource-Gathering Notes

### Search strategy
`paper-finder` at `localhost:8000` was **not running**, so literature discovery fell back to a
purpose-built arXiv Atom-API harness (`code/arxiv_search.py`) over **18 structured queries**
spanning the three contributing literatures → **473 unique papers**, screened by title, then 44
selected and downloaded. Crossref resolved the user-specified journal references; Semantic Scholar
was rate-limited (HTTP 429) and unusable.

### Selection criteria
Priority to (1) the user-specified reference, (2) papers with public code, (3) papers supplying
baselines/metrics we will actually run, (4) papers providing *counter*-evidence to the hypothesis
(deliberately over-sampled — see `literature_review.md` §2.4).

### Challenges encountered and how they were resolved

| Challenge | Resolution |
|---|---|
| `paper-finder` service down | Built arXiv API harness; documented the ranking-quality limitation |
| **ADNI/TADPOLE/AIBL/OASIS-3 all access-gated** | Switched to open cohorts (Allen ACT, OASIS-1/2) and replicated the *protocols*; documented that absolute numbers aren't comparable |
| No amyloid/tau in any open tabular AD dataset at first pass | Found the Allen ACT study via its open RMA API — **real quantified Aβ42/Aβ40/tau/ptau + neuropathology** |
| No LLM API keys in environment | Verified 4× RTX A6000; pre-downloaded Qwen2.5-7B/1.5B-Instruct for fully offline inference |
| TabPFN v2 refuses to download weights without a license token | Checkpoints are ungated on HF Hub → staged locally and passed via `model_path=` |
| `transformers` v5 broke `apply_chat_template` | Fixed (`return_dict=True`, `**enc`); documented in `models/README.md` |
| Ding et al. released no code | Reimplemented the L2C transform from the paper's equations |
| Semantic Scholar HTTP 429 | Not used; arXiv + Crossref sufficed |

### Gaps that remain (honest accounting)

1. **No ADNI/TADPOLE.** The headline benchmark of the field is unavailable. Cross-paper numeric
   comparison to Ding et al. and TAP-GPT is impossible; only within-study contrasts are valid.
2. **No clinician panel.** The hypothesis's "as assessed by human clinicians" clause **cannot be
   tested in this workspace.** Substituted by the Lanham causal-faithfulness battery plus an
   LLM-judged clinical-plausibility rubric. This substitution must be stated plainly in the
   write-up rather than presented as a clinician rating.
3. **Small n on the ATN arm** (107 donors). Appropriate for the few-shot framing that TAP-GPT and
   TabPFN target, but underpowered for small effect sizes. The synthetic Kaggle cohort is the
   fallback scale-up arm.
4. **No plasma biomarkers, no PET, no omics.** Allen ACT is post-mortem tissue, not in-vivo PET.
   The "amyloid/tau" arm therefore tests *neuropathological* rather than *imaging* ATN.
5. **Journal references for de Massy, Simancas-Racines, Shuja, Anwar** were resolved via Crossref
   but are paywalled/off-topic (two are oncology, not neurodegeneration) — not downloaded.

---

## Recommendations for Experiment Design

1. **Primary dataset:** `allen_act_atn.csv` (amyloid/tau arm). **Secondary:**
   `oasis_brainage.csv` (brain-age arm). **Tertiary:** `oasis_l2c.csv` (longitudinal arm).
   These map 1:1 to directions D1, D2, D3 in `planning.md`.
2. **Baselines:** XGBoost, TabPFN v2, logistic regression — **plus** the two LLM controls that
   make the hypothesis falsifiable: *direct-answer* (no CoT) and *filler-token* (compute control).
   Omitting these makes any "CoT helps" claim untestable.
3. **Metrics:** ROC-AUC / balanced accuracy (status), MAE + age-bias-corrected brain-age gap
   (brain age), MAUC + BCA (multi-class, via `code/TADPOLE/`); faithfulness via early-answering,
   mistake-injection, paraphrase, filler-token; robustness via MCAR sweep 0–50% and Gaussian noise
   sweep.
4. **Code to reuse:** `code/build_derived_datasets.py` (data + L2C), `code/TADPOLE/` (metrics),
   `code/TAP-GPT/core_functions/` (prompt & ICL-pool patterns), `code/tabpfn-extensions/`
   (post-hoc attribution arm).
5. **Non-negotiable hygiene:** split by `donor_id` / `Subject_ID` (never by row); fix decoding
   config across all LLM arms; accuracy-control any cross-model-size faithfulness comparison
   (`papers/2402.14897_*`); set `HF_HOME=$PWD/models`.
