# Literature Review

**Topic:** Chain-of-Thought Reasoning as an Interpretability Layer for Brain-Age and Amyloid-Tau Prediction Models
**Phase:** `resource_finder` | **Date:** 2026-08-26
**Corpus:** 45 PDFs in `papers/` (44 arXiv + 1 open-access journal), selected from 473 unique arXiv hits across 18 structured queries.

> **Search-tooling note.** The `paper-finder` service at `localhost:8000` was **not running**
> (verified; see `paper_search_results/*.jsonl`). All literature discovery therefore used the
> fallback path: a purpose-built arXiv Atom-API harness (`code/arxiv_search.py`) over 18 queries,
> plus Crossref for the user-specified journal references. Semantic Scholar returned HTTP 429
> (rate-limited) and was not usable. This is a real limitation: results are relevance-ranked by
> arXiv's own scoring, not by a dedicated ranker, and **non-arXiv venues are under-represented**.

---

## 1. Research Area Overview

The hypothesis sits at the intersection of three literatures that have, until very recently,
barely touched each other:

1. **AD biomarker / brain-age prediction.** A mature, benchmark-driven field (TADPOLE, ADNI,
   OASIS) where *gradient-boosted trees remain the model to beat* and interpretability is
   delivered **post-hoc** (almost universally SHAP).
2. **Tabular foundation models and tabular LLMs.** TabPFN and its descendants have displaced
   bespoke deep nets on small clinical tables; tabular-LLM variants (TAP-GPT, TabReason) add
   natural-language reasoning to the same task.
3. **Chain-of-thought faithfulness.** A skeptical, methodologically sharp literature showing that
   verbalised reasoning frequently is *not* a faithful account of the computation that produced
   the answer.

The central observation from this review: **strands (1) and (2) assert interpretability but never
measure it, while strand (3) has built rigorous measurement tools but has never applied them to a
clinical biomarker task.** That gap is precisely what this project can close.

---

## 2. Key Papers

### 2.1 Ding et al. 2025 — L2C-TabPFN (user-specified reference) — *deep read*
`papers/2508.17649_Ding2025_AD_progression_tabular_foundation_model.pdf` (ML4H 2025)

- **Method.** *Longitudinal-to-cross-sectional (L2C)* transform + TabPFN. For each numeric
  feature `x` and each visit `t`, it computes: most-recent value `mr_x` (eq. 1), time since
  most recent `Δt_mr` (eq. 2), change rate `mr_change_x` (eq. 3), min `low_x` (eq. 4) and
  `Δt_low` (eq. 5), max `high_x` (eq. 6) and `Δt_high` (eq. 7); plus DX-history indicators
  (most-recent / best / worst diagnosis, a "milder-state" flag). A `horizon` feature encodes the
  forecast interval. Augmentation yields `n(n-1)/2` examples per patient.
- **Data.** TADPOLE D1 (train) → D2 (test). Sizes after augmentation: DX 7,340 train / 2,538 test;
  ADAS 7,075 / 2,551; Ventricles 4,990 / 1,241. Splits are **by patient**.
- **Tasks & metrics.** DX (3-class CN/MCI/AD) → MAUC + BCA; ADAS-Cog → MAE; Ventricles
  (ICV-normalised) → MAE.
- **Baseline.** *Frog* = L2C + XGBoost, the TADPOLE SOTA. Optuna tuning (90 trials TabPFN,
  100 XGB); Wilcoxon signed-rank over 5 seeds.
- **Results.** Frog wins DX (MAUC 0.9258 vs 0.9138; BCA 0.8056 vs 0.7839) and ADAS
  (MAE 6.0030 vs 6.7932); TabPFN wins Ventricles (MAE 0.1577 vs 0.1781). All p = 0.0312.
- **Interpretability = SHAP summary plots only.** The paper's own comparison is qualitative:
  XGBoost gives "sparser, more focused" attributions, TabPFN spreads importance more broadly.
  No faithfulness test, no human evaluation.
- **Caveats the authors flag.** ADAS-Cog forecasting was *worse than predicting the median* for
  every TADPOLE team — so ADAS numbers are near-meaningless as a discriminator.
- **Code:** "available upon publication" — **not found in a public repo** (searched; see §6).
- **Relevance:** This is the exact "standard black-box model" the hypothesis proposes to
  CoT-augment, and its interpretability layer (SHAP) is exactly the incumbent to beat.

### 2.2 Kearney, Yang et al. 2026 — TAP-GPT (closest prior work) — *deep read*
`papers/2603.17191_Tabular_LLMs_interpretable_fewshot_AD_prediction.pdf` (IEEE JBHI)
**Code: https://github.com/sophie-kearney/TAP-GPT — cloned to `code/TAP-GPT/`**

- **Method.** TableGPT2 (schema-aware table encoder + Qwen2.5 decoder), LoRA-finetuned for
  few-shot AD-vs-CN classification. Prompts are *tables*, not serialized text. Constrained
  decoding to `{0,1}`. Disjoint per-split ICL pools (important design detail to avoid leakage).
- **Data.** Four ADNI-derived tables: QT-PAD (15 markers + 4 covariates), and region-level
  structural MRI, **amyloid PET**, **tau PET**.
- **Findings that matter for us:**
  - TAP-GPT ≈ TabPFN in low-shot settings; beats classical ML; competitive with GPT-4.1-mini.
  - **Missingness robustness:** stable to ~40% MCAR masking; at 50%, TabPFN catches up. Also
    evaluated on 433 patients with *naturally occurring* missingness (mean 0.247), k = 8 ICL.
    LLM prompts represent missingness directly — **no imputation needed**.
  - **Self-reflection stability:** GPT-4.1-mini degrades notably under a self-reflection round;
    TAP-GPT holds. Framed as a prerequisite for multi-agent debate.
- **The decisive gap.** The authors state explicitly: *"we did not enable extended 'thinking' or
  reasoning modes for these models, as our objective was constrained to binary classification."*
  Interpretability is assessed **qualitatively** ("aligned with established AD biology") — no
  faithfulness metric, no blinded rating, no accuracy-cost-of-CoT measurement.

### 2.3 Lanham et al. 2023 — Measuring Faithfulness in CoT — *deep read*
`papers/2307.13702_Lanham2023_measuring_faithfulness_CoT.pdf` (Anthropic)

Supplies the **measurement battery** this project needs. Four interventions on the CoT:

| Test | Intervention | Failure mode ruled out |
|---|---|---|
| **Early Answering** | truncate CoT after *k* steps, force an answer | post-hoc reasoning |
| **Adding Mistakes** | corrupt one step, regenerate the remainder | post-hoc reasoning |
| **Paraphrasing** | reword early CoT, regenerate remainder | encoded reasoning / steganography |
| **Filler Tokens** | replace CoT with `...` | gains-from-test-time-compute-alone |

Key results: reliance on CoT varies enormously *by task*; filler tokens give no gain (so compute
alone isn't the mechanism); paraphrase preserves performance (so phrasing isn't a hidden channel);
and **larger models produce *less* faithful reasoning** on most tasks. Sampling: 100 CoTs/question,
nucleus p = 0.95, T = 0.8; NLTK punkt for step splitting.

**Why this matters here:** the experiment plan calls for "clinician and neuroscientist ratings…
blinded evaluation." No clinician panel exists in this pipeline. The Lanham battery is the
defensible, fully automatable substitute — and it measures something *stronger* than a rating
(causal dependence of the answer on the stated reasoning), which is exactly what "interpretability
layer" has to mean if the claim is to be non-vacuous.

### 2.4 The skeptical evidence (must be taken seriously)

- **`2607.00260` — Do MLLMs Need Reasoning to Classify Dementia from Speech? (2026).**
  Naive text-rationale strategies produce *hallucinated and inconsistent rationales* and
  **inferior accuracy versus LLM-free baselines**. Direct counter-evidence to the hypothesis, in
  the dementia domain.
- **`2402.14897` — CoT Unfaithfulness as Disguised Accuracy (2024).** Lanham's
  size↔faithfulness result partly reflects *accuracy* differences; the metric must be
  accuracy-controlled or it measures the wrong thing.
- **`2508.19827` (2025)** — CoT *influence* and CoT *faithfulness* are not aligned; gains are
  limited on "soft-reasoning" tasks (which clinical judgement resembles far more than arithmetic).
- **`2405.15092` (2024)** — models reach correct answers *through invalid reasoning*
  ("unfaithful recovery"), so accuracy alone can never validate a reasoning chain.
- **`2311.17371` — Should we be going MAD? (2023).** Multi-agent debate does **not** reliably beat
  self-consistency or simple ensembling, and is far more expensive. Tempers the "debate-augmented
  variant" in the plan.
- **`2402.11863` (2024)** — argues interpretability must be scored on **three** axes —
  faithfulness, robustness, *and utility* — not faithfulness alone.

### 2.5 Interpretable brain-age and ATN modelling

- **`2305.18370` / `2501.01510` — coVariance Neural Networks (VNN).** Brain-age gap prediction
  where anatomical interpretability is *architectural* rather than post-hoc; explicitly targets
  neurodegenerative cohorts.
- **`2506.17597` — OpenMAP-BrainAge**: generalisable + interpretable brain-age predictor.
- **`2404.05748`** — multimodal normative modelling on **ATN** biomarkers; models heterogeneity
  rather than a single scalar.
- **`2109.03723`** — disentangling AD neurodegeneration from *typical* brain ageing: the
  conceptual justification for fitting the brain-age normative model on controls only.
- **`2212.03868`** — systematic review of DL brain-age; standard metric is **MAE in years**, with
  the well-known regression-to-the-mean bias requiring age-bias correction.

### 2.6 Concept bottlenecks — the rival interpretability paradigm

`2209.09056` (Concept Embedding Models) and `2407.06600` (clinical knowledge in CBMs) represent
the main alternative: force predictions through human-named concepts. A CoT chain is essentially a
*generative, unconstrained* bottleneck. Worth citing as the contrast class; out of scope to build.

---

## 3. Common Methodologies

| Approach | Papers |
|---|---|
| L2C feature engineering + GBDT (Frog) | Ding 2025; Marinescu 2020 (`2002.03419`) |
| TabPFN / tabular foundation models | Ding 2025; `2604.27195`; `2511.08667`; `2502.17361` |
| Tabular LLM w/ ICL prompting | TAP-GPT (`2603.17191`, `2507.23227`); `2505.13421` |
| RL-trained reasoning LLM for tabular | TabReason (`2505.21807`) |
| Post-hoc SHAP attribution | Ding 2025; broadly across AD-ML |
| CoT intervention batteries | Lanham (`2307.13702`); `2510.04040`; `2307.11768` |
| Multi-agent debate / adjudication | `2311.17371`; `2401.05998`; `2505.19477`; `2502.19559` |

## 4. Standard Baselines

1. **XGBoost (+ L2C when longitudinal)** — the strongest and most-used baseline; beat TabPFN on
   2 of 3 TADPOLE tasks.
2. **TabPFN v2** — SOTA on small clinical tables; AUC 0.892 for MCI→AD conversion (`2604.27195`),
   with its advantage *largest at N ≈ 50*.
3. **Logistic regression / Random Forest** — clinical standards, still competitive at small N.
4. **Direct-answer LLM (no CoT)** — the *essential* control for this hypothesis. Without it,
   "CoT helps" is unfalsifiable.
5. **Filler-token LLM** — controls for test-time compute (Lanham).

## 5. Evaluation Metrics

- **Brain age:** MAE (years) primary, RMSE, Pearson r; **age-bias-corrected brain-age gap (BAG)**;
  and BAG group separation (control vs demented).
- **Biomarker status:** ROC-AUC primary; balanced accuracy; F1. For 3-class, **MAUC + BCA**
  (Ding's protocol).
- **Interpretability (automatable):** Lanham battery →
  *early-answering AUC*, *same-answer rate after mistake injection*, *paraphrase stability*,
  *filler-token Δaccuracy*. Plus a structured **clinical-plausibility rubric** scored by a held-out
  LLM judge (grounding-in-cited-features, ATN-ordering validity, arithmetic consistency,
  hallucinated-feature rate). The rubric is the honest stand-in for the clinician panel — and
  *must be reported as such*, not as a clinician rating.
- **Robustness:** performance vs. MCAR missingness sweep (0–50%, per TAP-GPT) and vs. Gaussian
  feature noise sweep; report degradation slope, not just endpoint.

## 6. Gaps and Opportunities

1. **Nobody has measured CoT faithfulness on a clinical biomarker task.** The faithfulness
   literature is ~entirely multiple-choice QA (ARC, AQuA, MMLU, HellaSwag).
2. **CoT for *regression* is essentially unstudied.** Every faithfulness test above assumes
   discrete answer choices. Brain-age is continuous — adapting Early Answering / Adding Mistakes
   to a regression target is a genuine methodological contribution, not just an application.
3. **The accuracy cost of CoT is unmeasured in this domain.** TAP-GPT deliberately disabled
   reasoning modes; `2607.00260` found rationales *hurt*. The direct-answer vs CoT comparison at
   matched model/prompt is missing.
4. **Interpretability claims in AD-ML are asserted, not evaluated.** Ding et al. and TAP-GPT both
   assert clinical alignment from eyeballed SHAP plots / sample rationales.
5. **Debate is oversold.** `2311.17371` should keep the debate arm strictly optional.

## 7. Recommendations for Our Experiment

- **Datasets:** Allen ACT (real amyloid/tau + neuropath ground truth) for the ATN arm; OASIS-1+2
  for brain-age; OASIS-2 L2C for longitudinal. All three are **built and validated** in
  `datasets/derived/` — see `datasets/README.md`.
- **Baselines:** XGBoost, TabPFN v2 (local ckpt, verified working on GPU), logistic regression,
  **plus** direct-answer LLM and filler-token LLM.
- **Models:** Qwen2.5-7B-Instruct and Qwen2.5-1.5B-Instruct, both pre-downloaded. The size pair is
  not incidental — it tests Lanham's *smaller-models-are-more-faithful* finding in a clinical
  setting.
- **Metrics:** as §5. Report accuracy and faithfulness *jointly*; a chain that is plausible but
  non-causal is a negative result, and should be reported as one.
- **Methodological cautions:**
  - Split **by subject**, never by row (Allen has 4 regions/donor; OASIS has repeat visits).
  - Age in the Allen cohort is **binned above 90** — 34/107 donors are censored. Use the
    `age_censored` flag; do not run brain-age regression on Allen.
  - Apply age-bias correction to BAG; uncorrected BAG is a known artefact.
  - Accuracy-control any faithfulness comparison across model sizes (`2402.14897`).
  - Fix decoding config across arms; CoT-vs-direct comparisons are invalid at different temperatures.
