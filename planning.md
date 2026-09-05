# Planning — Direction Budget (Phase 1)

Ten plausible research directions were enumerated and scored on four axes (1–5):
**Ev** = evidence support in the reviewed literature, **Rel** = relevance to the stated hypothesis,
**IG** = expected information gain (how much a result changes what the field believes),
**Feas** = implementation feasibility *given the resources actually secured in this workspace*.

| # | Direction | Ev | Rel | IG | Feas | Total | Verdict |
|---|---|---|---|---|---|---|---|
| D1 | CoT as interpretability layer for **amyloid/tau (ATN) status** on Allen ACT, vs black-box baselines, with the Lanham faithfulness battery + noise/missingness sweeps | 5 | 5 | 5 | 5 | **20** | **KEEP #1** |
| D2 | CoT for **brain-age regression** on OASIS (+ brain-age gap), incl. adapting faithfulness tests to a continuous target | 4 | 5 | 5 | 5 | **19** | **KEEP #2** |
| D3 | **L2C longitudinal** progression + CoT on OASIS-2, replicating Ding et al.'s L2C-TabPFN/Frog protocol on open data | 5 | 4 | 4 | 5 | **18** | **KEEP #3** |
| D4 | Debate-augmented variant (two proposers + adjudicator) | 2 | 3 | 3 | 4 | 12 | Pruned → optional stretch inside D1 |
| D5 | Concept-bottleneck model as rival interpretability layer | 4 | 2 | 3 | 3 | 12 | Pruned |
| D6 | Reproduce/finetune TAP-GPT (TableGPT2 + LoRA) | 5 | 4 | 2 | 1 | 12 | Pruned |
| D7 | Multi-omics cross-domain transfer (de Massy / Simancas-Racines style) | 2 | 2 | 2 | 1 | 7 | Pruned |
| D8 | Blinded clinician/neuroscientist rating panel | 5 | 5 | 4 | 0 | 14 | Pruned (impossible here) → substituted |
| D9 | Encoded-reasoning / steganography analysis as its own study | 3 | 2 | 2 | 4 | 11 | Pruned → folded in as one test of D1 |
| D10 | Novel-hypothesis / "tipping point" discovery from reasoning chains | 1 | 3 | 3 | 2 | 9 | Pruned |

## The three retained directions

**D1 — CoT interpretability for amyloid/tau status (primary).**
Allen ACT gives *real, quantified* Aβ40/Aβ42/tau/ptau plus neuropathology ground truth
(Braak, CERAD, NIA-Reagan) — the ATN arm of the hypothesis with genuine biomarkers rather than
proxies. Compare: XGBoost / TabPFN v2 / logistic regression (black-box arm) vs Qwen2.5-7B and
-1.5B in three prompt conditions (**direct answer**, **CoT**, **filler-token**). Score accuracy
(ROC-AUC, balanced accuracy) *and* faithfulness (early-answering, mistake-injection, paraphrase,
filler) *and* robustness (MCAR sweep 0–50%, Gaussian noise sweep).

**D2 — CoT for brain-age regression (secondary).**
OASIS-1 + OASIS-2 pooled (809 scans, ages 18–98, 323 control scans). Fit the normative brain-age
model on controls, compute age-bias-corrected brain-age gap, test whether the gap separates
demented from non-demented. The novel methodological contribution: **the entire CoT-faithfulness
literature assumes a discrete answer set**, so early-answering and mistake-injection must be
redefined for a continuous target (e.g. answer-shift in years rather than answer-flip rate).

**D3 — L2C longitudinal progression + CoT (tie to the user-specified reference).**
`code/build_derived_datasets.py` already reimplements Ding et al.'s L2C equations (1)–(7) on
OASIS-2 (223 patient-visit rows, 150 subjects, 47% positive). This replicates the named baseline's
*protocol* on open data — ADNI/TADPOLE itself is access-gated (see below) — and adds the CoT arm
that Ding et al. lack.

## Why the others were pruned

- **D4 (debate).** `2311.17371` benchmarks multi-agent debate and finds it does *not* reliably beat
  self-consistency or plain ensembling while costing far more. The research plan already marks it
  "optional." Retained only as a stretch goal if D1–D3 complete with budget to spare.
- **D5 (concept bottlenecks).** Strong literature, but it is a *different* interpretability
  paradigm — building one tests a different hypothesis. Cited as contrast class instead.
- **D6 (TAP-GPT reproduction).** Requires the ADNI QT-PAD table (gated) and a LoRA finetuning run
  of TableGPT2. Low information gain: it would reproduce a published result rather than test our
  hypothesis. The repo is cloned for *reference* (prompt construction, ICL-pool discipline).
- **D7 (multi-omics transfer).** The two cited references are oncology (PDAC, breast cancer), not
  neurodegeneration, and neither dataset is openly available. Off-target.
- **D8 (clinician panel).** No clinicians are available to an automated pipeline. Rather than fake
  it, **substituted** by (a) the Lanham causal-faithfulness battery and (b) a structured
  clinical-plausibility rubric scored by a held-out LLM judge. This substitution must be stated
  plainly in the write-up — an LLM judge is *not* a clinician rating, and the hypothesis's
  "as assessed by human clinicians" clause **cannot be tested in this workspace**.
- **D9 (steganography).** Subsumed as the paraphrase test within D1.
- **D10 (tipping points).** No ground truth against which a "novel biological hypothesis" could be
  scored; would produce unfalsifiable output.

## Standing constraints discovered during Phase 1

- **ADNI / TADPOLE / AIBL / OASIS-3 are access-gated** (`ida.loni.usc.edu` requires login; HF
  `medarc/adni-*` return HTTP 401). The user-specified reference (Ding et al.) and the closest
  prior work (TAP-GPT) both use ADNI. We replicate their *protocols* on open cohorts instead;
  absolute numbers are therefore **not** directly comparable to their published tables.
- **No LLM API keys** are set in the environment. Mitigated: 4× RTX A6000 (48 GB) available and
  Qwen2.5-7B/1.5B-Instruct pre-downloaded locally, so all LLM work runs offline.
- **TabPFN v2 requires license/token acceptance** for its own weight download. Mitigated: the
  checkpoints are ungated on the HF Hub and are staged at `models/tabpfn/`, verified working on GPU.

## Revision policy

This ranking is fixed for the implementation phase. It will only be revised if new evidence
invalidates it — e.g. if D1 turns out to be underpowered at n = 107 donors, in which case the
Kaggle synthetic cohort (2,149 patients) becomes a scale-up arm and the change gets recorded in
`STATE.md` with the reason.

---
---

# Phase 0 — Motivation & Novelty Assessment
*(written by `experiment_runner` before any implementation)*

## Why This Research Matters

Alzheimer's-disease biomarker models (brain-age, amyloid/tau status, longitudinal progression) are
increasingly accurate and almost never adopted clinically, because a number without a rationale is
not actionable for a clinician deciding on a lumbar puncture, a PET scan, or an anti-amyloid
therapy. The field's current answer is *post-hoc* attribution (SHAP over an XGBoost or a tabular
foundation model — exactly what Ding et al. 2025 do). Post-hoc attributions explain the *model*,
not the *reasoning*, and they are known to be unstable and non-causal. Chain-of-thought (CoT)
promises something different: an *ante-hoc*, natural-language rationale produced by the same
forward pass that produces the answer. If that rationale is causally connected to the answer, it is
a genuinely new kind of clinical interpretability. If it is not, then deploying CoT in a clinical
biomarker pipeline actively manufactures unwarranted trust — a *worse* outcome than an honest
black box. Establishing which of these is true is the point of this study.

## Gap in Existing Work

From `literature_review.md` and the deep-read papers:

1. **Ding et al. (2025, L2C-TabPFN)** — the user-specified reference — achieves SOTA longitudinal
   AD progression prediction but supplies interpretability only through post-hoc SHAP. No reasoning.
2. **TAP-GPT (Kearney & Yang et al., 2026)** — the closest prior work, an LLM over AD tabular data —
   states *explicitly* that it "did not enable extended thinking or reasoning modes." The single
   most obvious experiment in the space has been deliberately left undone.
3. **Lanham et al. (2023)** built a four-test battery for measuring whether a CoT is *causally
   faithful* — but only on generic multiple-choice NLP benchmarks. **Nobody has measured CoT
   faithfulness on a clinical biomarker task**, where the cost of an unfaithful-but-plausible
   rationale is a misdirected diagnostic workup.
4. **CoT for regression is essentially unstudied.** The whole faithfulness literature assumes a
   discrete answer set; "did the answer flip?" is undefined for a continuous target like brain age.
5. **Counter-evidence exists and is deliberately over-sampled here**: arXiv `2607.00260` found text
   rationales gave *inferior* dementia-classification accuracy versus LLM-free baselines, and
   `2311.17371` found multi-agent debate does not reliably beat self-consistency. A negative result
   is a live and likely possibility, and the design must be able to detect it.

## Our Novel Contribution

1. **First measurement of CoT causal faithfulness on a clinical biomarker task** — the Lanham
   battery (early-answering, mistake-injection, paraphrasing, filler-tokens) transplanted onto real
   quantified Aβ40/Aβ42/tau/ptau neuropathology (Allen ACT) and onto longitudinal AD progression.
2. **A faithfulness battery redefined for a continuous target.** We replace "answer-flip rate" with
   a *normalised answer-shift* statistic (shift in predicted years divided by the model's own
   dispersion), making early-answering and mistake-injection well-defined for brain-age regression.
   To our knowledge this is new.
3. **The Cited-Feature Intervention Test (CFIT)** — a new metric proposed here. We parse which
   biomarkers a chain explicitly *cites as decisive*, then perturb cited versus uncited features
   and compare the resulting prediction shift. A faithful explanation must be more sensitive to
   what it says it used than to what it says it ignored. This converts "is the explanation true?"
   from a human-judgement question into a measurable causal quantity — which matters precisely
   because a clinician panel is not available to an automated pipeline.
4. **Explanation–Signal Alignment (ESA)** — do the biomarkers the chain cites coincide with the
   biomarkers that actually carry signal (established independently by permutation importance on
   the black-box arm)? This is an *objective* interpretability proxy, complementing the LLM-judged
   plausibility rubric.
5. **A properly-controlled accuracy comparison.** CoT is compared not only against black-box
   tabular models but against two matched LLM controls — *direct-answer* (same prompt, no reasoning)
   and *filler-token* (same number of extra forward passes, no semantic content). Without these two
   controls, "CoT helps" is untestable; with them, it is falsifiable.

## Experiment Justification

| # | Experiment | Why it is necessary |
|---|---|---|
| **E1** | Black-box tabular baselines (LogReg, XGBoost, TabPFN v2) on all three tasks, donor/subject-grouped CV | Establishes the accuracy ceiling the CoT arm must "match or outperform" per the hypothesis. Without it, LLM numbers are uninterpretable. |
| **E2** | LLM arm × 3 prompt conditions (direct / CoT / filler) × 3 tasks, identical decoding config | The core accuracy test. The filler control separates "reasoning helps" from "extra compute helps"; the direct control separates "reasoning helps" from "the LLM is just good at this". |
| **E3** | Lanham faithfulness battery on the CoT chains (+ the continuous-target adaptation for brain age) | The hypothesis claims *interpretability*. An explanation that does not causally drive the answer is not an explanation. This is the test that can falsify the interpretability claim even if accuracy holds. |
| **E4** | CFIT + ESA (novel metrics) | Faithfulness at the level of *individual cited biomarkers*, which is the granularity a clinician actually reads. ESA additionally checks the chain against independently-estimated ground-truth signal. |
| **E5** | Robustness sweeps: MCAR missingness 0–50%, Gaussian feature noise 0–1.0 SD, applied identically to every arm | Directly tests the hypothesis's "increased robustness to adversarial or noisy data" clause. Must be applied to *all* arms with identical perturbations to be a fair comparison. |
| **E6** | Model-size arm (Qwen2.5-1.5B vs 7B) on T1 | Lanham's central finding is that faithfulness is *inversely* related to capability on easy tasks. Testing two sizes checks whether that scaling result transfers to the clinical domain — and guards against attributing to "CoT" what is really "model size". |
| **E7** | Blinded LLM-judged clinical-plausibility rubric | The *substitute* for the impossible clinician panel. Reported as an LLM proxy, never as a clinician rating. Its value is mainly in contrast with E3/E4: if plausibility is high while faithfulness is low, that is the paper's central warning. |

**Pre-registered analysis commitments** (fixed before any result was seen):
- Primary accuracy endpoint: ROC-AUC (T1, T3) and MAE (T2), donor/subject-grouped 5-fold CV.
- Primary interpretability endpoint: early-answering AOC and CFIT cited-vs-uncited sensitivity ratio.
- α = 0.05, two-sided. Multiple comparisons across the three tasks controlled by
  Benjamini–Hochberg FDR within each metric family.
- CIs by **donor-clustered bootstrap** (2000 resamples), because Allen ACT has 4 regions per donor
  and OASIS has repeat visits per subject; row-level bootstrap would understate uncertainty.
- Equivalence ("match") is assessed by a TOST-style check against a pre-declared margin:
  ΔAUC = 0.05 for classification, ΔMAE = 1.5 years for brain age.
- **We commit to reporting a negative result as a negative result.** If CoT matches accuracy but
  fails the faithfulness battery, the conclusion is that CoT is *not* a valid interpretability layer
  here, and that is what will be written.

---

# Phase 1 — Experimental Plan

## Research Question
Can chain-of-thought reasoning serve as an interpretability layer for Alzheimer's biomarker models
— specifically, does an LLM that emits an explicit A/T/N reasoning chain (a) match black-box
accuracy on amyloid/tau status, brain-age, and progression prediction, (b) produce explanations
that causally drive its own predictions, and (c) degrade more gracefully under noise and
missingness?

## Hypothesis Decomposition

| ID | Sub-hypothesis | Endpoint | Falsified if |
|---|---|---|---|
| H1a | CoT-LLM matches black-box tabular accuracy | ROC-AUC (T1,T3), MAE (T2) | CoT AUC below best baseline by > 0.05 (or MAE worse by > 1.5 y) |
| H1b | CoT beats matched no-reasoning LLM controls | ΔAUC vs direct-answer and filler | CoT ≤ direct or ≤ filler |
| H2a | CoT chains are causally faithful | early-answering AOC, mistake-injection Δ | AOC ≈ 0 (answer known before reasoning); mistakes don't change answers |
| H2b | Chains cite features that actually drive the prediction | CFIT ratio > 1 | cited-feature perturbation ≤ uncited-feature perturbation |
| H2c | Chains cite features that actually carry signal | ESA vs permutation importance | no better than chance ordering |
| H3 | CoT is more robust to noise/missingness | AUC/MAE degradation slope | CoT slope ≥ baselines' slopes |
| H4 | Faithfulness declines with model capability (Lanham) | 1.5B vs 7B AOC | no difference or reversed |

Independent variables: prompt condition (direct / CoT / filler), model (7B / 1.5B), perturbation
type & magnitude, task. Dependent variables: accuracy metrics, faithfulness metrics, robustness
slopes, rubric scores.

## Datasets and Splits
- **T1 (ATN status)** `datasets/derived/allen_act_atn.csv`, 377 region-rows / 107 donors. Target
  `y_niareagan_AD` (NIA-Reagan intermediate/high). Region-level prediction, **GroupKFold by
  `donor_id`**, donor-clustered bootstrap CIs.
- **T2 (brain age)** `datasets/derived/oasis_brainage.csv`. Normative model fit on **controls only**
  (`is_control == 1`), GroupKFold by `Subject_ID`; brain-age gap then applied to demented scans
  with the standard Beheshti/Cole age-bias correction.
- **T3 (progression)** `datasets/derived/oasis_l2c.csv`, 223 rows / 150 subjects, Ding et al.'s L2C
  transform. Target `y_demented`. GroupKFold by `Subject_ID`.

## Baselines
Logistic regression (L2, standardised) · XGBoost · TabPFN v2 (local ckpt) · plus trivial references
(majority-class / mean-age) to detect ceiling and floor effects. The two *LLM* controls
(direct-answer, filler-token) are the ones that make H1b falsifiable.

## Statistical Analysis Plan
Donor/subject-clustered bootstrap (B = 2000) for all CIs and for paired differences between arms;
paired bootstrap p-values for arm-vs-arm contrasts; McNemar's test on paired binary correctness for
LLM condition contrasts; Wilcoxon signed-rank for paired absolute-error comparisons (T2); Spearman
ρ for ESA; ordinary least squares on the perturbation sweep for degradation slopes with clustered
resampling. BH-FDR within metric family. Effect sizes: ΔAUC, Cohen's d, rank-biserial.

## Timeline
Implementation 60–75 min · E1 (tabular, incl. robustness sweep) 20 min · E2 (LLM accuracy) 40 min ·
E3–E4 (faithfulness) 40 min · E5 (LLM robustness) 25 min · E6/E7 20 min · analysis 30 min ·
documentation 30 min. Buffer ~45 min.

## Potential Challenges & Contingencies
- *n = 107 donors is small* → region-level evaluation (377) with clustered CIs; wide CIs reported
  honestly; equivalence margins pre-declared rather than chosen post hoc.
- *LLM answer-parsing failures* → strict `ANSWER:` regex plus a fallback parser; parse-failure rate
  reported per condition and treated as an error, never silently dropped.
- *Prompt leakage* → the target label and all label-derived columns (Braak, CERAD, dementia status)
  are stripped from every prompt; ICL exemplars come only from the training fold.
- *Judge self-preference* → the judge is blinded to condition, sees chains in shuffled order, and
  its scores are reported as an LLM proxy with that limitation stated.
- *Time overrun* → task priority is T1 > T3 > T2 for the LLM arms; faithfulness battery is run on
  T1 first.

## Success Criteria
The study succeeds if it returns a *decisive* answer on H1a/H1b and H2a/H2b with adequately
characterised uncertainty — regardless of the direction of that answer. It fails only if the
experiments are underpowered or confounded to the point where no sub-hypothesis can be adjudicated.

---

## Amendment A1 (pre-implementation, no results seen)

**T1 primary target changed from `y_niareagan_AD` to `y_braak_ge4`.** Class-balance inspection of
the derived file (run before any model was fit) showed NIA-Reagan intermediate/high is positive in
81/107 donors (75.7%), leaving only **26 negative donors** — far too few for a donor-clustered
bootstrap to resolve a ΔAUC of 0.05, and prone to majority-class artefacts. `y_braak_ge4`
(Braak stage IV–VI, i.e. limbic/neocortical spread of neurofibrillary tau) is positive in 49/107
donors (45.8%), is squarely on-hypothesis (it *is* the tau axis of A/T/N), and yields a
well-powered comparison. `y_niareagan_AD` is retained as a **secondary** target for the tabular arm
so the change can be audited. Note for interpretation: AT8 and Tau-2 immunoreactivity in the four
sampled regions are quantitative tau measures and Braak is a semiquantitative topographic stage
assigned by a neuropathologist — related but distinct measurements, so this is a genuine
biomarker→stage prediction task, not label leakage. It is nonetheless expected to be *learnable*,
which is desirable: a task no arm can do would tell us nothing about CoT.

## Amendment A2 (pre-implementation, after inspecting baseline behaviour, before any LLM run)

**T3 feature set de-circularized.** With the full L2C feature block, *every* tabular model reached
ROC-AUC = 1.000 on OASIS-2 (logreg, XGBoost and TabPFN alike), and permutation importance showed
a single feature — `mr_demented` — carrying all of it (ΔAUC = 0.235, every other feature ≈ 1e-17).
The cause is definitional, not statistical: OASIS derives its `demented` label *from* CDR, so
prior-visit CDR and prior-visit dementia status are the label itself, lagged. A saturated task
cannot discriminate between arms, so T3 would have been useless for the hypothesis.

All CDR-derived and dementia-derived history features (`mr_CDR`, `low_CDR`, `high_CDR`,
`mr_change_CDR`, `dt_*_CDR`, `mr_demented`, `worst_demented`, `ever_milder`) are therefore removed.
The remaining 18 features are cognitive (MMSE history), structural (nWBV/eTIV/ASF history),
demographic, and temporal. This yields ROC-AUC ≈ 0.83–0.84 for XGBoost — a genuinely non-trivial
prediction problem with room both above and below for the LLM arms to land. Retaining MMSE is
deliberate: MMSE is a cognitive screening instrument, not the diagnostic criterion, so it is a
legitimate predictor rather than a lagged copy of the label. The saturated result is reported in
`results/tabular_results.csv` (superseded rows) and discussed as a data-provenance finding in its
own right.
