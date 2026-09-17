# Research State

- Current phase: `None`
- Pipeline completed: `True`

## Previous phases

resource_finder (succeeded), experiment_runner (succeeded)

## Current phase context

- Phase: `experiment_runner`
- Status: `completed`
- Started: `2026-08-26T06:47:45.006393Z`
- Next steps:
  - Validate the report and experimental artifacts before finalizing.

## Workspace check

- Expected: `/workspaces/chain_of_thought_reasoning_as__20260826_061743_2c65c00c`
- Actual: `/app`
- Directory usable: `True`
- Current process matches workspace: `False`

## Output validation

- Valid: `True`
- Expected: `REPORT.md`
- Missing: None
- Outside workspace: None

## Agent notes

<!-- NEURICO_AGENT_NOTES_START -->
### resource_finder
<!-- NEURICO_AGENT_NOTES_START:resource_finder -->
### Phase 1 (`resource_finder`) — COMPLETE

**Completed:** isolated `uv` venv + `pyproject.toml` (`[tool.uv] package = false`); literature
search, download and synthesis; dataset acquisition + validation + derivation; repo cloning;
model staging with GPU smoke tests; full documentation.

**Artifacts on disk**
- `papers/` — 45 PDFs (139 MB) + `papers/README.md`; chunks in `papers/pages/`
- `literature_review.md`, `planning.md` (direction budget), `resources.md`
- `datasets/` — 8 raw dirs + `datasets/derived/` (3 analysis-ready CSVs) + `.gitignore`,
  `README.md`, `download.sh`, `samples/`
- `code/` — 4 cloned repos + 3 project scripts + `README.md`
- `models/` — Qwen2.5-7B/1.5B-Instruct, TabPFN v2 clf+reg ckpts + `README.md`

**Key findings**
1. `paper-finder` (localhost:8000) was DOWN. Fell back to a custom arXiv Atom-API harness
   (`code/arxiv_search.py`), 18 queries -> 473 unique papers -> 44 downloaded. Semantic
   Scholar was rate-limited (429). Non-arXiv venues are under-represented as a result.
2. **ADNI / TADPOLE / AIBL / OASIS-3 are all access-gated** (LONI login; HF `medarc/adni-*`
   -> 401). Both the user-specified reference (Ding et al. 2025) and the closest prior work
   (TAP-GPT) use ADNI. We replicate their *protocols* on open cohorts; absolute numbers are
   NOT comparable to their published tables.
3. **Found real amyloid/tau data**: Allen Institute ACT study (`aging.brain-map.org`, open
   RMA API) — 107 donors x 4 regions, quantified Ab40/Ab42/tau/ptau/AT8 + Braak/CERAD/
   NIA-Reagan ground truth + APOE4. This is the ATN arm.
4. **The research gap is sharp and confirmed.** Ding et al. deliver interpretability via
   post-hoc SHAP only. TAP-GPT states explicitly it "did not enable extended thinking or
   reasoning modes." Nobody has measured CoT *faithfulness* on a clinical biomarker task,
   and CoT for *regression* (brain age) is essentially unstudied.
5. **Counter-evidence exists and must be respected**: `2607.00260` found text rationales gave
   INFERIOR dementia-classification accuracy vs LLM-free baselines; `2311.17371` found
   multi-agent debate does not reliably beat self-consistency. The hypothesis may well be
   falsified — design for that outcome.
6. Deep-read (all chunks): Ding et al. 2025 (L2C eqs 1-7, MAUC/BCA/MAE, Frog baseline),
   TAP-GPT (ICL pools, missingness protocol, self-reflection), Lanham et al. 2023
   (4-test faithfulness battery: early-answering / adding-mistakes / paraphrasing / filler).

**Direction budget** — top 3 kept (full scoring + pruning reasons in `planning.md`):
- **D1** CoT interpretability for amyloid/tau status on Allen ACT (primary)
- **D2** CoT for brain-age regression on OASIS (+ adapting faithfulness tests to a continuous target)
- **D3** L2C longitudinal progression + CoT on OASIS-2 (ties to the user-specified reference)
Pruned: debate (optional stretch only), concept bottlenecks, TAP-GPT finetune reproduction,
multi-omics transfer, clinician panel (impossible), standalone steganography study, tipping-points.

**Environment facts the next phase needs**
- 4x RTX A6000 (49 GB), 32 CPUs, 503 GB RAM, 774 GB free disk. CUDA available, torch 2.13.0+cu130.
- **No LLM API keys.** All LLM work must run locally; models are pre-staged. Set `HF_HOME=$PWD/models`.
- **TabPFN**: pass `model_path="models/tabpfn/tabpfn-v2-*.ckpt"` — the library's own downloader
  needs a license token we do not have. Verified working on GPU.
- **transformers is v5**: `apply_chat_template(..., return_dict=True)` then `generate(**enc)`;
  use `dtype=` not `torch_dtype=`. Snippet in `models/README.md`.
- End-to-end CoT generation was smoke-tested on real Allen ACT data and works.

**Unresolved uncertainty / risks**
- The hypothesis clause "as assessed by human clinicians" **cannot be tested here** — no clinician
  panel exists. Substituted by the Lanham causal-faithfulness battery + an LLM-judged
  clinical-plausibility rubric. This must be reported as a substitution, not as a clinician rating.
- n=107 donors on the ATN arm is small; fine for the few-shot framing but underpowered for small
  effects. Kaggle synthetic cohort (2,149) is the fallback scale-up arm (and is **synthetic** —
  label it as such).
- Allen ACT is post-mortem tissue, not in-vivo PET, and its `age` is binned above 89
  (`age_censored` flag). Do **not** run brain-age regression on Allen.

**Next phase:** `experiment_runner`. Concrete next steps:
1. `source .venv/bin/activate`; confirm `datasets/derived/*.csv` load and split **by
   `donor_id` / `Subject_ID`** (never by row — Allen has 4 regions/donor, OASIS repeat visits).
2. Build the black-box arm: XGBoost + TabPFN v2 + logistic regression on D1/D2/D3.
3. Build the LLM arm with three matched prompt conditions — **direct-answer**, **CoT**, and
   **filler-token** — at fixed decoding config. The direct-answer and filler controls are what
   make the hypothesis falsifiable; do not skip them.
4. Run the faithfulness battery on the CoT arm; accuracy-control across the 7B/1.5B pair.
5. Run robustness sweeps: MCAR 0-50% and Gaussian feature noise.
6. Report accuracy and faithfulness jointly. A plausible-but-non-causal chain is a negative
   result and should be written up as one.
<!-- NEURICO_AGENT_NOTES_END:resource_finder -->

### experiment_runner
<!-- NEURICO_AGENT_NOTES_START:experiment_runner -->
### Phase 2 (`experiment_runner`) — COMPLETE

**Deliverables on disk:** `REPORT.md` (primary), `README.md`, `CODE_WALKTHROUGH.md`,
`planning.md` (+ Phase-0 motivation, pre-registered analysis plan, amendments A1/A2),
`src/*.py` (9 scripts), `results/` (12 analysis tables + 7,500 raw generations in
`results/model_outputs/` + `results/faithfulness/`), `figures/fig1..fig7*.png`, `logs/`.

**What was run.** Three tasks (T1 amyloid/tau Braak stage on Allen ACT, 377 samples/107 donors;
T3 OASIS-2 L2C progression, 223/150; T2 OASIS brain age, 323/207), each with a black-box arm
(logreg / XGBoost / TabPFN v2 / trivial) and an LLM arm in three matched prompt conditions
(direct / CoT / filler-token) under identical grouped CV, identical features, and identical
perturbations. Plus: the Lanham faithfulness battery, two new metrics (CFIT, ESA), MCAR + Gaussian
robustness sweeps, a blinded LLM plausibility rubric with a shuffled-explanation control, the
brain-age-gap clinical endpoint, and a 1.5B-vs-7B model-size arm. ~14,500 LLM generations, ~2 h GPU,
$0 API cost (no keys in env; all local weights).

**Headline findings (all in REPORT.md with CIs and tests).**
1. **H1a refuted on all three tasks.** CoT AUC 0.616 vs XGBoost 0.794 (T1); 0.681 vs TabPFN 0.842
   (T3); MAE 11.4 y vs TabPFN 5.3 y (T2) — worse than predicting the cohort mean (9.1 y).
2. **H1b refuted on T1**: CoT is significantly *worse* than compute-matched filler tokens
   (dAUC -0.090 [-0.156,-0.024], q=0.048). Apparent CoT gain on T3 is largely a
   confidence-verbalisation artefact (direct emitted only 2 distinct confidence values;
   on balanced accuracy direct 0.702 > CoT 0.666). CoT does genuinely improve *calibration* on T3
   (ECE 0.083 vs 0.314).
3. **H2a refuted for the deployed model size.** Early-answering AOC 0.066 on T1; agreement with the
   full-chain answer at 0% of chain shown = 82.9%; mistake-injection changes the answer 3.6% of
   the time. The chain is post-hoc narration.
4. **The dissociation that is the paper's point:** the same chains score 3.67/4 on a blinded
   plausibility rubric whose shuffled-explanation control passes (p~1e-23). Persuasive + inert.
5. **Clinical endpoint is empty:** conventional models recover the textbook +4.3 y brain-age gap
   in dementia (AUC 0.70, p<1e-4); the CoT-derived gap is +0.15 y, AUC 0.490 = chance.
6. **Genuinely new positives.** (a) CoT is far more causally load-bearing on the *continuous*
   target (AOC 0.378 vs 0.066; mistake sensitivity 0.325 vs 0.036) — invisible to the
   multiple-choice faithfulness literature. (b) CFIT: cited features move the answer 1.4-2.1x more
   than uncited ones (p=0.0005 T3, p=0.007 T2). (c) **E6 reversal**: CoT *helps* the 1.5B model
   (dAUC +0.082, p=0.012) with 3.7x higher AOC and 6x higher mistake sensitivity, and *hurts* the
   7B model. CoT is faithful where the model needs it and decorative where you would deploy it.
7. **ESA refuted on the primary task** (rho=0.126, p=0.629): chains cite total tau and pTau in 100%
   of cases, so citation frequency carries no information about which biomarker matters.
8. **Robustness (H3) supported in slope, vacuous in level.** CoT is flat under MCAR on T1/T2 where
   XGBoost collapses (T2 MAE 5.7->14.2 y), and beats XGBoost on T2 beyond ~25% missingness — but
   TabPFN beats the LLM at every level, and on T3 the CoT arm is the *least* robust arm (-22.2%).

**Deviations from the plan, both pre-implementation and both audited in results.**
- **A1**: T1 target changed `y_niareagan_AD` -> `y_braak_ge4` (75.7% vs 45.8% positive; only 26
  negative donors under the original). Audit: every model is near chance on the original target
  (`results/tabular_T1_secondary_target.csv`, best AUC 0.674).
- **A2**: T3 feature block de-circularized. With the full L2C features every tabular model hit
  **AUC = 1.000** driven solely by `mr_demented`, because OASIS derives `demented` from CDR.
  Removing CDR/dementia-derived history features gives a real task (XGBoost 0.840). This is itself
  a reportable data-provenance finding about naive L2C on OASIS-2.
- The debate arm (D4) was not run; it was already marked optional and pruned in planning.md, and
  the budget went to the faithfulness battery and the model-size arm instead.

**Bugs found and fixed during the run (all in the final code).**
- The T1 secondary-target audit was overwriting the primary `oof_T1_atn_*.npy` files -> OOF files
  are now namespaced by target. Symptom was T1 tabular AUC ~0.29.
- Two stage-2 processes wrote the same `citation_rates_*.json` / `judge_scores_*.csv` paths
  concurrently, clobbering T1 -> judge output is now task-scoped and merged; citation rates were
  recomputed offline for all three tasks.
- The queued T3 launcher deadlocked because its own command line matched its `pgrep` guard.
- Batch 48 OOM'd on 3.2k-token prompts -> length-sorted batching + per-item OOM fallback in
  `llm.Engine.chat`.

**Validation performed.** Fold assignment hashes stable and zero group leakage on all three tasks;
MCAR and noise perturbations bit-identical on repeat; full XGBoost re-run reproduces stored metrics
to 1e-9 on all three tasks; final answer-parse success 100.0% (7B) / 99.5% (1.5B) with zero
generations discarded. No orphaned background processes remain.

**Honest gaps the next phase should know.**
- **No clinician panel** — the rubric is an LLM proxy scored by the same base model that wrote the
  chains, reported as such and never as a clinician rating. Its shuffled-explanation control passes,
  which validates discrimination but not absolute calibration to expert judgement.
- **No ADNI/TADPOLE/AIBL** (access-gated), so absolute numbers are NOT comparable to Ding et al.
  (2025) or TAP-GPT (2026); only within-study contrasts are valid.
- **One model family only.** No LLM API keys existed in the environment, so only Qwen2.5-7B/1.5B
  were tested. A frontier reasoning model (GPT-5 / Claude Sonnet 4.5 / Gemini 2.5 Pro) on the same
  harness is the single highest-value follow-up and could overturn the central claim; `src/llm.py`
  is the only file that would need to change.
- The 1.5B CoT arm had a 45.6% forced-answer rate (chains exceeded the 512-token budget), so the
  *magnitude* of the E6 effect is uncertain even though its direction is supported by three
  independent metrics.
- LLM robustness sweeps used a 150-case subset and fewer perturbation levels than the tabular
  sweeps; LLM noise slopes are two-point fits.
<!-- NEURICO_AGENT_NOTES_END:experiment_runner -->

<!-- NEURICO_AGENT_NOTES_END -->
