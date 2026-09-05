# Chain-of-Thought as an Interpretability Layer for Alzheimer's Biomarker Models

Does making a model *explain its reasoning* buy you interpretability in Alzheimer's biomarker
prediction — or just a persuasive story? This study runs chain-of-thought (CoT) LLMs against
conventional tabular baselines on three AD tasks (amyloid/tau neuropathological stage, dementia
progression, brain age), and then measures whether the reasoning chains actually *cause* the
predictions they accompany.

**Full write-up: [REPORT.md](REPORT.md) · Code guide: [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md) ·
Plan and amendments: [planning.md](planning.md)**

## Key findings

- **CoT did not match black-box accuracy on any task.** T1 amyloid/tau: CoT AUC 0.616 vs XGBoost
  0.794. T3 progression: 0.681 vs TabPFN 0.842. T2 brain age: MAE 11.4 y vs TabPFN 5.3 y — *worse
  than simply predicting the cohort mean* (9.1 y).
- **On the primary task CoT was significantly worse than semantically-empty filler tokens** of the
  same length (ΔAUC −0.090, 95% CI [−0.156, −0.024], q = 0.048) — so the extra tokens were not just
  unhelpful, the reasoning content itself cost accuracy.
- **The chains are largely post-hoc.** Forcing an answer with *none* of the chain shown reproduced
  the full-chain answer **82.9%** of the time (early-answering AOC 0.066), and injecting a 10×
  error into a biomarker value inside the chain changed the answer only **3.6%** of the time.
- **Yet a blinded rubric rated those same chains 3.67/4 for plausibility** (validated by a
  shuffled-explanation control that it passes, p ≈ 1e−23). Persuasive and inert is a worse
  combination than an honest black box.
- **The clinical endpoint is empty.** Conventional models recover the textbook **+4.3-year**
  brain-age gap in dementia (AUC 0.70, p < 1e−4); the CoT-derived gap is **+0.15 y**, AUC 0.49 —
  chance.
- **Two results cut the other way.** CoT is far more causally load-bearing on the *continuous*
  target (AOC 0.378 vs 0.066) — the faithfulness literature is entirely multiple-choice and could
  not have seen this. And CoT **helps the 1.5B model (ΔAUC +0.082, p = 0.012) while hurting the 7B
  model**, with 3.7× higher faithfulness at the smaller size: CoT is honest exactly where the model
  needs it, and decorative exactly where you would deploy it.

## Two new metrics introduced here

- **CFIT (Cited-Feature Intervention Test)** — perturb the biomarkers the chain *names as decisive*
  versus ones it ignores, and compare the induced prediction shift. Cited features move the answer
  1.4–2.1× more (significant on 2 of 3 tasks): local citations carry causal weight even when the
  global narrative does not.
- **ESA (Explanation–Signal Alignment)** — rank-correlate citation frequency against permutation
  importance from the black-box arm. On the primary task, ρ = 0.13 (p = 0.63): **no better than
  chance**. The chains cite total tau and pTau in 100% of cases; naming everything explains nothing.

Also contributed: a **continuous-target adaptation of the Lanham et al. (2023) faithfulness
battery**, replacing answer-flip rate with normalised answer shift, so early-answering and
mistake-injection are defined for regression.

## Reproducing

```bash
uv venv && source .venv/bin/activate
export HF_HOME=$PWD/models        # local weights; no API keys and no network needed

python src/run_tabular.py                                    # baselines + robustness + importances
python src/run_llm.py --models qwen7b --tasks T1_atn         # ~35 min / A6000; tasks are independent
python src/run_llm.py --models qwen7b --tasks T2_brainage,T3_l2c
python src/run_llm.py --models qwen1_5b --tasks T1_atn       # model-size arm
python src/run_faithfulness.py --model qwen7b                # Lanham battery + CFIT
python src/run_judge.py --model qwen7b                       # blinded rubric + shuffled control
python src/run_bag.py --model qwen7b                         # brain-age gap clinical endpoint
python src/analyze.py qwen7b && python src/analyze_faith.py qwen7b
python src/analyze_extra.py && python src/analyze_modelsize.py
```

Set `CUDA_VISIBLE_DEVICES` and pass different `--tasks` to run tasks concurrently on separate GPUs
(how the reported run was executed). Seed 42 and greedy decoding throughout; splits are assigned by
seeded permutation of donor/subject IDs, so any script reproduces byte-identical folds.

**Requirements:** 1–2 GPUs with ≥ 24 GB (Qwen2.5-7B bf16 + a 3.2k-token × 24 KV cache peaks ≈ 31 GB),
~20 GB disk for weights. Total ≈ 2 h GPU time, 7,500 LLM generations, **$0 in API cost**.

## Layout

```
REPORT.md              full research report with all results
CODE_WALKTHROUGH.md    code structure, design decisions, data flow
planning.md            direction budget, Phase-0 motivation, pre-registered analysis, amendments A1/A2
literature_review.md   45-paper review (from the resource-gathering phase)
resources.md           catalogue of datasets, papers, code, models

src/                   all experiment and analysis code (see CODE_WALKTHROUGH.md)
datasets/derived/      three analysis-ready CSVs (Allen ACT ATN, OASIS brain age, OASIS-2 L2C)
models/                locally staged Qwen2.5-7B/1.5B-Instruct and TabPFN v2 checkpoints
results/               all metric tables, plus every raw generation in results/model_outputs/
figures/               fig1 accuracy · fig2 early answering · fig3 robustness · fig4 CFIT+judge
                       fig5 ESA · fig6 calibration · fig7 model size
logs/                  complete run logs
```

## Honest caveats

ADNI/TADPOLE/AIBL/OASIS-3 are access-gated, so protocols were replicated on open cohorts and
**absolute numbers are not comparable** to Ding et al. (2025) or TAP-GPT (2026). Cohorts are small
(107 donors / 150 subjects / 207 subjects) and CIs are correspondingly wide. **No clinician panel
was available and none was simulated** — the plausibility rubric is an LLM proxy, reported as such.
Only one model family (Qwen2.5, two sizes) was tested because no LLM API keys existed in the
environment; testing a frontier reasoning model is the most important follow-up and could overturn
the central claim. Full limitations in [REPORT.md §6](REPORT.md).
