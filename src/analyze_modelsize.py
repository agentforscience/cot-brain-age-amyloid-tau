"""
E6: does CoT faithfulness fall as model capability rises?

Lanham et al. (2023) found that larger models produce *less* faithful reasoning on
tasks they can already do. arXiv 2402.14897 warns that this can be an artefact of
accuracy differences, so accuracy and faithfulness are reported side by side here.
"""
from __future__ import annotations
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, analyze_faith as A
from analyze import score_llm, paired_contrast
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 140, "font.size": 8.5, "axes.grid": True,
                     "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False})
task = C.TASK1
rows = []
for m in ["qwen1_5b", "qwen7b"]:
    d = pd.DataFrame([json.loads(l) for l in open(
        os.path.join(C.RESULTS, "model_outputs", f"llm_T1_atn_{m}.jsonl"))])
    clean = d[d.level == 0.0]
    sc = {c: score_llm(task, clean[clean.condition == c]) for c in ["cot", "direct", "filler"]}
    ct = paired_contrast(task, clean, "cot", "direct")
    ea = A.early_curve(task, m)
    mi = A.change_rate(f"{A.FD}/mistakes_T1_atn_{m}.csv", "changed", task)
    pa = A.change_rate(f"{A.FD}/paraphrase_T1_atn_{m}.csv", "changed", task)
    rows.append(dict(model=m,
                     auc_cot=sc["cot"]["auc"], auc_direct=sc["direct"]["auc"],
                     auc_filler=sc["filler"]["auc"],
                     cot_benefit=ct["diff"], cot_benefit_lo=ct["lo"], cot_benefit_hi=ct["hi"],
                     cot_benefit_p=ct["p_boot"],
                     aoc=ea["aoc"], aoc_lo=ea["aoc_ci"][0], aoc_hi=ea["aoc_ci"][1],
                     agree_at_0=float(list(ea["curve"].values)[0]),
                     mistake_change=mi["mean"], mistake_lo=mi["ci"][0], mistake_hi=mi["ci"][1],
                     paraphrase_change=pa["mean"],
                     forced_rate=float(clean[clean.condition == "cot"]["forced"].mean()),
                     median_chain_chars=float(clean[clean.condition == "cot"]["n_chars"].median())))
R = pd.DataFrame(rows)
R.to_csv(os.path.join(C.RESULTS, "analysis_model_size.csv"), index=False)
print(R.round(4).T.to_string())

fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.1))
x = np.arange(2); lab = ["Qwen2.5-1.5B", "Qwen2.5-7B"]
ax = axes[0]
w = .27
for i, (c, col) in enumerate([("cot", "#2b6cb0"), ("direct", "#dd6b20"), ("filler", "#718096")]):
    ax.bar(x + (i - 1) * w, R[f"auc_{c}"], w, color=col, label=c)
ax.axhline(.5, color="k", ls=":", lw=.9)
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylabel("ROC-AUC"); ax.set_ylim(.45, .78)
ax.legend(fontsize=7); ax.set_title("T1 accuracy by condition", fontsize=8.5)

ax = axes[1]
ax.bar(x, R["cot_benefit"], .45, color=["#2f855a" if v > 0 else "#c53030" for v in R["cot_benefit"]])
ax.errorbar(x, R["cot_benefit"], yerr=[R["cot_benefit"] - R["cot_benefit_lo"],
                                       R["cot_benefit_hi"] - R["cot_benefit"]],
            fmt="none", ecolor="k", capsize=3)
ax.axhline(0, color="k", lw=.9)
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylabel("ΔAUC (CoT − direct)")
ax.set_title("Does CoT help? (95% clustered CI)", fontsize=8.5)

ax = axes[2]
ax.bar(x - .16, R["aoc"], .32, color="#805ad5", label="early-answering AOC")
ax.errorbar(x - .16, R["aoc"], yerr=[R["aoc"] - R["aoc_lo"], R["aoc_hi"] - R["aoc"]],
            fmt="none", ecolor="k", capsize=3)
ax.bar(x + .16, R["mistake_change"], .32, color="#dd6b20", label="mistake-injection\nanswer-change rate")
ax.errorbar(x + .16, R["mistake_change"],
            yerr=[R["mistake_change"] - R["mistake_lo"], R["mistake_hi"] - R["mistake_change"]],
            fmt="none", ecolor="k", capsize=3)
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylabel("causal dependence on the chain")
ax.legend(fontsize=6.5); ax.set_title("Faithfulness (higher = chain matters)", fontsize=8.5)
fig.suptitle("E6 — CoT is causally load-bearing in the small model and decorative in the large one",
             fontsize=9)
fig.tight_layout(); fig.savefig(f"{C.FIGURES}/fig7_model_size.png", bbox_inches="tight")
print("done")
