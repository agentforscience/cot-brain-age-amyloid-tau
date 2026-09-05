"""
Supplementary analysis: confidence calibration, prediction-rate bias, and the
relationship between chain length and correctness.

Motivation: the classification AUCs are computed from the model's *stated*
confidence, so a coarse or mis-calibrated confidence scale is a candidate
explanation for any AUC difference between prompt conditions. This script checks
that explanation explicitly rather than leaving it as an untested confound.
"""
from __future__ import annotations
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({"figure.dpi": 140, "font.size": 8.5, "axes.grid": True,
                     "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False})
CO = {"cot": "#2b6cb0", "direct": "#dd6b20", "filler": "#718096"}


def ece(y, p, bins=10):
    """Expected calibration error of the stated confidence."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


rows, cal = [], []
for tk in ["T1_atn", "T3_l2c"]:
    task = C.TASKS[tk]
    d = pd.DataFrame([json.loads(l) for l in open(
        os.path.join(C.RESULTS, "model_outputs", f"llm_{tk}_qwen7b.jsonl"))])
    d = d[(d.level == 0)].dropna(subset=["pred"])
    for cond, g in d.groupby("condition"):
        y = g["y"].values.astype(float); p = g["conf"].values.astype(float)
        h = g["pred"].values.astype(float)
        rows.append(dict(task=tk, condition=cond, n=len(g),
                         base_rate=float(y.mean()), pred_rate=float(h.mean()),
                         n_distinct_conf=int(len(set(np.round(p, 3)))),
                         conf_entropy=float(stats.entropy(
                             pd.Series(np.round(p, 2)).value_counts(normalize=True))),
                         mean_conf=float(p.mean()), acc=float((h == y).mean()),
                         ece=ece(y, p), overconfidence=float(p.mean() - (h == y).mean())))
        for lo in np.arange(0, 1, .2):
            m = (p >= lo) & (p < lo + .2)
            if m.sum() >= 5:
                cal.append(dict(task=tk, condition=cond, bin_lo=lo, n=int(m.sum()),
                                mean_conf=float(p[m].mean()), frac_pos=float(y[m].mean())))
    # chain length vs correctness (CoT only) -- attach to the CoT row for this task
    c = d[d.condition == "cot"]
    corr = (c["pred"].values == c["y"].values).astype(float)
    r, pv = stats.pointbiserialr(corr, c["n_chars"].values)
    for rr in rows:
        if rr["task"] == tk and rr["condition"] == "cot":
            rr["len_corr_r"] = float(r); rr["len_corr_p"] = float(pv)

R = pd.DataFrame(rows)
R.to_csv(os.path.join(C.RESULTS, "analysis_calibration.csv"), index=False)
pd.DataFrame(cal).to_csv(os.path.join(C.RESULTS, "analysis_calibration_bins.csv"), index=False)
print(R.round(3).to_string(index=False))

# figure: reliability + confidence-resolution
cb = pd.DataFrame(cal)
fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
for ax, tk in zip(axes, ["T1_atn", "T3_l2c"]):
    s = cb[cb.task == tk]
    for cond, g in s.groupby("condition"):
        ax.plot(g["mean_conf"], g["frac_pos"], "o-", color=CO.get(cond, "k"), label=cond, ms=4)
    ax.plot([0, 1], [0, 1], "k--", lw=.9)
    ax.set_xlabel("stated confidence P(positive)")
    ax.set_ylabel("observed positive rate")
    ax.set_title(f"{tk} — reliability of stated confidence", fontsize=8.5)
    ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{C.FIGURES}/fig6_calibration.png", bbox_inches="tight")
print("done")
