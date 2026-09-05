"""
Phase 5: statistical analysis of every arm, plus all figures.

Reads results/{tabular_results.csv, model_outputs/*.jsonl, faithfulness/*, judge_scores_*}
and writes results/analysis_*.csv + figures/*.png.

All confidence intervals are DONOR/SUBJECT-CLUSTERED bootstraps (B=2000): Allen ACT has
four regions per donor and OASIS has repeat visits per subject, so a row-level bootstrap
would badly understate uncertainty.
"""
from __future__ import annotations
import os, sys, json, glob, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from scipy import stats

B = 2000
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False})
PAL = {"cot": "#2b6cb0", "direct": "#dd6b20", "filler": "#718096",
       "xgboost": "#2f855a", "logreg": "#805ad5", "tabpfn": "#c53030", "trivial": "#a0aec0"}


# ------------------------------------------------------------------ loading
def load_llm(task_key, model="qwen7b"):
    p = os.path.join(C.RESULTS, "model_outputs", f"llm_{task_key}_{model}.jsonl")
    if not os.path.exists(p):
        return pd.DataFrame()
    d = pd.DataFrame([json.loads(l) for l in open(p)])
    return d


def score_llm(task, d):
    """Per-arm metric with clustered-bootstrap CI."""
    d = d.dropna(subset=["pred"])
    y = d["y"].values.astype(float)
    if task.kind == "clf":
        s = d["conf"].values.astype(float)
        h = d["pred"].values.astype(float)

        def f(idx):
            yy = y[idx]
            return roc_auc_score(yy, s[idx]) if len(set(yy)) > 1 else np.nan
        auc = roc_auc_score(y, s) if len(set(y)) > 1 else np.nan
        bs = C.cluster_bootstrap(d["group"].values, f, B)
        bacc = balanced_accuracy_score(y, h)

        def f2(idx):
            yy = y[idx]
            return balanced_accuracy_score(yy, h[idx]) if len(set(yy)) > 1 else np.nan
        bsb = C.cluster_bootstrap(d["group"].values, f2, B)
        lo, hi = C.ci(bs); lo2, hi2 = C.ci(bsb)
        return dict(n=len(d), auc=auc, auc_lo=lo, auc_hi=hi,
                    bacc=bacc, bacc_lo=lo2, bacc_hi=hi2)
    p = d["pred"].values.astype(float)
    err = np.abs(p - y)
    bs = C.cluster_bootstrap(d["group"].values, lambda i: np.mean(err[i]), B)
    lo, hi = C.ci(bs)
    return dict(n=len(d), mae=float(np.mean(err)), mae_lo=lo, mae_hi=hi,
                rmse=float(np.sqrt(np.mean((p - y) ** 2))),
                r=float(np.corrcoef(p, y)[0, 1]) if np.std(p) > 0 else np.nan)


def paired_contrast(task, d, arm_a, arm_b, key="condition"):
    """Paired clustered bootstrap of arm_a - arm_b on the SAME rows."""
    A = d[d[key] == arm_a].set_index("row")
    Bd = d[d[key] == arm_b].set_index("row")
    common = sorted(set(A.index) & set(Bd.index))
    A, Bd = A.loc[common], Bd.loc[common]
    ok = A["pred"].notna() & Bd["pred"].notna()
    A, Bd = A[ok], Bd[ok]
    y = A["y"].values.astype(float); g = A["group"].values
    if task.kind == "clf":
        sa, sb = A["conf"].values.astype(float), Bd["conf"].values.astype(float)
        ha, hb = A["pred"].values.astype(float), Bd["pred"].values.astype(float)

        def f(idx):
            yy = y[idx]
            if len(set(yy)) < 2:
                return np.nan
            return roc_auc_score(yy, sa[idx]) - roc_auc_score(yy, sb[idx])
        diff = roc_auc_score(y, sa) - roc_auc_score(y, sb)
        bs = C.cluster_bootstrap(g, f, B)
        # McNemar on paired correctness
        ca, cb = (ha == y), (hb == y)
        n01, n10 = int((~ca & cb).sum()), int((ca & ~cb).sum())
        mc = stats.binomtest(n10, n01 + n10, 0.5).pvalue if (n01 + n10) > 0 else np.nan
        lo, hi = C.ci(bs)
        return dict(metric="dAUC", n_pairs=len(A), diff=diff, lo=lo, hi=hi,
                    p_boot=C.boot_p(bs), mcnemar_p=mc, n01=n01, n10=n10)
    ea = np.abs(A["pred"].values.astype(float) - y)
    eb = np.abs(Bd["pred"].values.astype(float) - y)
    bs = C.cluster_bootstrap(g, lambda i: np.mean(ea[i]) - np.mean(eb[i]), B)
    w = stats.wilcoxon(ea, eb).pvalue if len(ea) > 10 else np.nan
    lo, hi = C.ci(bs)
    return dict(metric="dMAE", n_pairs=len(A), diff=float(np.mean(ea) - np.mean(eb)),
                lo=lo, hi=hi, p_boot=C.boot_p(bs), wilcoxon_p=w,
                rank_biserial=float(np.mean(ea < eb) - np.mean(ea > eb)))


# ------------------------------------------------------------------ main
def main(model="qwen7b"):
    os.makedirs(C.FIGURES, exist_ok=True)
    tab = pd.read_csv(os.path.join(C.RESULTS, "tabular_results.csv"))
    acc_rows, con_rows = [], []

    for tk in ["T1_atn", "T3_l2c", "T2_brainage"]:
        task = C.TASKS[tk]
        d = load_llm(tk, model)
        if d.empty:
            continue
        clean = d[(d["level"] == 0.0)]
        for cond in ["cot", "direct", "filler"]:
            sub = clean[clean["condition"] == cond]
            if len(sub) == 0:
                continue
            m = score_llm(task, sub)
            acc_rows.append(dict(task=tk, family="llm", arm=cond, model=model,
                                 parse_ok=float(sub["parsed_ok"].mean()),
                                 forced=float(sub["forced"].mean()) if "forced" in sub else 0.0,
                                 median_chars=float(sub["n_chars"].median()), **m))
        # tabular arms with clustered CIs from the saved OOF predictions
        df = task.load(); g = df[task.group_col].astype(str).values
        y = df[task.target].values.astype(float)
        for arm in ["trivial", "logreg", "xgboost", "tabpfn"]:
            f = os.path.join(C.RESULTS, f"oof_{tk}_{arm}.npy")
            if not os.path.exists(f):
                continue
            s = np.load(f)
            if task.kind == "clf":
                bs = C.cluster_bootstrap(g, lambda i: roc_auc_score(y[i], s[i])
                                         if len(set(y[i])) > 1 else np.nan, B)
                bsb = C.cluster_bootstrap(g, lambda i: balanced_accuracy_score(
                    y[i], (s[i] > .5).astype(int)) if len(set(y[i])) > 1 else np.nan, B)
                lo, hi = C.ci(bs); lo2, hi2 = C.ci(bsb)
                acc_rows.append(dict(task=tk, family="tabular", arm=arm, model="-", n=len(y),
                                     auc=roc_auc_score(y, s), auc_lo=lo, auc_hi=hi,
                                     bacc=balanced_accuracy_score(y, (s > .5).astype(int)),
                                     bacc_lo=lo2, bacc_hi=hi2))
            else:
                e = np.abs(s - y)
                bs = C.cluster_bootstrap(g, lambda i: np.mean(e[i]), B)
                lo, hi = C.ci(bs)
                acc_rows.append(dict(task=tk, family="tabular", arm=arm, model="-", n=len(y),
                                     mae=float(np.mean(e)), mae_lo=lo, mae_hi=hi,
                                     rmse=float(np.sqrt(np.mean((s - y) ** 2))),
                                     r=float(np.corrcoef(s, y)[0, 1])))
        # LLM condition contrasts (H1b)
        for a_, b_ in [("cot", "direct"), ("cot", "filler"), ("direct", "filler")]:
            try:
                con_rows.append(dict(task=tk, model=model, contrast=f"{a_}-{b_}",
                                     **paired_contrast(task, clean, a_, b_)))
            except Exception as e:
                print("contrast failed", tk, a_, b_, e)

    acc = pd.DataFrame(acc_rows)
    con = pd.DataFrame(con_rows)
    if len(con):
        con["q_bh"] = C.bh_fdr(con["p_boot"].values)
    acc.to_csv(os.path.join(C.RESULTS, "analysis_accuracy.csv"), index=False)
    con.to_csv(os.path.join(C.RESULTS, "analysis_contrasts.csv"), index=False)
    print(acc.round(3).to_string(index=False)); print(); print(con.round(4).to_string(index=False))
    return acc, con, tab


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "qwen7b")
