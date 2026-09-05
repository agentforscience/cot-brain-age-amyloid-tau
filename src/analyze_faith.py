"""
Phase 5b: analysis of the faithfulness battery (E3/E4), the robustness sweeps (E5),
the model-size arm (E6) and the judge rubric (E7), plus all figures.
"""
from __future__ import annotations
import os, sys, json, glob, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, mean_absolute_error

B = 2000
FD = os.path.join(C.RESULTS, "faithfulness")
plt.rcParams.update({"figure.dpi": 140, "font.size": 8.5, "axes.grid": True,
                     "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False})
CO = {"cot": "#2b6cb0", "direct": "#dd6b20", "filler": "#718096",
      "xgboost": "#2f855a", "logreg": "#805ad5", "tabpfn": "#c53030", "trivial": "#a0aec0"}
TNAME = {"T1_atn": "T1 · amyloid/tau stage (Allen ACT)",
         "T3_l2c": "T3 · dementia progression (OASIS-2 L2C)",
         "T2_brainage": "T2 · brain age (OASIS controls)"}


# ---------------------------------------------------------------- E3.1 AOC
def early_curve(task, model="qwen7b"):
    p = f"{FD}/early_{task.key}_{model}.csv"
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p).dropna(subset=["pred", "full_pred"])
    if task.kind == "clf":
        d["agree"] = (d["pred"] == d["full_pred"]).astype(float)
    else:
        sd = d["full_pred"].std()
        d["agree"] = 1.0 - np.minimum(1.0, (d["pred"] - d["full_pred"]).abs() / (sd + 1e-9))
    g = d.groupby("frac")
    cur = g["agree"].mean()
    cis = {}
    for f, sub in g:
        bs = C.cluster_bootstrap(sub["group"].values,
                                 lambda i: sub["agree"].values[i].mean(), B)
        cis[f] = C.ci(bs)
    # AOC: area between the same-answer curve and the 100% reference (=1.0).
    # AOC = 0  -> the answer was already fixed before any reasoning was written.
    # AOC large -> the reasoning genuinely changes the answer as it accumulates.
    x = cur.index.values.astype(float); y = cur.values
    aoc = float(np.trapezoid(1.0 - y, x))
    bs_a = C.cluster_bootstrap(
        d["group"].values,
        lambda i: float(np.trapezoid(
            1.0 - d.iloc[i].groupby("frac")["agree"].mean().reindex(x).values, x)), B)
    return dict(curve=cur, cis=cis, aoc=aoc, aoc_ci=C.ci(bs_a),
                n=int(d.groupby("frac").size().max()))


# ---------------------------------------------------------------- E3.2/3.3
def change_rate(path, col, task):
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path).dropna(subset=[col])
    v = d[col].values.astype(float)
    if task.kind == "reg":                      # normalise the shift by prediction SD
        sd = d["full_pred"].std() + 1e-9
        v = v / sd
    bs = C.cluster_bootstrap(d["group"].values, lambda i: np.mean(v[i]), B)
    return dict(n=len(d), mean=float(np.mean(v)), ci=C.ci(bs))


# ---------------------------------------------------------------- E4 CFIT
def cfit_stat(task, model="qwen7b"):
    p = f"{FD}/cfit_{task.key}_{model}.csv"
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p).dropna(subset=["shift"])
    if task.kind == "reg":
        d["shift"] = d["shift"] / (d["base_pred"].std() + 1e-9)
    piv = d.pivot_table(index="row", columns="arm", values="shift", aggfunc="mean")
    piv = piv.dropna()
    grp = d.drop_duplicates("row").set_index("row")["group"].reindex(piv.index).values
    ci_, un = piv["cited"].values, piv["uncited"].values
    diff = float(np.mean(ci_) - np.mean(un))
    bs = C.cluster_bootstrap(grp, lambda i: np.mean(ci_[i]) - np.mean(un[i]), B)
    ratio = float((np.mean(ci_) + 1e-9) / (np.mean(un) + 1e-9))
    try:
        w = stats.wilcoxon(ci_, un).pvalue
    except Exception:
        w = np.nan
    return dict(n_pairs=len(piv), cited=float(np.mean(ci_)), uncited=float(np.mean(un)),
                diff=diff, ci=C.ci(bs), ratio=ratio, p_boot=C.boot_p(bs), wilcoxon_p=w)


# ---------------------------------------------------------------- E4 ESA
def esa_stat(task, model="qwen7b"):
    fp = f"{FD}/citation_rates_{model}.json"
    ip = os.path.join(C.RESULTS, "permutation_importance.json")
    if not (os.path.exists(fp) and os.path.exists(ip)):
        return None
    cit = json.load(open(fp)).get(task.key, {}).get("citation_rate", {})
    imp = json.load(open(ip))[task.key]["importance"]
    f = [k for k in task.features if k in cit and k in imp]
    a = np.array([cit[k] for k in f]); b = np.array([imp[k] for k in f])
    rho, p = stats.spearmanr(a, b)
    return dict(n_features=len(f), spearman_rho=float(rho), p=float(p),
                top_cited=sorted(cit.items(), key=lambda kv: -kv[1])[:5],
                top_signal=sorted(imp.items(), key=lambda kv: -kv[1])[:5])


# ---------------------------------------------------------------- E5 robustness
def robustness_table(model="qwen7b"):
    tab = pd.read_csv(os.path.join(C.RESULTS, "tabular_results.csv"))
    rows = []
    for tk in ["T1_atn", "T3_l2c", "T2_brainage"]:
        task = C.TASKS[tk]
        p = os.path.join(C.RESULTS, "model_outputs", f"llm_{tk}_{model}.jsonl")
        if not os.path.exists(p):
            continue
        d = pd.DataFrame([json.loads(l) for l in open(p)])
        # the sweep used a fixed subset; restrict the clean reference to the SAME rows
        sweep_rows = set(d[d["level"] > 0]["row"])
        for cond in ["cot", "direct"]:
            for (pk, lv), sub in d[d["condition"] == cond].groupby(["perturb", "level"]):
                if lv == 0.0:
                    sub = sub[sub["row"].isin(sweep_rows)]
                sub = sub.dropna(subset=["pred"])
                if len(sub) < 20:
                    continue
                y = sub["y"].values.astype(float)
                if task.kind == "clf":
                    m = dict(auc=roc_auc_score(y, sub["conf"].values) if len(set(y)) > 1 else np.nan,
                             bacc=balanced_accuracy_score(y, sub["pred"].values))
                else:
                    m = dict(mae=mean_absolute_error(y, sub["pred"].values))
                rows.append(dict(task=tk, family="llm", arm=cond, perturb=pk,
                                 level=lv, n=len(sub), **m))
        # tabular arms, restricted to the identical subset for a like-for-like curve
        for _, r in tab[(tab.task == tk)].iterrows():
            rows.append(dict(task=tk, family="tabular", arm=r["arm"], perturb=r["perturb"],
                             level=r["level"], n=int(r.get("n", 0) or 0),
                             **({"auc": r["auc"], "bacc": r["bacc"]} if task.kind == "clf"
                                else {"mae": r["mae"]})))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(C.RESULTS, "analysis_robustness.csv"), index=False)
    return d


# ---------------------------------------------------------------- figures
def fig_accuracy(acc):
    tasks = [t for t in ["T1_atn", "T3_l2c", "T2_brainage"] if t in set(acc.task)]
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.6 * len(tasks), 3.3))
    axes = np.atleast_1d(axes)
    for ax, tk in zip(axes, tasks):
        task = C.TASKS[tk]
        s = acc[acc.task == tk]
        key, lo, hi = ("auc", "auc_lo", "auc_hi") if task.kind == "clf" else ("mae", "mae_lo", "mae_hi")
        s = s.dropna(subset=[key]).sort_values(key, ascending=(task.kind == "reg"))
        xs = np.arange(len(s))
        cols = [CO.get(a, "#4a5568") for a in s["arm"]]
        ax.barh(xs, s[key], color=cols, alpha=.85,
                edgecolor=["black" if f == "llm" else "none" for f in s["family"]], linewidth=.8)
        ax.errorbar(s[key], xs, xerr=[s[key] - s[lo], s[hi] - s[key]], fmt="none",
                    ecolor="#1a202c", capsize=2.5, lw=1)
        ax.set_yticks(xs)
        ax.set_yticklabels([f"{a}{' (LLM)' if f=='llm' else ''}" for a, f in zip(s["arm"], s["family"])])
        ax.set_xlabel("ROC-AUC" if task.kind == "clf" else "MAE (years)")
        ax.set_title(TNAME[tk], fontsize=8.5)
        if task.kind == "clf":
            ax.axvline(.5, color="k", ls=":", lw=.9)
            ax.set_xlim(0.3, 1.0)
    fig.suptitle("Accuracy by arm — bars are LLM arms (black edge) vs tabular baselines;\n"
                 "error bars are 95% donor/subject-clustered bootstrap CIs", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{C.FIGURES}/fig1_accuracy.png", bbox_inches="tight")
    plt.close(fig)


def fig_early(model="qwen7b"):
    tasks = ["T1_atn", "T3_l2c", "T2_brainage"]
    res = {tk: early_curve(C.TASKS[tk], model) for tk in tasks}
    res = {k: v for k, v in res.items() if v}
    if not res:
        return {}
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for i, (tk, r) in enumerate(res.items()):
        x = r["curve"].index.values.astype(float); y = r["curve"].values
        lo = np.array([r["cis"][f][0] for f in x]); hi = np.array([r["cis"][f][1] for f in x])
        c = ["#2b6cb0", "#dd6b20", "#2f855a"][i]
        ax.plot(x * 100, y, "o-", color=c, label=f"{TNAME[tk].split(' · ')[0]}  AOC={r['aoc']:.3f}")
        ax.fill_between(x * 100, lo, hi, color=c, alpha=.15)
    ax.axhline(1.0, color="k", ls="--", lw=.9)
    ax.set_xlabel("% of the reasoning chain shown before forcing an answer")
    ax.set_ylabel("agreement with the full-chain answer")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("Early-answering test (Lanham et al. 2023)\n"
                 "flat near 1.0 = the answer was fixed before the reasoning was written", fontsize=8.5)
    fig.tight_layout(); fig.savefig(f"{C.FIGURES}/fig2_early_answering.png", bbox_inches="tight")
    plt.close(fig)
    return res


def fig_robustness(rob):
    tasks = [t for t in ["T1_atn", "T3_l2c", "T2_brainage"] if t in set(rob.task)]
    fig, axes = plt.subplots(2, len(tasks), figsize=(3.5 * len(tasks), 5.4), squeeze=False)
    for j, tk in enumerate(tasks):
        task = C.TASKS[tk]
        key = "auc" if task.kind == "clf" else "mae"
        for i, pk in enumerate(["mcar", "noise"]):
            ax = axes[i][j]
            s = rob[(rob.task == tk) & (rob.perturb == pk)]
            base = rob[(rob.task == tk) & (rob.perturb == "mcar") & (rob.level == 0)]
            for arm, g in s.groupby("arm"):
                if arm == "trivial":
                    continue
                b0 = base[base.arm == arm]
                g = pd.concat([b0.assign(level=0.0), g[g.level > 0]]).sort_values("level")
                ax.plot(g["level"], g[key], "o-", ms=3.5, lw=1.4,
                        color=CO.get(arm, "#4a5568"), label=arm,
                        ls="-" if arm in ("cot", "direct") else "--")
            ax.set_xlabel("MCAR missingness fraction" if pk == "mcar"
                          else "Gaussian noise (SD units)")
            ax.set_ylabel("ROC-AUC" if task.kind == "clf" else "MAE (y)")
            if i == 0:
                ax.set_title(TNAME[tk], fontsize=8.5)
            if task.kind == "clf":
                ax.axhline(.5, color="k", ls=":", lw=.8)
            if i == 0 and j == 0:
                ax.legend(fontsize=6.5, ncol=2)
    fig.suptitle("Robustness sweeps — solid = LLM arms, dashed = tabular baselines", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{C.FIGURES}/fig3_robustness.png", bbox_inches="tight")
    plt.close(fig)


def fig_cfit_judge(cf, jd):
    n = 1 + (jd is not None)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3))
    ax = axes[0]
    ks = [k for k in cf if cf[k]]
    x = np.arange(len(ks)); w = .36
    for o, (arm, col) in enumerate([("cited", "#2b6cb0"), ("uncited", "#a0aec0")]):
        v = [cf[k][arm] for k in ks]
        ax.bar(x + (o - .5) * w, v, w, color=col, label=f"{arm} features perturbed")
    ax.set_xticks(x); ax.set_xticklabels([TNAME[k].split(" · ")[0] for k in ks])
    ax.set_ylabel("mean answer shift after ±1 SD perturbation")
    ax.legend(fontsize=7)
    ax.set_title("CFIT — does perturbing what the chain SAYS it used\nmove the answer more than perturbing what it ignored?", fontsize=8)
    ax = axes[1]
    if jd is not None and len(jd):
        piv = jd.groupby(["task", "arm"])["MEAN"].mean().unstack()
        ks2 = [k for k in ["T1_atn", "T3_l2c", "T2_brainage"] if k in piv.index]
        x = np.arange(len(ks2))
        for o, (arm, col) in enumerate([("genuine", "#2f855a"), ("shuffled", "#c53030")]):
            if arm in piv.columns:
                ax.bar(x + (o - .5) * w, piv.loc[ks2, arm], w, color=col,
                       label=f"{arm} explanation")
        ax.set_xticks(x); ax.set_xticklabels([TNAME[k].split(" · ")[0] for k in ks2])
        ax.set_ylabel("mean rubric score (0-4)"); ax.set_ylim(0, 4); ax.legend(fontsize=7)
        ax.set_title("LLM-judged plausibility rubric + shuffled-explanation control\n"
                     "(proxy for the clinician panel, NOT a clinician rating)", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{C.FIGURES}/fig4_cfit_judge.png", bbox_inches="tight")
    plt.close(fig)


def fig_esa(model="qwen7b"):
    fp = f"{FD}/citation_rates_{model}.json"
    ip = os.path.join(C.RESULTS, "permutation_importance.json")
    if not (os.path.exists(fp) and os.path.exists(ip)):
        return
    cit_all = json.load(open(fp)); imp_all = json.load(open(ip))
    ks = [k for k in ["T1_atn", "T3_l2c", "T2_brainage"] if k in cit_all]
    fig, axes = plt.subplots(1, len(ks), figsize=(3.5 * len(ks), 3.2), squeeze=False)
    for ax, tk in zip(axes[0], ks):
        cit = cit_all[tk]["citation_rate"]; imp = imp_all[tk]["importance"]
        f = [k for k in C.TASKS[tk].features if k in cit and k in imp]
        a = np.array([cit[k] for k in f]); b = np.array([imp[k] for k in f])
        rho, p = stats.spearmanr(a, b)
        ax.scatter(b, a, s=18, color="#2b6cb0")
        for k, xx, yy in zip(f, b, a):
            if yy > np.percentile(a, 80) or xx > np.percentile(b, 85):
                ax.annotate(k, (xx, yy), fontsize=5.5, alpha=.8)
        ax.set_xlabel("permutation importance (black-box arm)")
        ax.set_ylabel("citation rate in CoT chains")
        ax.set_title(f"{TNAME[tk].split(' · ')[0]}  ESA ρ={rho:.2f}, p={p:.3f}", fontsize=8.5)
    fig.suptitle("Explanation–Signal Alignment: do the chains cite the features that carry signal?",
                 fontsize=9)
    fig.tight_layout(); fig.savefig(f"{C.FIGURES}/fig5_esa.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen7b"
    os.makedirs(C.FIGURES, exist_ok=True)
    out = {}
    acc = pd.read_csv(os.path.join(C.RESULTS, "analysis_accuracy.csv"))
    fig_accuracy(acc)

    early = fig_early(model)
    out["early_answering"] = {k: dict(aoc=v["aoc"], aoc_ci=v["aoc_ci"], n=v["n"],
                                      curve={float(a): float(b) for a, b in v["curve"].items()})
                              for k, v in early.items()}
    out["mistakes"] = {tk: change_rate(f"{FD}/mistakes_{tk}_{model}.csv", "changed", C.TASKS[tk])
                       for tk in ["T1_atn", "T3_l2c", "T2_brainage"]}
    out["paraphrase"] = {tk: change_rate(f"{FD}/paraphrase_{tk}_{model}.csv", "changed", C.TASKS[tk])
                         for tk in ["T1_atn", "T3_l2c", "T2_brainage"]}
    cf = {tk: cfit_stat(C.TASKS[tk], model) for tk in ["T1_atn", "T3_l2c", "T2_brainage"]}
    out["cfit"] = cf
    out["esa"] = {tk: esa_stat(C.TASKS[tk], model) for tk in ["T1_atn", "T3_l2c", "T2_brainage"]}

    rob = robustness_table(model)
    if len(rob):
        fig_robustness(rob)
    # per-task judge runs write judge_scores_<model>__<tasks>.csv; merge whatever exists
    jfs = sorted(glob.glob(os.path.join(C.RESULTS, f"judge_scores_{model}__*.csv")))
    if jfs:
        jd = pd.concat([pd.read_csv(f) for f in jfs], ignore_index=True)
        jd.to_csv(os.path.join(C.RESULTS, f"judge_scores_{model}.csv"), index=False)
    else:
        jp = os.path.join(C.RESULTS, f"judge_scores_{model}.csv")
        jd = pd.read_csv(jp) if os.path.exists(jp) else None
    if jd is not None:
        DIMS = ["GROUNDING", "BIOLOGY", "COHERENCE", "SPECIFICITY", "NO_FABRICATION", "CLINICAL_USE"]
        js = {}
        for tk, g in jd.groupby("task"):
            a = g[g.arm == "genuine"]["MEAN"].dropna().values
            b = g[g.arm == "shuffled"]["MEAN"].dropna().values
            t = stats.mannwhitneyu(a, b) if len(a) and len(b) else None
            js[tk] = dict(genuine=float(np.mean(a)), shuffled=float(np.mean(b)),
                          diff=float(np.mean(a) - np.mean(b)),
                          mwu_p=float(t.pvalue) if t else np.nan,
                          dims={d: float(g[g.arm == "genuine"][d].mean()) for d in DIMS})
        out["judge"] = js
    fig_cfit_judge({k: v for k, v in cf.items() if v}, jd)
    fig_esa(model)
    json.dump(out, open(os.path.join(C.RESULTS, f"analysis_faithfulness_{model}.json"), "w"),
              indent=2, default=float)
    print(json.dumps(out, indent=2, default=float)[:6000])
    print("done")
