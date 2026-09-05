"""
D2 clinical endpoint: the age-bias-corrected brain-age gap (BAG).

A brain-age model is only clinically interesting if the *gap* between predicted and
chronological age tracks disease. This script asks whether a BAG derived from a
chain-of-thought LLM carries the same clinical signal as a BAG derived from a
conventional normative model.

Protocol (standard in the brain-age literature, e.g. Cole/Beheshti):
  1. the normative model is fitted on cognitively normal scans ONLY;
  2. predictions on held-out controls come from subject-grouped CV;
  3. the well-known regression-to-the-mean bias is removed by regressing predicted
     age on chronological age *within controls* and subtracting the fitted line;
  4. the corrected BAG is then evaluated on the non-control scans for its ability to
     separate demented from non-demented (ROC-AUC).
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from run_llm import pick_shots, force_answer
from llm import Engine
from sklearn.metrics import roc_auc_score
from scipy import stats


def load_all_oasis():
    d = pd.read_csv(os.path.join(C.DATA, "oasis_brainage.csv"))
    d = d[d[C.TASK2.features].notna().all(axis=1)].reset_index(drop=True)
    return d


def llm_predict_noncontrols(eng, condition, batch=24):
    """Predict age for the NON-control scans using exemplars drawn from controls,
    exactly as the control-side evaluation did."""
    task = C.TASK2
    ctl = task.load()                       # controls only
    folds = C.grouped_folds(ctl, task)
    shots, labs = pick_shots(ctl, task, folds[0][0], seed=C.SEED)   # a fixed control ICL pool
    alld = load_all_oasis()
    nc = alld[alld["is_control"] != 1].reset_index(drop=True)
    convs = [C.build_prompt(nc.iloc[i], task, condition, shots, labs) for i in range(len(nc))]
    from run_llm import MAXTOK, FILLER_TOK
    txt = eng.chat(convs, max_new_tokens=MAXTOK[condition], batch_size=batch,
                   desc=f"bag/{condition}", verbose=False)
    txt, forced = force_answer(eng, task, txt, convs, batch)
    preds = [C.parse_answer(t, task)[0] for t in txt]
    nc = nc.copy(); nc["pred_age"] = preds; nc["condition"] = condition
    return nc


def bias_correct(pred_ctl, age_ctl, pred, age):
    """Fit pred = a*age + b on CONTROLS, then BAG = pred - (a*age + b)."""
    ok = np.isfinite(pred_ctl) & np.isfinite(age_ctl)
    a, b = np.polyfit(age_ctl[ok], pred_ctl[ok], 1)
    return pred - (a * age + b), (float(a), float(b))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen7b"); ap.add_argument("--batch", type=int, default=24)
    a = ap.parse_args()
    C.set_seed()
    task = C.TASK2
    ctl = task.load()
    out = {}
    rows = []

    # ---- tabular normative models (CV on controls; fit-on-all-controls for non-controls)
    from run_tabular import make_model
    alld = load_all_oasis()
    nc = alld[alld["is_control"] != 1].reset_index(drop=True)
    for arm in ["logreg", "xgboost", "tabpfn"]:
        oof = np.load(os.path.join(C.RESULTS, f"oof_{task.key}_{arm}.npy"))
        m = make_model(arm, "reg")
        m.fit(ctl[task.features].astype(float).values, ctl[task.target].values)
        p_nc = m.predict(nc[task.features].astype(float).values)
        bag, coef = bias_correct(oof, ctl[task.target].values, p_nc, nc["Age"].values)
        out[arm] = dict(family="tabular", coef=coef)
        nc[f"bag_{arm}"] = bag

    # ---- LLM arms
    eng = Engine(a.model)
    llm_ctl = {}
    p = os.path.join(C.RESULTS, "model_outputs", f"llm_{task.key}_{a.model}.jsonl")
    d = pd.DataFrame([json.loads(l) for l in open(p)])
    d = d[d["level"] == 0.0]
    for cond in ["cot", "direct"]:
        sub = d[d["condition"] == cond].dropna(subset=["pred"]).set_index("row")
        pc = np.full(len(ctl), np.nan)
        pc[sub.index.values] = sub["pred"].values.astype(float)
        ncp = llm_predict_noncontrols(eng, cond, a.batch)
        pv = np.array([np.nan if v is None else float(v) for v in ncp["pred_age"]])
        bag, coef = bias_correct(pc, ctl[task.target].values, pv, ncp["Age"].values)
        nc[f"bag_{cond}"] = bag
        nc[f"pred_{cond}"] = pv
        out[cond] = dict(family="llm", coef=coef,
                         mae_noncontrol=float(np.nanmean(np.abs(pv - ncp["Age"].values))))
    eng.close()

    # ---- clinical endpoint: does the brain-age gap separate demented scans from
    #      cognitively normal ones?
    #      NOTE: among the non-control scans, 283/285 are demented, so a
    #      demented-vs-non-demented contrast *within* the non-controls is degenerate.
    #      The standard field contrast is therefore used instead: cognitively normal
    #      controls (out-of-fold predictions, so genuinely held out) versus demented
    #      scans. The age-bias correction is fitted on the control OOF predictions,
    #      which is standard practice but does mean the control group is the group the
    #      correction was calibrated on -- stated explicitly as a limitation.
    ctl_eval = ctl.copy()
    dem = nc[nc["demented"] == 1].copy()
    res = []
    for arm in ["logreg", "xgboost", "tabpfn", "cot", "direct"]:
        col = f"bag_{arm}"
        if col not in dem:
            continue
        if arm in ("cot", "direct"):
            sub = d[d["condition"] == arm].dropna(subset=["pred"]).set_index("row")
            pc = np.full(len(ctl), np.nan); pc[sub.index.values] = sub["pred"].values.astype(float)
        else:
            pc = np.load(os.path.join(C.RESULTS, f"oof_{task.key}_{arm}.npy"))
        a, b = out[arm]["coef"]
        bag_ctl = pc - (a * ctl[task.target].values + b)
        e = dem.dropna(subset=[col])
        v = np.concatenate([bag_ctl, e[col].values.astype(float)])
        y = np.concatenate([np.zeros(len(bag_ctl)), np.ones(len(e))])
        g = np.concatenate([ctl["Subject_ID"].astype(str).values,
                            e["Subject_ID"].astype(str).values])
        ok = np.isfinite(v)
        v, y, g = v[ok], y[ok], g[ok]
        auc = roc_auc_score(y, v)
        bs = C.cluster_bootstrap(g, lambda i: roc_auc_score(y[i], v[i])
                                 if len(set(y[i])) > 1 else np.nan)
        lo, hi = C.ci(bs)
        t = stats.mannwhitneyu(v[y == 1], v[y == 0])
        d_ = (v[y == 1].mean() - v[y == 0].mean()) / np.sqrt(
            (v[y == 1].var(ddof=1) + v[y == 0].var(ddof=1)) / 2)
        res.append(dict(arm=arm, family="llm" if arm in ("cot", "direct") else "tabular",
                        n_ctl=int((y == 0).sum()), n_dem=int((y == 1).sum()),
                        bag_auc=auc, auc_lo=lo, auc_hi=hi,
                        bag_dem=float(v[y == 1].mean()), bag_ctl=float(v[y == 0].mean()),
                        cohens_d=float(d_), mwu_p=float(t.pvalue)))
    R = pd.DataFrame(res)
    R.to_csv(os.path.join(C.RESULTS, "analysis_brainage_gap.csv"), index=False)
    nc.to_csv(os.path.join(C.RESULTS, "brainage_noncontrol_predictions.csv"), index=False)
    print(R.round(4).to_string(index=False))
    print("done")
