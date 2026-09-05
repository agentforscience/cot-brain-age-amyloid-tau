"""
E1 + tabular half of E5.

Black-box baselines (logistic regression / XGBoost / TabPFN v2 / trivial reference)
on all three tasks, under grouped CV, with the identical MCAR-missingness and
Gaussian-noise perturbation sweeps that the LLM arm will face.

Also produces permutation-importance rankings used later as the ground-truth
"signal" against which chain-of-thought citations are scored (ESA metric, E4).
"""
from __future__ import annotations
import os, sys, json, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor

TABPFN_CLF = os.path.join(C.ROOT, "models", "tabpfn", "tabpfn-v2-classifier.ckpt")
TABPFN_REG = os.path.join(C.ROOT, "models", "tabpfn", "tabpfn-v2-regressor.ckpt")


def make_model(name: str, kind: str):
    """Return a fresh estimator. Imputation is part of the pipeline so that the
    MCAR sweep is handled the same way by every arm."""
    if name == "logreg":
        est = LogisticRegression(max_iter=2000, C=1.0) if kind == "clf" else Ridge(alpha=1.0)
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler()), ("m", est)])
    if name == "xgboost":                      # XGBoost handles NaN natively
        if kind == "clf":
            return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                 eval_metric="logloss", tree_method="hist",
                                 random_state=C.SEED, n_jobs=8)
        return XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                            tree_method="hist", random_state=C.SEED, n_jobs=8)
    if name == "tabpfn":
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        dev = os.environ.get("TABPFN_DEVICE", "cuda")
        if kind == "clf":
            return TabPFNClassifier(model_path=TABPFN_CLF, device=dev, random_state=C.SEED)
        return TabPFNRegressor(model_path=TABPFN_REG, device=dev, random_state=C.SEED)
    raise ValueError(name)


def fit_predict(name, task, Xtr, ytr, Xte):
    """Returns (score, hard_pred) where score is P(y=1) for clf, value for reg."""
    if name == "trivial":
        if task.kind == "clf":
            p = float(np.mean(ytr))
            return np.full(len(Xte), p), np.full(len(Xte), int(p > 0.5))
        m = float(np.mean(ytr))
        return np.full(len(Xte), m), np.full(len(Xte), m)
    m = make_model(name, task.kind)
    m.fit(Xtr.values, ytr)
    if task.kind == "clf":
        s = m.predict_proba(Xte.values)[:, 1]
        return s, (s > 0.5).astype(int)
    s = m.predict(Xte.values)
    return s, s


def metrics(task, y, score, hard):
    if task.kind == "clf":
        return {"auc": float(roc_auc_score(y, score)) if len(set(y)) > 1 else np.nan,
                "bacc": float(balanced_accuracy_score(y, hard)),
                "acc": float(np.mean(hard == y))}
    return {"mae": float(mean_absolute_error(y, score)),
            "rmse": float(np.sqrt(np.mean((np.asarray(score) - np.asarray(y)) ** 2))),
            "r": float(np.corrcoef(score, y)[0, 1]) if np.std(score) > 0 else np.nan}


def run_task(task: C.Task, models, sweeps, oof_tag=""):
    df = task.load()
    folds = C.grouped_folds(df, task)
    y = df[task.target].values.astype(float)
    X = df[task.features].astype(float)
    rows = []
    for pkind, levels in sweeps.items():
        for lvl in levels:
            Xp = C.perturb(X, task, pkind, lvl, seed=C.SEED + int(lvl * 1000))
            for name in models:
                oof_s = np.full(len(df), np.nan); oof_h = np.full(len(df), np.nan)
                t0 = time.time()
                for tr, te in folds:
                    # training data is always CLEAN; only the test case is perturbed,
                    # which is the deployment scenario the hypothesis is about.
                    s, h = fit_predict(name, task, X.iloc[tr], y[tr], Xp.iloc[te])
                    oof_s[te] = s; oof_h[te] = h
                m = metrics(task, y, oof_s, oof_h)
                rows.append(dict(task=task.key, arm=name, family="tabular",
                                 perturb=pkind, level=lvl, secs=round(time.time() - t0, 1), **m))
                print(f"  {task.key:12s} {name:8s} {pkind}={lvl:<5} " +
                      " ".join(f"{k}={v:.4f}" for k, v in m.items()), flush=True)
                if lvl == 0.0 and pkind == "mcar":       # save clean OOF predictions
                    np.save(os.path.join(C.RESULTS,
                            f"oof_{task.key}{oof_tag}_{name}.npy"), oof_s)
    return pd.DataFrame(rows)


def permutation_importance(task: C.Task, n_rep=20):
    """Ground-truth feature signal, from XGBoost OOF, used for the ESA metric."""
    df = task.load(); folds = C.grouped_folds(df, task)
    y = df[task.target].values.astype(float); X = df[task.features].astype(float)
    base_s = np.full(len(df), np.nan)
    models = []
    for tr, te in folds:
        m = make_model("xgboost", task.kind); m.fit(X.iloc[tr].values, y[tr])
        models.append((m, te))
        base_s[te] = m.predict_proba(X.iloc[te].values)[:, 1] if task.kind == "clf" \
            else m.predict(X.iloc[te].values)
    base = metrics(task, y, base_s, (base_s > .5).astype(int) if task.kind == "clf" else base_s)
    key = "auc" if task.kind == "clf" else "mae"
    rng = np.random.RandomState(C.SEED); imp = {}
    for j, c in enumerate(task.features):
        drops = []
        for _ in range(n_rep):
            s = np.full(len(df), np.nan)
            for m, te in models:
                Xt = X.iloc[te].copy()
                Xt[c] = rng.permutation(Xt[c].values)
                s[te] = m.predict_proba(Xt.values)[:, 1] if task.kind == "clf" else m.predict(Xt.values)
            mm = metrics(task, y, s, (s > .5).astype(int) if task.kind == "clf" else s)
            drops.append(base[key] - mm[key] if task.kind == "clf" else mm[key] - base[key])
        imp[c] = float(np.mean(drops))
    return imp, base


if __name__ == "__main__":
    C.set_seed()
    os.makedirs(C.RESULTS, exist_ok=True)
    MODELS = ["trivial", "logreg", "xgboost", "tabpfn"]
    SWEEPS = {"mcar": [0.0, 0.1, 0.2, 0.3, 0.5],
              "noise": [0.25, 0.5, 1.0]}
    all_rows, imps = [], {}
    for k in ["T1_atn", "T3_l2c", "T2_brainage"]:
        t = C.TASKS[k]
        print(f"== {t.name}", flush=True)
        all_rows.append(run_task(t, MODELS, SWEEPS))
        imp, base = permutation_importance(t)
        imps[k] = {"importance": imp, "xgb_base": base}
        print("  perm-importance top:",
              sorted(imp.items(), key=lambda kv: -kv[1])[:5], flush=True)
    pd.concat(all_rows).to_csv(os.path.join(C.RESULTS, "tabular_results.csv"), index=False)
    json.dump(imps, open(os.path.join(C.RESULTS, "permutation_importance.json"), "w"), indent=2)

    # secondary target audit for T1 (see planning.md amendment A1)
    t = C.TASK1
    sec = C.Task(**{**t.__dict__, "target": "y_niareagan_AD"})
    print("== T1 secondary target audit (y_niareagan_AD)", flush=True)
    run_task(sec, ["trivial", "logreg", "xgboost", "tabpfn"], {"mcar": [0.0]},
             oof_tag="_secondaryNIAReagan") \
        .to_csv(os.path.join(C.RESULTS, "tabular_T1_secondary_target.csv"), index=False)
    print("done")
