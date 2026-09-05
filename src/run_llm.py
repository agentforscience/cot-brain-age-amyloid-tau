"""
E2 (+ LLM half of E5, and E6).

Runs the LLM arm on all three tasks under three matched prompt conditions
(direct / cot / filler), with fold-local in-context exemplars, under the same
grouped CV splits and the same perturbation sweeps as the tabular arm.

Every raw generation is saved to results/model_outputs/ so that the faithfulness
battery (E3/E4) can operate on the exact chains that produced the scored answers.
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from llm import Engine

N_SHOTS = 8            # in-context exemplars, drawn ONLY from the training fold
MAXTOK = {"direct": 24, "cot": 512, "filler": 24}
FILLER_TOK = {}   # task.key -> number of '.' filler tokens, set to median CoT length


def pick_shots(df, task, tr_idx, seed):
    """Class-balanced (clf) / age-spread (reg) exemplars from the training fold,
    one row per group so no donor appears twice."""
    rng = np.random.RandomState(seed)
    tr = df.iloc[tr_idx]
    tr = tr.groupby(task.group_col, group_keys=False).apply(lambda g: g.iloc[[0]])
    if task.kind == "clf":
        per = N_SHOTS // 2
        sel = []
        for lab in (0, 1):
            pool = tr[tr[task.target] == lab]
            k = min(per, len(pool))
            sel.append(pool.iloc[rng.choice(len(pool), k, replace=False)])
        s = pd.concat(sel)
    else:
        q = tr.sort_values(task.target)
        idx = np.linspace(0, len(q) - 1, N_SHOTS).astype(int)
        s = q.iloc[idx]
    s = s.iloc[rng.permutation(len(s))]      # shuffle so label order carries no signal
    return [s.iloc[i] for i in range(len(s))], list(s[task.target].values)


def force_answer(eng, task, texts, convs, batch_size):
    """Any generation that did not emit a parseable ANSWER line is re-run as a
    continuation forced with a literal 'ANSWER:' prefill, rather than dropped.
    Dropping would silently bias the evaluated subset."""
    need = [i for i, t in enumerate(texts) if not C.parse_answer(t, task)[2]]
    if not need:
        return texts, np.zeros(len(texts), dtype=bool)
    pre = [texts[i].rstrip() + "\n\nANSWER:" for i in need]
    out = eng.chat([convs[i] for i in need], max_new_tokens=20, batch_size=batch_size,
                   prefill=pre, desc="force-answer", verbose=False)
    forced = np.zeros(len(texts), dtype=bool)
    for i, o in zip(need, out):
        texts[i] = texts[i].rstrip() + "\n\nANSWER:" + o
        forced[i] = True
    return texts, forced


def run(task: C.Task, eng: Engine, condition: str, perturb_kind="mcar", level=0.0,
        subset=None, batch_size=32):
    df = task.load()
    folds = C.grouped_folds(df, task)
    X = df[task.features].astype(float)
    Xp = C.perturb(X, task, perturb_kind, level, seed=C.SEED + int(level * 1000))
    dfp = df.copy()
    dfp[task.features] = Xp

    keep = np.arange(len(df)) if subset is None else np.asarray(subset)
    convs, meta = [], []
    for fi, (tr, te) in enumerate(folds):
        shots, labs = pick_shots(df, task, tr, seed=C.SEED + fi)
        for i in te:
            if i not in keep:
                continue
            convs.append(C.build_prompt(dfp.iloc[i], task, condition, shots, labs))
            meta.append(dict(row=int(i), fold=fi, group=str(df.iloc[i][task.group_col]),
                             y=float(df.iloc[i][task.target])))
    t0 = time.time()
    prefill = None
    if condition == "filler":
        prefill = ["." * FILLER_TOK.get(task.key, 150)] * len(convs)
    texts = eng.chat(convs, max_new_tokens=MAXTOK[condition], batch_size=batch_size,
                     prefill=prefill,
                     desc=f"{task.key}/{eng.key}/{condition}/{perturb_kind}{level}")
    texts, forced = force_answer(eng, task, texts, convs, batch_size)
    dt = time.time() - t0
    recs = []
    for m, tx, cv, fo in zip(meta, texts, convs, forced):
        val, conf, ok = C.parse_answer(tx, task)
        recs.append({**m, "condition": condition, "model": eng.key,
                     "perturb": perturb_kind, "level": level,
                     "pred": val, "conf": conf, "parsed_ok": bool(ok), "forced": bool(fo),
                     "n_chars": len(tx), "text": tx, "prompt": cv[1]["content"]})
    print(f"  -> {task.key} {eng.key} {condition} {perturb_kind}={level}: "
          f"{len(recs)} items in {dt:.0f}s, parse_ok={np.mean([r['parsed_ok'] for r in recs]):.2f}",
          flush=True)
    return recs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen7b")
    ap.add_argument("--tasks", default="T1_atn,T3_l2c,T2_brainage")
    ap.add_argument("--robust_n", type=int, default=150)
    ap.add_argument("--full_sweep", action="store_true")   # subset size for perturbation sweep
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    C.set_seed()
    os.makedirs(os.path.join(C.RESULTS, "model_outputs"), exist_ok=True)

    SWEEP = [("mcar", 0.2), ("mcar", 0.5), ("noise", 0.5), ("noise", 1.0)]
    SWEEP = SWEEP if a.full_sweep else [("mcar", 0.2), ("mcar", 0.5), ("noise", 1.0)]
    for mkey in a.models.split(","):
        eng = Engine(mkey)
        for tk in a.tasks.split(","):
            task = C.TASKS[tk]
            out = []
            cot_recs = run(task, eng, "cot", batch_size=a.batch)
            out += cot_recs
            # match the filler control's token budget to the median observed CoT length
            ntok = [len(eng.tok(r["text"])["input_ids"]) for r in cot_recs]
            FILLER_TOK[tk] = int(np.median(ntok))
            print(f"  filler control matched to {FILLER_TOK[tk]} tokens "
                  f"(median CoT length)", flush=True)
            for cond in ["direct", "filler"]:
                out += run(task, eng, cond, batch_size=a.batch)
            # robustness sweep on a fixed subset, cot + direct only
            df = task.load()
            rng = np.random.RandomState(C.SEED)
            sub = rng.choice(len(df), min(a.robust_n, len(df)), replace=False)
            for pk, lv in SWEEP:
                for cond in ["cot", "direct"]:
                    out += run(task, eng, cond, pk, lv, subset=sub, batch_size=a.batch)
            path = os.path.join(C.RESULTS, "model_outputs", f"llm_{tk}_{mkey}.jsonl")
            with open(path, "w") as f:
                for r in out:
                    f.write(json.dumps(r) + "\n")
            print(f"saved {path} ({len(out)} rows)", flush=True)
        eng.close()
    print("done")
