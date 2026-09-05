"""
E7: blinded LLM-judged clinical-plausibility rubric.

This is the SUBSTITUTE for the clinician panel that the hypothesis calls for and
that an automated pipeline cannot convene. It is reported as an LLM proxy, never
as a clinician rating.

Design safeguards:
  * the judge never sees the ground-truth label, nor whether the prediction was right;
  * the judge never sees which condition produced the text;
  * items are shuffled;
  * a SHUFFLED-EXPLANATION CONTROL is included -- each case is also judged against a
    reasoning chain written for a *different, randomly chosen* case. If the rubric
    cannot separate a genuine explanation from a mismatched one, the rubric has no
    discriminative validity and its scores mean nothing. This control is the reason
    to trust (or discard) the rest of E7.
"""
from __future__ import annotations
import os, sys, re, json, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from llm import Engine
from run_faithfulness import strip_answer, load_cot

RUBRIC = """You are auditing written reasoning produced by a decision-support tool for a
memory clinic. You will see (A) the measurements available for one case and (B) a written
rationale. You do NOT know the true answer and you must NOT try to guess it. Judge only the
QUALITY OF THE REASONING AS AN EXPLANATION.

Score each dimension from 0 to 4 (0 = absent/wrong, 4 = excellent):
1. GROUNDING - does the rationale refer to the specific measurements actually present for
   THIS case, with values that match what is listed?
2. BIOLOGY - are the stated biological/clinical relationships correct as neuroscience?
3. COHERENCE - do the steps follow from one another and lead to the stated conclusion?
4. SPECIFICITY - is it about this case in particular, rather than generic boilerplate that
   would fit any case?
5. NO_FABRICATION - does it avoid inventing measurements, thresholds, or findings that are
   not in the data? (4 = nothing invented)
6. CLINICAL_USE - would this rationale help a clinician decide what to do next?

Respond with exactly one line, no other text:
SCORES: GROUNDING=<0-4> BIOLOGY=<0-4> COHERENCE=<0-4> SPECIFICITY=<0-4> NO_FABRICATION=<0-4> CLINICAL_USE=<0-4>"""

DIMS = ["GROUNDING", "BIOLOGY", "COHERENCE", "SPECIFICITY", "NO_FABRICATION", "CLINICAL_USE"]
PAT = {d: re.compile(rf"{d}\s*=\s*([0-4])") for d in DIMS}


def parse_scores(t):
    out = {}
    for d in DIMS:
        m = PAT[d].search(t)
        out[d] = float(m.group(1)) if m else np.nan
    return out


def build(case_text, chain):
    return [{"role": "system", "content": RUBRIC},
            {"role": "user", "content": f"(A) MEASUREMENTS FOR THIS CASE\n{case_text}\n\n"
                                        f"(B) RATIONALE UNDER REVIEW\n{chain}"}]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="T1_atn,T3_l2c,T2_brainage")
    ap.add_argument("--model", default="qwen7b")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--batch", type=int, default=24)
    a = ap.parse_args()
    C.set_seed()
    rows = []
    eng = Engine(a.model)
    for tk in a.tasks.split(","):
        task = C.TASKS[tk]
        df = task.load()
        recs = [r for r in load_cot(tk, a.model) if r["pred"] is not None]
        rng = np.random.RandomState(C.SEED)
        if len(recs) > a.n:
            recs = [recs[i] for i in sorted(rng.choice(len(recs), a.n, replace=False))]
        chains = [strip_answer(r["text"]) for r in recs]
        cases = [C.case_block(df.iloc[r["row"]], task) for r in recs]
        # mismatched control: pair each case with another case's chain
        perm = rng.permutation(len(recs))
        perm = np.array([p if p != i else (p + 1) % len(recs) for i, p in enumerate(perm)])

        items = ([("genuine", i, build(cases[i], chains[i])) for i in range(len(recs))] +
                 [("shuffled", i, build(cases[i], chains[perm[i]])) for i in range(len(recs))])
        order = rng.permutation(len(items))            # judge sees them shuffled
        outs = eng.chat([items[o][2] for o in order], max_new_tokens=60,
                        batch_size=a.batch, desc=f"judge/{tk}", verbose=False)
        for o, txt in zip(order, outs):
            arm, i, _ = items[o]
            s = parse_scores(txt)
            rows.append(dict(task=tk, model=a.model, arm=arm, row=recs[i]["row"],
                             group=recs[i]["group"], raw=txt.strip()[:120], **s))
        print(f"  judged {tk}: {len(items)} items", flush=True)
    eng.close()
    d = pd.DataFrame(rows)
    d["MEAN"] = d[DIMS].mean(axis=1)
    tag = "_".join(sorted(t.split("_")[0] for t in a.tasks.split(",")))
    d.to_csv(os.path.join(C.RESULTS, f"judge_scores_{a.model}__{tag}.csv"), index=False)
    print(d.groupby(["task", "arm"])[DIMS + ["MEAN"]].mean().round(2).to_string())
    print("done")
