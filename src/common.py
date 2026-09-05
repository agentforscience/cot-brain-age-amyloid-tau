"""
Shared utilities: data loading, task definitions, splits, prompt construction,
LLM engine, and clustered-bootstrap statistics.

All three tasks (T1 ATN status, T2 brain age, T3 L2C progression) are described
by a `Task` object so that the tabular arm and the LLM arm see *identical*
features, splits, and perturbations.
"""
from __future__ import annotations
import os, re, json, math, random, hashlib
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
import pandas as pd

SEED = 42
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "datasets", "derived")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")


def set_seed(seed: int = SEED):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# ---------------------------------------------------------------- task specs
@dataclass
class Task:
    key: str
    name: str
    csv: str
    group_col: str                 # cluster unit -> never split across folds
    target: str
    kind: str                      # "clf" | "reg"
    features: list[str]            # numeric features given to *every* arm
    pretty: dict[str, str]         # feature -> clinician-readable label + unit
    context_cols: list[str] = field(default_factory=list)   # shown in prompt, not modelled
    row_filter: Callable | None = None
    positive_label: str = "AD"
    negative_label: str = "not AD"
    question: str = ""

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(os.path.join(DATA, self.csv))
        if self.row_filter is not None:
            df = df[self.row_filter(df)].copy()
        df = df.dropna(subset=[self.target]).reset_index(drop=True)
        return df


# --- T1: amyloid/tau (ATN) neuropathological status, Allen ACT ---------------
T1_FEATS = [
    "ab42_pg_per_mg", "ab40_pg_per_mg", "ab42_over_ab40_ratio",
    "tau_ng_per_mg", "ptau_ng_per_mg", "ptau_over_tau_ratio",
    "ihc_a_beta", "ihc_at8", "ihc_tau2_ffpe", "ihc_a_syn",
    "ihc_gfap_ffpe", "ihc_iba1_ffpe",
    "a_syn_pg_per_mg", "age_years", "sex_M", "apoe4", "education_years",
]
T1_PRETTY = {
    "ab42_pg_per_mg":      "Abeta-42 (pg/mg tissue)",
    "ab40_pg_per_mg":      "Abeta-40 (pg/mg tissue)",
    "ab42_over_ab40_ratio":"Abeta-42/40 ratio",
    "tau_ng_per_mg":       "total tau (ng/mg tissue)",
    "ptau_ng_per_mg":      "phospho-tau (ng/mg tissue)",
    "ptau_over_tau_ratio": "pTau/total-tau ratio",
    "ihc_a_beta":          "Abeta immunoreactivity, fraction of area",
    "ihc_at8":             "AT8 pTau immunoreactivity, fraction of area",
    "ihc_tau2_ffpe":       "Tau-2 immunoreactivity (FFPE), fraction of area",
    "ihc_a_syn":           "alpha-synuclein immunoreactivity, fraction of area",
    "ihc_gfap_ffpe":       "GFAP astrogliosis, fraction of area",
    "ihc_iba1_ffpe":       "IBA1 microgliosis, fraction of area",
    "a_syn_pg_per_mg":     "alpha-synuclein (pg/mg tissue)",
    "age_years":           "age at death (years)",
    "sex_M":               "male sex (1=male, 0=female)",
    "apoe4":               "APOE e4 allele carrier (1=yes, 0=no)",
    "education_years":     "years of education",
}
TASK1 = Task(
    key="T1_atn", name="Amyloid/tau neuropathological status (Allen ACT)",
    csv="allen_act_atn.csv", group_col="donor_id", target="y_braak_ge4", kind="clf",
    features=T1_FEATS, pretty=T1_PRETTY, context_cols=["structure_acronym"],
    positive_label="advanced tau", negative_label="early tau",
    question=("Braak staging describes how far neurofibrillary tau tangle pathology has spread. "
              "From the tissue measurements below, does this donor have ADVANCED tau pathology "
              "(Braak stage IV-VI, limbic or neocortical spread) or EARLY tau pathology "
              "(Braak stage 0-III)?"),
)

# --- T2: brain age regression on cognitively normal OASIS scans -------------
T2_FEATS = ["sex_M", "eTIV", "nWBV", "ASF"]
T2_PRETTY = {
    "sex_M": "male sex (1=male, 0=female)",
    "eTIV":  "estimated total intracranial volume (mm^3)",
    "nWBV":  "normalized whole-brain volume (fraction of intracranial volume)",
    "ASF":   "atlas scaling factor",
}
TASK2 = Task(
    key="T2_brainage", name="Brain age regression (OASIS-1/2 cognitively normal scans)",
    csv="oasis_brainage.csv", group_col="Subject_ID", target="Age", kind="reg",
    features=T2_FEATS, pretty=T2_PRETTY, context_cols=[],
    row_filter=lambda d: (d["is_control"] == 1) & d[T2_FEATS].notna().all(axis=1),
    question="Estimate this person's chronological age in years from their structural MRI summary.",
)

# --- T3: longitudinal progression, Ding et al. L2C transform on OASIS-2 -----
T3_FEATS = [
    # NOTE (amendment A2): prior-visit CDR- and dementia-derived features are EXCLUDED.
    # OASIS defines `demented` from CDR, so mr_CDR / mr_demented / worst_demented are
    # circular with the target and drive every model to AUC = 1.000 (verified).
    "horizon", "sex_M", "EDUC", "SES", "current_age", "n_prior_visits",
    "mr_MMSE", "time_since_mr_MMSE", "mr_change_MMSE", "low_MMSE", "high_MMSE",
    "mr_nWBV", "time_since_mr_nWBV", "mr_change_nWBV", "low_nWBV", "high_nWBV",
    "mr_eTIV", "mr_ASF",
]
T3_PRETTY = {
    "horizon": "months from last observed visit to the visit being predicted",
    "sex_M": "male sex (1=male, 0=female)", "EDUC": "years of education",
    "SES": "socioeconomic status (1=highest .. 5=lowest)",
    "current_age": "age at the visit being predicted (years)",
    "n_prior_visits": "number of prior visits observed",
    "mr_MMSE": "most recent MMSE (0-30)", "time_since_mr_MMSE": "months since that MMSE",
    "mr_change_MMSE": "MMSE change per month over prior visits",
    "low_MMSE": "lowest MMSE ever recorded", "high_MMSE": "highest MMSE ever recorded",
    "mr_CDR": "most recent Clinical Dementia Rating (0/0.5/1/2)",
    "time_since_mr_CDR": "months since that CDR",
    "mr_change_CDR": "CDR change per month over prior visits",
    "low_CDR": "lowest CDR ever recorded", "high_CDR": "highest CDR ever recorded",
    "mr_nWBV": "most recent normalized whole-brain volume",
    "time_since_mr_nWBV": "months since that nWBV measurement",
    "mr_change_nWBV": "nWBV change per month over prior visits",
    "low_nWBV": "lowest nWBV recorded", "high_nWBV": "highest nWBV recorded",
    "mr_eTIV": "most recent estimated total intracranial volume (mm^3)",
    "mr_ASF": "most recent atlas scaling factor",
    "mr_demented": "dementia status at the most recent prior visit (1=demented)",
    "worst_demented": "worst dementia status across prior visits (1=demented)",
    "ever_milder": "1 if status ever improved relative to its worst prior value",
}
TASK3 = Task(
    key="T3_l2c", name="Longitudinal dementia progression (OASIS-2, L2C transform)",
    csv="oasis_l2c.csv", group_col="Subject_ID", target="y_demented", kind="clf",
    features=T3_FEATS, pretty=T3_PRETTY, context_cols=[],
    positive_label="demented", negative_label="not demented",
    question=("Will this person be clinically demented at the future visit described "
              "by the follow-up horizon below?"),
)

TASKS = {t.key: t for t in (TASK1, TASK2, TASK3)}


# ---------------------------------------------------------------- splitting
def grouped_folds(df: pd.DataFrame, task: Task, n_splits: int = 5, seed: int = SEED):
    """Deterministic group-wise k-fold. Groups are hashed to folds so the split is
    identical across every arm and every script run."""
    groups = df[task.group_col].astype(str).values
    uniq = sorted(set(groups))
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(uniq))
    assign = {uniq[o]: i % n_splits for i, o in enumerate(order)}
    fold = np.array([assign[g] for g in groups])
    return [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in range(n_splits)]


# ---------------------------------------------------------------- perturbation
def perturb(X: pd.DataFrame, task: Task, kind: str, level: float, seed: int) -> pd.DataFrame:
    """Apply an identical perturbation to any arm's feature matrix.

    kind='mcar'  -> level = fraction of feature cells set to NaN (missing).
    kind='noise' -> level = SD of additive Gaussian noise in units of each
                    *continuous* feature's training SD. Binary features are left
                    alone (noise on a 0/1 indicator is not interpretable).
    """
    if level <= 0:
        return X.copy()
    rng = np.random.RandomState(seed)
    Xp = X.copy()
    cols = task.features
    if kind == "mcar":
        mask = rng.rand(len(Xp), len(cols)) < level
        for j, c in enumerate(cols):
            Xp.loc[Xp.index[mask[:, j]], c] = np.nan
    elif kind == "noise":
        for c in cols:
            v = Xp[c].astype(float)
            if v.dropna().nunique() <= 2:      # binary indicator -> skip
                continue
            sd = v.std()
            if not np.isfinite(sd) or sd == 0:
                continue
            Xp[c] = v + rng.normal(0, level * sd, size=len(v))
    else:
        raise ValueError(kind)
    return Xp


# ---------------------------------------------------------------- prompts
def _fmt_val(v, col):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "not measured"
    if col in ("sex_M", "apoe4", "mr_demented", "worst_demented", "ever_milder"):
        return str(int(round(float(v))))
    a = abs(float(v))
    if a != 0 and (a < 1e-3 or a >= 1e5):
        return f"{float(v):.3e}"
    return f"{float(v):.4g}" if a < 100 else f"{float(v):.1f}"


def case_block(row: pd.Series, task: Task) -> str:
    """Render one case as a clinician-readable measurement list."""
    lines = []
    for c in task.context_cols:
        if c == "structure_acronym":
            lines.append(f"- brain region sampled: {row[c]}")
        else:
            lines.append(f"- {c}: {row[c]}")
    for c in task.features:
        lines.append(f"- {task.pretty.get(c, c)}: {_fmt_val(row.get(c), c)}")
    return "\n".join(lines)


SYS = ("You are a careful neuropathology and neuroimaging consultant. You are given "
       "quantitative measurements from a research cohort and must produce a prediction. "
       "Always finish with a line of exactly the required ANSWER format.")


def answer_spec(task: Task) -> str:
    if task.kind == "clf":
        return (f"Finish with a final line exactly of the form:\n"
                f"ANSWER: <{task.positive_label}|{task.negative_label}> | CONFIDENCE: <0-100>\n"
                f"where CONFIDENCE is your probability (0-100) that the answer is "
                f"'{task.positive_label}'.")
    return ("Finish with a final line exactly of the form:\n"
            "ANSWER: <age in years, a single number>")


COND_INSTRUCTION = {
    # no reasoning at all: the model must answer immediately
    "direct": ("Do NOT explain, do NOT reason, do NOT write anything else. "
               "Output ONLY the final answer line."),
    # chain-of-thought: the intervention under test
    "cot":    ("Reason step by step before answering. Structure your reasoning as an explicit "
               "chain over intermediate biological factors, for example: "
               "amyloid burden -> tau pathology -> neurodegeneration/atrophy -> clinical outcome. "
               "At each step name the specific measurement you are using and what it implies. "
               "Then give the final answer line. Keep the whole chain under 180 words."),
    # compute-matched control: extra tokens, no semantic content (Lanham et al. 2023)
    # compute-matched control (Lanham et al. 2023): the assistant turn is FORCE-PREFILLED
    # with a run of '.' tokens matched in count to the median CoT length for that task,
    # so the model gets the same number of extra forward passes with zero semantic content.
    "filler": ("Do NOT explain and do NOT reason. Output ONLY the final answer line."),
}


def build_prompt(row: pd.Series, task: Task, condition: str, shots: list[pd.Series],
                 shot_labels: list) -> list[dict]:
    parts = [f"TASK: {task.question}\n"]
    if shots:
        parts.append("Here are labelled reference cases from the same cohort:\n")
        for s, lab in zip(shots, shot_labels):
            parts.append(case_block(s, task))
            if task.kind == "clf":
                lab_s = task.positive_label if int(lab) == 1 else task.negative_label
                parts.append(f"-> Correct answer: {lab_s}\n")
            else:
                parts.append(f"-> Correct answer: {float(lab):.0f} years\n")
    parts.append("Now the case to predict:\n")
    parts.append(case_block(row, task))
    parts.append("")
    parts.append(COND_INSTRUCTION[condition])
    parts.append(answer_spec(task))
    return [{"role": "system", "content": SYS},
            {"role": "user", "content": "\n".join(parts)}]


# ---------------------------------------------------------------- parsing
_ANS_CLF = re.compile(r"ANSWER\s*:\s*([A-Za-z \-]+?)\s*(?:\||$|\n)", re.I)
_CONF = re.compile(r"CONFIDENCE\s*:\s*([0-9]+(?:\.[0-9]+)?)", re.I)
_ANS_REG = re.compile(r"ANSWER\s*:\s*\**\s*(-?[0-9]+(?:\.[0-9]+)?)", re.I)


def parse_answer(text: str, task: Task):
    """Return (label_or_value, confidence_or_None, parsed_ok)."""
    if task.kind == "clf":
        pos, neg = task.positive_label.lower(), task.negative_label.lower()
        m = list(_ANS_CLF.finditer(text))
        lab = None
        if m:
            s = m[-1].group(1).strip().lower()
            if s.startswith(neg) or s in ("no", "negative", "not ad"):
                lab = 0
            elif s.startswith(pos) or s in ("yes", "positive"):
                lab = 1
        if lab is None:                       # fallback: last mention anywhere
            tail = text[-300:].lower()
            i_neg, i_pos = tail.rfind(neg), tail.rfind(pos)
            if i_neg == -1 and i_pos == -1:
                return None, None, False
            lab = 0 if i_neg >= i_pos else 1
            ok = False
        else:
            ok = True
        c = _CONF.findall(text)
        conf = float(c[-1]) / 100.0 if c else (0.75 if lab == 1 else 0.25)
        conf = min(max(conf, 0.0), 1.0)
        if lab == 0 and c:
            pass                              # confidence is always P(positive) by spec
        return lab, conf, ok
    m = _ANS_REG.findall(text)
    if not m:
        n = re.findall(r"(-?\d{2,3}(?:\.\d+)?)\s*(?:years|year|y\.o\.|yo)\b", text, re.I)
        if not n:
            return None, None, False
        return float(n[-1]), None, False
    return float(m[-1]), None, True


# ---------------------------------------------------------------- statistics
def cluster_bootstrap(groups, stat_fn, B=2000, seed=SEED):
    """Bootstrap over *clusters* (donors / subjects), not rows."""
    rng = np.random.RandomState(seed)
    g = np.asarray(groups)
    uniq = np.array(sorted(set(g.tolist())), dtype=object)
    idx_by_g = {u: np.where(g == u)[0] for u in uniq}
    out = []
    for _ in range(B):
        pick = rng.randint(0, len(uniq), len(uniq))
        idx = np.concatenate([idx_by_g[uniq[p]] for p in pick])
        try:
            v = stat_fn(idx)
        except Exception:
            v = np.nan
        out.append(v)
    return np.asarray(out, dtype=float)


def ci(vals, lo=2.5, hi=97.5):
    v = np.asarray(vals, dtype=float); v = v[np.isfinite(v)]
    if v.size == 0:
        return (np.nan, np.nan)
    return (float(np.percentile(v, lo)), float(np.percentile(v, hi)))


def boot_p(diffs):
    """Two-sided bootstrap p-value for H0: difference == 0."""
    d = np.asarray(diffs, dtype=float); d = d[np.isfinite(d)]
    if d.size == 0:
        return np.nan
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(min(1.0, max(p, 1.0 / len(d))))


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    q = np.full_like(p, np.nan)
    pp = p[ok]; n = len(pp)
    if n == 0:
        return q
    order = np.argsort(pp)
    ranked = pp[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n); out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q
