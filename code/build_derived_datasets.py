#!/usr/bin/env python3
"""
Build analysis-ready derived datasets for the CoT-interpretability study.

Outputs (datasets/derived/):
  oasis_brainage.csv      OASIS-1 + OASIS-2 pooled, brain-age regression + dementia status
  oasis_l2c.csv           OASIS-2 longitudinal, L2C-transformed (Ding et al. 2025 style)
  allen_act_atn.csv       Allen ACT donor x region amyloid/tau panel + neuropath ground truth
"""
import pandas as pd, numpy as np, os, re

OUT = "datasets/derived"; os.makedirs(OUT, exist_ok=True)

# ---------------- 1. OASIS pooled brain-age ----------------
x = pd.read_csv("datasets/kaggle_mri-and-alzheimers/oasis_cross-sectional.csv")
x = x.rename(columns={"Educ": "EDUC", "ID": "MRI_ID"})
x["Subject_ID"] = x["MRI_ID"].str.replace(r"_MR\d+$", "", regex=True)
x["cohort"] = "OASIS1"; x["Visit"] = 1; x["MR_Delay"] = 0

l = pd.read_csv("datasets/kaggle_mri-and-alzheimers/oasis_longitudinal.csv")
l = l.rename(columns={"Subject ID": "Subject_ID", "MRI ID": "MRI_ID", "MR Delay": "MR_Delay"})
l["cohort"] = "OASIS2"

keep = ["cohort","Subject_ID","MRI_ID","Visit","MR_Delay","M/F","Age","EDUC","SES","MMSE","CDR","eTIV","nWBV","ASF"]
oasis = pd.concat([x.reindex(columns=keep), l.reindex(columns=keep)], ignore_index=True)
oasis["Group"] = l.set_index("MRI_ID")["Group"].reindex(oasis["MRI_ID"]).values
# OASIS-1 has no Group column: derive from CDR (CDR>0 => Demented) where CDR is known
m = oasis["Group"].isna() & oasis["CDR"].notna()
oasis.loc[m, "Group"] = np.where(oasis.loc[m, "CDR"] > 0, "Demented", "Nondemented")
oasis["sex_M"] = (oasis["M/F"] == "M").astype(int)
oasis["demented"] = oasis["Group"].map({"Demented": 1, "Converted": 1, "Nondemented": 0})
# healthy-control mask for fitting the brain-age normative model
oasis["is_control"] = ((oasis["CDR"].fillna(-1) == 0) & (oasis["demented"] == 0)).astype(int)
oasis.to_csv(f"{OUT}/oasis_brainage.csv", index=False)
print(f"oasis_brainage.csv {oasis.shape}  controls={int(oasis.is_control.sum())} "
      f"demented={int(oasis.demented.sum())} age {oasis.Age.min()}-{oasis.Age.max()}")

# ---------------- 2. L2C transform on OASIS-2 (Ding et al. 2025 eqs 1-7) ----------------
NUM = ["MMSE","CDR","eTIV","nWBV","ASF"]
def l2c(df, time_col="MR_Delay"):
    rows = []
    for sid, g in df.sort_values(time_col).groupby("Subject_ID"):
        g = g.reset_index(drop=True)
        for i in range(1, len(g)):
            t = g.loc[i, time_col]; hist = g.iloc[:i]
            r = {"Subject_ID": sid, "target_visit": int(g.loc[i, "Visit"]),
                 "horizon": t - hist[time_col].iloc[-1],
                 "sex_M": g.loc[i,"sex_M"], "EDUC": g.loc[i,"EDUC"], "SES": g.loc[i,"SES"],
                 "current_age": g.loc[i,"Age"], "n_prior_visits": i}
            for c in NUM:
                s = hist[[time_col, c]].dropna()
                if len(s) == 0:
                    r.update({f"mr_{c}":np.nan,f"time_since_mr_{c}":np.nan,f"mr_change_{c}":np.nan,
                              f"low_{c}":np.nan,f"high_{c}":np.nan,f"dt_low_{c}":np.nan,f"dt_high_{c}":np.nan})
                    continue
                r[f"mr_{c}"] = s[c].iloc[-1]                       # most recent (eq 1)
                r[f"time_since_mr_{c}"] = t - s[time_col].iloc[-1] # eq 2
                if len(s) >= 2:                                     # change rate (eq 3)
                    dt = s[time_col].iloc[-1] - s[time_col].iloc[-2]
                    r[f"mr_change_{c}"] = (s[c].iloc[-1]-s[c].iloc[-2])/dt if dt else 0.0
                else:
                    r[f"mr_change_{c}"] = np.nan
                r[f"low_{c}"]  = s[c].min()                          # eq 4
                r[f"high_{c}"] = s[c].max()                          # eq 6
                r[f"dt_low_{c}"]  = t - s.loc[s[c].idxmin(), time_col]  # eq 5
                r[f"dt_high_{c}"] = t - s.loc[s[c].idxmax(), time_col]  # eq 7
            # diagnosis-history indicators
            hd = hist["demented"].dropna()
            r["mr_demented"]   = hd.iloc[-1] if len(hd) else np.nan
            r["worst_demented"] = hd.max() if len(hd) else np.nan
            r["ever_milder"]   = int(len(hd) > 1 and hd.iloc[-1] < hd.max())
            # targets at the current time point
            r["y_demented"] = g.loc[i,"demented"]; r["y_MMSE"] = g.loc[i,"MMSE"]
            r["y_CDR"] = g.loc[i,"CDR"];           r["y_nWBV"] = g.loc[i,"nWBV"]
            rows.append(r)
    return pd.DataFrame(rows)

o2 = oasis[oasis.cohort == "OASIS2"].copy()
L = l2c(o2)
L.to_csv(f"{OUT}/oasis_l2c.csv", index=False)
print(f"oasis_l2c.csv {L.shape}  subjects={L.Subject_ID.nunique()}  "
      f"y_demented pos-rate={L.y_demented.mean():.3f}")

# ---------------- 3. Allen ACT amyloid/tau panel ----------------
d = pd.read_csv("datasets/allen_aging_dementia_tbi/DonorInformation.csv")
p = pd.read_csv("datasets/allen_aging_dementia_tbi/ProteinAndPathologyQuantifications.csv")
a = p.merge(d, on="donor_id", how="left", suffixes=("", "_donor"))
# age is binned for the oldest-old ("90-94","95-99","100+"); take bin midpoint
def age_mid(v):
    v = str(v).strip()
    if re.fullmatch(r"\d+", v): return float(v)
    m = re.fullmatch(r"(\d+)-(\d+)", v)
    if m: return (float(m.group(1)) + float(m.group(2))) / 2
    m = re.fullmatch(r"(\d+)\+", v)
    return float(m.group(1)) if m else np.nan
a["age_years"] = a["age"].map(age_mid)
a["age_censored"] = (~a["age"].astype(str).str.fullmatch(r"\d+")).astype(int)
a["sex_M"]  = (a["sex"] == "M").astype(int)
a["apoe4"]  = a["apo_e4_allele"].map({"Y": 1, "N": 0})
a["y_dementia"] = (a["act_demented"] == "Dementia").astype(int)
# NIA-Reagan: 1=high,2=intermediate,3=low,4=no  -> AD-neuropathology positive
a["y_niareagan_AD"] = a["nia_reagan"].isin([1, 2]).astype(int)
a["y_braak_ge4"] = (pd.to_numeric(a["braak"], errors="coerce") >= 4).astype(int)  # 'T' proxy
a["y_cerad_ge2"] = (pd.to_numeric(a["cerad"], errors="coerce") >= 2).astype(int)  # 'A' proxy
a.to_csv(f"{OUT}/allen_act_atn.csv", index=False)
print(f"allen_act_atn.csv {a.shape}  donors={a.donor_id.nunique()} "
      f"regions={sorted(a.structure_acronym.unique())}")
print("  label rates:", {k: round(a[k].mean(),3) for k in
      ["y_dementia","y_niareagan_AD","y_braak_ge4","y_cerad_ge2"]})
