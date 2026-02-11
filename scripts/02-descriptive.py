"""
02-descriptive.py — Descriptive Epidemiology (Objective 4)

Generates manuscript tables and figures for the Brain AttACKs study.

Inputs:
    data/processed/brain_attacks_combined.parquet  (from 01-data-prep.py)
    data/processed/brain_attacks_llm_coded.csv     (from 01b-llm-coding.py, optional)

Outputs:
    outputs/tables/table1_demographics.csv
    outputs/tables/table2_mi_diagnoses.csv
    outputs/tables/tableS1_univariate.csv
    outputs/tables/tableS2_practice_variation.csv
    outputs/figures/fig_temporal_patterns.png
    outputs/figures/fig_symptom_prevalence.png
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data" / "processed"
TABLES = PROJECT / "outputs" / "tables"
FIGURES = PROJECT / "outputs" / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


# ===================================================================
# 1. LOAD DATA
# ===================================================================

def load_data():
    """Load combined dataset with proper types."""
    df = pd.read_parquet(DATA / "brain_attacks_combined.parquet")
    print(f"Loaded: {len(df)} records, {len(df.columns)} columns")

    # Merge LLM-coded columns if available
    llm_path = DATA / "brain_attacks_llm_coded.csv"
    if llm_path.exists():
        llm = pd.read_csv(llm_path, dtype=str)
        # Get new columns only
        new_cols = [c for c in llm.columns if c not in df.columns]
        if new_cols:
            key_cols = ["record_id", "year"]
            merge_cols = key_cols + new_cols
            llm_sub = llm[merge_cols].copy()
            df["record_id"] = df["record_id"].astype(str)
            df["year"] = df["year"].astype(str)
            llm_sub["record_id"] = llm_sub["record_id"].astype(str)
            llm_sub["year"] = llm_sub["year"].astype(str)
            df = df.merge(llm_sub, on=key_cols, how="left")
            print(f"  Merged {len(new_cols)} LLM-coded columns")

    return df


# ===================================================================
# 2. TABLE HELPERS
# ===================================================================

def _fmt_continuous(series, decimals=1):
    """Format as mean (SD)."""
    m = series.mean()
    s = series.std()
    return f"{m:.{decimals}f} ({s:.{decimals}f})"


def _fmt_median_iqr(series, decimals=1):
    """Format as median [IQR]."""
    med = series.median()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    return f"{med:.{decimals}f} [{q1:.{decimals}f}-{q3:.{decimals}f}]"


def _fmt_n_pct(count, total):
    """Format as n (%)."""
    pct = 100 * count / total if total > 0 else 0
    return f"{int(count)} ({pct:.1f}%)"


def _p_continuous(s1, s2):
    """Welch's t-test, return formatted p-value."""
    s1c = s1.dropna()
    s2c = s2.dropna()
    if len(s1c) < 2 or len(s2c) < 2:
        return ""
    _, p = stats.ttest_ind(s1c, s2c, equal_var=False)
    return _fmt_p(p)


def _p_binary(col, group, df):
    """Chi-squared or Fisher's exact for binary variable."""
    ct = pd.crosstab(df[col].fillna(0).astype(int), df[group])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return ""
    if ct.min().min() < 5:
        # Fisher exact (2x2 only)
        if ct.shape == (2, 2):
            _, p = stats.fisher_exact(ct)
        else:
            _, p, _, _ = stats.chi2_contingency(ct)
    else:
        _, p, _, _ = stats.chi2_contingency(ct)
    return _fmt_p(p)


def _fmt_p(p):
    if p < 0.001:
        return "<0.001"
    elif p < 0.01:
        return f"{p:.3f}"
    else:
        return f"{p:.2f}"


def _or_ci(col, outcome, df):
    """Odds ratio with 95% CI for binary predictor vs binary outcome."""
    ct = pd.crosstab(df[col].fillna(0).astype(int), df[outcome])
    if ct.shape != (2, 2):
        return "", ""
    a, b = ct.iloc[1, 1], ct.iloc[1, 0]  # exposed: outcome+, outcome-
    c, d = ct.iloc[0, 1], ct.iloc[0, 0]  # unexposed: outcome+, outcome-
    if a == 0 or b == 0 or c == 0 or d == 0:
        return "—", ""
    oratio = (a * d) / (b * c)
    log_or = np.log(oratio)
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    ci_lo = np.exp(log_or - 1.96 * se)
    ci_hi = np.exp(log_or + 1.96 * se)
    return f"{oratio:.2f}", f"{ci_lo:.2f}-{ci_hi:.2f}"


# ===================================================================
# 3. TABLE 1: DEMOGRAPHICS
# ===================================================================

def table1(df):
    """Demographics and clinical variables by outcome group."""
    print("\n=== TABLE 1: Demographics ===")

    mi = df[df["outcome_med_important"] == 1]
    bn = df[df["outcome_med_important"] == 0]
    n_all, n_mi, n_bn = len(df), len(mi), len(bn)

    rows = []

    def add_continuous(label, col, dec=1):
        if col not in df.columns:
            return
        rows.append({
            "Variable": f"{label}, mean (SD)",
            "Overall": _fmt_continuous(df[col], dec),
            f"MI (n={n_mi})": _fmt_continuous(mi[col], dec),
            f"Benign (n={n_bn})": _fmt_continuous(bn[col], dec),
            "p": _p_continuous(mi[col], bn[col]),
        })

    def add_binary(label, col):
        if col not in df.columns:
            return
        rows.append({
            "Variable": f"{label}, n (%)",
            "Overall": _fmt_n_pct((df[col] == 1).sum(), n_all),
            f"MI (n={n_mi})": _fmt_n_pct((mi[col] == 1).sum(), n_mi),
            f"Benign (n={n_bn})": _fmt_n_pct((bn[col] == 1).sum(), n_bn),
            "p": _p_binary(col, "outcome_med_important", df),
        })

    def add_categorical(label, col, value_labels):
        if col not in df.columns:
            return
        rows.append({"Variable": label, "Overall": "", f"MI (n={n_mi})": "",
                      f"Benign (n={n_bn})": "", "p": ""})
        for val, vlabel in value_labels.items():
            rows.append({
                "Variable": f"  {vlabel}",
                "Overall": _fmt_n_pct((df[col] == val).sum(), n_all),
                f"MI (n={n_mi})": _fmt_n_pct((mi[col] == val).sum(), n_mi),
                f"Benign (n={n_bn})": _fmt_n_pct((bn[col] == val).sum(), n_bn),
                "p": "",
            })

    # -- Demographics --
    rows.append({"Variable": "Demographics", "Overall": "", f"MI (n={n_mi})": "",
                  f"Benign (n={n_bn})": "", "p": ""})
    add_continuous("Age, years", "age")
    add_binary("Female sex", "gender")  # gender==2 is female; handle below

    # Gender: need to check coding (1=male, 2=female typical in REDCap)
    rows[-1] = {
        "Variable": "Female, n (%)",
        "Overall": _fmt_n_pct((df["gender"] == 2).sum(), n_all),
        f"MI (n={n_mi})": _fmt_n_pct((mi["gender"] == 2).sum(), n_mi),
        f"Benign (n={n_bn})": _fmt_n_pct((bn["gender"] == 2).sum(), n_bn),
        "p": _p_binary("gender", "outcome_med_important", df),
    }

    # -- Presentation --
    rows.append({"Variable": "Presentation", "Overall": "", f"MI (n={n_mi})": "",
                  f"Benign (n={n_bn})": "", "p": ""})
    add_binary("EMS transport", "ems")
    add_categorical("CTAS triage level", "ctas", {
        1: "1 - Resuscitation", 2: "2 - Emergent", 3: "3 - Urgent",
        4: "4 - Less urgent", 5: "5 - Non-urgent"
    })
    add_categorical("Season", "time_year", {
        1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"
    })
    add_categorical("Year", "year", {1: "Year 1", 2: "Year 2"})

    # -- Vitals --
    rows.append({"Variable": "Vitals", "Overall": "", f"MI (n={n_mi})": "",
                  f"Benign (n={n_bn})": "", "p": ""})
    add_continuous("Temperature, °C", "temp")
    add_continuous("Heart rate, bpm", "hr", 0)
    add_continuous("Systolic BP, mmHg", "sys_bp", 0)

    # -- Symptoms --
    rows.append({"Variable": "Presenting symptoms", "Overall": "",
                  f"MI (n={n_mi})": "", f"Benign (n={n_bn})": "", "p": ""})
    symptom_vars = [
        ("Focal weakness", "sx_focal_weakness"),
        ("Sensory changes", "sx_sensory"),
        ("Speech disturbance", "sx_speech"),
        ("Visual disturbance", "sx_visual"),
        ("Eye movement abnormality", "sx_eye_movement"),
        ("Altered consciousness", "sx_altered_consciousness"),
        ("Collapse", "sx_collapse"),
        ("Seizure", "sx_seizure"),
        ("Ataxia", "sx_ataxia"),
        ("Headache (any)", "sx_any_headache"),
        ("Fever", "sx_fever"),
    ]
    for label, col in symptom_vars:
        add_binary(label, col)

    # -- Other symptom subcategories --
    rows.append({"Variable": "Other symptom subcategories", "Overall": "",
                  f"MI (n={n_mi})": "", f"Benign (n={n_bn})": "", "p": ""})
    other_sx_vars = [
        ("  Pain", "other_pain"),
        ("  Movement", "other_movement"),
        ("  Psychiatric", "other_psych"),
        ("  GI", "other_gi"),
        ("  Cranial nerve", "other_cranial_nerve"),
        ("  Inflammation", "other_inflammation"),
        ("  Respiratory", "other_respiratory"),
        ("  Infection", "other_infection"),
        ("  Constitutional", "other_constitutional"),
        ("  Urinary", "other_urinary"),
        ("  Dermatological", "other_dermatological"),
        ("  Hemorrhage", "other_hemorrhage"),
        ("  Syncope", "other_syncope"),
        ("  Cardiac", "other_cardiac"),
        ("  Gynecological", "other_gyn"),
        ("  Ophthalmology", "other_ophthalmology"),
        ("  Sleep", "other_sleep"),
        ("  Hematological", "other_hematological"),
    ]
    for label, col in other_sx_vars:
        add_binary(label, col)

    # -- Exam findings --
    rows.append({"Variable": "Exam findings", "Overall": "",
                  f"MI (n={n_mi})": "", f"Benign (n={n_bn})": "", "p": ""})
    exam_vars = [
        ("Papilledema", "exam_papilledema"),
        ("Meningismus", "exam_meningismus"),
        ("Altered LOC (exam)", "exam_altered_loc"),
    ]
    for label, col in exam_vars:
        add_binary(label, col)

    # -- Prior diagnoses --
    rows.append({"Variable": "Prior diagnoses", "Overall": "",
                  f"MI (n={n_mi})": "", f"Benign (n={n_bn})": "", "p": ""})
    add_binary("Any previous diagnosis", "any_prev_dx")

    t1 = pd.DataFrame(rows)
    out = TABLES / "table1_demographics.csv"
    t1.to_csv(out, index=False)
    print(f"  Saved {out.name} ({len(t1)} rows)")

    # Print key findings
    print(f"\n  N = {n_all} ({n_mi} MI, {n_bn} benign)")
    print(f"  Age: MI {_fmt_continuous(mi['age'])} vs Benign {_fmt_continuous(bn['age'])}")
    print(f"  Female: MI {_fmt_n_pct((mi['gender']==2).sum(), n_mi)} "
          f"vs Benign {_fmt_n_pct((bn['gender']==2).sum(), n_bn)}")

    return t1


# ===================================================================
# 4. TABLE 2: MI DIAGNOSIS DISTRIBUTION
# ===================================================================

def table2(df):
    """Frequency of medically important diagnosis subtypes."""
    print("\n=== TABLE 2: MI Diagnosis Distribution ===")

    mi = df[df["outcome_med_important"] == 1]
    n_mi = len(mi)

    labels = {
        1: "Stroke", 2: "Intracerebral hemorrhage",
        3: "Vascular malformation", 4: "Vasculitis",
        5: "Venous thrombosis", 6: "Demyelinating disorder",
        7: "Brain/meningeal infection", 8: "Hydrocephalus",
        9: "Neoplasm", 10: "Inborn error of metabolism",
        11: "Other medically important",
        12: "IIH (idiopathic intracranial hypertension)",
        13: "Cerebellar ataxia",
    }

    rows = []
    for code, label in labels.items():
        col = f"med_imp_dx___{code}"
        if col not in df.columns:
            continue
        n_total = (df[col] == 1).sum()
        n_y1 = (df[(df[col] == 1) & (df["year"].astype(str) == "1")].shape[0])
        n_y2 = (df[(df[col] == 1) & (df["year"].astype(str) == "2")].shape[0])
        rows.append({
            "Code": code,
            "Diagnosis": label,
            "Total": n_total,
            "% of MI": f"{100 * n_total / n_mi:.1f}%" if n_mi > 0 else "",
            "% of cohort": f"{100 * n_total / len(df):.1f}%",
            "Year 1": n_y1,
            "Year 2": n_y2,
        })

    # Total row (patients may have >1 dx)
    rows.append({
        "Code": "",
        "Diagnosis": "Any medically important",
        "Total": n_mi,
        "% of MI": "100%",
        "% of cohort": f"{100 * n_mi / len(df):.1f}%",
        "Year 1": len(mi[mi["year"].astype(str) == "1"]),
        "Year 2": len(mi[mi["year"].astype(str) == "2"]),
    })

    t2 = pd.DataFrame(rows)
    out = TABLES / "table2_mi_diagnoses.csv"
    t2.to_csv(out, index=False)
    print(f"  Saved {out.name}")

    for _, r in t2.iterrows():
        if r["Total"] > 0:
            print(f"  {r['Diagnosis']}: {r['Total']} ({r['% of MI']})")

    return t2


# ===================================================================
# 5. TABLE S1: UNIVARIATE ASSOCIATIONS
# ===================================================================

def tableS1(df):
    """Complete univariate associations for all binary predictors."""
    print("\n=== TABLE S1: Univariate Associations ===")

    outcome = "outcome_med_important"
    mi = df[df[outcome] == 1]
    bn = df[df[outcome] == 0]
    n_mi, n_bn = len(mi), len(bn)

    # All binary predictor columns
    predictors = []
    for prefix, group in [
        ("sx_", "Symptoms"),
        ("exam_", "Exam"),
        ("ha_", "Headache features"),
        ("inv_", "Investigations"),
        ("other_", "Other symptom class"),
    ]:
        cols = sorted([c for c in df.columns if c.startswith(prefix)
                       and df[c].dtype in [np.int64, np.float64, int, float]])
        for col in cols:
            predictors.append((group, col))

    # Add key single variables
    for col in ["ems", "any_prev_dx", "any_neuroimaging", "age_under_12"]:
        if col in df.columns:
            predictors.append(("Other", col))

    rows = []
    for group, col in predictors:
        n_mi_pos = (mi[col] == 1).sum()
        n_bn_pos = (bn[col] == 1).sum()
        oratio, ci = _or_ci(col, outcome, df)
        p = _p_binary(col, outcome, df)
        rows.append({
            "Group": group,
            "Variable": col,
            "MI n (%)": _fmt_n_pct(n_mi_pos, n_mi),
            "Benign n (%)": _fmt_n_pct(n_bn_pos, n_bn),
            "OR": oratio,
            "95% CI": ci,
            "p": p,
        })

    ts1 = pd.DataFrame(rows)
    out = TABLES / "tableS1_univariate.csv"
    ts1.to_csv(out, index=False)
    print(f"  Saved {out.name} ({len(ts1)} predictors)")

    # Print significant associations
    sig = [r for r in rows if r["p"] and r["p"] not in ("", "—")
           and (r["p"] == "<0.001" or float(r["p"]) < 0.05)]
    print(f"  Significant (p<0.05): {len(sig)} predictors")
    for r in sorted(sig, key=lambda x: x["p"]):
        print(f"    {r['Variable']}: OR={r['OR']} ({r['95% CI']}), p={r['p']}")

    return ts1


# ===================================================================
# 6. TABLE S2: PRACTICE VARIATION
# ===================================================================

def tableS2(df):
    """Imaging, consultation, admission, and treatment patterns."""
    print("\n=== TABLE S2: Practice Variation ===")

    n = len(df)
    rows = []

    # -- Investigations --
    inv_vars = [
        ("EEG", "inv_eeg"), ("CT", "inv_ct"), ("LP", "inv_lp"),
        ("MRI", "inv_mri"), ("CTA", "inv_cta"), ("MRA", "inv_mra"),
        ("Cerebral arteriography", "inv_angiography"),
        ("Any neuroimaging", "any_neuroimaging"),
    ]
    rows.append({"Category": "Investigations", "Variable": "", "Overall": "",
                  "Year 1": "", "Year 2": "", "MI": "", "Benign": ""})
    y1 = df[df["year"].astype(str) == "1"]
    y2 = df[df["year"].astype(str) == "2"]
    mi = df[df["outcome_med_important"] == 1]
    bn = df[df["outcome_med_important"] == 0]

    for label, col in inv_vars:
        if col not in df.columns:
            continue
        rows.append({
            "Category": "Investigations",
            "Variable": label,
            "Overall": _fmt_n_pct((df[col] == 1).sum(), n),
            "Year 1": _fmt_n_pct((y1[col] == 1).sum(), len(y1)),
            "Year 2": _fmt_n_pct((y2[col] == 1).sum(), len(y2)),
            "MI": _fmt_n_pct((mi[col] == 1).sum(), len(mi)),
            "Benign": _fmt_n_pct((bn[col] == 1).sum(), len(bn)),
        })

    # -- Neuroimaging results --
    imaged = df[df["any_neuroimaging"] == 1]
    n_imaged = len(imaged)
    n_abnormal = (imaged["outcome_abnormal_imaging"] == 1).sum()
    n_mi_abn = (imaged["outcome_mi_imaging"] == 1).sum()
    rows.append({"Category": "Imaging results", "Variable": "Among imaged",
                  "Overall": f"n={n_imaged}", "Year 1": "", "Year 2": "",
                  "MI": "", "Benign": ""})
    rows.append({"Category": "Imaging results", "Variable": "Abnormal",
                  "Overall": _fmt_n_pct(n_abnormal, n_imaged),
                  "Year 1": "", "Year 2": "", "MI": "", "Benign": ""})
    rows.append({"Category": "Imaging results", "Variable": "MI cause",
                  "Overall": _fmt_n_pct(n_mi_abn, n_imaged),
                  "Year 1": "", "Year 2": "", "MI": "", "Benign": ""})

    # -- Consultations (Y2 by specialty) --
    rows.append({"Category": "Consultations", "Variable": "", "Overall": "",
                  "Year 1": "", "Year 2": "", "MI": "", "Benign": ""})
    if "consult" in df.columns:
        rows.append({
            "Category": "Consultations",
            "Variable": "Any consultation",
            "Overall": _fmt_n_pct((df["consult"] == 1).sum(), n),
            "Year 1": _fmt_n_pct((y1["consult"] == 1).sum(), len(y1)),
            "Year 2": _fmt_n_pct((y2["consult"] == 1).sum(), len(y2)),
            "MI": _fmt_n_pct((mi["consult"] == 1).sum(), len(mi)),
            "Benign": _fmt_n_pct((bn["consult"] == 1).sum(), len(bn)),
        })
    consult_labels = {
        1: "Pediatrics", 2: "Neurology", 3: "Ophthalmology",
        4: "Neurosurgery", 5: "Cardiology", 6: "PICU", 7: "Psychiatry",
        8: "Other",
    }
    for code, label in consult_labels.items():
        col = f"consult_specify___{code}"
        if col in df.columns:
            rows.append({
                "Category": "Consultations",
                "Variable": f"  {label}",
                "Overall": _fmt_n_pct((df[col] == 1).sum(), n),
                "Year 1": _fmt_n_pct((y1[col] == 1).sum(), len(y1)),
                "Year 2": _fmt_n_pct((y2[col] == 1).sum(), len(y2)),
                "MI": "", "Benign": "",
            })

    # -- Admissions --
    rows.append({"Category": "Disposition", "Variable": "", "Overall": "",
                  "Year 1": "", "Year 2": "", "MI": "", "Benign": ""})
    for label, col in [("Any admission", "admit_any"), ("ICU admission", "icu_any")]:
        if col in df.columns:
            rows.append({
                "Category": "Disposition",
                "Variable": label,
                "Overall": _fmt_n_pct((df[col] == 1).sum(), n),
                "Year 1": _fmt_n_pct((y1[col] == 1).sum(), len(y1)),
                "Year 2": _fmt_n_pct((y2[col] == 1).sum(), len(y2)),
                "MI": _fmt_n_pct((mi[col] == 1).sum(), len(mi)),
                "Benign": _fmt_n_pct((bn[col] == 1).sum(), len(bn)),
            })

    ts2 = pd.DataFrame(rows)
    out = TABLES / "tableS2_practice_variation.csv"
    ts2.to_csv(out, index=False)
    print(f"  Saved {out.name} ({len(ts2)} rows)")
    return ts2


# ===================================================================
# 7. FIGURES
# ===================================================================

def make_figures(df):
    """Generate descriptive figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("\n  matplotlib not available, skipping figures")
        return

    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

    # --- Figure: Symptom prevalence by outcome ---
    print("\n=== Figure: Symptom Prevalence ===")
    symptom_vars = [
        ("Headache", "sx_any_headache"), ("Seizure", "sx_seizure"),
        ("Collapse", "sx_collapse"), ("Visual", "sx_visual"),
        ("Alt. consciousness", "sx_altered_consciousness"),
        ("Fever", "sx_fever"), ("Ataxia", "sx_ataxia"),
        ("Speech", "sx_speech"), ("Focal weakness", "sx_focal_weakness"),
        ("Sensory", "sx_sensory"), ("Eye movement", "sx_eye_movement"),
        ("GI", "other_gi"), ("Syncope", "other_syncope"),
        ("Respiratory", "other_respiratory"), ("Pain", "other_pain"),
        ("Cranial nerve", "other_cranial_nerve"),
        ("Psychiatric", "other_psych"), ("Movement", "other_movement"),
    ]
    mi = df[df["outcome_med_important"] == 1]
    bn = df[df["outcome_med_important"] == 0]

    labels = [s[0] for s in symptom_vars]
    mi_pcts = [100 * (mi[s[1]] == 1).sum() / len(mi) for s in symptom_vars]
    bn_pcts = [100 * (bn[s[1]] == 1).sum() / len(bn) for s in symptom_vars]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, mi_pcts, w, label=f"MI (n={len(mi)})", color="#d62728")
    ax.bar(x + w/2, bn_pcts, w, label=f"Benign (n={len(bn)})", color="#1f77b4",
           alpha=0.7)
    ax.set_ylabel("Prevalence (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.set_title("Presenting Symptoms by Outcome Group")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_symptom_prevalence.png")
    plt.close()
    print(f"  Saved fig_symptom_prevalence.png")

    # --- Figure: Temporal patterns ---
    print("\n=== Figure: Temporal Patterns ===")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Season
    season_map = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
    season_counts = df["time_year"].value_counts().sort_index()
    axes[0].bar([season_map.get(int(s), str(s)) for s in season_counts.index],
                season_counts.values, color="#2ca02c")
    axes[0].set_title("Presentations by Season")
    axes[0].set_ylabel("Count")

    # Year comparison
    y_counts = df["year"].astype(str).value_counts().sort_index()
    mi_by_year = df.groupby(df["year"].astype(str))["outcome_med_important"].sum()
    bn_by_year = y_counts - mi_by_year
    x_yr = np.arange(len(y_counts))
    axes[1].bar(x_yr, mi_by_year.values, label="MI", color="#d62728")
    axes[1].bar(x_yr, bn_by_year.values, bottom=mi_by_year.values,
                label="Benign", color="#1f77b4", alpha=0.7)
    axes[1].set_xticks(x_yr)
    axes[1].set_xticklabels([f"Year {y}" for y in y_counts.index])
    axes[1].set_title("Cases by Year and Outcome")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(FIGURES / "fig_temporal_patterns.png")
    plt.close()
    print(f"  Saved fig_temporal_patterns.png")


# ===================================================================
# 8. NEUROIMAGING APPROPRIATENESS (HEADACHE PATIENTS)
# ===================================================================

def neuroimaging_appropriateness(df):
    """For headache patients: imaging rates vs appropriateness rating."""
    print("\n=== Neuroimaging Appropriateness (headache patients) ===")

    ha = df[df["sx_any_headache"] == 1]
    n_ha = len(ha)
    print(f"  Headache patients: {n_ha}")

    n_imaged = (ha["any_neuroimaging"] == 1).sum()
    print(f"  Imaged: {_fmt_n_pct(n_imaged, n_ha)}")

    if "ni_appropriateness" in ha.columns:
        rated = ha[ha["ni_appropriateness"].notna()]
        print(f"  With appropriateness rating: {len(rated)}")
        print(f"  Rating distribution:")
        for val in sorted(rated["ni_appropriateness"].dropna().unique()):
            n = (rated["ni_appropriateness"] == val).sum()
            print(f"    {int(val)}: {n} ({100*n/len(rated):.1f}%)")

    ha_mi = ha[ha["outcome_med_important"] == 1]
    ha_bn = ha[ha["outcome_med_important"] == 0]
    print(f"  Headache MI: {len(ha_mi)}, Headache benign: {len(ha_bn)}")
    print(f"  MI imaged: {_fmt_n_pct((ha_mi['any_neuroimaging']==1).sum(), len(ha_mi))}")
    print(f"  Benign imaged: {_fmt_n_pct((ha_bn['any_neuroimaging']==1).sum(), len(ha_bn))}")


# ===================================================================
# MAIN
# ===================================================================

def main():
    print("=" * 60)
    print("Brain AttACKs: Descriptive Epidemiology")
    print("=" * 60)

    df = load_data()

    # Ensure year is numeric-comparable for grouping
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    t1 = table1(df)
    t2 = table2(df)
    ts1 = tableS1(df)
    ts2 = tableS2(df)
    neuroimaging_appropriateness(df)
    make_figures(df)

    print("\n" + "=" * 60)
    print("Done. Outputs in outputs/tables/ and outputs/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
