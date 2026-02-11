"""
05-neuroimaging.py — Predictors of Abnormal Neuroimaging (Objective 3)

Among children who underwent neuroimaging, identifies clinical predictors
of abnormal results and medically important causes.

Inputs:
    data/processed/brain_attacks_combined.parquet

Outputs:
    outputs/tables/table4_neuroimaging_predictors.csv
    outputs/tables/table5_neuroimaging_yield.csv
    outputs/figures/fig_neuroimaging_yield.png
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, fbeta_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data" / "processed"
TABLES = PROJECT / "outputs" / "tables"
FIGURES = PROJECT / "outputs" / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

TARGET = "outcome_med_important"

# Triage-available predictors (same as 03)
PREDICTORS = [
    "sx_focal_weakness", "sx_sensory", "sx_speech", "sx_visual",
    "sx_eye_movement", "sx_altered_consciousness", "sx_collapse",
    "sx_seizure", "sx_ataxia", "sx_any_headache", "sx_fever",
    "other_pain", "other_movement", "other_psych", "other_gi",
    "other_cranial_nerve", "other_inflammation", "other_respiratory",
    "other_infection", "other_constitutional", "other_urinary",
    "other_dermatological", "other_hemorrhage", "other_syncope",
    "other_cardiac", "other_gyn", "other_ophthalmology",
    "other_sleep", "other_hematological",
    "exam_papilledema", "exam_meningismus", "exam_altered_loc",
    "ha_nausea", "ha_vomiting", "ha_photophobia", "ha_phonophobia",
    "ha_maximal_onset", "ha_progressive_days", "ha_progressive_months",
    "ha_non_progressive", "ha_abrupt_onset", "ha_gradual_onset",
    "ha_bilateral", "ha_unilateral", "ha_frontal", "ha_temporal", "ha_occipital",
    "age", "gender", "ems", "ctas", "temp", "hr",
    "any_prev_dx", "age_under_12",
]


def _fmt(n, total):
    pct = 100 * n / total if total > 0 else 0
    return f"{int(n)} ({pct:.1f}%)"


def _or_ci(ct):
    """OR with 95% CI from 2x2 contingency table."""
    if ct.shape != (2, 2):
        return np.nan, "", ""
    a, b = ct.iloc[1, 1], ct.iloc[1, 0]
    c, d = ct.iloc[0, 1], ct.iloc[0, 0]
    if any(v == 0 for v in [a, b, c, d]):
        return np.nan, "—", ""
    oratio = (a * d) / (b * c)
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    lo = np.exp(np.log(oratio) - 1.96 * se)
    hi = np.exp(np.log(oratio) + 1.96 * se)
    return oratio, f"{oratio:.2f}", f"{lo:.2f}-{hi:.2f}"


def _fmt_p(p):
    if p < 0.001:
        return "<0.001"
    elif p < 0.01:
        return f"{p:.3f}"
    return f"{p:.2f}"


# ===================================================================
# 1. LOAD & SUBSET
# ===================================================================

def load_data():
    df = pd.read_parquet(DATA / "brain_attacks_combined.parquet")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    print(f"Full cohort: {len(df)}")

    imaged = df[df["any_neuroimaging"] == 1].copy()
    print(f"Neuroimaged: {len(imaged)}")
    print(f"  Abnormal: {(imaged['outcome_abnormal_imaging']==1).sum()}")
    print(f"  MI cause: {(imaged['outcome_mi_imaging']==1).sum()}")
    print(f"  MI outcome: {(imaged[TARGET]==1).sum()}")

    return df, imaged


# ===================================================================
# 2. TABLE 4: PREDICTORS OF ABNORMAL NEUROIMAGING
# ===================================================================

def table4_neuroimaging_predictors(imaged):
    """Logistic regression for abnormal neuroimaging."""
    print("\n=== Table 4: Predictors of Abnormal Neuroimaging ===")
    from scipy import stats

    outcome = "outcome_abnormal_imaging"
    features = [f for f in PREDICTORS if f in imaged.columns]
    X = imaged[features].fillna(0)
    y = imaged[outcome].values

    pos = imaged[imaged[outcome] == 1]
    neg = imaged[imaged[outcome] == 0]
    n_pos, n_neg = len(pos), len(neg)
    print(f"  Abnormal: {n_pos}, Normal: {n_neg}")

    # Univariate ORs
    rows = []
    for f in features:
        ct = pd.crosstab(imaged[f].fillna(0).astype(int), imaged[outcome])
        if ct.shape != (2, 2):
            continue
        _, or_str, ci_str = _or_ci(ct)
        if ct.min().min() < 5:
            if ct.shape == (2, 2):
                _, p = stats.fisher_exact(ct)
            else:
                _, p, _, _ = stats.chi2_contingency(ct)
        else:
            _, p, _, _ = stats.chi2_contingency(ct)

        n_pos_val = (pos[f] == 1).sum()
        n_neg_val = (neg[f] == 1).sum()
        rows.append({
            "Variable": f,
            f"Abnormal (n={n_pos})": _fmt(n_pos_val, n_pos),
            f"Normal (n={n_neg})": _fmt(n_neg_val, n_neg),
            "OR": or_str,
            "95% CI": ci_str,
            "p": _fmt_p(p),
        })

    # Multivariate logistic regression (stepwise by univariate p<0.1)
    sig_features = [r["Variable"] for r in rows
                    if r["p"] and r["p"] != "—"
                    and (r["p"] == "<0.001" or float(r["p"]) < 0.1)]

    if len(sig_features) >= 2:
        from sklearn.preprocessing import StandardScaler
        X_mv = imaged[sig_features].fillna(0).values
        scaler = StandardScaler()
        X_mv_scaled = scaler.fit_transform(X_mv)
        lr = LogisticRegression(max_iter=1000, random_state=SEED, penalty="l2", C=1.0)
        lr.fit(X_mv_scaled, y)

        print(f"\n  Multivariate model ({len(sig_features)} features):")
        mv_ors = np.exp(lr.coef_[0])
        for feat, adjusted_or in sorted(zip(sig_features, mv_ors),
                                         key=lambda x: -abs(np.log(x[1]))):
            print(f"    {feat}: aOR={adjusted_or:.2f}")

        # CV AUROC
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        aurocs = []
        for train_idx, test_idx in cv.split(X_mv_scaled, y):
            m = LogisticRegression(max_iter=1000, random_state=SEED, penalty="l2")
            m.fit(X_mv_scaled[train_idx], y[train_idx])
            prob = m.predict_proba(X_mv_scaled[test_idx])[:, 1]
            aurocs.append(roc_auc_score(y[test_idx], prob))
        print(f"    AUROC (5-fold CV): {np.mean(aurocs):.3f} (+/- {np.std(aurocs):.3f})")

    # Repeat for MI imaging outcome
    outcome_mi = "outcome_mi_imaging"
    if outcome_mi in imaged.columns:
        y_mi = imaged[outcome_mi].values
        mi_pos = imaged[imaged[outcome_mi] == 1]
        mi_neg = imaged[imaged[outcome_mi] == 0]
        print(f"\n  MI imaging: {len(mi_pos)} positive, {len(mi_neg)} negative")

        mi_rows = []
        for f in features:
            ct = pd.crosstab(imaged[f].fillna(0).astype(int), imaged[outcome_mi])
            if ct.shape != (2, 2):
                continue
            _, or_str, ci_str = _or_ci(ct)
            mi_rows.append({
                "Variable": f,
                "OR (MI imaging)": or_str,
                "95% CI (MI imaging)": ci_str,
            })

        # Merge MI imaging ORs into main table
        mi_dict = {r["Variable"]: r for r in mi_rows}
        for r in rows:
            mi_r = mi_dict.get(r["Variable"], {})
            r["OR (MI imaging)"] = mi_r.get("OR (MI imaging)", "")
            r["95% CI (MI imaging)"] = mi_r.get("95% CI (MI imaging)", "")

    tbl = pd.DataFrame(rows)
    # Sort by p-value
    tbl["_p_sort"] = tbl["p"].apply(lambda x: 0 if x == "<0.001" else float(x) if x != "—" else 999)
    tbl = tbl.sort_values("_p_sort").drop(columns="_p_sort")

    out = TABLES / "table4_neuroimaging_predictors.csv"
    tbl.to_csv(out, index=False)
    print(f"\n  Saved {out.name} ({len(tbl)} predictors)")

    sig = tbl[(tbl["p"] == "<0.001") | (tbl["p"].apply(lambda x: float(x) < 0.05 if x not in ("<0.001", "—", "") else False))]
    print(f"  Significant (p<0.05): {len(sig)}")
    for _, r in sig.iterrows():
        print(f"    {r['Variable']}: OR={r['OR']} ({r['95% CI']}), p={r['p']}")

    return tbl


# ===================================================================
# 3. TABLE 5: NEUROIMAGING YIELD BY SYMPTOM
# ===================================================================

def table5_neuroimaging_yield(df, imaged):
    """Imaging rates and yield by presenting symptom."""
    print("\n=== Table 5: Neuroimaging Yield by Symptom ===")

    symptom_vars = [
        ("Headache", "sx_any_headache"), ("Seizure", "sx_seizure"),
        ("Collapse/syncope", "sx_collapse"), ("Visual symptoms", "sx_visual"),
        ("Altered consciousness", "sx_altered_consciousness"),
        ("Fever", "sx_fever"), ("Ataxia", "sx_ataxia"),
        ("Speech disturbance", "sx_speech"),
        ("Focal weakness", "sx_focal_weakness"),
        ("Sensory changes", "sx_sensory"),
        ("Eye movement", "sx_eye_movement"),
    ]

    n_total = len(df)
    rows = []
    for label, col in symptom_vars:
        if col not in df.columns:
            continue
        sx_df = df[df[col] == 1]
        n_sx = len(sx_df)
        n_imaged = (sx_df["any_neuroimaging"] == 1).sum()
        sx_imaged = sx_df[sx_df["any_neuroimaging"] == 1]
        n_abnormal = (sx_imaged["outcome_abnormal_imaging"] == 1).sum() if len(sx_imaged) > 0 else 0
        n_mi = (sx_imaged["outcome_mi_imaging"] == 1).sum() if len(sx_imaged) > 0 else 0
        n_mi_outcome = (sx_df[TARGET] == 1).sum()

        rows.append({
            "Symptom": label,
            "N with symptom": n_sx,
            "% of cohort": f"{100*n_sx/n_total:.1f}%",
            "Imaged": _fmt(n_imaged, n_sx),
            "Abnormal imaging": _fmt(n_abnormal, n_imaged) if n_imaged > 0 else "—",
            "MI imaging": _fmt(n_mi, n_imaged) if n_imaged > 0 else "—",
            "MI outcome": _fmt(n_mi_outcome, n_sx),
            "NNI (abnormal)": f"{n_imaged/n_abnormal:.1f}" if n_abnormal > 0 else "—",
            "NNI (MI)": f"{n_imaged/n_mi:.1f}" if n_mi > 0 else "—",
        })

    # Overall row
    n_imaged_total = len(imaged)
    n_abn = (imaged["outcome_abnormal_imaging"] == 1).sum()
    n_mi_img = (imaged["outcome_mi_imaging"] == 1).sum()
    rows.append({
        "Symptom": "Overall",
        "N with symptom": n_total,
        "% of cohort": "100%",
        "Imaged": _fmt(n_imaged_total, n_total),
        "Abnormal imaging": _fmt(n_abn, n_imaged_total),
        "MI imaging": _fmt(n_mi_img, n_imaged_total),
        "MI outcome": _fmt((df[TARGET]==1).sum(), n_total),
        "NNI (abnormal)": f"{n_imaged_total/n_abn:.1f}" if n_abn > 0 else "—",
        "NNI (MI)": f"{n_imaged_total/n_mi_img:.1f}" if n_mi_img > 0 else "—",
    })

    tbl = pd.DataFrame(rows)
    out = TABLES / "table5_neuroimaging_yield.csv"
    tbl.to_csv(out, index=False)
    print(f"  Saved {out.name}")

    for _, r in tbl.iterrows():
        print(f"  {r['Symptom']}: {r['Imaged']} imaged, {r['Abnormal imaging']} abnormal, NNI={r['NNI (MI)']}")

    return tbl


# ===================================================================
# 4. FIGURE
# ===================================================================

def make_figure(tbl5):
    """Neuroimaging yield bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

    print("\n=== Figure: Neuroimaging Yield ===")
    data = tbl5[tbl5["Symptom"] != "Overall"].copy()

    # Parse imaging rate and abnormal rate
    def parse_pct(s):
        if "—" in str(s) or not s:
            return 0
        try:
            return float(s.split("(")[1].rstrip("%)"))
        except (IndexError, ValueError):
            return 0

    data["img_rate"] = data["Imaged"].apply(parse_pct)
    data["abn_rate"] = data["Abnormal imaging"].apply(parse_pct)
    data["mi_rate"] = data["MI imaging"].apply(parse_pct)
    data = data.sort_values("img_rate", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(data))
    h = 0.25

    ax.barh(y - h, data["img_rate"].values, h, label="Imaging rate (%)", color="#1f77b4")
    ax.barh(y, data["abn_rate"].values, h, label="Abnormal result (%)", color="#ff7f0e")
    ax.barh(y + h, data["mi_rate"].values, h, label="MI cause (%)", color="#d62728")

    ax.set_yticks(y)
    ax.set_yticklabels(data["Symptom"].values)
    ax.set_xlabel("Percentage")
    ax.set_title("Neuroimaging Rates and Yield by Presenting Symptom")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_neuroimaging_yield.png")
    plt.close()
    print("  Saved fig_neuroimaging_yield.png")


# ===================================================================
# MAIN
# ===================================================================

def main():
    print("=" * 60)
    print("Brain AttACKs: Neuroimaging Predictors (Objective 3)")
    print("=" * 60)

    df, imaged = load_data()
    tbl4 = table4_neuroimaging_predictors(imaged)
    tbl5 = table5_neuroimaging_yield(df, imaged)
    make_figure(tbl5)

    print("\n" + "=" * 60)
    print("Done. Outputs in outputs/tables/ and outputs/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
