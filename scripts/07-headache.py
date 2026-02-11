"""
07-headache.py — DEPRECATED (MS1 Pipeline)

This script is DEPRECATED as of 2026-02-10.
All headache-specific analysis has moved to 13-headache-deep.py (Manuscript 2)
to avoid self-plagiarism between MS1 (J Pediatrics) and MS2 (Headache/Cephalalgia).

MS1 mentions headache as a symptom in descriptive tables but does NOT model it
separately. For the headache prediction model, imaging appropriateness, DCA,
calibration, and cost analysis, see 13-headache-deep.py.

Original purpose:
    Extended the IHC 2017 abstract analysis to the full 2-year dataset.
    Derived a prediction model for medically important causes of headache
    using Year 1 (derivation) and Year 2 (validation).

Inputs:
    data/processed/brain_attacks_combined.parquet

Outputs (no longer generated for MS1):
    outputs/tables/table6_headache_model.csv
    outputs/tables/table_headache_demographics.csv
    outputs/figures/fig_roc_headache.png
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, fbeta_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy import stats

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

# IHC 2017 predictors (replicating the abstract)
IHC_PREDICTORS = [
    "age_under_12",
    "gender",
    "ems",
    "exam_altered_loc",
    "sx_focal_weakness",
    "sx_ataxia",
    "sx_speech",
    "sx_visual",
    "ha_progressive_days",
    "ha_vomiting",
    "sx_fever",
]

# Extended predictors (additional headache features)
EXTENDED_PREDICTORS = IHC_PREDICTORS + [
    "ha_photophobia",
    "ha_phonophobia",
    "ha_maximal_onset",
    "ha_abrupt_onset",
    "ha_gradual_onset",
    "ha_nausea",
    "exam_papilledema",
    "exam_meningismus",
    "ctas",
    "any_prev_dx",
]


def _fmt(n, total):
    pct = 100 * n / total if total > 0 else 0
    return f"{int(n)} ({pct:.1f}%)"


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

    # Headache patients
    ha = df[df["sx_any_headache"] == 1].copy()
    print(f"Headache patients: {len(ha)}")

    # Exclude prior headache disorder (migraine=4, recurring headache=5)
    prior_ha = ((ha.get("previous_diagnoses___4", 0) == 1) |
                (ha.get("previous_diagnoses___5", 0) == 1))
    n_excluded = prior_ha.sum()
    ha_clean = ha[~prior_ha].copy()
    print(f"  Excluded prior headache disorder: {n_excluded}")
    print(f"  Analysis cohort: {len(ha_clean)}")

    mi = (ha_clean[TARGET] == 1).sum()
    print(f"  MI: {mi} ({100*mi/len(ha_clean):.1f}%)")

    y1 = ha_clean[ha_clean["year"] == 1]
    y2 = ha_clean[ha_clean["year"] == 2]
    print(f"  Year 1: {len(y1)} ({(y1[TARGET]==1).sum()} MI)")
    print(f"  Year 2: {len(y2)} ({(y2[TARGET]==1).sum()} MI)")

    return df, ha_clean, y1, y2


# ===================================================================
# 2. HEADACHE DEMOGRAPHICS TABLE
# ===================================================================

def headache_demographics(ha):
    """Demographics of headache sub-cohort."""
    print("\n=== Headache Demographics ===")

    mi = ha[ha[TARGET] == 1]
    bn = ha[ha[TARGET] == 0]
    n_all, n_mi, n_bn = len(ha), len(mi), len(bn)

    rows = []

    # Age
    rows.append({
        "Variable": "Age, years, mean (SD)",
        "Overall": f"{ha['age'].mean():.1f} ({ha['age'].std():.1f})",
        f"MI (n={n_mi})": f"{mi['age'].mean():.1f} ({mi['age'].std():.1f})",
        f"Benign (n={n_bn})": f"{bn['age'].mean():.1f} ({bn['age'].std():.1f})",
    })

    # Female
    rows.append({
        "Variable": "Female, n (%)",
        "Overall": _fmt((ha["gender"] == 2).sum(), n_all),
        f"MI (n={n_mi})": _fmt((mi["gender"] == 2).sum(), n_mi),
        f"Benign (n={n_bn})": _fmt((bn["gender"] == 2).sum(), n_bn),
    })

    # Key predictors
    for label, col in [
        ("EMS transport", "ems"),
        ("Altered consciousness (exam)", "exam_altered_loc"),
        ("Focal weakness", "sx_focal_weakness"),
        ("Ataxia", "sx_ataxia"),
        ("Fever", "sx_fever"),
        ("Progressive headache (days)", "ha_progressive_days"),
        ("Vomiting", "ha_vomiting"),
        ("Photophobia", "ha_photophobia"),
        ("Papilledema", "exam_papilledema"),
        ("Meningismus", "exam_meningismus"),
    ]:
        if col not in ha.columns:
            continue
        rows.append({
            "Variable": f"{label}, n (%)",
            "Overall": _fmt((ha[col] == 1).sum(), n_all),
            f"MI (n={n_mi})": _fmt((mi[col] == 1).sum(), n_mi),
            f"Benign (n={n_bn})": _fmt((bn[col] == 1).sum(), n_bn),
        })

    tbl = pd.DataFrame(rows)
    out = TABLES / "table_headache_demographics.csv"
    tbl.to_csv(out, index=False)
    print(f"  Saved {out.name}")
    return tbl


# ===================================================================
# 3. MODEL DERIVATION & VALIDATION
# ===================================================================

def _eval_model(model, X, y, label=""):
    """Evaluate model and return metrics dict + probabilities."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    auroc = roc_auc_score(y, y_prob)
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    result = {
        "accuracy": accuracy_score(y, y_pred),
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "auroc": auroc,
        "fnr": fnr,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }

    if label:
        print(f"\n  {label}:")
        for k in ["accuracy", "sensitivity", "specificity", "ppv", "npv", "auroc", "fnr"]:
            print(f"    {k}: {100*result[k]:.1f}%")
        print(f"    Confusion: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    return result, y_prob


def headache_model(ha, y1, y2):
    """Derive and validate headache prediction model."""
    print("\n=== Headache Prediction Model ===")

    features = [f for f in EXTENDED_PREDICTORS if f in ha.columns]
    print(f"  Features: {len(features)}")

    # Prepare data
    X1 = y1[features].fillna(0).values
    X2 = y2[features].fillna(0).values
    y1_out = y1[TARGET].values
    y2_out = y2[TARGET].values

    # Univariate associations on derivation set
    print("\n  Univariate associations (Year 1):")
    uv_rows = []
    for i, f in enumerate(features):
        pos = y1[y1[TARGET] == 1]
        neg = y1[y1[TARGET] == 0]
        ct = pd.crosstab(y1[f].fillna(0).astype(int), y1[TARGET])
        if ct.shape != (2, 2):
            continue
        a, b = ct.iloc[1, 1], ct.iloc[1, 0]
        c, d = ct.iloc[0, 1], ct.iloc[0, 0]
        if any(v == 0 for v in [a, b, c, d]):
            or_str, ci_str = "—", ""
        else:
            oratio = (a * d) / (b * c)
            se = np.sqrt(1/a + 1/b + 1/c + 1/d)
            lo = np.exp(np.log(oratio) - 1.96 * se)
            hi = np.exp(np.log(oratio) + 1.96 * se)
            or_str = f"{oratio:.2f}"
            ci_str = f"{lo:.2f}-{hi:.2f}"

        if ct.min().min() < 5:
            _, p = stats.fisher_exact(ct)
        else:
            _, p, _, _ = stats.chi2_contingency(ct)

        uv_rows.append({
            "Variable": f,
            "MI n (%)": _fmt((pos[f] == 1).sum(), len(pos)),
            "Benign n (%)": _fmt((neg[f] == 1).sum(), len(neg)),
            "OR": or_str,
            "95% CI": ci_str,
            "p": _fmt_p(p),
        })

    # Print significant
    for r in uv_rows:
        if r["p"] == "<0.001" or (r["p"] != "—" and float(r["p"]) < 0.05):
            print(f"    {r['Variable']}: OR={r['OR']} ({r['95% CI']}), p={r['p']}")

    # --- Derivation: Stepwise logistic regression ---
    # Select features with p < 0.1 from univariate
    sig_features = []
    sig_indices = []
    for r in uv_rows:
        if r["p"] == "<0.001" or (r["p"] != "—" and float(r["p"]) < 0.1):
            idx = features.index(r["Variable"])
            sig_features.append(r["Variable"])
            sig_indices.append(idx)

    if len(sig_features) < 2:
        print("  WARNING: Fewer than 2 significant features, using all")
        sig_features = features
        sig_indices = list(range(len(features)))

    print(f"\n  Selected features (p<0.1): {sig_features}")

    X1_sel = X1[:, sig_indices]
    X2_sel = X2[:, sig_indices]

    scaler = StandardScaler()
    X1_scaled = scaler.fit_transform(X1_sel)
    X2_scaled = scaler.transform(X2_sel)

    # Fit derivation model
    model = LogisticRegression(max_iter=1000, random_state=SEED, penalty="l2", C=1.0)
    model.fit(X1_scaled, y1_out)

    # Print coefficients
    print("\n  Model coefficients (derivation):")
    coefs = pd.Series(model.coef_[0], index=sig_features)
    mv_rows = []
    for f, c in coefs.sort_values(ascending=False).items():
        or_val = np.exp(c)
        print(f"    {f}: beta={c:.3f}, aOR={or_val:.2f}")
        mv_rows.append({
            "Variable": f,
            "Beta": f"{c:.3f}",
            "Adjusted OR": f"{or_val:.2f}",
        })

    # --- Evaluate ---
    deriv_result, y1_prob = _eval_model(model, X1_scaled, y1_out, "Derivation (Year 1)")
    valid_result, y2_prob = _eval_model(model, X2_scaled, y2_out, "Validation (Year 2)")

    # --- Pooled 2-year model for comparison ---
    X_all = np.vstack([X1_sel, X2_sel])
    y_all = np.concatenate([y1_out, y2_out])
    X_all_scaled = StandardScaler().fit_transform(X_all)

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    pooled_aurocs = []
    for train_idx, test_idx in cv.split(X_all_scaled, y_all):
        m = LogisticRegression(max_iter=1000, random_state=SEED, penalty="l2")
        m.fit(X_all_scaled[train_idx], y_all[train_idx])
        prob = m.predict_proba(X_all_scaled[test_idx])[:, 1]
        pooled_aurocs.append(roc_auc_score(y_all[test_idx], prob))
    print(f"\n  Pooled 2-year AUROC (10-fold CV): {np.mean(pooled_aurocs):.3f} "
          f"(+/- {np.std(pooled_aurocs):.3f})")

    # --- IHC 2017 comparison ---
    print("\n  Comparison with IHC 2017:")
    print("    IHC 2017 (Y1 only, 66/33 split): 94.5% accuracy, 33.3% FNR")
    print(f"    Current derivation (Y1 full): {100*deriv_result['accuracy']:.1f}% accuracy, "
          f"{100*deriv_result['fnr']:.1f}% FNR")
    print(f"    Current validation (Y2): {100*valid_result['accuracy']:.1f}% accuracy, "
          f"{100*valid_result['fnr']:.1f}% FNR")

    # --- Neuroimaging impact ---
    print("\n  Neuroimaging impact estimate:")
    ha_all = pd.concat([y1, y2])
    n_ha_imaged = (ha_all["any_neuroimaging"] == 1).sum()
    n_ha_total = len(ha_all)
    current_rate = n_ha_imaged / n_ha_total

    # If we used the model to screen, how many would we image?
    X_all_for_pred = np.vstack([X1_scaled, X2_scaled])
    y_all_prob = model.predict_proba(X_all_for_pred)[:, 1]
    # At default threshold
    n_flagged = (y_all_prob >= 0.5).sum()
    # At high-sensitivity threshold (flag top 20% by probability)
    thresh_80 = np.percentile(y_all_prob, 80)
    n_flagged_80 = (y_all_prob >= thresh_80).sum()

    print(f"    Current imaging rate: {_fmt(n_ha_imaged, n_ha_total)}")
    print(f"    Model-flagged (p>=0.5): {_fmt(n_flagged, n_ha_total)}")
    print(f"    Model-flagged (top 20%): {_fmt(n_flagged_80, n_ha_total)}")

    # --- Save table ---
    result_rows = []
    for metric in ["accuracy", "sensitivity", "specificity", "ppv", "npv", "auroc", "fnr"]:
        result_rows.append({
            "Metric": metric.upper() if metric in ("ppv", "npv", "fnr", "auroc") else metric.capitalize(),
            "Derivation (Y1)": f"{100*deriv_result[metric]:.1f}%",
            "Validation (Y2)": f"{100*valid_result[metric]:.1f}%",
            "Pooled (2-year CV)": f"{100*np.mean(pooled_aurocs):.1f}%" if metric == "auroc" else "—",
        })
    result_rows.append({
        "Metric": "IHC 2017 comparison",
        "Derivation (Y1)": f"Accuracy {100*deriv_result['accuracy']:.1f}% (vs 94.5%)",
        "Validation (Y2)": f"FNR {100*valid_result['fnr']:.1f}% (vs 33.3%)",
        "Pooled (2-year CV)": "—",
    })

    tbl = pd.DataFrame(result_rows)
    out = TABLES / "table6_headache_model.csv"
    tbl.to_csv(out, index=False)
    print(f"\n  Saved {out.name}")

    return deriv_result, valid_result, y1_out, y1_prob, y2_out, y2_prob


# ===================================================================
# 4. FIGURE
# ===================================================================

def make_roc_figure(y1_out, y1_prob, y2_out, y2_prob):
    """ROC curves for headache model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

    print("\n=== Figure 6: Headache ROC ===")

    fig, ax = plt.subplots(figsize=(7, 6))

    fpr, tpr, _ = roc_curve(y1_out, y1_prob)
    auc1 = roc_auc_score(y1_out, y1_prob)
    ax.plot(fpr, tpr, label=f"Derivation Y1 (AUC={auc1:.3f})", color="#1f77b4", linewidth=2)

    fpr, tpr, _ = roc_curve(y2_out, y2_prob)
    auc2 = roc_auc_score(y2_out, y2_prob)
    ax.plot(fpr, tpr, label=f"Validation Y2 (AUC={auc2:.3f})", color="#d62728", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("Headache Model: Derivation vs Validation")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_roc_headache.png")
    plt.close()
    print("  Saved fig_roc_headache.png")


# ===================================================================
# MAIN
# ===================================================================

def main():
    print("=" * 60)
    print("07-headache.py is DEPRECATED")
    print("=" * 60)
    print()
    print("This script has been deprecated to avoid self-plagiarism between")
    print("MS1 (J Pediatrics) and MS2 (Headache/Cephalalgia).")
    print()
    print("All headache analysis has moved to: scripts/13-headache-deep.py")
    print()
    print("To run the headache analysis for MS2:")
    print("  python scripts/13-headache-deep.py")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
