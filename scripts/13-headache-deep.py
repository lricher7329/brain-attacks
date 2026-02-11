"""
13-headache-deep.py — Extended Headache Analysis (Manuscript 2)

Self-contained analysis for the companion headache paper (target: Headache or
Cephalalgia). This script is the SOLE headache analysis for the two-manuscript
strategy; script 07-headache.py is deprecated for MS1.

Study design:
    Retrospective sub-cohort of children presenting to the Stollery Children's
    Hospital ED with non-traumatic acute brain dysfunction AND headache as a
    presenting symptom (sx_any_headache == 1). Children with prior known
    headache disorder (migraine, recurring headache) are excluded.

Cohort assembly:
    Full Brain AttACKs cohort (N=2,153) → headache sub-cohort → exclude prior
    headache disorder → analysis cohort. Temporal split: Year 1 derivation,
    Year 2 validation.

Outcome:
    Binary: medically important (MI) brain attack vs medically unimportant.
    Same definition as MS1 (11 MI categories per protocol PRO00055264).

Predictors:
    21 extended predictors: 11 from IHC 2017 analysis (age, sex, EMS, exam
    findings, symptoms) + 10 headache-specific features (photophobia,
    phonophobia, onset type, nausea, papilledema, meningismus, CTAS,
    prior diagnoses).

Missing data:
    Binary predictors: missing treated as absent (fillna(0)), consistent with
    REDCap branching logic where unchecked = absent.

Statistical methods:
    - Logistic regression (L2, C=1.0) and XGBoost with SMOTE oversampling
    - Temporal validation: Year 1 derivation → Year 2 validation
    - 3-fold inner CV for XGBoost hyperparameter tuning
    - Bootstrap 95% CIs (1000 iterations) for AUROC
    - Calibration: CITL, calibration slope, Brier score
    - Clinical utility: decision curve analysis (DCA)
    - Cost analysis: investigation costs from CADTH 2023 estimates

Reporting standards:
    - TRIPOD+AI (Collins et al., BMJ 2024) [Zotero: P6YDK85N]
    - ICHD-3 (IHS, Cephalalgia 2018) for headache classification [Zotero: EEV89RYF]
    - ACR Appropriateness Criteria Headache-Child (Hayes et al., 2018) for
      ni_appropriateness field [Zotero: HXY9U8RR]

Inputs:
    data/processed/brain_attacks_combined.parquet

Outputs:
    outputs/tables/table_ha_demographics_extended.csv
    outputs/tables/table_ha_appropriateness.csv
    outputs/tables/table_ha_model_extended.csv
    outputs/tables/table_ha_decision_curve.csv
    outputs/tables/table_ha_cost.csv
    outputs/figures/fig_ha_appropriateness.png
    outputs/figures/fig_ha_roc_extended.png
    outputs/figures/fig_ha_calibration.png
    outputs/figures/fig_ha_decision_curve.png
    outputs/figures/fig_ha_cost_threshold.png
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix,
    fbeta_score, make_scorer, roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy import stats
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

SEED = 42
N_BOOTSTRAP = 1000
np.random.seed(SEED)

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data" / "processed"
TABLES = PROJECT / "outputs" / "tables"
FIGURES = PROJECT / "outputs" / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

TARGET = "outcome_med_important"


def _print(msg=""):
    print(msg, flush=True)


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
# PREDICTOR DEFINITIONS (from 07-headache.py)
# ===================================================================

IHC_PREDICTORS = [
    "age_under_12", "gender", "ems", "exam_altered_loc",
    "sx_focal_weakness", "sx_ataxia", "sx_speech", "sx_visual",
    "ha_progressive_days", "ha_vomiting", "sx_fever",
]

EXTENDED_PREDICTORS = IHC_PREDICTORS + [
    "ha_photophobia", "ha_phonophobia", "ha_maximal_onset",
    "ha_abrupt_onset", "ha_gradual_onset", "ha_nausea",
    "exam_papilledema", "exam_meningismus", "ctas", "any_prev_dx",
]

# Investigation columns for cost analysis
INV_COL_MAP = {
    "inv_ct": "ct_head",
    "inv_mri": "mri_brain",
    "inv_cta": "cta_head",
    "inv_mra": "mra_head",
    "inv_lp": "lp",
    "inv_eeg": "eeg",
}

# Cost parameters — same published estimates as script 12
# Sources: CADTH 2023 (CT/MRI), estimated (CTA/MRA/LP/EEG)
# See 12-cost-analysis.py for full source documentation.
COST_PARAMS = {
    "ct_head": {"base": 95.0, "mid": 310.0, "high": 490.0},
    "mri_brain": {"base": 230.0, "mid": 762.0, "high": 990.0},
    "cta_head": {"base": 140.0, "mid": 465.0, "high": 670.0},
    "mra_head": {"base": 300.0, "mid": 990.0, "high": 1300.0},
    "lp": {"base": 80.0, "mid": 200.0, "high": 350.0},
    "eeg": {"base": 80.0, "mid": 250.0, "high": 400.0},
}


# ===================================================================
# 1. LOAD & SUBSET
# ===================================================================

def load_data():
    df = pd.read_parquet(DATA / "brain_attacks_combined.parquet")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    _print(f"Full cohort: {len(df)}")

    # Headache patients
    ha = df[df["sx_any_headache"] == 1].copy()
    _print(f"Headache patients: {len(ha)}")

    # Exclude prior headache disorder (migraine=4, recurring headache=5)
    prior_ha = ((ha.get("previous_diagnoses___4", pd.Series(0, index=ha.index)) == 1) |
                (ha.get("previous_diagnoses___5", pd.Series(0, index=ha.index)) == 1))
    n_excluded = prior_ha.sum()
    ha_clean = ha[~prior_ha].copy()
    _print(f"  Excluded prior headache disorder: {n_excluded}")
    _print(f"  Analysis cohort: {len(ha_clean)}")

    mi = (ha_clean[TARGET] == 1).sum()
    _print(f"  MI: {mi} ({100*mi/len(ha_clean):.1f}%)")

    y1 = ha_clean[ha_clean["year"] == 1]
    y2 = ha_clean[ha_clean["year"] == 2]
    _print(f"  Year 1: {len(y1)} ({(y1[TARGET]==1).sum()} MI)")
    _print(f"  Year 2: {len(y2)} ({(y2[TARGET]==1).sum()} MI)")

    return df, ha_clean, y1, y2


# ===================================================================
# 2. EXTENDED DEMOGRAPHICS TABLE
# ===================================================================

def extended_demographics(ha, y1, y2):
    """Full headache cohort characteristics including investigations."""
    _print("\n=== Extended Headache Demographics ===")

    mi = ha[ha[TARGET] == 1]
    bn = ha[ha[TARGET] == 0]
    n_all, n_mi, n_bn = len(ha), len(mi), len(bn)

    rows = []

    # N
    rows.append({"Variable": "N", "Overall": str(n_all),
                 f"MI (n={n_mi})": str(n_mi), f"Benign (n={n_bn})": str(n_bn), "p": ""})

    # Age
    _, p_age = stats.ttest_ind(mi["age"].dropna(), bn["age"].dropna())
    rows.append({
        "Variable": "Age, years, mean (SD)",
        "Overall": f"{ha['age'].mean():.1f} ({ha['age'].std():.1f})",
        f"MI (n={n_mi})": f"{mi['age'].mean():.1f} ({mi['age'].std():.1f})",
        f"Benign (n={n_bn})": f"{bn['age'].mean():.1f} ({bn['age'].std():.1f})",
        "p": _fmt_p(p_age),
    })

    # Age < 12
    ct = pd.crosstab(ha["age_under_12"].fillna(0), ha[TARGET])
    _, p = stats.chi2_contingency(ct)[:2] if ct.shape == (2, 2) else (0, 1.0)
    rows.append({
        "Variable": "Age < 12, n (%)",
        "Overall": _fmt((ha["age_under_12"] == 1).sum(), n_all),
        f"MI (n={n_mi})": _fmt((mi["age_under_12"] == 1).sum(), n_mi),
        f"Benign (n={n_bn})": _fmt((bn["age_under_12"] == 1).sum(), n_bn),
        "p": _fmt_p(p),
    })

    # Female
    ct = pd.crosstab((ha["gender"] == 2).astype(int), ha[TARGET])
    _, p = stats.chi2_contingency(ct)[:2] if ct.shape == (2, 2) else (0, 1.0)
    rows.append({
        "Variable": "Female, n (%)",
        "Overall": _fmt((ha["gender"] == 2).sum(), n_all),
        f"MI (n={n_mi})": _fmt((mi["gender"] == 2).sum(), n_mi),
        f"Benign (n={n_bn})": _fmt((bn["gender"] == 2).sum(), n_bn),
        "p": _fmt_p(p),
    })

    # Clinical features
    clinical_vars = [
        ("EMS transport", "ems"),
        ("Altered consciousness (exam)", "exam_altered_loc"),
        ("Focal weakness", "sx_focal_weakness"),
        ("Ataxia", "sx_ataxia"),
        ("Speech impairment", "sx_speech"),
        ("Visual impairment", "sx_visual"),
        ("Fever", "sx_fever"),
        ("Seizure", "sx_seizure"),
        ("Papilledema", "exam_papilledema"),
        ("Meningismus", "exam_meningismus"),
        ("Progressive headache (days)", "ha_progressive_days"),
        ("Vomiting", "ha_vomiting"),
        ("Nausea", "ha_nausea"),
        ("Photophobia", "ha_photophobia"),
        ("Phonophobia", "ha_phonophobia"),
        ("Maximal at onset", "ha_maximal_onset"),
        ("Abrupt onset", "ha_abrupt_onset"),
        ("Gradual onset", "ha_gradual_onset"),
    ]

    for label, col in clinical_vars:
        if col not in ha.columns:
            continue
        ct = pd.crosstab(ha[col].fillna(0).astype(int), ha[TARGET])
        if ct.shape == (2, 2):
            if ct.min().min() < 5:
                _, p = stats.fisher_exact(ct)
            else:
                _, p, _, _ = stats.chi2_contingency(ct)
        else:
            p = 1.0
        rows.append({
            "Variable": f"{label}, n (%)",
            "Overall": _fmt((ha[col] == 1).sum(), n_all),
            f"MI (n={n_mi})": _fmt((mi[col] == 1).sum(), n_mi),
            f"Benign (n={n_bn})": _fmt((bn[col] == 1).sum(), n_bn),
            "p": _fmt_p(p),
        })

    # Investigation rates
    inv_vars = [
        ("Any neuroimaging", "any_neuroimaging"),
        ("CT head", "inv_ct"),
        ("MRI brain", "inv_mri"),
        ("CTA", "inv_cta"),
        ("MRA", "inv_mra"),
        ("LP", "inv_lp"),
        ("EEG", "inv_eeg"),
    ]
    for label, col in inv_vars:
        if col not in ha.columns:
            continue
        ct = pd.crosstab(ha[col].fillna(0).astype(int), ha[TARGET])
        if ct.shape == (2, 2):
            if ct.min().min() < 5:
                _, p = stats.fisher_exact(ct)
            else:
                _, p, _, _ = stats.chi2_contingency(ct)
        else:
            p = 1.0
        rows.append({
            "Variable": f"{label}, n (%)",
            "Overall": _fmt((ha[col] == 1).sum(), n_all),
            f"MI (n={n_mi})": _fmt((mi[col] == 1).sum(), n_mi),
            f"Benign (n={n_bn})": _fmt((bn[col] == 1).sum(), n_bn),
            "p": _fmt_p(p),
        })

    # Consultation rate (Year 2 only)
    ha_y2 = ha[ha["year"] == 2]
    if "consult" in ha_y2.columns and len(ha_y2) > 0:
        n_consult = (ha_y2["consult"] == 1).sum()
        rows.append({
            "Variable": f"Consultation (Y2 only, n={len(ha_y2)}), n (%)",
            "Overall": _fmt(n_consult, len(ha_y2)),
            f"MI (n={n_mi})": "—",
            f"Benign (n={n_bn})": "—",
            "p": "",
        })

    # MI subtype breakdown
    rows.append({"Variable": "--- MI Subtype Breakdown ---", "Overall": "",
                 f"MI (n={n_mi})": "", f"Benign (n={n_bn})": "", "p": ""})

    mi_codes = {
        7: "Brain/meningeal infection",
        13: "Cerebellar ataxia",
        9: "Neoplasm",
        6: "Demyelinating disorder",
        8: "Hydrocephalus",
        12: "IIH",
        1: "Stroke",
        2: "ICH",
    }
    for code, name in mi_codes.items():
        col = f"med_imp_dx___{code}"
        if col in mi.columns:
            n = (mi[col] == 1).sum()
            if n > 0:
                rows.append({
                    "Variable": f"  {name}",
                    "Overall": "",
                    f"MI (n={n_mi})": _fmt(n, n_mi),
                    f"Benign (n={n_bn})": "",
                    "p": "",
                })

    # Year split
    rows.append({
        "Variable": "Year 1 / Year 2",
        "Overall": f"{len(y1)} / {len(y2)}",
        f"MI (n={n_mi})": f"{(y1[TARGET]==1).sum()} / {(y2[TARGET]==1).sum()}",
        f"Benign (n={n_bn})": f"{(y1[TARGET]==0).sum()} / {(y2[TARGET]==0).sum()}",
        "p": "",
    })

    tbl = pd.DataFrame(rows)
    tbl.to_csv(TABLES / "table_ha_demographics_extended.csv", index=False)
    _print(f"  Saved table_ha_demographics_extended.csv ({len(tbl)} rows)")
    return tbl


# ===================================================================
# 3. IMAGING APPROPRIATENESS ANALYSIS
# ===================================================================

def appropriateness_analysis(ha):
    """Cross-tabulate ni_appropriateness against imaging and outcomes."""
    _print("\n=== Imaging Appropriateness Analysis ===")

    if "ni_appropriateness" not in ha.columns:
        _print("  WARNING: ni_appropriateness not in dataset, skipping")
        return None

    # ni_appropriateness rated per ACR Appropriateness Criteria Headache-Child
    # (Hayes et al., JACR 2018) [Zotero: HXY9U8RR]:
    # 1=Usually NOT appropriate, 2=May be appropriate, 3=Usually appropriate
    ha_app = ha[ha["ni_appropriateness"].notna()].copy()
    ha_app["ni_appropriateness"] = ha_app["ni_appropriateness"].astype(int)
    _print(f"  Patients with appropriateness rating: {len(ha_app)}")

    app_labels = {1: "Usually NOT appropriate", 2: "May be appropriate", 3: "Usually appropriate"}

    rows = []
    for code in [1, 2, 3]:
        subset = ha_app[ha_app["ni_appropriateness"] == code]
        n = len(subset)
        if n == 0:
            continue

        n_imaged = (subset["any_neuroimaging"] == 1).sum() if "any_neuroimaging" in subset.columns else 0
        n_abnormal = (subset["outcome_abnormal_imaging"] == 1).sum() if "outcome_abnormal_imaging" in subset.columns else 0
        n_mi = (subset[TARGET] == 1).sum()

        rows.append({
            "Appropriateness": app_labels[code],
            "Code": code,
            "N": n,
            "N (%)": _fmt(n, len(ha_app)),
            "Imaged, n (%)": _fmt(n_imaged, n),
            "Abnormal imaging, n (%)": _fmt(n_abnormal, n) if n_imaged > 0 else "0 (0%)",
            "MI, n (%)": _fmt(n_mi, n),
            "Imaging rate": f"{100*n_imaged/n:.1f}%" if n > 0 else "—",
        })

    # Overtesting: imaged when "Usually NOT appropriate"
    not_app = ha_app[ha_app["ni_appropriateness"] == 1]
    if len(not_app) > 0 and "any_neuroimaging" in not_app.columns:
        n_overtested = (not_app["any_neuroimaging"] == 1).sum()
        _print(f"  Overtesting: {n_overtested}/{len(not_app)} "
               f"({100*n_overtested/len(not_app):.1f}%) imaged when 'Usually NOT appropriate'")

    # Undertesting: NOT imaged when "Usually appropriate"
    app = ha_app[ha_app["ni_appropriateness"] == 3]
    if len(app) > 0 and "any_neuroimaging" in app.columns:
        n_undertested = (app["any_neuroimaging"] == 0).sum()
        _print(f"  Undertesting: {n_undertested}/{len(app)} "
               f"({100*n_undertested/len(app):.1f}%) NOT imaged when 'Usually appropriate'")

    # Chi-squared: appropriateness vs imaging
    if "any_neuroimaging" in ha_app.columns:
        ct = pd.crosstab(ha_app["ni_appropriateness"], ha_app["any_neuroimaging"])
        if ct.shape[0] >= 2 and ct.shape[1] >= 2:
            chi2, p_chi, _, _ = stats.chi2_contingency(ct)
            _print(f"  Chi-squared (appropriateness vs imaging): χ²={chi2:.1f}, p={_fmt_p(p_chi)}")
            rows.append({
                "Appropriateness": "Chi-squared test",
                "Code": "",
                "N": "",
                "N (%)": "",
                "Imaged, n (%)": f"χ²={chi2:.1f}",
                "Abnormal imaging, n (%)": f"p={_fmt_p(p_chi)}",
                "MI, n (%)": "",
                "Imaging rate": "",
            })

    # Chi-squared: appropriateness vs MI
    ct_mi = pd.crosstab(ha_app["ni_appropriateness"], ha_app[TARGET])
    if ct_mi.shape[0] >= 2 and ct_mi.shape[1] >= 2:
        chi2_mi, p_mi, _, _ = stats.chi2_contingency(ct_mi)
        _print(f"  Chi-squared (appropriateness vs MI): χ²={chi2_mi:.1f}, p={_fmt_p(p_mi)}")

    tbl = pd.DataFrame(rows)
    tbl.to_csv(TABLES / "table_ha_appropriateness.csv", index=False)
    _print(f"  Saved table_ha_appropriateness.csv")
    return tbl


def make_appropriateness_figure(ha):
    """Imaging rates by appropriateness category."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

    if "ni_appropriateness" not in ha.columns:
        return

    ha_app = ha[ha["ni_appropriateness"].notna()].copy()
    ha_app["ni_appropriateness"] = ha_app["ni_appropriateness"].astype(int)

    app_labels = {1: "Usually NOT\nappropriate", 2: "May be\nappropriate", 3: "Usually\nappropriate"}
    categories = [1, 2, 3]
    labels = [app_labels[c] for c in categories]

    imaging_rates = []
    abnormal_rates = []
    mi_rates = []
    ns = []

    for code in categories:
        subset = ha_app[ha_app["ni_appropriateness"] == code]
        n = len(subset)
        ns.append(n)
        n_imaged = (subset["any_neuroimaging"] == 1).sum() if "any_neuroimaging" in subset.columns else 0
        n_abnormal = (subset["outcome_abnormal_imaging"] == 1).sum() if "outcome_abnormal_imaging" in subset.columns else 0
        n_mi = (subset[TARGET] == 1).sum()
        imaging_rates.append(100 * n_imaged / n if n > 0 else 0)
        abnormal_rates.append(100 * n_abnormal / n if n > 0 else 0)
        mi_rates.append(100 * n_mi / n if n > 0 else 0)

    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width, imaging_rates, width, label="Imaged", color="#1f77b4")
    bars2 = ax.bar(x, abnormal_rates, width, label="Abnormal imaging", color="#ff7f0e")
    bars3 = ax.bar(x + width, mi_rates, width, label="MI", color="#d62728")

    ax.set_xlabel("Neuroimaging Appropriateness Rating")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Headache Cohort: Imaging and Outcomes by Appropriateness")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(labels, ns)])
    ax.legend()
    ax.grid(alpha=0.2, axis="y")

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                        f"{h:.0f}%", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig_ha_appropriateness.png")
    plt.close()
    _print("  Saved fig_ha_appropriateness.png")


# ===================================================================
# 4. EXTENDED PREDICTION MODEL (LR + XGBoost)
# ===================================================================

def bootstrap_ci(y_true, y_prob, n_boot=N_BOOTSTRAP, seed=SEED):
    """Bootstrap 95% CIs for AUROC."""
    rng = np.random.RandomState(seed)
    aurocs = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            aurocs.append(roc_auc_score(y_true[idx], y_prob[idx]))
        except ValueError:
            continue
    if len(aurocs) == 0:
        return {"mean": 0, "ci_lower": 0, "ci_upper": 0}
    arr = np.array(aurocs)
    return {
        "mean": float(np.mean(arr)),
        "ci_lower": float(np.percentile(arr, 2.5)),
        "ci_upper": float(np.percentile(arr, 97.5)),
    }


def extended_model(ha, y1, y2):
    """Derive and validate extended headache model (LR + XGBoost)."""
    _print("\n=== Extended Prediction Model ===")

    features = [f for f in EXTENDED_PREDICTORS if f in ha.columns]
    _print(f"  Features: {len(features)} — {features}")

    X1 = y1[features].fillna(0).values
    X2 = y2[features].fillna(0).values
    y1_out = y1[TARGET].values
    y2_out = y2[TARGET].values

    # --- Logistic Regression ---
    _print("\n  --- Logistic Regression ---")
    scaler = StandardScaler()
    X1_scaled = scaler.fit_transform(X1)
    X2_scaled = scaler.transform(X2)

    lr = LogisticRegression(max_iter=1000, random_state=SEED, penalty="l2", C=1.0)
    lr.fit(X1_scaled, y1_out)

    lr_deriv_prob = lr.predict_proba(X1_scaled)[:, 1]
    lr_valid_prob = lr.predict_proba(X2_scaled)[:, 1]
    lr_deriv_pred = lr.predict(X1_scaled)
    lr_valid_pred = lr.predict(X2_scaled)

    lr_deriv_auroc = roc_auc_score(y1_out, lr_deriv_prob)
    lr_valid_auroc = roc_auc_score(y2_out, lr_valid_prob)

    lr_deriv_ci = bootstrap_ci(y1_out, lr_deriv_prob)
    lr_valid_ci = bootstrap_ci(y2_out, lr_valid_prob)

    _print(f"  LR Derivation AUROC: {lr_deriv_auroc:.3f} "
           f"({lr_deriv_ci['ci_lower']:.3f}-{lr_deriv_ci['ci_upper']:.3f})")
    _print(f"  LR Validation AUROC: {lr_valid_auroc:.3f} "
           f"({lr_valid_ci['ci_lower']:.3f}-{lr_valid_ci['ci_upper']:.3f})")

    # Coefficients
    coefs = pd.Series(lr.coef_[0], index=features)
    _print("\n  LR Coefficients:")
    coef_rows = []
    for f, c in coefs.sort_values(ascending=False).items():
        or_val = np.exp(c)
        _print(f"    {f}: beta={c:.3f}, aOR={or_val:.2f}")
        coef_rows.append({"Variable": f, "Beta": f"{c:.3f}", "aOR": f"{or_val:.2f}"})

    # --- XGBoost ---
    _print("\n  --- XGBoost ---")
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [3, 5],
        "learning_rate": [0.01, 0.1],
    }
    f2_scorer = make_scorer(fbeta_score, beta=2, zero_division=0)

    # SMOTE on derivation (headache MI is ~5%, very imbalanced)
    sm = SMOTE(sampling_strategy=0.5, random_state=SEED)
    try:
        X1_sm, y1_sm = sm.fit_resample(X1, y1_out)
    except ValueError:
        _print("  WARNING: SMOTE failed (too few MI), using original data")
        X1_sm, y1_sm = X1, y1_out

    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                        random_state=SEED, verbosity=0)
    grid = GridSearchCV(xgb, param_grid, scoring=f2_scorer,
                        cv=inner_cv, n_jobs=-1, refit=True)
    grid.fit(X1_sm, y1_sm)

    xgb_deriv_prob = grid.predict_proba(X1)[:, 1]
    xgb_valid_prob = grid.predict_proba(X2)[:, 1]
    xgb_deriv_pred = grid.predict(X1)
    xgb_valid_pred = grid.predict(X2)

    xgb_deriv_auroc = roc_auc_score(y1_out, xgb_deriv_prob)
    xgb_valid_auroc = roc_auc_score(y2_out, xgb_valid_prob)

    xgb_deriv_ci = bootstrap_ci(y1_out, xgb_deriv_prob)
    xgb_valid_ci = bootstrap_ci(y2_out, xgb_valid_prob)

    _print(f"  XGBoost best params: {grid.best_params_}")
    _print(f"  XGBoost Derivation AUROC: {xgb_deriv_auroc:.3f} "
           f"({xgb_deriv_ci['ci_lower']:.3f}-{xgb_deriv_ci['ci_upper']:.3f})")
    _print(f"  XGBoost Validation AUROC: {xgb_valid_auroc:.3f} "
           f"({xgb_valid_ci['ci_lower']:.3f}-{xgb_valid_ci['ci_upper']:.3f})")

    # --- Performance summary ---
    def _metrics(y_true, y_pred, y_prob):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return {
            "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "ppv": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "npv": tn / (tn + fn) if (tn + fn) > 0 else 0,
            "auroc": roc_auc_score(y_true, y_prob),
        }

    result_rows = []
    for label, y_t, y_pr, y_pb, ci in [
        ("LR Derivation (Y1)", y1_out, lr_deriv_pred, lr_deriv_prob, lr_deriv_ci),
        ("LR Validation (Y2)", y2_out, lr_valid_pred, lr_valid_prob, lr_valid_ci),
        ("XGBoost Derivation (Y1)", y1_out, xgb_deriv_pred, xgb_deriv_prob, xgb_deriv_ci),
        ("XGBoost Validation (Y2)", y2_out, xgb_valid_pred, xgb_valid_prob, xgb_valid_ci),
    ]:
        m = _metrics(y_t, y_pr, y_pb)
        result_rows.append({
            "Model": label,
            "N": len(y_t),
            "Events": int(y_t.sum()),
            "AUROC": f"{m['auroc']:.3f}",
            "AUROC 95% CI": f"({ci['ci_lower']:.3f}-{ci['ci_upper']:.3f})",
            "Sensitivity": f"{100*m['sensitivity']:.1f}%",
            "Specificity": f"{100*m['specificity']:.1f}%",
            "PPV": f"{100*m['ppv']:.1f}%",
            "NPV": f"{100*m['npv']:.1f}%",
        })

    # Add coefficient rows
    for row in coef_rows:
        result_rows.append({
            "Model": f"LR coef: {row['Variable']}",
            "N": "", "Events": "",
            "AUROC": row["Beta"],
            "AUROC 95% CI": f"aOR={row['aOR']}",
            "Sensitivity": "", "Specificity": "", "PPV": "", "NPV": "",
        })

    tbl = pd.DataFrame(result_rows)
    tbl.to_csv(TABLES / "table_ha_model_extended.csv", index=False)
    _print(f"  Saved table_ha_model_extended.csv")

    return {
        "lr": {"deriv": (y1_out, lr_deriv_prob, lr_deriv_pred),
               "valid": (y2_out, lr_valid_prob, lr_valid_pred)},
        "xgb": {"deriv": (y1_out, xgb_deriv_prob, xgb_deriv_pred),
                "valid": (y2_out, xgb_valid_prob, xgb_valid_pred)},
        "features": features,
    }


# ===================================================================
# 5. ROC FIGURE (LR + XGBoost, Derivation + Validation)
# ===================================================================

def make_roc_figure(model_data):
    """ROC curves for LR and XGBoost, derivation and validation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

    _print("\n=== ROC Figure (Extended) ===")

    fig, ax = plt.subplots(figsize=(7, 6))

    curves = [
        ("LR Derivation (Y1)", model_data["lr"]["deriv"], "#1f77b4", "-"),
        ("LR Validation (Y2)", model_data["lr"]["valid"], "#1f77b4", "--"),
        ("XGBoost Derivation (Y1)", model_data["xgb"]["deriv"], "#d62728", "-"),
        ("XGBoost Validation (Y2)", model_data["xgb"]["valid"], "#d62728", "--"),
    ]

    for label, (y_true, y_prob, _), color, ls in curves:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})",
                color=color, linestyle=ls, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("Headache Model: LR and XGBoost (Derivation vs Validation)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_ha_roc_extended.png")
    plt.close()
    _print("  Saved fig_ha_roc_extended.png")


# ===================================================================
# 6. HEADACHE CALIBRATION
# ===================================================================

def calibration_metrics(y_true, y_prob, n_bins=10):
    """Compute calibration metrics (matching 09-calibration.py)."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    brier = brier_score_loss(y_true, y_prob)
    prevalence = y_true.mean()
    brier_max = prevalence * (1 - prevalence)
    brier_scaled = 1 - brier / brier_max if brier_max > 0 else 0

    expected = y_prob.sum()
    observed = y_true.sum()
    eo_ratio = expected / observed if observed > 0 else np.inf
    citl = np.log(observed / expected) if expected > 0 else np.inf

    y_prob_clipped = np.clip(y_prob, 1e-6, 1 - 1e-6)
    lp = np.log(y_prob_clipped / (1 - y_prob_clipped))
    try:
        lr_cal = LogisticRegression(max_iter=1000, penalty=None, solver="lbfgs")
        lr_cal.fit(lp.reshape(-1, 1), y_true)
        slope = lr_cal.coef_[0][0]
    except Exception:
        slope = np.nan

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_data = []
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        if mask.sum() > 0:
            bin_data.append({
                "bin": i + 1,
                "n": int(mask.sum()),
                "mean_predicted": float(y_prob[mask].mean()),
                "mean_observed": float(y_true[mask].mean()),
            })

    return {
        "citl": citl, "slope": slope, "brier": brier,
        "brier_scaled": brier_scaled, "eo_ratio": eo_ratio, "bin_data": bin_data,
    }


def make_calibration_figure(model_data):
    """Calibration plots for headache models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

    _print("\n=== Calibration Figure ===")

    panels = [
        ("LR Derivation (Y1)", model_data["lr"]["deriv"]),
        ("LR Validation (Y2)", model_data["lr"]["valid"]),
        ("XGBoost Derivation (Y1)", model_data["xgb"]["deriv"]),
        ("XGBoost Validation (Y2)", model_data["xgb"]["valid"]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.flatten()
    colors = ["#1f77b4", "#1f77b4", "#d62728", "#d62728"]

    for i, (label, (y_true, y_prob, _)) in enumerate(panels):
        ax = axes[i]
        cal = calibration_metrics(y_true, y_prob)
        bins = cal["bin_data"]

        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Ideal")

        if bins:
            pred = [b["mean_predicted"] for b in bins]
            obs = [b["mean_observed"] for b in bins]
            sizes = [max(20, b["n"] / 5) for b in bins]
            ax.scatter(pred, obs, s=sizes, c=colors[i], alpha=0.8, zorder=3)
            ax.plot(pred, obs, "-", color=colors[i], alpha=0.6)

            ax.set_xlim(-0.02, max(0.3, max(pred) * 1.2))
            ax.set_ylim(-0.02, max(0.3, max(obs) * 1.2))

        txt = (f"CITL={cal['citl']:.3f}\n"
               f"Slope={cal['slope']:.2f}\n"
               f"Brier={cal['brier']:.4f}")
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=8,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed proportion")
        ax.set_title(label)

    fig.suptitle("Headache Model: Calibration Plots", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_ha_calibration.png", bbox_inches="tight")
    plt.close()
    _print("  Saved fig_ha_calibration.png")


# ===================================================================
# 7. HEADACHE DECISION CURVE
# ===================================================================

def headache_decision_curve(model_data):
    """Decision curve analysis for headache models."""
    _print("\n=== Headache Decision Curve ===")

    thresholds = np.arange(0.01, 0.31, 0.01)

    # Use LR validation predictions for decision curve
    y_true, y_prob, _ = model_data["lr"]["valid"]
    n = len(y_true)
    prevalence = y_true.mean()

    rows = []
    nb_model = []
    nb_all_list = []

    for t in thresholds:
        pred_pos = y_prob >= t
        tp = ((pred_pos) & (y_true == 1)).sum()
        fp = ((pred_pos) & (y_true == 0)).sum()
        nb = tp / n - fp / n * (t / (1 - t))
        nb_all = prevalence - (1 - prevalence) * (t / (1 - t))
        nb_model.append(nb)
        nb_all_list.append(nb_all)

        rows.append({
            "threshold": t,
            "net_benefit": nb,
            "net_benefit_all": nb_all,
            "n_treated": int(pred_pos.sum()),
            "tp": int(tp),
            "fp": int(fp),
        })

    # Save table at key thresholds
    key_t = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
    table_rows = []
    for t in key_t:
        idx = np.argmin(np.abs(thresholds - t))
        r = rows[idx]
        table_rows.append({
            "Threshold (%)": f"{100*t:.0f}%",
            "LR Net benefit": f"{r['net_benefit']:.4f}",
            "Investigate all NB": f"{r['net_benefit_all']:.4f}",
            "Investigate none NB": "0.0000",
            "N treated": r["n_treated"],
        })

    tbl = pd.DataFrame(table_rows)
    tbl.to_csv(TABLES / "table_ha_decision_curve.csv", index=False)
    _print(f"  Saved table_ha_decision_curve.csv")

    # Figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds * 100, nb_all_list, "k-", alpha=0.5, linewidth=1.5,
            label="Investigate all headache")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1,
               label="Investigate none")
    ax.plot(thresholds * 100, nb_model, "-", color="#1f77b4", linewidth=2,
            label="LR headache model")

    ax.set_xlabel("Threshold probability (%)")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision Curve: Headache Model Clinical Utility")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, 30)
    ax.grid(alpha=0.2)

    ax.axvline(x=prevalence * 100, color="gray", linestyle=":", alpha=0.4)
    ax.text(prevalence * 100 + 0.5, ax.get_ylim()[1] * 0.9,
            f"Prevalence\n({100*prevalence:.1f}%)", fontsize=8, color="gray")

    fig.tight_layout()
    fig.savefig(FIGURES / "fig_ha_decision_curve.png")
    plt.close()
    _print("  Saved fig_ha_decision_curve.png")

    return rows


# ===================================================================
# 8. HEADACHE COST ANALYSIS
# ===================================================================

def headache_cost_analysis(ha, model_data):
    """Cost analysis for headache imaging."""
    _print("\n=== Headache Cost Analysis ===")

    # Compute mean investigation cost among imaged headache patients
    inv_cols = [c for c in INV_COL_MAP.keys() if c in ha.columns]
    any_inv = ha[inv_cols].max(axis=1) == 1
    n_imaged = any_inv.sum()
    n_total = len(ha)

    _print(f"  Headache patients imaged: {n_imaged}/{n_total} ({100*n_imaged/n_total:.1f}%)")

    # Cost per imaged patient (base scenario, placeholder)
    patient_costs = np.zeros(len(ha))
    for col in inv_cols:
        cost_key = INV_COL_MAP[col]
        unit_cost = COST_PARAMS[cost_key]["base"]
        patient_costs += (ha[col].values == 1).astype(float) * unit_cost

    mean_cost = patient_costs[any_inv].mean() if n_imaged > 0 else 0.0
    _print(f"  Mean cost per imaged patient (base): ${mean_cost:.2f}")

    # Use LR validation for cost thresholds
    y_true, y_prob, _ = model_data["lr"]["valid"]
    n_mi = y_true.sum()
    thresholds = np.arange(0.01, 0.31, 0.01)

    rows = []
    cost_per_mi_vals = []
    valid_t = []

    for t in thresholds:
        flagged = y_prob >= t
        n_flagged = flagged.sum()
        tp = ((flagged) & (y_true == 1)).sum()
        fn = ((flagged == 0) & (y_true == 1)).sum()
        total_cost = n_flagged * mean_cost
        cost_per_mi = total_cost / tp if tp > 0 else np.inf

        rows.append({
            "threshold": t,
            "n_flagged": int(n_flagged),
            "pct_flagged": f"{100*n_flagged/len(y_true):.1f}%",
            "mi_detected": int(tp),
            "mi_missed": int(fn),
            "mi_detected_pct": f"{100*tp/n_mi:.1f}%" if n_mi > 0 else "0%",
            "total_cost": f"${total_cost:.0f}",
            "cost_per_mi": f"${cost_per_mi:.0f}" if cost_per_mi < np.inf else "—",
        })

        if tp > 0:
            cost_per_mi_vals.append(cost_per_mi)
            valid_t.append(t * 100)

    # Current practice comparison
    current_cost = n_imaged * mean_cost
    current_mi_detected = ((any_inv) & (ha[TARGET] == 1)).sum()
    _print(f"  Current practice: {n_imaged} imaged, "
           f"{current_mi_detected} MI detected, total cost ${current_cost:.0f}")

    # Potential savings at key thresholds
    key_t = [0.05, 0.10, 0.15, 0.20]
    table_rows = []
    for t in key_t:
        idx = np.argmin(np.abs(thresholds - t))
        r = rows[idx]
        table_rows.append({
            "Threshold (%)": f"{100*t:.0f}%",
            "N flagged": r["n_flagged"],
            "MI detected": r["mi_detected"],
            "MI missed": r["mi_missed"],
            "MI detected (%)": r["mi_detected_pct"],
            "Total cost": r["total_cost"],
            "Cost/MI": r["cost_per_mi"],
        })

    # Add current practice row
    table_rows.append({
        "Threshold (%)": "Current practice",
        "N flagged": int(n_imaged),
        "MI detected": int(current_mi_detected),
        "MI missed": int(n_mi - current_mi_detected) if n_mi > current_mi_detected else 0,
        "MI detected (%)": f"{100*current_mi_detected/n_mi:.1f}%" if n_mi > 0 else "—",
        "Total cost": f"${current_cost:.0f}",
        "Cost/MI": f"${current_cost/current_mi_detected:.0f}" if current_mi_detected > 0 else "—",
    })

    tbl = pd.DataFrame(table_rows)
    tbl.to_csv(TABLES / "table_ha_cost.csv", index=False)
    _print(f"  Saved table_ha_cost.csv")

    # Cost per MI figure
    if valid_t:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(valid_t, cost_per_mi_vals, "-o", color="#1f77b4",
                linewidth=2, markersize=4, label="LR headache model")

        if current_mi_detected > 0:
            ax.axhline(y=current_cost / current_mi_detected,
                       color="gray", linestyle="--", alpha=0.6,
                       label=f"Current practice (${current_cost/current_mi_detected:.0f}/MI)")

        ax.set_xlabel("Threshold probability (%)")
        ax.set_ylabel("Cost per MI detected ($)")
        ax.set_title("Headache Imaging: Cost per MI Case Detected")
        ax.legend(loc="upper left", fontsize=9)
        ax.set_xlim(0, 30)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(FIGURES / "fig_ha_cost_threshold.png")
        plt.close()
        _print("  Saved fig_ha_cost_threshold.png")

    return tbl


# ===================================================================
# MAIN
# ===================================================================

def main():
    t_start = time.time()
    _print("=" * 60)
    _print("Brain AttACKs: Extended Headache Analysis (Manuscript 2)")
    _print("=" * 60)

    df, ha, y1, y2 = load_data()

    # 1. Extended demographics
    extended_demographics(ha, y1, y2)

    # 2. Imaging appropriateness
    appropriateness_analysis(ha)
    make_appropriateness_figure(ha)

    # 3. Extended prediction model (LR + XGBoost)
    model_data = extended_model(ha, y1, y2)

    # 4. ROC figure
    make_roc_figure(model_data)

    # 5. Calibration
    make_calibration_figure(model_data)

    # 6. Decision curve
    headache_decision_curve(model_data)

    # 7. Cost analysis
    headache_cost_analysis(ha, model_data)

    # Check for placeholder costs
    all_zero = all(
        p[s] == 0.0
        for p in COST_PARAMS.values()
        for s in ["base", "mid", "high"]
    )
    if all_zero:
        _print("\n  *** NOTE: Cost parameters are $0 (placeholders). ***")
        _print("  *** Cost analysis shows counts only. Populate COST_PARAMS for dollar estimates. ***")

    elapsed = time.time() - t_start
    _print(f"\n{'='*60}")
    _print(f"Done in {elapsed:.0f}s. Outputs in outputs/tables/ and outputs/figures/")
    _print(f"{'='*60}")


if __name__ == "__main__":
    main()
