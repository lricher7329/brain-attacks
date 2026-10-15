"""
16-penalized-lr-headache.py — TRIPOD-Compliant Penalized Logistic Regression (Manuscript 2)

Motivation:
    The headache sub-cohort model in script 13 uses an unpenalized L2 logistic
    regression with fixed C=1.0. With only ~27 MI events in the derivation set
    and 21 predictors (EPV=1.3), the model overfits badly — calibration slope
    drops from ~1.10 (derivation) to ~0.51 (validation). This script implements
    a TRIPOD-compliant penalized LR for comparison, with bootstrap internal
    validation, restricted cubic splines (RCS) for age, and proper calibration.

Key methodological choices:
    - Ridge (L2) penalty with CV-tuned regularisation (neg_log_loss scoring)
    - No SMOTE: destroys calibration in LR; penalization handles overfitting
    - Age modelled continuously with RCS (3 knots, 2 df) replacing binary age_under_12
    - 500-iteration bootstrap with Harrell optimism correction for honest metrics
    - StandardScaler on all features for uniform penalization

Predictors (22 parameters):
    20 binary predictors from EXTENDED_PREDICTORS (minus age_under_12)
    + age (linear) + age_rcs1 (nonlinear RCS basis)

Reporting standards:
    - TRIPOD+AI (Collins et al., BMJ 2024)
    - Riley et al. (BMJ 2020) for sample size / EPV considerations
    - Van Calster et al. (BMC Med 2020) for calibration assessment

Inputs:
    data/processed/brain_attacks_combined.parquet

Outputs:
    outputs/tables/table_ha_penalized_coefficients.csv
    outputs/tables/table_ha_penalized_performance.csv
    outputs/tables/table_ha_penalized_comparison.csv
    outputs/tables/table_ha_penalized_dca.csv
    outputs/figures/fig_ha_pen_roc.png
    outputs/figures/fig_ha_pen_calibration.png
    outputs/figures/fig_ha_pen_forest.png
    outputs/figures/fig_ha_pen_dca.png
    outputs/figures/fig_ha_pen_age_spline.png
    outputs/figures/fig_ha_pen_bootstrap.png
"""

import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    brier_score_loss, confusion_matrix, roc_auc_score, roc_curve,
)
from sklearn.preprocessing import StandardScaler
from scipy import stats

warnings.filterwarnings("ignore")

SEED = 42
N_BOOT = 500
np.random.seed(SEED)

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data" / "processed"
TABLES = PROJECT / "outputs" / "tables"
FIGURES = PROJECT / "outputs" / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

TARGET = "outcome_med_important"

plt.rcParams.update({
    "font.size": 10,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _print(msg=""):
    print(msg, flush=True)


# ===================================================================
# SECTION 0: PREDICTOR DEFINITIONS
# ===================================================================

# Binary predictors (age_under_12 removed — replaced by continuous age + RCS)
BINARY_PREDICTORS = [
    "gender", "ems", "exam_altered_loc",
    "sx_focal_weakness", "sx_ataxia", "sx_speech", "sx_visual",
    "ha_progressive_days", "ha_vomiting", "sx_fever",
    "ha_photophobia", "ha_phonophobia", "ha_maximal_onset",
    "ha_abrupt_onset", "ha_gradual_onset", "ha_nausea",
    "exam_papilledema", "exam_meningismus", "ctas", "any_prev_dx",
]

# For model comparison: original 21 extended predictors from script 13
EXTENDED_PREDICTORS = [
    "age_under_12", "gender", "ems", "exam_altered_loc",
    "sx_focal_weakness", "sx_ataxia", "sx_speech", "sx_visual",
    "ha_progressive_days", "ha_vomiting", "sx_fever",
    "ha_photophobia", "ha_phonophobia", "ha_maximal_onset",
    "ha_abrupt_onset", "ha_gradual_onset", "ha_nausea",
    "exam_papilledema", "exam_meningismus", "ctas", "any_prev_dx",
]

# Human-readable labels for all features
PREDICTOR_LABELS = {
    "gender": "Male sex",
    "ems": "EMS transport",
    "exam_altered_loc": "Altered LOC (exam)",
    "sx_focal_weakness": "Focal weakness",
    "sx_ataxia": "Ataxia",
    "sx_speech": "Speech impairment",
    "sx_visual": "Visual impairment",
    "ha_progressive_days": "Progressive headache (days)",
    "ha_vomiting": "Vomiting",
    "sx_fever": "Fever",
    "ha_photophobia": "Photophobia",
    "ha_phonophobia": "Phonophobia",
    "ha_maximal_onset": "Maximal at onset",
    "ha_abrupt_onset": "Abrupt onset",
    "ha_gradual_onset": "Gradual onset",
    "ha_nausea": "Nausea",
    "exam_papilledema": "Papilledema",
    "exam_meningismus": "Meningismus",
    "ctas": "CTAS score",
    "any_prev_dx": "Any prior neurological Dx",
    "age": "Age (linear)",
    "age_rcs1": "Age (nonlinear RCS)",
    "age_under_12": "Age < 12 years",
}


# ===================================================================
# SECTION 1: RCS HELPER
# ===================================================================

def rcs_basis(x, knots):
    """Compute restricted cubic spline basis using Harrell (2001) truncated power basis.

    For k knots, produces k-2 nonlinear basis columns (plus the original
    linear term which the caller keeps separately). With k=3 knots this
    yields 1 nonlinear basis column.

    Parameters
    ----------
    x : array-like, shape (n,)
        Continuous predictor values.
    knots : array-like, shape (k,)
        Knot locations (k >= 3).

    Returns
    -------
    basis : ndarray, shape (n, k-2)
        Nonlinear RCS basis columns.
    """
    x = np.asarray(x, dtype=float)
    knots = np.sort(np.asarray(knots, dtype=float))
    k = len(knots)
    if k < 3:
        raise ValueError("Need at least 3 knots for RCS")

    def _tpow3(v):
        return np.where(v > 0, v ** 3, 0.0)

    basis = np.zeros((len(x), k - 2))
    t_last = knots[-1]
    t_second_last = knots[-2]
    denom = t_last - t_second_last

    for j in range(k - 2):
        t_j = knots[j]
        basis[:, j] = (
            _tpow3(x - t_j)
            - _tpow3(x - t_second_last) * (t_last - t_j) / denom
            + _tpow3(x - t_last) * (t_second_last - t_j) / denom
        )

    return basis


def compute_age_knots(age_values):
    """Compute 3 RCS knots at 10th, 50th, 90th percentiles (Harrell convention)."""
    return np.percentile(age_values, [10, 50, 90])


# ===================================================================
# SECTION 2: DATA LOADING
# ===================================================================

def load_data():
    """Load headache sub-cohort (same assembly as script 13)."""
    df = pd.read_parquet(DATA / "brain_attacks_combined.parquet")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    _print(f"Full cohort: {len(df)}")

    ha = df[df["sx_any_headache"] == 1].copy()
    _print(f"Headache patients: {len(ha)}")

    prior_ha = (
        (ha.get("previous_diagnoses___4", pd.Series(0, index=ha.index)) == 1) |
        (ha.get("previous_diagnoses___5", pd.Series(0, index=ha.index)) == 1)
    )
    n_excluded = prior_ha.sum()
    ha_clean = ha[~prior_ha].copy()
    _print(f"  Excluded prior headache disorder: {n_excluded}")
    _print(f"  Analysis cohort: {len(ha_clean)}")

    mi = (ha_clean[TARGET] == 1).sum()
    _print(f"  MI: {mi} ({100 * mi / len(ha_clean):.1f}%)")

    y1 = ha_clean[ha_clean["year"] == 1].copy()
    y2 = ha_clean[ha_clean["year"] == 2].copy()
    _print(f"  Year 1 (derivation): {len(y1)} ({(y1[TARGET] == 1).sum()} MI)")
    _print(f"  Year 2 (validation): {len(y2)} ({(y2[TARGET] == 1).sum()} MI)")

    return ha_clean, y1, y2


# ===================================================================
# SECTION 3: FEATURE PREPARATION
# ===================================================================

def prepare_features(y1, y2, knots=None):
    """Build feature matrices with binary predictors + age RCS.

    Returns
    -------
    X1, X2 : ndarray — raw feature matrices (pre-scaling)
    y1_out, y2_out : ndarray — outcome vectors
    feature_names : list — column names
    knots : ndarray — age RCS knots (computed from y1 if not provided)
    scaler : StandardScaler fitted on X1
    """
    features_binary = [f for f in BINARY_PREDICTORS if f in y1.columns]

    # Build binary feature matrices
    X1_bin = y1[features_binary].fillna(0).values
    X2_bin = y2[features_binary].fillna(0).values

    # Age: continuous linear + RCS nonlinear basis
    age1 = y1["age"].values.astype(float)
    age2 = y2["age"].values.astype(float)

    # Impute missing age with median (should be rare)
    age1_median = np.nanmedian(age1)
    age1 = np.where(np.isnan(age1), age1_median, age1)
    age2 = np.where(np.isnan(age2), age1_median, age2)

    if knots is None:
        knots = compute_age_knots(age1)
    _print(f"  Age RCS knots: {knots}")

    rcs1 = rcs_basis(age1, knots)
    rcs2 = rcs_basis(age2, knots)

    X1 = np.column_stack([X1_bin, age1, rcs1])
    X2 = np.column_stack([X2_bin, age2, rcs2])

    feature_names = features_binary + ["age"] + [f"age_rcs{i+1}" for i in range(rcs1.shape[1])]

    y1_out = y1[TARGET].values
    y2_out = y2[TARGET].values

    n_params = len(feature_names)
    n_events = y1_out.sum()
    epv = n_events / n_params
    _print(f"  Features: {n_params} ({len(features_binary)} binary + age linear + {rcs1.shape[1]} RCS)")
    _print(f"  Derivation events: {int(n_events)}, EPV: {epv:.1f}")

    # Scale all features
    scaler = StandardScaler()
    X1_scaled = scaler.fit_transform(X1)
    X2_scaled = scaler.transform(X2)

    return X1_scaled, X2_scaled, X1, X2, y1_out, y2_out, feature_names, knots, scaler


# ===================================================================
# SECTION 4: PENALIZED MODEL FITTING
# ===================================================================

def fit_penalized_lr(X1, y1_out):
    """Fit ridge logistic regression with CV-tuned regularization.

    Uses neg_log_loss scoring (proper scoring rule) rather than AUROC
    to preserve calibration during tuning.
    """
    _print("\n=== Section 4: Penalized LR Fitting ===")

    Cs = np.logspace(-4, 4, 50)

    lr_cv = LogisticRegressionCV(
        Cs=Cs,
        penalty="l2",
        scoring="neg_log_loss",
        cv=10,
        solver="lbfgs",
        max_iter=5000,
        random_state=SEED,
        refit=True,
    )
    lr_cv.fit(X1, y1_out)

    best_C = lr_cv.C_[0]
    _print(f"  Best C (inverse regularization): {best_C:.6f}")
    _print(f"  Lambda (1/C): {1/best_C:.6f}")

    return lr_cv, best_C


# ===================================================================
# SECTION 5: BOOTSTRAP INTERNAL VALIDATION (Harrell optimism correction)
# ===================================================================

def calibration_metrics(y_true, y_prob):
    """Compute CITL, calibration slope, and Brier score."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    brier = brier_score_loss(y_true, y_prob)

    expected = y_prob.sum()
    observed = y_true.sum()
    citl = np.log(observed / expected) if expected > 0 else np.inf

    y_prob_clipped = np.clip(y_prob, 1e-6, 1 - 1e-6)
    lp = np.log(y_prob_clipped / (1 - y_prob_clipped))
    try:
        lr_cal = LogisticRegression(max_iter=1000, penalty=None, solver="lbfgs")
        lr_cal.fit(lp.reshape(-1, 1), y_true)
        slope = lr_cal.coef_[0][0]
    except Exception:
        slope = np.nan

    return {"citl": citl, "slope": slope, "brier": brier}


def bootstrap_optimism(X_orig, y_orig, best_C, n_boot=N_BOOT):
    """Harrell optimism correction via bootstrap.

    For each bootstrap iteration:
      1. Draw bootstrap sample from original data
      2. Fit model on bootstrap sample, evaluate on bootstrap sample (apparent)
      3. Evaluate bootstrap model on original data (test)
      4. Optimism = apparent - test
    Final optimism-corrected = original apparent - mean(optimism)
    """
    _print(f"\n=== Section 5: Bootstrap Optimism Correction ({n_boot} iterations) ===")

    rng = np.random.RandomState(SEED)
    n = len(y_orig)

    optimism_auroc = []
    optimism_slope = []
    optimism_brier = []

    # Fit on original, evaluate on original (apparent performance)
    lr_orig = LogisticRegression(
        penalty="l2", C=best_C, solver="lbfgs",
        max_iter=5000, random_state=SEED,
    )
    lr_orig.fit(X_orig, y_orig)
    prob_orig = lr_orig.predict_proba(X_orig)[:, 1]

    apparent_auroc = roc_auc_score(y_orig, prob_orig)
    apparent_cal = calibration_metrics(y_orig, prob_orig)
    apparent_slope = apparent_cal["slope"]
    apparent_brier = apparent_cal["brier"]

    _print(f"  Apparent AUROC: {apparent_auroc:.3f}")
    _print(f"  Apparent cal slope: {apparent_slope:.3f}")
    _print(f"  Apparent Brier: {apparent_brier:.4f}")

    for b in range(n_boot):
        if (b + 1) % 100 == 0:
            _print(f"  Bootstrap {b + 1}/{n_boot}...")

        idx = rng.choice(n, size=n, replace=True)
        X_boot = X_orig[idx]
        y_boot = y_orig[idx]

        if len(np.unique(y_boot)) < 2:
            continue

        # Re-tune C within bootstrap (correct approach per Harrell)
        try:
            Cs_boot = np.logspace(-4, 4, 20)
            lr_boot_cv = LogisticRegressionCV(
                Cs=Cs_boot, penalty="l2", scoring="neg_log_loss",
                cv=5, solver="lbfgs", max_iter=5000, random_state=SEED, refit=True,
            )
            lr_boot_cv.fit(X_boot, y_boot)
            lr_boot = lr_boot_cv
        except Exception:
            continue

        # Apparent performance on bootstrap sample
        prob_boot_on_boot = lr_boot.predict_proba(X_boot)[:, 1]
        try:
            boot_auroc = roc_auc_score(y_boot, prob_boot_on_boot)
        except ValueError:
            continue
        boot_cal = calibration_metrics(y_boot, prob_boot_on_boot)

        # Test performance: bootstrap model on original data
        prob_boot_on_orig = lr_boot.predict_proba(X_orig)[:, 1]
        try:
            test_auroc = roc_auc_score(y_orig, prob_boot_on_orig)
        except ValueError:
            continue
        test_cal = calibration_metrics(y_orig, prob_boot_on_orig)

        optimism_auroc.append(boot_auroc - test_auroc)
        if not np.isnan(boot_cal["slope"]) and not np.isnan(test_cal["slope"]):
            optimism_slope.append(boot_cal["slope"] - test_cal["slope"])
        optimism_brier.append(boot_cal["brier"] - test_cal["brier"])

    mean_opt_auroc = np.mean(optimism_auroc) if optimism_auroc else 0
    mean_opt_slope = np.mean(optimism_slope) if optimism_slope else 0
    mean_opt_brier = np.mean(optimism_brier) if optimism_brier else 0

    corrected_auroc = apparent_auroc - mean_opt_auroc
    corrected_slope = apparent_slope - mean_opt_slope
    corrected_brier = apparent_brier - mean_opt_brier

    _print(f"  Mean optimism — AUROC: {mean_opt_auroc:.4f}, "
           f"Slope: {mean_opt_slope:.4f}, Brier: {mean_opt_brier:.6f}")
    _print(f"  Corrected AUROC: {corrected_auroc:.3f}")
    _print(f"  Corrected cal slope: {corrected_slope:.3f}")
    _print(f"  Corrected Brier: {corrected_brier:.4f}")

    return {
        "apparent_auroc": apparent_auroc,
        "apparent_slope": apparent_slope,
        "apparent_brier": apparent_brier,
        "mean_opt_auroc": mean_opt_auroc,
        "mean_opt_slope": mean_opt_slope,
        "mean_opt_brier": mean_opt_brier,
        "corrected_auroc": corrected_auroc,
        "corrected_slope": corrected_slope,
        "corrected_brier": corrected_brier,
        "optimism_auroc": np.array(optimism_auroc),
        "optimism_slope": np.array(optimism_slope),
        "optimism_brier": np.array(optimism_brier),
        "n_successful": len(optimism_auroc),
        "lr_orig": lr_orig,
        "prob_orig": prob_orig,
    }


# ===================================================================
# SECTION 6: TEMPORAL VALIDATION
# ===================================================================

def temporal_validation(lr_model, X1, X2, y1_out, y2_out):
    """Apply derivation model to Year 2 and compute full calibration metrics."""
    _print("\n=== Section 6: Temporal Validation ===")

    prob_deriv = lr_model.predict_proba(X1)[:, 1]
    prob_valid = lr_model.predict_proba(X2)[:, 1]

    auroc_deriv = roc_auc_score(y1_out, prob_deriv)
    auroc_valid = roc_auc_score(y2_out, prob_valid)

    cal_deriv = calibration_metrics(y1_out, prob_deriv)
    cal_valid = calibration_metrics(y2_out, prob_valid)

    # Bootstrap CIs for validation AUROC
    rng = np.random.RandomState(SEED)
    aurocs_valid = []
    for _ in range(N_BOOT):
        idx = rng.choice(len(y2_out), size=len(y2_out), replace=True)
        if len(np.unique(y2_out[idx])) < 2:
            continue
        try:
            aurocs_valid.append(roc_auc_score(y2_out[idx], prob_valid[idx]))
        except ValueError:
            continue

    aurocs_deriv = []
    rng2 = np.random.RandomState(SEED + 1)
    for _ in range(N_BOOT):
        idx = rng2.choice(len(y1_out), size=len(y1_out), replace=True)
        if len(np.unique(y1_out[idx])) < 2:
            continue
        try:
            aurocs_deriv.append(roc_auc_score(y1_out[idx], prob_deriv[idx]))
        except ValueError:
            continue

    ci_deriv = (np.percentile(aurocs_deriv, 2.5), np.percentile(aurocs_deriv, 97.5)) if aurocs_deriv else (0, 0)
    ci_valid = (np.percentile(aurocs_valid, 2.5), np.percentile(aurocs_valid, 97.5)) if aurocs_valid else (0, 0)

    _print(f"  Derivation — AUROC: {auroc_deriv:.3f} ({ci_deriv[0]:.3f}–{ci_deriv[1]:.3f})")
    _print(f"    CITL={cal_deriv['citl']:.3f}, Slope={cal_deriv['slope']:.3f}, Brier={cal_deriv['brier']:.4f}")
    _print(f"  Validation — AUROC: {auroc_valid:.3f} ({ci_valid[0]:.3f}–{ci_valid[1]:.3f})")
    _print(f"    CITL={cal_valid['citl']:.3f}, Slope={cal_valid['slope']:.3f}, Brier={cal_valid['brier']:.4f}")

    return {
        "prob_deriv": prob_deriv,
        "prob_valid": prob_valid,
        "auroc_deriv": auroc_deriv,
        "auroc_valid": auroc_valid,
        "ci_deriv": ci_deriv,
        "ci_valid": ci_valid,
        "cal_deriv": cal_deriv,
        "cal_valid": cal_valid,
    }


# ===================================================================
# SECTION 7: COEFFICIENT TABLE WITH BOOTSTRAP CIs
# ===================================================================

def coefficient_table(X1, y1_out, best_C, feature_names, scaler):
    """Bootstrap 95% CIs for penalized LR coefficients (betas and aORs)."""
    _print(f"\n=== Section 7: Coefficient Table ({N_BOOT} bootstrap iterations) ===")

    rng = np.random.RandomState(SEED)
    n = len(y1_out)

    # Fit final model
    lr_final = LogisticRegression(
        penalty="l2", C=best_C, solver="lbfgs",
        max_iter=5000, random_state=SEED,
    )
    lr_final.fit(X1, y1_out)
    betas = lr_final.coef_[0]
    intercept = lr_final.intercept_[0]

    # Bootstrap for CIs
    boot_betas = []
    for b in range(N_BOOT):
        idx = rng.choice(n, size=n, replace=True)
        X_b = X1[idx]
        y_b = y1_out[idx]
        if len(np.unique(y_b)) < 2:
            continue
        try:
            lr_b = LogisticRegression(
                penalty="l2", C=best_C, solver="lbfgs",
                max_iter=5000, random_state=SEED,
            )
            lr_b.fit(X_b, y_b)
            boot_betas.append(lr_b.coef_[0])
        except Exception:
            continue

    boot_betas = np.array(boot_betas)

    rows = []
    rows.append({
        "Variable": "Intercept",
        "Label": "Intercept",
        "Beta": f"{intercept:.4f}",
        "Beta 95% CI": "",
        "aOR": "",
        "aOR 95% CI": "",
    })

    for i, fname in enumerate(feature_names):
        beta = betas[i]
        aor = np.exp(beta)
        label = PREDICTOR_LABELS.get(fname, fname)

        if len(boot_betas) > 0:
            ci_beta_lo = np.percentile(boot_betas[:, i], 2.5)
            ci_beta_hi = np.percentile(boot_betas[:, i], 97.5)
            ci_aor_lo = np.exp(ci_beta_lo)
            ci_aor_hi = np.exp(ci_beta_hi)
            beta_ci_str = f"({ci_beta_lo:.4f}, {ci_beta_hi:.4f})"
            aor_ci_str = f"({ci_aor_lo:.2f}, {ci_aor_hi:.2f})"
        else:
            beta_ci_str = ""
            aor_ci_str = ""

        rows.append({
            "Variable": fname,
            "Label": label,
            "Beta": f"{beta:.4f}",
            "Beta 95% CI": beta_ci_str,
            "aOR": f"{aor:.2f}",
            "aOR 95% CI": aor_ci_str,
        })

    tbl = pd.DataFrame(rows)
    tbl.to_csv(TABLES / "table_ha_penalized_coefficients.csv", index=False)
    _print(f"  Saved table_ha_penalized_coefficients.csv ({len(tbl)} rows)")

    # Print summary
    for _, row in tbl.iterrows():
        if row["Variable"] == "Intercept":
            _print(f"    {row['Label']}: {row['Beta']}")
        else:
            _print(f"    {row['Label']}: beta={row['Beta']}, aOR={row['aOR']} {row['aOR 95% CI']}")

    return tbl, betas, boot_betas


# ===================================================================
# SECTION 8: DECISION CURVE ANALYSIS
# ===================================================================

def decision_curve_analysis(y_true, y_prob_pen, y_prob_lr=None, y_prob_xgb=None):
    """Net benefit calculation on validation set for multiple models."""
    _print("\n=== Section 8: Decision Curve Analysis ===")

    thresholds = np.arange(0.01, 0.51, 0.005)
    n = len(y_true)
    prevalence = y_true.mean()

    def _net_benefit(y_true, y_prob, t):
        pred_pos = y_prob >= t
        tp = ((pred_pos) & (y_true == 1)).sum()
        fp = ((pred_pos) & (y_true == 0)).sum()
        return tp / n - fp / n * (t / (1 - t))

    nb_pen = [_net_benefit(y_true, y_prob_pen, t) for t in thresholds]
    nb_all = [prevalence - (1 - prevalence) * (t / (1 - t)) for t in thresholds]

    nb_lr = [_net_benefit(y_true, y_prob_lr, t) for t in thresholds] if y_prob_lr is not None else None
    nb_xgb = [_net_benefit(y_true, y_prob_xgb, t) for t in thresholds] if y_prob_xgb is not None else None

    # Save table at key thresholds
    key_t = [0.02, 0.05, 0.10, 0.15, 0.20]
    table_rows = []
    for t in key_t:
        idx = np.argmin(np.abs(thresholds - t))
        row = {
            "Threshold (%)": f"{100 * t:.0f}%",
            "Penalized LR NB": f"{nb_pen[idx]:.4f}",
            "Investigate all NB": f"{nb_all[idx]:.4f}",
            "Investigate none NB": "0.0000",
        }
        if nb_lr is not None:
            row["Simple LR NB"] = f"{nb_lr[idx]:.4f}"
        if nb_xgb is not None:
            row["XGBoost NB"] = f"{nb_xgb[idx]:.4f}"
        table_rows.append(row)

    tbl = pd.DataFrame(table_rows)
    tbl.to_csv(TABLES / "table_ha_penalized_dca.csv", index=False)
    _print(f"  Saved table_ha_penalized_dca.csv")

    return {
        "thresholds": thresholds,
        "nb_pen": nb_pen,
        "nb_all": nb_all,
        "nb_lr": nb_lr,
        "nb_xgb": nb_xgb,
        "prevalence": prevalence,
    }


# ===================================================================
# SECTION 9: MODEL COMPARISON (simple LR + XGBoost from script 13)
# ===================================================================

def refit_comparison_models(y1, y2):
    """Refit simple LR and XGBoost for head-to-head comparison."""
    _print("\n=== Section 9: Model Comparison (refit simple LR + XGBoost) ===")

    features = [f for f in EXTENDED_PREDICTORS if f in y1.columns]

    X1 = y1[features].fillna(0).values
    X2 = y2[features].fillna(0).values
    y1_out = y1[TARGET].values
    y2_out = y2[TARGET].values

    # --- Simple LR (unpenalized, C=1.0, matching script 13) ---
    scaler_lr = StandardScaler()
    X1_lr = scaler_lr.fit_transform(X1)
    X2_lr = scaler_lr.transform(X2)

    lr_simple = LogisticRegression(max_iter=1000, random_state=SEED, penalty="l2", C=1.0)
    lr_simple.fit(X1_lr, y1_out)

    lr_deriv_prob = lr_simple.predict_proba(X1_lr)[:, 1]
    lr_valid_prob = lr_simple.predict_proba(X2_lr)[:, 1]

    # --- XGBoost with SMOTE (matching script 13) ---
    try:
        from imblearn.over_sampling import SMOTE
        from sklearn.metrics import fbeta_score, make_scorer
        from sklearn.model_selection import GridSearchCV, StratifiedKFold
        from xgboost import XGBClassifier

        sm = SMOTE(sampling_strategy=0.5, random_state=SEED)
        try:
            X1_sm, y1_sm = sm.fit_resample(X1, y1_out)
        except ValueError:
            X1_sm, y1_sm = X1, y1_out

        param_grid = {
            "n_estimators": [50, 100],
            "max_depth": [3, 5],
            "learning_rate": [0.01, 0.1],
        }
        f2_scorer = make_scorer(fbeta_score, beta=2, zero_division=0)
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        xgb = XGBClassifier(
            use_label_encoder=False, eval_metric="logloss",
            random_state=SEED, verbosity=0,
        )
        grid = GridSearchCV(xgb, param_grid, scoring=f2_scorer,
                            cv=inner_cv, n_jobs=-1, refit=True)
        grid.fit(X1_sm, y1_sm)

        xgb_deriv_prob = grid.predict_proba(X1)[:, 1]
        xgb_valid_prob = grid.predict_proba(X2)[:, 1]
        xgb_available = True
        _print(f"  XGBoost best params: {grid.best_params_}")
    except ImportError:
        _print("  WARNING: XGBoost/imblearn not available, skipping")
        xgb_deriv_prob = None
        xgb_valid_prob = None
        xgb_available = False

    # Metrics
    def _full_metrics(label, y_true, y_prob):
        auroc = roc_auc_score(y_true, y_prob)
        cal = calibration_metrics(y_true, y_prob)
        rng = np.random.RandomState(SEED)
        boots = []
        for _ in range(N_BOOT):
            idx = rng.choice(len(y_true), size=len(y_true), replace=True)
            if len(np.unique(y_true[idx])) < 2:
                continue
            try:
                boots.append(roc_auc_score(y_true[idx], y_prob[idx]))
            except ValueError:
                continue
        ci = (np.percentile(boots, 2.5), np.percentile(boots, 97.5)) if boots else (0, 0)
        _print(f"  {label}: AUROC={auroc:.3f} ({ci[0]:.3f}–{ci[1]:.3f}), "
               f"slope={cal['slope']:.3f}, Brier={cal['brier']:.4f}")
        return {
            "auroc": auroc, "ci": ci, "cal": cal,
        }

    results = {}
    results["lr_deriv"] = _full_metrics("Simple LR Derivation", y1_out, lr_deriv_prob)
    results["lr_valid"] = _full_metrics("Simple LR Validation", y2_out, lr_valid_prob)
    if xgb_available:
        results["xgb_deriv"] = _full_metrics("XGBoost Derivation", y1_out, xgb_deriv_prob)
        results["xgb_valid"] = _full_metrics("XGBoost Validation", y2_out, xgb_valid_prob)

    return {
        "results": results,
        "lr_deriv_prob": lr_deriv_prob,
        "lr_valid_prob": lr_valid_prob,
        "xgb_deriv_prob": xgb_deriv_prob,
        "xgb_valid_prob": xgb_valid_prob,
        "xgb_available": xgb_available,
        "y1_out": y1_out,
        "y2_out": y2_out,
    }


def save_comparison_table(pen_valid, pen_opt, comp):
    """Save head-to-head comparison table."""
    rows = []

    # Penalized LR
    rows.append({
        "Model": "Penalized LR",
        "Set": "Derivation",
        "AUROC (95% CI)": f"{pen_opt['apparent_auroc']:.3f}",
        "Optimism-corrected AUROC": f"{pen_opt['corrected_auroc']:.3f}",
        "Calibration slope": f"{pen_opt['apparent_slope']:.3f}",
        "Optimism-corrected slope": f"{pen_opt['corrected_slope']:.3f}",
        "Brier": f"{pen_opt['apparent_brier']:.4f}",
    })
    rows.append({
        "Model": "Penalized LR",
        "Set": "Validation",
        "AUROC (95% CI)": f"{pen_valid['auroc_valid']:.3f} ({pen_valid['ci_valid'][0]:.3f}–{pen_valid['ci_valid'][1]:.3f})",
        "Optimism-corrected AUROC": "",
        "Calibration slope": f"{pen_valid['cal_valid']['slope']:.3f}",
        "Optimism-corrected slope": "",
        "Brier": f"{pen_valid['cal_valid']['brier']:.4f}",
    })

    # Simple LR
    r = comp["results"]
    rows.append({
        "Model": "Simple LR (C=1.0)",
        "Set": "Derivation",
        "AUROC (95% CI)": f"{r['lr_deriv']['auroc']:.3f} ({r['lr_deriv']['ci'][0]:.3f}–{r['lr_deriv']['ci'][1]:.3f})",
        "Optimism-corrected AUROC": "",
        "Calibration slope": f"{r['lr_deriv']['cal']['slope']:.3f}",
        "Optimism-corrected slope": "",
        "Brier": f"{r['lr_deriv']['cal']['brier']:.4f}",
    })
    rows.append({
        "Model": "Simple LR (C=1.0)",
        "Set": "Validation",
        "AUROC (95% CI)": f"{r['lr_valid']['auroc']:.3f} ({r['lr_valid']['ci'][0]:.3f}–{r['lr_valid']['ci'][1]:.3f})",
        "Optimism-corrected AUROC": "",
        "Calibration slope": f"{r['lr_valid']['cal']['slope']:.3f}",
        "Optimism-corrected slope": "",
        "Brier": f"{r['lr_valid']['cal']['brier']:.4f}",
    })

    # XGBoost
    if comp["xgb_available"]:
        rows.append({
            "Model": "XGBoost + SMOTE",
            "Set": "Derivation",
            "AUROC (95% CI)": f"{r['xgb_deriv']['auroc']:.3f} ({r['xgb_deriv']['ci'][0]:.3f}–{r['xgb_deriv']['ci'][1]:.3f})",
            "Optimism-corrected AUROC": "",
            "Calibration slope": f"{r['xgb_deriv']['cal']['slope']:.3f}",
            "Optimism-corrected slope": "",
            "Brier": f"{r['xgb_deriv']['cal']['brier']:.4f}",
        })
        rows.append({
            "Model": "XGBoost + SMOTE",
            "Set": "Validation",
            "AUROC (95% CI)": f"{r['xgb_valid']['auroc']:.3f} ({r['xgb_valid']['ci'][0]:.3f}–{r['xgb_valid']['ci'][1]:.3f})",
            "Optimism-corrected AUROC": "",
            "Calibration slope": f"{r['xgb_valid']['cal']['slope']:.3f}",
            "Optimism-corrected slope": "",
            "Brier": f"{r['xgb_valid']['cal']['brier']:.4f}",
        })

    tbl = pd.DataFrame(rows)
    tbl.to_csv(TABLES / "table_ha_penalized_comparison.csv", index=False)
    _print(f"  Saved table_ha_penalized_comparison.csv")
    return tbl


def save_performance_table(pen_opt, pen_valid):
    """Save derivation + validation metrics with optimism-corrected estimates."""
    rows = [
        {
            "Set": "Derivation (apparent)",
            "AUROC": f"{pen_opt['apparent_auroc']:.3f}",
            "Calibration slope": f"{pen_opt['apparent_slope']:.3f}",
            "Brier": f"{pen_opt['apparent_brier']:.4f}",
        },
        {
            "Set": "Derivation (optimism-corrected)",
            "AUROC": f"{pen_opt['corrected_auroc']:.3f}",
            "Calibration slope": f"{pen_opt['corrected_slope']:.3f}",
            "Brier": f"{pen_opt['corrected_brier']:.4f}",
        },
        {
            "Set": "Mean optimism",
            "AUROC": f"{pen_opt['mean_opt_auroc']:.4f}",
            "Calibration slope": f"{pen_opt['mean_opt_slope']:.4f}",
            "Brier": f"{pen_opt['mean_opt_brier']:.6f}",
        },
        {
            "Set": f"Temporal validation (Year 2)",
            "AUROC": f"{pen_valid['auroc_valid']:.3f} ({pen_valid['ci_valid'][0]:.3f}–{pen_valid['ci_valid'][1]:.3f})",
            "Calibration slope": f"{pen_valid['cal_valid']['slope']:.3f}",
            "Brier": f"{pen_valid['cal_valid']['brier']:.4f}",
        },
    ]

    # Add CITL for validation
    rows[-1]["CITL"] = f"{pen_valid['cal_valid']['citl']:.3f}"

    tbl = pd.DataFrame(rows)
    tbl.to_csv(TABLES / "table_ha_penalized_performance.csv", index=False)
    _print(f"  Saved table_ha_penalized_performance.csv")
    return tbl


# ===================================================================
# SECTION 10: FIGURES
# ===================================================================

def _calibration_binned(y_true, y_prob, n_bins=10):
    """Bin predictions for calibration plots."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        else:
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() > 0:
            bins.append({
                "n": int(mask.sum()),
                "mean_predicted": float(y_prob[mask].mean()),
                "mean_observed": float(y_true[mask].mean()),
            })
    return bins


def fig_roc(pen_valid, comp, y1_out, y2_out):
    """Figure 1: ROC curves — penalized vs simple LR vs XGBoost (derivation + validation)."""
    _print("\n  Generating fig_ha_pen_roc.png...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # Derivation panel
    ax = axes[0]
    ax.set_title("Derivation (Year 1)", fontweight="bold")

    for label, y_t, y_p, color, ls in [
        ("Penalized LR", y1_out, pen_valid["prob_deriv"], "#2ca02c", "-"),
        ("Simple LR", comp["y1_out"], comp["lr_deriv_prob"], "#1f77b4", "--"),
    ]:
        fpr, tpr, _ = roc_curve(y_t, y_p)
        auc = roc_auc_score(y_t, y_p)
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})", color=color, linestyle=ls, linewidth=2)

    if comp["xgb_available"] and comp["xgb_deriv_prob"] is not None:
        fpr, tpr, _ = roc_curve(comp["y1_out"], comp["xgb_deriv_prob"])
        auc = roc_auc_score(comp["y1_out"], comp["xgb_deriv_prob"])
        ax.plot(fpr, tpr, label=f"XGBoost (AUC={auc:.3f})", color="#d62728", linestyle="-.", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("1 − Specificity")
    ax.set_ylabel("Sensitivity")
    ax.legend(loc="lower right", fontsize=9)

    # Validation panel
    ax = axes[1]
    ax.set_title("Validation (Year 2)", fontweight="bold")

    for label, y_t, y_p, color, ls in [
        ("Penalized LR", y2_out, pen_valid["prob_valid"], "#2ca02c", "-"),
        ("Simple LR", comp["y2_out"], comp["lr_valid_prob"], "#1f77b4", "--"),
    ]:
        fpr, tpr, _ = roc_curve(y_t, y_p)
        auc = roc_auc_score(y_t, y_p)
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})", color=color, linestyle=ls, linewidth=2)

    if comp["xgb_available"] and comp["xgb_valid_prob"] is not None:
        fpr, tpr, _ = roc_curve(comp["y2_out"], comp["xgb_valid_prob"])
        auc = roc_auc_score(comp["y2_out"], comp["xgb_valid_prob"])
        ax.plot(fpr, tpr, label=f"XGBoost (AUC={auc:.3f})", color="#d62728", linestyle="-.", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("1 − Specificity")
    ax.set_ylabel("Sensitivity")
    ax.legend(loc="lower right", fontsize=9)

    fig.suptitle("Headache Model: ROC Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_ha_pen_roc.png", bbox_inches="tight")
    plt.close()
    _print("    Saved fig_ha_pen_roc.png")


def fig_calibration(pen_valid, comp, y1_out, y2_out):
    """Figure 2: 2x2 calibration plots."""
    _print("  Generating fig_ha_pen_calibration.png...")

    panels = [
        ("Penalized LR\nDerivation (Year 1)", y1_out, pen_valid["prob_deriv"], "#2ca02c"),
        ("Penalized LR\nValidation (Year 2)", y2_out, pen_valid["prob_valid"], "#2ca02c"),
        ("Simple LR\nValidation (Year 2)", comp["y2_out"], comp["lr_valid_prob"], "#1f77b4"),
    ]
    if comp["xgb_available"] and comp["xgb_valid_prob"] is not None:
        panels.append(
            ("XGBoost\nValidation (Year 2)", comp["y2_out"], comp["xgb_valid_prob"], "#d62728")
        )

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.flatten()

    for i, (title, y_t, y_p, color) in enumerate(panels):
        ax = axes[i]
        cal = calibration_metrics(y_t, y_p)
        bins = _calibration_binned(y_t, y_p)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Ideal")
        if bins:
            pred = [b["mean_predicted"] for b in bins]
            obs = [b["mean_observed"] for b in bins]
            sizes = [max(20, b["n"] / 5) for b in bins]
            ax.scatter(pred, obs, s=sizes, c=color, alpha=0.8, zorder=3)
            ax.plot(pred, obs, "-", color=color, alpha=0.6)
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
        ax.set_title(title, fontsize=10)

    # Hide unused panel if only 3 models
    if len(panels) < 4:
        axes[3].axis("off")

    fig.suptitle("Calibration Plots", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_ha_pen_calibration.png", bbox_inches="tight")
    plt.close()
    _print("    Saved fig_ha_pen_calibration.png")


def fig_forest(coef_tbl):
    """Figure 3: Forest plot of penalized aORs with 95% CIs."""
    _print("  Generating fig_ha_pen_forest.png...")

    # Filter to non-intercept rows
    df = coef_tbl[coef_tbl["Variable"] != "Intercept"].copy()
    df["aOR_val"] = df["aOR"].astype(float)

    # Parse CIs
    ci_lo = []
    ci_hi = []
    for _, row in df.iterrows():
        ci_str = row["aOR 95% CI"]
        if ci_str and ci_str.startswith("("):
            parts = ci_str.strip("()").split(",")
            ci_lo.append(float(parts[0].strip()))
            ci_hi.append(float(parts[1].strip()))
        else:
            ci_lo.append(row["aOR_val"])
            ci_hi.append(row["aOR_val"])
    df["ci_lo"] = ci_lo
    df["ci_hi"] = ci_hi

    # Sort by aOR
    df = df.sort_values("aOR_val", ascending=True).reset_index(drop=True)

    n = len(df)
    y_pos = np.arange(n)

    fig, ax = plt.subplots(figsize=(9, max(6, n * 0.4)))

    colors = ["#c0392b" if aor > 1.0 else "#2980b9" for aor in df["aOR_val"]]

    # Error bars
    xerr_lo = df["aOR_val"].values - df["ci_lo"].values
    xerr_hi = df["ci_hi"].values - df["aOR_val"].values
    ax.errorbar(
        df["aOR_val"], y_pos,
        xerr=[xerr_lo, xerr_hi],
        fmt="none", ecolor="#555555", elinewidth=1.2, capsize=3,
    )
    ax.scatter(df["aOR_val"], y_pos, c=colors, s=60, zorder=5, edgecolors="white", linewidths=0.5)

    # Reference line at aOR=1
    ax.axvline(x=1.0, color="black", linewidth=1, linestyle="-", alpha=0.8)

    # Labels
    labels = [PREDICTOR_LABELS.get(v, v) for v in df["Variable"]]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Adjusted Odds Ratio (aOR)", fontsize=11)
    ax.set_title("Penalized Logistic Regression: Adjusted Odds Ratios\n(Ridge L2, bootstrap 95% CIs)",
                 fontsize=12, fontweight="bold")

    # aOR values on right
    for i, (aor, lo, hi) in enumerate(zip(df["aOR_val"], df["ci_lo"], df["ci_hi"])):
        ax.text(ax.get_xlim()[1] * 0.97, i, f"{aor:.2f} ({lo:.2f}–{hi:.2f})",
                va="center", ha="right", fontsize=7.5, color="#333333")

    # Shade regions
    ax.axvspan(0, 1.0, alpha=0.04, color="#2980b9")
    ax.axvspan(1.0, ax.get_xlim()[1], alpha=0.04, color="#c0392b")
    ax.set_ylim(-0.5, n + 1.5)
    ax.text(0.6, n + 0.5, "Protective", fontsize=8, ha="center", color="#2980b9", fontstyle="italic")
    ax.text(1.5, n + 0.5, "Risk factor", fontsize=8, ha="center", color="#c0392b", fontstyle="italic")

    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGURES / "fig_ha_pen_forest.png", bbox_inches="tight")
    plt.close()
    _print("    Saved fig_ha_pen_forest.png")


def fig_dca(dca_data):
    """Figure 4: Decision curve on validation data."""
    _print("  Generating fig_ha_pen_dca.png...")

    t = dca_data["thresholds"] * 100

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(t, dca_data["nb_all"], "k-", alpha=0.5, linewidth=1.5, label="Investigate all")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1, label="Investigate none")
    ax.plot(t, dca_data["nb_pen"], "-", color="#2ca02c", linewidth=2.5, label="Penalized LR")

    if dca_data["nb_lr"] is not None:
        ax.plot(t, dca_data["nb_lr"], "--", color="#1f77b4", linewidth=2, label="Simple LR")
    if dca_data["nb_xgb"] is not None:
        ax.plot(t, dca_data["nb_xgb"], "-.", color="#d62728", linewidth=2, label="XGBoost")

    ax.set_xlabel("Threshold probability (%)")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision Curve: Validation Set (Year 2)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, 50)
    ax.grid(alpha=0.2)

    prev = dca_data["prevalence"]
    ax.axvline(x=prev * 100, color="gray", linestyle=":", alpha=0.4)
    ax.text(prev * 100 + 0.8, ax.get_ylim()[0] * 0.3, f"MI rate\n({100 * prev:.1f}%)",
            fontsize=8, color="gray")

    fig.tight_layout()
    fig.savefig(FIGURES / "fig_ha_pen_dca.png", bbox_inches="tight")
    plt.close()
    _print("    Saved fig_ha_pen_dca.png")


def fig_age_spline(lr_model, feature_names, scaler, knots):
    """Figure 5: Partial effect of age showing RCS non-linearity."""
    _print("  Generating fig_ha_pen_age_spline.png...")

    # Create age range
    age_range = np.linspace(0, 18, 200)
    rcs_vals = rcs_basis(age_range, knots)

    # Get indices for age and age_rcs columns
    idx_age = feature_names.index("age")
    rcs_indices = [feature_names.index(f"age_rcs{i+1}") for i in range(rcs_vals.shape[1])]

    # Build a "zero" feature vector, varying only age
    betas = lr_model.coef_[0]
    intercept = lr_model.intercept_[0]
    means = scaler.mean_
    scales = scaler.scale_

    # Partial log-odds from age components only
    # After scaling: z_j = (x_j - mean_j) / scale_j
    # Linear predictor contribution = beta_age * (age - mean_age) / scale_age
    #                                + beta_rcs1 * (rcs1 - mean_rcs1) / scale_rcs1

    lp_age = betas[idx_age] * (age_range - means[idx_age]) / scales[idx_age]
    for j, rcs_idx in enumerate(rcs_indices):
        lp_age += betas[rcs_idx] * (rcs_vals[:, j] - means[rcs_idx]) / scales[rcs_idx]

    # Center so reference is median age
    median_age = knots[1]
    median_rcs = rcs_basis(np.array([median_age]), knots)
    lp_ref = betas[idx_age] * (median_age - means[idx_age]) / scales[idx_age]
    for j, rcs_idx in enumerate(rcs_indices):
        lp_ref += betas[rcs_idx] * (median_rcs[0, j] - means[rcs_idx]) / scales[rcs_idx]
    lp_centered = lp_age - lp_ref

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(age_range, lp_centered, "-", color="#2ca02c", linewidth=2.5)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Partial log-odds\n(centered at median age)")
    ax.set_title("Partial Effect of Age (Restricted Cubic Spline, 3 knots)")

    # Mark knots
    for kn in knots:
        ax.axvline(x=kn, color="#999999", linestyle=":", alpha=0.5)
    ax.text(knots[0], ax.get_ylim()[1] * 0.9, f"p10={knots[0]:.1f}", fontsize=8, ha="center", color="#666666")
    ax.text(knots[1], ax.get_ylim()[1] * 0.9, f"p50={knots[1]:.1f}", fontsize=8, ha="center", color="#666666")
    ax.text(knots[2], ax.get_ylim()[1] * 0.9, f"p90={knots[2]:.1f}", fontsize=8, ha="center", color="#666666")

    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_ha_pen_age_spline.png", bbox_inches="tight")
    plt.close()
    _print("    Saved fig_ha_pen_age_spline.png")


def fig_bootstrap(opt_data):
    """Figure 6: Optimism distribution histograms."""
    _print("  Generating fig_ha_pen_bootstrap.png...")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, vals, label, color in [
        (axes[0], opt_data["optimism_auroc"], "AUROC Optimism", "#2ca02c"),
        (axes[1], opt_data["optimism_slope"], "Cal Slope Optimism", "#ff7f0e"),
        (axes[2], opt_data["optimism_brier"], "Brier Optimism", "#9467bd"),
    ]:
        if len(vals) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label)
            continue

        ax.hist(vals, bins=30, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(x=np.mean(vals), color="black", linewidth=2, linestyle="-",
                   label=f"Mean={np.mean(vals):.4f}")
        ax.axvline(x=0, color="gray", linewidth=1, linestyle="--", alpha=0.5)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of {label}\n(n={len(vals)} bootstraps)")
        ax.legend(fontsize=9)

    fig.suptitle(f"Bootstrap Optimism Distributions ({opt_data['n_successful']}/{N_BOOT} successful)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_ha_pen_bootstrap.png", bbox_inches="tight")
    plt.close()
    _print("    Saved fig_ha_pen_bootstrap.png")


# ===================================================================
# SECTION 11: SENSITIVITY ANALYSIS — EXCLUDING CEREBELLAR ATAXIA
# ===================================================================

def sensitivity_exclude_cerebellar_ataxia(ha, y1, y2, primary_results):
    """Re-run penalized LR after reclassifying cerebellar ataxia as non-MI.

    Cerebellar ataxia (med_imp_dx___13) is arguably self-limiting and may not
    warrant the same imaging urgency as other MI diagnoses. This sensitivity
    analysis reclassifies CA-only cases as benign and re-evaluates model
    performance to assess robustness.

    Parameters
    ----------
    ha : DataFrame — full analysis cohort
    y1, y2 : DataFrames — Year 1, Year 2 splits
    primary_results : dict — primary analysis results for comparison
    """
    _print("\n" + "=" * 65)
    _print("SENSITIVITY ANALYSIS: Excluding Cerebellar Ataxia from MI")
    _print("=" * 65)

    ca_col = "med_imp_dx___13"
    mi_dx_cols = [c for c in ha.columns if c.startswith("med_imp_dx___")]

    if ca_col not in ha.columns:
        _print("  WARNING: med_imp_dx___13 not found, skipping sensitivity analysis")
        return None

    # Identify CA-only cases (cerebellar ataxia with no other MI dx)
    ca_mask = ha[ca_col].fillna(0) == 1
    other_mi = ha.loc[ca_mask, [c for c in mi_dx_cols if c != ca_col]].fillna(0).sum(axis=1)
    ca_only_idx = other_mi[other_mi == 0].index
    n_ca_only = len(ca_only_idx)

    _print(f"  Cerebellar ataxia MI cases: {ca_mask.sum()}")
    _print(f"  CA-only (no other MI dx): {n_ca_only}")

    # Reclassify: create modified outcome
    ha_sens = ha.copy()
    ha_sens.loc[ca_only_idx, TARGET] = 0

    y1_sens = ha_sens[ha_sens["year"] == 1].copy()
    y2_sens = ha_sens[ha_sens["year"] == 2].copy()

    n_mi_y1 = (y1_sens[TARGET] == 1).sum()
    n_mi_y2 = (y2_sens[TARGET] == 1).sum()
    _print(f"  Year 1: {len(y1_sens)} patients, {n_mi_y1} MI (was {(y1[TARGET] == 1).sum()})")
    _print(f"  Year 2: {len(y2_sens)} patients, {n_mi_y2} MI (was {(y2[TARGET] == 1).sum()})")
    _print(f"  New EPV: {n_mi_y1 / 22:.1f}")

    # Prepare features (reuse knots from primary analysis)
    primary_knots = primary_results["knots"]
    X1s, X2s, _, _, y1s_out, y2s_out, feat_names, _, scaler_s = prepare_features(
        y1_sens, y2_sens, knots=primary_knots,
    )

    # Fit penalized LR with CV
    _print("\n  Fitting penalized LR (sensitivity)...")
    Cs = np.logspace(-4, 4, 50)
    lr_sens = LogisticRegressionCV(
        Cs=Cs, penalty="l2", scoring="neg_log_loss",
        cv=10, solver="lbfgs", max_iter=5000, random_state=SEED, refit=True,
    )
    lr_sens.fit(X1s, y1s_out)
    best_C_sens = lr_sens.C_[0]
    _print(f"  Best C: {best_C_sens:.6f} (primary: {primary_results['best_C']:.6f})")

    # Derivation and validation performance
    prob_deriv_s = lr_sens.predict_proba(X1s)[:, 1]
    prob_valid_s = lr_sens.predict_proba(X2s)[:, 1]

    auroc_deriv_s = roc_auc_score(y1s_out, prob_deriv_s)
    auroc_valid_s = roc_auc_score(y2s_out, prob_valid_s)

    cal_deriv_s = calibration_metrics(y1s_out, prob_deriv_s)
    cal_valid_s = calibration_metrics(y2s_out, prob_valid_s)

    # Bootstrap CI for validation AUROC
    rng = np.random.RandomState(SEED)
    aurocs_boot = []
    for _ in range(N_BOOT):
        idx = rng.choice(len(y2s_out), size=len(y2s_out), replace=True)
        if len(np.unique(y2s_out[idx])) < 2:
            continue
        try:
            aurocs_boot.append(roc_auc_score(y2s_out[idx], prob_valid_s[idx]))
        except ValueError:
            continue
    ci_valid_s = (
        np.percentile(aurocs_boot, 2.5) if aurocs_boot else 0,
        np.percentile(aurocs_boot, 97.5) if aurocs_boot else 0,
    )

    _print(f"\n  Sensitivity results:")
    _print(f"    Derivation AUROC: {auroc_deriv_s:.3f} (primary: {primary_results['auroc_deriv']:.3f})")
    _print(f"    Validation AUROC: {auroc_valid_s:.3f} ({ci_valid_s[0]:.3f}–{ci_valid_s[1]:.3f}) "
           f"(primary: {primary_results['auroc_valid']:.3f})")
    _print(f"    Derivation cal slope: {cal_deriv_s['slope']:.3f} "
           f"(primary: {primary_results['cal_deriv_slope']:.3f})")
    _print(f"    Validation cal slope: {cal_valid_s['slope']:.3f} "
           f"(primary: {primary_results['cal_valid_slope']:.3f})")
    _print(f"    Validation Brier: {cal_valid_s['brier']:.4f} "
           f"(primary: {primary_results['brier_valid']:.4f})")

    # Save sensitivity table
    rows = [
        {
            "Analysis": "Primary (all MI)",
            "MI (Y1/Y2)": f"{(y1[TARGET] == 1).sum()} / {(y2[TARGET] == 1).sum()}",
            "EPV": f"{(y1[TARGET] == 1).sum() / 22:.1f}",
            "Best C": f"{primary_results['best_C']:.4f}",
            "Deriv AUROC": f"{primary_results['auroc_deriv']:.3f}",
            "Valid AUROC (95% CI)": (
                f"{primary_results['auroc_valid']:.3f} "
                f"({primary_results['ci_valid'][0]:.3f}–{primary_results['ci_valid'][1]:.3f})"
            ),
            "Valid cal slope": f"{primary_results['cal_valid_slope']:.3f}",
            "Valid Brier": f"{primary_results['brier_valid']:.4f}",
        },
        {
            "Analysis": "Excluding cerebellar ataxia",
            "MI (Y1/Y2)": f"{n_mi_y1} / {n_mi_y2}",
            "EPV": f"{n_mi_y1 / 22:.1f}",
            "Best C": f"{best_C_sens:.4f}",
            "Deriv AUROC": f"{auroc_deriv_s:.3f}",
            "Valid AUROC (95% CI)": f"{auroc_valid_s:.3f} ({ci_valid_s[0]:.3f}–{ci_valid_s[1]:.3f})",
            "Valid cal slope": f"{cal_valid_s['slope']:.3f}",
            "Valid Brier": f"{cal_valid_s['brier']:.4f}",
        },
    ]

    tbl = pd.DataFrame(rows)
    tbl.to_csv(TABLES / "table_ha_penalized_sensitivity_ca.csv", index=False)
    _print(f"\n  Saved table_ha_penalized_sensitivity_ca.csv")

    return {
        "auroc_deriv": auroc_deriv_s,
        "auroc_valid": auroc_valid_s,
        "ci_valid": ci_valid_s,
        "cal_deriv": cal_deriv_s,
        "cal_valid": cal_valid_s,
        "best_C": best_C_sens,
        "n_ca_excluded": n_ca_only,
        "n_mi_y1": n_mi_y1,
        "n_mi_y2": n_mi_y2,
    }


# ===================================================================
# SECTION 12: MAIN
# ===================================================================

def main():
    t_start = time.time()
    _print("=" * 65)
    _print("Brain AttACKs: Penalized LR for Headache Model (Manuscript 2)")
    _print("TRIPOD-compliant ridge regression with bootstrap validation")
    _print("=" * 65)

    # --- Load data ---
    ha, y1, y2 = load_data()

    # --- Prepare features ---
    _print("\n=== Section 3: Feature Preparation ===")
    X1, X2, X1_raw, X2_raw, y1_out, y2_out, feature_names, knots, scaler = prepare_features(y1, y2)

    # --- Fit penalized LR ---
    lr_cv, best_C = fit_penalized_lr(X1, y1_out)

    # --- Bootstrap internal validation ---
    opt_data = bootstrap_optimism(X1, y1_out, best_C)

    # --- Temporal validation ---
    pen_valid = temporal_validation(lr_cv, X1, X2, y1_out, y2_out)

    # --- Coefficient table ---
    coef_tbl, betas, boot_betas = coefficient_table(X1, y1_out, best_C, feature_names, scaler)

    # --- Refit comparison models ---
    comp = refit_comparison_models(y1, y2)

    # --- Decision curve ---
    dca_data = decision_curve_analysis(
        y2_out,
        pen_valid["prob_valid"],
        y_prob_lr=comp["lr_valid_prob"],
        y_prob_xgb=comp["xgb_valid_prob"],
    )

    # --- Save tables ---
    save_performance_table(opt_data, pen_valid)
    save_comparison_table(pen_valid, opt_data, comp)

    # --- Generate figures ---
    _print("\n=== Section 10: Figures ===")
    fig_roc(pen_valid, comp, y1_out, y2_out)
    fig_calibration(pen_valid, comp, y1_out, y2_out)
    fig_forest(coef_tbl)
    fig_dca(dca_data)
    fig_age_spline(lr_cv, feature_names, scaler, knots)
    fig_bootstrap(opt_data)

    # --- Sensitivity analysis: exclude cerebellar ataxia ---
    primary_results = {
        "best_C": best_C,
        "knots": knots,
        "auroc_deriv": pen_valid["auroc_deriv"],
        "auroc_valid": pen_valid["auroc_valid"],
        "ci_valid": pen_valid["ci_valid"],
        "cal_deriv_slope": pen_valid["cal_deriv"]["slope"],
        "cal_valid_slope": pen_valid["cal_valid"]["slope"],
        "brier_valid": pen_valid["cal_valid"]["brier"],
    }
    sens_ca = sensitivity_exclude_cerebellar_ataxia(ha, y1, y2, primary_results)

    # --- Summary ---
    elapsed = time.time() - t_start
    _print(f"\n{'=' * 65}")
    _print(f"SUMMARY")
    _print(f"{'=' * 65}")
    _print(f"  Penalized LR (ridge, C={best_C:.4f})")
    _print(f"  Derivation AUROC: {opt_data['apparent_auroc']:.3f} "
           f"(optimism-corrected: {opt_data['corrected_auroc']:.3f})")
    _print(f"  Derivation cal slope: {opt_data['apparent_slope']:.3f} "
           f"(optimism-corrected: {opt_data['corrected_slope']:.3f})")
    _print(f"  Validation AUROC: {pen_valid['auroc_valid']:.3f} "
           f"({pen_valid['ci_valid'][0]:.3f}–{pen_valid['ci_valid'][1]:.3f})")
    _print(f"  Validation cal slope: {pen_valid['cal_valid']['slope']:.3f}")
    _print(f"  Simple LR validation slope: {comp['results']['lr_valid']['cal']['slope']:.3f}")
    _print(f"  Bootstrap iterations: {opt_data['n_successful']}/{N_BOOT}")
    if sens_ca:
        _print(f"  Sensitivity (excl CA): valid AUROC {sens_ca['auroc_valid']:.3f}, "
               f"slope {sens_ca['cal_valid']['slope']:.3f}")
    _print(f"")
    _print(f"  Tables: 5 in outputs/tables/")
    _print(f"  Figures: 6 in outputs/figures/")
    _print(f"  Elapsed: {elapsed:.0f}s")
    _print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
