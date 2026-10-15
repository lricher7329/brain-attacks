"""
19-ciip-ms2-reanalysis.py — Consolidated MS2 Re-analysis (Headache) with CIIP

Headache sub-cohort re-analysis using the CIIP outcome (outcome_ciip).
Reuses code patterns from scripts 13 + 16.

Analyses:
    1. Headache sub-cohort assembly (sx_any_headache, exclude prior headache)
    2. LR + XGBoost temporal validation
    3. Penalized LR (ridge, RCS age, bootstrap optimism correction)
    4. Calibration, DCA, coefficient table with bootstrap CIs
    5. Built-in MI vs CIIP comparison

Inputs:
    data/processed/brain_attacks_ciip.parquet

Outputs:
    outputs/tables/table_ciip_ms2_performance.csv
    outputs/tables/table_ciip_ms2_pen_coefficients.csv
    outputs/tables/table_ciip_ms2_pen_performance.csv
    outputs/tables/table_ciip_ms2_comparison.csv
    outputs/figures/fig_ciip_ms2_roc.png
    outputs/figures/fig_ciip_ms2_calibration.png
    outputs/figures/fig_ciip_ms2_forest.png
    outputs/figures/fig_ciip_ms2_dca.png
"""

import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    brier_score_loss, confusion_matrix, fbeta_score,
    make_scorer, roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

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

TARGET = "outcome_ciip"

plt.rcParams.update({
    "font.size": 10,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _print(msg=""):
    print(msg, flush=True)


# ===================================================================
# PREDICTOR DEFINITIONS (from scripts 13/16)
# ===================================================================

EXTENDED_PREDICTORS = [
    "age_under_12", "gender", "ems", "exam_altered_loc",
    "sx_focal_weakness", "sx_ataxia", "sx_speech", "sx_visual",
    "ha_progressive_days", "ha_vomiting", "sx_fever",
    "ha_photophobia", "ha_phonophobia", "ha_maximal_onset",
    "ha_abrupt_onset", "ha_gradual_onset", "ha_nausea",
    "exam_papilledema", "exam_meningismus", "ctas", "any_prev_dx",
]

BINARY_PREDICTORS = [
    "gender", "ems", "exam_altered_loc",
    "sx_focal_weakness", "sx_ataxia", "sx_speech", "sx_visual",
    "ha_progressive_days", "ha_vomiting", "sx_fever",
    "ha_photophobia", "ha_phonophobia", "ha_maximal_onset",
    "ha_abrupt_onset", "ha_gradual_onset", "ha_nausea",
    "exam_papilledema", "exam_meningismus", "ctas", "any_prev_dx",
]

PREDICTOR_LABELS = {
    "gender": "Male sex", "ems": "EMS transport",
    "exam_altered_loc": "Altered LOC (exam)",
    "sx_focal_weakness": "Focal weakness", "sx_ataxia": "Ataxia",
    "sx_speech": "Speech impairment", "sx_visual": "Visual impairment",
    "ha_progressive_days": "Progressive HA (days)", "ha_vomiting": "Vomiting",
    "sx_fever": "Fever", "ha_photophobia": "Photophobia",
    "ha_phonophobia": "Phonophobia", "ha_maximal_onset": "Maximal at onset",
    "ha_abrupt_onset": "Abrupt onset", "ha_gradual_onset": "Gradual onset",
    "ha_nausea": "Nausea", "exam_papilledema": "Papilledema",
    "exam_meningismus": "Meningismus", "ctas": "CTAS score",
    "any_prev_dx": "Any prior neuro Dx", "age": "Age (linear)",
    "age_rcs1": "Age (nonlinear RCS)", "age_under_12": "Age < 12",
}


# ===================================================================
# RCS HELPER (from script 16)
# ===================================================================

def rcs_basis(x, knots):
    """Restricted cubic spline basis (Harrell 2001)."""
    x = np.asarray(x, dtype=float)
    knots = np.sort(np.asarray(knots, dtype=float))
    k = len(knots)

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
    """3 RCS knots at 10th, 50th, 90th percentiles."""
    return np.percentile(age_values, [10, 50, 90])


# ===================================================================
# CALIBRATION HELPER
# ===================================================================

def calibration_metrics(y_true, y_prob):
    """CITL, calibration slope, Brier."""
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


# ===================================================================
# 1. LOAD & SUBSET
# ===================================================================

def load_data():
    """Load CIIP dataset and assemble headache sub-cohort."""
    df = pd.read_parquet(DATA / "brain_attacks_ciip.parquet")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    _print(f"Full cohort: {len(df)}")

    ha = df[df["sx_any_headache"] == 1].copy()
    _print(f"Headache patients: {len(ha)}")

    # Exclude prior headache disorder (migraine=4, recurring headache=5)
    prior_ha = (
        (ha.get("previous_diagnoses___4", pd.Series(0, index=ha.index)) == 1) |
        (ha.get("previous_diagnoses___5", pd.Series(0, index=ha.index)) == 1)
    )
    n_excluded = prior_ha.sum()
    ha_clean = ha[~prior_ha].copy()
    _print(f"  Excluded prior headache disorder: {n_excluded}")
    _print(f"  Analysis cohort: {len(ha_clean)}")

    ciip = (ha_clean[TARGET] == 1).sum()
    mi = (ha_clean["outcome_med_important"] == 1).sum()
    _print(f"  CIIP: {ciip} ({100 * ciip / len(ha_clean):.1f}%)")
    _print(f"  MI (original): {mi} ({100 * mi / len(ha_clean):.1f}%)")

    y1 = ha_clean[ha_clean["year"] == 1].copy()
    y2 = ha_clean[ha_clean["year"] == 2].copy()
    _print(f"  Year 1: {len(y1)} ({(y1[TARGET] == 1).sum()} CIIP)")
    _print(f"  Year 2: {len(y2)} ({(y2[TARGET] == 1).sum()} CIIP)")

    return df, ha_clean, y1, y2


# ===================================================================
# 2. SIMPLE LR + XGBOOST TEMPORAL VALIDATION
# ===================================================================

def simple_models(ha, y1, y2):
    """LR + XGBoost temporal validation (matching script 13 patterns)."""
    _print("\n=== Simple LR + XGBoost ===")

    features = [f for f in EXTENDED_PREDICTORS if f in ha.columns]
    X1 = y1[features].fillna(0).values
    X2 = y2[features].fillna(0).values
    y1_out = y1[TARGET].values
    y2_out = y2[TARGET].values

    # --- LR ---
    scaler = StandardScaler()
    X1_s = scaler.fit_transform(X1)
    X2_s = scaler.transform(X2)

    lr = LogisticRegression(max_iter=1000, random_state=SEED, penalty="l2", C=1.0)
    lr.fit(X1_s, y1_out)
    lr_d_prob = lr.predict_proba(X1_s)[:, 1]
    lr_v_prob = lr.predict_proba(X2_s)[:, 1]

    lr_d_auroc = roc_auc_score(y1_out, lr_d_prob)
    lr_v_auroc = roc_auc_score(y2_out, lr_v_prob)

    _print(f"  LR Derivation AUROC: {lr_d_auroc:.3f}")
    _print(f"  LR Validation AUROC: {lr_v_auroc:.3f}")

    # --- XGBoost ---
    f2_scorer = make_scorer(fbeta_score, beta=2, zero_division=0)
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
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                        random_state=SEED, verbosity=0)
    grid = GridSearchCV(xgb, param_grid, scoring=f2_scorer,
                        cv=inner_cv, n_jobs=-1, refit=True)
    grid.fit(X1_sm, y1_sm)

    xgb_d_prob = grid.predict_proba(X1)[:, 1]
    xgb_v_prob = grid.predict_proba(X2)[:, 1]

    xgb_d_auroc = roc_auc_score(y1_out, xgb_d_prob)
    xgb_v_auroc = roc_auc_score(y2_out, xgb_v_prob)

    _print(f"  XGBoost Derivation AUROC: {xgb_d_auroc:.3f}")
    _print(f"  XGBoost Validation AUROC: {xgb_v_auroc:.3f}")

    # Bootstrap CIs for validation
    rng = np.random.RandomState(SEED)
    lr_boots, xgb_boots = [], []
    for _ in range(N_BOOT):
        idx = rng.choice(len(y2_out), size=len(y2_out), replace=True)
        if len(np.unique(y2_out[idx])) < 2:
            continue
        try:
            lr_boots.append(roc_auc_score(y2_out[idx], lr_v_prob[idx]))
            xgb_boots.append(roc_auc_score(y2_out[idx], xgb_v_prob[idx]))
        except ValueError:
            continue

    lr_ci = (np.percentile(lr_boots, 2.5), np.percentile(lr_boots, 97.5)) if lr_boots else (0, 0)
    xgb_ci = (np.percentile(xgb_boots, 2.5), np.percentile(xgb_boots, 97.5)) if xgb_boots else (0, 0)

    # Save performance table
    def _metrics(y_true, y_prob):
        y_pred = (y_prob >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return {
            "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "ppv": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "npv": tn / (tn + fn) if (tn + fn) > 0 else 0,
            "auroc": roc_auc_score(y_true, y_prob),
        }

    rows = []
    for label, y_t, y_p, ci in [
        ("LR Derivation (Y1)", y1_out, lr_d_prob, None),
        ("LR Validation (Y2)", y2_out, lr_v_prob, lr_ci),
        ("XGBoost Derivation (Y1)", y1_out, xgb_d_prob, None),
        ("XGBoost Validation (Y2)", y2_out, xgb_v_prob, xgb_ci),
    ]:
        m = _metrics(y_t, y_p)
        ci_str = f"({ci[0]:.3f}-{ci[1]:.3f})" if ci else ""
        rows.append({
            "Model": label, "N": len(y_t), "Events": int(y_t.sum()),
            "AUROC": f"{m['auroc']:.3f}", "AUROC 95% CI": ci_str,
            "Sensitivity": f"{100 * m['sensitivity']:.1f}%",
            "Specificity": f"{100 * m['specificity']:.1f}%",
            "PPV": f"{100 * m['ppv']:.1f}%",
            "NPV": f"{100 * m['npv']:.1f}%",
        })

    tbl = pd.DataFrame(rows)
    tbl.to_csv(TABLES / "table_ciip_ms2_performance.csv", index=False)
    _print(f"  Saved table_ciip_ms2_performance.csv")

    return {
        "lr_d_prob": lr_d_prob, "lr_v_prob": lr_v_prob,
        "xgb_d_prob": xgb_d_prob, "xgb_v_prob": xgb_v_prob,
        "y1_out": y1_out, "y2_out": y2_out,
        "lr_v_auroc": lr_v_auroc, "xgb_v_auroc": xgb_v_auroc,
        "lr_ci": lr_ci, "xgb_ci": xgb_ci,
    }


# ===================================================================
# 3. PENALIZED LR (Ridge + RCS age + bootstrap optimism)
# ===================================================================

def prepare_pen_features(y1, y2, knots=None):
    """Build feature matrices with binary predictors + age RCS."""
    features_binary = [f for f in BINARY_PREDICTORS if f in y1.columns]

    X1_bin = y1[features_binary].fillna(0).values
    X2_bin = y2[features_binary].fillna(0).values

    age1 = y1["age"].values.astype(float)
    age2 = y2["age"].values.astype(float)
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

    scaler = StandardScaler()
    X1_s = scaler.fit_transform(X1)
    X2_s = scaler.transform(X2)

    n_params = len(feature_names)
    n_events = y1_out.sum()
    _print(f"  Features: {n_params}, Events: {int(n_events)}, EPV: {n_events / n_params:.1f}")

    return X1_s, X2_s, y1_out, y2_out, feature_names, knots, scaler


def fit_penalized_lr(X1, y1_out):
    """Fit ridge LR with CV-tuned regularization."""
    _print("\n=== Penalized LR Fitting ===")

    lr_cv = LogisticRegressionCV(
        Cs=np.logspace(-4, 4, 50),
        penalty="l2", scoring="neg_log_loss",
        cv=10, solver="lbfgs", max_iter=5000,
        random_state=SEED, refit=True,
    )
    lr_cv.fit(X1, y1_out)

    best_C = lr_cv.C_[0]
    _print(f"  Best C: {best_C:.6f}")
    return lr_cv, best_C


def bootstrap_optimism(X_orig, y_orig, best_C):
    """Harrell optimism-corrected bootstrap validation."""
    _print(f"\n=== Bootstrap Optimism ({N_BOOT} iterations) ===")

    rng = np.random.RandomState(SEED)
    n = len(y_orig)

    lr_orig = LogisticRegression(penalty="l2", C=best_C, solver="lbfgs",
                                 max_iter=5000, random_state=SEED)
    lr_orig.fit(X_orig, y_orig)
    prob_orig = lr_orig.predict_proba(X_orig)[:, 1]

    apparent_auroc = roc_auc_score(y_orig, prob_orig)
    apparent_cal = calibration_metrics(y_orig, prob_orig)

    opt_auroc, opt_slope, opt_brier = [], [], []

    for b in range(N_BOOT):
        if (b + 1) % 100 == 0:
            _print(f"  Bootstrap {b + 1}/{N_BOOT}...")

        idx = rng.choice(n, size=n, replace=True)
        X_boot, y_boot = X_orig[idx], y_orig[idx]
        if len(np.unique(y_boot)) < 2:
            continue

        try:
            lr_b = LogisticRegressionCV(
                Cs=np.logspace(-4, 4, 20), penalty="l2", scoring="neg_log_loss",
                cv=5, solver="lbfgs", max_iter=5000, random_state=SEED, refit=True,
            )
            lr_b.fit(X_boot, y_boot)

            prob_boot = lr_b.predict_proba(X_boot)[:, 1]
            prob_test = lr_b.predict_proba(X_orig)[:, 1]

            boot_auroc = roc_auc_score(y_boot, prob_boot)
            test_auroc = roc_auc_score(y_orig, prob_test)

            boot_cal = calibration_metrics(y_boot, prob_boot)
            test_cal = calibration_metrics(y_orig, prob_test)

            opt_auroc.append(boot_auroc - test_auroc)
            if not np.isnan(boot_cal["slope"]) and not np.isnan(test_cal["slope"]):
                opt_slope.append(boot_cal["slope"] - test_cal["slope"])
            opt_brier.append(boot_cal["brier"] - test_cal["brier"])
        except Exception:
            continue

    mean_opt_auroc = np.mean(opt_auroc) if opt_auroc else 0
    mean_opt_slope = np.mean(opt_slope) if opt_slope else 0

    corrected_auroc = apparent_auroc - mean_opt_auroc
    corrected_slope = apparent_cal["slope"] - mean_opt_slope

    _print(f"  Apparent AUROC: {apparent_auroc:.3f}")
    _print(f"  Corrected AUROC: {corrected_auroc:.3f}")
    _print(f"  Apparent slope: {apparent_cal['slope']:.3f}")
    _print(f"  Corrected slope: {corrected_slope:.3f}")

    return {
        "apparent_auroc": apparent_auroc,
        "apparent_slope": apparent_cal["slope"],
        "apparent_brier": apparent_cal["brier"],
        "corrected_auroc": corrected_auroc,
        "corrected_slope": corrected_slope,
        "n_successful": len(opt_auroc),
    }


def penalized_temporal_validation(lr_model, X1, X2, y1_out, y2_out):
    """Temporal validation for penalized LR."""
    _print("\n=== Penalized LR Temporal Validation ===")

    prob_d = lr_model.predict_proba(X1)[:, 1]
    prob_v = lr_model.predict_proba(X2)[:, 1]

    auroc_d = roc_auc_score(y1_out, prob_d)
    auroc_v = roc_auc_score(y2_out, prob_v)
    cal_d = calibration_metrics(y1_out, prob_d)
    cal_v = calibration_metrics(y2_out, prob_v)

    # Bootstrap CI
    rng = np.random.RandomState(SEED)
    boots = []
    for _ in range(N_BOOT):
        idx = rng.choice(len(y2_out), size=len(y2_out), replace=True)
        if len(np.unique(y2_out[idx])) < 2:
            continue
        try:
            boots.append(roc_auc_score(y2_out[idx], prob_v[idx]))
        except ValueError:
            continue
    ci_v = (np.percentile(boots, 2.5), np.percentile(boots, 97.5)) if boots else (0, 0)

    _print(f"  Derivation AUROC: {auroc_d:.3f}, Slope={cal_d['slope']:.3f}")
    _print(f"  Validation AUROC: {auroc_v:.3f} ({ci_v[0]:.3f}-{ci_v[1]:.3f}), Slope={cal_v['slope']:.3f}")

    return {
        "prob_d": prob_d, "prob_v": prob_v,
        "auroc_d": auroc_d, "auroc_v": auroc_v,
        "cal_d": cal_d, "cal_v": cal_v, "ci_v": ci_v,
    }


def coefficient_table(X1, y1_out, best_C, feature_names):
    """Bootstrap 95% CIs for penalized LR coefficients."""
    _print(f"\n=== Coefficient Table ({N_BOOT} bootstrap) ===")

    rng = np.random.RandomState(SEED)
    n = len(y1_out)

    lr_final = LogisticRegression(penalty="l2", C=best_C, solver="lbfgs",
                                   max_iter=5000, random_state=SEED)
    lr_final.fit(X1, y1_out)
    betas = lr_final.coef_[0]
    intercept = lr_final.intercept_[0]

    boot_betas = []
    for _ in range(N_BOOT):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y1_out[idx])) < 2:
            continue
        try:
            lr_b = LogisticRegression(penalty="l2", C=best_C, solver="lbfgs",
                                      max_iter=5000, random_state=SEED)
            lr_b.fit(X1[idx], y1_out[idx])
            boot_betas.append(lr_b.coef_[0])
        except Exception:
            continue

    boot_betas = np.array(boot_betas)

    rows = [{"Variable": "Intercept", "Label": "Intercept",
             "Beta": f"{intercept:.4f}", "Beta 95% CI": "",
             "aOR": "", "aOR 95% CI": ""}]

    for i, fname in enumerate(feature_names):
        beta = betas[i]
        aor = np.exp(beta)
        label = PREDICTOR_LABELS.get(fname, fname)

        if len(boot_betas) > 0:
            ci_lo = np.percentile(boot_betas[:, i], 2.5)
            ci_hi = np.percentile(boot_betas[:, i], 97.5)
            beta_ci = f"({ci_lo:.4f}, {ci_hi:.4f})"
            aor_ci = f"({np.exp(ci_lo):.2f}, {np.exp(ci_hi):.2f})"
        else:
            beta_ci, aor_ci = "", ""

        rows.append({
            "Variable": fname, "Label": label,
            "Beta": f"{beta:.4f}", "Beta 95% CI": beta_ci,
            "aOR": f"{aor:.2f}", "aOR 95% CI": aor_ci,
        })

    tbl = pd.DataFrame(rows)
    tbl.to_csv(TABLES / "table_ciip_ms2_pen_coefficients.csv", index=False)
    _print(f"  Saved table_ciip_ms2_pen_coefficients.csv ({len(tbl)} rows)")

    return tbl


def save_pen_performance(opt_data, pen_valid):
    """Save penalized LR performance summary."""
    rows = [
        {"Set": "Derivation (apparent)",
         "AUROC": f"{opt_data['apparent_auroc']:.3f}",
         "Slope": f"{opt_data['apparent_slope']:.3f}",
         "Brier": f"{opt_data['apparent_brier']:.4f}"},
        {"Set": "Derivation (optimism-corrected)",
         "AUROC": f"{opt_data['corrected_auroc']:.3f}",
         "Slope": f"{opt_data['corrected_slope']:.3f}",
         "Brier": ""},
        {"Set": "Temporal validation (Y2)",
         "AUROC": f"{pen_valid['auroc_v']:.3f} ({pen_valid['ci_v'][0]:.3f}-{pen_valid['ci_v'][1]:.3f})",
         "Slope": f"{pen_valid['cal_v']['slope']:.3f}",
         "Brier": f"{pen_valid['cal_v']['brier']:.4f}"},
    ]
    tbl = pd.DataFrame(rows)
    tbl.to_csv(TABLES / "table_ciip_ms2_pen_performance.csv", index=False)
    _print(f"  Saved table_ciip_ms2_pen_performance.csv")
    return tbl


# ===================================================================
# 4. MI vs CIIP COMPARISON
# ===================================================================

def mi_vs_ciip_comparison(ha, y1, y2, simple_data, pen_valid, opt_data):
    """Side-by-side MI vs CIIP comparison for headache sub-cohort."""
    _print("\n=== MI vs CIIP Comparison (Headache) ===")

    n_mi = (ha["outcome_med_important"] == 1).sum()
    n_ciip = (ha[TARGET] == 1).sum()

    rows = [
        {"Metric": "Headache cohort N", "MI": str(len(ha)), "CIIP": str(len(ha)), "Delta": "0"},
        {"Metric": "Outcome count", "MI": str(n_mi), "CIIP": str(n_ciip),
         "Delta": str(n_ciip - n_mi)},
        {"Metric": "Outcome prevalence",
         "MI": f"{100 * n_mi / len(ha):.1f}%",
         "CIIP": f"{100 * n_ciip / len(ha):.1f}%",
         "Delta": f"{100 * (n_ciip - n_mi) / len(ha):.1f}pp"},
        {"Metric": "LR validation AUROC",
         "MI": "see scripts 13/16",
         "CIIP": f"{simple_data['lr_v_auroc']:.3f}",
         "Delta": ""},
        {"Metric": "XGBoost validation AUROC",
         "MI": "see scripts 13/16",
         "CIIP": f"{simple_data['xgb_v_auroc']:.3f}",
         "Delta": ""},
        {"Metric": "Penalized LR validation AUROC",
         "MI": "see script 16",
         "CIIP": f"{pen_valid['auroc_v']:.3f} ({pen_valid['ci_v'][0]:.3f}-{pen_valid['ci_v'][1]:.3f})",
         "Delta": ""},
        {"Metric": "Penalized LR optimism-corrected AUROC",
         "MI": "see script 16",
         "CIIP": f"{opt_data['corrected_auroc']:.3f}",
         "Delta": ""},
        {"Metric": "Penalized LR validation slope",
         "MI": "see script 16",
         "CIIP": f"{pen_valid['cal_v']['slope']:.3f}",
         "Delta": ""},
        {"Metric": "Penalized LR validation Brier",
         "MI": "see script 16",
         "CIIP": f"{pen_valid['cal_v']['brier']:.4f}",
         "Delta": ""},
    ]

    tbl = pd.DataFrame(rows)
    tbl.to_csv(TABLES / "table_ciip_ms2_comparison.csv", index=False)
    _print(f"  Saved table_ciip_ms2_comparison.csv")
    return tbl


# ===================================================================
# 5. FIGURES
# ===================================================================

def fig_roc(simple_data, pen_valid, y1_out, y2_out):
    """ROC curves: penalized LR vs simple LR vs XGBoost."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # Derivation
    ax = axes[0]
    ax.set_title("Derivation (Year 1)", fontweight="bold")
    for label, y_p, color, ls in [
        ("Penalized LR", pen_valid["prob_d"], "#2ca02c", "-"),
        ("Simple LR", simple_data["lr_d_prob"], "#1f77b4", "--"),
        ("XGBoost", simple_data["xgb_d_prob"], "#d62728", "-."),
    ]:
        fpr, tpr, _ = roc_curve(y1_out, y_p)
        auc = roc_auc_score(y1_out, y_p)
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})", color=color, linestyle=ls, linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.legend(loc="lower right", fontsize=9)

    # Validation
    ax = axes[1]
    ax.set_title("Validation (Year 2)", fontweight="bold")
    for label, y_p, color, ls in [
        ("Penalized LR", pen_valid["prob_v"], "#2ca02c", "-"),
        ("Simple LR", simple_data["lr_v_prob"], "#1f77b4", "--"),
        ("XGBoost", simple_data["xgb_v_prob"], "#d62728", "-."),
    ]:
        fpr, tpr, _ = roc_curve(y2_out, y_p)
        auc = roc_auc_score(y2_out, y_p)
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})", color=color, linestyle=ls, linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.legend(loc="lower right", fontsize=9)

    fig.suptitle("CIIP Headache Model: ROC Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_ciip_ms2_roc.png", bbox_inches="tight")
    plt.close()
    _print("  Saved fig_ciip_ms2_roc.png")


def fig_calibration(pen_valid, simple_data, y1_out, y2_out):
    """2x2 calibration plots."""
    panels = [
        ("Penalized LR\nDerivation (Y1)", y1_out, pen_valid["prob_d"], "#2ca02c"),
        ("Penalized LR\nValidation (Y2)", y2_out, pen_valid["prob_v"], "#2ca02c"),
        ("Simple LR\nValidation (Y2)", y2_out, simple_data["lr_v_prob"], "#1f77b4"),
        ("XGBoost\nValidation (Y2)", y2_out, simple_data["xgb_v_prob"], "#d62728"),
    ]

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

        txt = f"CITL={cal['citl']:.3f}\nSlope={cal['slope']:.2f}\nBrier={cal['brier']:.4f}"
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=8,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed proportion")
        ax.set_title(title, fontsize=10)

    fig.suptitle("CIIP Headache: Calibration Plots", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_ciip_ms2_calibration.png", bbox_inches="tight")
    plt.close()
    _print("  Saved fig_ciip_ms2_calibration.png")


def fig_forest(coef_tbl):
    """Forest plot of penalized aORs."""
    df = coef_tbl[coef_tbl["Variable"] != "Intercept"].copy()
    df["aOR_val"] = df["aOR"].astype(float)

    ci_lo, ci_hi = [], []
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
    df = df.sort_values("aOR_val", ascending=True).reset_index(drop=True)

    n = len(df)
    y_pos = np.arange(n)

    fig, ax = plt.subplots(figsize=(9, max(6, n * 0.4)))
    colors = ["#c0392b" if aor > 1.0 else "#2980b9" for aor in df["aOR_val"]]

    xerr_lo = df["aOR_val"].values - df["ci_lo"].values
    xerr_hi = df["ci_hi"].values - df["aOR_val"].values
    ax.errorbar(df["aOR_val"], y_pos, xerr=[xerr_lo, xerr_hi],
                fmt="none", ecolor="#555555", elinewidth=1.2, capsize=3)
    ax.scatter(df["aOR_val"], y_pos, c=colors, s=60, zorder=5, edgecolors="white", linewidths=0.5)

    ax.axvline(x=1.0, color="black", linewidth=1, linestyle="-", alpha=0.8)

    labels = [PREDICTOR_LABELS.get(v, v) for v in df["Variable"]]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Adjusted Odds Ratio (aOR)")
    ax.set_title("CIIP Headache: Penalized LR aORs (bootstrap 95% CIs)",
                 fontsize=12, fontweight="bold")

    for i, (aor, lo, hi) in enumerate(zip(df["aOR_val"], df["ci_lo"], df["ci_hi"])):
        ax.text(ax.get_xlim()[1] * 0.97, i, f"{aor:.2f} ({lo:.2f}-{hi:.2f})",
                va="center", ha="right", fontsize=7.5, color="#333333")

    ax.set_ylim(-0.5, n + 1.5)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_ciip_ms2_forest.png", bbox_inches="tight")
    plt.close()
    _print("  Saved fig_ciip_ms2_forest.png")


def fig_dca(pen_valid, simple_data, y2_out):
    """Decision curve on validation set."""
    thresholds = np.arange(0.01, 0.51, 0.005)
    n = len(y2_out)
    prevalence = y2_out.mean()

    def _nb(y_prob, t):
        pred_pos = y_prob >= t
        tp = (pred_pos & (y2_out == 1)).sum()
        fp = (pred_pos & (y2_out == 0)).sum()
        return tp / n - fp / n * (t / (1 - t))

    nb_pen = [_nb(pen_valid["prob_v"], t) for t in thresholds]
    nb_lr = [_nb(simple_data["lr_v_prob"], t) for t in thresholds]
    nb_xgb = [_nb(simple_data["xgb_v_prob"], t) for t in thresholds]
    nb_all = [prevalence - (1 - prevalence) * (t / (1 - t)) for t in thresholds]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds * 100, nb_all, "k-", alpha=0.5, linewidth=1.5, label="Investigate all")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1, label="Investigate none")
    ax.plot(thresholds * 100, nb_pen, "-", color="#2ca02c", linewidth=2.5, label="Penalized LR")
    ax.plot(thresholds * 100, nb_lr, "--", color="#1f77b4", linewidth=2, label="Simple LR")
    ax.plot(thresholds * 100, nb_xgb, "-.", color="#d62728", linewidth=2, label="XGBoost")

    ax.set_xlabel("Threshold probability (%)")
    ax.set_ylabel("Net benefit")
    ax.set_title("CIIP Headache: Decision Curve (Year 2)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, 50)
    ax.grid(alpha=0.2)

    ax.axvline(x=prevalence * 100, color="gray", linestyle=":", alpha=0.4)
    ax.text(prevalence * 100 + 0.8, ax.get_ylim()[0] * 0.3,
            f"CIIP rate\n({100 * prevalence:.1f}%)", fontsize=8, color="gray")

    fig.tight_layout()
    fig.savefig(FIGURES / "fig_ciip_ms2_dca.png", bbox_inches="tight")
    plt.close()
    _print("  Saved fig_ciip_ms2_dca.png")


# ===================================================================
# MAIN
# ===================================================================

def main():
    t_start = time.time()
    _print("=" * 65)
    _print("Brain AttACKs: MS2 Headache Re-analysis with CIIP Outcome")
    _print("=" * 65)

    # 1. Load data
    full_df, ha, y1, y2 = load_data()

    # 2. Simple LR + XGBoost
    simple_data = simple_models(ha, y1, y2)

    # 3. Penalized LR
    _print("\n=== Penalized LR (Ridge + RCS age) ===")
    X1_pen, X2_pen, y1_pen, y2_pen, feat_names, knots, scaler = prepare_pen_features(y1, y2)
    lr_cv, best_C = fit_penalized_lr(X1_pen, y1_pen)

    # 4. Bootstrap optimism correction
    opt_data = bootstrap_optimism(X1_pen, y1_pen, best_C)

    # 5. Penalized temporal validation
    pen_valid = penalized_temporal_validation(lr_cv, X1_pen, X2_pen, y1_pen, y2_pen)

    # 6. Coefficient table
    coef_tbl = coefficient_table(X1_pen, y1_pen, best_C, feat_names)

    # 7. Save penalized performance table
    save_pen_performance(opt_data, pen_valid)

    # 8. MI vs CIIP comparison
    mi_vs_ciip_comparison(ha, y1, y2, simple_data, pen_valid, opt_data)

    # 9. Figures
    _print("\n=== Generating Figures ===")
    fig_roc(simple_data, pen_valid, y1_pen, y2_pen)
    fig_calibration(pen_valid, simple_data, y1_pen, y2_pen)
    fig_forest(coef_tbl)
    fig_dca(pen_valid, simple_data, y2_pen)

    elapsed = time.time() - t_start
    _print(f"\n{'=' * 65}")
    _print(f"Done in {elapsed:.0f}s.")
    _print(f"  CIIP headache: {(ha[TARGET] == 1).sum()} ({100 * (ha[TARGET] == 1).mean():.1f}%)")
    _print(f"  Penalized LR validation AUROC: {pen_valid['auroc_v']:.3f}")
    _print(f"  Penalized LR corrected AUROC: {opt_data['corrected_auroc']:.3f}")
    _print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
