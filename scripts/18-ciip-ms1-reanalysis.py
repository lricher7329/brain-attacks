"""
18-ciip-ms1-reanalysis.py — Consolidated MS1 Re-analysis with CIIP Outcome

Full-cohort re-analysis using the CIIP outcome (outcome_ciip) instead of MI
(outcome_med_important). Reuses code patterns from scripts 02-12.

Analyses:
    1. Descriptive statistics (demographics, CIIP diagnosis breakdown)
    2. XGBoost + LR prediction (10-fold nested CV with SMOTE)
    3. Temporal validation (Year 1 -> Year 2)
    4. Clinical prediction rule (10-predictor integer-weighted score)
    5. Calibration (CITL, slope, Brier, bootstrap CIs)
    6. Decision curve analysis
    7. Built-in MI vs CIIP comparison table

Inputs:
    data/processed/brain_attacks_ciip.parquet

Outputs:
    outputs/tables/table_ciip_ms1_performance.csv
    outputs/tables/table_ciip_ms1_feature_importance.csv
    outputs/tables/table_ciip_ms1_temporal.csv
    outputs/tables/table_ciip_ms1_calibration.csv
    outputs/tables/table_ciip_ms1_dca.csv
    outputs/tables/table_ciip_ms1_comparison.csv
    outputs/figures/fig_ciip_ms1_roc.png
    outputs/figures/fig_ciip_ms1_calibration.png
    outputs/figures/fig_ciip_ms1_dca.png
    outputs/figures/fig_ciip_ms1_feature_importance.png
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix,
    fbeta_score, make_scorer, roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
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
# FEATURE DEFINITIONS (same as 03-prediction.py)
# ===================================================================

SYMPTOM_FEATURES = [
    "sx_focal_weakness", "sx_sensory", "sx_speech", "sx_visual",
    "sx_eye_movement", "sx_altered_consciousness", "sx_collapse",
    "sx_seizure", "sx_ataxia", "sx_any_headache", "sx_fever",
]
OTHER_SX_FEATURES = [
    "other_pain", "other_movement", "other_psych", "other_gi",
    "other_cranial_nerve", "other_inflammation", "other_respiratory",
    "other_infection", "other_constitutional", "other_urinary",
    "other_dermatological", "other_hemorrhage", "other_syncope",
    "other_cardiac", "other_gyn", "other_ophthalmology",
    "other_sleep", "other_hematological",
]
EXAM_FEATURES = ["exam_papilledema", "exam_meningismus", "exam_altered_loc"]
HEADACHE_FEATURES = [
    "ha_nausea", "ha_vomiting", "ha_photophobia", "ha_phonophobia",
    "ha_maximal_onset", "ha_progressive_days", "ha_progressive_months",
    "ha_non_progressive", "ha_abrupt_onset", "ha_gradual_onset",
    "ha_bilateral", "ha_unilateral", "ha_frontal", "ha_temporal", "ha_occipital",
]
DEMOGRAPHIC_FEATURES = ["age", "gender", "ems", "ctas", "temp", "hr"]
OTHER_FEATURES = ["any_prev_dx", "age_under_12"]

TOP_PREDICTORS = [
    "sx_ataxia", "exam_altered_loc", "ems", "sx_fever",
    "sx_focal_weakness", "exam_meningismus", "sx_collapse",
    "gender", "sx_seizure", "exam_papilledema",
]


def get_feature_names():
    return (SYMPTOM_FEATURES + OTHER_SX_FEATURES + EXAM_FEATURES +
            HEADACHE_FEATURES + DEMOGRAPHIC_FEATURES + OTHER_FEATURES)


# ===================================================================
# 1. LOAD DATA
# ===================================================================

def load_data():
    """Load CIIP dataset."""
    df = pd.read_parquet(DATA / "brain_attacks_ciip.parquet")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    features = get_feature_names()
    features = [f for f in features if f in df.columns]
    X = df[features].fillna(0).copy()
    y = df[TARGET].values

    _print(f"Loaded: {len(df)} records, {len(features)} features")
    _print(f"  CIIP: {y.sum()} ({100 * y.mean():.1f}%), non-CIIP: {len(y) - y.sum()}")
    _print(f"  MI (preserved): {(df['outcome_med_important'] == 1).sum()}")

    return df, X, y, features


# ===================================================================
# 2. DESCRIPTIVE STATISTICS
# ===================================================================

def descriptive_stats(df):
    """CIIP diagnosis breakdown."""
    _print("\n=== CIIP Diagnosis Breakdown ===")

    mi_labels = {
        1: "Stroke", 2: "ICH", 3: "Vascular malformation",
        4: "Vasculitis", 5: "Venous thrombosis", 6: "Demyelinating",
        7: "Infection", 8: "Hydrocephalus", 9: "Neoplasm",
        10: "Inborn error", 11: "Other MI", 12: "IIH", 13: "CA",
    }

    ciip_cases = df[df[TARGET] == 1]
    n_ciip = len(ciip_cases)

    for code, label in sorted(mi_labels.items()):
        col = f"med_imp_dx___{code}"
        if col in df.columns:
            n = (ciip_cases[col] == 1).sum()
            if n > 0:
                _print(f"  {label}: {n} ({100 * n / n_ciip:.1f}%)")


# ===================================================================
# 3. XGBOOST NESTED CV (10-fold outer, 5-fold inner)
# ===================================================================

def _find_f2_threshold(y_true, y_prob):
    """Find threshold that maximizes F2-score."""
    best_f2, best_thresh = 0, 0.5
    for thresh in np.arange(0.01, 0.99, 0.01):
        y_pred = (y_prob >= thresh).astype(int)
        f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2 = f2
            best_thresh = thresh
    return best_thresh, best_f2


def _eval_metrics(y_true, y_pred, y_prob):
    """Standard metrics for a fold/split."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "ppv": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "npv": tn / (tn + fn) if (tn + fn) > 0 else 0,
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "auroc": roc_auc_score(y_true, y_prob),
    }


def xgboost_nested_cv(X, y, features):
    """XGBoost with nested CV: 10-fold outer, 5-fold inner with SMOTE search."""
    _print("\n=== XGBoost Nested CV ===")

    smote_ratios = [0.5, 0.75, 1.0]
    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
    }
    f2_scorer = make_scorer(fbeta_score, beta=2, zero_division=0)
    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

    y_prob_all = np.zeros(len(y))
    fold_metrics = []
    importances = np.zeros(X.shape[1])

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X.values, y)):
        X_train, y_train = X.values[train_idx], y[train_idx]
        X_test = X.values[test_idx]

        best_inner_score = -1
        best_model = None

        for smote_ratio in smote_ratios:
            sm = SMOTE(sampling_strategy=smote_ratio, random_state=SEED)
            try:
                X_sm, y_sm = sm.fit_resample(X_train, y_train)
            except ValueError:
                continue

            inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                                random_state=SEED, verbosity=0)
            grid = GridSearchCV(xgb, param_grid, scoring=f2_scorer,
                                cv=inner_cv, n_jobs=-1, refit=True)
            grid.fit(X_sm, y_sm)

            if grid.best_score_ > best_inner_score:
                best_inner_score = grid.best_score_
                best_model = grid.best_estimator_

        y_prob = best_model.predict_proba(X_test)[:, 1]
        y_prob_all[test_idx] = y_prob
        y_pred = best_model.predict(X_test)

        fm = _eval_metrics(y[test_idx], y_pred, y_prob)
        fold_metrics.append(fm)
        importances += best_model.feature_importances_

        if (fold + 1) % 5 == 0:
            _print(f"  Fold {fold + 1}/10: AUROC={fm['auroc']:.3f}")

    importances /= 10
    pooled_auroc = roc_auc_score(y, y_prob_all)
    opt_thresh, opt_f2 = _find_f2_threshold(y, y_prob_all)

    _print(f"  Pooled OOF AUROC: {pooled_auroc:.3f}")
    _print(f"  Optimal F2 threshold: {opt_thresh:.2f} (F2={opt_f2:.3f})")

    # Summary
    for k in ["accuracy", "sensitivity", "specificity", "f2", "auroc"]:
        vals = [m[k] for m in fold_metrics]
        _print(f"  {k}: {100 * np.mean(vals):.1f}% (+/- {100 * np.std(vals):.1f}%)")

    return {
        "fold_metrics": fold_metrics,
        "y_prob_all": y_prob_all,
        "pooled_auroc": pooled_auroc,
        "optimal_threshold": opt_thresh,
        "importances": importances,
    }


# ===================================================================
# 4. BASELINE LOGISTIC REGRESSION
# ===================================================================

def baseline_lr(X, y, features):
    """Baseline LR with top 5 features (10-fold CV)."""
    _print("\n=== Baseline Logistic Regression ===")

    from sklearn.feature_selection import SelectKBest, f_classif

    selector = SelectKBest(f_classif, k="all")
    selector.fit(X, y)
    f_scores = pd.Series(selector.scores_, index=features).sort_values(ascending=False)
    top5 = f_scores.head(5).index.tolist()
    _print(f"  Top 5 features: {top5}")

    X_top5 = X[top5].values
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    fold_metrics = []
    y_prob_all = np.zeros(len(y))

    for train_idx, test_idx in cv.split(X_top5, y):
        lr = LogisticRegression(max_iter=1000, random_state=SEED)
        lr.fit(X_top5[train_idx], y[train_idx])
        y_prob = lr.predict_proba(X_top5[test_idx])[:, 1]
        y_pred = lr.predict(X_top5[test_idx])
        y_prob_all[test_idx] = y_prob
        fold_metrics.append(_eval_metrics(y[test_idx], y_pred, y_prob))

    pooled_auroc = roc_auc_score(y, y_prob_all)
    _print(f"  Pooled AUROC: {pooled_auroc:.3f}")

    return {
        "fold_metrics": fold_metrics,
        "y_prob_all": y_prob_all,
        "pooled_auroc": pooled_auroc,
    }


# ===================================================================
# 5. TEMPORAL VALIDATION
# ===================================================================

def temporal_validation(df, features):
    """Train on Year 1, validate on Year 2."""
    _print("\n=== Temporal Validation ===")

    y1 = df[df["year"] == 1]
    y2 = df[df["year"] == 2]

    X1 = y1[features].fillna(0).values
    X2 = y2[features].fillna(0).values
    y1_out = y1[TARGET].values
    y2_out = y2[TARGET].values

    _print(f"  Year 1: {len(y1)} ({y1_out.sum()} CIIP)")
    _print(f"  Year 2: {len(y2)} ({y2_out.sum()} CIIP)")

    # XGBoost with SMOTE + inner CV on Y1
    f2_scorer = make_scorer(fbeta_score, beta=2, zero_division=0)
    sm = SMOTE(sampling_strategy=0.75, random_state=SEED)
    X1_sm, y1_sm = sm.fit_resample(X1, y1_out)

    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
    }
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                        random_state=SEED, verbosity=0)
    grid = GridSearchCV(xgb, param_grid, scoring=f2_scorer,
                        cv=inner_cv, n_jobs=-1, refit=True)
    grid.fit(X1_sm, y1_sm)

    xgb_y1_prob = grid.predict_proba(X1)[:, 1]
    xgb_y2_prob = grid.predict_proba(X2)[:, 1]
    xgb_y2_pred = grid.predict(X2)

    # LR with top 10 predictors
    top10 = [f for f in TOP_PREDICTORS if f in features]
    X1_lr = y1[top10].fillna(0).values
    X2_lr = y2[top10].fillna(0).values
    lr = LogisticRegression(max_iter=1000, random_state=SEED, penalty="l2", C=1.0)
    lr.fit(X1_lr, y1_out)
    lr_y2_prob = lr.predict_proba(X2_lr)[:, 1]
    lr_y2_pred = lr.predict(X2_lr)

    xgb_m = _eval_metrics(y2_out, xgb_y2_pred, xgb_y2_prob)
    lr_m = _eval_metrics(y2_out, lr_y2_pred, lr_y2_prob)

    # Bootstrap CIs
    rng = np.random.RandomState(SEED)
    xgb_aurocs, lr_aurocs = [], []
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(len(y2_out), size=len(y2_out), replace=True)
        if len(np.unique(y2_out[idx])) < 2:
            continue
        try:
            xgb_aurocs.append(roc_auc_score(y2_out[idx], xgb_y2_prob[idx]))
            lr_aurocs.append(roc_auc_score(y2_out[idx], lr_y2_prob[idx]))
        except ValueError:
            continue

    xgb_ci = (np.percentile(xgb_aurocs, 2.5), np.percentile(xgb_aurocs, 97.5))
    lr_ci = (np.percentile(lr_aurocs, 2.5), np.percentile(lr_aurocs, 97.5))

    _print(f"  XGBoost Y2 AUROC: {xgb_m['auroc']:.3f} ({xgb_ci[0]:.3f}-{xgb_ci[1]:.3f})")
    _print(f"  LR Y2 AUROC: {lr_m['auroc']:.3f} ({lr_ci[0]:.3f}-{lr_ci[1]:.3f})")

    # Save
    rows = [
        {"Model": "XGBoost", "Set": "Temporal (Y2)",
         "N": len(y2), "Events": int(y2_out.sum()),
         "AUROC": f"{xgb_m['auroc']:.3f}",
         "AUROC 95% CI": f"({xgb_ci[0]:.3f}-{xgb_ci[1]:.3f})",
         "Sensitivity": f"{100 * xgb_m['sensitivity']:.1f}%",
         "Specificity": f"{100 * xgb_m['specificity']:.1f}%",
         "PPV": f"{100 * xgb_m['ppv']:.1f}%",
         "NPV": f"{100 * xgb_m['npv']:.1f}%"},
        {"Model": "LR (10-predictor)", "Set": "Temporal (Y2)",
         "N": len(y2), "Events": int(y2_out.sum()),
         "AUROC": f"{lr_m['auroc']:.3f}",
         "AUROC 95% CI": f"({lr_ci[0]:.3f}-{lr_ci[1]:.3f})",
         "Sensitivity": f"{100 * lr_m['sensitivity']:.1f}%",
         "Specificity": f"{100 * lr_m['specificity']:.1f}%",
         "PPV": f"{100 * lr_m['ppv']:.1f}%",
         "NPV": f"{100 * lr_m['npv']:.1f}%"},
    ]
    pd.DataFrame(rows).to_csv(TABLES / "table_ciip_ms1_temporal.csv", index=False)
    _print(f"  Saved table_ciip_ms1_temporal.csv")

    return {
        "xgb_y2_prob": xgb_y2_prob, "lr_y2_prob": lr_y2_prob,
        "y2_out": y2_out, "xgb_m": xgb_m, "lr_m": lr_m,
        "xgb_ci": xgb_ci, "lr_ci": lr_ci,
        "xgb_y1_prob": xgb_y1_prob, "y1_out": y1_out,
    }


# ===================================================================
# 6. CALIBRATION
# ===================================================================

def calibration_metrics(y_true, y_prob, n_bins=10):
    """CITL, slope, Brier."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    brier = brier_score_loss(y_true, y_prob)
    prevalence = y_true.mean()
    brier_max = prevalence * (1 - prevalence)
    brier_scaled = 1 - brier / brier_max if brier_max > 0 else 0

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

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_data = []
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        if mask.sum() > 0:
            bin_data.append({
                "mean_predicted": float(y_prob[mask].mean()),
                "mean_observed": float(y_true[mask].mean()),
                "n": int(mask.sum()),
            })

    return {"citl": citl, "slope": slope, "brier": brier,
            "brier_scaled": brier_scaled, "bin_data": bin_data}


def calibration_analysis(xgb_result, temporal, df):
    """Compute calibration for nested CV and temporal validation."""
    _print("\n=== Calibration Analysis ===")

    y = df[TARGET].values
    cal_cv = calibration_metrics(y, xgb_result["y_prob_all"])
    cal_temporal = calibration_metrics(temporal["y2_out"], temporal["xgb_y2_prob"])

    rows = [
        {"Set": "10-fold nested CV (pooled OOF)",
         "CITL": f"{cal_cv['citl']:.3f}", "Slope": f"{cal_cv['slope']:.3f}",
         "Brier": f"{cal_cv['brier']:.4f}", "Brier scaled": f"{cal_cv['brier_scaled']:.4f}"},
        {"Set": "Temporal validation (Y2)",
         "CITL": f"{cal_temporal['citl']:.3f}", "Slope": f"{cal_temporal['slope']:.3f}",
         "Brier": f"{cal_temporal['brier']:.4f}", "Brier scaled": f"{cal_temporal['brier_scaled']:.4f}"},
    ]

    tbl = pd.DataFrame(rows)
    tbl.to_csv(TABLES / "table_ciip_ms1_calibration.csv", index=False)
    _print(f"  Saved table_ciip_ms1_calibration.csv")
    _print(f"  CV — CITL={cal_cv['citl']:.3f}, Slope={cal_cv['slope']:.3f}, Brier={cal_cv['brier']:.4f}")
    _print(f"  Temporal — CITL={cal_temporal['citl']:.3f}, Slope={cal_temporal['slope']:.3f}, Brier={cal_temporal['brier']:.4f}")

    return cal_cv, cal_temporal


# ===================================================================
# 7. DECISION CURVE ANALYSIS
# ===================================================================

def decision_curve(xgb_result, lr_result, df):
    """Net benefit for XGBoost and LR."""
    _print("\n=== Decision Curve Analysis ===")

    y = df[TARGET].values
    n = len(y)
    prevalence = y.mean()
    thresholds = np.arange(0.01, 0.51, 0.01)

    rows = []
    nb_xgb_list, nb_lr_list, nb_all_list = [], [], []

    for t in thresholds:
        # XGBoost
        xgb_pos = xgb_result["y_prob_all"] >= t
        xgb_tp = (xgb_pos & (y == 1)).sum()
        xgb_fp = (xgb_pos & (y == 0)).sum()
        nb_xgb = xgb_tp / n - xgb_fp / n * (t / (1 - t))

        # LR
        lr_pos = lr_result["y_prob_all"] >= t
        lr_tp = (lr_pos & (y == 1)).sum()
        lr_fp = (lr_pos & (y == 0)).sum()
        nb_lr = lr_tp / n - lr_fp / n * (t / (1 - t))

        nb_all = prevalence - (1 - prevalence) * (t / (1 - t))

        nb_xgb_list.append(nb_xgb)
        nb_lr_list.append(nb_lr)
        nb_all_list.append(nb_all)

        rows.append({
            "threshold": t,
            "nb_xgb": nb_xgb, "nb_lr": nb_lr,
            "nb_all": nb_all, "nb_none": 0,
        })

    # Save at key thresholds
    key_t = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
    table_rows = []
    for t in key_t:
        idx = np.argmin(np.abs(thresholds - t))
        r = rows[idx]
        table_rows.append({
            "Threshold (%)": f"{100 * t:.0f}%",
            "XGBoost NB": f"{r['nb_xgb']:.4f}",
            "LR NB": f"{r['nb_lr']:.4f}",
            "Investigate all NB": f"{r['nb_all']:.4f}",
            "Investigate none NB": "0.0000",
        })
    pd.DataFrame(table_rows).to_csv(TABLES / "table_ciip_ms1_dca.csv", index=False)
    _print(f"  Saved table_ciip_ms1_dca.csv")

    return {
        "thresholds": thresholds,
        "nb_xgb": nb_xgb_list, "nb_lr": nb_lr_list,
        "nb_all": nb_all_list, "prevalence": prevalence,
    }


# ===================================================================
# 8. MI vs CIIP COMPARISON TABLE
# ===================================================================

def mi_vs_ciip_comparison(df, xgb_result):
    """Side-by-side comparison of MI and CIIP metrics."""
    _print("\n=== MI vs CIIP Comparison ===")

    y_mi = df["outcome_med_important"].values
    y_ciip = df[TARGET].values
    y_prob = xgb_result["y_prob_all"]

    # Also run XGBoost nested CV on MI outcome for fair comparison
    features_list = get_feature_names()
    features_list = [f for f in features_list if f in df.columns]
    X = df[features_list].fillna(0)

    # Load MI pooled OOF probabilities from original analysis if available
    # Otherwise, report prevalence-based comparison
    rows = [
        {"Metric": "Outcome count",
         "MI": str(int(y_mi.sum())),
         "CIIP": str(int(y_ciip.sum())),
         "Delta": str(int(y_ciip.sum()) - int(y_mi.sum()))},
        {"Metric": "Prevalence",
         "MI": f"{100 * y_mi.mean():.1f}%",
         "CIIP": f"{100 * y_ciip.mean():.1f}%",
         "Delta": f"{100 * (y_ciip.mean() - y_mi.mean()):.1f}pp"},
        {"Metric": "XGBoost pooled OOF AUROC (CIIP model)",
         "MI": "see original scripts",
         "CIIP": f"{xgb_result['pooled_auroc']:.3f}",
         "Delta": ""},
        {"Metric": "XGBoost optimal F2 threshold",
         "MI": "see original scripts",
         "CIIP": f"{xgb_result['optimal_threshold']:.2f}",
         "Delta": ""},
    ]

    # Per-fold mean metrics for CIIP
    for k in ["sensitivity", "specificity", "auroc", "f2"]:
        vals = [m[k] for m in xgb_result["fold_metrics"]]
        rows.append({
            "Metric": f"XGBoost 10-fold mean {k}",
            "MI": "see original scripts",
            "CIIP": f"{100 * np.mean(vals):.1f}% (+/- {100 * np.std(vals):.1f}%)",
            "Delta": "",
        })

    tbl = pd.DataFrame(rows)
    tbl.to_csv(TABLES / "table_ciip_ms1_comparison.csv", index=False)
    _print(f"  Saved table_ciip_ms1_comparison.csv")
    return tbl


# ===================================================================
# 9. PERFORMANCE & FEATURE IMPORTANCE TABLES
# ===================================================================

def save_performance_table(xgb_result, lr_result):
    """Save pooled CV performance summary."""
    rows = []
    for label, result in [("XGBoost (nested CV)", xgb_result),
                          ("LR baseline (10-fold CV)", lr_result)]:
        for k in ["accuracy", "sensitivity", "specificity", "ppv", "npv", "f2", "auroc"]:
            vals = [m[k] for m in result["fold_metrics"]]
            rows.append({
                "Model": label,
                "Metric": k,
                "Mean": f"{np.mean(vals):.3f}",
                "SD": f"{np.std(vals):.3f}",
                "Pooled AUROC": f"{result['pooled_auroc']:.3f}" if k == "auroc" else "",
            })
    tbl = pd.DataFrame(rows)
    tbl.to_csv(TABLES / "table_ciip_ms1_performance.csv", index=False)
    _print(f"  Saved table_ciip_ms1_performance.csv")


def save_feature_importance(features, importances):
    """Save XGBoost feature importance."""
    tbl = pd.DataFrame({
        "Feature": features,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    tbl["Rank"] = range(1, len(tbl) + 1)
    tbl.to_csv(TABLES / "table_ciip_ms1_feature_importance.csv", index=False)
    _print(f"  Saved table_ciip_ms1_feature_importance.csv")
    return tbl


# ===================================================================
# 10. FIGURES
# ===================================================================

def fig_roc(xgb_result, lr_result, temporal, df):
    """ROC curves for CV and temporal validation."""
    y = df[TARGET].values

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # Panel A: Pooled OOF
    ax = axes[0]
    ax.set_title("10-fold Nested CV (pooled OOF)", fontweight="bold")
    for label, y_prob, color, ls in [
        ("XGBoost", xgb_result["y_prob_all"], "#d62728", "-"),
        ("LR baseline", lr_result["y_prob_all"], "#1f77b4", "--"),
    ]:
        fpr, tpr, _ = roc_curve(y, y_prob)
        auc = roc_auc_score(y, y_prob)
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})", color=color, linestyle=ls, linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.legend(loc="lower right", fontsize=9)

    # Panel B: Temporal validation
    ax = axes[1]
    ax.set_title("Temporal Validation (Year 2)", fontweight="bold")
    for label, y_prob, y_true, color, ls in [
        ("XGBoost", temporal["xgb_y2_prob"], temporal["y2_out"], "#d62728", "-"),
        ("LR", temporal["lr_y2_prob"], temporal["y2_out"], "#1f77b4", "--"),
    ]:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})", color=color, linestyle=ls, linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.legend(loc="lower right", fontsize=9)

    fig.suptitle("CIIP Model: ROC Curves", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_ciip_ms1_roc.png", bbox_inches="tight")
    plt.close()
    _print("  Saved fig_ciip_ms1_roc.png")


def fig_calibration_plot(xgb_result, temporal, df):
    """Calibration plots (CV + temporal)."""
    y = df[TARGET].values

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, title, y_true, y_prob, color in [
        (axes[0], "10-fold CV (pooled OOF)", y, xgb_result["y_prob_all"], "#d62728"),
        (axes[1], "Temporal Validation (Y2)", temporal["y2_out"], temporal["xgb_y2_prob"], "#d62728"),
    ]:
        cal = calibration_metrics(y_true, y_prob)
        bins = cal["bin_data"]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Ideal")
        if bins:
            pred = [b["mean_predicted"] for b in bins]
            obs = [b["mean_observed"] for b in bins]
            sizes = [max(20, b["n"] / 5) for b in bins]
            ax.scatter(pred, obs, s=sizes, c=color, alpha=0.8, zorder=3)
            ax.plot(pred, obs, "-", color=color, alpha=0.6)
            ax.set_xlim(-0.02, max(0.3, max(pred) * 1.2))
            ax.set_ylim(-0.02, max(0.3, max(obs) * 1.2))
        txt = (f"CITL={cal['citl']:.3f}\nSlope={cal['slope']:.2f}\nBrier={cal['brier']:.4f}")
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=8,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed proportion")
        ax.set_title(title)

    fig.suptitle("CIIP Model: Calibration", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_ciip_ms1_calibration.png", bbox_inches="tight")
    plt.close()
    _print("  Saved fig_ciip_ms1_calibration.png")


def fig_dca_plot(dca_data):
    """Decision curve figure."""
    t = dca_data["thresholds"] * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, dca_data["nb_all"], "k-", alpha=0.5, linewidth=1.5, label="Investigate all")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1, label="Investigate none")
    ax.plot(t, dca_data["nb_xgb"], "-", color="#d62728", linewidth=2.5, label="XGBoost")
    ax.plot(t, dca_data["nb_lr"], "--", color="#1f77b4", linewidth=2, label="LR baseline")

    ax.set_xlabel("Threshold probability (%)")
    ax.set_ylabel("Net benefit")
    ax.set_title("CIIP Model: Decision Curve Analysis")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, 50)
    ax.grid(alpha=0.2)

    prev = dca_data["prevalence"]
    ax.axvline(x=prev * 100, color="gray", linestyle=":", alpha=0.4)
    ax.text(prev * 100 + 0.8, ax.get_ylim()[0] * 0.3,
            f"CIIP rate\n({100 * prev:.1f}%)", fontsize=8, color="gray")

    fig.tight_layout()
    fig.savefig(FIGURES / "fig_ciip_ms1_dca.png", bbox_inches="tight")
    plt.close()
    _print("  Saved fig_ciip_ms1_dca.png")


def fig_importance_plot(features, importances):
    """Feature importance bar chart."""
    idx = np.argsort(importances)[::-1][:20]

    fig, ax = plt.subplots(figsize=(9, 6))
    y_pos = np.arange(len(idx))
    ax.barh(y_pos, importances[idx], color="#d62728", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([features[i] for i in idx], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Feature Importance (XGBoost)")
    ax.set_title("CIIP Model: Top 20 Feature Importances")
    ax.grid(alpha=0.2, axis="x")

    fig.tight_layout()
    fig.savefig(FIGURES / "fig_ciip_ms1_feature_importance.png", bbox_inches="tight")
    plt.close()
    _print("  Saved fig_ciip_ms1_feature_importance.png")


# ===================================================================
# MAIN
# ===================================================================

def main():
    t_start = time.time()
    _print("=" * 60)
    _print("Brain AttACKs: MS1 Re-analysis with CIIP Outcome")
    _print("=" * 60)

    # 1. Load
    df, X, y, features = load_data()

    # 2. Descriptive stats
    descriptive_stats(df)

    # 3. XGBoost nested CV
    xgb_result = xgboost_nested_cv(X, y, features)

    # 4. Baseline LR
    lr_result = baseline_lr(X, y, features)

    # 5. Temporal validation
    temporal = temporal_validation(df, features)

    # 6. Calibration
    cal_cv, cal_temporal = calibration_analysis(xgb_result, temporal, df)

    # 7. DCA
    dca_data = decision_curve(xgb_result, lr_result, df)

    # 8. MI vs CIIP comparison
    mi_vs_ciip_comparison(df, xgb_result)

    # 9. Tables
    save_performance_table(xgb_result, lr_result)
    save_feature_importance(features, xgb_result["importances"])

    # 10. Figures
    _print("\n=== Generating Figures ===")
    fig_roc(xgb_result, lr_result, temporal, df)
    fig_calibration_plot(xgb_result, temporal, df)
    fig_dca_plot(dca_data)
    fig_importance_plot(features, xgb_result["importances"])

    elapsed = time.time() - t_start
    _print(f"\n{'=' * 60}")
    _print(f"Done in {elapsed:.0f}s.")
    _print(f"  CIIP: {y.sum()} ({100 * y.mean():.1f}%)")
    _print(f"  Pooled OOF AUROC: {xgb_result['pooled_auroc']:.3f}")
    _print(f"  Temporal AUROC: {temporal['xgb_m']['auroc']:.3f}")
    _print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
