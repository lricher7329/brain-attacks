"""
08-sensitivity.py — Sensitivity Analysis: MI Outcome Definition

Compares primary MI definition (all 147 MI, including IIH + cerebellar ataxia)
against a restrictive definition (excluding IIH + cerebellar ataxia, 113 MI)
across all key prediction analyses.

Clinical rationale:
  - IIH: requires LP for diagnosis, can damage vision — distinct from other MI
  - Cerebellar ataxia: requires imaging to rule out posterior fossa tumour/ADEM

There is clinical debate about whether these belong as MI. This script tests
the impact of excluding them on all key prediction results.

Inputs:
    data/processed/brain_attacks_combined.parquet

Outputs:
    outputs/tables/table_sensitivity_analysis.csv
    outputs/tables/table_imputation_sensitivity.csv
    outputs/figures/fig_sensitivity_comparison.png
"""

import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, fbeta_score,
    make_scorer, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data" / "processed"
TABLES = PROJECT / "outputs" / "tables"
FIGURES = PROJECT / "outputs" / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


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

PARAM_GRID = {
    "n_estimators": [50, 100, 150],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
}


def get_feature_names():
    return (SYMPTOM_FEATURES + OTHER_SX_FEATURES + EXAM_FEATURES +
            HEADACHE_FEATURES + DEMOGRAPHIC_FEATURES + OTHER_FEATURES)


# ===================================================================
# DATA LOADING
# ===================================================================

def load_data():
    """Load data and create both outcome definitions."""
    df = pd.read_parquet(DATA / "brain_attacks_combined.parquet")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    features = get_feature_names()
    features = [f for f in features if f in df.columns]

    # Primary outcome: all MI (codes 1-13)
    # Already defined as outcome_med_important

    # Restrictive outcome: exclude IIH (code 12) and cerebellar ataxia (code 13)
    iih_col = "med_imp_dx___12" if "med_imp_dx___12" in df.columns else None
    ca_col = "med_imp_dx___13" if "med_imp_dx___13" in df.columns else None

    df["outcome_mi_restrictive"] = df["outcome_med_important"].copy()
    if iih_col:
        df.loc[df[iih_col] == 1, "outcome_mi_restrictive"] = 0
    if ca_col:
        df.loc[df[ca_col] == 1, "outcome_mi_restrictive"] = 0

    # But if a record has BOTH a core MI code and IIH/CA, keep it MI
    core_mi_cols = [f"med_imp_dx___{i}" for i in range(1, 12)
                    if f"med_imp_dx___{i}" in df.columns]
    has_core = df[core_mi_cols].max(axis=1) == 1
    df.loc[has_core, "outcome_mi_restrictive"] = 1

    n_primary = int(df["outcome_med_important"].sum())
    n_restrictive = int(df["outcome_mi_restrictive"].sum())
    _print(f"Loaded: {len(df)} records, {len(features)} features")
    _print(f"  Primary MI (all):        {n_primary} ({100*n_primary/len(df):.1f}%)")
    _print(f"  Restrictive MI (no IIH/CA): {n_restrictive} ({100*n_restrictive/len(df):.1f}%)")
    _print(f"  Excluded: {n_primary - n_restrictive} (IIH + cerebellar ataxia only)")

    return df, features


# ===================================================================
# HELPERS
# ===================================================================

def _eval(y_true, y_pred, y_prob):
    """Compute standard metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "auroc": roc_auc_score(y_true, y_prob),
    }


def _prepare_fold(X_train, y_train, X_test, features, k, smote_ratio):
    """Apply feature selection + SMOTE to a training fold."""
    selector = None
    if k < len(features):
        selector = SelectKBest(f_classif, k=k)
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)
    if smote_ratio is not None:
        sm = SMOTE(sampling_strategy=smote_ratio, random_state=SEED)
        X_train, y_train = sm.fit_resample(X_train, y_train)
    return X_train, y_train, X_test, selector


# ===================================================================
# ANALYSIS 1: XGBoost 10-fold Nested CV
# ===================================================================

def xgboost_nested_cv(X, y, features, label=""):
    """XGBoost with all features, unified SMOTE search in 5-fold inner CV."""
    _print(f"\n--- XGBoost Nested CV [{label}] ---")
    _print(f"  N={len(y)}, MI={y.sum()} ({100*y.mean():.1f}%)")

    smote_ratios = [0.5, 0.75, 1.0]
    f2_scorer = make_scorer(fbeta_score, beta=2, zero_division=0)
    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        best_inner_score = -1
        best_model = None

        for smote_ratio in smote_ratios:
            sm = SMOTE(sampling_strategy=smote_ratio, random_state=SEED)
            X_sm, y_sm = sm.fit_resample(X_train, y_train)

            inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                                random_state=SEED, verbosity=0)
            grid = GridSearchCV(xgb, PARAM_GRID, scoring=f2_scorer,
                                cv=inner_cv, n_jobs=-1, refit=True)
            grid.fit(X_sm, y_sm)

            if grid.best_score_ > best_inner_score:
                best_inner_score = grid.best_score_
                best_model = grid.best_estimator_

        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        fold_metrics.append(_eval(y[test_idx], y_pred, y_prob))

    result = {}
    for k in ["accuracy", "sensitivity", "specificity", "f2", "auroc"]:
        vals = [m[k] for m in fold_metrics]
        result[k] = np.mean(vals)
        result[f"{k}_sd"] = np.std(vals)
        _print(f"    {k}: {100*np.mean(vals):.1f}% (+/- {100*np.std(vals):.1f}%)")

    return result


# ===================================================================
# ANALYSIS 2: Temporal Validation (Y1 → Y2)
# ===================================================================

def temporal_validation(X1, y1, X2, y2, features, label=""):
    """Train Year 1, validate Year 2."""
    _print(f"\n--- Temporal Validation [{label}] ---")
    _print(f"  Train: N={len(y1)}, MI={y1.sum()}")
    _print(f"  Test:  N={len(y2)}, MI={y2.sum()}")

    X_train = X1.copy()
    X_test = X2.copy()

    # Use all features, SMOTE 0.5 (matching 04-validation.py defaults)
    k_best = len(features)
    smote_ratio = 1.0

    Xt, yt, Xv, _ = _prepare_fold(X_train, y1, X_test, features, k_best, smote_ratio)

    f2_scorer = make_scorer(fbeta_score, beta=2, zero_division=0)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                        random_state=SEED, verbosity=0)
    grid = GridSearchCV(xgb, PARAM_GRID, scoring=f2_scorer,
                        cv=inner_cv, n_jobs=-1, refit=True)
    grid.fit(Xt, yt)
    _print(f"  Best params: {grid.best_params_}")

    y_pred = grid.predict(Xv)
    y_prob = grid.predict_proba(Xv)[:, 1]
    result = _eval(y2, y_pred, y_prob)

    for k in ["accuracy", "sensitivity", "specificity", "f2", "auroc"]:
        _print(f"    {k}: {100*result[k]:.1f}%")

    return result


# ===================================================================
# ANALYSIS 3: Clinical Prediction Rule
# ===================================================================

def clinical_rule(X, y, features, label=""):
    """Derive integer-weighted score, report CV AUROC."""
    _print(f"\n--- Clinical Prediction Rule [{label}] ---")
    _print(f"  N={len(y)}, MI={y.sum()}")

    pred_features = [f for f in TOP_PREDICTORS if f in features]
    X_sub = X[:, [features.index(f) for f in pred_features]]

    lr = LogisticRegression(max_iter=1000, random_state=SEED, penalty="l2", C=1.0)
    lr.fit(X_sub, y)

    coefs = pd.Series(lr.coef_[0], index=pred_features)
    abs_coefs = coefs.abs()
    min_coef = abs_coefs[abs_coefs > 0].min()
    weights = ((abs_coefs / min_coef).round() * np.sign(coefs)).astype(int)

    score = (X_sub * weights.values).sum(axis=1)

    # 10-fold CV AUROC of the score
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    aurocs = []
    for train_idx, test_idx in cv.split(score.reshape(-1, 1), y):
        m = LogisticRegression(max_iter=1000, random_state=SEED)
        m.fit(score[train_idx].reshape(-1, 1), y[train_idx])
        prob = m.predict_proba(score[test_idx].reshape(-1, 1))[:, 1]
        aurocs.append(roc_auc_score(y[test_idx], prob))

    auroc_mean = np.mean(aurocs)
    auroc_sd = np.std(aurocs)
    _print(f"  Score AUROC (10-fold CV): {auroc_mean:.3f} (+/- {auroc_sd:.3f})")

    # Performance at a key threshold (find threshold with sensitivity >= 90%)
    thresholds = sorted(set(score))
    best_thresh, best_sens, best_spec = None, 0, 0
    for thresh in thresholds:
        pred = (score >= thresh).astype(int)
        if pred.sum() == 0 or pred.sum() == len(pred):
            continue
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        if sens >= 0.90 and (best_thresh is None or spec > best_spec):
            best_thresh, best_sens, best_spec = thresh, sens, spec

    if best_thresh is not None:
        _print(f"  At threshold >= {best_thresh}: sens={100*best_sens:.1f}%, spec={100*best_spec:.1f}%")

    return {
        "auroc": auroc_mean,
        "auroc_sd": auroc_sd,
        "threshold_90sens": best_thresh,
        "sensitivity_at_thresh": best_sens,
        "specificity_at_thresh": best_spec,
    }


# ===================================================================
# COMPARISON TABLE
# ===================================================================

def build_comparison_table(primary, restrictive):
    """Build side-by-side comparison table."""
    rows = []

    # XGBoost nested CV
    for metric in ["accuracy", "sensitivity", "specificity", "f2", "auroc"]:
        rows.append({
            "Analysis": "XGBoost 10-fold nested CV",
            "Metric": metric.capitalize(),
            "Primary (N=147 MI)": f"{100*primary['xgb_cv'][metric]:.1f}% (+/-{100*primary['xgb_cv'][f'{metric}_sd']:.1f}%)",
            "Restrictive (N=113 MI)": f"{100*restrictive['xgb_cv'][metric]:.1f}% (+/-{100*restrictive['xgb_cv'][f'{metric}_sd']:.1f}%)",
            "Delta": f"{100*(restrictive['xgb_cv'][metric] - primary['xgb_cv'][metric]):+.1f}pp",
        })

    # Temporal validation
    for metric in ["accuracy", "sensitivity", "specificity", "f2", "auroc"]:
        rows.append({
            "Analysis": "Temporal validation (Y1->Y2)",
            "Metric": metric.capitalize(),
            "Primary (N=147 MI)": f"{100*primary['temporal'][metric]:.1f}%",
            "Restrictive (N=113 MI)": f"{100*restrictive['temporal'][metric]:.1f}%",
            "Delta": f"{100*(restrictive['temporal'][metric] - primary['temporal'][metric]):+.1f}pp",
        })

    # Clinical prediction rule
    rows.append({
        "Analysis": "Clinical prediction rule",
        "Metric": "AUROC (10-fold CV)",
        "Primary (N=147 MI)": f"{primary['rule']['auroc']:.3f} (+/-{primary['rule']['auroc_sd']:.3f})",
        "Restrictive (N=113 MI)": f"{restrictive['rule']['auroc']:.3f} (+/-{restrictive['rule']['auroc_sd']:.3f})",
        "Delta": f"{restrictive['rule']['auroc'] - primary['rule']['auroc']:+.3f}",
    })
    if primary['rule']['threshold_90sens'] is not None and restrictive['rule']['threshold_90sens'] is not None:
        rows.append({
            "Analysis": "Clinical prediction rule",
            "Metric": "Specificity at 90% sensitivity",
            "Primary (N=147 MI)": f"{100*primary['rule']['specificity_at_thresh']:.1f}%",
            "Restrictive (N=113 MI)": f"{100*restrictive['rule']['specificity_at_thresh']:.1f}%",
            "Delta": f"{100*(restrictive['rule']['specificity_at_thresh'] - primary['rule']['specificity_at_thresh']):+.1f}pp",
        })

    tbl = pd.DataFrame(rows)
    out = TABLES / "table_sensitivity_analysis.csv"
    tbl.to_csv(out, index=False)
    _print(f"\n  Saved {out.name}")
    return tbl


# ===================================================================
# IMPUTATION SENSITIVITY ANALYSIS (I4)
# ===================================================================

def imputation_sensitivity(df, features):
    """
    Compare XGBoost AUROC across three imputation strategies for
    temp, hr, and ctas (the three variables with meaningful missingness).

    Strategies:
      1. Complete-case analysis (drop records with any missing in temp/hr/ctas)
      2. Median imputation (current approach in 01-data-prep.py)
      3. MICE multiple imputation (iterative imputation)
    """
    _print("\n" + "=" * 60)
    _print("IMPUTATION SENSITIVITY ANALYSIS")
    _print("=" * 60)

    # Reload raw data to get pre-imputation missingness
    raw_df = pd.read_parquet(DATA / "brain_attacks_combined.parquet")
    raw_df["year"] = pd.to_numeric(raw_df["year"], errors="coerce")

    impute_vars = ["temp", "hr", "ctas"]
    present_vars = [v for v in impute_vars if v in raw_df.columns]

    # Report missingness
    for v in present_vars:
        n_miss = raw_df[v].isna().sum()
        _print(f"  {v}: {n_miss} missing ({100*n_miss/len(raw_df):.1f}%)")

    y = raw_df["outcome_med_important"].values
    feat_list = [f for f in features if f in raw_df.columns]

    param_grid = {
        "n_estimators": [100],
        "max_depth": [5],
        "learning_rate": [0.1],
    }

    results = {}

    # Strategy 1: Complete-case
    _print("\n--- Strategy 1: Complete-case analysis ---")
    complete_mask = raw_df[present_vars].notna().all(axis=1)
    n_complete = complete_mask.sum()
    _print(f"  N complete: {n_complete} ({100*n_complete/len(raw_df):.1f}%)")

    if n_complete > 100:
        X_cc = raw_df.loc[complete_mask, feat_list].fillna(0).values
        y_cc = y[complete_mask]
        mi_cc = y_cc.sum()
        _print(f"  MI in complete cases: {mi_cc}")

        if mi_cc >= 10:
            from sklearn.metrics import make_scorer
            f2_scorer = make_scorer(fbeta_score, beta=2, zero_division=0)
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
            aurocs = []
            for train_idx, test_idx in cv.split(X_cc, y_cc):
                from imblearn.over_sampling import SMOTE
                sm = SMOTE(sampling_strategy=1.0, random_state=SEED)
                X_tr, y_tr = sm.fit_resample(X_cc[train_idx], y_cc[train_idx])
                from xgboost import XGBClassifier
                xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                                    random_state=SEED, verbosity=0, **{k: v[0] for k, v in param_grid.items()})
                xgb.fit(X_tr, y_tr)
                prob = xgb.predict_proba(X_cc[test_idx])[:, 1]
                aurocs.append(roc_auc_score(y_cc[test_idx], prob))
            results["Complete-case"] = {"auroc": np.mean(aurocs), "sd": np.std(aurocs), "n": n_complete}
            _print(f"  AUROC: {np.mean(aurocs):.3f} (+/- {np.std(aurocs):.3f})")

    # Strategy 2: Median imputation (current approach)
    _print("\n--- Strategy 2: Median imputation (current) ---")
    X_med = raw_df[feat_list].copy()
    for v in present_vars:
        if v in X_med.columns:
            X_med[v] = X_med[v].fillna(X_med[v].median())
    X_med = X_med.fillna(0).values

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    aurocs = []
    for train_idx, test_idx in cv.split(X_med, y):
        sm = SMOTE(sampling_strategy=1.0, random_state=SEED)
        X_tr, y_tr = sm.fit_resample(X_med[train_idx], y[train_idx])
        xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                            random_state=SEED, verbosity=0, **{k: v[0] for k, v in param_grid.items()})
        xgb.fit(X_tr, y_tr)
        prob = xgb.predict_proba(X_med[test_idx])[:, 1]
        aurocs.append(roc_auc_score(y[test_idx], prob))
    results["Median imputation"] = {"auroc": np.mean(aurocs), "sd": np.std(aurocs), "n": len(y)}
    _print(f"  AUROC: {np.mean(aurocs):.3f} (+/- {np.std(aurocs):.3f})")

    # Strategy 3: MICE (iterative imputer)
    _print("\n--- Strategy 3: MICE multiple imputation ---")
    try:
        from sklearn.experimental import enable_iterative_imputer  # noqa
        from sklearn.impute import IterativeImputer

        X_mice_df = raw_df[feat_list].copy()
        # Only impute continuous vars; binary vars should be 0-filled
        binary_cols = [f for f in feat_list if f not in ["age", "temp", "hr", "ctas", "gender"]]
        for bc in binary_cols:
            if bc in X_mice_df.columns:
                X_mice_df[bc] = X_mice_df[bc].fillna(0)

        imputer = IterativeImputer(max_iter=10, random_state=SEED, sample_posterior=False)
        X_mice = imputer.fit_transform(X_mice_df.values)

        aurocs = []
        for train_idx, test_idx in cv.split(X_mice, y):
            sm = SMOTE(sampling_strategy=1.0, random_state=SEED)
            X_tr, y_tr = sm.fit_resample(X_mice[train_idx], y[train_idx])
            xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                                random_state=SEED, verbosity=0, **{k: v[0] for k, v in param_grid.items()})
            xgb.fit(X_tr, y_tr)
            prob = xgb.predict_proba(X_mice[test_idx])[:, 1]
            aurocs.append(roc_auc_score(y[test_idx], prob))
        results["MICE"] = {"auroc": np.mean(aurocs), "sd": np.std(aurocs), "n": len(y)}
        _print(f"  AUROC: {np.mean(aurocs):.3f} (+/- {np.std(aurocs):.3f})")
    except ImportError:
        _print("  WARNING: IterativeImputer not available, skipping MICE")

    # Save comparison table
    rows = []
    for strategy, res in results.items():
        rows.append({
            "Imputation strategy": strategy,
            "N": res["n"],
            "AUROC (mean)": f"{res['auroc']:.3f}",
            "AUROC (SD)": f"{res['sd']:.3f}",
        })
    tbl = pd.DataFrame(rows)
    out = TABLES / "table_imputation_sensitivity.csv"
    tbl.to_csv(out, index=False)
    _print(f"\n  Saved {out.name}")

    # Interpretation
    if len(results) >= 2:
        auroc_vals = [r["auroc"] for r in results.values()]
        delta = max(auroc_vals) - min(auroc_vals)
        _print(f"\n  Max AUROC difference across strategies: {delta:.3f}")
        if delta < 0.02:
            _print("  -> Minimal impact: imputation method does not meaningfully affect discrimination")
        else:
            _print("  -> Notable difference: imputation method affects model performance")

    return results


# ===================================================================
# FIGURE
# ===================================================================

def make_figure(primary, restrictive):
    """Grouped bar chart comparing key metrics across definitions."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        _print("  matplotlib not available, skipping figure")
        return

    _print("\n=== Figure: Sensitivity Comparison ===")

    metrics = ["accuracy", "sensitivity", "specificity", "f2", "auroc"]
    labels = [m.capitalize() for m in metrics]

    # XGBoost nested CV comparison
    primary_vals = [100 * primary['xgb_cv'][m] for m in metrics]
    restrict_vals = [100 * restrictive['xgb_cv'][m] for m in metrics]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: XGBoost nested CV
    x = np.arange(len(labels))
    w = 0.35
    axes[0].bar(x - w/2, primary_vals, w, label="Primary (147 MI)", color="#1f77b4")
    axes[0].bar(x + w/2, restrict_vals, w, label="Restrictive (113 MI)", color="#d62728", alpha=0.8)
    axes[0].set_ylabel("Performance (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha="right")
    axes[0].legend(fontsize=9)
    axes[0].set_title("A. XGBoost 10-fold Nested CV")
    axes[0].set_ylim(0, 105)

    # Panel B: Temporal validation
    primary_tv = [100 * primary['temporal'][m] for m in metrics]
    restrict_tv = [100 * restrictive['temporal'][m] for m in metrics]

    axes[1].bar(x - w/2, primary_tv, w, label="Primary (147 MI)", color="#1f77b4")
    axes[1].bar(x + w/2, restrict_tv, w, label="Restrictive (113 MI)", color="#d62728", alpha=0.8)
    axes[1].set_ylabel("Performance (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[1].legend(fontsize=9)
    axes[1].set_title("B. Temporal Validation (Y1 → Y2)")
    axes[1].set_ylim(0, 105)

    fig.suptitle("Sensitivity Analysis: Primary vs Restrictive MI Definition", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_sensitivity_comparison.png")
    plt.close()
    _print("  Saved fig_sensitivity_comparison.png")


# ===================================================================
# MAIN
# ===================================================================

def main():
    t_start = time.time()
    _print("=" * 60)
    _print("Brain AttACKs: Sensitivity Analysis — MI Outcome Definition")
    _print("=" * 60)

    df, features = load_data()

    X_all = df[features].fillna(0).values
    y_primary = df["outcome_med_important"].values
    y_restrictive = df["outcome_mi_restrictive"].values

    # Split by year for temporal validation
    y1_mask = df["year"] == 1
    y2_mask = df["year"] == 2
    X1, X2 = X_all[y1_mask], X_all[y2_mask]
    y1_primary, y2_primary = y_primary[y1_mask], y_primary[y2_mask]
    y1_restrictive, y2_restrictive = y_restrictive[y1_mask], y_restrictive[y2_mask]

    _print(f"\n  Year 1: primary MI={y1_primary.sum()}, restrictive MI={y1_restrictive.sum()}")
    _print(f"  Year 2: primary MI={y2_primary.sum()}, restrictive MI={y2_restrictive.sum()}")

    # ---- Run all analyses for both definitions ----
    primary = {}
    restrictive = {}

    _print("\n" + "=" * 60)
    _print("PRIMARY DEFINITION (all MI, N=147)")
    _print("=" * 60)

    primary["xgb_cv"] = xgboost_nested_cv(X_all, y_primary, features, "Primary")
    primary["temporal"] = temporal_validation(X1, y1_primary, X2, y2_primary, features, "Primary")
    primary["rule"] = clinical_rule(X_all, y_primary, features, "Primary")

    _print("\n" + "=" * 60)
    _print("RESTRICTIVE DEFINITION (excl. IIH + cerebellar ataxia, N=113)")
    _print("=" * 60)

    restrictive["xgb_cv"] = xgboost_nested_cv(X_all, y_restrictive, features, "Restrictive")
    restrictive["temporal"] = temporal_validation(X1, y1_restrictive, X2, y2_restrictive, features, "Restrictive")
    restrictive["rule"] = clinical_rule(X_all, y_restrictive, features, "Restrictive")

    # ---- Comparison ----
    _print("\n" + "=" * 60)
    _print("COMPARISON")
    _print("=" * 60)

    tbl = build_comparison_table(primary, restrictive)
    _print("\n" + tbl.to_string(index=False))

    make_figure(primary, restrictive)

    # Imputation sensitivity analysis (I4)
    imputation_sensitivity(df, features)

    # Interpretation
    auroc_delta = restrictive["xgb_cv"]["auroc"] - primary["xgb_cv"]["auroc"]
    _print(f"\n=== Interpretation ===")
    _print(f"  AUROC delta (XGBoost CV): {auroc_delta:+.3f}")
    if abs(auroc_delta) < 0.03:
        _print("  -> Minimal impact: IIH and cerebellar ataxia inclusion does not")
        _print("     substantially affect model discrimination.")
    elif auroc_delta > 0:
        _print("  -> Restrictive definition yields higher discrimination,")
        _print("     suggesting IIH/CA cases are harder to predict.")
    else:
        _print("  -> Primary definition yields higher discrimination,")
        _print("     suggesting IIH/CA cases help model calibration.")

    elapsed = time.time() - t_start
    _print(f"\n{'='*60}")
    _print(f"Done in {elapsed:.0f}s. Outputs in outputs/tables/ and outputs/figures/")
    _print(f"{'='*60}")


if __name__ == "__main__":
    main()
