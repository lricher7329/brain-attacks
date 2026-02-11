"""
03-prediction.py — Predictive Modeling (Objective 1)

XGBoost classification with unified inner CV (SMOTE + hyperparameter search):
  - Baseline logistic regression (top 5 features)
  - XGBoost with all features, SMOTE search, 5-fold inner CV (F2-score)
  - Metrics at default (0.50) and F2-optimized thresholds
  - F-score beta experiment (threshold optimization)
  - False negative analysis
  - Feature importance

Inputs:
    data/processed/brain_attacks_combined.parquet

Outputs:
    outputs/tables/table_model_performance.csv
    outputs/tables/table_feature_importance.csv
    outputs/figures/fig_model_comparison.png
    outputs/figures/fig_feature_importance.png
    outputs/figures/fig_beta_experiment.png
    outputs/figures/fig_false_negative_profiles.png
"""

import sys
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
# 1. DATA LOADING & FEATURE DEFINITION
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
TARGET = "outcome_med_important"


def get_feature_names():
    return (SYMPTOM_FEATURES + OTHER_SX_FEATURES + EXAM_FEATURES +
            HEADACHE_FEATURES + DEMOGRAPHIC_FEATURES + OTHER_FEATURES)


def load_data():
    df = pd.read_parquet(DATA / "brain_attacks_combined.parquet")
    features = get_feature_names()
    features = [f for f in features if f in df.columns]
    X = df[features].fillna(0).copy()
    y = df[TARGET].values
    _print(f"Loaded: {len(df)} records, {len(features)} features")
    _print(f"  Outcome: {y.sum()} MI ({100*y.mean():.1f}%), {len(y)-y.sum()} benign")
    return X, y, features, df


# ===================================================================
# 2. HELPERS
# ===================================================================

def _eval_fold(y_true, y_pred, y_prob):
    """Compute standard metrics for a single fold."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "auroc": roc_auc_score(y_true, y_prob),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def _summarize(fold_metrics, label=""):
    """Print mean +/- SD for fold metrics."""
    for k in ["accuracy", "sensitivity", "specificity", "f2", "auroc"]:
        vals = [m[k] for m in fold_metrics]
        _print(f"    {k}: {100*np.mean(vals):.1f}% (+/- {100*np.std(vals):.1f}%)")


def _find_f2_threshold(y_true, y_prob):
    """Find threshold that maximizes F2-score."""
    best_f2 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.01, 0.99, 0.01):
        y_pred = (y_prob >= thresh).astype(int)
        f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2 = f2
            best_thresh = thresh
    return best_thresh, best_f2


# ===================================================================
# 3. BASELINE LOGISTIC REGRESSION
# ===================================================================

def baseline_logistic(X, y, features):
    _print("\n=== Baseline Logistic Regression ===")

    selector = SelectKBest(f_classif, k="all")
    selector.fit(X, y)
    f_scores = pd.Series(selector.scores_, index=features).sort_values(ascending=False)
    _print("  Top 10 features by F-test:")
    for feat, score in f_scores.head(10).items():
        _print(f"    {feat}: F={score:.1f}")

    top5 = f_scores.head(5).index.tolist()
    X_top5 = X[top5].values

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    fold_metrics = []
    for train_idx, test_idx in cv.split(X_top5, y):
        model = LogisticRegression(max_iter=1000, random_state=SEED)
        model.fit(X_top5[train_idx], y[train_idx])
        y_pred = model.predict(X_top5[test_idx])
        y_prob = model.predict_proba(X_top5[test_idx])[:, 1]
        fold_metrics.append(_eval_fold(y[test_idx], y_pred, y_prob))

    result = {k: (np.mean([m[k] for m in fold_metrics]),
                   np.std([m[k] for m in fold_metrics]))
              for k in ["accuracy", "sensitivity", "specificity", "f2", "auroc"]}

    _print(f"\n  Results (10-fold CV, top 5 features: {top5}):")
    _summarize(fold_metrics)
    return result, top5, f_scores


# ===================================================================
# 4. XGBOOST WITH UNIFIED INNER CV
# ===================================================================

def xgboost_classification(X, y, features):
    """
    All features with SMOTE ratio search in unified 5-fold inner CV.
    Each outer fold independently selects optimal (SMOTE, hyperparameters).
    Reports metrics at both default (0.50) and F2-optimized thresholds.
    """
    _print("\n=== XGBoost Classification ===")
    _print("  Unified inner CV: SMOTE ratios x hyperparameters")
    _print("  10-fold outer x 5-fold inner, all features")

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
    fold_configs = []

    t0 = time.time()

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X.values, y)):
        X_train = X.values[train_idx]
        y_train = y[train_idx]
        X_test = X.values[test_idx]

        # Search SMOTE ratios x hyperparameters in inner CV
        best_inner_score = -1
        best_model = None
        best_smote = None
        best_params = None

        for smote_ratio in smote_ratios:
            sm = SMOTE(sampling_strategy=smote_ratio, random_state=SEED)
            X_sm, y_sm = sm.fit_resample(X_train, y_train)

            inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                                random_state=SEED, verbosity=0)
            grid = GridSearchCV(xgb, param_grid, scoring=f2_scorer,
                                cv=inner_cv, n_jobs=-1, refit=True)
            grid.fit(X_sm, y_sm)

            if grid.best_score_ > best_inner_score:
                best_inner_score = grid.best_score_
                best_model = grid.best_estimator_
                best_smote = smote_ratio
                best_params = grid.best_params_

        # Get test fold predictions
        y_prob = best_model.predict_proba(X_test)[:, 1]
        y_prob_all[test_idx] = y_prob

        # Evaluate at default threshold (0.50)
        y_pred = best_model.predict(X_test)
        fm = _eval_fold(y[test_idx], y_pred, y_prob)
        fm["best_params"] = best_params
        fm["smote_ratio"] = best_smote
        fold_metrics.append(fm)
        fold_configs.append({"smote": best_smote, "params": best_params})

        _print(f"    Fold {fold+1}/10: SMOTE={best_smote}, {best_params}")

    elapsed = time.time() - t0
    _print(f"  Took {elapsed:.0f}s")

    # --- Results at default threshold ---
    _print("\n  Results at default threshold (0.50):")
    _summarize(fold_metrics)

    # --- Pooled OOF AUROC ---
    pooled_auroc = roc_auc_score(y, y_prob_all)
    _print(f"\n  Pooled OOF AUROC: {100*pooled_auroc:.1f}%")

    # --- F2-optimal threshold on pooled OOF predictions ---
    opt_thresh, opt_f2 = _find_f2_threshold(y, y_prob_all)
    y_pred_opt = (y_prob_all >= opt_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred_opt).ravel()

    _print(f"\n  Results at F2-optimal threshold ({opt_thresh:.2f}):")
    _print(f"    accuracy: {100*(tp+tn)/len(y):.1f}%")
    _print(f"    sensitivity: {100*tp/(tp+fn):.1f}%")
    _print(f"    specificity: {100*tn/(tn+fp):.1f}%")
    _print(f"    f2: {100*opt_f2:.1f}%")
    _print(f"    auroc: {100*pooled_auroc:.1f}%")

    # Per-fold metrics at optimal threshold (for SD calculation)
    fold_metrics_opt = []
    for _, (_, test_idx) in enumerate(
            StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED).split(X.values, y)):
        y_prob_fold = y_prob_all[test_idx]
        y_pred_fold = (y_prob_fold >= opt_thresh).astype(int)
        fm_opt = _eval_fold(y[test_idx], y_pred_fold, y_prob_fold)
        fold_metrics_opt.append(fm_opt)

    best_config = {
        "metrics": fold_metrics,
        "metrics_optimized": fold_metrics_opt,
        "configs": fold_configs,
        "mean_f2": np.mean([m["f2"] for m in fold_metrics]),
        "y_prob_all": y_prob_all,
        "pooled_auroc": pooled_auroc,
        "optimal_threshold": opt_thresh,
    }

    # SMOTE selection summary
    smote_counts = Counter(c["smote"] for c in fold_configs)
    _print(f"\n  SMOTE selection: {dict(smote_counts)}")

    return best_config


# ===================================================================
# 5. FEATURE IMPORTANCE
# ===================================================================

def compute_feature_importance(X, y, features, best_config):
    _print("\n=== Feature Importance ===")

    param_strs = [str(c["params"]) for c in best_config["configs"]]
    best_params = eval(Counter(param_strs).most_common(1)[0][0])
    smote_counts = Counter(c["smote"] for c in best_config["configs"])
    best_smote = smote_counts.most_common(1)[0][0]
    _print(f"  Most common: params={best_params}, SMOTE={best_smote}")

    sm = SMOTE(sampling_strategy=best_smote, random_state=SEED)
    X_sm, y_sm = sm.fit_resample(X.values, y)

    model = XGBClassifier(**best_params, use_label_encoder=False,
                          eval_metric="logloss", random_state=SEED, verbosity=0)
    model.fit(X_sm, y_sm)

    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    _print("\n  Top 10 features:")
    for feat, imp in importances.head(10).items():
        _print(f"    {feat}: {imp:.3f}")

    imp_df = importances.reset_index()
    imp_df.columns = ["Feature", "Importance"]
    imp_df.to_csv(TABLES / "table_feature_importance.csv", index=False)
    _print(f"  Saved table_feature_importance.csv")

    return importances, model, features, best_params, best_smote


# ===================================================================
# 6. BETA EXPERIMENT — THRESHOLD OPTIMIZATION
# ===================================================================

def beta_experiment(X, y, features, best_config, n_runs=20):
    """
    For each run, train model and optimize threshold for each beta.
    Much faster than retraining for each beta.
    """
    _print(f"\n=== F-Score Beta Experiment ({n_runs} runs) ===")

    param_strs = [str(c["params"]) for c in best_config["configs"]]
    best_params = eval(Counter(param_strs).most_common(1)[0][0])
    smote_counts = Counter(c["smote"] for c in best_config["configs"])
    best_smote = smote_counts.most_common(1)[0][0]

    betas = [1.0, 1.5, 2.0, 2.5, 3.0]
    results = {b: {"fn": [], "fp": []} for b in betas}

    for run in range(n_runs):
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED + run)
        y_true_all = []
        y_prob_all = []

        for train_idx, test_idx in cv.split(X.values, y):
            sm = SMOTE(sampling_strategy=best_smote, random_state=SEED + run)
            X_sm, y_sm = sm.fit_resample(X.values[train_idx], y[train_idx])

            xgb = XGBClassifier(**best_params, use_label_encoder=False,
                                eval_metric="logloss", random_state=SEED, verbosity=0)
            xgb.fit(X_sm, y_sm)
            y_prob = xgb.predict_proba(X.values[test_idx])[:, 1]
            y_true_all.extend(y[test_idx])
            y_prob_all.extend(y_prob)

        y_true_arr = np.array(y_true_all)
        y_prob_arr = np.array(y_prob_all)

        for beta in betas:
            best_fbeta = -1
            best_thresh = 0.5
            for thresh in np.arange(0.05, 0.95, 0.05):
                y_pred = (y_prob_arr >= thresh).astype(int)
                fb = fbeta_score(y_true_arr, y_pred, beta=beta, zero_division=0)
                if fb > best_fbeta:
                    best_fbeta = fb
                    best_thresh = thresh

            y_pred = (y_prob_arr >= best_thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred).ravel()
            results[beta]["fn"].append(fn)
            results[beta]["fp"].append(fp)

        if (run + 1) % 5 == 0:
            _print(f"  Run {run + 1}/{n_runs}")

    _print("\n  Results:")
    _print(f"  {'Beta':>6} {'FN mean':>10} {'FN sd':>8} {'FP mean':>10} {'FP sd':>8}")
    for beta in betas:
        fn_m, fn_s = np.mean(results[beta]["fn"]), np.std(results[beta]["fn"])
        fp_m, fp_s = np.mean(results[beta]["fp"]), np.std(results[beta]["fp"])
        _print(f"  {beta:>6.1f} {fn_m:>10.1f} {fn_s:>8.1f} {fp_m:>10.1f} {fp_s:>8.1f}")

    return results, betas


# ===================================================================
# 7. FALSE NEGATIVE ANALYSIS
# ===================================================================

def false_negative_analysis(X, y, features, best_config, n_runs=20):
    _print(f"\n=== False Negative Analysis ({n_runs} runs) ===")

    param_strs = [str(c["params"]) for c in best_config["configs"]]
    best_params = eval(Counter(param_strs).most_common(1)[0][0])
    smote_counts = Counter(c["smote"] for c in best_config["configs"])
    best_smote = smote_counts.most_common(1)[0][0]

    mi_indices = np.where(y == 1)[0]
    misclass_counts = np.zeros(len(y), dtype=int)
    tested_counts = np.zeros(len(y), dtype=int)

    for run in range(n_runs):
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED + run)
        for train_idx, test_idx in cv.split(X.values, y):
            sm = SMOTE(sampling_strategy=best_smote, random_state=SEED + run)
            X_sm, y_sm = sm.fit_resample(X.values[train_idx], y[train_idx])

            xgb = XGBClassifier(**best_params, use_label_encoder=False,
                                eval_metric="logloss", random_state=SEED, verbosity=0)
            xgb.fit(X_sm, y_sm)
            y_pred = xgb.predict(X.values[test_idx])
            tested_counts[test_idx] += 1
            misclass_counts[test_idx] += (y_pred != y[test_idx]).astype(int)

        if (run + 1) % 5 == 0:
            _print(f"  Run {run + 1}/{n_runs}")

    mi_misclass_rate = misclass_counts[mi_indices] / tested_counts[mi_indices]
    always_wrong = np.sum(mi_misclass_rate > 0.90)
    always_right = np.sum(mi_misclass_rate < 0.10)
    _print(f"\n  MI cases (n={len(mi_indices)}):")
    _print(f"    Always misclassified (>90%): {always_wrong}")
    _print(f"    Always correct (<10%): {always_right}")
    _print(f"    Variable: {len(mi_indices) - always_wrong - always_right}")

    return mi_misclass_rate, mi_indices


# ===================================================================
# 8. FIGURES
# ===================================================================

def make_figures(lr_result, xgb_config, importances, beta_results, betas,
                 mi_misclass_rate, mi_indices, X, y, features):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

    # --- Figure 1: Model comparison ---
    _print("\n=== Figure 1: Model Comparison ===")
    mlist = ["accuracy", "sensitivity", "specificity", "f2", "auroc"]
    lr_means = [lr_result[m][0] * 100 for m in mlist]
    xgb_default = [np.mean([f[m] for f in xgb_config["metrics"]]) * 100 for m in mlist]
    xgb_opt = [np.mean([f[m] for f in xgb_config["metrics_optimized"]]) * 100 for m in mlist]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(mlist))
    w = 0.25
    ax.bar(x - w, lr_means, w, label="Logistic Regression", color="#1f77b4")
    ax.bar(x, xgb_default, w, label="XGBoost (t=0.50)", color="#d62728")
    opt_t = xgb_config["optimal_threshold"]
    ax.bar(x + w, xgb_opt, w, label=f"XGBoost (t={opt_t:.2f})", color="#ff7f0e")
    ax.set_ylabel("Performance (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in mlist])
    ax.legend()
    ax.set_title("Model Performance Comparison (10-fold CV)")
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_model_comparison.png")
    plt.close()
    _print("  Saved fig_model_comparison.png")

    # --- Figure 2: Feature importance ---
    _print("=== Figure 2: Feature Importance ===")
    top_n = min(15, len(importances))
    top_imp = importances.head(top_n)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(top_n), top_imp.values[::-1], color="#2ca02c")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_imp.index[::-1])
    ax.set_xlabel("Feature Importance")
    ax.set_title("Top Feature Importance Values")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_feature_importance.png")
    plt.close()
    _print("  Saved fig_feature_importance.png")

    # --- Figure 3: Beta experiment ---
    _print("=== Figure 3: Beta Experiment ===")
    fn_means = [np.mean(beta_results[b]["fn"]) for b in betas]
    fp_means = [np.mean(beta_results[b]["fp"]) for b in betas]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(betas, fn_means, "o-", color="#d62728", label="False Negatives", linewidth=2)
    ax.plot(betas, fp_means, "s-", color="#1f77b4", label="False Positives", linewidth=2)
    ax.set_xlabel("Beta value")
    ax.set_ylabel("Count")
    ax.set_title("False Positives vs False Negatives at Varying Beta")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_beta_experiment.png")
    plt.close()
    _print("  Saved fig_beta_experiment.png")

    # --- Figure 4: False negative symptom profiles ---
    _print("=== Figure 4: False Negative Profiles ===")
    always_wrong_idx = mi_indices[mi_misclass_rate > 0.90]
    always_right_idx = mi_indices[mi_misclass_rate < 0.10]

    if len(always_wrong_idx) > 0 and len(always_right_idx) > 0:
        sx_feats = [f for f in SYMPTOM_FEATURES if f in X.columns]
        wrong_pcts = [100 * X.iloc[always_wrong_idx][f].mean() for f in sx_feats]
        right_pcts = [100 * X.iloc[always_right_idx][f].mean() for f in sx_feats]
        labels = [f.replace("sx_", "").replace("_", " ") for f in sx_feats]

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w/2, wrong_pcts, w,
               label=f"Always wrong (n={len(always_wrong_idx)})", color="#d62728")
        ax.bar(x + w/2, right_pcts, w,
               label=f"Always correct (n={len(always_right_idx)})", color="#2ca02c")
        ax.set_ylabel("Prevalence (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend()
        ax.set_title("Symptom Profiles: Always-Misclassified vs Always-Correct MI Cases")
        fig.tight_layout()
        fig.savefig(FIGURES / "fig_false_negative_profiles.png")
        plt.close()
        _print("  Saved fig_false_negative_profiles.png")
    else:
        _print("  Skipped — insufficient always-wrong or always-correct cases")


# ===================================================================
# 9. SUMMARY TABLE
# ===================================================================

def save_performance_table(lr_result, xgb_config):
    mlist = ["accuracy", "sensitivity", "specificity", "f2", "auroc"]
    rows = []
    for m in mlist:
        lr_m, lr_s = lr_result[m]
        xgb_vals = [f[m] for f in xgb_config["metrics"]]
        xgb_opt_vals = [f[m] for f in xgb_config["metrics_optimized"]]
        rows.append({
            "Metric": m.capitalize(),
            "LR mean": f"{100*lr_m:.1f}%",
            "LR SD": f"{100*lr_s:.1f}%",
            "XGBoost mean": f"{100*np.mean(xgb_vals):.1f}%",
            "XGBoost SD": f"{100*np.std(xgb_vals):.1f}%",
            "XGBoost (F2-opt) mean": f"{100*np.mean(xgb_opt_vals):.1f}%",
            "XGBoost (F2-opt) SD": f"{100*np.std(xgb_opt_vals):.1f}%",
        })

    rows.append({
        "Metric": "Threshold",
        "LR mean": "0.50",
        "LR SD": "",
        "XGBoost mean": "0.50",
        "XGBoost SD": "",
        "XGBoost (F2-opt) mean": f"{xgb_config['optimal_threshold']:.2f}",
        "XGBoost (F2-opt) SD": "",
    })

    tbl = pd.DataFrame(rows)
    tbl.to_csv(TABLES / "table_model_performance.csv", index=False)
    _print(f"\n  Saved table_model_performance.csv")


# ===================================================================
# MAIN
# ===================================================================

def main():
    t_start = time.time()
    _print("=" * 60)
    _print("Brain AttACKs: Predictive Modeling (Objective 1)")
    _print("=" * 60)

    X, y, features, df = load_data()

    lr_result, top5, f_scores = baseline_logistic(X, y, features)
    best_config = xgboost_classification(X, y, features)
    importances, model, feat_names, best_params, best_smote = compute_feature_importance(
        X, y, features, best_config)
    save_performance_table(lr_result, best_config)
    beta_results, betas = beta_experiment(X, y, features, best_config, n_runs=20)
    mi_misclass_rate, mi_indices = false_negative_analysis(
        X, y, features, best_config, n_runs=20)
    make_figures(lr_result, best_config, importances, beta_results, betas,
                 mi_misclass_rate, mi_indices, X, y, features)

    elapsed = time.time() - t_start
    _print(f"\n{'='*60}")
    _print(f"Done in {elapsed:.0f}s. Outputs in outputs/tables/ and outputs/figures/")
    _print(f"{'='*60}")


if __name__ == "__main__":
    main()
