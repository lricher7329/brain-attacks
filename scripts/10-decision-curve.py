"""
10-decision-curve.py — Decision Curve Analysis (TRIPOD+AI Item 23d)

Evaluates clinical utility of the prediction models using decision curve
analysis (net benefit vs. threshold probability). Compares model-guided
strategies against default "investigate all" and "investigate none".

Reference:
    Vickers AJ, Van Calster B, Steyerberg EW. Net benefit approaches to
    the evaluation of prediction models, molecular markers, and diagnostic
    tests. BMJ. 2016;352:i6.

Inputs:
    data/processed/brain_attacks_combined.parquet

Outputs:
    outputs/tables/table_net_benefit.csv
    outputs/tables/tableS10_net_benefit.csv  (same, for supplementary ref)
    outputs/figures/fig_decision_curve.png
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
    confusion_matrix, fbeta_score, make_scorer,
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
TARGET = "outcome_med_important"

TOP_PREDICTORS = [
    "sx_ataxia", "exam_altered_loc", "ems", "sx_fever",
    "sx_focal_weakness", "exam_meningismus", "sx_collapse",
    "gender", "sx_seizure", "exam_papilledema",
]


def get_feature_names():
    return (SYMPTOM_FEATURES + OTHER_SX_FEATURES + EXAM_FEATURES +
            HEADACHE_FEATURES + DEMOGRAPHIC_FEATURES + OTHER_FEATURES)


# ===================================================================
# 1. DATA LOADING
# ===================================================================

def load_data():
    df = pd.read_parquet(DATA / "brain_attacks_combined.parquet")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    features = get_feature_names()
    features = [f for f in features if f in df.columns]
    X = df[features].fillna(0).copy()
    y = df[TARGET].values
    _print(f"Loaded: {len(df)} records, {len(features)} features")
    _print(f"  Outcome: {y.sum()} MI ({100*y.mean():.1f}%), {len(y)-y.sum()} benign")
    return df, X, y, features


# ===================================================================
# 2. NET BENEFIT CALCULATION
# ===================================================================

def net_benefit(y_true, y_prob, thresholds):
    """
    Compute net benefit at each threshold probability.

    Net benefit = (TP/N) - (FP/N) * (t / (1-t))

    where t is the threshold probability.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    prevalence = y_true.mean()

    results = []
    for t in thresholds:
        # Model-based strategy
        pred_pos = y_prob >= t
        tp = np.sum((pred_pos == 1) & (y_true == 1))
        fp = np.sum((pred_pos == 1) & (y_true == 0))
        nb = tp / n - fp / n * (t / (1 - t))

        # Treat-all strategy
        nb_all = prevalence - (1 - prevalence) * (t / (1 - t))

        results.append({
            "threshold": t,
            "net_benefit": nb,
            "net_benefit_all": nb_all,
            "tp": tp,
            "fp": fp,
            "n_treated": int(pred_pos.sum()),
        })

    return results


# ===================================================================
# 3. GENERATE OUT-OF-FOLD PREDICTIONS
# ===================================================================

def get_cv_predictions(X, y, features):
    """Get out-of-fold predictions for XGBoost, LR, and clinical score."""
    _print("\n=== Generating Out-of-Fold Predictions ===")

    k_best = len(features)
    smote_ratio = 1.0
    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
    }
    f2_scorer = make_scorer(fbeta_score, beta=2, zero_division=0)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

    xgb_probs = np.zeros(len(y))
    lr_probs = np.zeros(len(y))

    for fold, (train_idx, test_idx) in enumerate(cv.split(X.values, y)):
        X_train, X_test = X.values[train_idx], X.values[test_idx]
        y_train = y[train_idx]

        # XGBoost
        sm = SMOTE(sampling_strategy=smote_ratio, random_state=SEED)
        X_tr_sm, y_tr_sm = sm.fit_resample(X_train, y_train)

        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                            random_state=SEED, verbosity=0)
        grid = GridSearchCV(xgb, param_grid, scoring=f2_scorer,
                            cv=inner_cv, n_jobs=-1, refit=True)
        grid.fit(X_tr_sm, y_tr_sm)
        xgb_probs[test_idx] = grid.predict_proba(X_test)[:, 1]

        # LR (top 5)
        selector = SelectKBest(f_classif, k=5)
        X_train_lr = selector.fit_transform(X_train, y_train)
        X_test_lr = selector.transform(X_test)
        lr = LogisticRegression(max_iter=1000, random_state=SEED)
        lr.fit(X_train_lr, y_train)
        lr_probs[test_idx] = lr.predict_proba(X_test_lr)[:, 1]

        _print(f"  Fold {fold+1}/10 complete")

    # Clinical score
    _print("  Computing clinical score predictions...")
    score_features = [f for f in TOP_PREDICTORS if f in X.columns]
    X_score = X[score_features].fillna(0)
    lr_full = LogisticRegression(max_iter=1000, random_state=SEED, penalty="l2", C=1.0)
    lr_full.fit(X_score.values, y)

    coefs = pd.Series(lr_full.coef_[0], index=score_features)
    abs_coefs = coefs.abs()
    min_coef = abs_coefs[abs_coefs > 0].min()
    raw_weights = (abs_coefs / min_coef).round().astype(int)
    weights = raw_weights * np.sign(coefs).astype(int)
    score = (X_score.values * weights.values).sum(axis=1)

    score_probs = np.zeros(len(y))
    for train_idx, test_idx in cv.split(score.reshape(-1, 1), y):
        m = LogisticRegression(max_iter=1000, random_state=SEED)
        m.fit(score[train_idx].reshape(-1, 1), y[train_idx])
        score_probs[test_idx] = m.predict_proba(score[test_idx].reshape(-1, 1))[:, 1]

    return xgb_probs, lr_probs, score_probs


# ===================================================================
# 4. DECISION CURVE FIGURE
# ===================================================================

def make_figure(y, model_results, thresholds):
    """Plot decision curves for all strategies."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

    _print("\n=== Decision Curve Figure ===")

    fig, ax = plt.subplots(figsize=(9, 6))

    colors = {
        "XGBoost": "#d62728",
        "Logistic regression": "#1f77b4",
        "Clinical score": "#2ca02c",
    }

    # Treat-all line
    prevalence = y.mean()
    nb_all = [prevalence - (1 - prevalence) * (t / (1 - t)) for t in thresholds]
    ax.plot(thresholds * 100, nb_all, "k-", alpha=0.5, linewidth=1.5,
            label="Investigate all")

    # Treat-none line
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1,
               label="Investigate none")

    # Model lines
    for label, results in model_results.items():
        nb_vals = [r["net_benefit"] for r in results]
        ax.plot(thresholds * 100, nb_vals, "-", color=colors.get(label, "#333"),
                linewidth=2, label=label)

    ax.set_xlabel("Threshold probability (%)")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision Curve Analysis: Clinical Utility of Prediction Models")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, 50)

    # Find reasonable y-axis limits
    all_nb = []
    for results in model_results.values():
        all_nb.extend([r["net_benefit"] for r in results])
    all_nb.extend(nb_all)
    ymin = max(-0.05, min(all_nb) - 0.01)
    ymax = min(0.15, max(all_nb) + 0.02)
    ax.set_ylim(ymin, ymax)

    ax.grid(alpha=0.2)

    # Add prevalence annotation
    ax.axvline(x=prevalence * 100, color="gray", linestyle=":", alpha=0.4)
    ax.text(prevalence * 100 + 0.5, ymax * 0.9, f"Prevalence\n({100*prevalence:.1f}%)",
            fontsize=8, color="gray")

    fig.tight_layout()
    fig.savefig(FIGURES / "fig_decision_curve.png")
    plt.close()
    _print("  Saved fig_decision_curve.png")


# ===================================================================
# 5. RESULTS TABLE
# ===================================================================

def save_table(y, model_results, thresholds):
    """Save net benefit at key clinical thresholds."""
    key_thresholds = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    prevalence = y.mean()

    rows = []
    for t in key_thresholds:
        # Find closest threshold
        idx = np.argmin(np.abs(thresholds - t))
        actual_t = thresholds[idx]

        row = {"Threshold (%)": f"{100*actual_t:.0f}%"}

        # Treat-all
        nb_all = prevalence - (1 - prevalence) * (actual_t / (1 - actual_t))
        row["Investigate all"] = f"{nb_all:.4f}"

        for label, results in model_results.items():
            r = results[idx]
            row[f"{label} NB"] = f"{r['net_benefit']:.4f}"
            row[f"{label} N treated"] = r["n_treated"]

        row["Investigate none"] = "0.0000"
        rows.append(row)

    tbl = pd.DataFrame(rows)
    tbl.to_csv(TABLES / "table_net_benefit.csv", index=False)
    tbl.to_csv(TABLES / "tableS10_net_benefit.csv", index=False)
    _print(f"\n  Saved table_net_benefit.csv ({len(tbl)} thresholds)")

    _print("\n  Net Benefit at Key Thresholds:")
    _print(tbl.to_string(index=False))

    return tbl


# ===================================================================
# MAIN
# ===================================================================

def main():
    t_start = time.time()
    _print("=" * 60)
    _print("Brain AttACKs: Decision Curve Analysis (TRIPOD+AI)")
    _print("=" * 60)

    df, X, y, features = load_data()

    # Get predictions
    xgb_probs, lr_probs, score_probs = get_cv_predictions(X, y, features)

    # Threshold range
    thresholds = np.arange(0.01, 0.51, 0.01)

    # Compute net benefit for each model
    _print("\n=== Computing Net Benefit ===")
    model_results = {}
    for label, probs in [
        ("XGBoost", xgb_probs),
        ("Logistic regression", lr_probs),
        ("Clinical score", score_probs),
    ]:
        results = net_benefit(y, probs, thresholds)
        model_results[label] = results
        _print(f"  {label}: computed across {len(thresholds)} thresholds")

    # Identify where models provide net benefit
    _print("\n=== Clinical Utility Summary ===")
    for label, results in model_results.items():
        prevalence = y.mean()
        benefit_range = []
        for r in results:
            t = r["threshold"]
            nb_all = prevalence - (1 - prevalence) * (t / (1 - t))
            if r["net_benefit"] > max(0, nb_all):
                benefit_range.append(t)
        if benefit_range:
            _print(f"  {label}: net benefit over defaults at "
                   f"thresholds {100*min(benefit_range):.0f}%-{100*max(benefit_range):.0f}%")
        else:
            _print(f"  {label}: no net benefit range identified")

    # Save outputs
    save_table(y, model_results, thresholds)
    make_figure(y, model_results, thresholds)

    elapsed = time.time() - t_start
    _print(f"\n{'='*60}")
    _print(f"Done in {elapsed:.0f}s. Outputs in outputs/tables/ and outputs/figures/")
    _print(f"{'='*60}")


if __name__ == "__main__":
    main()
