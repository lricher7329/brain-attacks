"""
04-validation.py — Temporal Validation (Objective 2)

Year 1 derivation → Year 2 validation using best XGBoost model from 03.

Inputs:
    data/processed/brain_attacks_combined.parquet

Outputs:
    outputs/tables/table3_temporal_validation.csv
    outputs/tables/tableS5_case_mix_y1_vs_y2.csv     (TRIPOD+AI Item 22)
    outputs/figures/fig_roc_temporal.png
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, fbeta_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
np.random.seed(SEED)

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data" / "processed"
TABLES = PROJECT / "outputs" / "tables"
FIGURES = PROJECT / "outputs" / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# Import feature definitions from 03
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

# Feature definitions (replicated for standalone use)
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


# ===================================================================
# 1. DATA LOADING
# ===================================================================

def load_data():
    """Load and split by year."""
    df = pd.read_parquet(DATA / "brain_attacks_combined.parquet")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    features = get_feature_names()
    features = [f for f in features if f in df.columns]

    y1 = df[df["year"] == 1].copy()
    y2 = df[df["year"] == 2].copy()

    print(f"Year 1: {len(y1)} records ({(y1[TARGET]==1).sum()} MI, "
          f"{100*(y1[TARGET]==1).mean():.1f}%)")
    print(f"Year 2: {len(y2)} records ({(y2[TARGET]==1).sum()} MI, "
          f"{100*(y2[TARGET]==1).mean():.1f}%)")

    X1 = y1[features].fillna(0)
    X2 = y2[features].fillna(0)
    y1_out = y1[TARGET].values
    y2_out = y2[TARGET].values

    return X1, y1_out, X2, y2_out, features, df


# ===================================================================
# 2. POOLED CV (REFERENCE)
# ===================================================================

def pooled_cv(X, y, features, param_grid, k_best, smote_ratio):
    """10-fold pooled CV as reference comparison."""
    print("\n=== Pooled Cross-Validation (Reference) ===")

    f2_scorer = lambda y_t, y_p: fbeta_score(y_t, y_p, beta=2, zero_division=0)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    metrics = {"accuracy": [], "sensitivity": [], "specificity": [], "f2": [], "auroc": []}
    y_probs_all = np.zeros(len(y))

    for train_idx, test_idx in cv.split(X.values, y):
        X_train, X_test = X.values[train_idx], X.values[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if k_best < len(features):
            selector = SelectKBest(f_classif, k=k_best)
            X_train = selector.fit_transform(X_train, y_train)
            X_test = selector.transform(X_test)

        if smote_ratio is not None:
            smote = SMOTE(sampling_strategy=smote_ratio, random_state=SEED)
            X_train, y_train = smote.fit_resample(X_train, y_train)

        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        from sklearn.metrics import make_scorer
        xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                            random_state=SEED, verbosity=0)
        grid = GridSearchCV(xgb, param_grid,
                            scoring=make_scorer(fbeta_score, beta=2, zero_division=0),
                            cv=inner_cv, n_jobs=-1, refit=True)
        grid.fit(X_train, y_train)

        y_pred = grid.predict(X_test)
        y_prob = grid.predict_proba(X_test)[:, 1]
        y_probs_all[test_idx] = y_prob

        tn, fp, fn, tp = confusion_matrix(y[test_idx], y_pred).ravel()
        metrics["accuracy"].append(accuracy_score(y[test_idx], y_pred))
        metrics["sensitivity"].append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        metrics["specificity"].append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        metrics["f2"].append(fbeta_score(y[test_idx], y_pred, beta=2, zero_division=0))
        metrics["auroc"].append(roc_auc_score(y[test_idx], y_prob))

    result = {}
    for k, vals in metrics.items():
        result[k] = (np.mean(vals), np.std(vals))
        print(f"  {k}: {100*np.mean(vals):.1f}% (+/- {100*np.std(vals):.1f}%)")

    return result, y_probs_all


# ===================================================================
# 3. TEMPORAL VALIDATION
# ===================================================================

def temporal_validation(X1, y1, X2, y2, features, param_grid, k_best, smote_ratio):
    """Train on Year 1, validate on Year 2."""
    print("\n=== Temporal Validation: Year 1 → Year 2 ===")

    X_train = X1.values.copy()
    X_test = X2.values.copy()
    y_train = y1.copy()
    y_test = y2.copy()

    # Feature selection on training data
    feat_names = list(features)
    selector = None
    if k_best < len(features):
        selector = SelectKBest(f_classif, k=k_best)
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)
        mask = selector.get_support()
        feat_names = [f for f, m in zip(features, mask) if m]
        print(f"  Selected features: {feat_names}")

    # SMOTE on training data
    if smote_ratio is not None:
        smote = SMOTE(sampling_strategy=smote_ratio, random_state=SEED)
        X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    else:
        X_train_sm, y_train_sm = X_train, y_train

    # Hyperparameter search on Year 1 (inner CV)
    from sklearn.metrics import make_scorer
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                        random_state=SEED, verbosity=0)
    grid = GridSearchCV(
        xgb, param_grid,
        scoring=make_scorer(fbeta_score, beta=2, zero_division=0),
        cv=inner_cv, n_jobs=-1, refit=True,
    )
    grid.fit(X_train_sm, y_train_sm)
    print(f"  Best params: {grid.best_params_}")

    # Predict on Year 2
    y_pred = grid.predict(X_test)
    y_prob = grid.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    xgb_result = {
        "accuracy": accuracy_score(y_test, y_pred),
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "f2": fbeta_score(y_test, y_pred, beta=2, zero_division=0),
        "auroc": roc_auc_score(y_test, y_prob),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }

    print(f"\n  XGBoost Temporal Validation:")
    for k in ["accuracy", "sensitivity", "specificity", "f2", "auroc"]:
        print(f"    {k}: {100*xgb_result[k]:.1f}%")
    print(f"    Confusion: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    return xgb_result, y_prob, y_pred, grid


# ===================================================================
# 4. LOGISTIC REGRESSION SENSITIVITY ANALYSIS
# ===================================================================

def lr_temporal_validation(X1, y1, X2, y2, features, k_best):
    """Logistic regression: train Year 1, test Year 2."""
    print("\n=== LR Temporal Validation (Sensitivity Analysis) ===")

    X_train = X1.values.copy()
    X_test = X2.values.copy()

    if k_best < len(features):
        selector = SelectKBest(f_classif, k=k_best)
        X_train = selector.fit_transform(X_train, y1)
        X_test = selector.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=SEED)
    model.fit(X_train, y1)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y2, y_pred).ravel()
    result = {
        "accuracy": accuracy_score(y2, y_pred),
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "f2": fbeta_score(y2, y_pred, beta=2, zero_division=0),
        "auroc": roc_auc_score(y2, y_prob),
    }

    print(f"  Logistic Regression Temporal Validation:")
    for k, v in result.items():
        print(f"    {k}: {100*v:.1f}%")

    return result, y_prob


# ===================================================================
# 5. RESULTS TABLE
# ===================================================================

def save_table3(pooled_result, xgb_temporal, lr_temporal):
    """Save Table 3: Temporal validation comparison."""
    print("\n=== Table 3: Temporal Validation ===")

    metrics_list = ["accuracy", "sensitivity", "specificity", "f2", "auroc"]
    rows = []
    for m in metrics_list:
        rows.append({
            "Metric": m.capitalize(),
            "Pooled CV (XGBoost)": f"{100*pooled_result[m][0]:.1f}% (+/-{100*pooled_result[m][1]:.1f}%)",
            "Temporal (XGBoost)": f"{100*xgb_temporal[m]:.1f}%",
            "Temporal (LR)": f"{100*lr_temporal[m]:.1f}%",
        })

    # Add confusion matrix row
    cm = xgb_temporal["confusion"]
    rows.append({
        "Metric": "Confusion (XGBoost)",
        "Pooled CV (XGBoost)": "—",
        "Temporal (XGBoost)": f"TP={cm['tp']}, FP={cm['fp']}, FN={cm['fn']}, TN={cm['tn']}",
        "Temporal (LR)": "—",
    })

    tbl = pd.DataFrame(rows)
    out = TABLES / "table3_temporal_validation.csv"
    tbl.to_csv(out, index=False)
    print(f"  Saved {out.name}")
    print(tbl.to_string(index=False))


# ===================================================================
# 6. ROC FIGURE
# ===================================================================

def make_roc_figure(y_pooled, y_pooled_prob, y2, y2_xgb_prob, y2_lr_prob):
    """ROC curves: pooled CV vs temporal validation."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  matplotlib not available, skipping figures")
        return

    print("\n=== Figure 5: ROC Curves ===")
    fig, ax = plt.subplots(figsize=(7, 6))

    # Pooled CV
    fpr, tpr, _ = roc_curve(y_pooled, y_pooled_prob)
    auc_pooled = roc_auc_score(y_pooled, y_pooled_prob)
    ax.plot(fpr, tpr, label=f"Pooled CV XGBoost (AUC={auc_pooled:.3f})",
            color="#1f77b4", linewidth=2)

    # Temporal XGBoost
    fpr, tpr, _ = roc_curve(y2, y2_xgb_prob)
    auc_temp = roc_auc_score(y2, y2_xgb_prob)
    ax.plot(fpr, tpr, label=f"Temporal XGBoost (AUC={auc_temp:.3f})",
            color="#d62728", linewidth=2)

    # Temporal LR
    fpr, tpr, _ = roc_curve(y2, y2_lr_prob)
    auc_lr = roc_auc_score(y2, y2_lr_prob)
    ax.plot(fpr, tpr, label=f"Temporal LR (AUC={auc_lr:.3f})",
            color="#2ca02c", linewidth=2, linestyle="--")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("ROC Curves: Pooled CV vs Temporal Validation")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_roc_temporal.png")
    plt.close()
    print("  Saved fig_roc_temporal.png")


# ===================================================================
# 7. CASE-MIX COMPARISON (TRIPOD+AI Item 22)
# ===================================================================

def case_mix_comparison(X1, y1, X2, y2, features, df):
    """
    Compare Year 1 vs Year 2 baseline characteristics with SMDs.
    TRIPOD+AI Item 22: Comparison of development and validation datasets.
    """
    print("\n=== TableS5: Case-Mix Comparison (Y1 vs Y2) ===")

    y1_df = df[df["year"] == 1]
    y2_df = df[df["year"] == 2]

    rows = []

    # Age (continuous)
    for var in ["age", "temp", "hr", "ctas"]:
        if var not in df.columns:
            continue
        v1 = y1_df[var].dropna()
        v2 = y2_df[var].dropna()
        m1, s1 = v1.mean(), v1.std()
        m2, s2 = v2.mean(), v2.std()
        # SMD = (mean1 - mean2) / pooled SD
        pooled_sd = np.sqrt((s1**2 + s2**2) / 2) if (s1 > 0 or s2 > 0) else 1
        smd = (m1 - m2) / pooled_sd if pooled_sd > 0 else 0
        rows.append({
            "Variable": var,
            "Type": "continuous",
            "Year 1 (n={})".format(len(y1_df)): f"{m1:.1f} ({s1:.1f})",
            "Year 2 (n={})".format(len(y2_df)): f"{m2:.1f} ({s2:.1f})",
            "SMD": f"{abs(smd):.3f}",
            "SMD_raw": abs(smd),
        })

    # Binary variables
    binary_vars = [f for f in features if f not in ["age", "temp", "hr", "ctas", "gender"]]
    # Add gender separately
    binary_vars = ["gender"] + binary_vars

    for var in binary_vars:
        if var not in df.columns:
            continue
        if var == "gender":
            p1 = (y1_df[var] == 2).mean()  # female
            p2 = (y2_df[var] == 2).mean()
            label = "Female gender"
        else:
            p1 = (y1_df[var] == 1).mean()
            p2 = (y2_df[var] == 1).mean()
            label = var

        # SMD for binary: (p1 - p2) / sqrt((p1*(1-p1) + p2*(1-p2)) / 2)
        denom = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
        smd = (p1 - p2) / denom if denom > 0 else 0

        n1 = int(((y1_df[var] == 2) if var == "gender" else (y1_df[var] == 1)).sum())
        n2 = int(((y2_df[var] == 2) if var == "gender" else (y2_df[var] == 1)).sum())

        rows.append({
            "Variable": label,
            "Type": "binary",
            "Year 1 (n={})".format(len(y1_df)): f"{n1} ({100*p1:.1f}%)",
            "Year 2 (n={})".format(len(y2_df)): f"{n2} ({100*p2:.1f}%)",
            "SMD": f"{abs(smd):.3f}",
            "SMD_raw": abs(smd),
        })

    # Outcome prevalence
    p1_mi = y1.mean()
    p2_mi = y2.mean()
    denom = np.sqrt((p1_mi * (1 - p1_mi) + p2_mi * (1 - p2_mi)) / 2)
    smd_mi = (p1_mi - p2_mi) / denom if denom > 0 else 0
    rows.append({
        "Variable": "MI prevalence",
        "Type": "binary",
        "Year 1 (n={})".format(len(y1_df)): f"{int(y1.sum())} ({100*p1_mi:.1f}%)",
        "Year 2 (n={})".format(len(y2_df)): f"{int(y2.sum())} ({100*p2_mi:.1f}%)",
        "SMD": f"{abs(smd_mi):.3f}",
        "SMD_raw": abs(smd_mi),
    })

    tbl = pd.DataFrame(rows)

    # Flag imbalanced variables (SMD > 0.1)
    imbalanced = tbl[tbl["SMD_raw"] > 0.1]
    if len(imbalanced) > 0:
        print("  Variables with SMD > 0.1 (notable imbalance):")
        for _, r in imbalanced.iterrows():
            print(f"    {r['Variable']}: SMD = {r['SMD']}")
    else:
        print("  All variables have SMD <= 0.1 (well-balanced)")

    # Save (drop raw SMD column used for filtering)
    tbl_out = tbl.drop(columns=["SMD_raw"])
    out = TABLES / "tableS5_case_mix_y1_vs_y2.csv"
    tbl_out.to_csv(out, index=False)
    print(f"  Saved {out.name} ({len(tbl_out)} variables)")

    return tbl_out


# ===================================================================
# MAIN
# ===================================================================

def main():
    print("=" * 60)
    print("Brain AttACKs: Temporal Validation (Objective 2)")
    print("=" * 60)

    X1, y1, X2, y2, features, df = load_data()

    # Combine for pooled analysis
    X_all = pd.concat([X1, X2], ignore_index=True)
    y_all = np.concatenate([y1, y2])

    # Config from manuscript (will be refined if 03 is run first)
    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
    }
    k_best = len(features)  # Use all features (matches 03 best config)
    smote_ratio = 1.0       # Matches 03 unified inner CV selection

    # 1. Pooled CV reference
    pooled_result, y_pooled_prob = pooled_cv(X_all, y_all, features,
                                              param_grid, k_best, smote_ratio)

    # 2. Temporal validation: XGBoost
    xgb_result, y2_xgb_prob, y2_pred, grid = temporal_validation(
        X1, y1, X2, y2, features, param_grid, k_best, smote_ratio
    )

    # 3. Sensitivity analysis: Logistic regression
    lr_result, y2_lr_prob = lr_temporal_validation(X1, y1, X2, y2, features, k_best=5)

    # 4. Results table
    save_table3(pooled_result, xgb_result, lr_result)

    # 5. ROC figure
    make_roc_figure(y_all, y_pooled_prob, y2, y2_xgb_prob, y2_lr_prob)

    # 6. Case-mix comparison (TRIPOD+AI Item 22)
    case_mix_comparison(X1, y1, X2, y2, features, df)

    print("\n" + "=" * 60)
    print("Done. Outputs in outputs/tables/ and outputs/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
