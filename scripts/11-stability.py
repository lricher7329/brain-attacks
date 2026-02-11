"""
11-stability.py — Model Stability, Collinearity, Optimism Correction

Addresses PROBAST Analysis domain signaling questions on model complexity
and instability.

Analyses:
  1. Predictor collinearity (correlation matrix, VIF)
  2. Bootstrap model stability (feature selection frequency)
  3. Optimism-corrected AUROC (bootstrap)

Inputs:
    data/processed/brain_attacks_combined.parquet

Outputs:
    outputs/tables/tableS7_vif.csv
    outputs/tables/tableS8_bootstrap_stability.csv
    outputs/figures/fig_predictor_correlation.png
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
    fbeta_score, make_scorer, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

SEED = 42
N_BOOTSTRAP = 200
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
# 2. PREDICTOR COLLINEARITY
# ===================================================================

def compute_collinearity(X, features):
    """Correlation matrix and VIF for all predictors."""
    _print("\n=== Predictor Collinearity ===")

    # --- Correlation matrix ---
    corr = X.corr()

    # Flag highly correlated pairs
    high_corr = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            r = corr.iloc[i, j]
            if abs(r) > 0.5:
                high_corr.append((features[i], features[j], r))

    if high_corr:
        _print(f"\n  Predictor pairs with |r| > 0.5:")
        for f1, f2, r in sorted(high_corr, key=lambda x: -abs(x[2])):
            flag = " ***" if abs(r) > 0.7 else ""
            _print(f"    {f1} <-> {f2}: r={r:.3f}{flag}")
    else:
        _print("  No predictor pairs with |r| > 0.5")

    # --- VIF for clinical prediction rule predictors ---
    _print("\n  VIF (clinical prediction rule predictors):")
    score_features = [f for f in TOP_PREDICTORS if f in X.columns]
    X_score = X[score_features].copy()

    vif_rows = []
    for i, feat in enumerate(score_features):
        # VIF = 1 / (1 - R^2) where R^2 is from regressing feat on all others
        others = [f for f in score_features if f != feat]
        if len(others) == 0:
            continue
        X_others = X_score[others].values
        y_feat = X_score[feat].values

        # Handle zero-variance features
        if np.std(y_feat) == 0:
            vif_rows.append({"Predictor": feat, "VIF": 1.0, "Flag": ""})
            continue

        try:
            lr = LogisticRegression(max_iter=1000, random_state=SEED) \
                if len(np.unique(y_feat)) == 2 \
                else None

            # Use linear regression for VIF
            from numpy.linalg import lstsq
            X_aug = np.column_stack([np.ones(len(X_others)), X_others])
            beta, _, _, _ = lstsq(X_aug, y_feat, rcond=None)
            y_hat = X_aug @ beta
            ss_res = np.sum((y_feat - y_hat) ** 2)
            ss_tot = np.sum((y_feat - y_feat.mean()) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            vif = 1 / (1 - r_squared) if r_squared < 1 else np.inf
        except Exception:
            vif = np.nan

        flag = "HIGH" if vif > 5 else ("moderate" if vif > 2.5 else "")
        vif_rows.append({"Predictor": feat, "VIF": round(vif, 2), "Flag": flag})
        _print(f"    {feat}: VIF={vif:.2f} {flag}")

    vif_df = pd.DataFrame(vif_rows)
    vif_df.to_csv(TABLES / "tableS7_vif.csv", index=False)
    _print(f"\n  Saved tableS7_vif.csv")

    return corr, vif_df


# ===================================================================
# 3. BOOTSTRAP MODEL STABILITY
# ===================================================================

def bootstrap_stability(X, y, features, n_boot=N_BOOTSTRAP):
    """
    Refit XGBoost on bootstrap samples and track:
    - Feature importance rankings
    - AUROC variability
    - Top-10 selection frequency
    """
    _print(f"\n=== Bootstrap Model Stability ({n_boot} iterations) ===")

    smote_ratio = 1.0
    # Use fixed good params for efficiency (no inner grid search per bootstrap)
    fixed_params = {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1}

    rng = np.random.RandomState(SEED)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    importance_matrix = np.zeros((n_boot, len(features)))
    train_aurocs = []
    test_aurocs = []

    for b in range(n_boot):
        # Stratified bootstrap sample
        pos_sample = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        neg_sample = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        boot_idx = np.concatenate([pos_sample, neg_sample])

        # Out-of-bag indices
        oob_idx = np.setdiff1d(np.arange(len(y)), boot_idx)
        if len(oob_idx) < 10 or len(np.unique(y[oob_idx])) < 2:
            continue

        X_boot = X.values[boot_idx]
        y_boot = y[boot_idx]
        X_oob = X.values[oob_idx]
        y_oob = y[oob_idx]

        # SMOTE
        try:
            sm = SMOTE(sampling_strategy=smote_ratio, random_state=SEED)
            X_sm, y_sm = sm.fit_resample(X_boot, y_boot)
        except ValueError:
            X_sm, y_sm = X_boot, y_boot

        # Fit XGBoost
        xgb = XGBClassifier(**fixed_params, use_label_encoder=False,
                            eval_metric="logloss", random_state=SEED, verbosity=0)
        xgb.fit(X_sm, y_sm)

        # Feature importance
        importance_matrix[b] = xgb.feature_importances_

        # AUROC on bootstrap sample (apparent) and OOB (test)
        try:
            train_prob = xgb.predict_proba(X_boot)[:, 1]
            train_aurocs.append(roc_auc_score(y_boot, train_prob))
        except ValueError:
            pass

        try:
            oob_prob = xgb.predict_proba(X_oob)[:, 1]
            test_aurocs.append(roc_auc_score(y_oob, oob_prob))
        except ValueError:
            pass

        if (b + 1) % 50 == 0:
            _print(f"  Bootstrap {b+1}/{n_boot}")

    # --- Feature selection frequency ---
    # Top 10 in each bootstrap by importance
    top10_counts = np.zeros(len(features))
    for b in range(n_boot):
        if importance_matrix[b].sum() == 0:
            continue
        top10_idx = np.argsort(importance_matrix[b])[-10:]
        top10_counts[top10_idx] += 1

    valid_boots = np.sum(importance_matrix.sum(axis=1) > 0)
    selection_freq = top10_counts / valid_boots if valid_boots > 0 else top10_counts

    # Results table
    stability_rows = []
    for i, feat in enumerate(features):
        imps = importance_matrix[:, i]
        imps = imps[imps > 0] if imps.sum() > 0 else imps
        stability_rows.append({
            "Predictor": feat,
            "Mean importance": round(float(np.mean(importance_matrix[:, i])), 4),
            "SD importance": round(float(np.std(importance_matrix[:, i])), 4),
            "Top-10 frequency (%)": round(100 * selection_freq[i], 1),
            "Median rank": int(np.median([
                len(features) - np.where(np.argsort(importance_matrix[b]) == i)[0][0]
                for b in range(min(valid_boots, n_boot))
                if importance_matrix[b].sum() > 0
            ])) if valid_boots > 0 else 0,
        })

    stability_df = pd.DataFrame(stability_rows).sort_values(
        "Top-10 frequency (%)", ascending=False)
    stability_df.to_csv(TABLES / "tableS8_bootstrap_stability.csv", index=False)
    _print(f"\n  Saved tableS8_bootstrap_stability.csv")

    _print("\n  Top 10 most stable predictors:")
    for _, row in stability_df.head(10).iterrows():
        _print(f"    {row['Predictor']}: selected in {row['Top-10 frequency (%)']:.0f}% "
               f"of bootstraps, mean imp={row['Mean importance']:.4f}")

    return stability_df, train_aurocs, test_aurocs, importance_matrix


# ===================================================================
# 4. OPTIMISM CORRECTION
# ===================================================================

def optimism_correction(train_aurocs, test_aurocs):
    """
    Compute bootstrap optimism-corrected AUROC.

    Optimism = mean(apparent AUROC - OOB AUROC)
    Corrected AUROC = apparent AUROC on full data - optimism
    """
    _print("\n=== Optimism Correction ===")

    if not train_aurocs or not test_aurocs:
        _print("  Insufficient data for optimism correction")
        return None

    n = min(len(train_aurocs), len(test_aurocs))
    optimism_vals = [train_aurocs[i] - test_aurocs[i] for i in range(n)]
    mean_optimism = np.mean(optimism_vals)
    mean_apparent = np.mean(train_aurocs[:n])
    mean_test = np.mean(test_aurocs[:n])

    _print(f"  Mean apparent AUROC (bootstrap training): {mean_apparent:.4f}")
    _print(f"  Mean OOB AUROC (bootstrap test): {mean_test:.4f}")
    _print(f"  Mean optimism: {mean_optimism:.4f}")
    _print(f"  Optimism-corrected estimate: {mean_apparent - mean_optimism:.4f}")
    _print(f"  95% CI of optimism: ({np.percentile(optimism_vals, 2.5):.4f}, "
           f"{np.percentile(optimism_vals, 97.5):.4f})")

    return {
        "apparent_auroc": mean_apparent,
        "oob_auroc": mean_test,
        "optimism": mean_optimism,
        "corrected_auroc": mean_apparent - mean_optimism,
        "optimism_ci_lower": np.percentile(optimism_vals, 2.5),
        "optimism_ci_upper": np.percentile(optimism_vals, 97.5),
    }


# ===================================================================
# 5. LR OPTIMISM CORRECTION
# ===================================================================

def lr_optimism_correction(X, y, features, n_boot=N_BOOTSTRAP):
    """Bootstrap optimism correction for logistic regression (top 5 features)."""
    _print(f"\n=== LR Optimism Correction ({n_boot} iterations) ===")

    rng = np.random.RandomState(SEED + 1)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    train_aurocs = []
    test_aurocs = []

    for b in range(n_boot):
        pos_sample = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        neg_sample = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        boot_idx = np.concatenate([pos_sample, neg_sample])
        oob_idx = np.setdiff1d(np.arange(len(y)), boot_idx)

        if len(oob_idx) < 10 or len(np.unique(y[oob_idx])) < 2:
            continue

        X_boot = X.values[boot_idx]
        y_boot = y[boot_idx]
        X_oob = X.values[oob_idx]
        y_oob = y[oob_idx]

        selector = SelectKBest(f_classif, k=5)
        X_boot_sel = selector.fit_transform(X_boot, y_boot)
        X_oob_sel = selector.transform(X_oob)

        lr = LogisticRegression(max_iter=1000, random_state=SEED)
        lr.fit(X_boot_sel, y_boot)

        try:
            train_prob = lr.predict_proba(X_boot_sel)[:, 1]
            train_aurocs.append(roc_auc_score(y_boot, train_prob))
            oob_prob = lr.predict_proba(X_oob_sel)[:, 1]
            test_aurocs.append(roc_auc_score(y_oob, oob_prob))
        except ValueError:
            pass

        if (b + 1) % 50 == 0:
            _print(f"  Bootstrap {b+1}/{n_boot}")

    result = optimism_correction(train_aurocs, test_aurocs)
    if result:
        _print(f"  LR optimism-corrected AUROC: {result['corrected_auroc']:.4f}")
    return result


# ===================================================================
# 6. CORRELATION HEATMAP
# ===================================================================

def make_correlation_figure(corr, features):
    """Correlation heatmap for all predictors."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "figure.dpi": 150})

    _print("\n=== Correlation Heatmap ===")

    # Use shorter labels
    short_labels = [f.replace("sx_", "").replace("exam_", "ex:").replace("ha_", "ha:")
                    .replace("_", " ") for f in features]

    n = len(features)
    fig, ax = plt.subplots(figsize=(max(12, n * 0.4), max(10, n * 0.35)))

    # Plot heatmap
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_labels, rotation=90, ha="center", fontsize=7)
    ax.set_yticklabels(short_labels, fontsize=7)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Correlation coefficient", fontsize=9)

    # Highlight high correlations with text
    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            if abs(val) > 0.3 and i != j:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=5, color="white" if abs(val) > 0.5 else "black")

    ax.set_title("Predictor Correlation Matrix", fontsize=11, pad=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_predictor_correlation.png", bbox_inches="tight")
    plt.close()
    _print("  Saved fig_predictor_correlation.png")


# ===================================================================
# MAIN
# ===================================================================

def main():
    t_start = time.time()
    _print("=" * 60)
    _print("Brain AttACKs: Model Stability & Collinearity (PROBAST)")
    _print("=" * 60)

    df, X, y, features = load_data()

    # 1. Collinearity
    corr, vif_df = compute_collinearity(X, features)

    # 2. Bootstrap stability (XGBoost)
    stability_df, train_aurocs, test_aurocs, imp_matrix = \
        bootstrap_stability(X, y, features, n_boot=N_BOOTSTRAP)

    # 3. Optimism correction (XGBoost)
    xgb_optimism = optimism_correction(train_aurocs, test_aurocs)

    # 4. Optimism correction (LR)
    lr_optimism = lr_optimism_correction(X, y, features, n_boot=N_BOOTSTRAP)

    # 5. Summary
    _print("\n=== Optimism Summary ===")
    if xgb_optimism:
        _print(f"  XGBoost: apparent={xgb_optimism['apparent_auroc']:.4f}, "
               f"corrected={xgb_optimism['corrected_auroc']:.4f}, "
               f"optimism={xgb_optimism['optimism']:.4f}")
    if lr_optimism:
        _print(f"  LR:      apparent={lr_optimism['apparent_auroc']:.4f}, "
               f"corrected={lr_optimism['corrected_auroc']:.4f}, "
               f"optimism={lr_optimism['optimism']:.4f}")

    # 6. Figures
    make_correlation_figure(corr, features)

    elapsed = time.time() - t_start
    _print(f"\n{'='*60}")
    _print(f"Done in {elapsed:.0f}s. Outputs in outputs/tables/ and outputs/figures/")
    _print(f"{'='*60}")


if __name__ == "__main__":
    main()
