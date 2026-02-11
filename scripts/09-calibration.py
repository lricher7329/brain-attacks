"""
09-calibration.py — Calibration Assessment + Bootstrap CIs (TRIPOD+AI)

Generates calibration metrics and plots for all prediction models,
and computes bootstrap confidence intervals for performance metrics.

TRIPOD+AI compliance:
  - Item 23a: Discrimination with CIs (AUROC, sensitivity, specificity)
  - Item 23b: Calibration (calibration plot, CITL, calibration slope, Brier)

Models assessed (MS1):
  1. XGBoost (10-fold CV, pooled out-of-fold predictions)
  2. Logistic regression (10-fold CV, pooled out-of-fold predictions)
  3. XGBoost temporal validation (Year 1 → Year 2)
  4. LR temporal validation (Year 1 → Year 2)
  5. Clinical prediction score (06-prediction-rule.py)

Note: Headache model calibration (formerly model 6) has moved to
13-headache-deep.py (Manuscript 2) to avoid self-plagiarism.

Inputs:
    data/processed/brain_attacks_combined.parquet

Outputs:
    outputs/tables/table_calibration_metrics.csv
    outputs/tables/tableS9_calibration_metrics.csv  (same, for supplementary ref)
    outputs/figures/fig_calibration_combined.png
    outputs/figures/fig_calibration_temporal.png
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
    accuracy_score, brier_score_loss, confusion_matrix, fbeta_score,
    make_scorer, roc_auc_score,
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

# IHC_PREDICTORS removed — headache calibration moved to 13-headache-deep.py (MS2)


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
# 2. CALIBRATION METRICS
# ===================================================================

def calibration_metrics(y_true, y_prob, n_bins=10):
    """
    Compute calibration metrics for predicted probabilities.

    Returns dict with:
      - citl: calibration-in-the-large (log(O/E), 0 = ideal)
      - slope: calibration slope (logistic regression of outcome on logit(p))
      - brier: Brier score
      - brier_scaled: scaled Brier score (1 - Brier/Brier_max)
      - eo_ratio: expected-to-observed ratio
      - bin_data: per-bin predicted vs observed for plotting
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    # Brier score
    brier = brier_score_loss(y_true, y_prob)
    prevalence = y_true.mean()
    brier_max = prevalence * (1 - prevalence)
    brier_scaled = 1 - brier / brier_max if brier_max > 0 else 0

    # E:O ratio
    expected = y_prob.sum()
    observed = y_true.sum()
    eo_ratio = expected / observed if observed > 0 else np.inf

    # CITL
    citl = np.log(observed / expected) if expected > 0 else np.inf

    # Calibration slope
    # Logistic regression of outcome on logit(predicted probability)
    y_prob_clipped = np.clip(y_prob, 1e-6, 1 - 1e-6)
    lp = np.log(y_prob_clipped / (1 - y_prob_clipped))  # linear predictor (logit)
    try:
        from sklearn.linear_model import LogisticRegression as LR
        lr = LR(max_iter=1000, penalty=None, solver="lbfgs")
        lr.fit(lp.reshape(-1, 1), y_true)
        slope = lr.coef_[0][0]
    except Exception:
        slope = np.nan

    # Binned calibration data for plotting
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_data = []
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:  # include upper edge in last bin
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        if mask.sum() > 0:
            bin_data.append({
                "bin": i + 1,
                "n": int(mask.sum()),
                "mean_predicted": float(y_prob[mask].mean()),
                "mean_observed": float(y_true[mask].mean()),
            })

    return {
        "citl": citl,
        "slope": slope,
        "brier": brier,
        "brier_scaled": brier_scaled,
        "eo_ratio": eo_ratio,
        "bin_data": bin_data,
    }


# ===================================================================
# 3. BOOTSTRAP CONFIDENCE INTERVALS
# ===================================================================

def bootstrap_ci(y_true, y_prob, y_pred=None, n_boot=N_BOOTSTRAP, seed=SEED):
    """
    Bootstrap 95% CIs for discrimination and calibration metrics.
    Uses stratified resampling to maintain outcome prevalence.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_pred is None:
        y_pred = (y_prob >= 0.5).astype(int)
    else:
        y_pred = np.asarray(y_pred, dtype=int)

    rng = np.random.RandomState(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    boot_metrics = {
        "auroc": [], "sensitivity": [], "specificity": [],
        "ppv": [], "npv": [], "f2": [],
        "brier": [], "slope": [],
    }

    for _ in range(n_boot):
        # Stratified bootstrap
        pos_sample = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        neg_sample = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([pos_sample, neg_sample])

        yt = y_true[idx]
        yp = y_prob[idx]
        yd = y_pred[idx]

        # Skip degenerate samples
        if len(np.unique(yt)) < 2:
            continue

        try:
            boot_metrics["auroc"].append(roc_auc_score(yt, yp))
        except ValueError:
            continue

        tn, fp, fn, tp = confusion_matrix(yt, yd, labels=[0, 1]).ravel()
        boot_metrics["sensitivity"].append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        boot_metrics["specificity"].append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        boot_metrics["ppv"].append(tp / (tp + fp) if (tp + fp) > 0 else 0)
        boot_metrics["npv"].append(tn / (tn + fn) if (tn + fn) > 0 else 0)
        boot_metrics["f2"].append(fbeta_score(yt, yd, beta=2, zero_division=0))
        boot_metrics["brier"].append(brier_score_loss(yt, yp))

        # Calibration slope
        yp_clipped = np.clip(yp, 1e-6, 1 - 1e-6)
        lp = np.log(yp_clipped / (1 - yp_clipped))
        try:
            lr = LogisticRegression(max_iter=1000, penalty=None, solver="lbfgs")
            lr.fit(lp.reshape(-1, 1), yt)
            boot_metrics["slope"].append(lr.coef_[0][0])
        except Exception:
            pass

    # Compute percentile CIs
    results = {}
    for metric, vals in boot_metrics.items():
        if len(vals) > 0:
            arr = np.array(vals)
            results[metric] = {
                "mean": float(np.mean(arr)),
                "ci_lower": float(np.percentile(arr, 2.5)),
                "ci_upper": float(np.percentile(arr, 97.5)),
            }

    return results


# ===================================================================
# 4. MODEL REPRODUCTION — POOLED CV PREDICTIONS
# ===================================================================

def get_cv_predictions(X, y, features):
    """
    Reproduce XGBoost and LR models from 03-prediction.py,
    collecting out-of-fold predicted probabilities.
    """
    _print("\n=== Generating CV Predictions ===")

    k_best = len(features)  # all features (matches 03 best config)
    smote_ratio = 1.0
    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
    }
    f2_scorer = make_scorer(fbeta_score, beta=2, zero_division=0)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

    xgb_probs = np.zeros(len(y))
    xgb_preds = np.zeros(len(y), dtype=int)
    lr_probs = np.zeros(len(y))
    lr_preds = np.zeros(len(y), dtype=int)

    for fold, (train_idx, test_idx) in enumerate(cv.split(X.values, y)):
        X_train, X_test = X.values[train_idx], X.values[test_idx]
        y_train = y[train_idx]

        # --- XGBoost ---
        if smote_ratio is not None:
            sm = SMOTE(sampling_strategy=smote_ratio, random_state=SEED)
            X_tr_sm, y_tr_sm = sm.fit_resample(X_train, y_train)
        else:
            X_tr_sm, y_tr_sm = X_train, y_train

        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                            random_state=SEED, verbosity=0)
        grid = GridSearchCV(xgb, param_grid, scoring=f2_scorer,
                            cv=inner_cv, n_jobs=-1, refit=True)
        grid.fit(X_tr_sm, y_tr_sm)
        xgb_probs[test_idx] = grid.predict_proba(X_test)[:, 1]
        xgb_preds[test_idx] = grid.predict(X_test)

        # --- LR (top 5 features) ---
        selector = SelectKBest(f_classif, k=5)
        X_train_lr = selector.fit_transform(X_train, y_train)
        X_test_lr = selector.transform(X_test)
        lr = LogisticRegression(max_iter=1000, random_state=SEED)
        lr.fit(X_train_lr, y_train)
        lr_probs[test_idx] = lr.predict_proba(X_test_lr)[:, 1]
        lr_preds[test_idx] = lr.predict(X_test_lr)

        _print(f"  Fold {fold+1}/10 complete")

    return xgb_probs, xgb_preds, lr_probs, lr_preds


# ===================================================================
# 5. MODEL REPRODUCTION — TEMPORAL VALIDATION
# ===================================================================

def get_temporal_predictions(df, features):
    """Train on Year 1, predict Year 2 — XGBoost and LR."""
    _print("\n=== Generating Temporal Validation Predictions ===")

    y1_df = df[df["year"] == 1]
    y2_df = df[df["year"] == 2]
    X1 = y1_df[features].fillna(0).values
    X2 = y2_df[features].fillna(0).values
    y1 = y1_df[TARGET].values
    y2 = y2_df[TARGET].values

    _print(f"  Year 1: {len(y1)} ({y1.sum()} MI), Year 2: {len(y2)} ({y2.sum()} MI)")

    smote_ratio = 0.5
    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
    }

    # XGBoost
    sm = SMOTE(sampling_strategy=smote_ratio, random_state=SEED)
    X1_sm, y1_sm = sm.fit_resample(X1, y1)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                        random_state=SEED, verbosity=0)
    f2_scorer = make_scorer(fbeta_score, beta=2, zero_division=0)
    grid = GridSearchCV(xgb, param_grid, scoring=f2_scorer,
                        cv=inner_cv, n_jobs=-1, refit=True)
    grid.fit(X1_sm, y1_sm)
    xgb_probs = grid.predict_proba(X2)[:, 1]
    xgb_preds = grid.predict(X2)
    _print(f"  XGBoost best params: {grid.best_params_}")

    # LR (top 5 features)
    selector = SelectKBest(f_classif, k=5)
    X1_lr = selector.fit_transform(X1, y1)
    X2_lr = selector.transform(X2)
    scaler = StandardScaler()
    X1_lr = scaler.fit_transform(X1_lr)
    X2_lr = scaler.transform(X2_lr)
    lr = LogisticRegression(max_iter=1000, random_state=SEED)
    lr.fit(X1_lr, y1)
    lr_probs = lr.predict_proba(X2_lr)[:, 1]
    lr_preds = lr.predict(X2_lr)

    return y2, xgb_probs, xgb_preds, lr_probs, lr_preds


# ===================================================================
# 6. MODEL REPRODUCTION — CLINICAL SCORE
# ===================================================================

def get_score_predictions(df):
    """Reproduce clinical prediction score from 06-prediction-rule.py."""
    _print("\n=== Generating Clinical Score Predictions ===")

    score_features = [f for f in TOP_PREDICTORS if f in df.columns]
    X = df[score_features].fillna(0).copy()
    y = df[TARGET].values

    lr = LogisticRegression(max_iter=1000, random_state=SEED, penalty="l2", C=1.0)
    lr.fit(X.values, y)

    # Integer weights
    coefs = pd.Series(lr.coef_[0], index=score_features)
    abs_coefs = coefs.abs()
    min_coef = abs_coefs[abs_coefs > 0].min()
    raw_weights = (abs_coefs / min_coef).round().astype(int)
    weights = raw_weights * np.sign(coefs).astype(int)

    score = (X.values * weights.values).sum(axis=1)

    # Convert score to probability via CV logistic regression on score
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    probs = np.zeros(len(y))
    preds = np.zeros(len(y), dtype=int)
    for train_idx, test_idx in cv.split(score.reshape(-1, 1), y):
        m = LogisticRegression(max_iter=1000, random_state=SEED)
        m.fit(score[train_idx].reshape(-1, 1), y[train_idx])
        probs[test_idx] = m.predict_proba(score[test_idx].reshape(-1, 1))[:, 1]
        preds[test_idx] = m.predict(score[test_idx].reshape(-1, 1))

    _print(f"  Score range: {score.min()} to {score.max()}")
    return y, probs, preds


# ===================================================================
# 7. CALIBRATION PLOTS
# ===================================================================

def plot_calibration(models_data, filename, title, nrows=2, ncols=2):
    """
    Generate calibration plots for multiple models.
    models_data: list of (label, bin_data) tuples
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

    n_panels = len(models_data)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    if n_panels == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]

    for i, (label, cal) in enumerate(models_data):
        if i >= len(axes):
            break
        ax = axes[i]
        bins = cal["bin_data"]
        if not bins:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(label)
            continue

        pred = [b["mean_predicted"] for b in bins]
        obs = [b["mean_observed"] for b in bins]
        sizes = [b["n"] for b in bins]

        # Ideal line
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Ideal")
        # Calibration points
        ax.scatter(pred, obs, s=[max(20, s / 5) for s in sizes],
                   c=colors[i % len(colors)], alpha=0.8, zorder=3)
        ax.plot(pred, obs, "-", color=colors[i % len(colors)], alpha=0.6)

        # LOESS smoother (simple moving average approximation)
        if len(pred) >= 4:
            from numpy.polynomial.polynomial import polyfit, polyval
            try:
                coeffs = polyfit(pred, obs, deg=min(3, len(pred) - 1))
                x_smooth = np.linspace(min(pred), max(pred), 50)
                y_smooth = polyval(x_smooth, coeffs)
                ax.plot(x_smooth, y_smooth, "-", color=colors[i % len(colors)],
                        alpha=0.3, linewidth=2)
            except Exception:
                pass

        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed proportion")
        ax.set_title(label)
        ax.set_xlim(-0.02, max(0.3, max(pred) * 1.2))
        ax.set_ylim(-0.02, max(0.3, max(obs) * 1.2))

        # Annotate with metrics
        txt = (f"CITL={cal['citl']:.3f}\n"
               f"Slope={cal['slope']:.2f}\n"
               f"Brier={cal['brier']:.4f}")
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=8,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, bbox_inches="tight")
    plt.close()
    _print(f"  Saved {filename}")


# ===================================================================
# 9. RESULTS TABLE
# ===================================================================

def save_results(all_results):
    """Save combined calibration metrics table."""
    rows = []
    for label, data in all_results.items():
        cal = data["calibration"]
        boot = data.get("bootstrap", {})
        row = {
            "Model": label,
            "N": data.get("n", ""),
            "Events": data.get("events", ""),
            "AUROC": "",
            "AUROC 95% CI": "",
            "Sensitivity": "",
            "Sensitivity 95% CI": "",
            "Specificity": "",
            "Specificity 95% CI": "",
            "Brier": f"{cal['brier']:.4f}",
            "Brier scaled": f"{cal['brier_scaled']:.3f}",
            "CITL": f"{cal['citl']:.3f}",
            "Calibration slope": f"{cal['slope']:.3f}" if not np.isnan(cal['slope']) else "—",
            "E:O ratio": f"{cal['eo_ratio']:.3f}",
        }

        if "auroc" in boot:
            b = boot["auroc"]
            row["AUROC"] = f"{b['mean']:.3f}"
            row["AUROC 95% CI"] = f"({b['ci_lower']:.3f}-{b['ci_upper']:.3f})"
        if "sensitivity" in boot:
            b = boot["sensitivity"]
            row["Sensitivity"] = f"{100*b['mean']:.1f}%"
            row["Sensitivity 95% CI"] = f"({100*b['ci_lower']:.1f}-{100*b['ci_upper']:.1f}%)"
        if "specificity" in boot:
            b = boot["specificity"]
            row["Specificity"] = f"{100*b['mean']:.1f}%"
            row["Specificity 95% CI"] = f"({100*b['ci_lower']:.1f}-{100*b['ci_upper']:.1f}%)"

        rows.append(row)

    tbl = pd.DataFrame(rows)
    tbl.to_csv(TABLES / "table_calibration_metrics.csv", index=False)
    tbl.to_csv(TABLES / "tableS9_calibration_metrics.csv", index=False)
    _print(f"\n  Saved table_calibration_metrics.csv ({len(tbl)} models)")

    # Print summary
    _print("\n  Calibration Summary:")
    _print(f"  {'Model':<35} {'AUROC':>12} {'Brier':>8} {'CITL':>8} {'Slope':>8}")
    _print(f"  {'-'*35} {'-'*12} {'-'*8} {'-'*8} {'-'*8}")
    for _, r in tbl.iterrows():
        auroc_str = f"{r['AUROC']} {r['AUROC 95% CI']}" if r['AUROC'] else "—"
        _print(f"  {r['Model']:<35} {auroc_str:>12} {r['Brier']:>8} "
               f"{r['CITL']:>8} {r['Calibration slope']:>8}")

    return tbl


# ===================================================================
# MAIN
# ===================================================================

def main():
    t_start = time.time()
    _print("=" * 60)
    _print("Brain AttACKs: Calibration Assessment (TRIPOD+AI)")
    _print("=" * 60)

    df, X, y, features = load_data()
    all_results = {}

    # --- 1. Pooled CV: XGBoost + LR ---
    _print("\n--- Pooled Cross-Validation Models ---")
    xgb_probs, xgb_preds, lr_probs, lr_preds = get_cv_predictions(X, y, features)

    _print("  Computing XGBoost CV calibration + CIs...")
    xgb_cal = calibration_metrics(y, xgb_probs)
    xgb_boot = bootstrap_ci(y, xgb_probs, xgb_preds)
    all_results["XGBoost (10-fold CV)"] = {
        "calibration": xgb_cal, "bootstrap": xgb_boot,
        "n": len(y), "events": int(y.sum()),
    }

    _print("  Computing LR CV calibration + CIs...")
    lr_cal = calibration_metrics(y, lr_probs)
    lr_boot = bootstrap_ci(y, lr_probs, lr_preds)
    all_results["LR (10-fold CV)"] = {
        "calibration": lr_cal, "bootstrap": lr_boot,
        "n": len(y), "events": int(y.sum()),
    }

    # --- 2. Temporal Validation ---
    _print("\n--- Temporal Validation Models ---")
    y2, tv_xgb_probs, tv_xgb_preds, tv_lr_probs, tv_lr_preds = \
        get_temporal_predictions(df, features)

    _print("  Computing XGBoost temporal calibration + CIs...")
    tv_xgb_cal = calibration_metrics(y2, tv_xgb_probs)
    tv_xgb_boot = bootstrap_ci(y2, tv_xgb_probs, tv_xgb_preds)
    all_results["XGBoost (Year 1→2)"] = {
        "calibration": tv_xgb_cal, "bootstrap": tv_xgb_boot,
        "n": len(y2), "events": int(y2.sum()),
    }

    _print("  Computing LR temporal calibration + CIs...")
    tv_lr_cal = calibration_metrics(y2, tv_lr_probs)
    tv_lr_boot = bootstrap_ci(y2, tv_lr_probs, tv_lr_preds)
    all_results["LR (Year 1→2)"] = {
        "calibration": tv_lr_cal, "bootstrap": tv_lr_boot,
        "n": len(y2), "events": int(y2.sum()),
    }

    # --- 3. Clinical Score ---
    _print("\n--- Clinical Prediction Score ---")
    score_y, score_probs, score_preds = get_score_predictions(df)
    score_cal = calibration_metrics(score_y, score_probs)
    score_boot = bootstrap_ci(score_y, score_probs, score_preds)
    all_results["Clinical score (CV)"] = {
        "calibration": score_cal, "bootstrap": score_boot,
        "n": len(score_y), "events": int(score_y.sum()),
    }

    # Note: Headache sub-analysis calibration has moved to 13-headache-deep.py (MS2)

    # --- Save results ---
    save_results(all_results)

    # --- Figures ---
    _print("\n--- Generating Figures ---")

    # Main calibration figure (2x2 panel)
    plot_calibration(
        [
            ("XGBoost (10-fold CV)", xgb_cal),
            ("LR (10-fold CV)", lr_cal),
            ("XGBoost (Year 1→2)", tv_xgb_cal),
            ("Clinical Score (CV)", score_cal),
        ],
        "fig_calibration_combined.png",
        "Calibration Plots",
        nrows=2, ncols=2,
    )

    # Temporal validation calibration
    plot_calibration(
        [
            ("XGBoost (Year 1→2)", tv_xgb_cal),
            ("LR (Year 1→2)", tv_lr_cal),
        ],
        "fig_calibration_temporal.png",
        "Temporal Validation: Calibration",
        nrows=1, ncols=2,
    )

    # Note: Headache calibration figure moved to 13-headache-deep.py (MS2)

    elapsed = time.time() - t_start
    _print(f"\n{'='*60}")
    _print(f"Done in {elapsed:.0f}s. Outputs in outputs/tables/ and outputs/figures/")
    _print(f"{'='*60}")


if __name__ == "__main__":
    main()
