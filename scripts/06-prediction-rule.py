"""
06-prediction-rule.py — Clinical Prediction Rule Derivation

Derives a simple clinical score from the XGBoost model's top predictors,
using logistic regression coefficients as weights.

Inputs:
    data/processed/brain_attacks_combined.parquet

Outputs:
    outputs/tables/table_prediction_rule.csv
    outputs/tables/table_score_performance.csv
    outputs/tables/tableS6_full_model_coefficients.csv  (TRIPOD+AI Item 24)
    outputs/tables/table_score_temporal_validation.csv
    outputs/figures/fig_score_performance.png
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data" / "processed"
TABLES = PROJECT / "outputs" / "tables"
FIGURES = PROJECT / "outputs" / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

TARGET = "outcome_med_important"

# Top predictors from XGBoost feature importance (03-prediction.py)
# Using clinical features only (binary or easily categorized at triage)
TOP_PREDICTORS = [
    "sx_ataxia",
    "exam_altered_loc",
    "ems",
    "sx_fever",
    "sx_focal_weakness",
    "exam_meningismus",
    "sx_collapse",        # protective
    "gender",             # male = higher risk
    "sx_seizure",
    "exam_papilledema",
]


# ===================================================================
# 1. LOAD DATA
# ===================================================================

def load_data():
    df = pd.read_parquet(DATA / "brain_attacks_combined.parquet")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    features = [f for f in TOP_PREDICTORS if f in df.columns]
    X = df[features].fillna(0).copy()
    y = df[TARGET].values
    print(f"Loaded: {len(df)} records, {len(features)} predictors")
    print(f"  Outcome: {y.sum()} MI ({100*y.mean():.1f}%)")
    return df, X, y, features


# ===================================================================
# 2. DERIVE CLINICAL SCORE
# ===================================================================

def derive_score(df, X, y, features):
    """
    Fit multivariate logistic regression, convert coefficients to integer weights.
    Protocol method: lowest absolute coefficient = 1 point, others scaled proportionally.
    """
    print("\n=== Clinical Score Derivation ===")

    # Fit on full cohort (derivation)
    lr = LogisticRegression(max_iter=1000, random_state=SEED, penalty="l2", C=1.0)
    lr.fit(X.values, y)

    coefs = pd.Series(lr.coef_[0], index=features)
    print("\n  Logistic regression coefficients:")
    for f, c in coefs.sort_values(ascending=False).items():
        or_val = np.exp(c)
        print(f"    {f}: beta={c:.3f}, OR={or_val:.2f}")

    # Convert to integer score weights
    # Use absolute values, round to nearest integer with minimum weight = 1
    abs_coefs = coefs.abs()
    min_coef = abs_coefs[abs_coefs > 0].min()
    raw_weights = (abs_coefs / min_coef).round().astype(int)

    # Assign sign (negative coefficients become negative points)
    weights = raw_weights * np.sign(coefs).astype(int)

    print("\n  Score weights:")
    rule_rows = []
    for f in features:
        direction = "risk" if coefs[f] > 0 else "protective"
        rule_rows.append({
            "Predictor": f,
            "Beta": f"{coefs[f]:.3f}",
            "OR": f"{np.exp(coefs[f]):.2f}",
            "Points": int(weights[f]),
            "Direction": direction,
        })
        print(f"    {f}: {int(weights[f])} points ({direction})")

    rule_df = pd.DataFrame(rule_rows)
    out = TABLES / "table_prediction_rule.csv"
    rule_df.to_csv(out, index=False)
    print(f"\n  Saved {out.name}")

    # Compute scores for each patient
    score = (X.values * weights.values).sum(axis=1)
    df["prediction_score"] = score
    print(f"\n  Score range: {score.min()} to {score.max()}")
    print(f"  Score distribution: mean={score.mean():.1f}, median={np.median(score):.0f}")

    return weights, score, rule_df


# ===================================================================
# 3. SCORE PERFORMANCE AT THRESHOLDS
# ===================================================================

def score_performance(df, score, y):
    """Evaluate score performance at each possible threshold."""
    print("\n=== Score Performance at Thresholds ===")

    thresholds = sorted(set(score))
    rows = []

    for thresh in thresholds:
        pred = (score >= thresh).astype(int)
        if pred.sum() == 0 or pred.sum() == len(pred):
            continue
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        lr_pos = sens / (1 - spec) if spec < 1 else np.inf
        lr_neg = (1 - sens) / spec if spec > 0 else np.inf

        rows.append({
            "Threshold": int(thresh),
            "Sensitivity": f"{100*sens:.1f}%",
            "Specificity": f"{100*spec:.1f}%",
            "PPV": f"{100*ppv:.1f}%",
            "NPV": f"{100*npv:.1f}%",
            "LR+": f"{lr_pos:.2f}" if lr_pos < 100 else ">100",
            "LR-": f"{lr_neg:.2f}",
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "_sens": sens, "_spec": spec,
        })

    tbl = pd.DataFrame(rows)
    out = TABLES / "table_score_performance.csv"
    tbl.drop(columns=["_sens", "_spec"]).to_csv(out, index=False)
    print(f"  Saved {out.name} ({len(tbl)} thresholds)")

    # Print key thresholds
    print("\n  Key thresholds:")
    for _, r in tbl.iterrows():
        t = r["Threshold"]
        print(f"    Score >= {t}: Sens={r['Sensitivity']}, Spec={r['Specificity']}, "
              f"PPV={r['PPV']}, NPV={r['NPV']}, LR+={r['LR+']}, LR-={r['LR-']}")

    # Risk stratification
    print("\n  Suggested risk strata:")
    med_thresh = tbl.iloc[len(tbl)//2]["Threshold"] if len(tbl) > 2 else 1

    # Find threshold with sensitivity >= 90%
    high_sens = tbl[tbl["_sens"] >= 0.90]
    if len(high_sens) > 0:
        low_risk_thresh = high_sens.iloc[-1]["Threshold"]
        print(f"    Low risk: score < {low_risk_thresh} (sensitivity >= 90%)")
    else:
        low_risk_thresh = tbl.iloc[0]["Threshold"]
        print(f"    Low risk: score < {low_risk_thresh} (best available)")

    # Find threshold with specificity >= 90%
    high_spec = tbl[tbl["_spec"] >= 0.90]
    if len(high_spec) > 0:
        high_risk_thresh = high_spec.iloc[0]["Threshold"]
        print(f"    High risk: score >= {high_risk_thresh} (specificity >= 90%)")
    else:
        high_risk_thresh = tbl.iloc[-1]["Threshold"]

    print(f"    Moderate risk: {low_risk_thresh} <= score < {high_risk_thresh}")

    # Cross-validated AUROC using the raw score
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    aurocs = []
    for train_idx, test_idx in cv.split(score.reshape(-1, 1), y):
        from sklearn.linear_model import LogisticRegression as LR
        m = LR(max_iter=1000, random_state=SEED)
        m.fit(score[train_idx].reshape(-1, 1), y[train_idx])
        prob = m.predict_proba(score[test_idx].reshape(-1, 1))[:, 1]
        aurocs.append(roc_auc_score(y[test_idx], prob))

    print(f"\n  Score AUROC (10-fold CV): {np.mean(aurocs):.3f} (+/- {np.std(aurocs):.3f})")

    return tbl


# ===================================================================
# 4. FIGURE
# ===================================================================

def make_figure(tbl):
    """Sensitivity/specificity tradeoff by score threshold."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

    print("\n=== Figure: Score Performance ===")

    thresholds = tbl["Threshold"].values
    sens = tbl["_sens"].values * 100
    spec = tbl["_spec"].values * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, sens, "o-", color="#d62728", label="Sensitivity", linewidth=2)
    ax.plot(thresholds, spec, "s-", color="#1f77b4", label="Specificity", linewidth=2)
    ax.set_xlabel("Score Threshold")
    ax.set_ylabel("Performance (%)")
    ax.set_title("Clinical Prediction Score: Sensitivity vs Specificity")
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)

    # Mark where curves cross
    for i in range(len(thresholds) - 1):
        if (sens[i] >= spec[i]) != (sens[i+1] >= spec[i+1]):
            ax.axvline(x=(thresholds[i] + thresholds[i+1]) / 2,
                       color="gray", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig_score_performance.png")
    plt.close()
    print("  Saved fig_score_performance.png")


# ===================================================================
# 5. FULL MODEL COEFFICIENTS (TRIPOD+AI Item 24)
# ===================================================================

def full_model_coefficients(X, y, features):
    """
    Report full logistic regression specification: coefficients, SEs,
    ORs with 95% CIs, and intercept (TRIPOD+AI Item 24 / TableS6).
    """
    print("\n=== Full Model Coefficients (TRIPOD+AI Item 24) ===")

    lr = LogisticRegression(max_iter=1000, random_state=SEED, penalty="l2", C=1.0)
    lr.fit(X.values, y)

    coefs = lr.coef_[0]
    intercept = lr.intercept_[0]

    # Bootstrap SEs (1000 iterations, stratified)
    n_boot = 1000
    rng = np.random.RandomState(SEED)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    boot_coefs = []

    for _ in range(n_boot):
        pos_sample = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        neg_sample = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([pos_sample, neg_sample])
        m = LogisticRegression(max_iter=1000, random_state=SEED, penalty="l2", C=1.0)
        try:
            m.fit(X.values[idx], y[idx])
            boot_coefs.append(np.append(m.coef_[0], m.intercept_[0]))
        except Exception:
            continue

    boot_arr = np.array(boot_coefs)
    se_vals = boot_arr.std(axis=0)

    rows = []
    for i, f in enumerate(features):
        beta = coefs[i]
        se = se_vals[i]
        or_val = np.exp(beta)
        or_lo = np.exp(beta - 1.96 * se)
        or_hi = np.exp(beta + 1.96 * se)
        rows.append({
            "Variable": f,
            "Beta": f"{beta:.4f}",
            "SE": f"{se:.4f}",
            "OR": f"{or_val:.3f}",
            "OR 95% CI lower": f"{or_lo:.3f}",
            "OR 95% CI upper": f"{or_hi:.3f}",
        })

    # Intercept
    rows.append({
        "Variable": "(Intercept)",
        "Beta": f"{intercept:.4f}",
        "SE": f"{se_vals[-1]:.4f}",
        "OR": "—",
        "OR 95% CI lower": "—",
        "OR 95% CI upper": "—",
    })

    tbl = pd.DataFrame(rows)
    out = TABLES / "tableS6_full_model_coefficients.csv"
    tbl.to_csv(out, index=False)
    print(f"  Saved {out.name} ({len(tbl)} rows)")

    for _, r in tbl.iterrows():
        print(f"    {r['Variable']}: beta={r['Beta']}, SE={r['SE']}, "
              f"OR={r['OR']} ({r['OR 95% CI lower']}-{r['OR 95% CI upper']})")

    return tbl


# ===================================================================
# 6. TEMPORAL VALIDATION OF CLINICAL SCORE (C2)
# ===================================================================

def temporal_validation_score(df, features):
    """
    Derive score from Year 1 only, apply to Year 2.
    Reports AUROC, sensitivity/specificity at risk strata, calibration.
    """
    print("\n=== Temporal Validation: Clinical Score (Y1 → Y2) ===")

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    y1 = df[df["year"] == 1]
    y2 = df[df["year"] == 2]

    feat_cols = [f for f in features if f in df.columns]
    X1 = y1[feat_cols].fillna(0)
    X2 = y2[feat_cols].fillna(0)
    y1_out = y1[TARGET].values
    y2_out = y2[TARGET].values

    print(f"  Year 1: {len(y1)} records ({y1_out.sum()} MI)")
    print(f"  Year 2: {len(y2)} records ({y2_out.sum()} MI)")

    # Derive score weights from Year 1 only
    lr = LogisticRegression(max_iter=1000, random_state=SEED, penalty="l2", C=1.0)
    lr.fit(X1.values, y1_out)

    coefs = pd.Series(lr.coef_[0], index=feat_cols)
    abs_coefs = coefs.abs()
    min_coef = abs_coefs[abs_coefs > 0].min()
    raw_weights = (abs_coefs / min_coef).round().astype(int)
    weights = raw_weights * np.sign(coefs).astype(int)

    print("\n  Score weights (derived from Y1):")
    for f in feat_cols:
        print(f"    {f}: {int(weights[f])} points")

    # Apply to Year 2
    score_y2 = (X2.values * weights.values).sum(axis=1)

    # Convert to probabilities via LR on score (trained on Y1)
    score_y1 = (X1.values * weights.values).sum(axis=1)
    lr_score = LogisticRegression(max_iter=1000, random_state=SEED)
    lr_score.fit(score_y1.reshape(-1, 1), y1_out)
    prob_y2 = lr_score.predict_proba(score_y2.reshape(-1, 1))[:, 1]
    pred_y2 = lr_score.predict(score_y2.reshape(-1, 1))

    # AUROC
    auroc = roc_auc_score(y2_out, prob_y2)
    print(f"\n  Temporal validation AUROC: {auroc:.3f}")

    # Performance at thresholds
    from sklearn.metrics import confusion_matrix as cm_func
    tn, fp, fn, tp = cm_func(y2_out, pred_y2).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"  Sensitivity: {100*sens:.1f}%")
    print(f"  Specificity: {100*spec:.1f}%")
    print(f"  Confusion: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    # Score thresholds on Y2
    thresholds = sorted(set(score_y2))
    threshold_rows = []
    for thresh in thresholds:
        p = (score_y2 >= thresh).astype(int)
        if p.sum() == 0 or p.sum() == len(p):
            continue
        tn_t, fp_t, fn_t, tp_t = cm_func(y2_out, p).ravel()
        s = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
        sp = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0
        ppv = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
        npv = tn_t / (tn_t + fn_t) if (tn_t + fn_t) > 0 else 0
        threshold_rows.append({
            "Threshold": int(thresh),
            "Sensitivity": f"{100*s:.1f}%",
            "Specificity": f"{100*sp:.1f}%",
            "PPV": f"{100*ppv:.1f}%",
            "NPV": f"{100*npv:.1f}%",
            "TP": tp_t, "FP": fp_t, "FN": fn_t, "TN": tn_t,
        })

    # Calibration metrics
    from sklearn.metrics import brier_score_loss
    brier = brier_score_loss(y2_out, prob_y2)
    prob_clipped = np.clip(prob_y2, 1e-6, 1 - 1e-6)
    lp = np.log(prob_clipped / (1 - prob_clipped))
    try:
        lr_cal = LogisticRegression(max_iter=1000, penalty=None, solver="lbfgs")
        lr_cal.fit(lp.reshape(-1, 1), y2_out)
        cal_slope = lr_cal.coef_[0][0]
    except Exception:
        cal_slope = float("nan")

    expected = prob_y2.sum()
    observed = y2_out.sum()
    citl = np.log(observed / expected) if expected > 0 else float("inf")

    print(f"\n  Calibration:")
    print(f"    Brier score: {brier:.4f}")
    print(f"    CITL: {citl:.3f}")
    print(f"    Calibration slope: {cal_slope:.3f}")

    # Save temporal validation table
    result_rows = [
        {"Metric": "N (Year 2)", "Value": str(len(y2))},
        {"Metric": "Events (Year 2)", "Value": str(int(y2_out.sum()))},
        {"Metric": "AUROC", "Value": f"{auroc:.3f}"},
        {"Metric": "Sensitivity", "Value": f"{100*sens:.1f}%"},
        {"Metric": "Specificity", "Value": f"{100*spec:.1f}%"},
        {"Metric": "Brier score", "Value": f"{brier:.4f}"},
        {"Metric": "CITL", "Value": f"{citl:.3f}"},
        {"Metric": "Calibration slope", "Value": f"{cal_slope:.3f}"},
    ]
    result_tbl = pd.DataFrame(result_rows)
    out = TABLES / "table_score_temporal_validation.csv"
    result_tbl.to_csv(out, index=False)
    print(f"\n  Saved {out.name}")

    if threshold_rows:
        thresh_tbl = pd.DataFrame(threshold_rows)
        out2 = TABLES / "table_score_temporal_thresholds.csv"
        thresh_tbl.to_csv(out2, index=False)
        print(f"  Saved {out2.name}")

    return auroc, brier, cal_slope


# ===================================================================
# MAIN
# ===================================================================

def main():
    print("=" * 60)
    print("Brain AttACKs: Clinical Prediction Rule (Objective 1)")
    print("=" * 60)

    df, X, y, features = load_data()
    weights, score, rule_df = derive_score(df, X, y, features)
    tbl = score_performance(df, score, y)
    make_figure(tbl)

    # TRIPOD+AI Item 24: Full model coefficients
    full_model_coefficients(X, y, features)

    # C2: Temporal validation of clinical score
    temporal_validation_score(df, features)

    print("\n" + "=" * 60)
    print("Done. Outputs in outputs/tables/ and outputs/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
