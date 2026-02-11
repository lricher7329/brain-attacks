"""
12-cost-analysis.py — Cost-Informed Threshold Analysis (Manuscript 1)

Extends the decision curve analysis (script 10) by incorporating investigation
costs from the Alberta Schedule of Medical Benefits (SOMB). Translates model
performance into healthcare dollars: cost per MI detected and net monetary
benefit at each threshold probability.

Three cost scenarios:
    1. Base: Alberta SOMB professional fee only
    2. Mid: Professional fee × 2.5 (total institutional cost proxy)
    3. High: CADTH/published per-scan costs

Cost parameters are populated from CADTH 2023 Health Technology Review (CT/MRI
private facility costs) and estimated Alberta institutional costs. Three scenarios
(base/mid/high) are used for deterministic sensitivity analysis.

Reporting framework: CHEERS 2022 (Husereau et al., BMC Medicine 2022) applies
to this cost-consequence analysis. Non-applicable items (e.g., QALYs, ICERs,
time horizon, discounting) are marked N/A.

Inputs:
    data/processed/brain_attacks_combined.parquet

Outputs:
    outputs/tables/table_cost_parameters.csv
    outputs/tables/table_cost_thresholds.csv
    outputs/tables/tableS11_cost_sensitivity.csv
    outputs/tables/tableS12_investigation_mix.csv
    outputs/figures/fig_nmb_curve.png
    outputs/figures/fig_cost_per_case.png
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, make_scorer
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
# COST PARAMETERS — Published Canadian Investigation Costs (CAD)
# ===================================================================
# Three scenarios for sensitivity analysis:
#   base: Estimated professional fee component (~30% of institutional cost)
#   mid:  CADTH 2023 average institutional cost (published mean)
#   high: Upper range / private facility pricing
#
# Sources:
#   CT/MRI: CADTH Health Technology Review, Oct 2023 (NBK598233)
#           "Use of MRI and CT in Private Imaging Facilities in Canada
#            (2021-2023)". Mean CT=$310, range $192-490. Mean MRI=$762,
#            range $640-990. Based on facilities in BC, AB, SK, QC.
#   CTA/MRA: Estimated from CT/MRI with contrast/angiography premium.
#   LP/EEG:  No published Canadian per-procedure cost available.
#            Estimated from provincial fee schedules and US literature
#            adjusted for CAD/USD and system differences.
#   Consultation: Alberta SOMB 03.08A ~$79 (professional fee);
#                 institutional cost estimated at ~$200.
#
# Pediatric modifier: +30% for patients <= 12 years applies to some
# diagnostic radiology codes. Applied automatically in cost calculation.
#
# NOTE: These are literature-based estimates. For Alberta-specific
# professional fees, consult the SOMB (open.alberta.ca/publications/somb)
# or AMA Fee Navigator (apps.albertadoctors.org/fee-navigator/).

COST_PARAMS = {
    "ct_head": {
        "hsc_code": "X125A / X125",
        "somb_section": "Diagnostic Radiology - CT",
        "base": 95.0,    # ~30% of CADTH mean ($310)
        "mid": 310.0,    # CADTH 2023 mean
        "high": 490.0,   # CADTH 2023 upper range
    },
    "mri_brain": {
        "hsc_code": "X482A / X482",
        "somb_section": "Diagnostic Radiology - MRI",
        "base": 230.0,   # ~30% of CADTH mean ($762)
        "mid": 762.0,    # CADTH 2023 mean
        "high": 990.0,   # CADTH 2023 upper range
    },
    "cta_head": {
        "hsc_code": "X127 / X126",
        "somb_section": "Diagnostic Radiology - CT",
        "base": 140.0,   # ~1.5× CT base (contrast premium)
        "mid": 465.0,    # ~1.5× CT mid
        "high": 670.0,   # Private clinic CT with contrast
    },
    "mra_head": {
        "hsc_code": "X484",
        "somb_section": "Diagnostic Radiology - MRI",
        "base": 300.0,   # ~1.3× MRI base (angiography premium)
        "mid": 990.0,    # ~1.3× MRI mid
        "high": 1300.0,  # Estimated upper range
    },
    "lp": {
        "hsc_code": "13.99C",
        "somb_section": "General Procedures",
        "base": 80.0,    # Estimated professional fee
        "mid": 200.0,    # Estimated institutional cost
        "high": 350.0,   # Estimated upper range
    },
    "eeg": {
        "hsc_code": "X600",
        "somb_section": "Diagnostic Services",
        "base": 80.0,    # Estimated professional fee
        "mid": 250.0,    # Estimated institutional cost
        "high": 400.0,   # Estimated upper range
    },
    "consultation": {
        "hsc_code": "03.08A",
        "somb_section": "Consultation Services",
        "base": 80.0,    # Alberta SOMB 03.08A ~$79
        "mid": 200.0,    # Estimated institutional cost
        "high": 350.0,   # Estimated upper range
    },
}

# Map from dataset investigation columns to cost parameter keys
INV_COL_MAP = {
    "inv_ct": "ct_head",
    "inv_mri": "mri_brain",
    "inv_cta": "cta_head",
    "inv_mra": "mra_head",
    "inv_lp": "lp",
    "inv_eeg": "eeg",
}

PEDIATRIC_MODIFIER_COLS = ["inv_ct", "inv_mri", "inv_cta", "inv_mra"]
PEDIATRIC_MODIFIER = 0.30  # +30% for patients <= 12 years


# ===================================================================
# FEATURE DEFINITIONS (same as 03-prediction.py / 10-decision-curve.py)
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
# 2. COST PARAMETERS TABLE
# ===================================================================

def save_cost_parameters():
    """Save cost parameter documentation table."""
    rows = []
    for inv, params in COST_PARAMS.items():
        rows.append({
            "Investigation": inv.replace("_", " ").title(),
            "HSC Code": params["hsc_code"],
            "SOMB Section": params["somb_section"],
            "Base (Prof. fee)": f"${params['base']:.2f}",
            "Mid (×2.5)": f"${params['mid']:.2f}",
            "High (CADTH)": f"${params['high']:.2f}",
        })
    tbl = pd.DataFrame(rows)
    tbl.to_csv(TABLES / "table_cost_parameters.csv", index=False)
    _print(f"  Saved table_cost_parameters.csv")

    # Check for placeholders
    all_zero = all(
        p[s] == 0.0
        for p in COST_PARAMS.values()
        for s in ["base", "mid", "high"]
    )
    if all_zero:
        _print("\n  *** WARNING: All cost parameters are $0.00 (placeholders). ***")
        _print("  *** Populate from Alberta SOMB before interpreting results. ***\n")

    return tbl


# ===================================================================
# 3. OBSERVED INVESTIGATION MIX
# ===================================================================

def investigation_mix(df):
    """Compute investigation rates and expected cost per investigated patient."""
    _print("\n=== Observed Investigation Mix ===")

    inv_cols = [c for c in INV_COL_MAP.keys() if c in df.columns]
    n_total = len(df)

    rows = []
    for col in inv_cols:
        n_inv = (df[col] == 1).sum()
        cost_key = INV_COL_MAP[col]

        rows.append({
            "Investigation": cost_key.replace("_", " ").title(),
            "Column": col,
            "N": int(n_inv),
            "Rate (all patients)": f"{100*n_inv/n_total:.1f}%",
            "Rate (among investigated)": "",  # filled below
        })

    # Any investigation
    any_inv = df[[c for c in inv_cols if c in df.columns]].max(axis=1)
    n_any = (any_inv == 1).sum()

    for row in rows:
        col = row["Column"]
        n_inv = (df[col] == 1).sum()
        row["Rate (among investigated)"] = f"{100*n_inv/n_any:.1f}%" if n_any > 0 else "0%"

    # Consultation (Year 2 only)
    y2 = df[df["year"] == 2]
    if "consult" in y2.columns:
        n_consult = (y2["consult"] == 1).sum()
        rows.append({
            "Investigation": "Consultation (Y2 only)",
            "Column": "consult",
            "N": int(n_consult),
            "Rate (all patients)": f"{100*n_consult/len(y2):.1f}% (of Y2)",
            "Rate (among investigated)": "—",
        })

    rows.append({
        "Investigation": "Any investigation",
        "Column": "—",
        "N": int(n_any),
        "Rate (all patients)": f"{100*n_any/n_total:.1f}%",
        "Rate (among investigated)": "100.0%",
    })

    tbl = pd.DataFrame(rows)
    tbl.to_csv(TABLES / "tableS12_investigation_mix.csv", index=False)
    _print(f"  Saved tableS12_investigation_mix.csv")
    _print(tbl[["Investigation", "N", "Rate (all patients)"]].to_string(index=False))
    return tbl


# ===================================================================
# 4. PER-PATIENT INVESTIGATION COST
# ===================================================================

def compute_patient_cost(df, scenario="base"):
    """
    Compute expected investigation cost per patient based on actual
    investigations received. Returns array of per-patient costs.
    """
    costs = np.zeros(len(df))
    inv_cols = [c for c in INV_COL_MAP.keys() if c in df.columns]
    is_pediatric = (df["age_under_12"].values == 1) if "age_under_12" in df.columns else np.zeros(len(df), dtype=bool)

    for col in inv_cols:
        cost_key = INV_COL_MAP[col]
        unit_cost = COST_PARAMS[cost_key][scenario]
        had_inv = (df[col].values == 1).astype(float)

        # Apply pediatric modifier for radiology
        if col in PEDIATRIC_MODIFIER_COLS:
            modifier = np.where(is_pediatric, 1 + PEDIATRIC_MODIFIER, 1.0)
        else:
            modifier = np.ones(len(df))

        costs += had_inv * unit_cost * modifier

    return costs


def mean_investigation_cost(df, scenario="base"):
    """Mean investigation cost among those who had any investigation."""
    inv_cols = [c for c in INV_COL_MAP.keys() if c in df.columns]
    any_inv = df[inv_cols].max(axis=1) == 1
    if any_inv.sum() == 0:
        return 0.0
    patient_costs = compute_patient_cost(df, scenario)
    return patient_costs[any_inv].mean()


# ===================================================================
# 5. OUT-OF-FOLD PREDICTIONS (replicated from script 10)
# ===================================================================

def get_cv_predictions(X, y, features):
    """Get out-of-fold predictions for XGBoost, LR, and clinical score."""
    _print("\n=== Generating Out-of-Fold Predictions ===")

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
# 6. COST AT EACH THRESHOLD
# ===================================================================

def cost_at_thresholds(y, probs, mean_inv_cost, thresholds, label="Model"):
    """
    For each threshold: N flagged, investigation cost, MI detected/missed,
    cost per MI detected.
    """
    n_total = len(y)
    n_mi = y.sum()
    results = []

    for t in thresholds:
        flagged = probs >= t
        n_flagged = flagged.sum()
        tp = ((flagged == 1) & (y == 1)).sum()
        fp = ((flagged == 1) & (y == 0)).sum()
        fn = ((flagged == 0) & (y == 1)).sum()

        total_cost = n_flagged * mean_inv_cost
        cost_per_mi = total_cost / tp if tp > 0 else np.inf

        results.append({
            "threshold": t,
            "n_flagged": int(n_flagged),
            "pct_flagged": 100 * n_flagged / n_total,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "mi_detected_pct": 100 * tp / n_mi if n_mi > 0 else 0,
            "total_cost": total_cost,
            "cost_per_mi": cost_per_mi,
        })

    return results


# ===================================================================
# 7. NET MONETARY BENEFIT
# ===================================================================

WTP_SENSITIVITY_RANGE = [5000, 10000, 25000, 50000]  # CAD per MI detected


def net_monetary_benefit(y, probs, mean_inv_cost, thresholds, willingness_per_mi=None):
    """
    NMB(t) = (TP/N) * W_tp - (FP/N) * C_inv

    Where W_tp is the willingness-to-pay per true positive detected.
    If not provided, we use 10× the mean investigation cost as a default
    (i.e., society values detecting one MI case at 10× the cost of one
    investigation workup).
    """
    if willingness_per_mi is None:
        # Default: 10× investigation cost, or $10,000 if costs are zero
        willingness_per_mi = max(mean_inv_cost * 10, 10000.0)

    n = len(y)
    prevalence = y.mean()
    results = []

    for t in thresholds:
        flagged = probs >= t
        tp = ((flagged == 1) & (y == 1)).sum()
        fp = ((flagged == 1) & (y == 0)).sum()

        nmb = (tp / n) * willingness_per_mi - (fp / n) * mean_inv_cost

        # Investigate-all NMB
        nmb_all = prevalence * willingness_per_mi - (1 - prevalence) * mean_inv_cost

        results.append({
            "threshold": t,
            "nmb": nmb,
            "nmb_all": nmb_all,
            "tp": int(tp),
            "fp": int(fp),
        })

    return results


# ===================================================================
# 8. FIGURES
# ===================================================================

def make_nmb_figure(y, model_nmb, thresholds):
    """Net monetary benefit vs. threshold for all models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

    _print("\n=== NMB Figure ===")

    fig, ax = plt.subplots(figsize=(9, 6))

    colors = {
        "XGBoost": "#d62728",
        "Logistic regression": "#1f77b4",
        "Clinical score": "#2ca02c",
    }

    # Investigate-all line (same across models, use first)
    first_label = list(model_nmb.keys())[0]
    nmb_all = [r["nmb_all"] for r in model_nmb[first_label]]
    ax.plot(thresholds * 100, nmb_all, "k-", alpha=0.5, linewidth=1.5,
            label="Investigate all")

    # Investigate-none line
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1,
               label="Investigate none")

    # Model lines
    for label, results in model_nmb.items():
        nmb_vals = [r["nmb"] for r in results]
        ax.plot(thresholds * 100, nmb_vals, "-", color=colors.get(label, "#333"),
                linewidth=2, label=label)

    ax.set_xlabel("Threshold probability (%)")
    ax.set_ylabel("Net monetary benefit ($)")
    ax.set_title("Cost-Informed Decision Curve: Net Monetary Benefit")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, 50)
    ax.grid(alpha=0.2)

    prevalence = y.mean()
    ax.axvline(x=prevalence * 100, color="gray", linestyle=":", alpha=0.4)
    ax.text(prevalence * 100 + 0.5, ax.get_ylim()[1] * 0.9,
            f"Prevalence\n({100*prevalence:.1f}%)", fontsize=8, color="gray")

    fig.tight_layout()
    fig.savefig(FIGURES / "fig_nmb_curve.png")
    plt.close()
    _print("  Saved fig_nmb_curve.png")


def make_cost_per_case_figure(model_cost_results, thresholds):
    """Cost per MI detected vs. threshold for all models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

    _print("\n=== Cost Per MI Detected Figure ===")

    fig, ax = plt.subplots(figsize=(9, 6))

    colors = {
        "XGBoost": "#d62728",
        "Logistic regression": "#1f77b4",
        "Clinical score": "#2ca02c",
    }

    for label, results in model_cost_results.items():
        costs = []
        valid_t = []
        for r in results:
            if r["cost_per_mi"] < np.inf and r["tp"] > 0:
                costs.append(r["cost_per_mi"])
                valid_t.append(r["threshold"] * 100)
        if valid_t:
            ax.plot(valid_t, costs, "-o", color=colors.get(label, "#333"),
                    linewidth=2, markersize=3, label=label)

    ax.set_xlabel("Threshold probability (%)")
    ax.set_ylabel("Cost per MI detected ($)")
    ax.set_title("Investigation Cost per Medically Important Case Detected")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(0, 50)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig_cost_per_case.png")
    plt.close()
    _print("  Saved fig_cost_per_case.png")


# ===================================================================
# 9. RESULTS TABLES
# ===================================================================

def save_threshold_table(y, model_cost_results, scenario_label="base"):
    """Save cost at key thresholds for all models."""
    key_thresholds = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
    thresholds_arr = np.arange(0.01, 0.51, 0.01)

    rows = []
    for t in key_thresholds:
        idx = np.argmin(np.abs(thresholds_arr - t))
        row = {"Threshold (%)": f"{100*t:.0f}%"}

        for label, results in model_cost_results.items():
            r = results[idx]
            row[f"{label} N flagged"] = r["n_flagged"]
            row[f"{label} MI detected"] = r["tp"]
            row[f"{label} MI missed"] = r["fn"]
            row[f"{label} MI detected (%)"] = f"{r['mi_detected_pct']:.1f}%"
            row[f"{label} Total cost"] = f"${r['total_cost']:.0f}"
            row[f"{label} Cost/MI"] = f"${r['cost_per_mi']:.0f}" if r["cost_per_mi"] < np.inf else "—"

        # Investigate-all reference
        n_total = len(y)
        n_mi = y.sum()
        inv_cost = results[0]["total_cost"] / results[0]["n_flagged"] if results[0]["n_flagged"] > 0 else 0
        row["Investigate all N"] = n_total
        row["Investigate all cost"] = f"${n_total * inv_cost:.0f}" if inv_cost > 0 else "$0"

        rows.append(row)

    tbl = pd.DataFrame(rows)
    return tbl


def save_sensitivity_table(y, all_scenario_results, thresholds):
    """Save results across cost scenarios (base, mid, high)."""
    key_thresholds = [0.05, 0.10, 0.20]

    rows = []
    for scenario, model_results in all_scenario_results.items():
        for t in key_thresholds:
            idx = np.argmin(np.abs(thresholds - t))
            for label, results in model_results.items():
                r = results[idx]
                rows.append({
                    "Cost scenario": scenario,
                    "Threshold (%)": f"{100*t:.0f}%",
                    "Model": label,
                    "N flagged": r["n_flagged"],
                    "MI detected": r["tp"],
                    "MI detected (%)": f"{r['mi_detected_pct']:.1f}%",
                    "Total cost": f"${r['total_cost']:.0f}",
                    "Cost/MI detected": f"${r['cost_per_mi']:.0f}" if r["cost_per_mi"] < np.inf else "—",
                })

    tbl = pd.DataFrame(rows)
    tbl.to_csv(TABLES / "tableS11_cost_sensitivity.csv", index=False)
    _print(f"  Saved tableS11_cost_sensitivity.csv ({len(tbl)} rows)")
    return tbl


# ===================================================================
# MAIN
# ===================================================================

def main():
    t_start = time.time()
    _print("=" * 60)
    _print("Brain AttACKs: Cost-Informed Threshold Analysis")
    _print("=" * 60)

    df, X, y, features = load_data()

    # --- Cost parameters ---
    save_cost_parameters()

    # --- Investigation mix ---
    investigation_mix(df)

    # --- Mean investigation cost per investigated patient ---
    scenarios = ["base", "mid", "high"]
    mean_costs = {}
    for s in scenarios:
        mc = mean_investigation_cost(df, scenario=s)
        mean_costs[s] = mc
        _print(f"  Mean investigation cost ({s}): ${mc:.2f}")

    # --- Get predictions ---
    xgb_probs, lr_probs, score_probs = get_cv_predictions(X, y, features)

    # --- Cost analysis at each threshold ---
    thresholds = np.arange(0.01, 0.51, 0.01)
    _print("\n=== Cost at Each Threshold ===")

    # Base scenario for main tables/figures
    base_cost = mean_costs["base"]
    model_cost_results = {}
    for label, probs in [
        ("XGBoost", xgb_probs),
        ("Logistic regression", lr_probs),
        ("Clinical score", score_probs),
    ]:
        results = cost_at_thresholds(y, probs, base_cost, thresholds, label)
        model_cost_results[label] = results
        _print(f"  {label}: computed across {len(thresholds)} thresholds")

    # --- Main threshold table ---
    tbl = save_threshold_table(y, model_cost_results)
    tbl.to_csv(TABLES / "table_cost_thresholds.csv", index=False)
    _print(f"\n  Saved table_cost_thresholds.csv")

    _print("\n  Cost at Key Thresholds (base scenario):")
    for _, row in tbl.iterrows():
        _print(f"    {row['Threshold (%)']}: "
               f"XGB {row.get('XGBoost MI detected (%)', '—')} detected, "
               f"cost/MI={row.get('XGBoost Cost/MI', '—')}")

    # --- Net monetary benefit ---
    _print("\n=== Net Monetary Benefit ===")
    model_nmb = {}
    for label, probs in [
        ("XGBoost", xgb_probs),
        ("Logistic regression", lr_probs),
        ("Clinical score", score_probs),
    ]:
        results = net_monetary_benefit(y, probs, base_cost, thresholds)
        model_nmb[label] = results

    # --- WTP sensitivity analysis (M4) ---
    _print("\n=== WTP Threshold Sensitivity Analysis ===")
    _print(f"  WTP range: ${min(WTP_SENSITIVITY_RANGE):,} - ${max(WTP_SENSITIVITY_RANGE):,}")
    wtp_rows = []
    key_thresh = [0.05, 0.10, 0.20]
    for wtp in WTP_SENSITIVITY_RANGE:
        for label, probs in [("XGBoost", xgb_probs), ("Clinical score", score_probs)]:
            nmb_results = net_monetary_benefit(y, probs, base_cost, thresholds,
                                                willingness_per_mi=wtp)
            for t in key_thresh:
                idx = np.argmin(np.abs(thresholds - t))
                r = nmb_results[idx]
                wtp_rows.append({
                    "WTP ($/MI detected)": f"${wtp:,}",
                    "Model": label,
                    "Threshold (%)": f"{100*t:.0f}%",
                    "NMB ($)": f"${r['nmb']:.0f}",
                    "TP": r["tp"],
                    "FP": r["fp"],
                })
    wtp_tbl = pd.DataFrame(wtp_rows)
    wtp_out = TABLES / "tableS13_wtp_sensitivity.csv"
    wtp_tbl.to_csv(wtp_out, index=False)
    _print(f"  Saved {wtp_out.name} ({len(wtp_tbl)} rows)")

    # --- Sensitivity analysis across cost scenarios ---
    _print("\n=== Cost Sensitivity Analysis ===")
    all_scenario_results = {}
    for s in scenarios:
        mc = mean_costs[s]
        scenario_results = {}
        for label, probs in [
            ("XGBoost", xgb_probs),
            ("Logistic regression", lr_probs),
            ("Clinical score", score_probs),
        ]:
            scenario_results[label] = cost_at_thresholds(y, probs, mc, thresholds, label)
        all_scenario_results[s] = scenario_results
        _print(f"  {s}: mean cost ${mc:.2f}")

    save_sensitivity_table(y, all_scenario_results, thresholds)

    # --- Figures ---
    make_nmb_figure(y, model_nmb, thresholds)
    make_cost_per_case_figure(model_cost_results, thresholds)

    # --- Summary ---
    _print("\n=== Cost Analysis Summary ===")
    if base_cost == 0:
        _print("  NOTE: All costs are $0 (placeholders). Results show counts only.")
        _print("  Populate COST_PARAMS from Alberta SOMB for dollar estimates.")
    else:
        _print(f"  Mean investigation cost (base): ${base_cost:.2f}")
        # Report key threshold findings
        for label, results in model_cost_results.items():
            idx_10 = np.argmin(np.abs(thresholds - 0.10))
            r = results[idx_10]
            _print(f"  {label} at 10%: {r['n_flagged']} flagged, "
                   f"{r['tp']} MI detected ({r['mi_detected_pct']:.0f}%), "
                   f"cost/MI=${r['cost_per_mi']:.0f}")

    elapsed = time.time() - t_start
    _print(f"\n{'='*60}")
    _print(f"Done in {elapsed:.0f}s. Outputs in outputs/tables/ and outputs/figures/")
    _print(f"{'='*60}")


if __name__ == "__main__":
    main()
