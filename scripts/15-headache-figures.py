"""
15-headache-figures.py — Generate figures and supplementary material for Manuscript 2

Generates:
  Main figures:
    - fig_ha_flow.png          (Figure 1: Patient flow diagram)
    - fig_ha_forest_plot.png   (Figure 6: Forest plot of predictor aORs)

  Supplementary tables:
    - etable1_appropriateness_by_year.csv
    - etable2_calibration_metrics.csv

  Also copies existing figures to the manuscripts output folder.

Output folder: ~/Outputs/manuscripts/brain-attacks/
"""

import shutil
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy import stats
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data" / "processed"
TABLES = PROJECT / "outputs" / "tables"
FIGURES = PROJECT / "outputs" / "figures"
MS_OUT = Path.home() / "Outputs" / "manuscripts" / "brain-attacks"
MS_OUT.mkdir(parents=True, exist_ok=True)

TARGET = "outcome_med_important"

# Predictor definitions (same as 13-headache-deep.py)
IHC_PREDICTORS = [
    "age_under_12", "gender", "ems", "exam_altered_loc",
    "sx_focal_weakness", "sx_ataxia", "sx_speech", "sx_visual",
    "ha_progressive_days", "ha_vomiting", "sx_fever",
]
EXTENDED_PREDICTORS = IHC_PREDICTORS + [
    "ha_photophobia", "ha_phonophobia", "ha_maximal_onset",
    "ha_abrupt_onset", "ha_gradual_onset", "ha_nausea",
    "exam_papilledema", "exam_meningismus", "ctas", "any_prev_dx",
]

# Human-readable predictor labels
PREDICTOR_LABELS = {
    "ha_vomiting": "Vomiting",
    "exam_altered_loc": "Altered LOC (exam)",
    "ems": "EMS transport",
    "exam_papilledema": "Papilledema",
    "age_under_12": "Age < 12 years",
    "ha_progressive_days": "Progressive headache (days)",
    "sx_focal_weakness": "Focal weakness",
    "sx_ataxia": "Ataxia",
    "ha_nausea": "Nausea",
    "ha_photophobia": "Photophobia",
    "ha_abrupt_onset": "Abrupt onset",
    "ha_maximal_onset": "Maximal at onset",
    "exam_meningismus": "Meningismus",
    "any_prev_dx": "Any prior neurological diagnosis",
    "sx_visual": "Visual impairment",
    "gender": "Male sex",
    "sx_fever": "Fever",
    "ctas": "CTAS score",
    "ha_phonophobia": "Phonophobia",
    "sx_speech": "Speech impairment",
    "ha_gradual_onset": "Gradual onset",
}


def load_data():
    """Load and subset the headache cohort."""
    df = pd.read_parquet(DATA / "brain_attacks_combined.parquet")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    ha = df[df["sx_any_headache"] == 1].copy()
    prior_ha = (
        (ha.get("previous_diagnoses___4", pd.Series(0, index=ha.index)) == 1) |
        (ha.get("previous_diagnoses___5", pd.Series(0, index=ha.index)) == 1)
    )
    ha_clean = ha[~prior_ha].copy()

    y1 = ha_clean[ha_clean["year"] == 1]
    y2 = ha_clean[ha_clean["year"] == 2]

    n_ha_total = (df["sx_any_headache"] == 1).sum()
    n_excluded = prior_ha.sum()

    print(f"Full cohort: {len(df)}")
    print(f"Headache patients (total): {n_ha_total}")
    print(f"Excluded prior HA disorder: {n_excluded}")
    print(f"Analysis cohort: {len(ha_clean)} (MI: {(ha_clean[TARGET]==1).sum()})")
    print(f"Year 1: {len(y1)} (MI: {(y1[TARGET]==1).sum()})")
    print(f"Year 2: {len(y2)} (MI: {(y2[TARGET]==1).sum()})")

    return df, ha_clean, y1, y2, n_ha_total, n_excluded


# ===================================================================
# FIGURE 1: PATIENT FLOW DIAGRAM
# ===================================================================

def make_flow_diagram(n_full, n_ha_total, n_excluded, ha, y1, y2):
    """Create a CONSORT-style patient flow diagram."""
    print("\n=== Figure 1: Patient Flow Diagram ===")

    n_analysis = len(ha)
    n_mi = (ha[TARGET] == 1).sum()
    n_benign = n_analysis - n_mi

    n_y1 = len(y1)
    n_y1_mi = (y1[TARGET] == 1).sum()
    n_y2 = len(y2)
    n_y2_mi = (y2[TARGET] == 1).sum()

    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    # Style
    box_fc = "#f0f4f8"
    box_ec = "#2c3e50"
    excl_fc = "#fef3e2"
    excl_ec = "#e67e22"
    split_fc = "#e8f5e9"
    split_ec = "#27ae60"
    font_main = 10
    font_small = 9

    def draw_box(x, y, w, h, text, fc=box_fc, ec=box_ec, fs=font_main, bold=False):
        rect = mpatches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.15", facecolor=fc, edgecolor=ec, linewidth=1.5
        )
        ax.add_patch(rect)
        weight = "bold" if bold else "normal"
        ax.text(x, y, text, ha="center", va="center", fontsize=fs,
                fontweight=weight, wrap=True,
                bbox=dict(facecolor="none", edgecolor="none"))

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="-|>", color=box_ec, lw=1.5))

    def arrow_right(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="-|>", color=excl_ec, lw=1.5))

    # Box 1: Full cohort
    draw_box(5, 11, 5.5, 1.0,
             f"Brain AttACKs Cohort\nN = {n_full:,}",
             bold=True)

    # Arrow down
    arrow(5, 10.5, 5, 9.8)

    # Box 2: Headache subset
    draw_box(5, 9.3, 5.5, 1.0,
             f"Headache as presenting symptom\nn = {n_ha_total:,} ({100*n_ha_total/n_full:.1f}%)")

    # Arrow right to exclusion
    arrow_right(7.75, 9.3, 8.8, 9.3)

    # Exclusion box
    draw_box(9.2, 9.3, 1.5, 0.8,
             f"Excluded:\nPrior headache\ndisorder\nn = {n_excluded}",
             fc=excl_fc, ec=excl_ec, fs=font_small)

    # Arrow down
    arrow(5, 8.8, 5, 8.1)

    # Box 3: Analysis cohort
    draw_box(5, 7.5, 5.5, 1.2,
             f"Analysis Cohort\nN = {n_analysis:,}\nMI: {n_mi} ({100*n_mi/n_analysis:.1f}%) | "
             f"Benign: {n_benign} ({100*n_benign/n_analysis:.1f}%)",
             bold=True)

    # Arrow splits to two
    arrow(3.5, 6.9, 3.0, 6.2)
    arrow(6.5, 6.9, 7.0, 6.2)

    # Temporal split label
    ax.text(5, 6.5, "Temporal Split", ha="center", va="center",
            fontsize=9, fontstyle="italic", color="#7f8c8d")

    # Box 4a: Year 1 (Derivation)
    draw_box(3.0, 5.5, 3.5, 1.2,
             f"Derivation (Year 1)\nn = {n_y1}\nMI: {n_y1_mi} ({100*n_y1_mi/n_y1:.1f}%)\n"
             f"EPV: {n_y1_mi/len(EXTENDED_PREDICTORS):.1f}",
             fc=split_fc, ec=split_ec)

    # Box 4b: Year 2 (Validation)
    draw_box(7.0, 5.5, 3.5, 1.2,
             f"Validation (Year 2)\nn = {n_y2}\nMI: {n_y2_mi} ({100*n_y2_mi/n_y2:.1f}%)",
             fc=split_fc, ec=split_ec)

    # Arrow down from Year 1
    arrow(3.0, 4.9, 3.0, 4.2)

    # Box 5a: Model development
    draw_box(3.0, 3.6, 3.5, 1.0,
             "Model Development\nL2 Logistic Regression\nXGBoost (3-fold CV)\nSMOTE oversampling",
             fs=font_small)

    # Arrow from model to validation
    ax.annotate("", xy=(7.0, 4.2), xytext=(4.75, 3.6),
                arrowprops=dict(arrowstyle="-|>", color="#27ae60", lw=1.5,
                                connectionstyle="arc3,rad=-0.2"))
    ax.text(5.9, 4.1, "Apply\nmodel", ha="center", va="center",
            fontsize=8, fontstyle="italic", color="#27ae60")

    # Arrow down from Year 2
    arrow(7.0, 4.9, 7.0, 4.2)

    # Box 5b: Temporal validation
    draw_box(7.0, 3.6, 3.5, 1.0,
             "Temporal Validation\nAUROC, Calibration\nDCA, Cost Analysis\nACR Concordance",
             fs=font_small)

    # Arrow down to outputs
    arrow(3.0, 3.1, 3.0, 2.4)
    arrow(7.0, 3.1, 7.0, 2.4)

    # Output boxes
    draw_box(3.0, 1.8, 3.5, 1.0,
             "LR AUROC: 0.928\n(95% CI: 0.873\u20130.978)",
             fc="#e3f2fd", ec="#1565c0", fs=font_small)

    draw_box(7.0, 1.8, 3.5, 1.0,
             "LR AUROC: 0.813\n(95% CI: 0.706\u20130.898)",
             fc="#e3f2fd", ec="#1565c0", fs=font_small)

    fig.tight_layout()
    outpath = MS_OUT / "fig_ha_flow.png"
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved {outpath}")


# ===================================================================
# FIGURE 6: FOREST PLOT OF PREDICTOR aORs
# ===================================================================

def make_forest_plot():
    """Horizontal forest plot of the 21 predictor adjusted odds ratios."""
    print("\n=== Figure 6: Forest Plot of Predictor aORs ===")

    # Read coefficient data from existing table
    tbl = pd.read_csv(TABLES / "table_ha_model_extended.csv")
    coef_rows = tbl[tbl["Model"].str.startswith("LR coef:")].copy()
    coef_rows["Variable"] = coef_rows["Model"].str.replace("LR coef: ", "", regex=False)
    coef_rows["Beta"] = coef_rows["AUROC"].astype(float)
    coef_rows["aOR"] = coef_rows["AUROC 95% CI"].str.replace("aOR=", "", regex=False).astype(float)

    # Sort by aOR descending (risk factors at top)
    coef_rows = coef_rows.sort_values("aOR", ascending=True).reset_index(drop=True)

    # Create human-readable labels
    coef_rows["Label"] = coef_rows["Variable"].map(PREDICTOR_LABELS).fillna(coef_rows["Variable"])

    n = len(coef_rows)
    y_pos = np.arange(n)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Color code: risk (aOR > 1) vs protective (aOR < 1)
    colors = ["#c0392b" if aor > 1.0 else "#2980b9" for aor in coef_rows["aOR"]]

    # Plot points
    ax.barh(y_pos, coef_rows["aOR"] - 1.0, left=1.0, height=0.6,
            color=colors, alpha=0.7, edgecolor="none")
    ax.scatter(coef_rows["aOR"], y_pos, c=colors, s=60, zorder=5, edgecolors="white", linewidths=0.5)

    # Reference line at aOR = 1
    ax.axvline(x=1.0, color="black", linewidth=1, linestyle="-", alpha=0.8)

    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(coef_rows["Label"], fontsize=9)
    ax.set_xlabel("Adjusted Odds Ratio (aOR)", fontsize=11)
    ax.set_title("Logistic Regression Predictor Adjusted Odds Ratios\n(21 Extended Predictors)",
                 fontsize=12, fontweight="bold")

    # Add aOR values on right
    for i, (aor, label) in enumerate(zip(coef_rows["aOR"], coef_rows["Label"])):
        ax.text(max(coef_rows["aOR"]) + 0.08, i, f"{aor:.2f}",
                va="center", ha="left", fontsize=8, color="#333333")

    # Add column header
    ax.text(max(coef_rows["aOR"]) + 0.08, n + 0.3, "aOR",
            va="center", ha="left", fontsize=9, fontweight="bold", color="#333333")

    # Shade risk vs protective regions
    ax.axvspan(0, 1.0, alpha=0.04, color="#2980b9")
    ax.axvspan(1.0, ax.get_xlim()[1], alpha=0.04, color="#c0392b")
    ax.text(0.55, n + 0.3, "Protective", fontsize=8, ha="center",
            color="#2980b9", fontstyle="italic")
    ax.text(1.55, n + 0.3, "Risk factor", fontsize=8, ha="center",
            color="#c0392b", fontstyle="italic")

    ax.set_xlim(0.2, max(coef_rows["aOR"]) + 0.35)
    ax.grid(axis="x", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    outpath = MS_OUT / "fig_ha_forest_plot.png"
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved {outpath}")


# ===================================================================
# COPY EXISTING FIGURES
# ===================================================================

def copy_existing_figures():
    """Copy existing analysis figures to the manuscripts output folder."""
    print("\n=== Copying Existing Figures ===")

    figures_to_copy = [
        "fig_ha_roc_extended.png",
        "fig_ha_calibration.png",
        "fig_ha_decision_curve.png",
        "fig_ha_cost_threshold.png",
        "fig_ha_appropriateness.png",
    ]

    for fname in figures_to_copy:
        src = FIGURES / fname
        dst = MS_OUT / fname
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  Copied {fname}")
        else:
            print(f"  WARNING: {fname} not found")


# ===================================================================
# eTABLE 1: ACR APPROPRIATENESS BY STUDY YEAR
# ===================================================================

def etable1_appropriateness_by_year(ha, y1, y2):
    """ACR appropriateness breakdown by study year."""
    print("\n=== eTable 1: ACR Appropriateness by Study Year ===")

    if "ni_appropriateness" not in ha.columns:
        print("  WARNING: ni_appropriateness not available")
        return

    app_labels = {
        1: "Usually NOT appropriate",
        2: "May be appropriate",
        3: "Usually appropriate",
    }

    rows = []
    for year_label, year_data in [("Year 1", y1), ("Year 2", y2), ("Overall", ha)]:
        subset = year_data[year_data["ni_appropriateness"].notna()].copy()
        subset["ni_appropriateness"] = subset["ni_appropriateness"].astype(int)
        n_rated = len(subset)

        for code in [1, 2, 3]:
            cat = subset[subset["ni_appropriateness"] == code]
            n = len(cat)
            if n == 0:
                continue

            n_imaged = (cat["any_neuroimaging"] == 1).sum() if "any_neuroimaging" in cat.columns else 0
            n_abnormal = (cat["outcome_abnormal_imaging"] == 1).sum() if "outcome_abnormal_imaging" in cat.columns else 0
            n_mi = (cat[TARGET] == 1).sum()

            rows.append({
                "Year": year_label,
                "ACR Category": app_labels[code],
                "N": n,
                "% of rated": f"{100*n/n_rated:.1f}%",
                "Imaged, n (%)": f"{n_imaged} ({100*n_imaged/n:.1f}%)" if n > 0 else "0",
                "Abnormal imaging, n (%)": f"{n_abnormal} ({100*n_abnormal/n:.1f}%)" if n > 0 else "0",
                "MI, n (%)": f"{n_mi} ({100*n_mi/n:.1f}%)" if n > 0 else "0",
            })

        # Total row for this year
        n_imaged_total = (subset["any_neuroimaging"] == 1).sum() if "any_neuroimaging" in subset.columns else 0
        n_mi_total = (subset[TARGET] == 1).sum()
        rows.append({
            "Year": year_label,
            "ACR Category": "Total rated",
            "N": n_rated,
            "% of rated": "100%",
            "Imaged, n (%)": f"{n_imaged_total} ({100*n_imaged_total/n_rated:.1f}%)",
            "Abnormal imaging, n (%)": "",
            "MI, n (%)": f"{n_mi_total} ({100*n_mi_total/n_rated:.1f}%)",
        })

    tbl = pd.DataFrame(rows)
    outpath = MS_OUT / "etable1_appropriateness_by_year.csv"
    tbl.to_csv(outpath, index=False)
    print(f"  Saved {outpath} ({len(tbl)} rows)")

    # Also save to project tables folder
    tbl.to_csv(TABLES / "etable1_appropriateness_by_year.csv", index=False)

    return tbl


# ===================================================================
# eTABLE 2: FULL MODEL CALIBRATION METRICS
# ===================================================================

def calibration_metrics(y_true, y_prob, n_bins=10):
    """Compute calibration metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    brier = brier_score_loss(y_true, y_prob)
    prevalence = y_true.mean()
    brier_max = prevalence * (1 - prevalence)
    brier_scaled = 1 - brier / brier_max if brier_max > 0 else 0

    expected = y_prob.sum()
    observed = y_true.sum()
    eo_ratio = expected / observed if observed > 0 else np.inf
    citl = np.log(observed / expected) if expected > 0 else np.inf

    # Calibration slope
    y_prob_clipped = np.clip(y_prob, 1e-6, 1 - 1e-6)
    lp = np.log(y_prob_clipped / (1 - y_prob_clipped))
    try:
        lr_cal = LogisticRegression(max_iter=1000, penalty=None, solver="lbfgs")
        lr_cal.fit(lp.reshape(-1, 1), y_true)
        slope = lr_cal.coef_[0][0]
    except Exception:
        slope = np.nan

    return {
        "brier": brier,
        "brier_scaled": brier_scaled,
        "citl": citl,
        "slope": slope,
        "eo_ratio": eo_ratio,
        "prevalence": prevalence,
    }


def bootstrap_ci(y_true, y_prob, n_boot=1000, seed=SEED):
    """Bootstrap 95% CIs for AUROC."""
    rng = np.random.RandomState(seed)
    aurocs = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            aurocs.append(roc_auc_score(y_true[idx], y_prob[idx]))
        except ValueError:
            continue
    arr = np.array(aurocs)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def etable2_calibration(y1, y2):
    """Generate full calibration metrics for LR and XGBoost."""
    print("\n=== eTable 2: Full Model Calibration Metrics ===")

    features = [f for f in EXTENDED_PREDICTORS if f in y1.columns]

    X1 = y1[features].fillna(0).values
    X2 = y2[features].fillna(0).values
    y1_out = y1[TARGET].values
    y2_out = y2[TARGET].values

    # --- LR ---
    scaler = StandardScaler()
    X1_scaled = scaler.fit_transform(X1)
    X2_scaled = scaler.transform(X2)

    lr = LogisticRegression(max_iter=1000, random_state=SEED, penalty="l2", C=1.0)
    lr.fit(X1_scaled, y1_out)

    lr_deriv_prob = lr.predict_proba(X1_scaled)[:, 1]
    lr_valid_prob = lr.predict_proba(X2_scaled)[:, 1]

    # --- XGBoost ---
    from sklearn.metrics import fbeta_score, make_scorer
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
    f2_scorer = make_scorer(fbeta_score, beta=2, zero_division=0)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                         random_state=SEED, verbosity=0)
    grid = GridSearchCV(xgb, param_grid, scoring=f2_scorer,
                         cv=inner_cv, n_jobs=-1, refit=True)
    grid.fit(X1_sm, y1_sm)

    xgb_deriv_prob = grid.predict_proba(X1)[:, 1]
    xgb_valid_prob = grid.predict_proba(X2)[:, 1]

    # Compute metrics for all 4 model-dataset combos
    rows = []
    for label, y_true, y_prob in [
        ("LR Derivation (Year 1)", y1_out, lr_deriv_prob),
        ("LR Validation (Year 2)", y2_out, lr_valid_prob),
        ("XGBoost Derivation (Year 1)", y1_out, xgb_deriv_prob),
        ("XGBoost Validation (Year 2)", y2_out, xgb_valid_prob),
    ]:
        auroc = roc_auc_score(y_true, y_prob)
        ci_lo, ci_hi = bootstrap_ci(y_true, y_prob)
        cal = calibration_metrics(y_true, y_prob)

        rows.append({
            "Model": label,
            "N": len(y_true),
            "Events": int(y_true.sum()),
            "Prevalence": f"{100*cal['prevalence']:.1f}%",
            "AUROC (95% CI)": f"{auroc:.3f} ({ci_lo:.3f}\u2013{ci_hi:.3f})",
            "Brier score": f"{cal['brier']:.4f}",
            "Brier scaled": f"{cal['brier_scaled']:.3f}",
            "CITL": f"{cal['citl']:.3f}",
            "Calibration slope": f"{cal['slope']:.3f}" if not np.isnan(cal['slope']) else "N/A",
            "E:O ratio": f"{cal['eo_ratio']:.3f}",
        })

        print(f"  {label}: AUROC={auroc:.3f}, Brier={cal['brier']:.4f}, "
              f"CITL={cal['citl']:.3f}, Slope={cal['slope']:.3f}")

    tbl = pd.DataFrame(rows)
    outpath = MS_OUT / "etable2_calibration_metrics.csv"
    tbl.to_csv(outpath, index=False)
    print(f"  Saved {outpath}")

    tbl.to_csv(TABLES / "etable2_calibration_metrics.csv", index=False)

    return tbl


# ===================================================================
# MAIN
# ===================================================================

def main():
    print("=" * 60)
    print("Headache Manuscript: Figure & Supplementary Generation")
    print("=" * 60)

    df, ha, y1, y2, n_ha_total, n_excluded = load_data()

    # Figure 1: Flow diagram
    make_flow_diagram(len(df), n_ha_total, n_excluded, ha, y1, y2)

    # Figure 6: Forest plot
    make_forest_plot()

    # Copy existing figures
    copy_existing_figures()

    # eTable 1: Appropriateness by year
    etable1_appropriateness_by_year(ha, y1, y2)

    # eTable 2: Calibration metrics
    etable2_calibration(y1, y2)

    print(f"\n{'='*60}")
    print(f"All outputs saved to: {MS_OUT}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
