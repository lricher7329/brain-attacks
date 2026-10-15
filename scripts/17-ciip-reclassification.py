"""
17-ciip-reclassification.py — CIIP Outcome Reclassification

Reclassifies the primary outcome from "Medically Important Brain Attack (MI)"
to "Clinically Important Intracranial Pathology (CIIP)". Cerebellar ataxia
cases (med_imp_dx___13) that lack imaging-confirmed intracranial pathology
AND qualifying therapy are reclassified as non-CIIP. IIH (code 12) remains CIIP.

Reclassification logic:
    FOR each case with med_imp_dx___13 == 1 (cerebellar ataxia):
        IF any other med_imp_dx___1 through ___12 == 1:
            -> CIIP (other MI code qualifies independently)
        ELIF results == 2 (abnormal imaging) AND inv_specific_tx == 1:
            -> CIIP (imaging-confirmed pathology + qualifying therapy)
        ELSE:
            -> non-CIIP (reclassify)

Inputs:
    data/processed/brain_attacks_combined.parquet

Outputs:
    data/processed/brain_attacks_ciip.parquet
    data/processed/brain_attacks_ciip.csv
    outputs/tables/table_ciip_ca_verification.csv
    outputs/tables/table_ciip_reclassification_summary.csv
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data" / "processed"
TABLES = PROJECT / "outputs" / "tables"
DATA.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)


def _print(msg=""):
    print(msg, flush=True)


# ===================================================================
# 1. LOAD ORIGINAL DATA
# ===================================================================

def load_original_data():
    """Load brain_attacks_combined.parquet (the MI dataset)."""
    df = pd.read_parquet(DATA / "brain_attacks_combined.parquet")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    n_mi = (df["outcome_med_important"] == 1).sum()
    _print(f"Loaded: {len(df)} records")
    _print(f"  MI: {n_mi} ({100 * n_mi / len(df):.1f}%)")
    _print(f"  Benign: {len(df) - n_mi}")
    return df


# ===================================================================
# 2. IDENTIFY CEREBELLAR ATAXIA CASES
# ===================================================================

def identify_ca_cases(df):
    """Find all cerebellar ataxia cases (med_imp_dx___13 == 1).

    Returns
    -------
    ca_idx : Index — all CA cases
    ca_only_idx : Index — CA cases with NO other MI code
    ca_plus_idx : Index — CA cases WITH at least one other MI code
    """
    ca_col = "med_imp_dx___13"
    if ca_col not in df.columns:
        _print("  WARNING: med_imp_dx___13 not found in dataset")
        return pd.Index([]), pd.Index([]), pd.Index([])

    ca_mask = df[ca_col].fillna(0) == 1
    ca_idx = df.index[ca_mask]

    # Check for other MI codes (1-12)
    other_mi_cols = [f"med_imp_dx___{i}" for i in range(1, 13)
                     if f"med_imp_dx___{i}" in df.columns]
    other_mi = df.loc[ca_idx, other_mi_cols].fillna(0).max(axis=1)

    ca_only_idx = other_mi[other_mi == 0].index
    ca_plus_idx = other_mi[other_mi == 1].index

    _print(f"\n=== Cerebellar Ataxia Cases ===")
    _print(f"  Total CA (med_imp_dx___13 == 1): {len(ca_idx)}")
    _print(f"  CA only (no other MI code): {len(ca_only_idx)}")
    _print(f"  CA + other MI code: {len(ca_plus_idx)}")

    return ca_idx, ca_only_idx, ca_plus_idx


# ===================================================================
# 3. BUILD VERIFICATION TABLE
# ===================================================================

def build_verification_table(df, ca_idx):
    """Build detailed verification table for PI review of all CA cases.

    Includes clinical data needed to verify reclassification decisions.
    """
    _print(f"\n=== Building Verification Table ({len(ca_idx)} CA cases) ===")

    mi_dx_cols = [f"med_imp_dx___{i}" for i in range(1, 14)
                  if f"med_imp_dx___{i}" in df.columns]
    mi_labels = {
        1: "Stroke", 2: "ICH", 3: "Vascular malformation",
        4: "Vasculitis", 5: "Venous thrombosis", 6: "Demyelinating",
        7: "Infection", 8: "Hydrocephalus", 9: "Neoplasm",
        10: "Inborn error", 11: "Other MI", 12: "IIH", 13: "CA",
    }

    rows = []
    for idx in ca_idx:
        row = df.loc[idx]

        # Determine other MI codes present
        other_codes = []
        for i in range(1, 13):
            col = f"med_imp_dx___{i}"
            if col in df.columns and row.get(col, 0) == 1:
                other_codes.append(f"{i}={mi_labels.get(i, f'Code{i}')}")
        has_other_mi = len(other_codes) > 0

        # Imaging results
        results_val = row.get("results", np.nan)
        abnormal_imaging = (results_val == 2) if pd.notna(results_val) else False

        # Specific treatment
        inv_specific_tx = row.get("inv_specific_tx", 0)
        has_specific_tx = (inv_specific_tx == 1)

        # Apply reclassification rule
        if has_other_mi:
            proposed_status = "CIIP"
            reason = f"Other MI code qualifies: {'; '.join(other_codes)}"
        elif abnormal_imaging and has_specific_tx:
            proposed_status = "CIIP"
            reason = "Abnormal imaging + specific treatment"
        else:
            proposed_status = "non-CIIP"
            parts = []
            if not abnormal_imaging:
                parts.append(f"imaging result={results_val}")
            if not has_specific_tx:
                parts.append("no specific tx")
            reason = "CA only: " + ", ".join(parts)

        rows.append({
            "record_id": row.get("record_id", idx),
            "year": row.get("year", ""),
            "age": row.get("age", ""),
            "gender": row.get("gender", ""),
            "diagnosis_specify": row.get("diagnosis_specify", ""),
            "med_benign_specify": row.get("med_benign_specify", ""),
            # Imaging
            "any_neuroimaging": row.get("any_neuroimaging", 0),
            "inv_ct": row.get("inv_ct", 0),
            "inv_mri": row.get("inv_mri", 0),
            "results": results_val,
            "results_abnormal": int(abnormal_imaging),
            # Treatment
            "inv_specific_tx": inv_specific_tx,
            # Course
            "admit": row.get("admit", 0),
            "icu_any": row.get("icu_any", 0),
            "consult": row.get("consult", 0),
            "consult_specify___4": row.get("consult_specify___4", 0),  # Neurosurgery
            # MI codes
            "has_other_mi_dx": int(has_other_mi),
            "other_mi_codes": "; ".join(other_codes) if other_codes else "",
            # Reclassification
            "proposed_ciip_status": proposed_status,
            "reclassification_reason": reason,
            # PI override columns (empty for PI to fill in)
            "pi_override": "",
            "pi_notes": "",
        })

    tbl = pd.DataFrame(rows)
    tbl.to_csv(TABLES / "table_ciip_ca_verification.csv", index=False)
    _print(f"  Saved table_ciip_ca_verification.csv ({len(tbl)} rows)")

    # Summary
    n_ciip = (tbl["proposed_ciip_status"] == "CIIP").sum()
    n_non_ciip = (tbl["proposed_ciip_status"] == "non-CIIP").sum()
    _print(f"  Proposed CIIP: {n_ciip}")
    _print(f"  Proposed non-CIIP: {n_non_ciip}")

    return tbl


# ===================================================================
# 4. APPLY RECLASSIFICATION
# ===================================================================

def apply_reclassification(df, ca_idx, ca_only_idx):
    """Apply CIIP reclassification rule.

    Creates outcome_ciip column alongside preserved outcome_med_important.

    Logic:
        - CA + other MI code -> CIIP (other code qualifies)
        - CA + abnormal imaging + specific treatment -> CIIP
        - All other CA-only -> non-CIIP
        - All non-CA MI codes -> CIIP (unchanged)
        - All benign -> non-CIIP (unchanged)
    """
    _print(f"\n=== Applying Reclassification ===")

    # Start with MI outcome as baseline for CIIP
    df["outcome_ciip"] = df["outcome_med_important"].copy()

    # For CA-only cases, check imaging + treatment criteria
    reclassified_to_non_ciip = []

    for idx in ca_only_idx:
        row = df.loc[idx]
        results_val = row.get("results", np.nan)
        abnormal_imaging = (results_val == 2) if pd.notna(results_val) else False
        has_specific_tx = (row.get("inv_specific_tx", 0) == 1)

        if abnormal_imaging and has_specific_tx:
            # Keep as CIIP — imaging-confirmed + treated
            pass
        else:
            # Reclassify to non-CIIP
            df.loc[idx, "outcome_ciip"] = 0
            reclassified_to_non_ciip.append(idx)

    n_reclassified = len(reclassified_to_non_ciip)
    n_mi_original = (df["outcome_med_important"] == 1).sum()
    n_ciip = (df["outcome_ciip"] == 1).sum()

    _print(f"  Original MI count: {n_mi_original}")
    _print(f"  CA cases reclassified to non-CIIP: {n_reclassified}")
    _print(f"  New CIIP count: {n_ciip}")
    _print(f"  CIIP prevalence: {100 * n_ciip / len(df):.1f}%")
    _print(f"  MI prevalence was: {100 * n_mi_original / len(df):.1f}%")

    # Verify preservation
    assert (df["outcome_med_important"] == 1).sum() == n_mi_original, \
        "outcome_med_important was modified — this should not happen"

    return df, reclassified_to_non_ciip


# ===================================================================
# 5. SAVE CIIP DATASET
# ===================================================================

def save_ciip_dataset(df):
    """Export CIIP dataset as parquet and CSV."""
    _print(f"\n=== Saving CIIP Dataset ===")

    df.to_parquet(DATA / "brain_attacks_ciip.parquet", index=False)
    df.to_csv(DATA / "brain_attacks_ciip.csv", index=False)

    _print(f"  Parquet: {DATA / 'brain_attacks_ciip.parquet'}")
    _print(f"  CSV: {DATA / 'brain_attacks_ciip.csv'}")
    _print(f"  Shape: {df.shape}")


# ===================================================================
# 6. GENERATE SUMMARY
# ===================================================================

def generate_summary(df, ca_idx, ca_only_idx, ca_plus_idx, reclassified_idx):
    """Generate reclassification summary table."""
    _print(f"\n=== Reclassification Summary ===")

    n_total = len(df)
    n_mi = (df["outcome_med_important"] == 1).sum()
    n_ciip = (df["outcome_ciip"] == 1).sum()
    n_reclassified = len(reclassified_idx)

    mi_labels = {
        1: "Stroke", 2: "ICH", 3: "Vascular malformation",
        4: "Vasculitis", 5: "Venous thrombosis", 6: "Demyelinating",
        7: "Infection", 8: "Hydrocephalus", 9: "Neoplasm",
        10: "Inborn error", 11: "Other MI", 12: "IIH", 13: "CA",
    }

    rows = []

    # Overall counts
    rows.append({"Metric": "Total cohort N", "Value": n_total})
    rows.append({"Metric": "Original MI count", "Value": n_mi})
    rows.append({"Metric": "Original MI prevalence", "Value": f"{100 * n_mi / n_total:.1f}%"})
    rows.append({"Metric": "New CIIP count", "Value": n_ciip})
    rows.append({"Metric": "New CIIP prevalence", "Value": f"{100 * n_ciip / n_total:.1f}%"})
    rows.append({"Metric": "Net change (MI - CIIP)", "Value": n_mi - n_ciip})

    # CA breakdown
    rows.append({"Metric": "---", "Value": "---"})
    rows.append({"Metric": "Total CA cases", "Value": len(ca_idx)})
    rows.append({"Metric": "CA + other MI code (kept CIIP)", "Value": len(ca_plus_idx)})
    rows.append({"Metric": "CA only (evaluated)", "Value": len(ca_only_idx)})
    ca_kept = len(ca_only_idx) - n_reclassified
    rows.append({"Metric": "CA only -> CIIP (imaging+tx)", "Value": ca_kept})
    rows.append({"Metric": "CA only -> non-CIIP (reclassified)", "Value": n_reclassified})

    # Per-year breakdown
    rows.append({"Metric": "---", "Value": "---"})
    for yr in [1, 2]:
        yr_df = df[df["year"] == yr]
        n_yr = len(yr_df)
        mi_yr = (yr_df["outcome_med_important"] == 1).sum()
        ciip_yr = (yr_df["outcome_ciip"] == 1).sum()
        rows.append({"Metric": f"Year {yr} N", "Value": n_yr})
        rows.append({"Metric": f"Year {yr} MI", "Value": mi_yr})
        rows.append({"Metric": f"Year {yr} CIIP", "Value": ciip_yr})
        rows.append({"Metric": f"Year {yr} reclassified", "Value": mi_yr - ciip_yr})

    # CIIP diagnosis breakdown
    rows.append({"Metric": "---", "Value": "---"})
    rows.append({"Metric": "CIIP diagnosis breakdown:", "Value": ""})
    ciip_cases = df[df["outcome_ciip"] == 1]
    for code, label in sorted(mi_labels.items()):
        col = f"med_imp_dx___{code}"
        if col in df.columns:
            n = (ciip_cases[col] == 1).sum()
            if n > 0:
                rows.append({"Metric": f"  {label} (code {code})", "Value": n})

    tbl = pd.DataFrame(rows)
    tbl.to_csv(TABLES / "table_ciip_reclassification_summary.csv", index=False)
    _print(f"  Saved table_ciip_reclassification_summary.csv ({len(tbl)} rows)")

    return tbl


# ===================================================================
# MAIN
# ===================================================================

def main():
    _print("=" * 60)
    _print("Brain AttACKs: CIIP Reclassification (MI -> CIIP)")
    _print("=" * 60)

    # 1. Load
    df = load_original_data()

    # 2. Identify CA cases
    ca_idx, ca_only_idx, ca_plus_idx = identify_ca_cases(df)

    # 3. Verification table for PI review
    build_verification_table(df, ca_idx)

    # 4. Apply reclassification
    df, reclassified_idx = apply_reclassification(df, ca_idx, ca_only_idx)

    # 5. Save CIIP dataset
    save_ciip_dataset(df)

    # 6. Summary
    generate_summary(df, ca_idx, ca_only_idx, ca_plus_idx, reclassified_idx)

    _print(f"\n{'=' * 60}")
    _print("Done. Review table_ciip_ca_verification.csv before proceeding.")
    _print(f"{'=' * 60}")

    return df


if __name__ == "__main__":
    df = main()
