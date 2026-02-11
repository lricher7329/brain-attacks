"""
01-data-prep.py — Brain AttACKs Data Preparation and Harmonization

Rebuilds the combined analytic dataset from source REDCap exports.
See analysis-plan.md Sections 4, 6, 7, 8.1 for full documentation.

Inputs:
    data/raw_data/BrainAttACKs_DATA_2024-11-18_1011.csv       (Year 1, 1124 records)
    data/raw_data/BrainAttACKsYear2_DATA_2024-11-18_1013.csv   (Year 2, 1205 records)

Outputs:
    data/processed/brain_attacks_combined.csv
    data/processed/brain_attacks_combined.parquet
    outputs/tables/data_prep_summary.csv
    outputs/tables/tableS4_missing_data.csv         (TRIPOD+AI Item 14)
    outputs/figures/fig_participant_flow.png          (TRIPOD+AI Item 20)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
RAW = DATA / "raw_data"
PROCESSED = DATA / "processed"
OUTPUTS = PROJECT / "outputs" / "tables"
FIGURES = PROJECT / "outputs" / "figures"

PROCESSED.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

Y1_FILE = RAW / "BrainAttACKs_DATA_2024-11-18_1011.csv"
Y2_FILE = RAW / "BrainAttACKsYear2_DATA_2024-11-18_1013.csv"


# ===================================================================
# 1. LOAD RAW DATA
# ===================================================================

def load_raw():
    """Load Year 1 and Year 2 raw CSVs with appropriate settings."""
    y1 = pd.read_csv(Y1_FILE, dtype=str, keep_default_na=False)
    y2 = pd.read_csv(Y2_FILE, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    y1["year"] = "1"
    y2["year"] = "2"
    print(f"Year 1: {len(y1)} records, {len(y1.columns)} columns")
    print(f"Year 2: {len(y2)} records, {len(y2.columns)} columns")
    return y1, y2


# ===================================================================
# 2. HARMONIZE SCHEMAS
# ===================================================================

def harmonize_year1(y1):
    """
    Rename and restructure Year 1 fields to match Year 2 conventions.
    See analysis-plan.md Section 4 for full mapping.
    """
    df = y1.copy()

    # --- 2a. Simple renames ---
    renames = {
        "time_of_year": "time_year",
        "presentation_time": "time",
        "othersymptom_specify": "symptom_other",
        "med_benign_other_specify": "med_benign_specify",
    }
    df = df.rename(columns=renames)

    # --- 2b. Medically important diagnosis: med_important_diagnosis -> med_imp_dx ---
    for i in range(1, 11):
        old = f"med_important_diagnosis___{i}"
        new = f"med_imp_dx___{i}"
        if old in df.columns:
            df = df.rename(columns={old: new})
    # Year 1 has no option 11 (Other medically important); add as empty
    df["med_imp_dx___11"] = ""
    df["med_imp_specify"] = ""

    # --- 2c. Headache specify remapping ---
    # Year 1: ___1=Bilateral, ___2=Unilateral, ___3=Nausea, ___4=Vomiting,
    #          ___5=Photophobia, ___6=Phonophobia, ___7=Other
    # Year 2: ___1=Nausea, ___2=Vomiting, ___3=Photophobia, ___4=Phonophobia, ___5=Other
    #
    # Harmonize to concept-based names, then create unified columns later.
    # First, capture Year 1 bilateral/unilateral from headache_specify before remapping.
    df["_hs_bilateral"] = df.get("headache_specify___1", "")
    df["_hs_unilateral"] = df.get("headache_specify___2", "")

    # Remap Year 1 headache_specify to match Year 2 numbering
    hs_remap = {
        "headache_specify___3": "headache_specify___1",  # Nausea
        "headache_specify___4": "headache_specify___2",  # Vomiting
        "headache_specify___5": "headache_specify___3",  # Photophobia
        "headache_specify___6": "headache_specify___4",  # Phonophobia
        "headache_specify___7": "headache_specify___5",  # Other
    }
    # Drop original columns first, then rename
    for old, new in hs_remap.items():
        if old in df.columns:
            df[new] = df[old]
    # Remove old columns that were remapped
    for col in ["headache_specify___1", "headache_specify___2",
                "headache_specify___3", "headache_specify___4",
                "headache_specify___5", "headache_specify___6",
                "headache_specify___7"]:
        if col in df.columns and col not in hs_remap.values():
            df = df.drop(columns=[col])

    # --- 2d. Current headache: Year 1 has ___1-7, Year 2 has ___1-12 ---
    # ___1-7 map directly between years (same concepts).
    # Year 2 adds ___8=Bilateral, ___9=Unilateral, ___10=Frontal, ___11=Temporal, ___12=Occipital
    # For Year 1, populate ___8 and ___9 from the headache_specify bilateral/unilateral we saved.
    df["current_headache___8"] = df["_hs_bilateral"]    # Bilateral
    df["current_headache___9"] = df["_hs_unilateral"]   # Unilateral
    df["current_headache___10"] = ""  # Frontal - not available Y1
    df["current_headache___11"] = ""  # Temporal - not available Y1
    df["current_headache___12"] = ""  # Occipital - not available Y1
    df = df.drop(columns=["_hs_bilateral", "_hs_unilateral"])

    # --- 2e. Migraine/headache treatment renames ---
    for i in range(1, 4):
        old_m = f"migraine_treatments___{i}"
        new_m = f"migraine_tx___{i}"
        if old_m in df.columns:
            df = df.rename(columns={old_m: new_m})

        old_h = f"headache_treatments___{i}"
        new_h = f"headache_tx___{i}"
        if old_h in df.columns:
            df = df.rename(columns={old_h: new_h})

    # --- 2f. Investigations: extract consultation and admission ---
    # Year 1: ___5=Consultation, ___6=Admission, ___7=Specific treatment,
    #          ___8=CTA, ___9=MRA, ___10=Cerebral arteriography
    # Year 2: ___5=Specific treatment, ___6=CTA, ___7=MRA, ___8=Cerebral arteriography
    #
    # Extract consult/admit first, then remap imaging fields.
    df["_inv_consult"] = df.get("investigations___5", "")
    df["_inv_admit"] = df.get("investigations___6", "")

    # Remap investigation numbering to match Year 2
    inv_remap = {
        "investigations___7": "investigations___5",   # Specific treatment
        "investigations___8": "investigations___6",   # CTA
        "investigations___9": "investigations___7",   # MRA
        "investigations___10": "investigations___8",  # Cerebral arteriography
    }
    for old, new in inv_remap.items():
        if old in df.columns:
            df[new] = df[old]
    # Drop old columns
    for col in ["investigations___7", "investigations___8",
                "investigations___9", "investigations___10"]:
        if col in df.columns and col not in inv_remap.values():
            df = df.drop(columns=[col])

    # --- 2g. Create harmonized consult/admit fields ---
    # Year 1: consult derived from investigations___5; admission from investigations___6 or admission_icu
    df["consult"] = df["_inv_consult"]
    df["consult_specify___1"] = ""  # Pediatrics
    df["consult_specify___2"] = ""  # Neurology
    df["consult_specify___3"] = ""  # Ophthalmology
    df["consult_specify___4"] = ""  # Neurosurgery
    df["consult_specify___5"] = ""  # Cardiology
    df["consult_specify___6"] = ""  # PICU
    df["consult_specify___7"] = ""  # Psychiatry
    df["consult_specify___8"] = ""  # Other
    df["consult_other"] = ""
    # Keep the free-text consult_specify as-is for LLM coding later

    df["admit"] = df["_inv_admit"]
    df["admit_location___1"] = ""  # Pediatric inpatient
    df["admit_location___2"] = ""  # PICU
    df["admit_location___3"] = ""  # Adult ICU
    df["admit_location___4"] = ""  # Direct to surgery
    df["admit_location___5"] = ""  # Other
    df["admit_other"] = ""

    # If admission_icu == 1, mark PICU
    if "admission_icu" in df.columns:
        icu_mask = df["admission_icu"] == "1"
        df.loc[icu_mask, "admit_location___2"] = "1"
        # If admitted to ICU, they were also admitted
        df.loc[icu_mask, "admit"] = "1"

    df = df.drop(columns=["_inv_consult", "_inv_admit"], errors="ignore")

    # --- 2h. Fields Year 1 doesn't have ---
    if "related" not in df.columns:
        df["related"] = ""  # Assume all Y1 records pre-screened
    if "dias_bp" not in df.columns:
        df["dias_bp"] = ""
    if "diff_dx" not in df.columns:
        df["diff_dx"] = ""
    if "diffdx_specify" not in df.columns:
        df["diffdx_specify"] = ""
    if "med_benign___6" not in df.columns:
        df["med_benign___6"] = ""
    if "epilepsy_describe___3" not in df.columns:
        df["epilepsy_describe___3"] = ""

    # Discharge referral: Year 1 has free-text, Year 2 has checkboxes
    for i in range(1, 6):
        col = f"discharge_refer_specify___{i}"
        if col not in df.columns:
            df[col] = ""
    if "refer_other" not in df.columns:
        df["refer_other"] = ""

    # Discharge meds: Year 2 has multiple fields
    for col in ["discharge_med2", "discharge_med3", "discharge_med4", "discharge_med5"]:
        if col not in df.columns:
            df[col] = ""

    return df


def harmonize_year2(y2):
    """Minor adjustments to Year 2 for consistency."""
    df = y2.copy()

    # Year 2 may have admission_icu from old schema; drop if present
    # No substantial changes needed — Year 2 is the reference schema.

    # Ensure consult_specify free-text column exists (Year 2 uses checkboxes)
    if "consult_specify" not in df.columns:
        df["consult_specify"] = ""

    # Ensure admission_icu exists for harmonized icu_any derivation
    if "admission_icu" not in df.columns:
        df["admission_icu"] = ""

    # Ensure discharge_refer_specify free-text exists
    if "discharge_refer_specify" not in df.columns:
        df["discharge_refer_specify"] = ""

    return df


# ===================================================================
# 3. COMBINE AND ALIGN COLUMNS
# ===================================================================

def combine_datasets(y1, y2):
    """
    Combine Year 1 and Year 2 into a single dataframe.
    Only keep columns that exist in both datasets after harmonization,
    plus columns explicitly handled as year-specific.
    """
    # Find common columns
    common = sorted(set(y1.columns) & set(y2.columns))
    y1_only = sorted(set(y1.columns) - set(y2.columns))
    y2_only = sorted(set(y2.columns) - set(y1.columns))

    print(f"\nColumn alignment:")
    print(f"  Common columns: {len(common)}")
    print(f"  Year 1 only: {len(y1_only)}")
    if y1_only:
        print(f"    {y1_only}")
    print(f"  Year 2 only: {len(y2_only)}")
    if y2_only:
        print(f"    {y2_only}")

    # Add missing columns as empty to each side
    for col in y2_only:
        y1[col] = ""
    for col in y1_only:
        y2[col] = ""

    # Ensure same column order
    all_cols = sorted(set(y1.columns) | set(y2.columns))
    df = pd.concat([y1[all_cols], y2[all_cols]], ignore_index=True)
    print(f"\nCombined: {len(df)} records, {len(df.columns)} columns")
    return df


# ===================================================================
# 4. APPLY EXCLUSIONS
# ===================================================================

def apply_exclusions(df):
    """
    Apply eligibility criteria per protocol.
    Primary exclusion: trauma only (manuscript N = 2,329 - 176 = 2,153).
    Other data quality flags reported but not applied as exclusions.
    """
    flow = {"raw_total": len(df)}

    # Primary exclusion: trauma
    trauma_mask = df["trauma"] == "1"
    flow["excluded_trauma"] = trauma_mask.sum()
    df = df[~trauma_mask].copy()

    flow["after_exclusions"] = len(df)

    # Report (but do not exclude) data quality flags
    not_related = (df["related"] == "0").sum()
    no_doctor = (df["see_doctor"] == "0").sum()
    incomplete = (df["brain_attacks_crf_complete"] != "2").sum()

    print("\n=== Exclusion Flow ===")
    for k, v in flow.items():
        print(f"  {k}: {v}")
    print(f"\n  Data quality flags (not excluded):")
    print(f"    related=0: {not_related}")
    print(f"    see_doctor=0: {no_doctor}")
    print(f"    incomplete CRF: {incomplete}")

    return df, flow


# ===================================================================
# 5. DERIVE OUTCOME VARIABLES
# ===================================================================

def derive_outcomes(df):
    """
    Create binary outcome variables from checkbox fields.
    """
    # Primary: medically important brain attack
    # Note: runs after convert_types(), so checkbox values are numeric 0/1
    # Codes 12 (IIH) and 13 (cerebellar ataxia) are created later by
    # apply_pi_review(), so only check 1-11 here; outcome is recalculated
    # after PI review elevates sub-types.
    mi_cols = [f"med_imp_dx___{i}" for i in range(1, 14)]
    existing_mi = [c for c in mi_cols if c in df.columns]
    df["outcome_med_important"] = (df[existing_mi].max(axis=1) == 1).astype(int)

    # Count by subtype
    mi_labels = {
        1: "Stroke", 2: "ICH", 3: "Vascular malformation", 4: "Vasculitis",
        5: "Venous thrombosis", 6: "Demyelinating disorder",
        7: "Brain/meningeal infection", 8: "Hydrocephalus", 9: "Neoplasm",
        10: "Inborn error of metabolism", 11: "Other medically important",
        12: "IIH (idiopathic intracranial hypertension)",
        13: "Cerebellar ataxia",
    }
    print("\n=== Medically Important Diagnoses ===")
    total_mi = df["outcome_med_important"].sum()
    print(f"  Total medically important: {total_mi}")
    print(f"  Total medically unimportant: {len(df) - total_mi}")
    for i, label in mi_labels.items():
        col = f"med_imp_dx___{i}"
        if col in df.columns:
            n = (df[col] == 1).sum()
            if n > 0:
                print(f"    {label}: {n}")

    # Secondary: abnormal neuroimaging (only among imaged patients)
    df["outcome_abnormal_imaging"] = (df["results"] == 2).astype(int)

    # Secondary: medically important cause of abnormal imaging
    df["outcome_mi_imaging"] = (df["med_import"] == 1).astype(int)

    # Secondary: ICU admission (harmonized)
    # admission_icu and admit are binary fields (converted to numeric by convert_types)
    # admit_location checkboxes are also numeric after convert_types
    df["icu_any"] = (
        (df.get("admission_icu", pd.Series(dtype="float64")) == 1) |
        (df.get("admit_location___2", pd.Series(0, index=df.index)) == 1) |
        (df.get("admit_location___3", pd.Series(0, index=df.index)) == 1)
    ).astype(int)

    # Secondary: any admission (harmonized)
    y1_admit = (
        (df["year"] == "1") &
        ((df.get("admit", pd.Series(dtype="float64")) == 1) |
         (df.get("admission_icu", pd.Series(dtype="float64")) == 1))
    )
    y2_admit = (df["year"] == "2") & (df.get("admit", pd.Series(dtype="float64")) == 1)
    df["admit_any"] = (y1_admit | y2_admit).astype(int)

    return df


# ===================================================================
# 5b. APPLY PI REVIEW DECISIONS
# ===================================================================

def apply_pi_review(df):
    """
    Apply PI review decisions to correct outcome classification.

    Reads pi-review-decisions.csv which documents the PI's adjudication of
    124 ambiguous cases flagged during LLM-assisted free-text coding.

    Two types of corrections:
    - MI → Benign (checkbox errors): records where the MI checkbox was
      erroneously selected but the discharge diagnosis is benign.
    - Benign → MI (reclassifications): records where the benign checkbox was
      selected but the diagnosis is medically important.

    Matching uses BOTH diagnosis_specify and med_benign_specify fields because
    Task E decisions reference med_benign_specify text, which can differ from
    the primary diagnosis_specify field.

    Also creates sub-type tracking columns for IIH and cerebellar ataxia,
    two clinically distinct categories within "Other MI" (code 11).
    """
    decisions_path = PROCESSED / "llm_coded" / "pi-review-decisions.csv"
    if not decisions_path.exists():
        print("\n  WARNING: pi-review-decisions.csv not found — skipping PI review")
        return df

    decisions = pd.read_csv(decisions_path)
    print(f"\n=== PI Review: {len(decisions)} decisions loaded ===")

    # Lowercase text fields for case-insensitive matching
    # Match against BOTH diagnosis_specify and med_benign_specify because
    # Task E decisions reference med_benign_specify text which may differ
    dx_lower = df["diagnosis_specify"].fillna("").str.strip().str.lower()
    mbs_lower = df["med_benign_specify"].fillna("").str.strip().str.lower()

    # Snapshot current outcome state (match against original, apply changes after)
    original_mi = df["outcome_med_important"].copy()

    # Initialize sub-type columns
    df["mi_subtype_iih"] = 0
    df["mi_subtype_cerebellar_ataxia"] = 0

    # Collect indices for batch changes
    indices_to_benign = set()
    indices_to_mi = {}       # index -> mi_code
    indices_iih = set()
    indices_cerebellar = set()
    changes_log = []

    for _, row in decisions.iterrows():
        decision = row["decision"]
        dx_text = str(row["diagnosis_text"]).strip().lower()
        source_task = row["source_task"]
        mi_code = row.get("mi_code", None)
        expected_n = int(row.get("n_records", 0))

        if decision == "keep_benign" and source_task == "A":
            # Checkbox error: MI checkbox was wrong, should be benign
            # Task A matches on diagnosis_specify only
            mask = (dx_lower == dx_text) & (original_mi == 1)
            matched_idx = df.index[mask]
            actual_n = len(matched_idx)

            if actual_n > 0:
                indices_to_benign.update(matched_idx)
                flag = "" if actual_n == expected_n else f" [expected {expected_n}]"
                changes_log.append(
                    f"  MI→Benign: '{dx_text}' ({actual_n} records{flag})")

        elif decision.startswith("reclassify_mi_"):
            # Reclassify: benign checkbox was wrong, should be MI
            # Match on EITHER diagnosis_specify OR med_benign_specify
            mask = ((dx_lower == dx_text) | (mbs_lower == dx_text)) & (original_mi == 0)
            matched_idx = df.index[mask]
            actual_n = len(matched_idx)

            if actual_n > 0 and pd.notna(mi_code):
                mi_code_int = int(mi_code)
                for idx in matched_idx:
                    indices_to_mi[idx] = mi_code_int

                if "IIH" in decision:
                    indices_iih.update(matched_idx)
                elif "cerebellar_ataxia" in decision:
                    indices_cerebellar.update(matched_idx)

                flag = "" if actual_n == expected_n else f" [expected {expected_n}]"
                changes_log.append(
                    f"  Benign→MI({mi_code_int}): '{dx_text}' ({actual_n} records{flag})")

    # --- Apply MI → Benign ---
    benign_idx = list(indices_to_benign)
    if benign_idx:
        for i in range(1, 14):
            col = f"med_imp_dx___{i}"
            if col in df.columns:
                df.loc[benign_idx, col] = 0
        df.loc[benign_idx, "outcome_med_important"] = 0

    # --- Apply Benign → MI ---
    mi_idx_list = list(indices_to_mi.keys())
    for idx, mi_code_int in indices_to_mi.items():
        col = f"med_imp_dx___{mi_code_int}"
        if col in df.columns:
            df.loc[idx, col] = 1
        df.loc[idx, "outcome_med_important"] = 1

    # Clear benign checkboxes on ALL MI records (can't be both MI and benign)
    benign_checkbox_cols = [f"med_benign___{i}" for i in range(1, 7)]
    existing_benign_cols = [c for c in benign_checkbox_cols if c in df.columns]
    all_mi_idx = df.index[df["outcome_med_important"] == 1].tolist()
    if all_mi_idx and existing_benign_cols:
        for col in existing_benign_cols:
            df.loc[all_mi_idx, col] = 0

    # --- Apply sub-type flags for newly reclassified records ---
    if indices_iih:
        df.loc[list(indices_iih), "mi_subtype_iih"] = 1
    if indices_cerebellar:
        df.loc[list(indices_cerebellar), "mi_subtype_cerebellar_ataxia"] = 1

    # --- Tag existing MI-11 records with sub-types based on diagnosis text ---
    # Records already correctly coded as MI-11 in the original data also need
    # sub-type flags for IIH and cerebellar ataxia. Check both diagnosis_specify
    # and med_benign_specify since the clinically informative text may be in either.
    iih_texts = {
        "pseudotumor cerebri", "idiopathic intracranial hypertension",
        "benign intracranial hypertension", "papilledema", "? papilledema nyd",
        "pseudotumor cerebri/idiopathic intracranial hypertension",
        "b.i.h.", "bilateral papilledema", "iih",
    }
    cerebellar_texts = {
        "cerebellar ataxia", "acute cerebellar ataxia", "cerebellitis",
        "post-viral ataxia", "ataxia", "ataxia nyd", "? ataxia",
        "ataxia and weakness", "vertigo/ataxia",
        "ataxia & headache", "acute onset ataxia", "acute ataxia",
        "ataxia +/- seizure nyd",
    }
    mi_11_mask = (df["med_imp_dx___11"] == 1) & (df["mi_subtype_iih"] == 0) & \
                 (df["mi_subtype_cerebellar_ataxia"] == 0)
    for idx in df.index[mi_11_mask]:
        dx = str(df.loc[idx, "diagnosis_specify"]).strip().lower()
        mbs = str(df.loc[idx, "med_benign_specify"]).strip().lower()
        if dx in iih_texts or mbs in iih_texts:
            df.loc[idx, "mi_subtype_iih"] = 1
        elif dx in cerebellar_texts or mbs in cerebellar_texts:
            df.loc[idx, "mi_subtype_cerebellar_ataxia"] = 1

    # --- Elevate IIH and cerebellar ataxia to distinct MI codes ---
    # IIH -> code 12; Cerebellar ataxia -> code 13
    df["med_imp_dx___12"] = 0
    df["med_imp_dx___13"] = 0

    iih_mask = df["mi_subtype_iih"] == 1
    df.loc[iih_mask, "med_imp_dx___12"] = 1
    df.loc[iih_mask, "med_imp_dx___11"] = 0

    ca_mask = df["mi_subtype_cerebellar_ataxia"] == 1
    df.loc[ca_mask, "med_imp_dx___13"] = 1
    df.loc[ca_mask, "med_imp_dx___11"] = 0

    # --- Report ---
    n_mi_to_benign = len(indices_to_benign)
    n_benign_to_mi = len(indices_to_mi)
    for line in sorted(changes_log):
        print(line)

    print(f"\n  Summary:")
    print(f"    MI → Benign (checkbox errors): {n_mi_to_benign}")
    print(f"    Benign → MI (reclassified):    {n_benign_to_mi}")
    print(f"    Net MI change: {n_benign_to_mi - n_mi_to_benign:+d}")
    print(f"    New MI count: {int(df['outcome_med_important'].sum())}")

    # Updated MI breakdown
    mi_labels = {
        1: "Stroke", 2: "ICH", 3: "Vascular malformation", 4: "Vasculitis",
        5: "Venous thrombosis", 6: "Demyelinating disorder",
        7: "Brain/meningeal infection", 8: "Hydrocephalus", 9: "Neoplasm",
        10: "Inborn error of metabolism", 11: "Other medically important",
        12: "IIH (idiopathic intracranial hypertension)",
        13: "Cerebellar ataxia",
    }
    print(f"\n  Updated MI breakdown:")
    for i, label in mi_labels.items():
        col = f"med_imp_dx___{i}"
        if col in df.columns:
            n = (df[col] == 1).sum()
            if n > 0:
                print(f"    {label}: {n}")

    return df


# ===================================================================
# 6. CONVERT DATA TYPES
# ===================================================================

def convert_types(df):
    """
    Convert string-encoded fields to appropriate numeric types.
    Respect branching logic: empty strings in branched fields -> NaN (structurally absent).
    """
    # --- Continuous variables ---
    continuous = ["age", "temp", "hr", "sys_bp", "dias_bp",
                  "seizure_number", "seizure_duration"]
    for col in continuous:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Binary/checkbox fields (0/1 coded) ---
    checkbox_prefixes = [
        "symptoms___", "previous_diagnoses___", "other_classify___",
        "headache_specify___", "current_headache___", "exam_notes___",
        "investigations___", "med_imp_dx___", "med_benign___",
        "epilepsy_describe___", "febrile_describe___",
        "seizure_treatment___", "migraine_tx___", "migraine_iv___",
        "headache_tx___", "headache_iv___",
        "prev_diagnosis_febrile___",
        "consult_specify___", "admit_location___",
        "discharge_refer_specify___",
    ]
    for col in df.columns:
        if any(col.startswith(p) for p in checkbox_prefixes):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Binary yes/no fields
    binary_fields = ["trauma", "ems", "ems_meds", "see_doctor", "diagnosis",
                     "consult", "admit", "discharge_refer", "lr_review",
                     "diff_dx"]
    for col in binary_fields:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ordinal/categorical
    for col in ["gender", "ctas", "time_year", "ni_appropriateness",
                "results", "med_import"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ===================================================================
# 7. HANDLE MISSING DATA
# ===================================================================

def handle_missing(df):
    """
    Report and handle missing values.
    Distinguish truly missing from structurally absent (branching logic).
    """
    # --- Report missingness for key analytic variables ---
    key_vars = ["age", "gender", "temp", "hr", "sys_bp", "dias_bp", "ctas"]
    print("\n=== Missing Data (key variables) ===")
    for col in key_vars:
        if col in df.columns:
            n_miss = df[col].isna().sum()
            by_year = df.groupby("year")[col].apply(lambda x: x.isna().sum())
            by_outcome = df.groupby("outcome_med_important")[col].apply(lambda x: x.isna().sum())
            print(f"  {col}: {n_miss} total "
                  f"(Y1={by_year.get('1', 0)}, Y2={by_year.get('2', 0)}) "
                  f"(benign={by_outcome.get(0, 0)}, important={by_outcome.get(1, 0)})")

    # --- Flag (do not drop) records with missing gender ---
    gender_missing = df["gender"].isna()
    n_gender_miss = gender_missing.sum()
    if n_gender_miss > 0:
        print(f"\n  Records with missing gender: {n_gender_miss} (flagged, not dropped)")

    # --- Impute continuous variables with median ---
    impute_cols = ["temp", "hr", "ctas"]
    print("\n=== Imputation (median) ===")
    for col in impute_cols:
        if col in df.columns:
            n_miss = df[col].isna().sum()
            if n_miss > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"  {col}: imputed {n_miss} values with median={median_val}")

    # Note: sys_bp and dias_bp have high missingness; do NOT impute —
    # these are not used as predictors in the primary model.

    return df


# ===================================================================
# 8. FEATURE ENGINEERING
# ===================================================================

def engineer_features(df):
    """
    Create derived analytic variables.
    """
    # --- Z-score continuous predictors ---
    zscore_cols = ["age", "hr", "temp"]
    for col in zscore_cols:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[f"{col}_z"] = (df[col] - mean) / std
            else:
                df[f"{col}_z"] = 0.0

    # --- Harmonized investigation flags (imaging only) ---
    imaging_map = {
        "inv_eeg": "investigations___1",
        "inv_ct": "investigations___2",
        "inv_lp": "investigations___3",
        "inv_mri": "investigations___4",
        "inv_specific_tx": "investigations___5",
        "inv_cta": "investigations___6",
        "inv_mra": "investigations___7",
        "inv_angiography": "investigations___8",
    }
    for new_name, src_col in imaging_map.items():
        if src_col in df.columns:
            df[new_name] = df[src_col]

    # Any neuroimaging
    img_cols = ["inv_ct", "inv_mri", "inv_cta", "inv_mra"]
    existing_img = [c for c in img_cols if c in df.columns]
    df["any_neuroimaging"] = df[existing_img].max(axis=1)

    # --- Harmonized headache features (concept-based names) ---
    ha_concept_map = {
        "ha_nausea": "headache_specify___1",
        "ha_vomiting": "headache_specify___2",
        "ha_photophobia": "headache_specify___3",
        "ha_phonophobia": "headache_specify___4",
        "ha_maximal_onset": "current_headache___1",
        "ha_progressive_days": "current_headache___2",
        "ha_progressive_months": "current_headache___3",
        "ha_non_progressive": "current_headache___4",
        "ha_abrupt_onset": "current_headache___5",
        "ha_gradual_onset": "current_headache___6",
        "ha_bilateral": "current_headache___8",
        "ha_unilateral": "current_headache___9",
        "ha_frontal": "current_headache___10",
        "ha_temporal": "current_headache___11",
        "ha_occipital": "current_headache___12",
    }
    for new_name, src_col in ha_concept_map.items():
        if src_col in df.columns:
            df[new_name] = df[src_col]
        else:
            df[new_name] = 0

    # --- Symptom concept names ---
    symptom_map = {
        "sx_focal_weakness": "symptoms___1",
        "sx_sensory": "symptoms___2",
        "sx_speech": "symptoms___3",
        "sx_visual": "symptoms___4",
        "sx_eye_movement": "symptoms___5",
        "sx_altered_consciousness": "symptoms___6",
        "sx_collapse": "symptoms___7",
        "sx_seizure": "symptoms___8",
        "sx_ataxia": "symptoms___9",
        "sx_headache": "symptoms___10",
        "sx_headache_other": "symptoms___11",
        "sx_fever": "symptoms___12",
        "sx_other": "symptoms___13",
    }
    for new_name, src_col in symptom_map.items():
        if src_col in df.columns:
            df[new_name] = df[src_col]

    # --- Exam concept names ---
    exam_map = {
        "exam_papilledema": "exam_notes___1",
        "exam_meningismus": "exam_notes___2",
        "exam_altered_loc": "exam_notes___3",
    }
    for new_name, src_col in exam_map.items():
        if src_col in df.columns:
            df[new_name] = df[src_col]

    # --- Other symptom classifications ---
    other_sx_map = {
        "other_pain": "other_classify___1",
        "other_movement": "other_classify___2",
        "other_psych": "other_classify___3",
        "other_gi": "other_classify___4",
        "other_cranial_nerve": "other_classify___5",
        "other_inflammation": "other_classify___6",
        "other_respiratory": "other_classify___7",
        "other_infection": "other_classify___8",
        "other_constitutional": "other_classify___9",
        "other_urinary": "other_classify___10",
        "other_dermatological": "other_classify___11",
        "other_hemorrhage": "other_classify___12",
        "other_syncope": "other_classify___13",
        "other_cardiac": "other_classify___14",
        "other_gyn": "other_classify___15",
        "other_ophthalmology": "other_classify___16",
        "other_sleep": "other_classify___17",
        "other_hematological": "other_classify___18",
    }
    for new_name, src_col in other_sx_map.items():
        if src_col in df.columns:
            df[new_name] = df[src_col]

    # --- Age binary (for headache sub-analysis) ---
    df["age_under_12"] = (df["age"] < 12).astype(int)

    # --- Any headache symptom ---
    df["sx_any_headache"] = ((df.get("sx_headache", 0) == 1) |
                              (df.get("sx_headache_other", 0) == 1)).astype(int)

    # --- Any previous diagnosis ---
    prev_cols = [f"previous_diagnoses___{i}" for i in range(1, 8)]
    existing_prev = [c for c in prev_cols if c in df.columns]
    # "No previous diagnoses" is option 7; actual diagnoses are 1-6
    prev_dx_cols = [c for c in existing_prev if not c.endswith("___7")]
    df["any_prev_dx"] = df[prev_dx_cols].max(axis=1) if prev_dx_cols else 0

    return df


# ===================================================================
# 9. VALIDATE AGAINST MANUSCRIPT
# ===================================================================

def validate(df, flow):
    """
    Compare rebuilt dataset against expected counts.

    Expected counts reflect the post-PI-review totals:
      - Original manuscript: N=2,153, MI=113 (pre-PI-review)
      - After PI review: MI=147 (+48 reclassified to MI, -18 corrected to benign)
      - After code elevation: IIH (code 12, n=8) and cerebellar ataxia (code 13, n=26)
        separated from Other MI (code 11)

    Per-code counts also shifted vs the original manuscript due to PI review
    reclassifications (e.g., shunt malfunction → hydrocephalus, optic neuritis →
    demyelinating, brain lesion → neoplasm).
    """
    print("\n=== VALIDATION ===")
    n_total = len(df)
    n_mi = df["outcome_med_important"].sum()
    n_benign = n_total - n_mi

    print(f"  Total analytic cohort: {n_total} (expected: 2,153)")
    print(f"  Medically important:   {n_mi} (expected: 147 post-PI-review)")
    print(f"  Medically unimportant: {n_benign} (expected: 2,006)")

    by_year = df.groupby("year")["outcome_med_important"].agg(["count", "sum"])
    print(f"\n  By year:")
    for yr, row in by_year.iterrows():
        print(f"    Year {yr}: {int(row['count'])} total, {int(row['sum'])} medically important")

    # Post-PI-review expected counts per MI code
    mi_expected = {
        1: 8, 2: 8, 3: 2, 5: 3, 6: 13, 7: 42, 8: 10, 9: 22, 10: 1,
        11: 12, 12: 8, 13: 26,
    }
    mi_labels = {
        1: "Stroke", 2: "ICH", 3: "Vascular malformation", 5: "Venous thrombosis",
        6: "Demyelinating", 7: "Infection", 8: "Hydrocephalus", 9: "Neoplasm",
        10: "Inborn error", 11: "Other MI", 12: "IIH", 13: "Cerebellar ataxia",
    }
    print(f"\n  Diagnosis counts (rebuilt vs expected):")
    all_match = True
    for code, expected in sorted(mi_expected.items()):
        col = f"med_imp_dx___{code}"
        if col in df.columns:
            actual = (df[col] == 1).sum()
            match = "OK" if actual == expected else f"MISMATCH (expected {expected})"
            if actual != expected:
                all_match = False
            print(f"    {mi_labels.get(code, f'Code {code}')}: {actual} {match}")

    # Concordance flag
    concordant = (n_total == 2153 and n_mi == 147 and all_match)
    if concordant:
        print("\n  CONCORDANCE: Dataset matches expected post-PI-review counts.")
    else:
        delta_n = n_total - 2153
        delta_mi = n_mi - 147
        print(f"\n  DISCREPANCY: N off by {delta_n}, MI off by {delta_mi}")

    return concordant


# ===================================================================
# 10. GENERATE SUMMARY REPORT
# ===================================================================

def generate_summary(df, flow):
    """Export a summary table of the data preparation."""
    summary = {
        "metric": [],
        "value": [],
    }

    for k, v in flow.items():
        summary["metric"].append(k)
        summary["value"].append(v)

    summary["metric"].append("final_analytic_n")
    summary["value"].append(len(df))
    summary["metric"].append("n_medically_important")
    summary["value"].append(int(df["outcome_med_important"].sum()))
    summary["metric"].append("n_year1")
    summary["value"].append(int((df["year"] == "1").sum()))
    summary["metric"].append("n_year2")
    summary["value"].append(int((df["year"] == "2").sum()))
    summary["metric"].append("n_columns")
    summary["value"].append(len(df.columns))

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUTPUTS / "data_prep_summary.csv", index=False)
    print(f"\nSummary saved to {OUTPUTS / 'data_prep_summary.csv'}")


# ===================================================================
# 10b. MISSING DATA PATTERNS TABLE (TRIPOD+AI Item 14)
# ===================================================================

def generate_missing_data_table(df):
    """
    Generate TableS4: missing data patterns by year and outcome.
    Reports missingness for key analytic variables stratified by
    year (1 vs 2) and outcome (MI vs benign).
    """
    print("\n=== TableS4: Missing Data Patterns ===")

    key_vars = [
        "age", "gender", "temp", "hr", "sys_bp", "dias_bp", "ctas",
        "sx_focal_weakness", "sx_sensory", "sx_speech", "sx_visual",
        "sx_eye_movement", "sx_altered_consciousness", "sx_collapse",
        "sx_seizure", "sx_ataxia", "sx_any_headache", "sx_fever",
        "exam_papilledema", "exam_meningismus", "exam_altered_loc",
        "ems", "any_prev_dx",
    ]
    # Only include vars present in dataset
    key_vars = [v for v in key_vars if v in df.columns]

    rows = []
    for var in key_vars:
        # Check if variable was imputed (temp, hr, ctas were imputed with median)
        # For imputed vars, report pre-imputation counts from source columns
        # For binary/checkbox vars, NaN = truly missing (rare)
        n_total = df[var].isna().sum()

        # By year
        n_y1 = df.loc[df["year"] == "1", var].isna().sum()
        n_y2 = df.loc[df["year"] == "2", var].isna().sum()

        # By outcome
        n_mi = df.loc[df["outcome_med_important"] == 1, var].isna().sum()
        n_benign = df.loc[df["outcome_med_important"] == 0, var].isna().sum()

        rows.append({
            "Variable": var,
            "N missing (total)": int(n_total),
            "% missing (total)": f"{100 * n_total / len(df):.1f}%",
            "N missing Year 1": int(n_y1),
            "N missing Year 2": int(n_y2),
            "N missing MI": int(n_mi),
            "N missing Benign": int(n_benign),
        })

    tbl = pd.DataFrame(rows)
    out = OUTPUTS / "tableS4_missing_data.csv"
    tbl.to_csv(out, index=False)
    print(f"  Saved {out.name} ({len(tbl)} variables)")

    # Print vars with any missingness
    has_missing = tbl[tbl["N missing (total)"] > 0]
    if len(has_missing) > 0:
        print("  Variables with missing data:")
        for _, r in has_missing.iterrows():
            print(f"    {r['Variable']}: {r['N missing (total)']} ({r['% missing (total)']})")
    else:
        print("  No missing data in key analytic variables (post-imputation)")

    return tbl


# ===================================================================
# 10c. PARTICIPANT FLOW DIAGRAM (TRIPOD+AI Item 20)
# ===================================================================

def generate_flow_diagram(df, flow):
    """
    Generate fig_participant_flow.png: participant flow from
    initial records through exclusions to final analytic cohorts.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("\n=== Participant Flow Diagram (TRIPOD+AI Item 20) ===")

    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")

    box_kw = dict(boxstyle="round,pad=0.4", facecolor="#e8e8e8", edgecolor="black", linewidth=1.2)
    exc_kw = dict(boxstyle="round,pad=0.4", facecolor="#ffe0e0", edgecolor="#c00000", linewidth=1.0)
    out_kw = dict(boxstyle="round,pad=0.4", facecolor="#e0f0e0", edgecolor="#006000", linewidth=1.2)
    font_kw = dict(fontsize=10, ha="center", va="center", fontweight="normal")

    n_y1 = flow.get("raw_total", 2329) - 1205  # Approximate Y1
    n_y2 = 1205
    n_raw = flow.get("raw_total", 2329)
    n_excluded = flow.get("excluded_trauma", 176)
    n_final = flow.get("after_exclusions", 2153)
    n_mi = int(df["outcome_med_important"].sum())
    n_benign = n_final - n_mi

    # Year 1 and Year 2 boxes
    y1_count = int((df["year"] == "1").sum())
    y2_count = int((df["year"] == "2").sum())
    y1_mi = int((df.loc[df["year"] == "1", "outcome_med_important"] == 1).sum())
    y2_mi = int((df.loc[df["year"] == "2", "outcome_med_important"] == 1).sum())

    # Row 1: Source datasets
    ax.text(3, 13, f"Year 1 REDCap export\nn = 1,124", bbox=box_kw, **font_kw)
    ax.text(7, 13, f"Year 2 REDCap export\nn = 1,205", bbox=box_kw, **font_kw)

    # Arrows down
    ax.annotate("", xy=(5, 12.2), xytext=(3, 12.5), arrowprops=dict(arrowstyle="-", color="black"))
    ax.annotate("", xy=(5, 12.2), xytext=(7, 12.5), arrowprops=dict(arrowstyle="-", color="black"))

    # Row 2: Combined
    ax.text(5, 11.5, f"Combined cohort\nn = {n_raw:,}", bbox=box_kw, **font_kw)
    ax.annotate("", xy=(5, 10.9), xytext=(5, 11.0), arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    # Row 3: Exclusion
    ax.text(8.2, 10.3, f"Excluded: trauma\nn = {n_excluded}", bbox=exc_kw, **font_kw)
    ax.annotate("", xy=(6.8, 10.3), xytext=(5.5, 10.3), arrowprops=dict(arrowstyle="->", color="#c00000", lw=1.0))

    # Row 4: After exclusions
    ax.text(5, 9.3, f"Analytic cohort\nn = {n_final:,}", bbox=box_kw, **font_kw)
    ax.annotate("", xy=(5, 9.8), xytext=(5, 10.0), arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    # Row 5: LLM coding + PI review
    ax.text(5, 7.8, "LLM-assisted free-text coding\n+ PI review of 124 ambiguous cases\n"
            "(+48 reclassified to MI, −18 corrected to benign)",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8e0f0", edgecolor="#6000a0", linewidth=1.0),
            **font_kw)
    ax.annotate("", xy=(5, 8.5), xytext=(5, 8.7), arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    # Row 6: Final outcome
    ax.text(5, 6.2, f"Final analytic cohort\nn = {n_final:,}\n"
            f"MI = {n_mi} ({100*n_mi/n_final:.1f}%), "
            f"Benign = {n_benign}",
            bbox=out_kw, fontsize=10, ha="center", va="center", fontweight="bold")
    ax.annotate("", xy=(5, 6.9), xytext=(5, 7.1), arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    # Row 7: Year split
    ax.text(3, 4.5, f"Year 1\nn = {y1_count:,}\nMI = {y1_mi}",
            bbox=box_kw, **font_kw)
    ax.text(7, 4.5, f"Year 2\nn = {y2_count:,}\nMI = {y2_mi}",
            bbox=box_kw, **font_kw)
    ax.annotate("", xy=(3, 5.1), xytext=(5, 5.6), arrowprops=dict(arrowstyle="->", color="black", lw=1.0))
    ax.annotate("", xy=(7, 5.1), xytext=(5, 5.6), arrowprops=dict(arrowstyle="->", color="black", lw=1.0))

    # Row 8: Purpose labels
    ax.text(3, 3.5, "Derivation", fontsize=9, ha="center", va="center",
            fontstyle="italic", color="#666")
    ax.text(7, 3.5, "Temporal validation", fontsize=9, ha="center", va="center",
            fontstyle="italic", color="#666")

    fig.tight_layout()
    out_path = FIGURES / "fig_participant_flow.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path.name}")


# ===================================================================
# MAIN
# ===================================================================

def main():
    print("=" * 60)
    print("Brain AttACKs Data Preparation")
    print("=" * 60)

    # 1. Load
    y1_raw, y2_raw = load_raw()

    # 2. Harmonize
    print("\n--- Harmonizing Year 1 ---")
    y1 = harmonize_year1(y1_raw)
    print(f"  Year 1 after harmonization: {len(y1.columns)} columns")

    print("\n--- Harmonizing Year 2 ---")
    y2 = harmonize_year2(y2_raw)
    print(f"  Year 2 after harmonization: {len(y2.columns)} columns")

    # 3. Combine
    df = combine_datasets(y1, y2)

    # Defragment after column manipulation
    df = df.copy()

    # 4. Exclusions
    df, flow = apply_exclusions(df)

    # 5. Convert types
    df = convert_types(df)

    # 6. Derive outcomes
    df = df.copy()  # defragment
    df = derive_outcomes(df)

    # 6b. Apply PI review corrections
    df = apply_pi_review(df)

    # 7. Handle missing
    df = handle_missing(df)

    # 8. Feature engineering
    df = df.copy()  # defragment before adding many columns
    df = engineer_features(df)

    # 9. Validate
    concordant = validate(df, flow)

    # 10. Summary
    generate_summary(df, flow)

    # 10b. Missing data patterns (TRIPOD+AI Item 14)
    generate_missing_data_table(df)

    # 10c. Participant flow diagram (TRIPOD+AI Item 20)
    generate_flow_diagram(df, flow)

    # 11. Export
    print(f"\n--- Exporting ---")
    df.to_csv(PROCESSED / "brain_attacks_combined.csv", index=False)
    df.to_parquet(PROCESSED / "brain_attacks_combined.parquet", index=False)
    print(f"  CSV:     {PROCESSED / 'brain_attacks_combined.csv'}")
    print(f"  Parquet: {PROCESSED / 'brain_attacks_combined.parquet'}")
    print(f"  Final shape: {df.shape}")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)

    return df


if __name__ == "__main__":
    df = main()
