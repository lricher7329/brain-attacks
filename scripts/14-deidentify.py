"""
14-deidentify.py — Create De-Identified Dataset for Public Sharing

Reads the combined Brain AttACKs parquet (N=2,153) and LLM-coded CSV,
removes all direct/quasi-identifiers, applies date shifting, suppresses
rare diagnoses, and writes a shareable CSV with data dictionary and README.

De-identification steps:
  1. Replace record_id with shuffled sequential study_id
  2. Date shift visit dates (±180 days per record, seeded)
  3. Drop dob, time, free-text fields, internal/admin fields
  4. Cap extreme ages (< 0.1 → 0.1, > 18 → 18), round to 1 decimal
  5. Suppress rare MI subtypes (≤ 3 cases → merge into Other MI code 11)
  6. Shuffle row order

Inputs:
    data/processed/brain_attacks_combined.parquet
    data/processed/brain_attacks_llm_coded.csv  (for comorbid_* columns)

Outputs:
    data/sharing/brain_attacks_deidentified.csv
    data/sharing/DATA_DICTIONARY.csv
    data/sharing/README.md
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEED = 42
rng = np.random.default_rng(SEED)

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data" / "processed"
SHARING = PROJECT / "data" / "sharing"
SHARING.mkdir(parents=True, exist_ok=True)


def _print(msg=""):
    print(msg, flush=True)


# ===================================================================
# 1. LOAD DATA
# ===================================================================

def load_data():
    """Load parquet and merge LLM-coded columns (comorbid_*)."""
    df = pd.read_parquet(DATA / "brain_attacks_combined.parquet")
    _print(f"Loaded parquet: {len(df)} records, {len(df.columns)} columns")

    llm_path = DATA / "brain_attacks_llm_coded.csv"
    if llm_path.exists():
        llm = pd.read_csv(llm_path, dtype=str)
        new_cols = [c for c in llm.columns if c not in df.columns]
        if new_cols:
            key_cols = ["record_id", "year"]
            merge_cols = key_cols + new_cols
            llm_sub = llm[merge_cols].copy()
            df["record_id"] = df["record_id"].astype(str)
            df["year"] = df["year"].astype(str)
            llm_sub["record_id"] = llm_sub["record_id"].astype(str)
            llm_sub["year"] = llm_sub["year"].astype(str)
            df = df.merge(llm_sub, on=key_cols, how="left")
            _print(f"  Merged {len(new_cols)} LLM-coded columns")

    return df


# ===================================================================
# 2. DEFINE COLUMNS TO RETAIN
# ===================================================================

# Demographics
DEMO_COLS = ["age", "age_under_12", "gender", "year", "time_year"]

# Vitals / Triage
VITALS_COLS = ["temp", "hr", "sys_bp", "dias_bp", "ctas"]

# Transport
TRANSPORT_COLS = ["ems", "ems_meds"]

# Symptoms (12 standardized)
SYMPTOM_COLS = [
    "sx_focal_weakness", "sx_sensory", "sx_speech", "sx_visual",
    "sx_eye_movement", "sx_altered_consciousness", "sx_collapse",
    "sx_seizure", "sx_ataxia", "sx_fever", "sx_any_headache", "sx_other",
]

# Other symptom categories (18)
OTHER_SX_COLS = [
    "other_pain", "other_movement", "other_psych", "other_gi",
    "other_cranial_nerve", "other_inflammation", "other_respiratory",
    "other_infection", "other_constitutional", "other_urinary",
    "other_dermatological", "other_hemorrhage", "other_syncope",
    "other_cardiac", "other_gyn", "other_ophthalmology",
    "other_sleep", "other_hematological",
]

# Exam findings
EXAM_COLS = ["exam_papilledema", "exam_meningismus", "exam_altered_loc"]

# Headache features (temporal + location)
HEADACHE_COLS = [
    "ha_nausea", "ha_vomiting", "ha_photophobia", "ha_phonophobia",
    "ha_maximal_onset", "ha_progressive_days", "ha_progressive_months",
    "ha_non_progressive", "ha_abrupt_onset", "ha_gradual_onset",
    "ha_bilateral", "ha_unilateral", "ha_frontal", "ha_temporal", "ha_occipital",
]

# Prior history
PREV_DX_COLS = [f"previous_diagnoses___{i}" for i in range(1, 8)]
PREV_DX_OTHER = ["any_prev_dx"]
COMORBID_COLS = [
    "comorbid_neurological", "comorbid_developmental", "comorbid_psychiatric",
    "comorbid_cardiac", "comorbid_respiratory", "comorbid_autoimmune",
    "comorbid_hematological", "comorbid_oncological", "comorbid_metabolic",
    "comorbid_genetic", "comorbid_migraine_hx", "comorbid_other",
]

# Investigations
INV_COLS = [
    "inv_eeg", "inv_ct", "inv_lp", "inv_mri", "inv_cta", "inv_mra",
    "inv_angiography", "inv_specific_tx", "any_neuroimaging",
    "results", "med_import", "ni_appropriateness",
]

# Outcomes
MI_DX_COLS = [f"med_imp_dx___{i}" for i in range(1, 14)]
BENIGN_COLS = [f"med_benign___{i}" for i in range(1, 7)]
OUTCOME_COLS = [
    "outcome_med_important", "outcome_abnormal_imaging", "outcome_mi_imaging",
] + MI_DX_COLS + BENIGN_COLS + [
    "mi_subtype_iih", "mi_subtype_cerebellar_ataxia",
    "icu_any", "admit_any",
]

# Admission / Consult / Disposition
ADMIT_COLS = [
    "admit", "admit_location___1", "admit_location___2", "admit_location___3",
    "admit_location___4", "admit_location___5",
    "consult", "consult_specify___1", "consult_specify___2",
    "consult_specify___3", "consult_specify___4", "consult_specify___5",
    "consult_specify___6", "consult_specify___7", "consult_specify___8",
    "discharge_refer", "discharge_refer_specify___1",
    "discharge_refer_specify___2", "discharge_refer_specify___3",
    "discharge_refer_specify___4", "discharge_refer_specify___5",
]

# Seizure-specific
SEIZURE_COLS = [
    "seizure_number", "seizure_duration",
    "seizure_treatment___1", "seizure_treatment___2",
    "seizure_treatment___3", "seizure_treatment___4",
    "epilepsy_describe___1", "epilepsy_describe___2", "epilepsy_describe___3",
    "febrile_describe___1", "febrile_describe___2",
    "febrile_describe___3", "febrile_describe___4",
]

# Diagnosis flags
DIAG_FLAG_COLS = ["diagnosis", "diff_dx"]

# Combine all retained columns
ALL_RETAIN = (
    DEMO_COLS + VITALS_COLS + TRANSPORT_COLS + SYMPTOM_COLS +
    OTHER_SX_COLS + EXAM_COLS + HEADACHE_COLS +
    PREV_DX_COLS + PREV_DX_OTHER + COMORBID_COLS +
    INV_COLS + OUTCOME_COLS + ADMIT_COLS + SEIZURE_COLS + DIAG_FLAG_COLS
)


# ===================================================================
# 3. DE-IDENTIFICATION
# ===================================================================

def deidentify(df):
    """Apply all de-identification steps and return clean DataFrame."""

    n_original = len(df)
    _print(f"\n{'='*60}")
    _print("DE-IDENTIFICATION REPORT")
    _print(f"{'='*60}")
    _print(f"Source records: {n_original}")

    # --- 3a. Select retained columns (skip any missing from source) ---
    available = [c for c in ALL_RETAIN if c in df.columns]
    missing = [c for c in ALL_RETAIN if c not in df.columns]
    if missing:
        _print(f"\nWARNING — {len(missing)} planned columns not in source:")
        for c in missing:
            _print(f"  - {c}")

    out = df[available].copy()
    removed = set(df.columns) - set(available)
    _print(f"\nColumns retained: {len(available)}")
    _print(f"Columns removed:  {len(removed)}")

    # --- 3b. Date shifting (guarantee no same-record overlap) ---
    if "date_of_visit" in df.columns:
        # Draw from ±180 excluding 0 so no record keeps its original date
        offsets = rng.choice(
            np.concatenate([np.arange(-180, 0), np.arange(1, 181)]),
            size=n_original, replace=True,
        )
        dates = pd.to_datetime(df["date_of_visit"], errors="coerce")
        shifted = dates + pd.to_timedelta(offsets, unit="D")
        out.insert(0, "visit_date_shifted", shifted.dt.strftime("%Y-%m-%d"))
        # Verify no same-record match
        same = (dates.dt.strftime("%Y-%m-%d") == shifted.dt.strftime("%Y-%m-%d")).sum()
        _print(f"\nDate shifting: ±180 days applied to {dates.notna().sum()} dates")
        _print(f"  Same-record date matches: {same} (must be 0)")
        assert same == 0, "Date shift failed: same-record overlap detected"
    else:
        _print("\nWARNING — date_of_visit not found, skipping date shift")

    # --- 3c. Age generalization ---
    if "age" in out.columns:
        age = pd.to_numeric(out["age"], errors="coerce")
        age = age.round(1)
        n_low = (age < 0.1).sum()
        n_high = (age > 18).sum()
        age = age.clip(lower=0.1, upper=18.0)
        out["age"] = age
        _print(f"\nAge generalization: rounded to 1 decimal, "
               f"capped {n_low} < 0.1, {n_high} > 18")

    # --- 3d. Rare MI subtype suppression ---
    mi_dx_present = [c for c in MI_DX_COLS if c in out.columns]
    suppressed = []
    for col in mi_dx_present:
        if col == "med_imp_dx___11":
            continue  # "Other MI" — the merge target
        n_cases = pd.to_numeric(out[col], errors="coerce").fillna(0).sum()
        if 0 < n_cases <= 3:
            # Merge into Other MI (code 11)
            other_col = "med_imp_dx___11"
            if other_col in out.columns:
                out[other_col] = (
                    pd.to_numeric(out[other_col], errors="coerce").fillna(0).astype(int) |
                    pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
                )
            out[col] = 0
            suppressed.append((col, int(n_cases)))

    if suppressed:
        _print(f"\nRare MI subtype suppression (≤ 3 cases → merged into code 11):")
        for col, n in suppressed:
            _print(f"  {col}: {n} cases → merged into med_imp_dx___11")
    else:
        _print("\nNo rare MI subtypes to suppress")

    # --- 3e. Generate study_id and shuffle ---
    study_ids = np.arange(1, n_original + 1)
    rng.shuffle(study_ids)
    out.insert(0, "study_id", study_ids)

    # Shuffle row order
    shuffle_idx = rng.permutation(n_original)
    out = out.iloc[shuffle_idx].reset_index(drop=True)
    _print(f"\nRow order shuffled (seed={SEED})")

    # --- 3f. Verify ---
    assert len(out) == n_original, f"Row count mismatch: {len(out)} vs {n_original}"
    phi_cols = {"record_id", "dob", "date_of_visit", "time", "redcap_event_name",
                "diagnosis_specify", "med_benign_specify", "med_imp_specify",
                "symptom_other", "prev_diagnoses_other", "diffdx_specify",
                "headache_other", "treatment_specify", "ems_meds_specify",
                "consult_other", "admit_other", "refer_other",
                "consult_specify", "discharge_refer_specify",
                "lr_review", "related", "see_doctor", "trauma",
                "brain_attacks_crf_complete", "dx_discrepancy",
                "benign_review_flag", "benign_suggested_mi",
                "llm_dx_class", "llm_dx_code", "llm_dx_confidence"}
    leaked = phi_cols & set(out.columns)
    assert not leaked, f"PHI columns still present: {leaked}"

    _print(f"\nFinal dataset: {len(out)} rows × {len(out.columns)} columns")
    _print(f"Verification: no PHI columns detected ✓")

    return out


# ===================================================================
# 4. DATA DICTIONARY
# ===================================================================

DICTIONARY_ENTRIES = [
    ("study_id", "integer", "Anonymous sequential identifier (1–N), randomly assigned"),
    ("visit_date_shifted", "date (YYYY-MM-DD)", "Date of ED visit, shifted ±180 days per record"),
    # Demographics
    ("age", "float", "Age in years (rounded to 1 decimal, capped 0.1–18.0)"),
    ("age_under_12", "binary (0/1)", "Age < 12 years"),
    ("gender", "binary (0/1)", "0 = Female, 1 = Male"),
    ("year", "integer (1/2)", "Study year: 1 = Year 1, 2 = Year 2"),
    ("time_year", "integer (1–4)", "Quarter of year of presentation"),
    # Vitals
    ("temp", "float", "Temperature (°C)"),
    ("hr", "float", "Heart rate (bpm)"),
    ("sys_bp", "float", "Systolic blood pressure (mmHg)"),
    ("dias_bp", "float", "Diastolic blood pressure (mmHg)"),
    ("ctas", "integer (1–5)", "Canadian Triage Acuity Scale score"),
    # Transport
    ("ems", "binary (0/1)", "Arrived by EMS"),
    ("ems_meds", "binary (0/1)", "Medications administered by EMS"),
    # Symptoms
    ("sx_focal_weakness", "binary (0/1)", "Presenting symptom: focal weakness"),
    ("sx_sensory", "binary (0/1)", "Presenting symptom: sensory change"),
    ("sx_speech", "binary (0/1)", "Presenting symptom: speech disturbance"),
    ("sx_visual", "binary (0/1)", "Presenting symptom: visual disturbance"),
    ("sx_eye_movement", "binary (0/1)", "Presenting symptom: eye movement abnormality"),
    ("sx_altered_consciousness", "binary (0/1)", "Presenting symptom: altered consciousness"),
    ("sx_collapse", "binary (0/1)", "Presenting symptom: collapse"),
    ("sx_seizure", "binary (0/1)", "Presenting symptom: seizure"),
    ("sx_ataxia", "binary (0/1)", "Presenting symptom: ataxia"),
    ("sx_fever", "binary (0/1)", "Presenting symptom: fever"),
    ("sx_any_headache", "binary (0/1)", "Presenting symptom: any headache"),
    ("sx_other", "binary (0/1)", "Presenting symptom: other (not in standard list)"),
    # Other symptom categories
    ("other_pain", "binary (0/1)", "Other symptom category: pain"),
    ("other_movement", "binary (0/1)", "Other symptom category: movement disorder"),
    ("other_psych", "binary (0/1)", "Other symptom category: psychiatric"),
    ("other_gi", "binary (0/1)", "Other symptom category: gastrointestinal"),
    ("other_cranial_nerve", "binary (0/1)", "Other symptom category: cranial nerve"),
    ("other_inflammation", "binary (0/1)", "Other symptom category: inflammation"),
    ("other_respiratory", "binary (0/1)", "Other symptom category: respiratory"),
    ("other_infection", "binary (0/1)", "Other symptom category: infection"),
    ("other_constitutional", "binary (0/1)", "Other symptom category: constitutional"),
    ("other_urinary", "binary (0/1)", "Other symptom category: urinary"),
    ("other_dermatological", "binary (0/1)", "Other symptom category: dermatological"),
    ("other_hemorrhage", "binary (0/1)", "Other symptom category: hemorrhage"),
    ("other_syncope", "binary (0/1)", "Other symptom category: syncope"),
    ("other_cardiac", "binary (0/1)", "Other symptom category: cardiac"),
    ("other_gyn", "binary (0/1)", "Other symptom category: gynecological"),
    ("other_ophthalmology", "binary (0/1)", "Other symptom category: ophthalmological"),
    ("other_sleep", "binary (0/1)", "Other symptom category: sleep"),
    ("other_hematological", "binary (0/1)", "Other symptom category: hematological"),
    # Exam
    ("exam_papilledema", "binary (0/1)", "Exam finding: papilledema"),
    ("exam_meningismus", "binary (0/1)", "Exam finding: meningismus"),
    ("exam_altered_loc", "binary (0/1)", "Exam finding: altered level of consciousness"),
    # Headache features
    ("ha_nausea", "binary (0/1)", "Headache feature: nausea"),
    ("ha_vomiting", "binary (0/1)", "Headache feature: vomiting"),
    ("ha_photophobia", "binary (0/1)", "Headache feature: photophobia"),
    ("ha_phonophobia", "binary (0/1)", "Headache feature: phonophobia"),
    ("ha_maximal_onset", "binary (0/1)", "Headache feature: maximal at onset"),
    ("ha_progressive_days", "binary (0/1)", "Headache feature: progressive over days"),
    ("ha_progressive_months", "binary (0/1)", "Headache feature: progressive over months"),
    ("ha_non_progressive", "binary (0/1)", "Headache feature: non-progressive"),
    ("ha_abrupt_onset", "binary (0/1)", "Headache feature: abrupt onset"),
    ("ha_gradual_onset", "binary (0/1)", "Headache feature: gradual onset"),
    ("ha_bilateral", "binary (0/1)", "Headache location: bilateral"),
    ("ha_unilateral", "binary (0/1)", "Headache location: unilateral"),
    ("ha_frontal", "binary (0/1)", "Headache location: frontal"),
    ("ha_temporal", "binary (0/1)", "Headache location: temporal"),
    ("ha_occipital", "binary (0/1)", "Headache location: occipital"),
    # Prior history
    ("previous_diagnoses___1", "binary (0/1)", "Prior diagnosis: epilepsy"),
    ("previous_diagnoses___2", "binary (0/1)", "Prior diagnosis: febrile seizures"),
    ("previous_diagnoses___3", "binary (0/1)", "Prior diagnosis: migraine"),
    ("previous_diagnoses___4", "binary (0/1)", "Prior diagnosis: previous stroke"),
    ("previous_diagnoses___5", "binary (0/1)", "Prior diagnosis: brain tumour"),
    ("previous_diagnoses___6", "binary (0/1)", "Prior diagnosis: VP shunt"),
    ("previous_diagnoses___7", "binary (0/1)", "Prior diagnosis: none"),
    ("any_prev_dx", "binary (0/1)", "Any prior neurological diagnosis"),
    # Comorbidities (LLM-coded)
    ("comorbid_neurological", "binary (0/1)", "Comorbidity: neurological"),
    ("comorbid_developmental", "binary (0/1)", "Comorbidity: developmental"),
    ("comorbid_psychiatric", "binary (0/1)", "Comorbidity: psychiatric"),
    ("comorbid_cardiac", "binary (0/1)", "Comorbidity: cardiac"),
    ("comorbid_respiratory", "binary (0/1)", "Comorbidity: respiratory"),
    ("comorbid_autoimmune", "binary (0/1)", "Comorbidity: autoimmune"),
    ("comorbid_hematological", "binary (0/1)", "Comorbidity: hematological"),
    ("comorbid_oncological", "binary (0/1)", "Comorbidity: oncological"),
    ("comorbid_metabolic", "binary (0/1)", "Comorbidity: metabolic"),
    ("comorbid_genetic", "binary (0/1)", "Comorbidity: genetic"),
    ("comorbid_migraine_hx", "binary (0/1)", "Comorbidity: history of migraine"),
    ("comorbid_other", "binary (0/1)", "Comorbidity: other"),
    # Investigations
    ("inv_eeg", "binary (0/1)", "Investigation: EEG"),
    ("inv_ct", "binary (0/1)", "Investigation: CT head"),
    ("inv_lp", "binary (0/1)", "Investigation: lumbar puncture"),
    ("inv_mri", "binary (0/1)", "Investigation: MRI brain"),
    ("inv_cta", "binary (0/1)", "Investigation: CT angiography"),
    ("inv_mra", "binary (0/1)", "Investigation: MR angiography"),
    ("inv_angiography", "binary (0/1)", "Investigation: conventional angiography"),
    ("inv_specific_tx", "binary (0/1)", "Investigation: specific treatment given"),
    ("any_neuroimaging", "binary (0/1)", "Any neuroimaging performed (CT/MRI/CTA/MRA/angio)"),
    ("results", "categorical", "Investigation results category"),
    ("med_import", "binary (0/1)", "Medically important investigation result"),
    ("ni_appropriateness", "categorical", "Neuroimaging appropriateness rating"),
    # Outcomes
    ("outcome_med_important", "binary (0/1)", "Primary outcome: medically important brain attack"),
    ("outcome_abnormal_imaging", "binary (0/1)", "Secondary outcome: abnormal neuroimaging"),
    ("outcome_mi_imaging", "binary (0/1)", "Secondary outcome: MI cause of abnormal imaging"),
    ("med_imp_dx___1", "binary (0/1)", "MI diagnosis: stroke"),
    ("med_imp_dx___2", "binary (0/1)", "MI diagnosis: intracranial hemorrhage"),
    ("med_imp_dx___3", "binary (0/1)", "MI diagnosis: vascular malformation (suppressed if ≤3 cases)"),
    ("med_imp_dx___4", "binary (0/1)", "MI diagnosis: vasculitis (suppressed if ≤3 cases)"),
    ("med_imp_dx___5", "binary (0/1)", "MI diagnosis: venous thrombosis (suppressed if ≤3 cases)"),
    ("med_imp_dx___6", "binary (0/1)", "MI diagnosis: demyelinating disorder"),
    ("med_imp_dx___7", "binary (0/1)", "MI diagnosis: brain/meningeal infection"),
    ("med_imp_dx___8", "binary (0/1)", "MI diagnosis: hydrocephalus"),
    ("med_imp_dx___9", "binary (0/1)", "MI diagnosis: neoplasm"),
    ("med_imp_dx___10", "binary (0/1)", "MI diagnosis: inborn error of metabolism (suppressed if ≤3 cases)"),
    ("med_imp_dx___11", "binary (0/1)", "MI diagnosis: other medically important (includes suppressed rare subtypes)"),
    ("med_imp_dx___12", "binary (0/1)", "MI diagnosis: idiopathic intracranial hypertension"),
    ("med_imp_dx___13", "binary (0/1)", "MI diagnosis: cerebellar ataxia"),
    ("med_benign___1", "binary (0/1)", "Benign diagnosis: category 1"),
    ("med_benign___2", "binary (0/1)", "Benign diagnosis: category 2"),
    ("med_benign___3", "binary (0/1)", "Benign diagnosis: category 3"),
    ("med_benign___4", "binary (0/1)", "Benign diagnosis: category 4"),
    ("med_benign___5", "binary (0/1)", "Benign diagnosis: category 5"),
    ("med_benign___6", "binary (0/1)", "Benign diagnosis: category 6"),
    ("mi_subtype_iih", "binary (0/1)", "MI subtype flag: idiopathic intracranial hypertension"),
    ("mi_subtype_cerebellar_ataxia", "binary (0/1)", "MI subtype flag: cerebellar ataxia"),
    ("icu_any", "binary (0/1)", "ICU admission (harmonized across years)"),
    ("admit_any", "binary (0/1)", "Any hospital admission (harmonized across years)"),
    # Admission / Consult
    ("admit", "binary (0/1)", "Admitted from ED"),
    ("admit_location___1", "binary (0/1)", "Admission location: category 1"),
    ("admit_location___2", "binary (0/1)", "Admission location: category 2"),
    ("admit_location___3", "binary (0/1)", "Admission location: category 3"),
    ("admit_location___4", "binary (0/1)", "Admission location: category 4"),
    ("admit_location___5", "binary (0/1)", "Admission location: category 5"),
    ("consult", "binary (0/1)", "Consultation obtained"),
    ("consult_specify___1", "binary (0/1)", "Consultation: category 1"),
    ("consult_specify___2", "binary (0/1)", "Consultation: category 2"),
    ("consult_specify___3", "binary (0/1)", "Consultation: category 3"),
    ("consult_specify___4", "binary (0/1)", "Consultation: category 4"),
    ("consult_specify___5", "binary (0/1)", "Consultation: category 5"),
    ("consult_specify___6", "binary (0/1)", "Consultation: category 6"),
    ("consult_specify___7", "binary (0/1)", "Consultation: category 7"),
    ("consult_specify___8", "binary (0/1)", "Consultation: category 8"),
    ("discharge_refer", "binary (0/1)", "Referred on discharge"),
    ("discharge_refer_specify___1", "binary (0/1)", "Discharge referral: category 1"),
    ("discharge_refer_specify___2", "binary (0/1)", "Discharge referral: category 2"),
    ("discharge_refer_specify___3", "binary (0/1)", "Discharge referral: category 3"),
    ("discharge_refer_specify___4", "binary (0/1)", "Discharge referral: category 4"),
    ("discharge_refer_specify___5", "binary (0/1)", "Discharge referral: category 5"),
    # Seizure
    ("seizure_number", "integer", "Number of seizures"),
    ("seizure_duration", "float", "Seizure duration (minutes)"),
    ("seizure_treatment___1", "binary (0/1)", "Seizure treatment: category 1"),
    ("seizure_treatment___2", "binary (0/1)", "Seizure treatment: category 2"),
    ("seizure_treatment___3", "binary (0/1)", "Seizure treatment: category 3"),
    ("seizure_treatment___4", "binary (0/1)", "Seizure treatment: category 4"),
    ("epilepsy_describe___1", "binary (0/1)", "Epilepsy description: category 1"),
    ("epilepsy_describe___2", "binary (0/1)", "Epilepsy description: category 2"),
    ("epilepsy_describe___3", "binary (0/1)", "Epilepsy description: category 3"),
    ("febrile_describe___1", "binary (0/1)", "Febrile seizure description: category 1"),
    ("febrile_describe___2", "binary (0/1)", "Febrile seizure description: category 2"),
    ("febrile_describe___3", "binary (0/1)", "Febrile seizure description: category 3"),
    ("febrile_describe___4", "binary (0/1)", "Febrile seizure description: category 4"),
    # Diagnosis flags
    ("diagnosis", "categorical", "ED diagnosis category"),
    ("diff_dx", "binary (0/1)", "Differential diagnosis considered"),
]


def write_data_dictionary():
    """Write data dictionary CSV."""
    dd = pd.DataFrame(DICTIONARY_ENTRIES, columns=["variable", "type", "description"])
    dd.to_csv(SHARING / "DATA_DICTIONARY.csv", index=False)
    _print(f"\nData dictionary: {len(dd)} variables → {SHARING / 'DATA_DICTIONARY.csv'}")


# ===================================================================
# 5. README
# ===================================================================

README_TEXT = """\
# Brain AttACKs De-Identified Dataset

## Overview

De-identified dataset from the Brain AttACKs study: a two-year prospective
cohort of pediatric patients presenting to the emergency department with
acute neurological symptoms (N=2,153). This dataset supports replication
of all analyses in the accompanying manuscript.

## Files

| File | Description |
|------|-------------|
| `brain_attacks_deidentified.csv` | De-identified analytic dataset |
| `DATA_DICTIONARY.csv` | Variable names, types, and descriptions |
| `README.md` | This file |

## Dataset Summary

- **Rows:** 2,153
- **Columns:** 156
- **Primary outcome:** `outcome_med_important` — binary indicator of medically important brain attack (147 events, 6.8% prevalence)
- **Temporal split:** `year` (1 = derivation, 2 = temporal validation)

## De-Identification

The following steps were applied to protect patient privacy (script: `scripts/14-deidentify.py`, seed=42):

1. **Record IDs removed** — `record_id` replaced with randomly shuffled sequential `study_id` (1–2,153); no link back to source REDCap records
2. **Dates shifted** — each record offset by a random ±180-day shift (excluding zero offset; zero same-record date matches verified)
3. **Date of birth removed** — continuous `age` (years, rounded to 1 decimal) retained
4. **Time of presentation removed** — `time` (hour) dropped; quarter of year (`time_year`) retained for seasonality
5. **Age capped** — extreme values capped at 0.1–18.0 years (56 neonatal ages capped at lower bound)
6. **Free-text fields removed** — all narrative/specify fields excluded (15+ fields including `diagnosis_specify`, `symptom_other`, `prev_diagnoses_other`, medication free-text); coded checkbox equivalents retained
7. **Rare diagnoses suppressed** — MI subtypes with ≤3 cases merged into "Other MI" (`med_imp_dx___11`):
   - `med_imp_dx___3` (vascular malformation): 2 cases
   - `med_imp_dx___5` (venous thrombosis): 3 cases
   - `med_imp_dx___10` (inborn error of metabolism): 1 case
8. **Row order shuffled** — no correspondence to chronological or original ordering
9. **Administrative fields removed** — REDCap form status (`brain_attacks_crf_complete`), review flags (`lr_review`), exclusion criteria already applied (`related`, `see_doctor`, `trauma`), QC fields (`dx_discrepancy`, `benign_review_flag`, `benign_suggested_mi`), LLM intermediates (`llm_dx_class`, `llm_dx_code`, `llm_dx_confidence`)

**Result:** 133 columns removed, 154 retained + `study_id` + `visit_date_shifted` = 156 total.

## Variable Groups

| Group | Count | Examples |
|-------|------:|---------|
| Demographics | 7 | `study_id`, `age`, `gender`, `year`, `visit_date_shifted` |
| Vitals / Triage | 5 | `temp`, `hr`, `sys_bp`, `dias_bp`, `ctas` |
| Transport | 2 | `ems`, `ems_meds` |
| Symptoms | 12 | `sx_focal_weakness`, `sx_seizure`, `sx_any_headache` |
| Other symptom categories | 18 | `other_pain`, `other_syncope`, `other_cardiac` |
| Exam findings | 3 | `exam_papilledema`, `exam_meningismus`, `exam_altered_loc` |
| Headache features | 15 | `ha_nausea`, `ha_maximal_onset`, `ha_frontal` |
| Prior history | 20 | `previous_diagnoses___1`–`___7`, `comorbid_*` (12) |
| Investigations | 12 | `inv_ct`, `inv_mri`, `any_neuroimaging`, `results` |
| Outcomes / Diagnoses | 24 | `outcome_med_important`, `med_imp_dx___1`–`___13`, `icu_any` |
| Admission / Consult | 20 | `admit`, `consult_specify___1`–`___8`, `discharge_refer` |
| Seizure-specific | 15 | `seizure_number`, `seizure_duration`, `febrile_describe___*` |
| Diagnosis flags | 2 | `diagnosis`, `diff_dx` |

## Analysis Replication

All 55 features used in the predictive model (`get_feature_names()` in analysis scripts 03–13) are present in this dataset. To verify:

```python
import pandas as pd
df = pd.read_csv("brain_attacks_deidentified.csv")

# 55 model features
FEATURES = [
    "sx_focal_weakness", "sx_sensory", "sx_speech", "sx_visual",
    "sx_eye_movement", "sx_altered_consciousness", "sx_collapse",
    "sx_seizure", "sx_ataxia", "sx_any_headache", "sx_fever",
    "other_pain", "other_movement", "other_psych", "other_gi",
    "other_cranial_nerve", "other_inflammation", "other_respiratory",
    "other_infection", "other_constitutional", "other_urinary",
    "other_dermatological", "other_hemorrhage", "other_syncope",
    "other_cardiac", "other_gyn", "other_ophthalmology",
    "other_sleep", "other_hematological",
    "exam_papilledema", "exam_meningismus", "exam_altered_loc",
    "ha_nausea", "ha_vomiting", "ha_photophobia", "ha_phonophobia",
    "ha_maximal_onset", "ha_progressive_days", "ha_progressive_months",
    "ha_non_progressive", "ha_abrupt_onset", "ha_gradual_onset",
    "ha_bilateral", "ha_unilateral", "ha_frontal", "ha_temporal", "ha_occipital",
    "age", "gender", "ems", "ctas", "temp", "hr",
    "any_prev_dx", "age_under_12",
]

assert all(f in df.columns for f in FEATURES), "Missing features"
assert len(df) == 2153
assert df["outcome_med_important"].sum() == 147
print("All checks passed")
```

## Citation

[To be added upon publication]

## License

This dataset is shared under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
"""


def write_readme():
    """Write README.md."""
    with open(SHARING / "README.md", "w") as f:
        f.write(README_TEXT)
    _print(f"README → {SHARING / 'README.md'}")


# ===================================================================
# 6. MAIN
# ===================================================================

def main():
    _print("=" * 60)
    _print("14-deidentify.py — Brain AttACKs De-Identification")
    _print("=" * 60)

    df = load_data()
    out = deidentify(df)

    # Write outputs
    out.to_csv(SHARING / "brain_attacks_deidentified.csv", index=False)
    _print(f"\nDataset → {SHARING / 'brain_attacks_deidentified.csv'}")

    write_data_dictionary()
    write_readme()

    # --- Final verification summary ---
    _print(f"\n{'='*60}")
    _print("VERIFICATION SUMMARY")
    _print(f"{'='*60}")
    _print(f"Rows: {len(out)} (expected 2,153)")
    _print(f"Columns: {len(out.columns)}")
    _print(f"study_id range: {out['study_id'].min()}–{out['study_id'].max()}")

    # Check features for analysis replication
    model_features = [
        "sx_focal_weakness", "sx_sensory", "sx_speech", "sx_visual",
        "sx_eye_movement", "sx_altered_consciousness", "sx_collapse",
        "sx_seizure", "sx_ataxia", "sx_any_headache", "sx_fever",
        "other_pain", "other_movement", "other_psych", "other_gi",
        "other_cranial_nerve", "other_inflammation", "other_respiratory",
        "other_infection", "other_constitutional", "other_urinary",
        "other_dermatological", "other_hemorrhage", "other_syncope",
        "other_cardiac", "other_gyn", "other_ophthalmology",
        "other_sleep", "other_hematological",
        "exam_papilledema", "exam_meningismus", "exam_altered_loc",
        "ha_nausea", "ha_vomiting", "ha_photophobia", "ha_phonophobia",
        "ha_maximal_onset", "ha_progressive_days", "ha_progressive_months",
        "ha_non_progressive", "ha_abrupt_onset", "ha_gradual_onset",
        "ha_bilateral", "ha_unilateral", "ha_frontal", "ha_temporal", "ha_occipital",
        "age", "gender", "ems", "ctas", "temp", "hr",
        "any_prev_dx", "age_under_12",
    ]
    present = [f for f in model_features if f in out.columns]
    missing_feats = [f for f in model_features if f not in out.columns]
    _print(f"Model features present: {len(present)}/{len(model_features)}")
    if missing_feats:
        _print(f"  Missing: {missing_feats}")

    # Outcome check
    if "outcome_med_important" in out.columns:
        n_events = pd.to_numeric(out["outcome_med_important"], errors="coerce").sum()
        _print(f"outcome_med_important events: {int(n_events)} (expected 147)")

    # Date shift verification (same-record match already asserted above)
    if "visit_date_shifted" in out.columns:
        _print(f"Date shift verified: zero same-record matches ✓")

    _print(f"\nDone.")


if __name__ == "__main__":
    main()
