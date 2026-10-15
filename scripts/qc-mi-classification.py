"""
qc-mi-classification.py — Line-by-line quality review of MI classification

Loads the analytic dataset and performs exhaustive checks on every record's
medically important (MI) classification, producing a detailed QC report.

Outputs:
    outputs/tables/qc_mi_report.csv        — Per-record flags for all 2,153 records
    outputs/tables/qc_mi_summary.md        — Summary of all findings
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import Counter

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
PROCESSED = DATA / "processed"
OUTPUTS = PROJECT / "outputs" / "tables"

import json
PI_OVERRIDES = json.loads((DATA / "metadata" / "pi-review-overrides.json").read_text())

# ===================================================================
# LOAD DATA
# ===================================================================

df = pd.read_parquet(PROCESSED / "brain_attacks_combined.parquet")
print(f"Loaded: {len(df)} records, {len(df.columns)} columns")

# Load PI review decisions for cross-reference
pi_decisions = pd.read_csv(PROCESSED / "llm_coded" / "pi-review-decisions.csv")

# ===================================================================
# SETUP: QC FLAGS PER RECORD
# ===================================================================

# Every record gets a row in the QC output
qc = pd.DataFrame({
    "record_id": df["record_id"],
    "year": df["year"],
    "outcome_mi": df["outcome_med_important"],
    "diagnosis_specify": df["diagnosis_specify"].fillna(""),
})

# We'll accumulate flags as columns
flags = []  # list of (flag_name, description, series_of_booleans)

# ===================================================================
# CHECK 1: outcome_med_important vs med_imp_dx checkboxes
# ===================================================================

mi_cols = [f"med_imp_dx___{i}" for i in range(1, 14)]
existing_mi = [c for c in mi_cols if c in df.columns]

any_mi_checkbox = (df[existing_mi].max(axis=1) == 1)
outcome_flag = df["outcome_med_important"] == 1

# MI=1 but no checkbox
mismatch_mi_no_checkbox = outcome_flag & ~any_mi_checkbox
flags.append(("mi_no_checkbox", "MI=1 but no med_imp_dx checkbox is set", mismatch_mi_no_checkbox))

# Checkbox set but MI=0
mismatch_checkbox_no_mi = ~outcome_flag & any_mi_checkbox
flags.append(("checkbox_no_mi", "med_imp_dx checkbox set but outcome_med_important=0", mismatch_checkbox_no_mi))

# ===================================================================
# CHECK 2: Multiple MI category codes
# ===================================================================

n_mi_codes = df[existing_mi].sum(axis=1)
multiple_mi = (n_mi_codes > 1) & outcome_flag
flags.append(("multiple_mi_codes", "More than one MI category checkbox (valid comorbid diagnoses)", multiple_mi))

# ===================================================================
# CHECK 3: MI cases without diagnosis text
# ===================================================================

dx_text = df["diagnosis_specify"].fillna("").str.strip()
mi_no_dx = outcome_flag & (dx_text == "")
flags.append(("mi_no_diagnosis_text", "MI=1 but diagnosis_specify is empty", mi_no_dx))

# ===================================================================
# CHECK 4: MI sub-type flags consistency
# ===================================================================

# IIH sub-type should be on MI-12 records (elevated from code 11)
if "mi_subtype_iih" in df.columns:
    iih_flag = df["mi_subtype_iih"] == 1
    iih_col = "med_imp_dx___12" if "med_imp_dx___12" in df.columns else "med_imp_dx___11"
    iih_not_code = iih_flag & ~(df[iih_col] == 1)
    flags.append(("iih_not_mi12", f"IIH sub-type flagged but {iih_col} != 1", iih_not_code))

    iih_not_mi = iih_flag & ~outcome_flag
    flags.append(("iih_not_mi", "IIH sub-type flagged but outcome_med_important=0", iih_not_mi))

# Cerebellar ataxia sub-type should be on MI-13 records (elevated from code 11)
if "mi_subtype_cerebellar_ataxia" in df.columns:
    ca_flag = df["mi_subtype_cerebellar_ataxia"] == 1
    ca_col = "med_imp_dx___13" if "med_imp_dx___13" in df.columns else "med_imp_dx___11"
    ca_not_code = ca_flag & ~(df[ca_col] == 1)
    flags.append(("cerebellar_not_mi13", f"Cerebellar ataxia sub-type but {ca_col} != 1", ca_not_code))

    ca_not_mi = ca_flag & ~outcome_flag
    flags.append(("cerebellar_not_mi", "Cerebellar ataxia sub-type but outcome_med_important=0", ca_not_mi))

# MI-11 without any sub-type — now that IIH/CA are elevated, code 11 should have no sub-types
if "mi_subtype_iih" in df.columns and "mi_subtype_cerebellar_ataxia" in df.columns:
    mi_11_mask = (df["med_imp_dx___11"] == 1) & outcome_flag
    no_subtype = mi_11_mask & (df["mi_subtype_iih"] == 0) & (df["mi_subtype_cerebellar_ataxia"] == 0)
    flags.append(("mi11_no_subtype", "MI code 11 (Other) — expected after IIH/CA elevation", no_subtype))

# ===================================================================
# CHECK 5: Diagnosis text plausibility for each MI category
# ===================================================================

# Records reviewed and accepted by PI despite diagnosis-code mismatch.
# key = record_id, value = set of MI codes confirmed correct.
# Record-level overrides (with PI notes) live in data/metadata/pi-review-overrides.json (git-ignored).
reviewed_dx_mismatches = {rid: set(codes) for rid, codes in PI_OVERRIDES["reviewed_dx_mismatches"].items()}

# Define expected diagnosis text patterns for each MI code
mi_dx_patterns = {
    1: r"stroke|dissection|ischemi|infarct|tia|cerebrovascular|\bcva\b",
    2: r"ich|hemorrh|haemorrh|haemmorrh|bleed|hematoma|haematoma",
    3: r"vascular malform|avm|cavernoma|aneurysm|arteriovenous",
    5: r"thrombos|svt|sinus thromb|venous",
    6: r"demyelin|optic neuritis|transverse myelitis|ms |multiple sclerosis|adem|nmo|neuromyelitis|encephalomyelitis|visual loss|visual disturbance",
    7: r"mening|encephalit|sepsis|infect|abscess|abcess|h influenza|cerebellitis",
    8: r"hydrocephalus|shunt|vp shunt|csf|ventriculomegaly",
    9: r"neoplas|tumor|tumour|mass|lesion|glioma|medulloblast|craniopharyngioma|radiation necrosis|brain lesion|adenoma|cancer",
    10: r"metaboli|inborn|regression|storage disease|leukodystrop",
    11: r"vision loss|abnormal mri|shunt|cerebellitis|opsoclonus|tumou?r|glioma|cerebellar|nerve palsy|myelitis|ataxia",
    12: r"iih|intracranial hypertension|pseudotumor|papilledema|b\.i\.h\.",
    13: r"ataxia|cerebellitis|post.?viral ataxia",
}

dx_lower = dx_text.str.lower()

for code, pattern in mi_dx_patterns.items():
    col = f"med_imp_dx___{code}"
    if col in df.columns:
        has_code = (df[col] == 1)
        text_matches = dx_lower.str.contains(pattern, na=False, regex=True)
        # MI code set but diagnosis text doesn't match expected pattern
        code_no_match = has_code & ~text_matches & (dx_lower != "")
        # Exclude PI-reviewed records for this specific code
        for rid, reviewed_codes in reviewed_dx_mismatches.items():
            if code in reviewed_codes:
                code_no_match = code_no_match & (df["record_id"] != rid)
        mi_labels = {
            1: "Stroke", 2: "ICH", 3: "Vascular malformation", 5: "Venous thrombosis",
            6: "Demyelinating", 7: "Infection", 8: "Hydrocephalus", 9: "Neoplasm",
            10: "Inborn error", 11: "Other MI", 12: "IIH", 13: "Cerebellar ataxia"
        }
        label = mi_labels.get(code, f"Code {code}")
        flags.append((
            f"dx_mismatch_code{code}",
            f"MI code {code} ({label}) but diagnosis text doesn't match expected pattern",
            code_no_match
        ))

# ===================================================================
# CHECK 6: Benign cases with suspicious diagnosis text
# ===================================================================

# Diagnosis terms that should raise concern if classified benign
mi_suspicious_terms = [
    (r"\bstroke\b", "stroke"),
    (r"\binfarct", "infarct"),
    (r"\bdissection\b", "dissection"),
    (r"\bich\b", "ICH"),
    (r"\bhemorrhag", "hemorrhage"),
    (r"\bhaemorrhag", "haemorrhage"),
    (r"\btumou?r\b", "tumor/tumour"),
    (r"\bneoplas", "neoplasm"),
    (r"\bglioma\b", "glioma"),
    (r"\bmedulloblast", "medulloblastoma"),
    (r"\bdemyelin", "demyelinating"),
    (r"\boptic neuritis\b", "optic neuritis"),
    (r"\btransverse myelitis\b", "transverse myelitis"),
    (r"\bencephalitis\b", "encephalitis"),
    (r"\bmeningitis\b(?!\s*prophylaxis)", "meningitis (not prophylaxis)"),
    (r"\babscess\b", "abscess"),
    (r"\bhydrocephalus\b", "hydrocephalus"),
    (r"\bshunt malfunction\b", "shunt malfunction"),
    (r"\bvp shunt malfunction\b", "VP shunt malfunction"),
    (r"\bpseudotumor\b", "pseudotumor"),
    (r"\biih\b", "IIH"),
    (r"\bpapilledema\b", "papilledema"),
    (r"\bintracranial hypertension\b", "intracranial hypertension"),
    (r"\bcerebellar ataxia\b", "cerebellar ataxia"),
    (r"\bcerebellitis\b", "cerebellitis"),
    (r"\bvenous.*thromb", "venous thrombosis"),
    (r"\bsinus thromb", "sinus thrombosis"),
    (r"\bavm\b", "AVM"),
    (r"\barteriovenous", "arteriovenous malformation"),
    (r"\bcraniopharyngioma\b", "craniopharyngioma"),
    (r"\bradiation necrosis\b", "radiation necrosis"),
]

benign_mask = df["outcome_med_important"] == 0
suspicious_benign = pd.Series(False, index=df.index)
suspicious_terms_found = pd.Series("", index=df.index)

for pattern, label in mi_suspicious_terms:
    match = dx_lower.str.contains(pattern, na=False, regex=True) & benign_mask
    # Exclude r/o (rule-out) diagnoses — these were deliberately kept benign
    # Also exclude known benign mimics: pseudo-papilledema, subconjunctival hemorrhage
    ro_mask = dx_lower.str.contains(r"r/o|rule.?out|query|\?", na=False, regex=True)
    benign_mimic_mask = dx_lower.str.contains(r"pseudo|subconjunctival", na=False, regex=True)
    # Still flag if suspicious even with r/o prefix, but mark separately
    flagged = match & ~ro_mask & ~benign_mimic_mask
    suspicious_benign = suspicious_benign | flagged
    suspicious_terms_found = suspicious_terms_found.where(
        ~flagged, suspicious_terms_found + label + "; "
    )

flags.append(("benign_suspicious_dx",
              "Benign but diagnosis text contains MI-suggestive terms (not r/o)",
              suspicious_benign))

# Also flag benign r/o cases for informational completeness
ro_suspicious = pd.Series(False, index=df.index)
for pattern, label in mi_suspicious_terms:
    match = dx_lower.str.contains(pattern, na=False, regex=True) & benign_mask
    ro_match = dx_lower.str.contains(r"r/o|rule.?out|query|\?", na=False, regex=True)
    flagged_ro = match & ro_match
    ro_suspicious = ro_suspicious | flagged_ro

flags.append(("benign_ro_suspicious_dx",
              "Benign with r/o prefix but diagnosis mentions MI condition (kept benign per PI rule-out principle)",
              ro_suspicious))

# ===================================================================
# CHECK 7: PI review decision coverage
# ===================================================================

# Check that all reclassify decisions actually matched records
reclassify_decisions = pi_decisions[pi_decisions["decision"].str.startswith("reclassify_mi_")]
unmatched_reclassify = []

for _, row in reclassify_decisions.iterrows():
    dx = str(row["diagnosis_text"]).strip().lower()
    expected_n = int(row["n_records"])
    mi_code = int(row["mi_code"])

    # These should now be MI — check both diagnosis_specify and med_benign_specify
    mbs_lower = df["med_benign_specify"].fillna("").str.strip().str.lower()
    mask = ((dx_lower == dx) | (mbs_lower == dx)) & (df["outcome_med_important"] == 1)
    actual_n = mask.sum()

    if actual_n < expected_n:
        unmatched_reclassify.append({
            "decision": row["decision"],
            "diagnosis_text": dx,
            "mi_code": mi_code,
            "expected_n": expected_n,
            "actual_mi_n": actual_n,
        })

# Check keep_benign decisions
keep_benign_task_a = pi_decisions[
    (pi_decisions["decision"] == "keep_benign") &
    (pi_decisions["source_task"] == "A")
]
unmatched_benign = []

for _, row in keep_benign_task_a.iterrows():
    dx = str(row["diagnosis_text"]).strip().lower()
    expected_n = int(row["n_records"])

    # These should now be benign
    mask = (dx_lower == dx) & (df["outcome_med_important"] == 0)
    actual_n = mask.sum()

    # For keep_benign Task A, these were originally MI and should now be benign
    # We can't perfectly verify without original state, but at least check they exist as benign
    if actual_n < expected_n:
        unmatched_benign.append({
            "decision": "keep_benign",
            "diagnosis_text": dx,
            "expected_n": expected_n,
            "actual_benign_n": actual_n,
        })

# ===================================================================
# CHECK 8: med_benign fields vs MI status
# ===================================================================

benign_cols = [f"med_benign___{i}" for i in range(1, 7)]
existing_benign = [c for c in benign_cols if c in df.columns]

if existing_benign:
    any_benign_checkbox = (df[existing_benign].max(axis=1) == 1)
    # Both MI and benign checkboxes set
    both_mi_benign = outcome_flag & any_benign_checkbox
    flags.append(("both_mi_and_benign", "Both MI and benign checkboxes are set", both_mi_benign))

    # Neither MI nor benign checkbox
    neither = ~outcome_flag & ~any_benign_checkbox
    # But this might be expected for some records, check diagnosis field
    has_diagnosis = (df["diagnosis"] == 1) if "diagnosis" in df.columns else pd.Series(False, index=df.index)
    neither_with_dx = neither & has_diagnosis
    flags.append(("neither_mi_nor_benign", "Neither MI nor benign checkbox set, but has diagnosis", neither_with_dx))

# ===================================================================
# CHECK 9: Year-specific checks
# ===================================================================

# Year 1 should not have med_imp_dx___11 (it was added for Year 2)
# After PI review, some Year 1 records may now have code 11
y1_mi11 = (df["year"] == "1") & (df["med_imp_dx___11"] == 1)
flags.append(("y1_mi_code11", "Year 1 record with MI code 11 (added via PI review)", y1_mi11))

# ===================================================================
# CHECK 10: Duplicate diagnosis text among MI cases
# ===================================================================

mi_records = df[outcome_flag].copy()
mi_records["dx_lower"] = dx_lower[outcome_flag]
dx_counts = mi_records["dx_lower"].value_counts()
# Flag MI records with very common diagnosis text (might indicate coding issues)
common_dx = dx_counts[dx_counts >= 5].index
mi_common_dx = outcome_flag & dx_lower.isin(common_dx)
flags.append(("mi_common_dx", "MI case with diagnosis text shared by 5+ other MI cases", mi_common_dx))

# ===================================================================
# CHECK 11: Ataxia-specific check (largest reclassified group)
# ===================================================================

# All records with "ataxia" in diagnosis — verify classification is correct
ataxia_in_dx = dx_lower.str.contains(r"\bataxia\b", na=False, regex=True)
ataxia_benign = ataxia_in_dx & ~outcome_flag
# After PI review, most ataxia should be MI unless it's a rule-out
ataxia_benign_not_ro = ataxia_benign & ~dx_lower.str.contains(r"r/o|rule.?out|\?", na=False, regex=True)
flags.append(("ataxia_benign_not_ro",
              "Diagnosis contains 'ataxia' but classified benign (not rule-out)",
              ataxia_benign_not_ro))

# ===================================================================
# CHECK 12: CTAS vs MI (clinical plausibility)
# ===================================================================

if "ctas" in df.columns:
    # CTAS 4-5 (less urgent) with MI is unusual but not impossible
    low_acuity_mi = outcome_flag & (df["ctas"] >= 4)
    flags.append(("low_acuity_mi", "MI case with CTAS 4 or 5 (less urgent)", low_acuity_mi))

# ===================================================================
# CHECK 13: Exam findings consistency
# ===================================================================

# Papilledema on exam but no MI (should trigger IIH consideration)
# Exclude cases with clear clinical explanations for benign classification:
#   - "pseudo papilledema" / "pseudo-papilledema" — not true papilledema
#   - post-LP intracranial hypotension — opposite of IIH mechanism
# PI-reviewed and kept benign: (record_id, year) pairs in data/metadata/pi-review-overrides.json.
reviewed_papilledema = {tuple(pair) for pair in PI_OVERRIDES["reviewed_papilledema"]}
if "exam_papilledema" in df.columns:
    papilledema_benign_raw = (df["exam_papilledema"] == 1) & ~outcome_flag
    pap_explained = dx_lower.str.contains(
        r"pseudo.?papilledema|intracranial hypotension|post.?lumbar puncture|post.?lp",
        na=False, regex=True,
    )
    pap_reviewed = pd.Series(False, index=df.index)
    for rid, yr in reviewed_papilledema:
        pap_reviewed = pap_reviewed | ((df["record_id"] == rid) & (df["year"] == yr))
    papilledema_benign = papilledema_benign_raw & ~pap_explained & ~pap_reviewed
    flags.append(("papilledema_benign", "Papilledema on exam but classified benign", papilledema_benign))

# Meningismus on exam but no MI (should trigger infection consideration)
if "exam_meningismus" in df.columns:
    meningismus_benign = (df["exam_meningismus"] == 1) & ~outcome_flag
    flags.append(("meningismus_benign", "Meningismus on exam but classified benign", meningismus_benign))

# ===================================================================
# CHECK 14: Symptom-outcome logical checks
# ===================================================================

# Fever + altered consciousness + MI infection code — check consistency
if "sx_fever" in df.columns and "sx_altered_consciousness" in df.columns:
    fever_aloc_benign = (df["sx_fever"] == 1) & (df["sx_altered_consciousness"] == 1) & ~outcome_flag
    flags.append(("fever_aloc_benign",
                  "Fever + altered consciousness but classified benign",
                  fever_aloc_benign))

# ===================================================================
# COMPILE QC REPORT
# ===================================================================

print(f"\n{'='*60}")
print("MI CLASSIFICATION QUALITY REVIEW")
print(f"{'='*60}")

# Add all flags to qc dataframe
all_flag_names = []
for flag_name, description, flag_series in flags:
    qc[flag_name] = flag_series.astype(int)
    all_flag_names.append(flag_name)

# Count total flags per record
qc["total_flags"] = qc[all_flag_names].sum(axis=1)

# Add MI category code for MI records
for i in range(1, 14):
    col = f"med_imp_dx___{i}"
    if col in df.columns:
        qc[f"code_{i}"] = df[col]

# Add sub-type columns
if "mi_subtype_iih" in df.columns:
    qc["subtype_iih"] = df["mi_subtype_iih"]
if "mi_subtype_cerebellar_ataxia" in df.columns:
    qc["subtype_cerebellar"] = df["mi_subtype_cerebellar_ataxia"]

# Add suspicious terms column
qc["suspicious_terms"] = suspicious_terms_found.str.rstrip("; ")

# ===================================================================
# SUMMARY REPORT
# ===================================================================

summary_lines = []
summary_lines.append("# MI Classification Quality Review\n")
summary_lines.append(f"**Dataset:** `data/processed/brain_attacks_combined.parquet`")
summary_lines.append(f"**Records:** {len(df):,}")
summary_lines.append(f"**MI cases:** {int(df['outcome_med_important'].sum())}")
summary_lines.append(f"**Benign cases:** {int((df['outcome_med_important'] == 0).sum())}")
summary_lines.append("")

# --- Flag summary table ---
summary_lines.append("## Flag Summary\n")
summary_lines.append("| # | Flag | Description | N Flagged | Severity |")
summary_lines.append("|---|------|-------------|-----------|----------|")

severity_map = {
    "mi_no_checkbox": "ERROR",
    "checkbox_no_mi": "ERROR",
    "multiple_mi_codes": "INFO",
    "mi_no_diagnosis_text": "INFO",
    "iih_not_mi12": "ERROR",
    "iih_not_mi": "ERROR",
    "cerebellar_not_mi13": "ERROR",
    "cerebellar_not_mi": "ERROR",
    "mi11_no_subtype": "INFO",
    "benign_suspicious_dx": "REVIEW",
    "benign_ro_suspicious_dx": "INFO",
    "both_mi_and_benign": "WARNING",
    "neither_mi_nor_benign": "INFO",
    "y1_mi_code11": "INFO",
    "mi_common_dx": "INFO",
    "ataxia_benign_not_ro": "REVIEW",
    "low_acuity_mi": "INFO",
    "papilledema_benign": "REVIEW",
    "meningismus_benign": "INFO",
    "fever_aloc_benign": "INFO",
}

error_count = 0
warning_count = 0
review_count = 0

for i, (flag_name, description, flag_series) in enumerate(flags, 1):
    n = flag_series.sum()
    severity = severity_map.get(flag_name, "INFO")
    # Count diagnosis-mismatch flags
    if flag_name.startswith("dx_mismatch"):
        severity = "REVIEW" if n > 0 else "INFO"

    if n > 0:
        if severity == "ERROR": error_count += n
        elif severity == "WARNING": warning_count += n
        elif severity == "REVIEW": review_count += n

    summary_lines.append(f"| {i} | `{flag_name}` | {description} | {n} | {severity} |")

summary_lines.append("")
summary_lines.append(f"**Totals:** {error_count} errors, {warning_count} warnings, {review_count} requiring review")
summary_lines.append("")

# --- Detailed findings ---
summary_lines.append("## Detailed Findings\n")

# Check 1: Internal consistency
summary_lines.append("### 1. Outcome vs Checkbox Consistency\n")
n1a = mismatch_mi_no_checkbox.sum()
n1b = mismatch_checkbox_no_mi.sum()
summary_lines.append(f"- MI=1 but no checkbox: **{n1a}**")
summary_lines.append(f"- Checkbox set but MI=0: **{n1b}**")
if n1a == 0 and n1b == 0:
    summary_lines.append(f"- **PASS**: outcome_med_important is perfectly consistent with med_imp_dx checkboxes")
summary_lines.append("")

# Check 2: Multiple codes
summary_lines.append("### 2. Multiple MI Category Codes\n")
n_mult = multiple_mi.sum()
summary_lines.append(f"- Records with >1 MI code: **{n_mult}**")
if n_mult > 0:
    mult_records = df[multiple_mi][["record_id", "year", "diagnosis_specify"] + existing_mi]
    for _, row in mult_records.iterrows():
        codes = [str(i) for i in range(1, 14) if row.get(f"med_imp_dx___{i}", 0) == 1]
        summary_lines.append(f"  - {row['record_id']} (Y{row['year']}): codes [{', '.join(codes)}] — \"{row['diagnosis_specify']}\"")
summary_lines.append("")

# Check 3: MI no diagnosis text
summary_lines.append("### 3. MI Cases Without Diagnosis Text\n")
n_no_dx = mi_no_dx.sum()
summary_lines.append(f"- MI cases with empty diagnosis_specify: **{n_no_dx}**")
if n_no_dx > 0:
    no_dx_records = df[mi_no_dx][["record_id", "year"] + existing_mi]
    for _, row in no_dx_records.iterrows():
        codes = [str(i) for i in range(1, 14) if row.get(f"med_imp_dx___{i}", 0) == 1]
        summary_lines.append(f"  - {row['record_id']} (Y{row['year']}): codes [{', '.join(codes)}]")
summary_lines.append("")

# Check 4: Sub-type consistency
summary_lines.append("### 4. MI Sub-type Flags\n")
if "mi_subtype_iih" in df.columns:
    n_iih = int(df["mi_subtype_iih"].sum())
    summary_lines.append(f"- IIH sub-type: **{n_iih}** records")
if "mi_subtype_cerebellar_ataxia" in df.columns:
    n_ca = int(df["mi_subtype_cerebellar_ataxia"].sum())
    summary_lines.append(f"- Cerebellar ataxia sub-type: **{n_ca}** records")
n_mi11 = int((df["med_imp_dx___11"] == 1).sum())
n_untyped = int(no_subtype.sum()) if "mi_subtype_iih" in df.columns else 0
summary_lines.append(f"- MI code 11 total: **{n_mi11}** (untyped: {n_untyped})")
if n_untyped > 0:
    untyped_records = df[no_subtype][["record_id", "year", "diagnosis_specify"]]
    for _, row in untyped_records.iterrows():
        summary_lines.append(f"  - {row['record_id']} (Y{row['year']}): \"{row['diagnosis_specify']}\"")
summary_lines.append("")

# Check 5: Diagnosis text plausibility
summary_lines.append("### 5. Diagnosis Text vs MI Code Plausibility\n")
for flag_name, description, flag_series in flags:
    if flag_name.startswith("dx_mismatch") and flag_series.sum() > 0:
        code = int(flag_name.split("code")[1])
        col = f"med_imp_dx___{code}"
        mismatch_records = df[flag_series][["record_id", "year", "diagnosis_specify"]]
        summary_lines.append(f"**Code {code}** — {flag_series.sum()} records with non-matching diagnosis text:")
        for _, row in mismatch_records.iterrows():
            summary_lines.append(f"  - {row['record_id']} (Y{row['year']}): \"{row['diagnosis_specify']}\"")
        summary_lines.append("")

# Check 6: Suspicious benign diagnoses
summary_lines.append("### 6. Benign Cases with Suspicious Diagnosis Text\n")
n_susp = suspicious_benign.sum()
summary_lines.append(f"- Benign with MI-suggestive diagnosis (non-r/o): **{n_susp}**")
if n_susp > 0:
    susp_records = df[suspicious_benign][["record_id", "year", "diagnosis_specify"]].copy()
    susp_records["terms"] = suspicious_terms_found[suspicious_benign].values
    for _, row in susp_records.iterrows():
        summary_lines.append(f"  - {row['record_id']} (Y{row['year']}): \"{row['diagnosis_specify']}\" [{row['terms'].rstrip('; ')}]")
summary_lines.append("")

n_ro = ro_suspicious.sum()
summary_lines.append(f"- Benign with r/o MI diagnosis (kept benign per PI rule-out principle): **{n_ro}**")
if n_ro > 0:
    ro_records = df[ro_suspicious][["record_id", "year", "diagnosis_specify"]]
    for _, row in ro_records.iterrows():
        summary_lines.append(f"  - {row['record_id']} (Y{row['year']}): \"{row['diagnosis_specify']}\"")
summary_lines.append("")

# Check 7: PI review coverage
summary_lines.append("### 7. PI Review Decision Verification\n")
if unmatched_reclassify:
    summary_lines.append(f"**Unmatched reclassify decisions:** {len(unmatched_reclassify)}")
    for item in unmatched_reclassify:
        summary_lines.append(
            f"  - {item['decision']}: \"{item['diagnosis_text']}\" — expected {item['expected_n']}, "
            f"found {item['actual_mi_n']} MI records")
else:
    summary_lines.append("- All reclassify-to-MI decisions verified: **PASS**")

summary_lines.append("")
if unmatched_benign:
    summary_lines.append(f"**Unmatched keep_benign (Task A) decisions:** {len(unmatched_benign)}")
    for item in unmatched_benign:
        summary_lines.append(
            f"  - \"{item['diagnosis_text']}\" — expected {item['expected_n']}, "
            f"found {item['actual_benign_n']} benign records")
else:
    summary_lines.append("- All keep_benign (Task A) decisions verified: **PASS**")
summary_lines.append("")

# Check 8: Both MI and benign
summary_lines.append("### 8. Both MI and Benign Checkboxes Set\n")
if existing_benign:
    n_both = both_mi_benign.sum()
    summary_lines.append(f"- Records with both MI and benign checkboxes: **{n_both}**")
    if n_both > 0:
        both_records = df[both_mi_benign][["record_id", "year", "diagnosis_specify"]]
        for _, row in both_records.iterrows():
            summary_lines.append(f"  - {row['record_id']} (Y{row['year']}): \"{row['diagnosis_specify']}\"")
summary_lines.append("")

# Check 9: Ataxia
summary_lines.append("### 9. Ataxia Classification Check\n")
n_ataxia_total = ataxia_in_dx.sum()
n_ataxia_mi = (ataxia_in_dx & outcome_flag).sum()
n_ataxia_benign = ataxia_benign.sum()
n_ataxia_benign_not_ro_val = ataxia_benign_not_ro.sum()
summary_lines.append(f"- Records with 'ataxia' in diagnosis: **{n_ataxia_total}**")
summary_lines.append(f"  - MI: {n_ataxia_mi}")
summary_lines.append(f"  - Benign: {n_ataxia_benign} (of which {n_ataxia_benign_not_ro_val} not rule-out)")
if n_ataxia_benign_not_ro_val > 0:
    ataxia_records = df[ataxia_benign_not_ro][["record_id", "year", "diagnosis_specify"]]
    for _, row in ataxia_records.iterrows():
        summary_lines.append(f"    - {row['record_id']} (Y{row['year']}): \"{row['diagnosis_specify']}\"")
summary_lines.append("")

# Check 10: Clinical red flags in benign
summary_lines.append("### 10. Clinical Red Flags in Benign Cases\n")
if "exam_papilledema" in df.columns:
    n_pap = papilledema_benign.sum()
    summary_lines.append(f"- Papilledema on exam but benign: **{n_pap}**")
    if n_pap > 0:
        pap_records = df[papilledema_benign][["record_id", "year", "diagnosis_specify"]]
        for _, row in pap_records.iterrows():
            summary_lines.append(f"  - {row['record_id']} (Y{row['year']}): \"{row['diagnosis_specify']}\"")

if "exam_meningismus" in df.columns:
    n_men = meningismus_benign.sum()
    summary_lines.append(f"- Meningismus on exam but benign: **{n_men}**")

if "ctas" in df.columns:
    n_low = low_acuity_mi.sum()
    summary_lines.append(f"- MI with CTAS 4-5 (low acuity): **{n_low}**")
    if n_low > 0:
        low_records = df[low_acuity_mi][["record_id", "year", "diagnosis_specify", "ctas"]]
        for _, row in low_records.iterrows():
            summary_lines.append(f"  - {row['record_id']} (Y{row['year']}): CTAS {int(row['ctas'])} — \"{row['diagnosis_specify']}\"")

summary_lines.append("")

# Check 11: Year distribution
summary_lines.append("### 11. MI Distribution by Year\n")
mi_by_year = df.groupby("year")["outcome_med_important"].agg(["count", "sum", "mean"])
for yr, row in mi_by_year.iterrows():
    summary_lines.append(f"- Year {yr}: {int(row['sum'])}/{int(row['count'])} MI ({row['mean']:.1%})")
summary_lines.append("")

# MI code by year
summary_lines.append("**MI categories by year:**\n")
summary_lines.append("| Category | Year 1 | Year 2 | Total |")
summary_lines.append("|----------|--------|--------|-------|")
mi_labels = {
    1: "Stroke", 2: "ICH", 3: "Vascular malformation",
    5: "Venous thrombosis", 6: "Demyelinating", 7: "Infection",
    8: "Hydrocephalus", 9: "Neoplasm", 10: "Inborn error",
    11: "Other MI", 12: "IIH", 13: "Cerebellar ataxia",
}
for code, label in sorted(mi_labels.items()):
    col = f"med_imp_dx___{code}"
    if col in df.columns:
        y1_n = int(((df["year"] == "1") & (df[col] == 1)).sum())
        y2_n = int(((df["year"] == "2") & (df[col] == 1)).sum())
        total = y1_n + y2_n
        if total > 0:
            summary_lines.append(f"| {label} | {y1_n} | {y2_n} | {total} |")

summary_lines.append("")

# Final summary
summary_lines.append("## Overall Assessment\n")
n_flagged = int((qc["total_flags"] > 0).sum())
n_clean = int((qc["total_flags"] == 0).sum())
summary_lines.append(f"- Records with zero flags: **{n_clean}** ({n_clean/len(df):.1%})")
summary_lines.append(f"- Records with 1+ flags: **{n_flagged}** ({n_flagged/len(df):.1%})")
summary_lines.append(f"  - Of which MI: {int(((qc['total_flags'] > 0) & (qc['outcome_mi'] == 1)).sum())}")
summary_lines.append(f"  - Of which benign: {int(((qc['total_flags'] > 0) & (qc['outcome_mi'] == 0)).sum())}")
summary_lines.append("")

# Print to console
for line in summary_lines:
    print(line)

# ===================================================================
# SAVE OUTPUTS
# ===================================================================

# Save per-record QC report
qc.to_csv(OUTPUTS / "qc_mi_report.csv", index=False)
print(f"\nPer-record QC report: {OUTPUTS / 'qc_mi_report.csv'}")

# Save summary
with open(OUTPUTS / "qc_mi_summary.md", "w") as f:
    f.write("\n".join(summary_lines))
print(f"Summary report: {OUTPUTS / 'qc_mi_summary.md'}")
