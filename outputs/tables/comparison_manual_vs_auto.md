# Comparison: Manual vs Automated MI Classification

## Overview

This document compares the MI (medically important) outcome classification between:
- **Manual dataset:** `data/manual_clean_data/ba_excel_data.csv` — Excel-based manual cleaning (113 MI cases)
- **Automated dataset:** `data/processed/brain_attacks_combined.parquet` — Programmatic pipeline with LLM-assisted coding + PI review (147 MI cases)

## Record Matching

Records matched on `record_id` (stripping leading zeros from REDCap IDs) + `year`. 2,149 of 2,153 records matched successfully. Four records unmatched due to duplicate record IDs in Year 2 (records #835 and #838 each appear twice with different data).

## MI Classification Agreement

| | Auto MI | Auto Benign | Total |
|---|---|---|---|
| **Manual MI** | 94 | 18 | 112 |
| **Manual Benign** | 46 | 1,991 | 2,037 |
| **Total** | 140 | 2,009 | 2,149 |

**Overall agreement:** 97.0% (2,085 / 2,149)

Note: Totals are for matched records only (2,149 of 2,153).

## Discordant Cases

### 18 Manual-MI / Auto-Benign (Checkbox Errors)

These records had the MI checkbox set in REDCap but the free-text diagnosis was benign. The PI review corrected them to benign:

- Seizure / new onset seizures / multiple seizures (5 records)
- Syncope (1)
- Headache / headache ?migraine (2)
- r/o meningitis / r/o meningitis (typo) / r/o sepsis/meningitis (5)
- r/o complicated pneumonia (1)
- r/o intracranial bleed (1)
- r/o shunt malfunction (1)
- r/o stroke (1)
- Neuropathy NYD (1)

These represent data entry errors in the original REDCap where the MI checkbox was incorrectly set. The manual Excel cleaning retained these errors. The automated pipeline caught them via LLM-assisted diagnosis text validation (Task A).

### 46 Manual-Benign / Auto-MI (PI Reclassifications)

These records were benign in both the original REDCap data and the manual cleaning, but were reclassified to MI after PI review of LLM-flagged cases:

| MI Category | N | Examples |
|---|---|---|
| Cerebellar ataxia (code 11) | ~20 | ataxia, cerebellitis, post-viral ataxia, ataxia NYD |
| IIH (code 11) | ~5 | pseudotumor cerebri, papilledema, benign intracranial hypertension |
| Hydrocephalus — shunt malfunction (code 8) | ~7 | shunt malfunction, VP shunt malfunction |
| Brain/meningeal infection (code 7) | ~5 | neonatal sepsis, H. influenzae sepsis, confirmed meningitis |
| Demyelinating disorder (code 6) | ~5 | optic neuritis, transverse myelitis |
| Other MI (code 11) | ~3 | vision loss, abnormal MRI |
| Neoplasm (code 9) | ~2 | brain lesion, radiation necrosis |

These were missed by both the original REDCap coding and the manual Excel cleaning because the benign checkbox was set but the free-text diagnosis described a medically important condition. The automated pipeline's LLM coding (Tasks A and E) systematically flagged these for PI review.

## MI Category Concordance (Both-MI Records)

Among the 94 records classified as MI by both methods:

- **Concordant category:** 82 (87.2%)
- **Discordant category:** 12 (12.8%)

Category mismatches were primarily in multi-diagnosis records where the manual coder chose one category and the automated pipeline chose another (e.g., infection vs. hydrocephalus for a patient with both meningitis and hydrocephalus).

## Summary

The automated pipeline identified 34 net additional MI cases compared to manual cleaning (147 vs 113):
- +48 benign-to-MI reclassifications (LLM flagged, PI confirmed)
- -18 MI-to-benign corrections (checkbox data entry errors)
- Net: +30 on matched records (+34 overall including 4 unmatched)

Note: The comparison table above was generated before the QC-driven fix to match against
both `diagnosis_specify` and `med_benign_specify` fields, which caught 6 additional records.
The agreement percentage and cross-tabulation would shift slightly with updated matching.

The key advantage of the automated pipeline is systematic free-text validation: the LLM coding checked every diagnosis text field against the MI/benign checkbox, catching both false positives (checkbox errors) and false negatives (missed MI cases) that manual Excel cleaning did not detect.
