# Brain AttACKs

Prediction of Medically Important Pediatric Brain Attack Based on Presenting Characteristics

**Protocol:** v1.1 (May 19, 2016) | PRO00055264
**Site:** Stollery Children's Hospital, Edmonton, AB

## Locations

| Layer | Path | Access |
|-------|------|--------|
| Documents | `~/Library/CloudStorage/GoogleDrive-lricher@ualberta.ca/My Drive/Research/brain-attacks` | Always |
| Code | This repo | Always (local git) |
| Data | `/Volumes/fomd-richerlab/data/brain_attacks` | VPN required |
| Manuscript | `~/Outputs/manuscripts/brain-attacks` | Always |

## Strategy

- Project ID: `ms_brain_attacks`
- Strategy note: `~/Strategy/projects/ms-brain-attacks.md`

---

## Study Design

Retrospective observational cohort of 2,153 children presenting to the ED with non-traumatic acute focal brain dysfunction ("brain attack") over two years. Data collected via chart review into REDCap using two CRF versions (Year 1: 1,124 records; Year 2: 1,205 records; 176 excluded for trauma).

**Primary outcome:** Medically important brain attack (147 cases, 6.8% prevalence after PI review).

## Data Preparation Pipeline

The pipeline rebuilds the analytic dataset from source REDCap exports through the following steps:

### Step 1: Schema Harmonization

Year 1 and Year 2 used different CRF versions requiring field renaming, checkbox remapping, and structural restructuring. Full mapping documented in:
- `analysis-plan.md` Section 4
- `outputs/tables/tableS3_harmonization_mapping.csv`

### Step 2: LLM-Assisted Free-Text Coding

Six coding tasks classified free-text fields into structured variables using Claude, with confidence-based triage for human review:

| Task | Field | Records | Purpose |
|------|-------|---------|---------|
| A | `diagnosis_specify` | 1,868 | Validate outcome classification against free-text diagnosis |
| B | `consult_specify`, `discharge_refer_specify` | 515 | Harmonize Year 1 free-text to Year 2 checkboxes |
| C | `symptom_other` | 1,024 | Classify other symptoms into 18 categories |
| D | `prev_diagnoses_other` | 970 | Categorize comorbidities |
| E | `med_benign_specify` | 157 | Verify benign classifications are truly benign |

Outputs: `data/processed/llm_coded/task_*.csv`

### Step 3: PI Review of Ambiguous Cases

124 cases flagged by LLM coding were reviewed by the PI (pediatric neurologist). Cases organized into two tiers:

- **Tier 1** (74 cases): High-confidence LLM flags requiring clinical adjudication
  - Task A: MI checkbox set but diagnosis text appears benign (checkbox errors)
  - Task A: Benign checkbox set but LLM identified MI-consistent diagnosis (missed MI)
  - Task E: Benign diagnosis that LLM flagged as potentially MI
- **Tier 2** (50 cases): Moderate-confidence flags for batch review

**Key decisions:**
- 18 records corrected from MI to benign (checkbox errors: seizure, syncope, headache, r/o diagnoses)
- 48 records reclassified from benign to MI (missed cases: cerebellar ataxia, IIH, shunt malfunction, infections, demyelinating, neoplasm)
- 2 new MI sub-categories created under "Other MI" (code 11):
  - **IIH** (Idiopathic Intracranial Hypertension): 8 cases (pseudotumor cerebri, papilledema, B.I.H.)
  - **Cerebellar ataxia**: 26 cases (cerebellitis, post-viral ataxia, ataxia NOS)
- **Rule-out principle:** "r/o" diagnoses kept benign unless confirmatory testing documented elsewhere
- **Symptom-only principle:** Isolated symptoms (altered LOC, syncope, weakness NOS) kept benign without confirmed MI etiology

**Documentation:**
- `data/processed/llm_coded/pi-review-decisions.csv` — All 124 decisions with MI codes, diagnosis text, record counts, source tier/task, and clinical rationale
- `data/processed/llm_coded/pi_review_focused.csv` — Raw review file with per-case decisions

### Validation Against Manual Cleaning

The automated pipeline was compared against a previously manually cleaned dataset (`data/manual_clean_data/ba_excel_data.csv`, 113 MI). Overall agreement was 97.0% on MI classification across 2,149 matched records. The automated pipeline identified 18 checkbox data entry errors missed by manual cleaning and 42 additional MI cases through systematic free-text validation. Full comparison in [`outputs/tables/comparison_manual_vs_auto.md`](outputs/tables/comparison_manual_vs_auto.md).

### Step 4: Outcome Derivation

After PI review, the final outcome counts are:

| MI Category | Code | N |
|-------------|------|---|
| Brain/meningeal infection | 7 | 42 |
| Cerebellar ataxia | 13 | 26 |
| Neoplasm | 9 | 22 |
| Demyelinating disorder | 6 | 13 |
| Other medically important | 11 | 12 |
| Hydrocephalus | 8 | 10 |
| Stroke | 1 | 8 |
| ICH | 2 | 8 |
| IIH (idiopathic intracranial hypertension) | 12 | 8 |
| Venous thrombosis | 5 | 3 |
| Vascular malformation | 3 | 2 |
| Inborn error of metabolism | 10 | 1 |
| **Total MI** | | **147** |
| **Total benign** | | **2,006** |

---

## Scripts

All scripts run sequentially from `scripts/`. Each reads from `data/processed/brain_attacks_combined.parquet` (except 01 which creates it).

| Script | Objective | Key Outputs |
|--------|-----------|-------------|
| `01-data-prep.py` | Data import, harmonization, PI review, feature engineering | `brain_attacks_combined.parquet`, `data_prep_summary.csv`, `tableS3_harmonization_mapping.csv` |
| `01b-llm-coding.py` | LLM-assisted free-text classification | `data/processed/llm_coded/task_*.csv` |
| `02-descriptive.py` | Epidemiology (Obj 4) | `table1_demographics.csv`, `table2_mi_diagnoses.csv`, `tableS1_univariate.csv`, `tableS2_practice_variation.csv`, 2 figures |
| `03-prediction.py` | XGBoost + LR prediction (Obj 1) | `table_model_performance.csv`, `table_feature_importance.csv`, 4 figures |
| `04-validation.py` | Temporal validation (Obj 2) | `table3_temporal_validation.csv`, 1 figure |
| `05-neuroimaging.py` | Neuroimaging predictors (Obj 3) | `table4_neuroimaging_predictors.csv`, `table5_neuroimaging_yield.csv`, 1 figure |
| `06-prediction-rule.py` | Clinical prediction score (Obj 1) | `table_prediction_rule.csv`, `table_score_performance.csv`, 1 figure |
| `07-headache.py` | Headache sub-analysis (IHC 2017 extension) | `table6_headache_model.csv`, `table_headache_demographics.csv`, 1 figure |
| `08-sensitivity.py` | Sensitivity analysis: MI outcome definition | `table_sensitivity_analysis.csv`, 1 figure |
| `09-calibration.py` | Calibration + bootstrap CIs (TRIPOD+AI) | `table_calibration_metrics.csv`, 3 figures |
| `10-decision-curve.py` | Decision curve analysis (TRIPOD+AI) | `table_net_benefit.csv`, 1 figure |
| `11-stability.py` | Model stability, collinearity, optimism | `tableS7_vif.csv`, `tableS8_bootstrap_stability.csv`, 1 figure |
| `12-cost-analysis.py` | Cost-informed threshold analysis (Manuscript 1) | `table_cost_thresholds.csv`, `tableS11_cost_sensitivity.csv`, `tableS12_investigation_mix.csv`, 2 figures |
| `13-headache-deep.py` | Extended headache analysis (Manuscript 2) | `table_ha_*.csv` (5 tables), 5 figures |

### Two-Manuscript Structure

**Manuscript 1** (J Pediatrics): Primary brain attack prediction — scripts 01–12 outputs, plus brief headache section from script 07.

**Manuscript 2** (Headache / Cephalalgia): Deep headache-specific prediction, imaging appropriateness, and cost analysis — script 13 outputs (all `ha_` prefixed). References Manuscript 1 for full cohort methods. Extends the IHC 2017 abstract to the full 2-year analysis.

### Running the Pipeline

```bash
python scripts/01-data-prep.py    # Must run first
python scripts/02-descriptive.py
python scripts/03-prediction.py   # ~2 min (nested CV)
python scripts/04-validation.py
python scripts/05-neuroimaging.py
python scripts/06-prediction-rule.py
python scripts/07-headache.py
python scripts/08-sensitivity.py  # ~3 min (repeated nested CV)
python scripts/09-calibration.py  # ~5 min (CV + bootstrap)
python scripts/10-decision-curve.py
python scripts/11-stability.py
python scripts/12-cost-analysis.py  # ~2 min (CV for predictions)
python scripts/13-headache-deep.py  # ~2 min (headache deep dive)
```

---

## Key Results (Post-PI Review)

| Analysis | AUROC | Notes |
|----------|-------|-------|
| XGBoost (10-fold CV) | 89.8% | Pooled CV, k=5 features after QC |
| Temporal validation (Y1 to Y2) | 86.6% | XGBoost; LR=80.4% |
| Clinical score (10-fold CV) | 85.0% | 10-predictor integer-weighted score |
| Headache model (derivation) | 89.3% | Year 1 derivation |
| Headache model (validation) | 81.9% | Year 2 validation |

**Top predictors:** Ataxia (0.455), papilledema (0.255), altered LOC exam (0.118), focal weakness (0.103), age (0.068).

---

## Directory Structure

```
brain-attacks/
+-- README.md                          # This file
+-- analysis-plan.md                   # Detailed statistical analysis plan
+-- config.yml                         # Project configuration
+-- .gitignore
+-- scripts/
|   +-- 01-data-prep.py                # Data import, harmonization, PI review
|   +-- 01b-llm-coding.py             # LLM-assisted free-text coding
|   +-- 02-descriptive.py             # Epidemiology tables and figures
|   +-- 03-prediction.py              # XGBoost + logistic regression
|   +-- 04-validation.py              # Temporal validation (Y1 -> Y2)
|   +-- 05-neuroimaging.py            # Neuroimaging predictors
|   +-- 06-prediction-rule.py         # Clinical prediction score
|   +-- 07-headache.py                # Headache sub-analysis
|   +-- 08-sensitivity.py             # Sensitivity analysis: MI definition
|   +-- 09-calibration.py             # Calibration + bootstrap CIs
|   +-- 10-decision-curve.py          # Decision curve analysis
|   +-- 11-stability.py               # Model stability, collinearity
|   +-- 12-cost-analysis.py           # Cost-informed threshold analysis
|   +-- 13-headache-deep.py           # Extended headache analysis
+-- data/                              # .gitignored
|   +-- raw_data/                      # Source REDCap exports
|   +-- labelled_data/                 # REDCap exports with labels
|   +-- manual_clean_data/             # Previous manually cleaned dataset
|   |   +-- ba_excel_data.csv          # Excel-based manual cleaning (113 MI)
|   +-- metadata/                      # Data dictionaries
|   +-- processed/
|       +-- brain_attacks_combined.csv
|       +-- brain_attacks_combined.parquet
|       +-- llm_coded/
|           +-- task_a_diagnosis.csv       # LLM outcome validation
|           +-- task_b_consult.csv         # LLM consultation coding
|           +-- task_b_referral.csv        # LLM referral coding
|           +-- task_c_symptom.csv         # LLM symptom classification
|           +-- task_d_comorbidity.csv     # LLM comorbidity coding
|           +-- task_e_benign_review.csv   # LLM benign review
|           +-- human_review.csv           # Combined LLM flags for review
|           +-- pi_review_focused.csv      # 124 ambiguous cases with PI decisions
|           +-- pi-review-decisions.csv    # Audit trail: all decisions with rationale
+-- outputs/
    +-- tables/
    |   +-- data_prep_summary.csv
    |   +-- table1_demographics.csv
    |   +-- table2_mi_diagnoses.csv
    |   +-- table3_temporal_validation.csv
    |   +-- table4_neuroimaging_predictors.csv
    |   +-- table5_neuroimaging_yield.csv
    |   +-- table6_headache_model.csv
    |   +-- table_feature_importance.csv
    |   +-- table_headache_demographics.csv
    |   +-- table_model_performance.csv
    |   +-- table_prediction_rule.csv
    |   +-- table_score_performance.csv
    |   +-- tableS1_univariate.csv
    |   +-- tableS2_practice_variation.csv
    |   +-- tableS3_harmonization_mapping.csv
    |   +-- table_sensitivity_analysis.csv  # Sensitivity: primary vs restrictive MI
    |   +-- table_calibration_metrics.csv  # Calibration + bootstrap CIs
    |   +-- table_net_benefit.csv          # Decision curve analysis
    |   +-- tableS7_vif.csv               # VIF collinearity
    |   +-- tableS8_bootstrap_stability.csv # Feature stability
    |   +-- table_cost_parameters.csv      # Cost parameter documentation
    |   +-- table_cost_thresholds.csv      # Cost at key thresholds
    |   +-- tableS11_cost_sensitivity.csv  # Cost sensitivity (3 scenarios)
    |   +-- tableS12_investigation_mix.csv # Investigation rates
    |   +-- table_ha_demographics_extended.csv  # Headache extended demographics
    |   +-- table_ha_appropriateness.csv   # Imaging appropriateness
    |   +-- table_ha_model_extended.csv    # Extended headache model
    |   +-- table_ha_decision_curve.csv    # Headache decision curve
    |   +-- table_ha_cost.csv             # Headache cost analysis
    |   +-- comparison_manual_vs_auto.md  # Manual vs automated MI comparison
    +-- figures/
        +-- fig_beta_experiment.png
        +-- fig_false_negative_profiles.png
        +-- fig_feature_importance.png
        +-- fig_model_comparison.png
        +-- fig_neuroimaging_yield.png
        +-- fig_roc_headache.png
        +-- fig_roc_temporal.png
        +-- fig_score_performance.png
        +-- fig_sensitivity_comparison.png
        +-- fig_symptom_prevalence.png
        +-- fig_temporal_patterns.png
        +-- fig_calibration_combined.png
        +-- fig_calibration_temporal.png
        +-- fig_calibration_headache.png
        +-- fig_decision_curve.png
        +-- fig_predictor_correlation.png
        +-- fig_nmb_curve.png             # Net monetary benefit
        +-- fig_cost_per_case.png         # Cost per MI detected
        +-- fig_ha_appropriateness.png    # Headache imaging appropriateness
        +-- fig_ha_roc_extended.png       # Headache ROC (LR + XGBoost)
        +-- fig_ha_calibration.png        # Headache calibration
        +-- fig_ha_decision_curve.png     # Headache decision curve
        +-- fig_ha_cost_threshold.png     # Headache cost per MI
```

## Reproducibility

- **Python:** 3.11+
- **Key packages:** pandas, scikit-learn, xgboost, imbalanced-learn, matplotlib, scipy
- **Random seed:** 42 (all scripts)
- **Data:** Gitignored; copy from `/Volumes/fomd-richerlab/data/brain_attacks` to `data/raw_data/`
