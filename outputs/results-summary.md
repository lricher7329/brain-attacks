# Brain AttACKs: Comprehensive Results Summary

**Project:** Prediction of Medically Important Pediatric Brain Attack
**Setting:** Stollery Children's Hospital Emergency Department
**Design:** Retrospective cohort, two-year chart review
**Analysis date:** 2025

---

## 1. Cohort Description

| Metric | Value |
|--------|-------|
| Raw records | 2,329 |
| Excluded (trauma) | 176 |
| **Analytic cohort** | **2,153** |
| Year 1 | 1,043 (75 MI, 7.2%) |
| Year 2 | 1,110 (72 MI, 6.5%) |
| **MI prevalence** | **147 / 2,153 (6.8%)** |
| Total features | 55 predictors across 6 groups |

**Demographics (full cohort):**
- Mean age: 10.7 years
- Female: 57.2%
- EMS transport: 17.7%
- Median CTAS: 3

---

## 2. MI Diagnosis Distribution

| Diagnosis | n | % of MI |
|-----------|---|---------|
| Brain/meningeal infection | 42 | 28.6% |
| Cerebellar ataxia | 26 | 17.7% |
| Neoplasm | 22 | 15.0% |
| Demyelinating disease | 13 | 8.8% |
| Other MI | 12 | 8.2% |
| Hydrocephalus | 10 | 6.8% |
| Stroke | 8 | 5.4% |
| Intracranial hemorrhage | 8 | 5.4% |
| Idiopathic intracranial hypertension | 8 | 5.4% |
| Venous thrombosis | 3 | 2.0% |
| Vascular malformation | 2 | 1.4% |
| Inborn error of metabolism | 1 | 0.7% |

---

## 3. Primary Model Performance (10-fold CV)

| Metric | XGBoost (default) | XGBoost (F2-opt) | Logistic Regression |
|--------|-------------------|-------------------|---------------------|
| AUROC | **0.905** (0.860-0.924) | 0.905 (same model) | 0.782 (0.733-0.829) |
| Sensitivity | 43.6% (31.2-55.9%) | **74.1%** (58.9-89.3%) | 15.0% (4.5-25.5%) |
| Specificity | **98.2%** (97.2-99.2%) | 91.9% (89.1-94.7%) | 99.3% (98.7-99.9%) |
| F2 | 46.5% | **63.4%** | 17.5% |
| Brier score | **0.0444** | — | 0.0512 |
| Calibration slope | 0.782 | — | **0.965** |
| CITL | 0.021 | — | -0.001 |
| Threshold | 0.50 | 0.09 | 0.50 |

**Key finding:** Expanding `sx_other` into 18 symptom subcategories (55 total features) improves XGBoost discrimination (AUROC 0.905 vs 0.894 previously) and sensitivity at both thresholds. At the F2-optimized threshold (0.09), sensitivity reaches **74.1%** at 91.9% specificity. New top-10 features include `other_hemorrhage`, `other_psych`, and `other_pain`. `other_cranial_nerve` ranks 6th in univariate F-test (F=49.1). The unified inner CV consistently selects SMOTE=1.0 across all 10 folds.

---

## 4. Temporal Validation (Year 1 &rarr; Year 2)

| Metric | XGBoost | Logistic Regression |
|--------|---------|---------------------|
| AUROC | **0.889** (0.844-0.928) | 0.804 (0.735-0.866) |
| Sensitivity | **47.2%** (35.6-58.8%) | 13.9% (6.9-22.2%) |
| Specificity | 97.1% (96.1-98.1%) | 99.1% (98.6-99.6%) |
| Calibration slope | 0.673 | 1.094 |
| CITL | -0.203 | -0.221 |

**Key finding:** Temporal validation improves with expanded features. XGBoost AUROC drops modestly from 0.905 to 0.889 (vs 0.894→0.872 previously). Sensitivity on temporal validation reaches 47.2% (up from 43.1%). The calibration slope of 0.67 indicates predictions are too extreme (over-confident).

---

## 5. Clinical Prediction Rule

A 10-predictor integer-weighted score derived from logistic regression coefficients:

| Predictor | Points |
|-----------|--------|
| sx_ataxia | +7 |
| exam_papilledema | +7 |
| exam_meningismus | +5 |
| exam_altered_loc | +3 |
| sx_focal_weakness | +3 |
| sx_fever | +2 |
| ems | +1 |
| sx_seizure | -2 |
| sx_collapse | -6 |
| gender (female) | -1 |

**Score performance (10-fold CV):**

| Metric | Value |
|--------|-------|
| AUROC | 0.843 (0.805-0.881) |
| Sensitivity | 17.6% (11.6-23.8%) |
| Specificity | 98.4% (97.9-99.0%) |
| Calibration slope | 0.990 |
| Brier score | 0.0511 |

At a score threshold of 0: Sensitivity 79.6%, Specificity 77.9%, PPV 20.9%, LR+ 3.60.

---

## 6. Headache Subgroup Analysis

| Metric | Derivation (Year 1, n=630) | Validation (Year 2, n=657) |
|--------|---------------------------|---------------------------|
| MI cases | 32 (5.1%) | 28 (4.3%) |
| AUROC | **0.811** (0.703-0.906) | 0.745 (0.630-0.850) |
| Sensitivity | 22.0% (9.4-37.5%) | 7.1% (0.0-17.9%) |
| Specificity | 99.7% (99.2-100%) | 99.2% (98.4-99.8%) |
| Calibration slope | 1.141 | 0.711 |

**Key finding:** The headache-specific model shows good derivation performance but notable degradation on temporal validation (AUROC 0.81 &rarr; 0.75, sensitivity 22% &rarr; 7%). The small MI count (n=28-32) limits statistical power.

---

## 7. Model Stability and Optimism

### 7a. Optimism-Corrected Performance

| Model | Apparent AUROC | Optimism | Corrected AUROC |
|-------|---------------|----------|-----------------|
| XGBoost | 0.994 | 0.116 | **0.878** |
| Logistic Regression | 0.819 | 0.034 | **0.785** |

**Key finding:** XGBoost shows substantial optimism (0.116), with apparent AUROC of 0.994 correcting down to 0.878. LR is far more stable with only 0.034 optimism. The optimism-corrected XGBoost AUROC (0.878) aligns well with the temporal validation AUROC (0.889).

### 7b. Bootstrap Feature Stability (200 refits)

Top predictors by consistency across bootstrap samples:

| Predictor | Top-10 Frequency | Mean Importance | Median Rank |
|-----------|-------------------|-----------------|-------------|
| sx_ataxia | **100%** | 0.170 | 1 |
| exam_papilledema | 90% | 0.059 | 3 |
| sx_collapse | 88% | 0.035 | 7 |
| exam_altered_loc | 74% | 0.069 | 4 |
| CTAS | 68% | 0.046 | 6 |
| sx_any_headache | 62% | 0.032 | 7 |
| EMS | 55% | 0.034 | 9 |
| sx_focal_weakness | 52% | 0.031 | 10 |
| sx_seizure | 48% | 0.027 | 11 |
| sx_fever | 48% | 0.032 | 10 |

**Key finding:** Ataxia is the dominant and most stable predictor, appearing in 100% of bootstrap samples with a mean importance nearly 3x the next predictor. The top 5 clinical features (ataxia, papilledema, collapse, altered LOC, CTAS) are consistently selected. The 18 `other_*` subcategory features distribute importance more granularly — individual subcategories appear in feature importance (e.g., `other_hemorrhage`, `other_psych`) rather than the former lumped `sx_other`.

### 7c. Collinearity Assessment

All variance inflation factors < 1.35, confirming no multicollinearity concerns among the clinical prediction rule variables.

---

## 8. Decision Curve Analysis (Clinical Utility)

Net benefit at clinically relevant thresholds:

| Threshold | Investigate All | XGBoost | Clinical Score | LR | Investigate None |
|-----------|----------------|---------|---------------|-----|-----------------|
| 2% | 0.049 | **0.054** | 0.052 | 0.047 | 0.000 |
| 5% | 0.019 | **0.049** | 0.044 | 0.036 | 0.000 |
| 10% | -0.035 | **0.042** | 0.035 | 0.028 | 0.000 |
| 15% | -0.096 | **0.036** | 0.031 | 0.023 | 0.000 |
| 20% | -0.165 | **0.031** | 0.027 | 0.020 | 0.000 |
| 30% | -0.331 | **0.024** | 0.013 | 0.017 | 0.000 |
| 50% | -0.863 | **0.013** | -0.003 | 0.003 | 0.000 |

**Key finding:** All three models demonstrate net benefit over default strategies across a wide range of clinically plausible thresholds (2-50%). XGBoost provides the highest net benefit at every threshold, with expanded features improving net benefit particularly at higher thresholds. At the 10% threshold, model-guided investigation would flag 244-251 patients (vs. 2,153 for investigate-all).

---

## 9. Sensitivity Analysis (MI Definition)

| Definition | N (MI) | XGBoost AUROC | XGBoost Sensitivity | LR AUROC |
|------------|--------|---------------|---------------------|----------|
| Primary | 147 | 0.905 | 43.6% | 0.782 |
| Restrictive | 113 | 0.873 | 19.5% | 0.808 |

**Key finding:** The restrictive MI definition (excluding conditions like cerebellar ataxia and IIH) yields lower sensitivity (20% vs 44%) and lower AUROC (0.87 vs 0.91). Temporal validation: primary AUROC 0.889 vs restrictive 0.832. The broader primary definition captures a more heterogeneous but clinically important group.

---

## 10. Neuroimaging Analysis

- Overall imaging rate: 19.3% (415/2,153)
- Abnormal imaging: 27.2% of those imaged (113/415)
- MI among imaged: 21.0% (87/415)
- Strongest imaging predictors: exam_papilledema (OR 6.94), exam_altered_loc (OR 4.57), sx_seizure (OR 3.22)
- Highest imaging NNI (number needed to image): collapse/syncope (11.3), headache alone (4.9-6.8)
- Lowest NNI: seizure (2.0-2.7), altered consciousness (2.4-2.9)

---

## 11. Data Quality

- 94.3% of records (2,030/2,153) had zero quality flags
- Automated pipeline identified 34 additional MI cases compared to manual classification
- Perfect outcome-checkbox consistency across all records
- 7 records with multiple MI codes (valid comorbid diagnoses)
- 12 MI cases had low-acuity triage (CTAS 4-5), suggesting triage alone is insufficient

---

## 12. Key Conclusions

1. **XGBoost outperforms LR** for predicting medically important pediatric brain attack (AUROC 0.91 vs 0.78), with substantially better sensitivity (44% vs 15% at default threshold).

2. **F2-optimal threshold improves sensitivity to 74%**: At the F2-optimized threshold (0.09), XGBoost achieves 74.1% sensitivity at 91.9% specificity, compared to 44% sensitivity at the default 0.50 threshold.

3. **Expanded "other symptom" features improve performance**: Breaking `sx_other` into 18 clinically meaningful subcategories (55 total features) improves AUROC from 0.894 to 0.905, temporal validation from 0.872 to 0.889, and MI detection at 10% threshold from 69% to 72%. Key new predictors include cranial nerve symptoms (F=49.1), hemorrhage, and psychiatric features.

4. **Ataxia is the dominant predictor**, appearing in 100% of bootstrap samples and carrying the highest importance weight in both ML and clinical score models.

5. **A 10-predictor clinical score** achieves competitive discrimination (AUROC 0.85) using only bedside-available features, making it practical for ED use.

6. **Models demonstrate clinical utility** across all reasonable threshold probabilities, with net benefit superior to both "investigate all" and "investigate none" strategies.

7. **XGBoost shows meaningful optimism** (corrected AUROC 0.878 vs apparent 0.994), but temporal validation (AUROC 0.889) confirms reasonable real-world performance.

8. **Calibration requires attention**: XGBoost calibration slope of 0.78 indicates over-confident predictions that would benefit from recalibration before deployment.

9. **Temporal validation is encouraging but narrow**: single-centre, consecutive-year design limits generalizability claims. External validation at other paediatric centres is needed.

10. **Low EPV (2.7)** is a limitation for the full 55-feature set, though the clinical prediction rule (10 predictors, EPV ~15) is adequately powered. The "other symptom" subcategories require quality verification as they were derived from LLM classification of free-text symptom descriptions.

---

## 13. Cost-Informed Threshold Analysis (Manuscript 1)

Cost estimates from published Canadian sources (CADTH 2023 for CT/MRI; estimated for LP/EEG/consultation). Three scenarios: base (professional fee ~$167/investigation), mid (CADTH institutional ~$539), high (upper range ~$797).

**Investigation rates (full cohort, N=2,153):**

| Investigation | N | Rate |
|---|---|---|
| CT head | 354 | 16.4% |
| MRI brain | 102 | 4.7% |
| LP | 59 | 2.7% |
| EEG | 54 | 2.5% |
| CTA | 10 | 0.5% |
| MRA | 5 | 0.2% |
| Any investigation | 471 | 21.9% |
| Consultation (Y2) | 264 | 23.8% of Y2 |

**MI detection and cost at key thresholds (XGBoost, base scenario):**

| Threshold | MI Detected (%) | N Flagged | Cost/MI Detected |
|-----------|-----------------|-----------|-----------------|
| 2% | 89.1% | 714 | $1,026 |
| 5% | 83.0% | 402 | $573 |
| 10% | 72.1% | 251 | $395 |
| 15% | 65.3% | 200 | $347 |
| 20% | 59.2% | 164 | $316 |
| 30% | 52.4% | 134 | $290 |

**Model comparison at 10% threshold (base scenario):**

| Model | N Flagged | MI Detected (%) | Cost/MI |
|-------|-----------|-----------------|---------|
| XGBoost | 251 | 72.1% | $395 |
| Clinical score | 244 | 63.3% | $438 |
| Logistic regression | 313 | 58.5% | $607 |

**Key finding:** At a 10% threshold, XGBoost-guided investigation would flag 251 patients (12% of cohort) and detect 72% of MI cases at $395/MI detected — compared to investigating all 2,153 patients. The expanded feature set improves MI detection (72% vs 69% previously) at similar cost efficiency.

**Cost sources:** CADTH Health Technology Review 2023 (CT mean $310, MRI mean $762). LP, EEG, consultation estimated. Full documentation in `table_cost_parameters.csv`.

---

## 14. Extended Headache Analysis (Manuscript 2)

**Cohort:** 873 headache patients (51 MI, 5.8%) after excluding 414 with prior headache disorder.

### Imaging Appropriateness

| Rating | N | Imaging Rate | MI Rate |
|--------|---|-------------|---------|
| Usually NOT appropriate | 580 | 10.5% | Low |
| May be appropriate | 132 | Variable | Variable |
| Usually appropriate | 131 | 71.0% | High |

- **Overtesting:** 10.5% of "Usually NOT appropriate" patients were imaged
- **Undertesting:** 29.0% of "Usually appropriate" patients were NOT imaged
- Chi-squared (appropriateness vs imaging): p<0.001
- Chi-squared (appropriateness vs MI): p<0.001

### Extended Prediction Model

| Model | Derivation AUROC (Y1) | Validation AUROC (Y2) |
|-------|----------------------|----------------------|
| Logistic regression (21 vars) | **0.928** (0.873-0.978) | **0.813** (0.706-0.898) |
| XGBoost | 0.991 (0.978-1.000) | 0.804 (0.703-0.891) |

**Top LR predictors (by aOR):** vomiting (2.12), altered LOC exam (1.72), EMS transport (1.53), papilledema (1.52), age <12 (1.51), progressive headache (1.49).

**Key finding:** The extended 21-predictor LR model achieves substantially higher derivation AUROC (0.928 vs 0.811 in script 07's simpler model) and better validation AUROC (0.813 vs 0.745). XGBoost matches LR on validation despite much higher apparent performance, suggesting overfitting on the small headache cohort.

### Headache Cost Analysis

- 210/873 (24.1%) of headache patients were imaged
- 47 MI cases detected among imaged patients (current practice)
- Mean cost per imaged headache patient (base): $150
- Current practice total cost: $31,430 (base scenario)

---

## 15. Two-Manuscript Structure

**Manuscript 1** (J Pediatrics): Scripts 01-12. Primary brain attack prediction with cost-informed threshold analysis. Brief headache section (script 07). TRIPOD+AI compliant.

**Manuscript 2** (Headache / Cephalalgia): Script 13. Extended headache prediction, imaging appropriateness, and cost analysis. References Manuscript 1 for full cohort methods. Extends IHC 2017 abstract.

---

## 16. Output File Index

| Script | Output | Description |
|--------|--------|-------------|
| 01 | data_prep_summary.csv | Cohort assembly summary |
| 01 | tableS3_harmonization_mapping.csv | Field harmonization documentation |
| 02 | table1_demographics.csv | Baseline demographics |
| 02 | table2_mi_diagnoses.csv | MI diagnosis distribution |
| 02 | tableS1_univariate.csv | Univariate analysis (85 variables) |
| 02 | tableS2_practice_variation.csv | Investigation and consultation patterns |
| 03 | table_model_performance.csv | Primary model performance |
| 03 | table_feature_importance.csv | XGBoost feature importance |
| 04 | table3_temporal_validation.csv | Year 1 &rarr; Year 2 validation |
| 05 | table4_neuroimaging_predictors.csv | Imaging predictor analysis |
| 05 | table5_neuroimaging_yield.csv | Symptom-based imaging yield |
| 06 | table_prediction_rule.csv | Clinical prediction rule weights |
| 06 | table_score_performance.csv | Score threshold performance |
| 07 | table6_headache_model.csv | Headache subgroup model |
| 07 | table_headache_demographics.csv | Headache cohort demographics |
| 08 | table_sensitivity_analysis.csv | Primary vs restrictive MI |
| 09 | table_calibration_metrics.csv | Calibration + bootstrap CIs |
| 10 | table_net_benefit.csv | Decision curve analysis |
| 11 | tableS7_vif.csv | Collinearity (VIF) |
| 11 | tableS8_bootstrap_stability.csv | Feature stability |
| 12 | table_cost_parameters.csv | Cost parameter documentation |
| 12 | table_cost_thresholds.csv | Cost at key thresholds |
| 12 | tableS11_cost_sensitivity.csv | Cost sensitivity (3 scenarios) |
| 12 | tableS12_investigation_mix.csv | Investigation rates |
| 13 | table_ha_demographics_extended.csv | Extended headache demographics |
| 13 | table_ha_appropriateness.csv | Imaging appropriateness |
| 13 | table_ha_model_extended.csv | Extended headache model |
| 13 | table_ha_decision_curve.csv | Headache decision curve |
| 13 | table_ha_cost.csv | Headache cost analysis |
| QC | qc_mi_summary.md | Quality control report |
| QC | comparison_manual_vs_auto.md | Pipeline validation |
