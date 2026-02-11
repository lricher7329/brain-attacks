# Manuscript vs. Current Analysis: Discrepancy Report

**Manuscript:** Richer2025_BrainAttackManuscript_JPediatrics.md (draft)
**Analysis:** Scripts 01-11 outputs in outputs/tables/

---

## CRITICAL DISCREPANCIES

### 1. MI Case Count: 113 (manuscript) vs 147 (analysis)

The manuscript reports **113 medically important cases** and 2,040 benign cases. The current automated pipeline identifies **147 MI cases** and 2,006 benign cases. This is a 30% increase in MI cases.

The `comparison_manual_vs_auto.md` QC report documents this difference:
- 18 cases were Manual-MI / Auto-Benign (reclassified as benign)
- 46 cases were Manual-Benign / Auto-MI (newly identified as MI)
- Net change: +34 additional MI cases
- New MI cases include: ~20 cerebellar ataxia, ~7 hydrocephalus, ~5 IIH, ~5 infection, ~5 demyelinating

**Impact:** This fundamentally changes prevalence (5.2% to 6.8%), all performance metrics, and the clinical prediction rule. Every number in the manuscript needs updating.

---

### 2. XGBoost Performance Metrics

| Metric | Manuscript | Current Analysis | Direction |
|--------|-----------|-----------------|-----------|
| Accuracy | 91.3% +/- 3.0% | 90.9% (CV) | Similar |
| Sensitivity | **57.6% +/- 14.6%** | **40.1% (32.0-48.3%)** | Lower |
| Specificity | **92.7% +/- 2.5%** | **98.2% (97.7-98.8%)** | Higher |
| AUROC | Not reported | 0.892 (0.859-0.919) | New metric |

The manuscript reports substantially higher sensitivity (57.6% vs 40.1%) but lower specificity (92.7% vs 98.2%). This likely reflects:
- Different MI definition (113 vs 147 cases)
- Different model tuning (manuscript feature selection k={5,15,25,all} vs current k=all)
- Different SMOTE ratios tested (manuscript 0.5/0.75/1.0 vs current fixed 0.5)
- Manuscript reports mean +/- SD across folds; current analysis reports point estimates with bootstrap CIs

### 3. Logistic Regression Performance

| Metric | Manuscript | Current Analysis | Direction |
|--------|-----------|-----------------|-----------|
| Accuracy | 94.8% +/- 0.3% | Not directly reported | - |
| Sensitivity | **4.3% +/- 5.9%** | **15.6% (10.2-21.1%)** | Higher |
| Specificity | **99.8% +/- 0.2%** | **99.1% (98.8-99.6%)** | Lower |
| AUROC | Not reported | 0.795 (0.748-0.842) | New metric |

The LR baseline differs significantly. The manuscript uses the top 5 univariate features selected once on the entire dataset (data leakage concern); the current analysis uses SelectKBest(k=5) fit within each fold.

---

### 4. MI Diagnosis Distribution

| Diagnosis | Manuscript (N=113) | Analysis (N=147) | Delta |
|-----------|-------------------|------------------|-------|
| Infection | 43 (19%*) | 42 (28.6%) | -1 |
| Cerebellar ataxia | Not listed separately | 26 (17.7%) | +26 |
| Neoplasm | 21 (19%*) | 22 (15.0%) | +1 |
| Demyelinating | 11 (10%) | 13 (8.8%) | +2 |
| Other MI | 16 (14%) | 12 (8.2%) | -4 |
| Hydrocephalus | 8 (7%) | 10 (6.8%) | +2 |
| Stroke | 8 (7%) | 8 (5.4%) | 0 |
| ICH | 8 (7%) | 8 (5.4%) | 0 |
| IIH | Not listed separately | 8 (5.4%) | +8 |
| Venous thrombosis | 2 (1.8%) | 3 (2.0%) | +1 |
| Vascular malformation | 3 (3%) | 2 (1.4%) | -1 |
| Inborn error | 1 (1%) | 1 (0.7%) | 0 |
| Vasculitis | 0 (0%) | Not listed | 0 |

*Manuscript percentages appear incorrect (infection 43/113 = 38%, not 19%; neoplasm 21/113 = 19%, correct).

Key changes: Cerebellar ataxia (26 cases) and IIH (8 cases) were previously grouped under "other" or classified as benign. The automated pipeline correctly identifies these as MI.

---

### 5. Feature Importance Rankings

| Rank | Manuscript | Current Analysis |
|------|-----------|-----------------|
| 1 | Ataxia (0.19) | sx_ataxia (0.210) |
| 2 | Altered LOC: Exam (0.13) | exam_papilledema (0.061) |
| 3 | Gender (0.10) | exam_altered_loc (0.075) |
| 4 | Altered LOC: Symptoms (0.09) | sx_focal_weakness (0.034) |
| 5 | Fever (0.07) | age (varies) |

Ataxia remains #1 in both. However, gender drops from #3 to not appearing in the top 5. Papilledema, not prominent in the manuscript, becomes the #2 predictor. This likely reflects the additional MI cases (especially IIH, which presents with papilledema).

---

## IMPORTANT DISCREPANCIES

### 6. Analyses Present in Current Pipeline but Absent from Manuscript

The manuscript is missing several analyses now completed in the pipeline:

| Analysis | Status in Manuscript |
|----------|---------------------|
| AUROC (any model) | **Not reported** |
| Bootstrap confidence intervals | **Not reported** (uses SD across folds instead) |
| Temporal validation (Year 1 to Year 2) | **Not performed** |
| Clinical prediction rule (10-predictor score) | **Not performed** |
| Headache subgroup analysis | **Not performed** |
| Calibration assessment (CITL, slope, Brier) | **Not performed** |
| Decision curve analysis | **Not performed** |
| Model stability / bootstrap feature selection | **Not performed** |
| Optimism-corrected AUROC | **Not performed** |
| Collinearity assessment (VIF) | **Not performed** |
| Sensitivity analysis (restrictive MI definition) | **Not performed** |
| Neuroimaging yield analysis | **Not performed** |

### 7. Analyses Present in Manuscript but Absent from Current Pipeline

| Analysis | Status in Pipeline |
|----------|-------------------|
| F-score beta experiment (beta 1.0-3.0, 100 runs) | **Not implemented** |
| False negative case exploration (6 groups) | **Not implemented** |
| Always-misclassified vs always-correct symptom comparison | **Not implemented** |
| Z-score normalization of continuous variables | **Not used** (fillna(0) instead) |

---

### 8. Methodological Differences

| Aspect | Manuscript | Current Pipeline |
|--------|-----------|-----------------|
| **MI definition** | 113 cases (manual) | 147 cases (automated) |
| **Feature count** | 48 variables | 38 predictors (5 groups) |
| **Missing data** | Median imputation + Z-scores | fillna(0) |
| **Inner CV** | 5-fold | 3-fold |
| **Feature selection** | k = {5, 15, 25, all} | k = all (no selection) |
| **SMOTE ratios** | {0.5, 0.75, 1.0} | Fixed 0.5 |
| **LR features** | Top 5 from full-data F-test | SelectKBest(k=5) per fold |
| **Metric reported** | Mean +/- SD across folds | Point estimate + bootstrap CI |
| **Optimization** | F2-score | F2-score (same) |
| **Repeated runs** | 100 permutations x 10-fold | Single 10-fold CV |
| **Exclusions** | 1 removed (missing gender) | 176 excluded (trauma) |

### 9. Table 1 Demographics

| Variable | Manuscript MI (N=113) | Analysis MI (N=147) |
|----------|----------------------|---------------------|
| Age | 7 +/- 6 years | 10.7 mean (full cohort) |
| Male | 43 (38.1%) | ~42.8% (gender) |
| EMS | 45 (39.8%) | 17.7% (full cohort) |
| Heart rate | 109 +/- 32 bpm | Not split by MI/benign |
| Temperature | 37 +/- 1 C | Not split by MI/benign |

The manuscript reports MI mean age of 7 years vs overall cohort mean of 10.7; this age difference is clinically meaningful (MI children are younger).

---

## MINOR DISCREPANCIES

### 10. Percentage Errors in Manuscript

The manuscript Table 2 / diagnosis text contains apparent percentage errors:
- Infection listed as "19%" but 43/113 = 38.1%
- The manuscript text says "brain and/or meningeal infection (43 cases, 19% of total)" - this percentage is wrong

### 11. Exclusion Criteria

- Manuscript: "Patients were excluded who had a history of head trauma within 1 week preceding presentation." States 1 subject removed for missing gender.
- Pipeline: 176 excluded for trauma (from 2,329 raw to 2,153).
- The manuscript doesn't mention the 176 trauma exclusions or the 2,329 starting number.

### 12. Sample Size Justification

Neither the manuscript nor the original pipeline addresses events-per-variable (EPV). The updated DAP now documents EPV = 4.5 (below the 10 threshold), which should be acknowledged as a limitation.

---

## SUMMARY OF REQUIRED MANUSCRIPT UPDATES

### Numbers that must change:
1. MI count: 113 &rarr; 147
2. Benign count: 2,040 &rarr; 2,006
3. Prevalence: ~5.2% &rarr; 6.8%
4. All performance metrics (sensitivity, specificity, accuracy)
5. Feature importance rankings and values
6. MI diagnosis distribution table (add cerebellar ataxia, IIH as separate categories)
7. Table 1 demographics (reflect 147 MI cohort)
8. Correct percentage errors in diagnosis distribution

### Sections that need major revision:
1. **Results 3.1**: All performance numbers
2. **Results 3.2**: Feature importance rankings
3. **Methods 2.2**: Missing data approach
4. **Methods 2.3**: Model training parameters
5. **Discussion 4.1**: Performance comparison narrative
6. **Discussion 4.5**: Limitations (add EPV, single-centre, narrow temporal validation)

### New sections needed:
1. AUROC reporting (primary discrimination metric)
2. Temporal validation results
3. Clinical prediction rule
4. Calibration assessment
5. Decision curve / clinical utility
6. Headache subgroup (if retained)

### Sections from manuscript to decide on:
1. **F-score beta analysis** (100-run experiment): Retain, update, or replace with decision curve analysis?
2. **False negative exploration**: Retain and update with 147 MI cases?
3. **Z-score preprocessing**: Harmonize approach between manuscript and pipeline?

---

## RECOMMENDATION

The manuscript draft predates significant analytical work. The current pipeline represents a substantially more rigorous analysis with:
- Corrected MI classification (+34 cases)
- Bootstrap confidence intervals instead of fold-level SD
- Temporal validation
- Calibration assessment
- Decision curve analysis
- Clinical prediction rule
- TRIPOD+AI compliance

The manuscript should be substantially rewritten to reflect the current pipeline results. The F-score beta analysis and false negative exploration from the original manuscript are valuable and could be re-run with the updated 147-case MI definition and incorporated into the revised manuscript.
