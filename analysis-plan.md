# Brain AttACKs Data Analysis Plan

**Study:** Prediction of Clinically Important Neurologic Presentation in Children with Acute Brain Attack
**Protocol:** v1.1 (May 19, 2016) | PRO00055264
**Manuscript target:** Pediatrics (AAP)
**Date:** 2026-02-12

---

## 1. Study Overview

Retrospective observational cohort of children presenting to the Stollery Children's Hospital ED with non-traumatic acute focal brain dysfunction ("brain attack"). Two years of data collected via chart review into REDCap.

### 1.1 Case Ascertainment (RECORD Item 6.1)

Cases were identified from the Stollery Children's Hospital ED census using ICD-10-CA discharge diagnosis codes consistent with acute neurological presentations (stroke, seizure, headache, altered consciousness, focal deficit, etc.). Research assistants screened the ED census daily during the study period and flagged all visits meeting the "brain attack" clinical definition: acute onset of focal or diffuse neurological dysfunction of presumed vascular, infectious, inflammatory, neoplastic, or metabolic origin, excluding trauma. Charts were then manually reviewed to confirm eligibility against protocol inclusion/exclusion criteria before data abstraction into REDCap. The full case identification algorithm and ICD code list are documented in protocol PRO00055264.

- **Year 1:** `BrainAttACKs_DATA_2024-11-18_1011.csv` — 1,124 records, 158 columns
- **Year 2:** `BrainAttACKsYear2_DATA_2024-11-18_1013.csv` — 1,205 records, 191 columns
- **Combined N (post-exclusion):** 2,153 (147 CINP [clinically important neurologic presentation] after PI review, 2,006 non-CINP; 121 CIIP [clinically important intracranial pathology])

## 1b. Reporting Standards

This manuscript reports the development and temporal validation of a clinical prediction model from a retrospective chart-review cohort. The following reporting frameworks apply in a layered fashion and should be referenced during manuscript preparation.

| Layer | Guideline | Scope | Zotero Key |
|-------|-----------|-------|------------|
| **Primary** | TRIPOD+AI (Collins et al., BMJ 2024) | Prediction model reporting — 27-item checklist (supersedes TRIPOD 2015) | `P6YDK85N` |
| **Study design** | STROBE (von Elm et al., 2007) | Observational study reporting — 22-item checklist | `NR7G5TLX` |
| **Data source** | RECORD (Benchimol et al., 2015) | Routinely-collected health data — 13 additional items over STROBE | `8TQWKRAM` |
| **Quality self-check** | PROBAST (Wolff et al., 2019) | Risk of bias assessment — 20 signaling questions across 4 domains | `2D276DAT` |

**Practical application:**

1. Use TRIPOD+AI as the primary checklist. Submit the completed checklist as a supplementary file with page numbers for each item.
2. Supplement with RECORD items for chart-review data-specific reporting (case identification methods, code/algorithm lists, data quality assessment).
3. Reference STROBE for any general observational study items not already covered by TRIPOD+AI.
4. Use PROBAST proactively as a self-assessment tool — address each of the 4 domains (Participants, Predictors, Outcome, Analysis) in the methods and discussion.

**Key TRIPOD+AI items requiring dedicated analysis (addressed in Sections 8.9-8.11):**

| TRIPOD+AI Item | Requirement | DAP Section |
|----------------|-------------|-------------|
| Item 11 | Sample size justification | 8.1.1 |
| Item 14 | Missing data reporting | 8.1.2 |
| Item 20 | Participant flow diagram | 8.1.3 |
| Item 23a | Discrimination with CIs (AUROC, sensitivity, specificity) | 8.9 |
| Item 23b | Calibration (calibration plot, calibration-in-the-large, calibration slope) | 8.9 |
| Item 23d | Clinical utility (decision curve analysis) | 8.10 |
| Item 24 | Full model specification | 8.11 |
| Item 26 | Limitations including generalizability | Manuscript discussion |

## 2. Objectives

Per protocol, the study has four aims. These are organized below with references to prior work.

| # | Objective | Status |
|---|-----------|--------|
| 1 | Determine clinical predictors of clinically important neurologic presentation (CINP) and derive a clinical prediction rule | Completed (manuscript draft: XGBoost + logistic regression on combined cohort) |
| 2 | Validate clinical predictors using Year 2 as temporal validation cohort | Completed (`04-validation.py`: XGBoost + LR temporal Y1→Y2, `tableS5_case_mix_y1_vs_y2.csv`) |
| 3 | Determine clinical predictors of abnormal neuroimaging | Completed (`05-neuroimaging.py`: univariate + multivariate predictors, neuroimaging yield) |
| 4 | Describe the epidemiology of brain attack (incidence, symptoms, etiologies, practice variation, neuroimaging use) | Completed (`02-descriptive.py`: Table 1 demographics, Table S2 practice variation, investigation mix) |

### 2.1 Prior Analyses

**IHC 2017 Abstract (Richer et al.):** A sub-analysis of Year 1 headache cases (n=630 from 1,134 total). Stepwise logistic regression on a 66/33 random derivation/validation split identified predictors of medically urgent secondary causes of headache (4.4% prevalence, n=27). Key predictors: ambulance transport (OR 8.5), decreased consciousness (OR 37.6), progressive headache over days (OR 9.9), vomiting (OR 4.0). The model achieved 94.5% correct classification but a 33.3% false negative rate. This analysis should be extended to the full 2-year dataset as a secondary analysis (see Section 8.7).

**Manuscript Draft (Richer et al., 2025):** XGBoost classification on the combined 2-year cohort (N=2,153) using 10-fold stratified cross-validation. Achieved 91.3% accuracy, 57.6% sensitivity, 92.7% specificity. Top predictors by feature importance: ataxia, altered consciousness (exam finding), gender, altered consciousness (symptom), fever. The F-score beta experiment (100 runs x 5 beta values) and false negative analysis are described. This analysis is documented in Section 8.3 and does not need to be repeated, but the code should be archived for reproducibility.

## 3. Data Sources

All data is copied locally to `data/` (gitignored).

| Dataset | Path | Records | Columns |
|---------|------|---------|---------|
| Year 1 raw | `data/raw_data/BrainAttACKs_DATA_2024-11-18_1011.csv` | 1,124 | 158 |
| Year 2 raw | `data/raw_data/BrainAttACKsYear2_DATA_2024-11-18_1013.csv` | 1,205 | 191 |
| Year 1 labelled | `data/labelled_data/BrainAttACKs_DATA_LABELS_2024-11-18_1010.csv` | 1,124 | 158 |
| Year 2 labelled | `data/labelled_data/BrainAttACKsYear2_DATA_LABELS_2024-11-18_1013.csv` | 1,205 | 191 |
| Data dictionary (Year 2) | `data/metadata/BrainAttACKsYear2_DataDictionary_2024-11-18.csv` | — | 88 fields |

**Note:** The data dictionary corresponds to the Year 2 CRF only. Year 1 used a different CRF version with significant structural differences (see Section 4).

## 4. Data Harmonization

The Year 1 and Year 2 REDCap projects used different CRF versions. The following differences must be resolved during data preparation.

### 4.1 Renamed Fields (same concept, different field names)

| Concept | Year 1 field | Year 2 field |
|---------|-------------|-------------|
| Season | `time_of_year` | `time_year` |
| Presentation time | `presentation_time` | `time` |
| Other symptom text | `othersymptom_specify` | `symptom_other` |
| Medically important dx | `med_important_diagnosis___1-10` | `med_imp_dx___1-11` |
| Medically benign other | `med_benign_other_specify` | `med_benign_specify` |
| Migraine treatments | `migraine_treatments___1-3` | `migraine_tx___1-3` |
| Headache treatments | `headache_treatments___1-3` | `headache_tx___1-3` |

### 4.2 Restructured Fields (different data model)

| Concept | Year 1 | Year 2 | Harmonization |
|---------|--------|--------|---------------|
| **Headache specifiers** | `headache_specify___1-7` (Bilateral, Unilateral, Nausea, Vomiting, Photophobia, Phonophobia, Other) | `headache_specify___1-5` (Nausea, Vomiting, Photophobia, Phonophobia, Other) | Year 1 options 1-2 (Bilateral, Unilateral) moved to `current_headache` in Year 2. Map Year 1 `___3-6` to harmonized Nausea/Vomiting/Photophobia/Phonophobia. Map Year 1 `___1-2` to harmonized headache laterality variables. |
| **Current headache** | `current_headache___1-7` (Maximal at onset, Progressive days/weeks, Progressive months, Non-progressive, Abrupt onset, Gradual onset, Not described) | `current_headache___1-12` (same 7 + Bilateral, Unilateral, Frontal, Temporal, Occipital) | Common fields 1-7 map directly. Year 2 options 8-12 (laterality/location) have no Year 1 equivalent in this field, but Bilateral/Unilateral exist in Year 1's `headache_specify___1-2`. Frontal/Temporal/Occipital are Year 2 only. |
| **Investigations** | `investigations___1-10` (EEG, CT, LP, MRI, **Consultation**, **Admission**, Specific treatment, CTA, MRA, Cerebral arteriography) | `investigations___1-8` (EEG, CT, LP, MRI, Specific treatment, CTA, MRA, Cerebral arteriography) | Year 1 embeds Consultation (___5) and Admission (___6) as investigation options. Year 2 separates these into `consult` and `admit` fields. Harmonize: create unified `consult_any` and `admit_any` from respective sources. For imaging-only investigations, map Year 1 `___7` (Specific treatment) to harmonized `___5`, Year 1 `___8-10` (CTA, MRA, Cerebral arteriography) to harmonized `___6-8`. |
| **Consultation details** | `consult_specify` (free text) | `consult` (yes/no) + `consult_specify___1-8` (checkboxes: Pediatrics, Neurology, Ophthalmology, Neurosurgery, Cardiology, PICU, Psychiatry, Other) | Year 1 consultation details available as free text only; checkbox breakdown not possible without NLP parsing. |
| **Admission** | `admission_icu` (yes/no for ICU only) | `admit` (yes/no) + `admit_location___1-5` (Pediatric inpatient, PICU, Adult ICU, Direct to surgery, Other) | Year 1 only captures ICU admission (not ward). Create: `icu_any` = Year 1 `admission_icu` OR Year 2 `admit_location___2` or `___3`. Ward admission available Year 2 only. |
| **Discharge referrals** | `discharge_refer_specify` (free text) | `discharge_refer_specify___1-5` (checkboxes) | Year 1 is free text; Year 2 is structured. |

### 4.3 Fields Present in Only One Year

| Field | Present | Absent | Impact |
|-------|---------|--------|--------|
| `related` (visit related to study question) | Year 2 | Year 1 | Cannot apply this exclusion to Year 1; assume all Year 1 records passed manual screening |
| `dias_bp` (diastolic blood pressure) | Year 2 | Year 1 | **Cannot use as predictor in combined model**. Year 2-only analyses may include it. |
| `med_imp_dx___11` (Other medically important) | Year 2 | Year 1 | Year 1 has no "other" category; cases may be uncategorized or missing. Review Year 1 `diagnosis_specify` free text for potential reclassification. |
| `med_benign___6` (6th benign option) | Year 2 | Year 1 | Minor; Year 1 used "Other" (option 5) to capture overflow. |
| `epilepsy_describe___3` (Not described) | Year 2 | Year 1 | Minor; Year 1 had Focal/Generalized only. |
| `diff_dx` / `diffdx_specify` | Year 2 | Year 1 | Differential diagnosis tracking not available for Year 1. |

### 4.4 Record Counts and Exclusions

| | Year 1 | Year 2 | Combined |
|---|--------|--------|----------|
| Raw records | 1,124 | 1,205 | 2,329 |
| Excluded (trauma) | TBD | TBD | TBD |
| Excluded (not related) | N/A (no field) | TBD | TBD |
| Excluded (no doctor seen) | TBD | TBD | TBD |
| Excluded (missing gender) | TBD | TBD | 1 (per manuscript) |
| **Analytic cohort** | **TBD** | **TBD** | **2,153** (per manuscript) |

## 5. LLM-Assisted Free-Text Coding

The REDCap data contains substantial free-text content in "Other/specify" fields that was captured during chart review but never systematically coded. An LLM (e.g., Claude) can classify these entries into structured variables, with human review of uncertain cases.

### 5.1 Free-Text Field Inventory

| Field | Year 1 | Year 2 | Total | Priority | Coding Task |
|-------|--------|--------|-------|----------|-------------|
| `diagnosis_specify` | 928 | 940 | 1,868 | **High** | Validate/correct `med_imp_dx` and `med_benign` checkbox classification against free-text diagnosis |
| `othersymptom_specify` / `symptom_other` | 378 | 646 | 1,024 | **High** | Classify into `other_classify` categories (18 options) or flag new symptom variables |
| `prev_diagnoses_other` | 468 | 502 | 970 | **Medium** | Categorize into comorbidity groups (neurological, developmental, cardiac, respiratory, psychiatric, etc.) |
| `headache_other` | 258 | 91 | 349 | **Medium** | Extract additional headache features (e.g., dizziness, neck pain, visual aura) |
| `discharge_refer_specify` (Y1 free text) | 317 | — | 317 | **High** | Parse into Year 2 checkbox categories (Family physician, Neurology, Pediatric inpatient, Cardiology, Other) to harmonize |
| `consult_specify` (Y1 free text) | 198 | — | 198 | **High** | Parse into Year 2 checkbox categories (Pediatrics, Neurology, Ophthalmology, Neurosurgery, Cardiology, PICU, Psychiatry, Other) to harmonize |
| `med_benign_specify` | — | 157 | 157 | **Medium** | Recode into diagnostic categories; verify not misclassified medically important cases |
| `discharge_meds` (+ med2-5) | 87 | 163+ | 250+ | **Low** | Classify medications by therapeutic class |
| `diffdx_specify` | — | 64 | 64 | **Medium** | Cross-reference with final diagnosis for discordance analysis |
| `med_imp_specify` | — | 22 | 22 | **High** | Already maps to manuscript "other medically important" subtypes — verify completeness |
| `refer_other` | — | 41 | 41 | **Low** | Categorize referral specialty |
| `consult_other` | — | 18 | 18 | **Low** | Categorize consultation specialty |
| `treatment_specify` | 18 | 0 | 18 | **Low** | Classify treatment type |

### 5.2 LLM Coding Approach

```
scripts/01b-llm-coding.py
```

**Step 1: Automated classification.** For each free-text field, provide the LLM with:
- The target coding scheme (e.g., the 18 `other_classify` categories for symptom coding)
- The free-text entry
- Clinical context (age, gender, presenting symptoms) where helpful
- Instructions to return structured output (category code + confidence)

**Step 2: Confidence triage.** Partition results into:
- **High confidence (>90%):** Accept automated classification
- **Moderate confidence (70-90%):** Flag for batch human review
- **Low confidence (<70%):** Require individual human review

**Step 3: Human review.** PI reviews flagged cases. Output: reviewed classification + override flag.

**Step 4: Merge.** Integrate coded variables into the analytic dataset.

### 5.2b LLM Coding Methodology Reporting

**The use of LLM-assisted free-text coding is a novel methodological approach that will be scrutinized by reviewers.** The following must be reported in the manuscript (Methods section or Supplementary Methods):

1. **Model specification:** Claude Haiku 4.5 (`claude-haiku-4-5-20251001`), accessed via the Anthropic API. Haiku was selected for its speed and cost-efficiency on structured classification tasks; the smaller model size is appropriate given the well-defined coding schemes and confidence-based triage to PI review.
2. **Prompt engineering:** Provide representative prompts for each coding task (A-E) in a supplementary appendix. Include the system prompt, the structured output format requested, and any few-shot examples provided.
3. **Confidence calibration:** Describe how confidence scores (>90%, 70-90%, <70%) were assigned and how thresholds were determined.
4. **Agreement metrics:** Computed via `python 01b-llm-coding.py --agreement`. Among the 123 PI-reviewed cases (flagged subset, not full cohort): percent agreement = 41.5% (case-level), Cohen's kappa = 0.079. Confusion matrix: TP=42, FP=72, FN=0, TN=9. The low kappa reflects the LLM's deliberately conservative design: it over-flags benign cases as potentially MI (72 false positives) but never misses a true MI (0 false negatives). Task A (diagnosis discrepancies): 55.3% agreement; Task E (benign review): 32.9% agreement. These metrics apply only to the flagged subset; the full-cohort agreement is substantially higher since 94% of cases were unambiguous. Output: `outputs/table_llm_pi_agreement.csv`.
5. **Outcome impact:** Report the number and direction of outcome reclassifications (already documented in Section 5b.3).
6. **Limitations:** Acknowledge that LLM classifications are not independently validated against a gold-standard manual review of all 2,153 records (only the 124 flagged cases were reviewed by the PI). Discuss potential for systematic bias if the LLM consistently misclassifies certain diagnosis types.
7. **Reproducibility:** Note that LLM outputs are stochastic — running the same prompts may produce slightly different results. Document the temperature setting used (ideally 0 for maximum determinism). Archive all LLM outputs in `data/processed/llm_coded/`.

### 5.3 Priority Coding Tasks

**Task A — Outcome validation (`diagnosis_specify`, n=1,868):**
Cross-reference free-text ED diagnosis against `med_imp_dx` and `med_benign` checkboxes. Flag cases where the free text suggests a medically important diagnosis but no `med_imp_dx` checkbox was checked (potential missed cases), or vice versa. This is critical for outcome accuracy.

**Task B — Year 1 harmonization (`consult_specify`, `discharge_refer_specify`, n=515):**
Parse Year 1 free-text consultation and referral fields into the structured checkbox categories used in Year 2. This enables consistent cross-year analysis of consultation patterns and referral rates.

**Task C — Symptom enrichment (`othersymptom_specify`/`symptom_other`, n=1,024):**
Classify free-text symptoms into the existing `other_classify` checkbox scheme (18 categories). Additionally, identify recurring themes not captured by the existing scheme (e.g., dizziness/vertigo, neck pain/stiffness, fatigue/lethargy) that may warrant new binary variables.

**Task D — Comorbidity coding (`prev_diagnoses_other`, n=970):**
Classify prior medical history into clinically meaningful groups for use as potential confounders or predictors (e.g., neurological, developmental, psychiatric, cardiac, respiratory, autoimmune).

**Task E — Benign diagnosis review (`med_benign_specify`, n=157):**
Verify that cases classified as "other medically benign" are truly benign. Flag any that may represent medically important conditions misclassified as benign.

### 5.4 Medication Coding

The agent survey identified ~4,100 medication free-text entries across home medications (`meds_1` through `meds_other`, n=1,544), ED treatments (migraine IV/oral, headache IV/oral, seizure treatments, n=~2,100), EMS medications (n=118), and discharge medications (n=457). These follow a "Drug name + dose + route" pattern but with inconsistent formatting and abbreviations. LLM coding can extract:
- **Drug name** (standardized to generic)
- **Therapeutic class** (e.g., anticonvulsant, analgesic, triptan, benzodiazepine, NSAID, antiemetic)
- **Route** (PO, IV, PR, IN)
- **Dose** (numeric + unit)

This is lower priority than diagnostic and symptom coding but valuable for the practice variation analysis (Section 8.2).

## 5b. PI Review of Ambiguous Cases

After LLM coding, 124 ambiguous cases were flagged for PI adjudication. These were organized into two tiers and reviewed by the PI (pediatric neurologist).

### 5b.1 Review Tiers

| Tier | Task | Description | N cases |
|------|------|-------------|---------|
| 1 | A | MI checkbox set but diagnosis text appears benign (checkbox errors) | 12 |
| 1 | A | Benign checkbox set but LLM identified MI-consistent diagnosis | 36 |
| 1 | E | Benign diagnosis flagged by LLM as potentially MI | 26 |
| 2 | E | Moderate-confidence LLM flags for batch review | 50 |
| **Total** | | | **124** |

### 5b.2 Decision Principles

The PI established the following principles during review:

1. **Rule-out principle:** Diagnoses phrased as "r/o [condition]" are kept benign unless confirmatory testing (abnormal imaging, abnormal CSF) is documented elsewhere in the record.
2. **Symptom-only principle:** Isolated symptom-diagnoses (altered LOC, syncope, weakness NOS, gait disturbance) are kept benign without a confirmed MI etiology, as these represent discharge diagnoses where no serious cause was identified.
3. **Checkbox error correction:** When the MI checkbox is checked but the free-text diagnosis is clearly benign (seizure, headache, syncope, neuropathy), the checkbox is considered erroneous and the record is corrected to benign.
4. **New sub-categories:** Two clinically meaningful sub-categories were created within "Other MI" (code 11):
   - **IIH** (Idiopathic Intracranial Hypertension): pseudotumor cerebri, benign intracranial hypertension, papilledema
   - **Cerebellar ataxia:** cerebellitis, acute cerebellar ataxia, post-viral ataxia, ataxia NOS

### 5b.3 Outcome Impact

| Change | N records |
|--------|-----------|
| MI to Benign (checkbox errors) | 18 |
| Benign to MI (reclassified) | 48 |
| **Net MI change** | **+30** |
| Original MI count | 117 |
| **Final MI count** | **147** |

Final MI breakdown:

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

### 5b.4 Implementation

PI review decisions are applied programmatically by the `apply_pi_review()` function in `01-data-prep.py`, which:
1. Reads `data/processed/llm_coded/pi-review-decisions.csv`
2. Matches records by text (case-insensitive) against **both** `diagnosis_specify` and `med_benign_specify` fields, combined with current `outcome_med_important` value. Dual-field matching is necessary because Task E decisions reference `med_benign_specify` text, which can differ from `diagnosis_specify`.
3. For checkbox errors (Task A, keep_benign): clears all `med_imp_dx` checkboxes and sets outcome to 0
4. For reclassifications: sets the appropriate `med_imp_dx` checkbox and outcome to 1
5. Clears all `med_benign___` checkboxes on MI records (a record cannot be both MI and benign)
6. Tags records with `mi_subtype_iih` and `mi_subtype_cerebellar_ataxia` columns, using expanded keyword sets checked against both `diagnosis_specify` and `med_benign_specify`

### 5b.5 Audit Trail

All decisions are documented in `data/processed/llm_coded/pi-review-decisions.csv` with columns:
- `decision`: keep_benign, reclassify_mi_1, ..., reclassify_mi_11_IIH, reclassify_mi_11_cerebellar_ataxia
- `mi_code`: target MI code (1-11) for reclassifications
- `mi_label`: human-readable MI category label
- `diagnosis_text`: text matched (from `diagnosis_specify` or `med_benign_specify`)
- `n_records`: expected number of records matching this text
- `source_tier`: review tier (1 or 2)
- `source_task`: LLM coding task (A or E)
- `rationale`: clinical reasoning for the decision

## 6. REDCap Branching Logic

The CRF uses conditional display logic: fields are only shown (and therefore only populated) when a parent condition is met. **Empty values in branched fields are structurally absent, not missing data**, and must be coded as 0/absent/NA — never imputed.

### 6.1 Branching Dependencies

| Parent Condition | Child Fields Displayed |
|-----------------|----------------------|
| `trauma == 1` | `trauma_stop` (exclusion message) |
| `related == 0` (Y2 only) | `related_stop` (exclusion message) |
| `ems == 1` | `ems_meds` |
| `ems_meds == 1` | `ems_meds_specify` |
| `previous_diagnoses___1 == 1` (febrile seizure) | `prev_diagnosis_febrile___1-3` |
| `previous_diagnoses___6 == 1` (other) | `prev_diagnoses_other` |
| `symptoms___8 == 1` (seizure) | `seizure_number`, `seizure_duration` |
| `symptoms___10 == 1` OR `symptoms___11 == 1` (headache) | `current_headache___*` |
| `symptoms___11 == 1` (headache with other symptoms) | `headache_specify___*` |
| `symptoms___13 == 1` (other symptom) | `symptom_other`, `other_classify___*` |
| `headache_specify___5 == 1` (other headache) | `headache_other` |
| `see_doctor == 0` | `nodoc_stop` (exclusion message) |
| `diagnosis == 1` | `diagnosis_specify`, `med_imp_dx___*`, `med_benign___*` |
| `med_imp_dx___11 == 1` (other important) | `med_imp_specify` |
| `med_benign___3 == 1` (epilepsy) | `epilepsy_describe___*` |
| `med_benign___4 == 1` (febrile/afebrile seizure) | `febrile_describe___*` |
| `med_benign___3 == 1` OR `med_benign___4 == 1` | `seizure_treatment___*`, `seizure_tx_*` |
| `med_benign___2 == 1` (migraine) | `migraine_tx___*`, `migraine_oral`, `migraine_iv___*` |
| `med_benign___1 == 1` (headache) | `headache_tx___*`, `headache_oral`, `headache_iv___*` |
| `migraine_tx___2 == 1` (IV started) | `migraine_iv___*`, `migraine_iv_fluid`, `migraine_iv_med1/2` |
| `headache_tx___2 == 1` (IV started) | `headache_iv___*`, `headache_iv_fluid`, `headache_iv_med1/2` |
| `symptoms___10 == 1` OR `symptoms___11 == 1` | `ni_appropriateness` |
| Any investigation ordered | `results` |
| `results == 2` (abnormal) | `med_import` |
| `investigations___5 == 1` (specific treatment) | `treatment_specify` |
| `consult == 1` (Y2) | `consult_specify___*` |
| `admit == 1` (Y2) | `admit_location___*` |
| `admit_location___5 == 1` (other, Y2) | `admit_other` |
| `discharge_refer == 1` | `discharge_refer_specify___*` |
| `diff_dx == 1` (Y2) | `diffdx_specify` |

### 6.2 Implications for Analysis

1. **Missing vs. structurally absent:** When `symptoms___8 == 0` (no seizure), then `seizure_number` and `seizure_duration` are empty by design. These should be coded as 0 or NA, not treated as missing for imputation.
2. **Conditional treatment fields:** Migraine and headache treatment data only exists for patients diagnosed with those conditions. These cannot be used as predictors (they are post-diagnosis), only for descriptive analysis of practice variation.
3. **Headache feature availability:** Headache descriptors (`current_headache`, `headache_specify`) are only populated for patients with headache symptoms. For the headache sub-analysis (Section 8.7) these are key predictors; for the full-cohort analysis they should be coded as 0 when headache is not a presenting symptom.
4. **Nested branching:** Some fields have multi-level dependencies (e.g., `migraine_iv_med1` requires `med_benign___2 == 1` AND `migraine_tx___2 == 1` AND `migraine_iv___2 == 1`). The data prep script must respect this chain.

## 7. Outcome Definitions

### 7.1 Primary Outcome: Clinically Important Neurologic Presentation (CINP)

CINP is a decision-based composite outcome answering the clinical question: "Does this child require urgent neurodiagnostic investigation?" A case meets CINP criteria if **either** criterion is satisfied:

- **Criterion 1 (CIIP):** Confirmed intracranial pathology requiring treatment or ongoing management (see Section 7.2)
- **Criterion 2 (Objective finding):** Documented objective neurologic finding that mandates investigation to exclude serious intracranial pathology, even if no pathology is ultimately confirmed

**Qualifying objective neurologic findings for Criterion 2:**

| Finding | Column |
|---------|--------|
| Cerebellar ataxia | `sx_ataxia` |
| Focal weakness | `sx_focal_weakness` |
| Sensory disturbance | `sx_sensory` |
| Speech disturbance | `sx_speech` |
| Visual disturbance | `sx_visual` |
| Eye movement abnormality | `sx_eye_movement` |
| Altered consciousness (symptom) | `sx_altered_consciousness` |
| Altered LOC (exam) | `exam_altered_loc` |
| Papilledema | `exam_papilledema` |
| Meningismus | `exam_meningismus` |
| Seizure | `sx_seizure` |

**Circularity avoidance:** CINP is defined by the clinical presentation (documented findings), NOT by post-presentation decisions (imaging ordered, neurology consulted, admission). This prevents the outcome from being determined by the very decisions the model aims to support.

**Implementation:** The original `outcome_med_important` variable (MI, N=147) is numerically equivalent to CINP because all 121 CIIP cases qualify via Criterion 1, and all 26 cerebellar ataxia cases (which are the only MI cases not meeting CIIP) qualify via Criterion 2 (all have documented `sx_ataxia = 1`). This equivalence is verified by `scripts/21-cinp-framework.py`.

Binary outcome: `outcome_cinp` = `outcome_med_important` = 1 if CINP criteria met, else 0. (N=147, 6.8%)

**REDCap source fields:** The underlying diagnostic codes remain as follows:

| Code | Diagnosis | Year 1 field | Year 2 field |
|------|-----------|-------------|-------------|
| 1 | Stroke | `med_important_diagnosis___1` | `med_imp_dx___1` |
| 2 | Intracerebral haemorrhage | `med_important_diagnosis___2` | `med_imp_dx___2` |
| 3 | Vascular malformation | `med_important_diagnosis___3` | `med_imp_dx___3` |
| 4 | Vasculitis | `med_important_diagnosis___4` | `med_imp_dx___4` |
| 5 | Venous thrombosis | `med_important_diagnosis___5` | `med_imp_dx___5` |
| 6 | Demyelinating disorder | `med_important_diagnosis___6` | `med_imp_dx___6` |
| 7 | Brain and/or meningeal infection | `med_important_diagnosis___7` | `med_imp_dx___7` |
| 8 | Hydrocephalus | `med_important_diagnosis___8` | `med_imp_dx___8` |
| 9 | Neoplasm | `med_important_diagnosis___9` | `med_imp_dx___9` |
| 10 | Inborn error of metabolism | `med_important_diagnosis___10` | `med_imp_dx___10` |
| 11 | Other medically important | — | `med_imp_dx___11` |
| 12 | IIH (idiopathic intracranial hypertension) | — | `med_imp_dx___12` |
| 13 | Cerebellar ataxia | — | `med_imp_dx___13` |

Codes 12 and 13 are derived from code 11 sub-types by `apply_pi_review()` in 01-data-prep.py.

### 7.2 Secondary Outcome: Clinically Important Intracranial Pathology (CIIP)

CIIP is the strict pathology-confirmed endpoint: confirmed intracranial pathology requiring treatment or ongoing management. A case meets CIIP criteria if it has an MI diagnosis (any `med_imp_dx` checkbox) AND is NOT reclassified by the CIIP algorithm in `scripts/17-ciip-reclassification.py`.

**CIIP reclassification logic:** Cerebellar ataxia cases (code 13) are reclassified as non-CIIP unless:
- They have another MI diagnosis code (codes 1-12), OR
- They have both abnormal imaging AND specific treatment

Binary outcome: `outcome_ciip` = 1 if CIIP criteria met, else 0. (N=121, 5.6%)

**Relationship:** CIIP is a subset of CINP. All CIIP cases are CINP (via Criterion 1). CINP cases that are not CIIP (N=26) are cerebellar ataxia presentations meeting Criterion 2 only.

### 7.3 Other Secondary Outcomes

- **Abnormal neuroimaging:** `results == 2` (among those who had imaging)
- **Medically important abnormal imaging:** `med_import == 1`
- **ICU admission:** `icu_any` (harmonized, see 4.2)
- **Any admission:** Year 2 `admit == 1`; Year 1 derived from `investigations___6` or `admission_icu`

### 7.4 CINP Adjudication

`scripts/21-cinp-framework.py` generates:
- `data/processed/brain_attacks_cinp.parquet` — dataset with `outcome_cinp` and `outcome_ciip` columns
- `outputs/tables/table_cinp_adjudication.csv` — per-case adjudication (record_id, year, objective findings, imaging, therapy, CINP status, CIIP status, reason)
- `outputs/tables/table_cinp_vs_mi_vs_ciip.csv` — three-way metric comparison using existing analysis results
- `outputs/cinp_methods_text.md` — publication-ready rationale paragraph

## 8. Analysis Pipeline

### 8.1 Data Preparation (Rebuild from Source)

```
scripts/01-data-prep.py
```

**The combined analytic dataset used in the manuscript was assembled manually and should be recreated programmatically from the source REDCap exports.** This ensures reproducibility, documents all harmonization decisions, and provides an auditable pipeline from raw data to analytic dataset. The original manual process is not documented and cannot be verified.

1. **Import:** Load Year 1 and Year 2 raw CSVs separately
2. **Harmonize schemas** (per Section 4):
   - Rename fields to common names
   - Remap `headache_specify` checkbox codes (Year 1 ___1-2 to laterality; ___3-6 to nausea/vomiting/photophobia/phonophobia)
   - Remap `investigations` (extract consultation and admission from Year 1 investigations)
   - Remap `med_important_diagnosis` to `med_imp_dx`
   - Create `consult_any`, `admit_any`, `icu_any` from respective year-specific fields
3. **Eligibility exclusions:**
   - Exclude `trauma == 1` (both years)
   - Exclude `related == 0` (Year 2 only; Year 1 assumed pre-screened)
   - Exclude `see_doctor == 0` (both years)
4. **Outcome derivation:** Create `outcome_med_important` from harmonized medically important diagnosis checkboxes
5. **Feature engineering:**
   - Expand all checkbox fields into binary indicator columns with harmonized names
   - Confirm `age` calculated correctly (from `dob` and `date_of_visit`)
   - Z-score continuous variables (age, hr, temp, sys_bp) — **not** `dias_bp` (Year 2 only)
6. **Missing data:**
   - Report missingness per variable, stratified by year
   - Impute continuous variables with median (per manuscript: temp 172, hr 163, ctas 54 missing in benign group)
   - Remove records with missing gender (1 case per manuscript)
7. **Cohort tagging:** Add `year` column (1 or 2)
8. **Validation:** Compare rebuilt cohort counts against manuscript (N=2,153; 147 CINP after PI review, 121 CIIP) to confirm concordance. Flag and investigate any discrepancies.
9. **Export:** Combined analytic dataset + year-specific datasets as CSV and parquet to `data/processed/`

#### 8.1.1 Sample Size Justification (TRIPOD+AI Item 11)

Report the effective sample size relative to the number of candidate predictors and outcome events. For the primary analysis:

- **Events (CINP cases):** 147
- **Non-events (non-CINP):** 2,006
- **Candidate predictors (raw features):** 55 (11 symptom + 18 other symptom + 3 exam + 15 headache + 6 demographic + 2 other — per `get_feature_names()` in `scripts/11-stability.py`)
- **Events per variable (EPV) for 55 features:** 147 / 55 = 2.7

EPV < 10 indicates a risk of model overfitting for traditional logistic regression (Peduzzi et al., 1996). For the XGBoost model, the nested CV with SMOTE partially mitigates this via internal regularisation (max_depth, learning_rate constraints), reducing effective degrees of freedom below the nominal 55 features. For the clinical prediction rule (10 predictors), EPV = 147 / 10 = 14.7, which is adequate.

**Riley et al. (2020) minimum sample size (pmsampsize, R 4.5.2):**

The Riley calculation uses 33 effective parameters — the expected number after XGBoost's internal regularisation reduces the 55 raw features to approximately 33 effective degrees of freedom. This estimate is based on the observed feature importance distribution where ~33 features carry non-negligible importance weights.

Using `pmsampsize` with prevalence = 0.0683, target C-statistic = 0.90:

| Model | Parameters | Min N | Min events | EPP | Actual N | Actual events | Adequate? |
|-------|-----------|-------|------------|-----|----------|---------------|-----------|
| Full (33 effective) | 33 | 1,644 | 113 | 3.4 | 2,153 | 147 | Yes |
| Full (55 raw features) | 55 | 2,738 | 188 | 3.4 | 2,153 | 147 | No — shortfall of 41 events |
| Clinical score | 10 | 499 | 35 | 3.4 | 2,153 | 147 | Yes |
| Full (conservative, C=0.85) | 33 | 2,372 | 163 | 4.9 | 2,153 | 147 | No — shortfall of 16 events |

The actual sample (N=2,153, 147 events) meets the Riley criteria for the target C-statistic of 0.90 with 33 effective parameters. Under the full 55-feature count, the sample falls short (147 vs 188 required events), which must be acknowledged as a limitation. The traditional EPV for 55 features is 2.7, well below the conventional threshold of 10. XGBoost's internal regularisation and nested CV partially mitigate overfitting risk, but this remains the study's primary statistical limitation.

**Reference:** Riley RD, et al. Calculating the sample size required for developing a clinical prediction model. BMJ. 2020;368:m441. [Zotero: `H44ZUWJ6`]

#### 8.1.2 Missing Data Reporting (TRIPOD+AI Item 14)

The current pipeline handles missing data as follows:

- **Structurally absent values** (REDCap branching logic, Section 6): coded as 0 by `.fillna(0)` in analysis scripts. These are not truly missing and must not be imputed.
- **Truly missing continuous variables:** temp (172 missing), hr (163), CTAS (54) — imputed with median.
- **Missing gender:** 1 record excluded.

**Enhancements required for TRIPOD+AI compliance:**

1. **Table of missingness:** Generate a supplementary table reporting the number and percentage of missing values for each predictor and the outcome, stratified by year and outcome group. Add to `01-data-prep.py`.
2. **Missing data pattern:** Report whether missingness is MCAR/MAR/MNAR. For vitals (temp, hr), missingness likely correlates with acuity (MAR) — sicker patients more likely to have vitals recorded.
3. **Sensitivity analysis for imputation approach:** As a supplement, repeat the primary XGBoost analysis using:
   - (a) Complete-case analysis (exclude records with any missing predictor)
   - (b) Current approach (median imputation)
   - Report whether results are materially different. If so, consider multiple imputation (MICE) as the primary approach.

**Output:** `outputs/tables/tableS4_missing_data.csv`

#### 8.1.3 Participant Flow Diagram (TRIPOD+AI Item 20, STROBE Item 13)

Generate a CONSORT-style participant flow diagram showing the path from raw data to analytic cohort. The exclusion flow is already tracked programmatically in `01-data-prep.py` via the `flow` dictionary; this needs to be rendered as a figure.

**Flow structure:**

```
Year 1 raw (1,124) + Year 2 raw (1,205)
    = 2,329 total records
    |
    ├─ Excluded: trauma (n=TBD)
    ├─ Excluded: not related to study question (Year 2 only, n=TBD)
    └─ Excluded: no doctor seen (n=TBD)
    |
    = Analytic cohort (2,153)
        ├─ CINP after PI review: 147 (6.8%) [of which CIIP: 121 (5.6%)]
        └─ Non-CINP: 2,006 (93.2%)
        |
        ├─ Year 1 (derivation): n=1,043 (CINP=75)
        └─ Year 2 (validation): n=1,110 (CINP=72)
```

**Implementation:** Add flow diagram generation to `01-data-prep.py` or create a dedicated figure script using matplotlib. Populate TBD values from the `flow` dictionary.

**Output:** `outputs/figures/fig_participant_flow.png`

### 8.2 Descriptive Epidemiology (Objective 4)

```
scripts/02-descriptive.py
```

**Case ascertainment (RECORD Item 6.1):** The manuscript methods must describe how "brain attack" presentations were identified from the ED census. Specifically: (a) the screening criteria used to identify eligible presentations from the ED tracking system, (b) who performed the screening (research assistant, physician, automated), (c) the time period of enrollment, (d) whether screening was prospective or retrospective chart review, and (e) any validation of the screening process (e.g., sensitivity of the identification method). This information comes from the study protocol (PRO00055264) and should be reported verbatim in the Methods section.

1. **Demographics table (Table 1):** Age, gender, vitals, CTAS by CINP vs. non-CINP
   - Continuous: mean +/- SD, range; two-sample t-test
   - Categorical: n (%); chi-squared or Fisher exact test
   - Report overall and stratified by year
2. **Diagnosis distribution (Table 2):** Frequency of each CINP subtype (with CIIP status indicated)
3. **Symptom prevalence:** Frequency of each presenting symptom by outcome group
4. **Practice variation:**
   - Neuroimaging rates (CT, MRI, CTA, MRA) overall and by presenting symptom
   - EEG, LP ordering rates
   - Consultation rates (Year 2 by specialty; Year 1 overall only)
   - Admission rates: ICU (both years), ward (Year 2 only)
   - Treatments: seizure medications, migraine treatments, headache treatments
5. **Neuroimaging appropriateness:** For headache patients, cross-tabulate `ni_appropriateness` against imaging ordered and results
6. **Temporal patterns:** Presentations by season and time of day

### 8.3 Predictive Modeling — XGBoost (Objective 1, Completed)

```
scripts/03-prediction.py
```

**This analysis has been completed and is described in the manuscript draft.** The code should be archived here for reproducibility. Summary of the approach:

#### 8.3.1 Baseline Logistic Regression (completed)

- Univariate F-tests on combined data to rank features
- Top 5 features selected; multivariate logistic regression
- 10-fold stratified cross-validation
- Result: 94.8% accuracy, 4.3% sensitivity, 99.8% specificity

#### 8.3.2 XGBoost Classification (completed)

- Feature selection: k = {5, 15, 25, all} by F-test
- SMOTE resampling: ratios = {none, 0.5, 0.75, 1.0}
- Hyperparameter grid: `n_estimators` {50, 100, 150}, `max_depth` {3, 5, 7}, `learning_rate` {0.01, 0.1, 0.2}
- Inner 5-fold CV for parameter selection (F2-score), outer 10-fold for evaluation
- Result: 91.3% accuracy, 57.6% sensitivity, 92.7% specificity
- Top features: ataxia (0.19), altered consciousness exam (0.13), gender (0.10), altered consciousness symptom (0.09), fever (0.07)

#### 8.3.3 F-Score Beta Experiment (completed)

- Beta = {1.0, 1.5, 2.0, 2.5, 3.0}, 100 runs per beta
- False negatives decreased with increasing beta (61 -> 44) but plateaued
- False positives increased (62 -> 178)

#### 8.3.4 False Negative Analysis (completed)

- 19 cases always misclassified; 21 always correct
- Demyelinating disorders and neoplasms consistently misclassified
- Infections and strokes more reliably identified

#### 8.3.5 Calibration and Confidence Intervals (New — TRIPOD+AI Items 23a-b)

**The existing 03-prediction.py reports discrimination (AUROC, accuracy, sensitivity, specificity) but lacks calibration assessment and confidence intervals.** These must be added to comply with TRIPOD+AI.

**Calibration (add to 03-prediction.py or new script 09-calibration.py):**
1. **Calibration plot:** Predicted probability vs. observed frequency (10 decile bins). Plot for both XGBoost and logistic regression.
2. **Calibration-in-the-large (CITL):** Mean predicted probability vs. observed event rate. Values near 0 indicate good calibration.
3. **Calibration slope:** Logistic regression of outcome on linear predictor (log-odds of predicted probability). Slope = 1.0 is ideal; slope < 1.0 suggests overfitting.
4. **Brier score:** Mean squared difference between predicted probabilities and observed outcomes. Lower is better; compare to null model (prevalence-based).

**Confidence intervals (add to all analysis scripts):**

All performance metrics must be reported with 95% CIs. Use bootstrap resampling (1,000 iterations, stratified by outcome) to generate percentile CIs for:
- AUROC (alternatively, DeLong method for paired ROC comparison)
- Sensitivity, specificity, PPV, NPV (at the selected operating threshold)
- F2-score
- Brier score
- Calibration slope

**Outputs:**
- `outputs/figures/fig_calibration_xgboost.png` — calibration plot (XGBoost, 10-fold CV)
- `outputs/figures/fig_calibration_lr.png` — calibration plot (logistic regression, 10-fold CV)
- Updated `table_model_performance.csv` with CIs for all metrics

### 8.4 Temporal Validation (Objective 2, New)

```
scripts/04-validation.py
```

**This analysis is new and addresses the protocol's Objective 2.** The manuscript used only pooled cross-validation; temporal validation using the intended derivation/validation design has not been performed.

1. **Derivation:** Train XGBoost model on Year 1 data only using best parameters from 8.3.2
2. **Validation:** Apply to Year 2 data (narrow temporal validation from same population)
3. **Report:** accuracy, sensitivity, specificity, F2-score, AUROC, confusion matrix — all with 95% bootstrap CIs (1,000 iterations)
4. **Compare** Year 2 validation performance to pooled cross-validated performance
5. **Sensitivity analysis:** Repeat with logistic regression for comparison
6. **Case-mix comparison (TRIPOD+AI Item 22):** Compare Year 1 vs Year 2 baseline characteristics (demographics, symptom prevalence, CINP rate) using standardized mean differences. Report as a supplementary table. Any differences inform interpretation of validation performance (e.g., if CINP prevalence differs between years, calibration-in-the-large may shift).
7. **Calibration in validation:** Generate calibration plot for Year 2 predictions. Report calibration-in-the-large and calibration slope in the validation cohort. These are critical — a model can maintain discrimination (AUROC) while losing calibration when applied to new data.
8. **Optimism-corrected performance:** Report bootstrap optimism-corrected AUROC from the derivation cohort to estimate expected shrinkage in new data.

**Additional outputs:**
- `outputs/tables/tableS5_case_mix_y1_vs_y2.csv` — Year 1 vs Year 2 baseline comparison
- `outputs/figures/fig_calibration_temporal.png` — calibration plot in Year 2 validation

### 8.5 Predictors of Abnormal Neuroimaging (Objective 3, New)

```
scripts/05-neuroimaging.py
```

**This analysis is new and addresses the protocol's Objective 3.**

1. **Cohort:** Restrict to patients who had neuroimaging (CT, MRI, CTA, or MRA)
2. **Outcome:** Abnormal result (`results == 2`)
3. **Secondary outcome:** Medically important cause (`med_import == 1`)
4. **Predictors:** Same triage-available clinical features as primary analysis
5. **Model:** Logistic regression with stepwise variable selection
6. **Report:** Odds ratios with 95% CI, AUROC, sensitivity/specificity
7. **Neuroimaging yield:** Proportion with abnormal results by presenting symptom cluster

### 8.6 Clinical Prediction Rule (Objective 1, protocol)

```
scripts/06-prediction-rule.py
```

1. From final model, extract top predictors by feature importance
2. Weight by beta coefficient (per protocol: lowest coefficient = 1 point)
3. Compute AUROC and likelihood ratios at score thresholds — with 95% bootstrap CIs
4. Identify risk strata (low / moderate / high)
5. **Full model specification (TRIPOD+AI Item 24):** Report all logistic regression coefficients (betas), odds ratios with 95% CIs, and the intercept. This enables external validation by other centres. Present as a supplementary table.
6. **Calibration of the clinical score:** Generate calibration plot for the integer-weighted score. Report Brier score.
7. **Temporal validation of the score:** Apply the Year 1-derived score to Year 2 and report AUROC, calibration slope, and threshold-specific sensitivity/specificity.

**Output additions:**
- `outputs/tables/tableS6_full_model_coefficients.csv` — complete logistic regression model specification

### 8.7 Headache Sub-Analysis (Secondary, Extending IHC 2017)

```
scripts/07-headache.py
```

**This extends the IHC 2017 abstract analysis to the full 2-year dataset.** The original analysis used Year 1 only (n=630 headache cases) with stepwise logistic regression (66/33 split). The current analysis uses both years.

1. **Cohort:** Restrict to patients with headache as a presenting symptom (`symptoms___10 == 1` OR `symptoms___11 == 1`), excluding those with prior headache disorder (e.g., `previous_diagnoses___4` (migraine) or `previous_diagnoses___5` (recurring headache))
2. **Outcome:** CINP (same definition as primary outcome)
3. **Predictors** (replicating IHC 2017 model):
   - Demographics: age (<12 vs >=12), gender, EMS transport
   - History: prior headache
   - Examination: decreased consciousness, focal motor weakness, ataxia, speech impairment, visual impairment
   - Headache features: progressive over days, acute onset, vomiting
4. **Derivation (Year 1):** Stepwise logistic regression on Year 1 headache cohort (replicating IHC 2017 but using full Year 1, not 66/33 split)
5. **Validation (Year 2):** Apply derived model to Year 2 headache cohort
6. **Report:** ORs with 95% CI, AUROC (derivation + validation), sensitivity, specificity, PPV, NPV, false negative rate
7. **Comparison:** Compare 2-year model performance to original IHC 2017 results
8. **Neuroimaging impact:** Estimate reduction in imaging if model were applied (as per IHC 2017 conclusion)

### 8.8 Sensitivity Analysis — Outcome Definition Hierarchy

```
scripts/08-sensitivity.py
scripts/17-ciip-reclassification.py
scripts/18-ciip-ms1-reanalysis.py
scripts/19-ciip-ms2-reanalysis.py
scripts/20-ciip-comparison.py
scripts/21-cinp-framework.py
```

**Clinical rationale:** The study uses a hierarchical outcome framework:

| Outcome | Definition | N | Prevalence |
|---------|-----------|---|-----------|
| **CINP (primary)** | Requires urgent neurodiagnostic investigation (Criterion 1 OR 2) | 147 | 6.8% |
| **CIIP (secondary)** | Confirmed intracranial pathology requiring treatment | 121 | 5.6% |
| **CIIP excl. IIH** | CIIP excluding IIH (code 12, n=8) | 113 | 5.2% |

The 26 cases that are CINP but not CIIP are all cerebellar ataxia (code 13): documented ataxia mandates investigation (Criterion 2) but the condition is typically self-limiting (does not meet CIIP pathology threshold).

**Sensitivity analyses:**

1. **Primary (CINP, N=147):** All existing MI analyses serve as the CINP primary analysis (scripts 03-12). No reanalysis required because CINP = MI numerically.

2. **Sensitivity 1 — CIIP (N=121):** Full reanalysis excluding cerebellar ataxia cases that lack imaging+treatment criteria. Implemented in:
   - `scripts/17-ciip-reclassification.py` — creates `outcome_ciip` column and `brain_attacks_ciip.parquet`
   - `scripts/18-ciip-ms1-reanalysis.py` — reruns XGBoost CV and temporal validation on CIIP outcome
   - `scripts/19-ciip-ms2-reanalysis.py` — reruns penalized LR headache analysis on CIIP outcome
   - `scripts/20-ciip-comparison.py` — generates side-by-side CINP vs CIIP comparison tables

3. **Sensitivity 2 — CIIP excl. IIH (N=113):** Further excludes IIH cases. Implemented in `scripts/08-sensitivity.py`.

**Design decisions:**
- CINP = MI numerically, so no reanalysis of existing scripts is needed for the primary outcome
- CIIP is derived by `scripts/17-ciip-reclassification.py` from the MI parquet
- CINP adjudication is documented by `scripts/21-cinp-framework.py`
- Uses the same feature set, SMOTE ratios, and hyperparameter grid as the primary analysis

**Outputs:**
- `outputs/tables/table_cinp_vs_mi_vs_ciip.csv` — three-way metric comparison
- `outputs/tables/table_cinp_adjudication.csv` — per-case CINP adjudication
- `outputs/tables/table_ciip_ms1_performance.csv` — CIIP XGBoost performance
- `outputs/tables/table_ciip_ms2_pen_performance.csv` — CIIP penalized LR performance
- `outputs/tables/table_master_mi_vs_ciip.csv` — comprehensive comparison
- `outputs/tables/table_sensitivity_analysis.csv` — CIIP excl. IIH comparison
- `outputs/figures/fig_sensitivity_comparison.png` — grouped bar chart

### 8.9 Calibration Assessment (New — TRIPOD+AI Item 23b)

```
scripts/09-calibration.py
```

**This is a new cross-cutting analysis that generates calibration metrics and plots for all prediction models.** Calibration assesses whether predicted probabilities match observed event rates — a model can discriminate well (high AUROC) yet produce systematically biased probability estimates.

**Models assessed:**
1. XGBoost (10-fold CV, pooled out-of-fold predictions from 03-prediction.py)
2. Logistic regression (10-fold CV)
3. XGBoost temporal validation (Year 1 → Year 2)
4. Logistic regression temporal validation (Year 1 → Year 2)
5. Clinical prediction score (06-prediction-rule.py)
6. Headache sub-analysis model (07-headache.py, derivation + validation)

**Metrics for each:**
- **Calibration plot:** Predicted probability (x-axis) vs. observed proportion (y-axis) in decile bins, with ideal line and LOESS smoother
- **Calibration-in-the-large (CITL):** log(O/E) where O = observed events, E = sum of predicted probabilities. CITL = 0 is ideal.
- **Calibration slope:** Coefficient from logistic regression of outcome on linear predictor. Slope = 1.0 is ideal; < 1.0 indicates overfitting.
- **Brier score:** Mean(predicted - observed)^2. Report alongside the null Brier score (prevalence * (1 - prevalence)) and scaled Brier score.
- **E:O ratio (expected-to-observed):** Total predicted / total observed events.

**Calibration method rationale:** We use calibration slope, CITL, and calibration plots (binned + LOESS) rather than the Hosmer-Lemeshow (H-L) goodness-of-fit test. The H-L test is no longer recommended for prediction model evaluation because it is oversensitive in large samples (rejecting well-calibrated models), provides no measure of calibration magnitude, and depends on arbitrary grouping choices (Van Calster et al., BMC Med 2019 [Zotero: `2YAI7UYZ`]). Calibration slope and CITL provide continuous, interpretable measures of calibration that directly inform model updating decisions.

**Bootstrap confidence intervals (1,000 iterations):** For AUROC, sensitivity, specificity, PPV, NPV, Brier score, calibration slope. Use stratified resampling to maintain outcome prevalence in each bootstrap sample.

**Outputs:**
- `outputs/figures/fig_calibration_combined.png` — 2x3 panel of calibration plots for all models
- `outputs/tables/table_calibration_metrics.csv` — CITL, slope, Brier, E:O for all models
- Updated performance tables in scripts 03, 04, 06, 07 to include CIs

### 8.10 Decision Curve Analysis (New — TRIPOD+AI Item 23d)

```
scripts/10-decision-curve.py
```

**Clinical utility assessment using decision curve analysis (Vickers & Elkin, 2006).** This evaluates the net benefit of using the prediction model to guide clinical decisions (e.g., ordering neuroimaging) across a range of threshold probabilities, compared to the default strategies of "investigate all" and "investigate none."

**Rationale:** With CINP prevalence of 6.8%, the clinical question is: at what threshold probability of CINP should a clinician order neuroimaging or escalate care? Decision curve analysis answers this by plotting net benefit vs. threshold probability.

**Models compared:**
1. XGBoost (primary model)
2. Logistic regression (baseline)
3. Clinical prediction score (simplified)
4. Investigate-all strategy (current practice)
5. Investigate-none strategy (reference)

**Analysis:**
1. For each threshold probability t from 0.01 to 0.50:
   - Classify patients as positive (predicted probability >= t) or negative
   - Net benefit = (true positives / N) - (false positives / N) * (t / (1 - t))
2. Plot net benefit vs. threshold probability for all strategies
3. Identify the range of clinically relevant thresholds where the model provides net benefit over default strategies
4. Report the net benefit at key clinical thresholds (e.g., 5%, 10%, 20%)

**Reference:** Vickers AJ, Van Calster B, Steyerberg EW. Net benefit approaches to the evaluation of prediction models, molecular markers, and diagnostic tests. BMJ. 2016;352:i6. (Zotero key: `KLZA8XGJ`)

**Outputs:**
- `outputs/figures/fig_decision_curve.png` — decision curve analysis plot
- `outputs/tables/table_net_benefit.csv` — net benefit at key thresholds

### 8.11 Model Stability and Collinearity Assessment (New — PROBAST Analysis Domain)

```
scripts/11-stability.py
```

**Addresses PROBAST signaling questions on model complexity and instability.**

1. **Predictor collinearity:** Generate a correlation matrix (Pearson for continuous, phi coefficient for binary) for all candidate predictors. Report variance inflation factors (VIF) for the logistic regression model. Flag predictor pairs with |r| > 0.7 or VIF > 5.
2. **Bootstrap model stability (XGBoost):** In each of 200 bootstrap samples, refit the XGBoost model and record:
   - Feature importance rankings (are the same features consistently selected?)
   - Performance metrics (how variable is AUROC across resamples?)
   - Report the selection frequency for each predictor (% of bootstraps where it appears in the top 10)
3. **Optimism correction:** Calculate bootstrap optimism-corrected AUROC for both the XGBoost and logistic regression models. This estimates the expected performance degradation in new data.

**Outputs:**
- `outputs/figures/fig_predictor_correlation.png` — correlation heatmap
- `outputs/tables/tableS7_vif.csv` — VIF values for logistic regression predictors
- `outputs/tables/tableS8_bootstrap_stability.csv` — predictor selection frequency across bootstraps

### 8.12 Cost-Informed Threshold Analysis (Manuscript 1)

```
scripts/12-cost-analysis.py
```

**Extends the decision curve analysis (Section 8.10) by incorporating investigation costs.** Decision curves show net benefit in probability units; this analysis translates model performance into healthcare dollars — how much does it cost per CINP case detected at each threshold, and what is the net monetary benefit of model-guided investigation versus investigate-all?

**Cost parameters:**

Investigation costs are estimated from the Alberta Schedule of Medical Benefits (SOMB) professional fees, with two additional cost scenarios for sensitivity analysis:

| Investigation | HSC Code | SOMB Section | Base (Prof. fee) | Mid (×2.5 total) | High (CADTH) |
|---|---|---|---|---|---|
| CT head | X125A / X125 | Diagnostic Radiology – CT | $95 | $310 | $490 |
| MRI brain | X482A / X482 | Diagnostic Radiology – MRI | $230 | $762 | $990 |
| CTA head | X127 / X126 | Diagnostic Radiology – CT | $140 | $465 | $670 |
| MRA head | X484 | Diagnostic Radiology – MRI | $300 | $990 | $1,300 |
| LP | 13.99C | General Procedures | $80 | $200 | $350 |
| EEG | X600 | Diagnostic Services | $80 | $250 | $400 |
| Consultation (Y2 only) | 03.08A | Consultation Services | $80 | $200 | $350 |

Pediatric modifier: +30% for patients ≤12 years (some diagnostic radiology codes).

**Three cost scenarios:**
1. **Base:** Estimated professional fee component (~30% of institutional cost)
2. **Mid:** CADTH 2023 published average institutional cost
3. **High:** CADTH upper range / private facility pricing

**Sources:** CT/MRI costs from CADTH Health Technology Review, Oct 2023 ("Use of MRI and CT in Private Imaging Facilities in Canada, 2021-2023"; mean CT=$310, MRI=$762, from facilities in BC, AB, SK, QC). CTA/MRA estimated with contrast/angiography premium. LP/EEG estimated from provincial fee schedules. Consultation from Alberta SOMB 03.08A (~$79). For Alberta-specific professional fees, consult the [Alberta SOMB](https://open.alberta.ca/publications/somb) or [AMA Fee Navigator](https://apps.albertadoctors.org/fee-navigator/).

**Key analyses:**

1. **Observed investigation mix:** From `inv_ct`, `inv_mri`, `inv_lp`, `inv_eeg`, `inv_cta`, `inv_mra` — proportion receiving each investigation type among investigated vs. all patients.
2. **Cost at each threshold (1–50%):** For XGBoost, LR, and clinical score: N flagged, mean investigation cost per flagged patient, total investigation cost, CINP detected, CINP missed, cost per CINP detected.
3. **Net monetary benefit:** NMB(t) = (TP/N) × W_tp − (FP/N) × C_inv, across all thresholds and cost scenarios.

**Consultation cost handling:** Subspecialty consultation data is available for Year 2 only. For Year 1, patients with `consult == 1` are assigned the Year 2 mean consultation cost. This is reported as a supplementary sensitivity analysis; the main analysis uses investigation costs only.

**Outputs:**

| File | Type | Content |
|------|------|---------|
| `table_cost_parameters.csv` | Table | Fee schedule, HSC codes, cost scenarios |
| `table_cost_thresholds.csv` | Table (main) | Cost at key thresholds (2%, 5%, 10%, 15%, 20%, 30%) per model |
| `tableS11_cost_sensitivity.csv` | Supp table | Results under 3 cost scenarios |
| `tableS12_investigation_mix.csv` | Supp table | Observed investigation rates |
| `fig_nmb_curve.png` | Figure (main) | Net monetary benefit vs. threshold, 3 models + investigate-all |
| `fig_cost_per_case.png` | Figure (main) | Cost per CINP detected vs. threshold |

### 8.13 Extended Headache Analysis (Manuscript 2)

```
scripts/13-headache-deep.py
```

**Deep headache-specific analysis for the companion paper (Headache / Cephalalgia).** Extends the brief headache sub-analysis in Section 8.7 (which produces 1 table + 1 figure for Manuscript 1) into a full headache paper covering imaging appropriateness, extended prediction modeling, and cost analysis.

**Key analyses:**

1. **Cohort selection:** Replicate from Section 8.7: `sx_any_headache == 1`, exclude prior headache disorder. Split Year 1 (derivation) / Year 2 (validation).
2. **Expanded demographics:** Full headache cohort characteristics including investigation rates, consultation rates (Year 2), imaging appropriateness distribution, CINP subtype breakdown.
3. **Imaging appropriateness analysis:** Cross-tabulate `ni_appropriateness` (1=Usually NOT appropriate, 2=May be appropriate, 3=Usually appropriate) against:
   - `any_neuroimaging` (was imaging done?)
   - `outcome_abnormal_imaging` (was it abnormal?)
   - `outcome_med_important` (was the case CINP?)
   - Compute overtesting/undertesting rates, chi-squared tests.
4. **Extended prediction model:** Use `EXTENDED_PREDICTORS` (21 vars) from Section 8.7. Add XGBoost comparison (new — 07-headache.py uses LR only). Derivation Year 1, validation Year 2. Bootstrap CIs.
5. **Headache decision curve:** Net benefit for headache LR model vs. investigate-all-headache vs. investigate-none. Threshold range 1–30%.
6. **Headache calibration:** Following pattern from Section 8.9.
7. **Headache cost analysis:** Same cost framework as Section 8.12 applied to headache subgroup. Key question: "How many imaging studies could be avoided, at what cost saving, detecting what % of CINP?"

**Outputs:**

| File | Type | Content |
|------|------|---------|
| `table_ha_demographics_extended.csv` | Table | Extended headache cohort characteristics |
| `table_ha_appropriateness.csv` | Table | Imaging appropriateness cross-tabulation |
| `table_ha_model_extended.csv` | Table | Full model coefficients + validation metrics |
| `table_ha_decision_curve.csv` | Table | Net benefit at key thresholds |
| `table_ha_cost.csv` | Table | Cost analysis for headache imaging |
| `fig_ha_appropriateness.png` | Figure | Imaging rates by appropriateness category |
| `fig_ha_roc_extended.png` | Figure | ROC curves (LR + XGBoost, derivation + validation) |
| `fig_ha_calibration.png` | Figure | Calibration plot |
| `fig_ha_decision_curve.png` | Figure | Decision curve |
| `fig_ha_cost_threshold.png` | Figure | Cost per CINP detected vs. threshold |

## 9. Figures and Tables

### Two-Manuscript Structure

**Manuscript 1** (Pediatrics, AAP): Primary brain attack prediction — all current analyses (scripts 01–12) including cost-informed threshold analysis, with a brief headache section (script 07).

**Manuscript 2** (Headache / Cephalalgia): Deep headache-specific prediction, imaging appropriateness, and cost analysis (script 13). References Manuscript 1 for full cohort methods.

### Manuscript 1 — Primary Brain Attack Prediction (Pediatrics, AAP)

| # | Type | Content | Script |
|---|------|---------|--------|
| Figure 1 | Flow diagram | Participant flow from raw data to analytic cohort | 01 |
| Table 1 | Table | Demographics and clinical variables by outcome group | 02 |
| Table 2 | Table | Diagnosis distribution among medically important cases | 02 |
| Figure 2 | Bar chart | Model performance: XGBoost vs logistic regression (with CIs) | 03 |
| Figure 3 | Horizontal bar | Feature importance values across 10 folds | 03 |
| Table 3 | Table | Temporal validation performance (Year 1 → Year 2, with CIs) | 04 |
| Figure 4 | ROC curves | Temporal validation: derivation vs validation | 04 |
| Figure 5 | Calibration | Calibration plots: XGBoost and LR (CV + temporal validation) | 09 |
| Figure 6 | Decision curve | Net benefit vs. threshold probability | 10 |
| Table 4 | Table | Predictors of abnormal neuroimaging (logistic regression) | 05 |
| Table 5 | Table | Neuroimaging yield by presenting symptom | 05 |
| Table 6 | Table | Headache sub-analysis: prediction model (2-year) | 07 |
| Figure 7 | ROC curves | Headache model: derivation vs validation | 07 |
| Table 7 | Table | Clinical prediction score weights and threshold performance | 06 |
| Table 8 | Table | Sensitivity analysis: CINP vs CIIP vs CIIP-excl-IIH | 08, 17-21 |
| Table 9 | Table | Cost at key thresholds per model | 12 |
| Figure 8 | Line plot | Net monetary benefit vs. threshold | 12 |
| Figure 9 | Line plot | Cost per CINP detected vs. threshold | 12 |

### Additional Main Figures

| # | Type | Content | Script |
|---|------|---------|--------|
| Figure 10 | Line plot | F-score beta experiment: false positives vs false negatives | 03 |
| Figure 11 | Grouped bar | Symptom profiles: always-misclassified vs always-correct | 03 |
| Figure 12 | Grouped bar | Sensitivity analysis: metric comparison | 08 |

### Manuscript 1 Supplementary

| # | Type | Content | Script |
|---|------|---------|--------|
| Table S1 | Table | Complete univariate associations for all predictors | 02 |
| Table S2 | Table | Practice variation: imaging, consults, admissions, treatments | 02 |
| Table S3 | Table | Data harmonization mapping between Year 1 and Year 2 CRFs | 01 |
| Table S4 | Table | Missing data: count and percentage by predictor, year, outcome | 01 |
| Table S5 | Table | Case-mix comparison: Year 1 vs Year 2 baseline characteristics | 04 |
| Table S6 | Table | Full model specification: LR coefficients, ORs, intercept | 06 |
| Table S7 | Table | Variance inflation factors for logistic regression predictors | 11 |
| Table S8 | Table | Bootstrap stability: predictor selection frequency | 11 |
| Table S9 | Table | Calibration metrics: CITL, slope, Brier score for all models | 09 |
| Table S10 | Table | Net benefit at key clinical thresholds | 10 |
| Table S11 | Table | Cost sensitivity analysis: results under 3 cost scenarios | 12 |
| Table S12 | Table | Observed investigation rates and costs | 12 |
| Figure S1 | Heatmap | Predictor correlation matrix | 11 |
| Figure S2 | Calibration | Calibration plot: Year 2 temporal validation | 09 |
| Figure S3 | Calibration | Calibration plot: headache model (derivation + validation) | 09 |
| Checklist S1 | Checklist | Completed TRIPOD+AI checklist with page numbers | — |

### Manuscript 2 — Extended Headache Analysis (Headache / Cephalalgia)

| # | Type | Content | Script |
|---|------|---------|--------|
| Table 1 | Table | Extended headache cohort characteristics | 13 |
| Table 2 | Table | Imaging appropriateness cross-tabulation | 13 |
| Table 3 | Table | Extended prediction model coefficients + validation | 13 |
| Table 4 | Table | Decision curve: net benefit at key thresholds | 13 |
| Table 5 | Table | Cost analysis for headache imaging | 13 |
| Figure 1 | Grouped bar | Imaging rates by appropriateness category | 13 |
| Figure 2 | ROC curves | LR + XGBoost, derivation + validation | 13 |
| Figure 3 | Calibration | Headache model calibration plot | 13 |
| Figure 4 | Decision curve | Headache decision curve | 13 |
| Figure 5 | Line plot | Cost per CINP detected vs. threshold (headache) | 13 |

## 10. Software and Reproducibility

- **Language:** Python 3.11+
- **Key packages:** pandas, scikit-learn, xgboost, imbalanced-learn (SMOTE), matplotlib, seaborn, tableone
- **Additional packages (new analyses):** `scikit-learn` (calibration_curve, brier_score_loss), `statsmodels` (VIF, logistic calibration slope), `dcurves` or custom implementation (decision curve analysis)
- **Data location:** `data/` (gitignored; copied from `/Volumes/fomd-richerlab/data/brain_attacks`)
- **Outputs:** All figures and tables exported to `outputs/`
- **Random seed:** 42 (set globally)
- **Bootstrap seed:** 42 (for reproducible CIs; 1,000 iterations unless otherwise specified)

## 11. Project Directory Structure

```
brain-attacks/
├── analysis-plan.md            # This document
├── config.yml                  # Project configuration
├── README.md                   # Project overview and file index
├── scripts/
│   ├── 01-data-prep.py         # Data import, harmonization, PI review, feature engineering
│   ├── 01b-llm-coding.py      # LLM-assisted free-text classification
│   ├── 02-descriptive.py       # Epidemiology tables and figures
│   ├── 03-prediction.py        # XGBoost + logistic regression
│   ├── 04-validation.py        # Year 1 -> Year 2 temporal validation
│   ├── 05-neuroimaging.py      # Predictors of abnormal neuroimaging
│   ├── 06-prediction-rule.py   # Clinical score derivation
│   ├── 07-headache.py          # Headache sub-analysis extending IHC 2017
│   ├── 08-sensitivity.py       # Sensitivity analysis: CIIP excl. IIH outcome definition
│   ├── 09-calibration.py       # Calibration assessment + bootstrap CIs (TRIPOD+AI)
│   ├── 10-decision-curve.py    # Decision curve analysis / clinical utility
│   ├── 11-stability.py         # Model stability, collinearity, optimism correction
│   ├── 12-cost-analysis.py     # Cost-informed threshold analysis (Manuscript 1)
│   ├── 13-headache-deep.py     # Extended headache analysis (Manuscript 2)
│   ├── 15-headache-figures.py  # Headache manuscript figures
│   ├── 16-penalized-lr-headache.py # Penalized LR headache model (Manuscript 2)
│   ├── 17-ciip-reclassification.py # CIIP outcome reclassification
│   ├── 18-ciip-ms1-reanalysis.py   # CIIP reanalysis for Manuscript 1
│   ├── 19-ciip-ms2-reanalysis.py   # CIIP reanalysis for Manuscript 2
│   ├── 20-ciip-comparison.py       # CINP vs CIIP comparison tables
│   ├── 21-cinp-framework.py        # CINP adjudication framework + methods text
│   └── qc-mi-classification.py # Line-by-line MI classification QC audit
├── outputs/
│   ├── tables/                 # CSV tables + QC reports (see README.md for full list)
│   └── figures/                # PNG figures (see README.md for full list)
├── data/                       # .gitignored
│   ├── raw_data/               # Source REDCap exports
│   ├── labelled_data/          # REDCap exports with labels
│   ├── metadata/               # Data dictionaries
│   └── processed/
│       ├── brain_attacks_combined.csv
│       ├── brain_attacks_combined.parquet
│       ├── brain_attacks_ciip.parquet      # CIIP-reclassified dataset (script 17)
│       ├── brain_attacks_cinp.parquet      # CINP-adjudicated dataset (script 21)
│       └── llm_coded/
│           ├── task_a_diagnosis.csv       # LLM outcome validation
│           ├── task_b_consult.csv         # LLM consultation coding
│           ├── task_b_referral.csv        # LLM referral coding
│           ├── task_c_symptom.csv         # LLM symptom classification
│           ├── task_d_comorbidity.csv     # LLM comorbidity coding
│           ├── task_e_benign_review.csv   # LLM benign review
│           ├── human_review.csv           # Combined LLM flags for review
│           ├── pi_review_focused.csv      # 124 ambiguous cases with PI decisions
│           └── pi-review-decisions.csv    # Audit trail with rationale
└── .gitignore
```

## 12. Analysis Priority

| Priority | Script | Analysis | Rationale |
|----------|--------|----------|-----------|
| 1 | 01 | Data preparation + harmonization + flow diagram | Everything depends on this |
| 1b | 01b | LLM-assisted free-text coding | Enriches predictors, validates outcomes, harmonizes Y1 fields |
| 2 | 02 | Descriptive epidemiology | Needed for manuscript Table 1 |
| 3 | 03 | Archive existing ML code | Document completed work |
| 4 | 04 | Temporal validation + case-mix comparison | Addresses protocol Objective 2; strengthens manuscript |
| 5 | 09 | Calibration assessment + bootstrap CIs | TRIPOD+AI compliance; applies to all models |
| 6 | 10 | Decision curve analysis | TRIPOD+AI clinical utility; strengthens clinical relevance |
| 7 | 11 | Model stability + collinearity | PROBAST compliance; addresses reviewer concerns |
| 8 | 07 | Headache sub-analysis | Extends IHC 2017; potential secondary paper |
| 9 | 05 | Neuroimaging predictors | Addresses protocol Objective 3 |
| 10 | 06 | Clinical prediction rule + full model specification | Translational output |
| 11 | 08 | Sensitivity analysis | Robustness across CINP/CIIP/CIIP-excl-IIH definitions |
| 12 | 12 | Cost-informed threshold analysis | Translates model utility into healthcare dollars (Manuscript 1) |
| 13 | 13 | Extended headache analysis | Deep headache paper: appropriateness, cost, prediction (Manuscript 2) |
| 14 | 15 | Headache manuscript figures | Figures for Manuscript 2 |
| 15 | 16 | Penalized LR headache model | Ridge-penalized LR for headache subgroup (Manuscript 2) |
| 16 | 17 | CIIP reclassification | Reclassify cerebellar ataxia for CIIP secondary outcome |
| 17 | 18 | CIIP MS1 reanalysis | Rerun XGBoost + clinical score on CIIP outcome |
| 18 | 19 | CIIP MS2 reanalysis | Rerun penalized LR headache on CIIP outcome |
| 19 | 20 | CINP vs CIIP comparison | Generate master comparison tables |
| 20 | 21 | CINP framework | CINP adjudication, three-way comparison, methods text |

## 13. Key Limitations to Address in Manuscript Discussion

The following limitations should be explicitly discussed in the manuscript, per TRIPOD+AI Item 26 and PROBAST:

1. **Single-centre study:** All data from the Stollery Children's Hospital ED. Geographic and institutional generalizability is unknown. External validation at other paediatric centres is needed before clinical implementation.
2. **Low events per variable (EPV):** With 147 CINP events and 55 candidate features, EPV = 2.7 for the full model. This raises concerns about overfitting, partially mitigated by XGBoost's internal regularisation (reducing effective degrees of freedom to ~33), nested CV with SMOTE, and the simplified 10-predictor clinical score (EPV = 14.7). The bootstrap optimism correction (Section 8.11) quantifies the expected performance shrinkage. The Riley (2020) sample size criterion is met for 33 effective parameters but not for the full 55 raw features.
3. **Retrospective design with chart-review data:** Predictor and outcome ascertainment depends on documentation quality. Predictors not documented in the chart are coded as absent, which may misclassify subtle findings. The CINP outcome relies on the final ED diagnosis and documented objective findings, not a standardized adjudication process — though the LLM-assisted review with PI adjudication (Section 5b) substantially strengthened outcome accuracy.
4. **Temporal validation only (not external):** Year 1 → Year 2 validation from the same institution provides narrow temporal validation but does not test geographic or case-mix generalizability. The single-centre design means unmeasured institutional factors (e.g., referral patterns, diagnostic thresholds) may influence model performance.
5. **Missing data:** Median imputation used for vitals (temp, hr, CTAS). Missingness may not be MCAR — sicker patients may be more likely to have vitals recorded. Sensitivity analysis (Section 8.1.2) will test the impact of the imputation approach.
6. **LLM-assisted outcome classification:** Novel methodology with limited precedent. Only 124/2,153 cases (5.8%) were reviewed by the PI; the remaining 94% rely on the LLM-validated checkbox data. Systematic LLM errors could bias the outcome if certain diagnosis types are consistently misclassified.
7. **Outcome definition sensitivity:** The CINP framework includes cerebellar ataxia (Criterion 2: objective neurologic finding mandating investigation) even though these cases do not meet the stricter CIIP pathology threshold. The CIIP secondary analysis (Section 8.8) tests robustness to this definitional choice. The inclusion of IIH is also clinically debatable; the CIIP-excl-IIH analysis addresses this.
8. **Class imbalance:** CINP prevalence of 6.8% creates challenges for classifier sensitivity. Despite SMOTE and F2-score optimization, the XGBoost model achieves only ~58% sensitivity — meaning ~42% of CINP cases are missed. This has important clinical implications that must be discussed transparently.
9. **Cost analysis uses estimated costs:** The cost-informed threshold analysis (Section 8.12) uses Alberta SOMB professional fees as a proxy for total investigation costs. True institutional costs include facility fees, technical costs, and supplies not captured in physician billing. The three-scenario sensitivity analysis (base, mid, high) partially addresses this, but real-world costs will vary by centre. Downstream costs (treatment, follow-up, missed diagnoses) are not modelled.
