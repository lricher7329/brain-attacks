"""
01b-llm-coding.py — LLM-Assisted Free-Text Classification

Uses Claude API to classify free-text entries from REDCap "Other/specify"
fields into structured analytic variables.

See analysis-plan.md Section 5 for full documentation.

Tasks:
  A — Outcome validation: classify diagnosis_specify, flag checkbox discrepancies
  B — Year 1 harmonization: parse consult_specify & discharge_refer_specify
  C — Symptom enrichment: classify symptom_other into other_classify categories
  D — Comorbidity coding: classify prev_diagnoses_other into clinical groups
  E — Benign review: verify med_benign_specify classifications

Inputs:
    data/processed/brain_attacks_combined.csv  (from 01-data-prep.py)

Outputs:
    data/processed/llm_coded/task_*.csv        (per-task cached classifications)
    data/processed/llm_coded/human_review.csv  (uncertain cases for PI review)
    data/processed/brain_attacks_llm_coded.csv  (updated combined dataset)
    data/processed/brain_attacks_llm_coded.parquet

Usage:
    python 01b-llm-coding.py                  # Run all tasks + merge
    python 01b-llm-coding.py --task a b       # Run specific tasks
    python 01b-llm-coding.py --dry-run        # Survey data, estimate cost
    python 01b-llm-coding.py --merge-only     # Merge existing cached results
    python 01b-llm-coding.py --force          # Re-run ignoring cache

Requires:
    pip install anthropic pandas pyarrow
    export ANTHROPIC_API_KEY=sk-ant-...
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
PROCESSED = DATA / "processed"
LLM_DIR = PROCESSED / "llm_coded"
LLM_DIR.mkdir(parents=True, exist_ok=True)

COMBINED = PROCESSED / "brain_attacks_combined.csv"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
BATCH_SIZE = 20
MAX_RETRIES = 3
RETRY_DELAY = 2.0


# ===================================================================
# CODING SCHEMES
# ===================================================================

MI_CATEGORIES = {
    1: "Stroke (arterial ischemic stroke, AIS)",
    2: "Intracerebral hemorrhage (non-traumatic ICH, intraparenchymal)",
    3: "Vascular malformation (AVM, aneurysm, cavernoma, Moyamoya)",
    4: "Vasculitis (CNS vasculitis, cerebral vasculitis)",
    5: "Venous thrombosis (CVST, cerebral venous sinus thrombosis)",
    6: "Demyelinating disorder (ADEM, MS, transverse myelitis, optic neuritis, NMO)",
    7: "Brain/meningeal infection (meningitis, encephalitis, brain abscess)",
    8: "Hydrocephalus (including VP shunt malfunction, shunt obstruction)",
    9: "Neoplasm (brain tumor, CNS tumor, intracranial mass, craniopharyngioma)",
    10: "Inborn error of metabolism (metabolic crisis, mitochondrial disease)",
    11: "Other medically important (IIH/pseudotumor cerebri, other urgent)",
}

BENIGN_CATEGORIES = {
    1: "Headache NOS (tension-type, unspecified, non-migraine headache)",
    2: "Migraine (with/without aura, hemiplegic, basilar, abdominal, vestibular)",
    3: "Epilepsy/seizure disorder (known epilepsy, breakthrough seizure)",
    4: "Febrile or afebrile seizure (first seizure, febrile seizure, provoked)",
    5: "Other benign (conversion/functional, syncope, anxiety, BPV, Bell palsy, tic)",
}

CONSULT_CATEGORIES = {
    1: "Pediatrics (general pediatrics, paediatrics, pediatrician)",
    2: "Neurology (pediatric neurology, paediatric neurology, neurologist)",
    3: "Ophthalmology (eye, neuro-ophthalmology)",
    4: "Neurosurgery (neurosurgeon)",
    5: "Cardiology (cardiologist)",
    6: "PICU (pediatric intensive care, ICU, critical care)",
    7: "Psychiatry (psychology, mental health, social work)",
    8: "Other (any specialty not listed above)",
}

REFERRAL_CATEGORIES = {
    1: "Family physician (GP, primary care, family doctor)",
    2: "Neurology (pediatric neurology, neurologist)",
    3: "Pediatric inpatient (admission, admitted, pediatric ward)",
    4: "Cardiology (cardiologist)",
    5: "Other (any referral not listed above)",
}

OTHER_CLASSIFY_CATEGORIES = {
    1: "Pain (non-headache pain, chest pain, limb pain, abdominal pain)",
    2: "Movement disorder (tremor, dystonia, chorea, tics, myoclonus)",
    3: "Psychiatric (anxiety, panic, behavioral, self-harm, psychogenic)",
    4: "Gastrointestinal (nausea, vomiting not as headache feature, abdominal)",
    5: "Cranial nerve (facial droop, hearing loss, vertigo, tinnitus, dysphagia)",
    6: "Inflammation (swelling, erythema, joint, autoimmune symptom)",
    7: "Respiratory (dyspnea, cough, apnea, hyperventilation)",
    8: "Infection (fever as primary, pharyngitis, URI, viral illness)",
    9: "Constitutional (fatigue, lethargy, malaise, weight loss, pallor)",
    10: "Urinary (incontinence, retention, enuresis)",
    11: "Dermatological (rash, petechiae, skin changes)",
    12: "Hemorrhage (epistaxis, non-CNS bleeding)",
    13: "Syncope (fainting, presyncope, orthostatic, vasovagal)",
    14: "Cardiac (palpitations, chest tightness, murmur)",
    15: "Gynecological",
    16: "Ophthalmological (eye pain, blurred vision as primary, papilledema)",
    17: "Sleep (insomnia, parasomnia, narcolepsy, sleep disturbance)",
    18: "Hematological (anemia, bruising, coagulopathy)",
}

COMORBIDITY_GROUPS = {
    "neurological": "Neurological (epilepsy, prior stroke, brain injury, hydrocephalus, CP)",
    "developmental": "Developmental (autism/ASD, ADHD, learning disability, GDD, ID)",
    "psychiatric": "Psychiatric (anxiety, depression, eating disorder, self-harm, OCD)",
    "cardiac": "Cardiac (congenital heart disease, arrhythmia, murmur)",
    "respiratory": "Respiratory (asthma, reactive airway disease)",
    "autoimmune": "Autoimmune/inflammatory (lupus, JIA, IBD, celiac, PANDAS)",
    "hematological": "Hematological (sickle cell, bleeding disorder, anemia)",
    "oncological": "Oncological (cancer, prior tumor, leukemia)",
    "metabolic": "Metabolic/endocrine (diabetes, thyroid, metabolic disorder, obesity)",
    "genetic": "Genetic/syndromic (chromosomal, genetic syndrome, NF, TSC)",
    "migraine_hx": "Migraine/headache history (known migraine, chronic headache)",
    "other": "Other (surgical history, allergies, prematurity, miscellaneous)",
    "none": "No significant history / unable to determine",
}


# ===================================================================
# CLI
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="LLM-Assisted Free-Text Classification for Brain AttACKs"
    )
    p.add_argument("--task", nargs="+", choices=["a", "b", "c", "d", "e", "all"],
                    default=["all"], help="Tasks to run (default: all)")
    p.add_argument("--dry-run", action="store_true",
                    help="Survey data and estimate cost without API calls")
    p.add_argument("--merge-only", action="store_true",
                    help="Merge existing cached results without new API calls")
    p.add_argument("--force", action="store_true",
                    help="Re-run even if cached results exist")
    p.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"Claude model (default: {DEFAULT_MODEL})")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                    help=f"Entries per API call (default: {BATCH_SIZE})")
    return p.parse_args()


# ===================================================================
# LLM CLIENT
# ===================================================================

def create_client():
    try:
        import anthropic  # noqa: F811
    except ImportError:
        print("ERROR: anthropic package not installed. Run: pip install anthropic")
        sys.exit(1)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)
    return anthropic.Anthropic()


def call_llm(client, prompt, system_prompt=None, model=DEFAULT_MODEL,
             max_tokens=4096):
    import anthropic

    messages = [{"role": "user", "content": prompt}]
    kwargs = dict(model=model, max_tokens=max_tokens, messages=messages)
    if system_prompt:
        kwargs["system"] = system_prompt

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.messages.create(**kwargs)
            return resp.content[0].text
        except anthropic.RateLimitError:
            wait = RETRY_DELAY * (2 ** attempt)
            print(f"\n  Rate limited, waiting {wait:.0f}s...", end="", flush=True)
            time.sleep(wait)
        except anthropic.APIError as e:
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"\n  API error ({e}), retrying...", end="", flush=True)
            time.sleep(RETRY_DELAY)
    raise RuntimeError("Max retries exceeded")


def parse_json_response(text):
    """Extract JSON array from LLM response, handling markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        result = json.loads(text)
        return result if isinstance(result, list) else [result]
    except json.JSONDecodeError:
        pass
    for sc, ec in [("[", "]"), ("{", "}")]:
        s = text.find(sc)
        e = text.rfind(ec)
        if s != -1 and e > s:
            try:
                result = json.loads(text[s:e + 1])
                return result if isinstance(result, list) else [result]
            except json.JSONDecodeError:
                continue
    print(f"\n  WARNING: JSON parse failed: {text[:200]}")
    return []


# ===================================================================
# UTILITIES
# ===================================================================

def fmt_scheme(categories):
    """Format coding scheme as numbered list for prompts."""
    return "\n".join(f"  {k} = {v}" for k, v in categories.items())


def get_unique_texts(df, col):
    """Unique non-empty values with counts, case-normalized."""
    mask = df[col].notna() & (df[col].astype(str).str.strip() != "")
    vals = df.loc[mask, col].astype(str).str.strip().str.lower()
    vc = vals.value_counts().reset_index()
    vc.columns = ["text", "count"]
    return vc


def split_delimited(df, col, sep=";"):
    """Split multi-value field into individual terms, deduplicate."""
    mask = df[col].notna() & (df[col].astype(str).str.strip() != "")
    all_terms = []
    for v in df.loc[mask, col].astype(str):
        all_terms.extend(t.strip().lower() for t in v.split(sep) if t.strip())
    vc = pd.Series(all_terms).value_counts().reset_index()
    vc.columns = ["text", "count"]
    return vc


def batched(items, n):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def classify_batch(client, texts, system_prompt, prompt_template,
                   model=DEFAULT_MODEL, batch_size=BATCH_SIZE):
    """Send texts to LLM in batches; return list of result dicts."""
    results = []
    total = (len(texts) + batch_size - 1) // batch_size
    for bnum, batch in enumerate(batched(texts, batch_size), 1):
        print(f"  Batch {bnum}/{total} ({len(batch)})...", end="", flush=True)
        entries = "\n".join(f'{i+1}. "{t}"' for i, t in enumerate(batch))
        prompt = prompt_template.format(entries=entries)
        raw = call_llm(client, prompt, system_prompt=system_prompt, model=model)
        parsed = parse_json_response(raw)
        # Match responses back to input texts
        for idx, item in enumerate(parsed):
            if not isinstance(item, dict):
                continue
            resp_text = str(item.get("text", "")).strip().lower()
            matched = None
            for bt in batch:
                if bt.strip().lower() == resp_text:
                    matched = bt
                    break
            if not matched and idx < len(batch):
                matched = batch[idx]
            item["_input"] = matched or resp_text
            results.append(item)
        print(f" -> {len(parsed)}")
        time.sleep(0.3)
    return results


# ===================================================================
# TASK A: OUTCOME VALIDATION
# ===================================================================

_A_SYS = (
    "You are a clinical data coding assistant specializing in pediatric emergency "
    "neurology. Classify ED diagnoses for children presenting with acute brain "
    "dysfunction symptoms. Return ONLY a JSON array, no other text."
)

_A_PROMPT = (
    "Classify each pediatric ED diagnosis as medically important (MI) or "
    "medically benign (B).\n\nMEDICALLY IMPORTANT:\n"
    + fmt_scheme(MI_CATEGORIES)
    + "\n\nMEDICALLY BENIGN:\n"
    + fmt_scheme(BENIGN_CATEGORIES)
    + """

Rules:
- VP shunt malfunction/obstruction -> MI 8
- Intracranial hypertension / pseudotumor cerebri / IIH -> MI 11
- Transverse myelitis / optic neuritis / ADEM / NMO -> MI 6
- Brain abscess / empyema -> MI 7
- Moyamoya -> MI 3
- "seizure" alone -> B 4
- "headache" alone -> B 1
- "migraine" (any type) -> B 2
- Syncope / vasovagal -> B 5
- Conversion / functional / psychogenic -> B 5
- Bell's palsy -> B 5
- If both benign and important elements present, classify as MI
- If truly ambiguous, give best guess with confidence "low"

Classify:
{entries}

JSON array format:
[{{"text": "...", "class": "mi" or "benign", "code": <int>, "confidence": "high" or "moderate" or "low", "notes": ""}}]"""
)


def task_a(df, client, model, bs, force=False):
    cache = LLM_DIR / "task_a_diagnosis.csv"
    print("\n" + "=" * 60)
    print("TASK A: Outcome Validation (diagnosis_specify)")
    print("=" * 60)
    if cache.exists() and not force:
        print(f"  Using cache: {cache.name}")
        return pd.read_csv(cache)
    unique = get_unique_texts(df, "diagnosis_specify")
    print(f"  Entries: {unique['count'].sum()}, Unique: {len(unique)}")
    results = classify_batch(client, unique["text"].tolist(), _A_SYS, _A_PROMPT,
                             model=model, batch_size=bs)
    out = pd.DataFrame(results).rename(columns={
        "_input": "text_normalized", "class": "llm_class", "code": "llm_code"
    })
    if not out.empty:
        out = out.merge(unique, left_on="text_normalized", right_on="text",
                        how="left", suffixes=("", "_dup"))
        out = out.drop(columns=[c for c in out.columns if c.endswith("_dup")],
                       errors="ignore")
    out.to_csv(cache, index=False)
    print(f"  Saved {len(out)} -> {cache.name}")
    return out


# ===================================================================
# TASK B: YEAR 1 HARMONIZATION
# ===================================================================

_B_SYS = (
    "You are a clinical data coding assistant. Parse free-text consultation and "
    "referral notes from a pediatric ED into structured specialty categories. "
    "Return ONLY a JSON array."
)

_B_CONSULT = (
    "Parse each consultation note into specialty categories.\n\n"
    "CONSULTATION CATEGORIES:\n"
    + fmt_scheme(CONSULT_CATEGORIES)
    + """

Rules:
- One entry may list MULTIPLE specialties (e.g., "Neuro and Ophthalmology" -> [2, 3])
- "Paeds" / "Peds" = Pediatrics (1)
- "Neuro" = Neurology (2)
- "NSx" / "Neurosurg" = Neurosurgery (4)
- If unclear, use Other (8)

Parse:
{entries}

JSON: [{{"text": "...", "codes": [<int>], "confidence": "high"/"moderate"/"low", "notes": ""}}]"""
)

_B_REFERRAL = (
    "Parse each discharge referral note into referral categories.\n\n"
    "REFERRAL CATEGORIES:\n"
    + fmt_scheme(REFERRAL_CATEGORIES)
    + """

Rules:
- One entry may list MULTIPLE referrals
- "GP" / "FP" / "family doc" = Family physician (1)
- "Neuro" = Neurology (2)
- "Admitted" / "ward" = Pediatric inpatient (3)
- If unclear, use Other (5)

Parse:
{entries}

JSON: [{{"text": "...", "codes": [<int>], "confidence": "high"/"moderate"/"low", "notes": ""}}]"""
)


def task_b(df, client, model, bs, force=False):
    cache_c = LLM_DIR / "task_b_consult.csv"
    cache_r = LLM_DIR / "task_b_referral.csv"
    print("\n" + "=" * 60)
    print("TASK B: Year 1 Harmonization (consult + referral)")
    print("=" * 60)

    y1 = df[df["year"] == "1"]

    # Consultations
    if cache_c.exists() and not force:
        print(f"  Consult: using cache")
        con_df = pd.read_csv(cache_c)
    else:
        uq = get_unique_texts(y1, "consult_specify")
        print(f"  Consult entries: {uq['count'].sum()}, Unique: {len(uq)}")
        if len(uq):
            res = classify_batch(client, uq["text"].tolist(), _B_SYS, _B_CONSULT,
                                 model=model, batch_size=bs)
            con_df = pd.DataFrame(res).rename(columns={"_input": "text_normalized"})
            con_df.to_csv(cache_c, index=False)
            print(f"  Saved {len(con_df)} -> {cache_c.name}")
        else:
            con_df = pd.DataFrame()

    # Referrals
    if cache_r.exists() and not force:
        print(f"  Referral: using cache")
        ref_df = pd.read_csv(cache_r)
    else:
        uq = get_unique_texts(y1, "discharge_refer_specify")
        print(f"  Referral entries: {uq['count'].sum()}, Unique: {len(uq)}")
        if len(uq):
            res = classify_batch(client, uq["text"].tolist(), _B_SYS, _B_REFERRAL,
                                 model=model, batch_size=bs)
            ref_df = pd.DataFrame(res).rename(columns={"_input": "text_normalized"})
            ref_df.to_csv(cache_r, index=False)
            print(f"  Saved {len(ref_df)} -> {cache_r.name}")
        else:
            ref_df = pd.DataFrame()

    return con_df, ref_df


# ===================================================================
# TASK C: SYMPTOM ENRICHMENT
# ===================================================================

_C_SYS = (
    "You are a clinical data coding assistant. Classify free-text symptoms from "
    "a pediatric ED into standardized categories. Return ONLY a JSON array."
)

_C_PROMPT = (
    "Classify each symptom into one or more categories.\n\n"
    "SYMPTOM CATEGORIES:\n"
    + fmt_scheme(OTHER_CLASSIFY_CATEGORIES)
    + """

Rules:
- A symptom may fit multiple categories; list all that apply
- "Dizziness" -> 5 (cranial nerve)
- "Vertigo" -> 5 (cranial nerve)
- "Neck stiffness" -> 1 (pain)
- "Vomiting" alone -> 4 (GI)
- "Lethargy" / "drowsy" -> 9 (constitutional)
- "Numbness" / "tingling" -> use closest match with confidence "low"

Classify:
{entries}

JSON: [{{"text": "...", "codes": [<int>], "confidence": "high"/"moderate"/"low", "notes": ""}}]"""
)


def task_c(df, client, model, bs, force=False):
    cache = LLM_DIR / "task_c_symptom.csv"
    print("\n" + "=" * 60)
    print("TASK C: Symptom Enrichment (symptom_other)")
    print("=" * 60)
    if cache.exists() and not force:
        print(f"  Using cache: {cache.name}")
        return pd.read_csv(cache)
    uq = split_delimited(df, "symptom_other")
    print(f"  Unique terms (after splitting): {len(uq)}")
    results = classify_batch(client, uq["text"].tolist(), _C_SYS, _C_PROMPT,
                             model=model, batch_size=bs)
    out = pd.DataFrame(results).rename(columns={"_input": "text_normalized"})
    out.to_csv(cache, index=False)
    print(f"  Saved {len(out)} -> {cache.name}")
    return out


# ===================================================================
# TASK D: COMORBIDITY CODING
# ===================================================================

_D_SYS = (
    "You are a clinical data coding assistant. Classify prior medical history "
    "entries from a pediatric ED into clinical comorbidity groups. "
    "Return ONLY a JSON array."
)

_D_PROMPT = (
    "Classify each prior diagnosis into comorbidity groups.\n\n"
    "COMORBIDITY GROUPS:\n"
    + "\n".join(f'  "{k}" = {v}' for k, v in COMORBIDITY_GROUPS.items())
    + """

Rules:
- An entry may fit multiple groups (e.g., "epilepsy and autism" -> ["neurological", "developmental"])
- "Migraine" / "headaches" -> "migraine_hx"
- "ADHD" -> "developmental"
- "Anxiety" -> "psychiatric"
- "Asthma" -> "respiratory"
- If uninterpretable, use "none"

Classify:
{entries}

JSON: [{{"text": "...", "groups": ["group1"], "confidence": "high"/"moderate"/"low", "notes": ""}}]"""
)


def task_d(df, client, model, bs, force=False):
    cache = LLM_DIR / "task_d_comorbidity.csv"
    print("\n" + "=" * 60)
    print("TASK D: Comorbidity Coding (prev_diagnoses_other)")
    print("=" * 60)
    if cache.exists() and not force:
        print(f"  Using cache: {cache.name}")
        return pd.read_csv(cache)
    uq = split_delimited(df, "prev_diagnoses_other")
    print(f"  Unique terms (after splitting): {len(uq)}")
    results = classify_batch(client, uq["text"].tolist(), _D_SYS, _D_PROMPT,
                             model=model, batch_size=bs)
    out = pd.DataFrame(results).rename(columns={"_input": "text_normalized"})
    out.to_csv(cache, index=False)
    print(f"  Saved {len(out)} -> {cache.name}")
    return out


# ===================================================================
# TASK E: BENIGN DIAGNOSIS REVIEW
# ===================================================================

_E_SYS = (
    "You are a pediatric neurology expert reviewing diagnoses classified as "
    "'medically benign' in a brain attack study. Flag any that may actually be "
    "medically important. Return ONLY a JSON array."
)

_E_PROMPT = (
    "Review these diagnoses classified as MEDICALLY BENIGN. Flag any that may "
    "actually be MEDICALLY IMPORTANT (requiring urgent treatment).\n\n"
    "Medically important categories:\n"
    + fmt_scheme(MI_CATEGORIES)
    + """

For each, determine:
- "confirmed_benign" = truly benign, correct classification
- "potentially_important" = may be medically important, needs PI review
- "uncertain" = cannot determine without more context

Review:
{entries}

JSON: [{{"text": "...", "review": "confirmed_benign"/"potentially_important"/"uncertain", "suggested_mi_code": null or <int>, "confidence": "high"/"moderate"/"low", "notes": ""}}]"""
)


def task_e(df, client, model, bs, force=False):
    cache = LLM_DIR / "task_e_benign_review.csv"
    print("\n" + "=" * 60)
    print("TASK E: Benign Diagnosis Review (med_benign_specify)")
    print("=" * 60)
    if cache.exists() and not force:
        print(f"  Using cache: {cache.name}")
        return pd.read_csv(cache)
    uq = get_unique_texts(df, "med_benign_specify")
    print(f"  Entries: {uq['count'].sum()}, Unique: {len(uq)}")
    results = classify_batch(client, uq["text"].tolist(), _E_SYS, _E_PROMPT,
                             model=model, batch_size=bs)
    out = pd.DataFrame(results).rename(columns={"_input": "text_normalized"})
    out.to_csv(cache, index=False)
    print(f"  Saved {len(out)} -> {cache.name}")
    return out


# ===================================================================
# HUMAN REVIEW FILE
# ===================================================================

def generate_human_review(task_results):
    """Collect uncertain/flagged cases into a single PI review file."""
    rows = []

    for task_key, rdf in task_results.items():
        if rdf is None or not isinstance(rdf, pd.DataFrame) or rdf.empty:
            continue
        # Non-high confidence
        if "confidence" in rdf.columns:
            for _, r in rdf[rdf["confidence"].isin(["moderate", "low"])].iterrows():
                rows.append({
                    "task": task_key,
                    "text": r.get("text_normalized", r.get("text", "")),
                    "confidence": r.get("confidence", ""),
                    "llm_result": json.dumps({k: v for k, v in r.items()
                                              if k not in ("text_normalized", "text")}),
                    "reason": "Non-high confidence",
                    "pi_decision": "",
                    "pi_override_code": "",
                })
        # Task E flagged
        if task_key == "E" and "review" in rdf.columns:
            flagged = rdf[rdf["review"].isin(["potentially_important", "uncertain"])]
            for _, r in flagged.iterrows():
                rows.append({
                    "task": task_key,
                    "text": r.get("text_normalized", r.get("text", "")),
                    "confidence": r.get("confidence", ""),
                    "llm_result": json.dumps({k: v for k, v in r.items()
                                              if k not in ("text_normalized", "text")}),
                    "reason": f"Benign flagged: {r.get('review', '')}",
                    "pi_decision": "",
                    "pi_override_code": "",
                })

    rdf = pd.DataFrame(rows).drop_duplicates(subset=["task", "text"])
    path = LLM_DIR / "human_review.csv"
    rdf.to_csv(path, index=False)
    print(f"\n  Human review: {len(rdf)} cases -> {path.name}")
    return rdf


# ===================================================================
# MERGE
# ===================================================================

def _safe_codes(raw):
    """Parse a codes field that may be a JSON string, list, or scalar."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, (int, float)):
        return [int(raw)] if pd.notna(raw) else []
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        try:
            val = json.loads(s.replace("'", '"'))
            return val if isinstance(val, list) else [val]
        except (json.JSONDecodeError, TypeError):
            return []
    return []


def _safe_groups(raw):
    """Parse a groups field (list of strings)."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        try:
            val = json.loads(s.replace("'", '"'))
            return val if isinstance(val, list) else [val]
        except (json.JSONDecodeError, TypeError):
            return []
    return []


def merge_results(df, task_results):
    """Merge LLM classifications back into the combined dataset."""
    print("\n" + "=" * 60)
    print("MERGE: Applying LLM-coded variables")
    print("=" * 60)

    changes = {}

    # ------------------------------------------------------------------
    # Task A: diagnosis classification + discrepancy flags
    # ------------------------------------------------------------------
    a_df = task_results.get("A")
    if a_df is not None and not a_df.empty:
        lookup = {}
        for _, r in a_df.iterrows():
            t = str(r.get("text_normalized", r.get("text", ""))).strip().lower()
            if t:
                lookup[t] = {
                    "llm_dx_class": r.get("llm_class", ""),
                    "llm_dx_code": r.get("llm_code", ""),
                    "llm_dx_confidence": r.get("confidence", ""),
                }
        dx_norm = df["diagnosis_specify"].fillna("").astype(str).str.strip().str.lower()
        df["llm_dx_class"] = dx_norm.map(lambda t: lookup.get(t, {}).get("llm_dx_class", ""))
        df["llm_dx_code"] = dx_norm.map(lambda t: lookup.get(t, {}).get("llm_dx_code", ""))
        df["llm_dx_confidence"] = dx_norm.map(
            lambda t: lookup.get(t, {}).get("llm_dx_confidence", ""))
        # Discrepancies
        llm_mi = df["llm_dx_class"] == "mi"
        chk_mi = df["outcome_med_important"] == 1
        df["dx_discrepancy"] = ""
        df.loc[llm_mi & ~chk_mi, "dx_discrepancy"] = "llm_mi_checkbox_benign"
        df.loc[~llm_mi & chk_mi & (dx_norm != ""), "dx_discrepancy"] = "llm_benign_checkbox_mi"
        n_disc = (df["dx_discrepancy"] != "").sum()
        changes["A"] = f"dx classification cols + {n_disc} discrepancies"
        print(f"  Task A: {n_disc} outcome discrepancies flagged")

    # ------------------------------------------------------------------
    # Task B: populate Y1 consult/referral checkboxes
    # ------------------------------------------------------------------
    for label, key, col_text, col_prefix, n_opts in [
        ("consult", "B_consult", "consult_specify", "consult_specify", 8),
        ("referral", "B_referral", "discharge_refer_specify",
         "discharge_refer_specify", 5),
    ]:
        b_df = task_results.get(key)
        if b_df is None or b_df.empty:
            continue
        lookup = {}
        for _, r in b_df.iterrows():
            t = str(r.get("text_normalized", r.get("text", ""))).strip().lower()
            codes = _safe_codes(r.get("codes", "[]"))
            if t:
                lookup[t] = codes
        y1_mask = df["year"] == "1"
        norm = df[col_text].fillna("").astype(str).str.strip().str.lower()
        n_coded = 0
        for t, codes in lookup.items():
            mask = y1_mask & (norm == t)
            n_match = mask.sum()
            if n_match == 0:
                continue
            for c in codes:
                ccol = f"{col_prefix}___{c}"
                if ccol in df.columns:
                    df[ccol] = df[ccol].astype(str)
                    df.loc[mask, ccol] = "1"
                    n_coded += n_match
        changes[f"B_{label}"] = f"{n_coded} Y1 {label} checkbox values coded"
        print(f"  Task B ({label}): {n_coded} Y1 checkbox values populated")

    # ------------------------------------------------------------------
    # Task C: add LLM symptom category column
    # ------------------------------------------------------------------
    c_df = task_results.get("C")
    if c_df is not None and not c_df.empty:
        term_lookup = {}
        for _, r in c_df.iterrows():
            t = str(r.get("text_normalized", r.get("text", ""))).strip().lower()
            codes = _safe_codes(r.get("codes", "[]"))
            if t:
                term_lookup[t] = codes

        def _map_symptoms(text):
            if not text or str(text).strip() == "":
                return ""
            terms = [t.strip().lower() for t in str(text).split(";") if t.strip()]
            all_codes = set()
            for term in terms:
                all_codes.update(term_lookup.get(term, []))
            return json.dumps(sorted(all_codes)) if all_codes else ""

        df["llm_symptom_categories"] = (
            df["symptom_other"].fillna("").apply(_map_symptoms)
        )
        n_coded = (df["llm_symptom_categories"] != "").sum()
        changes["C"] = f"llm_symptom_categories for {n_coded} records"
        print(f"  Task C: {n_coded} records with coded symptom categories")

    # ------------------------------------------------------------------
    # Task D: binary comorbidity columns
    # ------------------------------------------------------------------
    d_df = task_results.get("D")
    if d_df is not None and not d_df.empty:
        term_lookup = {}
        for _, r in d_df.iterrows():
            t = str(r.get("text_normalized", r.get("text", ""))).strip().lower()
            groups = _safe_groups(r.get("groups", "[]"))
            if t:
                term_lookup[t] = groups
        # Create columns
        group_keys = [k for k in COMORBIDITY_GROUPS if k != "none"]
        for gk in group_keys:
            df[f"comorbid_{gk}"] = "0"
        mask = (df["prev_diagnoses_other"].notna()
                & (df["prev_diagnoses_other"].astype(str).str.strip() != ""))
        for idx in df[mask].index:
            text = str(df.at[idx, "prev_diagnoses_other"]).strip()
            terms = [t.strip().lower() for t in text.split(";") if t.strip()]
            for term in terms:
                for g in term_lookup.get(term, []):
                    col = f"comorbid_{g}"
                    if col in df.columns:
                        df.at[idx, col] = "1"
        n_any = (df[[f"comorbid_{g}" for g in group_keys]].astype(int).max(axis=1) > 0).sum()
        changes["D"] = f"{len(group_keys)} comorbid_* cols; {n_any} records coded"
        print(f"  Task D: {n_any} records with coded comorbidities")

    # ------------------------------------------------------------------
    # Task E: benign review flags
    # ------------------------------------------------------------------
    e_df = task_results.get("E")
    if e_df is not None and not e_df.empty:
        lookup = {}
        for _, r in e_df.iterrows():
            t = str(r.get("text_normalized", r.get("text", ""))).strip().lower()
            if t:
                lookup[t] = {
                    "review": r.get("review", ""),
                    "mi_code": r.get("suggested_mi_code", ""),
                }
        norm = df["med_benign_specify"].fillna("").astype(str).str.strip().str.lower()
        df["benign_review_flag"] = norm.map(
            lambda t: lookup.get(t, {}).get("review", ""))
        df["benign_suggested_mi"] = norm.map(
            lambda t: lookup.get(t, {}).get("mi_code", ""))
        n_flag = df["benign_review_flag"].isin(
            ["potentially_important", "uncertain"]).sum()
        changes["E"] = f"{n_flag} benign diagnoses flagged"
        print(f"  Task E: {n_flag} benign diagnoses flagged for review")

    # Summary
    print("\n  Merge summary:")
    for k, v in changes.items():
        print(f"    {k}: {v}")

    return df


# ===================================================================
# DRY RUN
# ===================================================================

def dry_run(df):
    print("\n" + "=" * 60)
    print("DRY RUN: Data Survey and Cost Estimate")
    print("=" * 60)

    y1 = df[df["year"] == "1"]
    tasks = {
        "A": ("diagnosis_specify", get_unique_texts(df, "diagnosis_specify")),
        "B-consult": ("consult_specify (Y1)", get_unique_texts(y1, "consult_specify")),
        "B-referral": ("discharge_refer_specify (Y1)",
                       get_unique_texts(y1, "discharge_refer_specify")),
        "C": ("symptom_other", split_delimited(df, "symptom_other")),
        "D": ("prev_diagnoses_other", split_delimited(df, "prev_diagnoses_other")),
        "E": ("med_benign_specify", get_unique_texts(df, "med_benign_specify")),
    }

    total_u = total_b = 0
    print(f"\n  {'Task':<14} {'Field':<32} {'Entries':>8} {'Unique':>8} {'Batches':>8}")
    print("  " + "-" * 74)
    for name, (field, uq) in tasks.items():
        n_entries = uq["count"].sum()
        n_unique = len(uq)
        n_batches = (n_unique + BATCH_SIZE - 1) // BATCH_SIZE
        total_u += n_unique
        total_b += n_batches
        print(f"  {name:<14} {field:<32} {n_entries:>8} {n_unique:>8} {n_batches:>8}")
    print(f"  {'TOTAL':<14} {'':<32} {'':>8} {total_u:>8} {total_b:>8}")

    # Haiku pricing: $0.80/MTok input, $4/MTok output
    est_in = total_b * 800
    est_out = total_u * 50
    est_cost = (est_in * 0.80 + est_out * 4.0) / 1_000_000
    print(f"\n  Estimated API calls: {total_b}")
    print(f"  Estimated tokens:    ~{est_in + est_out:,}")
    print(f"  Estimated cost:      ~${est_cost:.2f} (Haiku)")
    print(f"\n  Run: python 01b-llm-coding.py")


# ===================================================================
# MAIN
# ===================================================================

def main():
    args = parse_args()

    print("=" * 60)
    print("Brain AttACKs: LLM-Assisted Free-Text Classification")
    print("=" * 60)

    if not COMBINED.exists():
        print(f"ERROR: {COMBINED} not found. Run 01-data-prep.py first.")
        sys.exit(1)

    df = pd.read_csv(COMBINED, dtype=str, keep_default_na=False)
    df["outcome_med_important"] = (
        pd.to_numeric(df["outcome_med_important"], errors="coerce")
        .fillna(0).astype(int)
    )
    print(f"Loaded: {len(df)} records, {len(df.columns)} columns")

    run = set(args.task)
    if "all" in run:
        run = {"a", "b", "c", "d", "e"}

    if args.dry_run:
        dry_run(df)
        return

    # Create API client unless merge-only
    client = None if args.merge_only else create_client()

    # --- Run or load each task ---
    tr = {}

    if "a" in run and not args.merge_only:
        tr["A"] = task_a(df, client, args.model, args.batch_size, args.force)
    elif (LLM_DIR / "task_a_diagnosis.csv").exists():
        tr["A"] = pd.read_csv(LLM_DIR / "task_a_diagnosis.csv")

    if "b" in run and not args.merge_only:
        con, ref = task_b(df, client, args.model, args.batch_size, args.force)
        tr["B_consult"], tr["B_referral"] = con, ref
    else:
        p = LLM_DIR / "task_b_consult.csv"
        if p.exists():
            tr["B_consult"] = pd.read_csv(p)
        p = LLM_DIR / "task_b_referral.csv"
        if p.exists():
            tr["B_referral"] = pd.read_csv(p)

    if "c" in run and not args.merge_only:
        tr["C"] = task_c(df, client, args.model, args.batch_size, args.force)
    elif (LLM_DIR / "task_c_symptom.csv").exists():
        tr["C"] = pd.read_csv(LLM_DIR / "task_c_symptom.csv")

    if "d" in run and not args.merge_only:
        tr["D"] = task_d(df, client, args.model, args.batch_size, args.force)
    elif (LLM_DIR / "task_d_comorbidity.csv").exists():
        tr["D"] = pd.read_csv(LLM_DIR / "task_d_comorbidity.csv")

    if "e" in run and not args.merge_only:
        tr["E"] = task_e(df, client, args.model, args.batch_size, args.force)
    elif (LLM_DIR / "task_e_benign_review.csv").exists():
        tr["E"] = pd.read_csv(LLM_DIR / "task_e_benign_review.csv")

    # Human review
    if tr:
        generate_human_review(tr)

    # Merge
    df_out = merge_results(df, tr)

    # Export
    out_csv = PROCESSED / "brain_attacks_llm_coded.csv"
    out_pq = PROCESSED / "brain_attacks_llm_coded.parquet"
    df_out.to_csv(out_csv, index=False)
    # For parquet: ensure consistent column types (mixed int/str columns -> str)
    df_pq = df_out.copy()
    for col in df_pq.columns:
        if df_pq[col].dtype == object:
            df_pq[col] = df_pq[col].astype(str)
    df_pq.to_parquet(out_pq, index=False)
    print(f"\n  -> {out_csv.name} ({len(df_out)} rows, {len(df_out.columns)} cols)")
    print(f"  -> {out_pq.name}")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


def compute_agreement_metrics():
    """
    Compute Cohen's kappa and percent agreement between LLM classification
    and PI adjudication for the 124 flagged cases (TRIPOD+AI reporting).

    Reads pi_review_focused.csv (LLM suggestions) and pi-review-decisions.csv
    (PI final decisions). For each case:
      - LLM recommendation derived from task + issue description
      - PI decision from the pi_decision / decision column

    Outputs:
        outputs/table_llm_pi_agreement.csv
    """
    OUTPUTS = PROJECT / "outputs"
    OUTPUTS.mkdir(exist_ok=True)

    focused_path = LLM_DIR / "pi_review_focused.csv"
    decisions_path = LLM_DIR / "pi-review-decisions.csv"

    if not focused_path.exists() or not decisions_path.exists():
        print("WARNING: PI review files not found — skipping agreement metrics")
        return

    print("\n" + "=" * 60)
    print("LLM-PI Agreement Metrics")
    print("=" * 60)

    focused = pd.read_csv(focused_path)
    decisions = pd.read_csv(decisions_path)

    # --- Map each focused case to LLM suggestion and PI decision ---
    records = []
    for _, row in focused.iterrows():
        task = row["task"]
        issue = str(row.get("issue", ""))
        pi_dec = str(row.get("pi_decision", ""))
        text = str(row.get("text", ""))
        n_records = int(row.get("n_records", 1))

        # Determine LLM suggestion
        if "LLM says MI" in issue:
            llm_suggestion = "mi"
        elif "Checkbox says MI" in issue and "appears benign" in issue:
            llm_suggestion = "benign"
        elif task == "E":
            # Task E: benign review — LLM flagged as potentially important
            llm_suggestion = "mi"
        else:
            llm_suggestion = "uncertain"

        # Determine PI decision
        if pi_dec.startswith("reclassify_mi"):
            pi_final = "mi"
        elif pi_dec == "keep_benign":
            pi_final = "benign"
        else:
            pi_final = "uncertain"

        records.append({
            "task": task,
            "text": text,
            "n_records": n_records,
            "llm_suggestion": llm_suggestion,
            "pi_decision": pi_final,
            "agree": llm_suggestion == pi_final,
        })

    agree_df = pd.DataFrame(records)

    # --- Overall agreement (case-level, not record-weighted) ---
    n_total = len(agree_df)
    n_agree = agree_df["agree"].sum()
    pct_agree = n_agree / n_total * 100

    # --- Cohen's kappa (2x2: mi vs benign) ---
    valid = agree_df[
        (agree_df["llm_suggestion"].isin(["mi", "benign"]))
        & (agree_df["pi_decision"].isin(["mi", "benign"]))
    ]
    n_valid = len(valid)
    a = ((valid["llm_suggestion"] == "mi") & (valid["pi_decision"] == "mi")).sum()
    b = ((valid["llm_suggestion"] == "mi") & (valid["pi_decision"] == "benign")).sum()
    c = ((valid["llm_suggestion"] == "benign") & (valid["pi_decision"] == "mi")).sum()
    d = ((valid["llm_suggestion"] == "benign") & (valid["pi_decision"] == "benign")).sum()

    po = (a + d) / n_valid if n_valid else 0
    p_llm_mi = (a + b) / n_valid if n_valid else 0
    p_pi_mi = (a + c) / n_valid if n_valid else 0
    pe = p_llm_mi * p_pi_mi + (1 - p_llm_mi) * (1 - p_pi_mi)
    kappa = (po - pe) / (1 - pe) if (1 - pe) != 0 else 0

    # --- Record-weighted agreement (accounting for n_records) ---
    total_records = agree_df["n_records"].sum()
    agree_records = agree_df.loc[agree_df["agree"], "n_records"].sum()
    pct_agree_weighted = agree_records / total_records * 100 if total_records else 0

    # --- By-task breakdown ---
    task_stats = []
    for task_name in sorted(agree_df["task"].unique()):
        t = agree_df[agree_df["task"] == task_name]
        t_valid = t[
            (t["llm_suggestion"].isin(["mi", "benign"]))
            & (t["pi_decision"].isin(["mi", "benign"]))
        ]
        task_stats.append({
            "task": task_name,
            "n_cases": len(t),
            "n_agree": t["agree"].sum(),
            "pct_agree": t["agree"].mean() * 100,
            "n_llm_mi": (t["llm_suggestion"] == "mi").sum(),
            "n_pi_mi": (t["pi_decision"] == "mi").sum(),
        })

    # --- Output ---
    summary = {
        "metric": [
            "Total cases reviewed",
            "Cases with LLM-PI agreement",
            "Percent agreement (case-level)",
            "Percent agreement (record-weighted)",
            "Cohen's kappa",
            "Confusion: LLM=MI, PI=MI (true positive)",
            "Confusion: LLM=MI, PI=Benign (false positive)",
            "Confusion: LLM=Benign, PI=MI (false negative)",
            "Confusion: LLM=Benign, PI=Benign (true negative)",
        ],
        "value": [
            n_total, n_agree, f"{pct_agree:.1f}%", f"{pct_agree_weighted:.1f}%",
            f"{kappa:.3f}",
            a, b, c, d,
        ],
    }
    summary_df = pd.DataFrame(summary)

    # Combine into single output
    out_path = OUTPUTS / "table_llm_pi_agreement.csv"
    with open(out_path, "w") as f:
        f.write("# LLM-PI Agreement Metrics\n")
        f.write(f"# Model: {DEFAULT_MODEL}\n")
        f.write(f"# Cases reviewed by PI: {n_total}\n\n")
        summary_df.to_csv(f, index=False)
        f.write("\n# By-task breakdown\n")
        pd.DataFrame(task_stats).to_csv(f, index=False)
        f.write("\n# Per-case detail\n")
        agree_df.to_csv(f, index=False)

    print(f"\n  Cases: {n_total}")
    print(f"  Agreement: {n_agree}/{n_total} ({pct_agree:.1f}%)")
    print(f"  Agreement (record-weighted): {agree_records}/{total_records} ({pct_agree_weighted:.1f}%)")
    print(f"  Cohen's kappa: {kappa:.3f}")
    print(f"  Confusion matrix: TP={a}, FP={b}, FN={c}, TN={d}")
    for ts in task_stats:
        print(f"  Task {ts['task']}: {ts['n_agree']}/{ts['n_cases']} ({ts['pct_agree']:.1f}%)")
    print(f"\n  -> {out_path.name}")


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == "--agreement":
        compute_agreement_metrics()
    else:
        main()
