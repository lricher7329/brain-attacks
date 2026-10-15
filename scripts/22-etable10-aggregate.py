#!/usr/bin/env python3
"""
eTable 10 — CINP adjudication summary, aggregated by diagnosis category.

Reads the record-level adjudication table written by 21-cinp-framework.py
(outputs/tables/table_cinp_adjudication.csv — git-ignored, never published) and
writes aggregate-only outputs that are safe to share:

    outputs/tables/etable10_cinp_adjudication_by_category.csv
    outputs/tables/etable10_cinp_objective_findings.csv
    outputs/tables/etable10_cinp_adjudication.md

Diagnosis categories match manuscript Table 2: a case with more than one CINP
diagnosis is counted once in each of its categories, so category rows sum to
more than the number of cases.
"""

from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
TABLES = PROJECT / "outputs" / "tables"

LABELS = {"ICH": "Intracranial haemorrhage", "Other MI": "Other"}


def pct(n, d):
    return f"{n} ({100 * n / d:.0f}%)" if d else "0"


def main():
    df = pd.read_csv(TABLES / "table_cinp_adjudication.csv")
    n_cases = len(df)
    df["has_finding"] = df["n_objective_findings"] > 0
    df["criterion"] = df["cinp_criterion"].str.extract(r"(\d)").astype(int)

    long = df.assign(category=df["mi_diagnoses"].str.split("; ")).explode("category")
    long["category"] = long["category"].replace(LABELS)

    rows = []
    for cat, g in long.groupby("category"):
        n = len(g)
        q1, med, q3 = g["n_objective_findings"].quantile([0.25, 0.5, 0.75])
        rows.append({
            "Diagnosis category": cat,
            "n": n,
            "Year 1": int((g["year"] == 1).sum()),
            "Year 2": int((g["year"] == 2).sum()),
            "Criterion 1": int((g["criterion"] == 1).sum()),
            "Criterion 2": int((g["criterion"] == 2).sum()),
            "CIIP": int(g["ciip_status"].sum()),
            "Abnormal imaging": pct(int(g["abnormal_imaging"].sum()), n),
            "Specific treatment": pct(int(g["specific_treatment"].sum()), n),
            "≥1 objective finding": pct(int(g["has_finding"].sum()), n),
            "Objective findings, median (IQR)": f"{med:.0f} ({q1:.0f}–{q3:.0f})",
        })
    by_cat = pd.DataFrame(rows).sort_values(["n", "Diagnosis category"], ascending=[False, True])

    q1, med, q3 = df["n_objective_findings"].quantile([0.25, 0.5, 0.75])
    total = {
        "Diagnosis category": "All CINP cases",
        "n": n_cases,
        "Year 1": int((df["year"] == 1).sum()),
        "Year 2": int((df["year"] == 2).sum()),
        "Criterion 1": int((df["criterion"] == 1).sum()),
        "Criterion 2": int((df["criterion"] == 2).sum()),
        "CIIP": int(df["ciip_status"].sum()),
        "Abnormal imaging": pct(int(df["abnormal_imaging"].sum()), n_cases),
        "Specific treatment": pct(int(df["specific_treatment"].sum()), n_cases),
        "≥1 objective finding": pct(int(df["has_finding"].sum()), n_cases),
        "Objective findings, median (IQR)": f"{med:.0f} ({q1:.0f}–{q3:.0f})",
    }
    by_cat = pd.concat([by_cat, pd.DataFrame([total])], ignore_index=True)
    by_cat.to_csv(TABLES / "etable10_cinp_adjudication_by_category.csv", index=False)

    findings = (df.loc[df["has_finding"], "objective_findings"].str.split("; ").explode().value_counts())
    findings = findings.rename_axis("Objective finding").reset_index(name="n")
    findings["% of CINP cases"] = (100 * findings["n"] / n_cases).round(0).astype(int)
    findings.to_csv(TABLES / "etable10_cinp_objective_findings.csv", index=False)

    multi = int((df["mi_diagnoses"].str.count("; ") > 0).sum())
    md = [
        "**eTable 10. CINP adjudication summary by diagnosis category**",
        "",
        "*Part A. Adjudication by diagnosis category*",
        "",
        "| " + " | ".join(by_cat.columns) + " |",
        "|" + "---|" * len(by_cat.columns),
    ]
    for _, r in by_cat.iterrows():
        cells = [str(v) for v in r.values]
        if r["Diagnosis category"] == "All CINP cases":
            cells = [f"**{c}**" for c in cells]
        md.append("| " + " | ".join(cells) + " |")
    md += [
        "",
        f"A case with more than one CINP diagnosis is counted in each of its categories ({multi} of {n_cases} cases had more than one), "
        "so category rows sum to more than the total. Criterion 1: confirmed intracranial pathology requiring treatment, imaging surveillance "
        "or specialist referral. Criterion 2: documented objective neurologic finding mandating investigation. CIIP, clinically important "
        "intracranial pathology; IQR, interquartile range.",
        "",
        "*Part B. Objective neurologic findings documented among CINP cases*",
        "",
        "| Objective finding | n | % of CINP cases |",
        "|---|---:|---:|",
    ]
    md += [f"| {r['Objective finding']} | {r['n']} | {r['% of CINP cases']} |" for _, r in findings.iterrows()]
    md += ["", f"Cases may have more than one finding. {int((~df['has_finding']).sum())} of {n_cases} cases had no objective finding documented."]
    (TABLES / "etable10_cinp_adjudication.md").write_text("\n".join(md) + "\n")
    print(f"eTable 10: {len(by_cat) - 1} categories, {len(findings)} findings, {n_cases} cases ({multi} multi-diagnosis)")


if __name__ == "__main__":
    main()
