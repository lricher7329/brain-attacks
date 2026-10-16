#!/usr/bin/env python3
"""
Minimum sample size for the CINP prediction models (Riley et al., BMJ 2020;368:m441).

Binary outcome, three criteria: (1) expected uniform shrinkage >= 0.90,
(2) optimism in Nagelkerke R^2 <= 0.05, (3) overall risk estimated within +/-0.05.
The anticipated Cox-Snell R^2 is approximated from the anticipated C-statistic by
simulation, following pmsampsize (Riley et al., Stat Med 2022): a large sample of
linear predictors from two normals separated by sqrt(2)*Phi^-1(C), recalibrated by
logistic regression.

Reproduces the values in the Pediatrics manuscript (Methods, Sample Size):
    55 parameters -> 2,734 participants (186 events); expected shrinkage 0.88 in the
    full cohort and 0.78 in the Year 1 derivation cohort.
    10 parameters -> 498 participants (34 events); expected shrinkage 0.97.

Usage:
    python3 scripts/23-riley-sample-size.py
"""

import math
from statistics import NormalDist

import numpy as np

PREVALENCE = 0.068   # CINP, 147 / 2,153
C_STATISTIC = 0.90   # anticipated
COHORTS = {"Full cohort": 2153, "Year 1 derivation cohort": 1043}
PARAMETERS = (10, 33, 55)
SEED = 20260215


def approximate_r2(auc, prev, n=1_000_000, seed=SEED):
    """Cox-Snell R^2 and its maximum for a given C-statistic and prevalence."""
    rng = np.random.default_rng(seed)
    mu = math.sqrt(2) * NormalDist().inv_cdf(auc)
    n1 = int(prev * n)
    n0 = n - n1
    lp = np.concatenate([rng.normal(0, 1, n0), rng.normal(mu, 1, n1)])
    y = np.concatenate([np.zeros(n0), np.ones(n1)])
    X = np.column_stack([np.ones(n), lp])
    b = np.zeros(2)
    for _ in range(50):  # Newton-Raphson logistic regression
        p = 1 / (1 + np.exp(-X @ b))
        step = np.linalg.solve((X * (p * (1 - p))[:, None]).T @ X, X.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-10:
            break
    p = np.clip(1 / (1 + np.exp(-X @ b)), 1e-15, 1 - 1e-15)
    ll_model = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    ll_null = n1 * math.log(prev) + n0 * math.log(1 - prev)
    r2_cs = 1 - math.exp(2 * (ll_null - ll_model) / n)
    max_r2 = 1 - math.exp(2 * ll_null / n)
    return r2_cs, max_r2


def minimum_sample_size(P, prev, r2_cs, max_r2, S=0.90, delta=0.05, margin=0.05):
    n1 = P / ((S - 1) * math.log(1 - r2_cs / S))
    S2 = r2_cs / (r2_cs + delta * max_r2)
    n2 = P / ((S2 - 1) * math.log(1 - r2_cs / S2))
    n3 = (1.96 / margin) ** 2 * prev * (1 - prev)
    n = math.ceil(max(n1, n2, n3))
    return {"criterion 1 (shrinkage)": math.ceil(n1), "criterion 2 (optimism)": math.ceil(n2),
            "criterion 3 (intercept)": math.ceil(n3), "minimum n": n,
            "events": math.ceil(n * prev), "events per parameter": round(n * prev / P, 2)}


def expected_shrinkage(P, n, r2_cs):
    """Solve S = 1 + P / (n * ln(1 - R2_cs / S)) by fixed-point iteration."""
    S = 0.9
    for _ in range(500):
        S = 1 + P / (n * math.log(1 - r2_cs / S))
    return S


def main():
    r2_cs, max_r2 = approximate_r2(C_STATISTIC, PREVALENCE)
    print(f"Prevalence {PREVALENCE}, C-statistic {C_STATISTIC}: "
          f"Cox-Snell R^2 {r2_cs:.3f} (max {max_r2:.3f})\n")
    for P in PARAMETERS:
        res = minimum_sample_size(P, PREVALENCE, r2_cs, max_r2)
        print(f"{P} parameters: " + ", ".join(f"{k} {v}" for k, v in res.items()))
        for label, n in COHORTS.items():
            print(f"    expected shrinkage, {label} (n={n:,}): {expected_shrinkage(P, n, r2_cs):.2f}")


if __name__ == "__main__":
    main()
