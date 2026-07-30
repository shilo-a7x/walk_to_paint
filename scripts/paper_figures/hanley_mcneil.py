"""Hanley-McNeil (1982) closed-form standard error for an AUC point estimate.

SE(AUC) = sqrt( [AUC(1-AUC) + (n_pos-1)(Q1-AUC^2) + (n_neg-1)(Q2-AUC^2)] / (n_pos*n_neg) )
with Q1 = AUC/(2-AUC), Q2 = 2*AUC^2/(1+AUC).

Only needs the AUC point estimate plus the positive/negative counts of the test
set it was measured on -- no raw per-edge scores required. This is the
single source of truth for every AUC +/- SE reported anywhere in the paper
(tables and plot error bars alike), per the advisor's guidance logged in
PEWTER_ASSETS_CHECKLIST.md item #17 (single-split Hanley-McNeil/DeLong SE,
no multi-seed/CV).
"""
import math


def auc_se(auc, n_pos, n_neg):
    """Hanley-McNeil standard error for a single AUC estimate."""
    if n_pos <= 1 or n_neg <= 1:
        return float("nan")
    q1 = auc / (2 - auc)
    q2 = (2 * auc ** 2) / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    return math.sqrt(max(var, 0.0))


def fmt_auc_se(auc, n_pos, n_neg, decimals=4):
    se = auc_se(auc, n_pos, n_neg)
    return f"{auc:.{decimals}f} $\\pm$ {se:.{decimals}f}"


if __name__ == "__main__":
    # plausibility check: SE should shrink as n grows, for a fixed AUC
    for n in (50, 500, 5000, 50000):
        print(f"AUC=0.90, n_pos=n_neg={n} -> SE={auc_se(0.90, n, n):.5f}")
