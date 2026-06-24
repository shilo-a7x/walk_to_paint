"""
Lead 3 (first-hop signal swamping / "fog of war") -- Step 1: synthetic SNR test.

Model-free, graph-free. Tests the core swamping hypothesis in isolation: does
mean aggregation suppress a weak target signal as a function of the magnitude
ratio between a strong diluting signal and the weak target signal, while
concatenation (the walk/attention analogue -- nothing forced to share a slot)
stays unaffected? See ~/.claude/plans/plan-research-leads.md Lead 3 writeup
("Synthetic SNR test") for the full design rationale.

Per-example construction:
  - k 1-hop-like messages: m1_i = s1 * x1_i + noise,  x1_i ~ Bernoulli(0.5) in {-1,+1}
    (x1_i is an arbitrary per-neighbor d=1 label, not itself the quantity being
    recovered -- it's just what's "in the way").
  - one 2-hop-like message: m2 = s2 * (2*z2 - 1) + noise, z2 ~ Bernoulli(0.5) is the
    target label we try to recover.
  - mean aggregation: avg(m1_1..m1_k, m2)  -- a single scalar, target diluted by k+1.
  - concatenation: [m1_1, ..., m1_k, m2]  -- a (k+1)-dim vector, nothing summed away.

For each (s1, s2, k), recoverability of z2 is measured via 5-fold CV logistic
regression AUC (same probe pattern as lead1_layer_probe.py:probe_auc), holding
s2 and noise level fixed and only varying s1/k -- this isolates dilution-by-k and
swamping-by-s1 from any change in z2's own SNR.

Usage:
    python scripts/lead3_synthetic_swamping.py
"""
import argparse
import itertools
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs", "lead3_swamping")

NOISE_STD = 1.0
N_SAMPLES = 4000
SEED = 42


def probe_auc(X, y, seed=SEED):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
    scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
    return scores.mean(), scores.std()


def make_example_batch(n, k, s1, s2, noise_std, rng):
    """Returns (m1: (n,k), m2: (n,), z2: (n,) in {0,1})."""
    x1 = rng.choice([-1.0, 1.0], size=(n, k))
    m1 = s1 * x1 + rng.normal(0, noise_std, size=(n, k))
    z2 = rng.integers(0, 2, size=n)
    m2 = s2 * (2 * z2 - 1) + rng.normal(0, noise_std, size=n)
    return m1, m2, z2


def run_setting(k, s1, s2, n=N_SAMPLES, noise_std=NOISE_STD, seed=SEED):
    rng = np.random.default_rng(seed)
    m1, m2, z2 = make_example_batch(n, k, s1, s2, noise_std, rng)

    mean_repr = np.concatenate([m1, m2[:, None]], axis=1).mean(axis=1, keepdims=True)
    concat_repr = np.concatenate([m1, m2[:, None]], axis=1)

    mean_auc, mean_std = probe_auc(mean_repr, z2, seed)
    concat_auc, concat_std = probe_auc(concat_repr, z2, seed)
    return mean_auc, mean_std, concat_auc, concat_std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s2", type=float, default=1.0, help="fixed target-signal magnitude")
    parser.add_argument("--s1-values", type=float, nargs="+", default=[0.0, 1.0, 2.0, 4.0, 8.0, 16.0])
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    report_path = os.path.join(args.out_dir, "swamping_report.txt")
    rows = []

    print(f"=== Lead 3 synthetic swamping test (s2={args.s2}, noise_std={NOISE_STD}, n={N_SAMPLES}) ===\n")
    header = f"{'k':>3} {'s1':>6} {'mean_AUC':>10} {'concat_AUC':>11} {'gap':>8}"
    print(header)
    lines = [header]

    for k, s1 in itertools.product(args.k_values, args.s1_values):
        mean_auc, mean_std, concat_auc, concat_std = run_setting(k, s1, args.s2)
        gap = concat_auc - mean_auc
        row = f"{k:>3} {s1:>6.2f} {mean_auc:>10.4f} {concat_auc:>11.4f} {gap:>8.4f}"
        print(row)
        lines.append(row)
        rows.append(dict(k=k, s1=s1, s2=args.s2, mean_auc=mean_auc, mean_std=mean_std,
                          concat_auc=concat_auc, concat_std=concat_std, gap=gap))

    print("\n=== Sanity checks ===")
    zero_s2_mean, _, zero_s2_concat, _ = run_setting(k=4, s1=4.0, s2=0.0)
    print(f"s2=0 (no real signal), k=4,s1=4: mean_AUC={zero_s2_mean:.4f}, concat_AUC={zero_s2_concat:.4f} "
          f"(expect both ~0.5)")
    lines.append(f"\nSanity: s2=0,k=4,s1=4 -> mean_AUC={zero_s2_mean:.4f}, concat_AUC={zero_s2_concat:.4f} (expect ~0.5 both)")

    # s1=0 alone is not sufficient to make mean == concat: even pure-noise
    # diluters (s1=0) change the mean's variance via averaging-in extra
    # independent noise (d-prime ~ 1/sqrt(k+1)), a "dilution-by-count" effect
    # distinct from swamping-by-magnitude. The clean isolation needs k=0 too.
    zero_s1_k0_mean, _, zero_s1_k0_concat, _ = run_setting(k=0, s1=0.0, s2=args.s2)
    print(f"s1=0, k=0 (no diluters at all): mean_AUC={zero_s1_k0_mean:.4f}, concat_AUC={zero_s1_k0_concat:.4f} "
          f"(expect identical -- mean/concat of a single value are the same thing)")
    lines.append(f"Sanity: s1=0,k=0 -> mean_AUC={zero_s1_k0_mean:.4f}, concat_AUC={zero_s1_k0_concat:.4f} "
                  f"(expect identical)")

    zero_s1_k4_mean, _, zero_s1_k4_concat, _ = run_setting(k=4, s1=0.0, s2=args.s2)
    print(f"s1=0, k=4 (pure-noise diluters present): mean_AUC={zero_s1_k4_mean:.4f}, "
          f"concat_AUC={zero_s1_k4_concat:.4f} (expect mean < concat: dilution-by-count alone, "
          f"NOT swamping-by-magnitude, since s1=0 -- concat is invariant to k via logistic weights ~0 "
          f"on uninformative dims, mean is not)")
    lines.append(f"Sanity: s1=0,k=4 -> mean_AUC={zero_s1_k4_mean:.4f}, concat_AUC={zero_s1_k4_concat:.4f} "
                  f"(expect mean < concat from dilution-by-count alone, distinct from swamping)")

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
