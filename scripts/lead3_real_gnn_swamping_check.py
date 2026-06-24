"""
Lead 3 -- Step 2: real-GNN magnitude-ratio check.

Plugs Lead 2's already-measured per-edge dilution levels (contribution_share,
from scripts/lead2_edge_sensitivity.py: ||message(v,q)|| / sum_q' ||message(v,q')||)
into Lead 3 Step 1's synthetic swamping curve (scripts/lead3_synthetic_swamping.py)
to predict, per dataset / GNN type / degree bucket, whether real GNNs are
operating in the "signal destroyed" or "still recoverable" swamping regime.

No retraining, no new GNN forward passes -- reads the existing
outputs/lead2_gnn_bottleneck/*_edge_sensitivity.pkl artifacts directly.

Inversion: contribution_share for an edge (v,q) is, by definition, q's
message norm as a fraction of v's *total* aggregate message norm. For k
*equal-magnitude* messages sharing an aggregate, share = 1/k exactly. So
k_eff := 1/mean_share is the "effective number of equal-weight diluters"
implied by the measured share -- the same k the synthetic test sweeps over.
This is an approximation (real messages aren't equal-magnitude) but it is
the natural, parameter-free way to map a measured share onto the synthetic
curve's k-axis without inventing a new free parameter.

Because the true ratio between d=1-like and d=2-like message magnitudes in
a real GNN is unknown, two representative ratios are evaluated side by side:
  - s1=s2 (matched magnitude) -- a conservative/neutral assumption.
  - s1=4*s2 (d=1 evidence 4x stronger) -- plausible given Lead 1/2's finding
    that d=1 sign MI is 10-1000x larger than d=2+ MI; this is the more
    swamping-prone, arguably more realistic regime.

Usage:
    python scripts/lead3_real_gnn_swamping_check.py
"""
import os
import pickle

import numpy as np

from scripts.lead3_synthetic_swamping import run_setting

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART_DIR = os.path.join(ROOT, "outputs", "lead2_gnn_bottleneck")
OUT_DIR = os.path.join(ROOT, "outputs", "lead3_swamping")

DATASETS = ["bitcoin-alpha", "bitcoin-otc", "epinions", "wiki-elec", "wiki-rfa", "slashdot090221"]
MODELS = ["CSG", "GINEConv"]
RATIOS = [1.0, 4.0]  # s1/s2
MAX_K_EFF_CAP = 64  # synthetic curve sampled densely up to this k; beyond, regime is unambiguous
N_BUCKETS = 4  # matches lead2_edge_sensitivity.py's N_BUCKETS, duplicated here to avoid
               # importing that module's torch_geometric dependency for one helper


def quantile_buckets(values: np.ndarray, n_buckets: int = N_BUCKETS):
    """Verbatim copy of lead2_edge_sensitivity.py:quantile_buckets -- kept identical so
    bucket boundaries here exactly reproduce Lead 2's published buckets."""
    edges = np.quantile(values, np.linspace(0, 1, n_buckets + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return np.digitize(values, edges[1:-1], right=True)


def build_swamping_lookup(ratios, k_grid, s2=1.0):
    """{ratio: {k: mean_auc}} via lead3's exact run_setting (same probe, same noise)."""
    lookup = {}
    for ratio in ratios:
        lookup[ratio] = {}
        for k in k_grid:
            mean_auc, _, concat_auc, _ = run_setting(max(k, 0), s1=ratio * s2, s2=s2)
            lookup[ratio][k] = (mean_auc, concat_auc)
    return lookup


def predict_mean_auc(lookup, ratio, k_eff):
    """Nearest-neighbor lookup against the precomputed integer-k grid (k must be int for the
    synthetic test's array shapes; k_eff from real data is continuous)."""
    grid = sorted(lookup[ratio].keys())
    k_clamped = min(max(k_eff, grid[0]), grid[-1])
    nearest = min(grid, key=lambda k: abs(k - k_clamped))
    return lookup[ratio][nearest]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    k_grid = [0, 1, 2, 4, 8, 16, 24, 32, 48, 64]
    lookup = build_swamping_lookup(RATIOS, k_grid)

    lines = ["=" * 100,
             "  LEAD 3 STEP 2 -- real-GNN dilution levels plugged into the synthetic swamping curve",
             "=" * 100,
             "",
             "k_eff = 1/mean_share (effective # of equal-weight diluters implied by Lead 2's measured",
             "contribution_share). predicted_mean_auc looks up the nearest k on the Lead 3 Step 1",
             "synthetic curve at s1=ratio*s2 -- i.e. 'if this dataset's measured dilution were plugged",
             "into the idealized swamping experiment, how recoverable would the weak signal be.'",
             "regime: 'destroyed' if predicted_mean_auc < 0.55 (near chance), 'partial' if < 0.75,",
             "else 'recoverable'.",
             "",
             ]
    header = (f"{'dataset':<16}{'model':<10}{'bucket':>7}{'degree range':>16}{'mean_share':>11}"
               f"{'k_eff':>8}" + "".join(f"  pred_auc(r={r:g})" for r in RATIOS) + "  regime(r=4)")
    print(header)
    lines.append(header)

    for ds in DATASETS:
        for model in MODELS:
            pkl_path = os.path.join(ART_DIR, f"{ds}_{model}_edge_sensitivity.pkl")
            if not os.path.exists(pkl_path):
                continue
            with open(pkl_path, "rb") as f:
                res = pickle.load(f)
            degree = res["degree"]
            share = res["contribution_share"]
            buckets = quantile_buckets(degree)
            for b in range(N_BUCKETS):
                mask = buckets == b
                if mask.sum() == 0:
                    continue
                mean_share = float(share[mask].mean())
                k_eff = min(1.0 / max(mean_share, 1e-9), MAX_K_EFF_CAP)
                deg_range = f"[{degree[mask].min():.0f},{degree[mask].max():.0f}]"
                preds = []
                for ratio in RATIOS:
                    mean_auc, concat_auc = predict_mean_auc(lookup, ratio, k_eff)
                    preds.append(mean_auc)
                regime = "destroyed" if preds[-1] < 0.55 else ("partial" if preds[-1] < 0.75 else "recoverable")
                row = (f"{ds:<16}{model:<10}{b:>7}{deg_range:>16}{mean_share:>11.5f}{k_eff:>8.2f}"
                       + "".join(f"{p:>17.4f}" for p in preds) + f"  {regime}")
                print(row)
                lines.append(row)

    with open(os.path.join(OUT_DIR, "real_gnn_swamping_check.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {os.path.join(OUT_DIR, 'real_gnn_swamping_check.txt')}")


if __name__ == "__main__":
    main()
