"""Thesis version of the delta-heatmap entropy regression (WSDM Sec. "The gain concentrates where
the entropy is high"): scripts/paper_figures/delta_heatmap_entropy_regression.py run unchanged,
with Pewter(local)'s per-seed predictions swapped for H3's (extract_eid_entropy_heatmaps.h3_raw_seed).

Usage: .venv/bin/python experiments/edge_identity_tokens/eid_delta_regression.py
Output: experiments/edge_identity_tokens/thesis_figure_data/eid_delta_heatmap_entropy_regression.csv
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "paper_figures"))

import delta_heatmap_entropy_regression as reg  # noqa: E402
from experiments.edge_identity_tokens.extract_eid_entropy_heatmaps import h3_raw_seed  # noqa: E402

reg.walk_raw_seed = lambda ds, seed, variant: h3_raw_seed(ds, seed)
reg.OUT_CSV = "experiments/edge_identity_tokens/thesis_figure_data/eid_delta_heatmap_entropy_regression.csv"

if __name__ == "__main__":
    reg.main()
