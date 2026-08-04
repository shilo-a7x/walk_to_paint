"""Extract step for the Attention Directionality figure, Panels B & C -- cross-dataset
forward/backward and node/edge attention-mass summary, LocalAttn4 only, layer 0 only.

Source: outputs/attention_directionality/attention_directionality_<ds>_local_result.pkl
per dataset (already computed by scripts/attention_directionality.py -- no new
inference). Layer-0-restricted mean over heads, matching the "focus on layer 0"
convention already established this session for interpretability (Panel A uses the
same restriction). forward_total/backward_total/node_total/edge_total already exclude
self by construction (self is its own separate category in the source pkl -- the four
totals plus self sum to 1), so no extra filtering is needed to "exclude self" here.
"""
import csv
import os
import pickle

DATASETS = ["bitcoin-alpha", "bitcoin-otc", "epinions", "slashdot090221", "wiki-elec", "wiki-rfa"]
OUT_CSV = "aaai2027/figure_data/attndir_panelBC_summary.csv"
LAYER = 0


def main():
    rows = []
    for ds in DATASETS:
        pkl_path = f"outputs/attention_directionality/attention_directionality_{ds}_local_result.pkl"
        with open(pkl_path, "rb") as f:
            res = pickle.load(f)
        fwd = float(res["forward_total"][LAYER].mean())
        bwd = float(res["backward_total"][LAYER].mean())
        node = float(res["node_total"][LAYER].mean())
        edge = float(res["edge_total"][LAYER].mean())
        slf = float(res["self_total"][LAYER].mean())
        rows.append({
            "dataset": ds, "layer": LAYER, "nhead": res["nhead"],
            "forward": fwd, "backward": bwd, "node": node, "edge": edge, "self": slf,
        })
        print(f"{ds:16s} fwd={fwd:.4f} bwd={bwd:.4f} (fwd-bwd={fwd-bwd:+.4f})  "
              f"node={node:.4f} edge={edge:.4f} (node-edge={node-edge:+.4f})  self={slf:.4f}")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "layer", "nhead", "forward", "backward",
                                           "node", "edge", "self"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
