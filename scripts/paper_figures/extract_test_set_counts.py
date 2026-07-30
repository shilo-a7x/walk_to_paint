"""Extract the real positive/negative test-edge counts per dataset from the
canonical split (baselines/splits_canonical/<ds>.pt). These are the (n_pos,
n_neg) values fed into the Hanley-McNeil SE formula (hanley_mcneil.py) for
every AUC reported in the paper.

Verified 2026-07-26: these counts are identical to the ones baked into
outputs/lead4_entropy_heterogeneity/computed_data.pkl for walk_full,
walk_localattn4, GINEConv, and SiGAT alike (all four share the exact same
y-array length and class split per dataset) -- i.e. with edge_cover's ~100%
coverage, "each model's own full test set" and the "shared apples-to-apples
set" are now the same set in practice, so a single n_pos/n_neg per dataset is
valid to use across every baseline's AUC, not just the ones we have raw
per-edge scores for.
"""
import csv
import os

import torch

DATASETS = ["bitcoin-alpha", "bitcoin-otc", "epinions", "wiki-elec", "wiki-rfa", "slashdot090221"]
SPLITS_DIR = "baselines/splits_canonical"
OUT_CSV = "aaai2027/figure_data/test_set_counts.csv"


def main():
    rows = []
    for ds in DATASETS:
        d = torch.load(os.path.join(SPLITS_DIR, f"{ds}.pt"), weights_only=False)
        tst_mask = d["tst_mask"]
        y = d["edge_weight"][tst_mask]
        n_pos = int((y > 0).sum().item())
        n_neg = int((y <= 0).sum().item())
        rows.append({"dataset": ds, "n_pos": n_pos, "n_neg": n_neg, "n_total": n_pos + n_neg})
        print(ds, n_pos, n_neg)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "n_pos", "n_neg", "n_total"])
        w.writeheader()
        w.writerows(rows)
    print(f"saved {OUT_CSV}")


if __name__ == "__main__":
    main()
