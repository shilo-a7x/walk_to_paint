"""Plot for Ablation A -- full attention vs. LocalAttn4, all 6 datasets.

Data (aaai2027/figure_data/ablationA_full_vs_local.csv) is a direct
transcription of CLAUDE.md's "Attention variant: full vs. local" table
(E25/E26 full-attention vs. E27 LocalAttn4, current edge_cover-sampler
production budgets) -- not recomputed here. NOTE (2026-07-26): an earlier
attempt to recompute these AUCs directly from
outputs/lead4_entropy_heterogeneity/computed_data.pkl's raw y/p arrays
reversed the full-vs-local winner on 3/6 datasets (alpha, otc, wiki-rfa) --
that pickle is almost certainly built from older, pre-`edge_cover` checkpoints,
not the E25/E26/E27 pair this table refers to, so it was NOT used here.
Standard error per bar is computed from the real per-dataset test-set
n_pos/n_neg (test_set_counts.csv) via the Hanley-McNeil closed form
(hanley_mcneil.py) -- the AUC point estimates themselves are still the
already-published CLAUDE.md numbers, just with an SE attached.

This is a rough sketch for early feedback: this exact ablation (full vs.
local, matched budgets/sampler) is ON HOLD per your call pending a
from-scratch ablation another agent session is doing -- swap this figure out
once those numbers land.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hanley_mcneil import auc_se

IN_CSV = "aaai2027/figure_data/ablationA_full_vs_local.csv"
COUNTS_CSV = "aaai2027/figure_data/test_set_counts.csv"
OUT_PNG = "aaai2027/figures/ablationA_full_vs_local.png"

DATASET_ORDER = ["bitcoin-alpha", "bitcoin-otc", "epinions", "slashdot090221", "wiki-elec", "wiki-rfa"]
DISPLAY_LABEL = {"slashdot090221": "slashdot"}
FULL_COLOR = "#2a78d6"
LOCAL_COLOR = "#eb6834"


def load():
    aucs = {r["dataset"]: r for r in csv.DictReader(open(IN_CSV))}
    counts = {r["dataset"]: r for r in csv.DictReader(open(COUNTS_CSV))}
    full, full_se, local, local_se = [], [], [], []
    for ds in DATASET_ORDER:
        n_pos, n_neg = int(counts[ds]["n_pos"]), int(counts[ds]["n_neg"])
        fa, la = float(aucs[ds]["full_auc"]), float(aucs[ds]["local_auc"])
        full.append(fa)
        local.append(la)
        full_se.append(auc_se(fa, n_pos, n_neg))
        local_se.append(auc_se(la, n_pos, n_neg))
    return full, full_se, local, local_se


def main():
    full, full_se, local, local_se = load()
    x = np.arange(len(DATASET_ORDER))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 3.4))

    ax.bar(x - w / 2, full, width=w, yerr=full_se, capsize=3, color=FULL_COLOR, label="Full attention")
    ax.bar(x + w / 2, local, width=w, yerr=local_se, capsize=3, color=LOCAL_COLOR, label="LocalAttn4")

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_LABEL.get(d, d) for d in DATASET_ORDER], fontsize=8, rotation=15)
    ax.set_ylabel("test AUC")
    ax.set_ylim(0.85, 1.0)
    ax.grid(True, axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
