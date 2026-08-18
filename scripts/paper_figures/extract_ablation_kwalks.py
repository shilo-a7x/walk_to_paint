"""Extract step for the K-ablation (Section 6.5(B) tail): AUC as a function of K, the
number of walk occurrences aggregated per edge, from K=1 upward.

Motivation (see the closeout plan, `\\ph{TODO}` in the tex): PEWTER's headline number
pools many walk-level predictions per edge. If AUC is already near-final at K=1, the
per-walk edge-centric representation is doing the work and ensembling is a minor
bonus; if AUC only becomes competitive at large K, much of the reported gain is
ensembling variance-reduction. This sweep answers that directly from already-saved
per-walk predictions -- no retraining, no new model runs.

Source data: each seed's `test_predictions.pkl` (LocalAttn4, one row per
walk-occurrence, not per edge -- has `edge_ids`, `probabilities`, `targets`). Seed 42
uses the separately-pinned `E32_PY314_LOCALATTN4` checkpoints (predates the
`MULTISEED_*` naming, see `scripts/attention_directionality.py`'s `LOCAL_RUN_INFO`);
seeds 43-51 use `MULTISEED_s<seed>_local_*`.

Aggregation: plain uniform mean of the K sampled walks' probabilities per edge (not
the fitted confidence-weighted aggregator) -- Ablation B already found aggregator
choice barely matters (full spread < 0.0022 AUC), so using the simplest aggregator
here avoids confounding K's effect with a per-K aggregator refit.

K grid: log-spaced (1, 2, 4, 8, ...) since per-edge occurrence counts range from
single digits (wiki-elec, median 3) to the hundreds (bitcoin-alpha, mean ~214, max
~1089) -- a linear grid would either waste points on the sparse datasets or barely
move on the dense ones. For an edge with fewer than K occurrences, all its occurrences
are used (no padding). Sampling is a fixed-seed random draw without replacement, not
occurrence order, to avoid confounding with the walk-corpus generation order.

Usage: .venv/bin/python scripts/paper_figures/extract_ablation_kwalks.py
"""
import csv
import os
import pickle

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASETS = ["bitcoin-alpha", "bitcoin-otc", "epinions", "slashdot090221", "wiki-elec", "wiki-rfa"]
SEEDS = list(range(42, 52))
K_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
OUT_CSV = os.path.join(ROOT, "aaai2027", "figure_data", "ablation_kwalks.csv")

SEED42_RUN_INFO = {
    "bitcoin-alpha": ("E32_PY314_LOCALATTN4_20260804-225948", 35),
    "bitcoin-otc": ("E32_PY314_LOCALATTN4_20260804-231122", 35),
    "epinions": ("E32_PY314_LOCALATTN4_20260804-225948", 30),
    "wiki-elec": ("E32_PY314_LOCALATTN4_20260804-232140", 44),
    "wiki-rfa": ("E32_PY314_LOCALATTN4_20260804-232239", 33),
    "slashdot090221": ("E32_PY314_LOCALATTN4_20260804-225948", 34),
}


def find_predictions_path(ds, seed):
    if seed == 42:
        run_dir, epoch = SEED42_RUN_INFO[ds]
        path = os.path.join(ROOT, "outputs", ds, run_dir, "checkpoints",
                             f"{ds}_predictions", f"epoch_{epoch:03d}", "test_predictions.pkl")
        return path if os.path.exists(path) else None
    import glob
    pattern = os.path.join(ROOT, "outputs", ds, f"MULTISEED_s{seed}_local_*",
                            "checkpoints", f"{ds}_predictions", "epoch_*", "test_predictions.pkl")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def auc_at_k(edge_ids, q, y_occ, k, rng):
    """Subsample up to k occurrences per edge (without replacement), average
    probabilities per edge, compute edge-level AUC."""
    order = np.argsort(edge_ids, kind="stable")
    eids_s, q_s, y_s = edge_ids[order], q[order], y_occ[order]
    uniq, start_idx, counts = np.unique(eids_s, return_index=True, return_counts=True)

    scores = np.empty(len(uniq))
    labels = np.empty(len(uniq), dtype=int)
    for i, (start, cnt) in enumerate(zip(start_idx, counts)):
        idx = np.arange(start, start + cnt)
        if cnt > k:
            idx = rng.choice(idx, size=k, replace=False)
        scores[i] = q_s[idx].mean()
        labels[i] = y_s[idx[0]]
    return roc_auc_score(labels, scores)


def main():
    rows = []
    for ds in DATASETS:
        for seed in SEEDS:
            path = find_predictions_path(ds, seed)
            if path is None:
                print(f"  {ds} seed={seed}: NOT FOUND, skipping")
                continue
            with open(path, "rb") as f:
                preds = pickle.load(f)
            edge_ids = np.asarray(preds["edge_ids"]).astype(np.int64)
            q = preds["probabilities"][:, 1].astype(np.float64)
            y_occ = np.asarray(preds["targets"]).astype(int)

            max_occ = int(np.bincount(edge_ids - edge_ids.min()).max())
            rng = np.random.default_rng(seed)
            for k in K_GRID:
                # once k reaches every edge's max occurrence count, larger k is
                # identical (every edge already uses all its occurrences) -- compute
                # this final saturated point, then stop growing further.
                auc = auc_at_k(edge_ids, q, y_occ, k, rng)
                rows.append({"dataset": ds, "seed": seed, "k": k, "auc": auc, "max_occ": max_occ})
                print(f"  {ds:16s} seed={seed} K={k:5d} AUC={auc:.4f} (max_occ/edge={max_occ})")
                if k >= max_occ:
                    break

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "seed", "k", "auc", "max_occ"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
