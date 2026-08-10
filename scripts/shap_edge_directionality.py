"""Exact-Shapley edge directionality analysis (companion to attention_directionality.py).

Question: attention weight shows where the model LOOKS, not necessarily what actually
DRIVES its prediction. This script measures the latter directly: for a masked target
edge, how much does each nearby labeled context edge causally move the model's
predicted P(positive), as a function of hop-distance (1 or 2) and direction
(forward toward v / backward toward u)?

Scope, confirmed with the user (2026-08-10): LocalAttn4 only, restricted to the local
attention window. local_attention_window=4 is a TOKEN-distance threshold and
1 graph-hop = 2 token positions (CLAUDE.md, "Walk encoding"), so `dist <= 4` covers TWO
hops on each side, not one: token offsets {-4,-2,+2,+4} relative to the target edge's own
position i, i.e. 4 context EDGE features (node tokens are never treated as features --
edge-tokens-only, per the user's call):
    bwd hop2 (i-4)   bwd hop1 (i-2)   [target i]   fwd hop1 (i+2)   fwd hop2 (i+4)

Feature granularity: whole edge (its sign token), not sub-token. Masking: reuse the
model's own <MASK> token (tokenizer["MASK_ID"]) -- the same placeholder already used for
the target edge itself and for split-disallowed edges (src/data/stage_dataset.py). This
keeps every constructed input in-distribution (something the model was trained to handle),
rather than inventing a new out-of-distribution "missing" value.

value(S) = model's predicted P(positive) for the target edge, with exactly the context
edges in S showing their real (already-dataset-masked-as-appropriate) sign token, and every
other context edge (not in S, or outside the walk's bounds / window) replaced by <MASK>.
S ranges over all subsets of whichever of the 4 slots are actually valid for a given walk
instance (in-bounds, real edge position) -- n <= 4, so exact Shapley (up to 2^4=16 forward
evaluations per instance) needs no sampling approximation.

Aggregation: mean |shap| per (dataset, direction, hop) bucket, cluster-robust SE
(cluster = target edge_id, since one edge can occur in several walks) -- one-way cluster
SE (cluster means, then SE of cluster means), simpler than lead4c's two-way
Cameron-Gelbach-Miller machinery since this is a plain mean, not a regression coefficient.

Inference only -- no retraining, no checkpoint modification. Reuses
attention_directionality.py's LOCAL_RUN_INFO / load_model_and_dataset (same E32
checkpoints, already verified fresh) rather than re-deriving checkpoint pins.

Usage
-----
  python scripts/shap_edge_directionality.py [--datasets all] [--max-samples 3000] [--device cuda:1]
  python scripts/shap_edge_directionality.py --datasets bitcoin-alpha --max-samples 200  # smoke test
"""

import argparse
import csv
import itertools
import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.dataset_cache import load_dataset_cache
from scripts.attention_directionality import load_model_and_dataset, ALL_DATASETS

OUT_DIR_DEFAULT = os.path.join(ROOT, "outputs", "shap_edge_directionality")
MAX_SAMPLES_DEFAULT = 20000

# (direction, hop, token offset relative to the target edge's own position i)
FEATURES = [("bwd", 2, -4), ("bwd", 1, -2), ("fwd", 1, 2), ("fwd", 2, 4)]


# ── Exact Shapley over a small feature set ──────────────────────────────────────

def shapley_values(value_of, n):
    """value_of: dict {frozenset(subset indices in range(n)): float}.
    Returns array of length n, exact Shapley value per feature index."""
    import math
    phi = np.zeros(n)
    all_idx = list(range(n))
    for i in all_idx:
        others = [j for j in all_idx if j != i]
        total = 0.0
        for s in range(len(others) + 1):
            weight = math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
            for combo in itertools.combinations(others, s):
                S = frozenset(combo)
                S_plus_i = S | {i}
                total += weight * (value_of[S_plus_i] - value_of[S])
        phi[i] = total
    return phi


# ── Per-dataset analysis ──────────────────────────────────────────────────────

def analyse_dataset(ds_name, out_dir, max_samples=MAX_SAMPLES_DEFAULT, batch_size=32, device="cpu"):
    print(f"\n{'=' * 80}\nDATASET: {ds_name}  [SHAP directionality, LocalAttn4]\n{'=' * 80}")
    t0 = time.time()

    bundle = load_model_and_dataset(ds_name, "local", stage="test")
    if bundle is None:
        return None
    model = bundle["model"].to(device).eval()
    ds = bundle["dataset"]
    ignore_index = bundle["ignore_index"]

    cache_data = load_dataset_cache(bundle["cache_path"], use_mmap=False)
    mask_id = int(cache_data["tokenizer"]["MASK_ID"])
    del cache_data

    n = len(ds)
    if max_samples is not None and n > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, max_samples, replace=False).tolist()
        ds_run = torch.utils.data.Subset(ds, idx)
    else:
        ds_run = ds

    loader = torch.utils.data.DataLoader(
        ds_run, batch_size=batch_size, shuffle=False, collate_fn=bundle["collate"]
    )
    print(f"  N={n:,} test walks (using {len(ds_run):,}), ckpt={os.path.basename(bundle['ckpt_path'])}")

    # records[direction][hop] -> list of (|shap|, edge_id)
    records = {"bwd": {1: [], 2: []}, "fwd": {1: [], 2: []}}
    n_instances = 0
    n_skipped_no_context = 0
    efficiency_gaps = []  # sanity check: sum(shap) vs value(full)-value(empty)

    with torch.no_grad():
        for batch in loader:
            input_ids, labels, attention_mask, metadata = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            edge_ids = metadata["edge_ids"].to(device)
            B, S = input_ids.shape

            target_mask = labels != ignore_index
            rows, cols = target_mask.nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue

            # Build every masked variant for every target in this batch, all at once.
            job_rows = []          # which input row (into input_ids) each job's base sequence comes from
            job_variants = []      # [S] tensor per job
            job_meta = []          # (instance_idx, subset_frozenset, target_pos)
            instance_info = []     # per target instance: (valid_offsets, n_valid, edge_id)

            for k in range(rows.numel()):
                r = int(rows[k]); i = int(cols[k])
                valid = []
                for direction, hop, off in FEATURES:
                    j = i + off
                    if 0 <= j < S and bool(attention_mask[r, j]):
                        valid.append((direction, hop, off, j))
                if len(valid) == 0:
                    n_skipped_no_context += 1
                    continue
                inst_idx = len(instance_info)
                instance_info.append((valid, int(edge_ids[r, i])))
                base = input_ids[r].clone()
                nv = len(valid)
                for bitmask in range(2 ** nv):
                    S_subset = frozenset(fi for fi in range(nv) if (bitmask >> fi) & 1)
                    variant = base.clone()
                    for fi, (direction, hop, off, j) in enumerate(valid):
                        if fi not in S_subset:
                            variant[j] = mask_id
                    job_rows.append(r)
                    job_variants.append(variant)
                    job_meta.append((inst_idx, S_subset, i))

            if not job_variants:
                continue

            big_batch = torch.stack(job_variants, dim=0)  # [n_jobs, S]
            # attention_mask/key-padding is identical across variants of the same base row
            am_expand = attention_mask[job_rows]  # [n_jobs, S]
            logits = model(big_batch, attention_mask=am_expand)  # [n_jobs, S, C]
            probs_pos = torch.softmax(logits, dim=-1)[:, :, 1]  # [n_jobs, S]

            # Gather value(S) per instance
            value_of_per_instance = defaultdict(dict)
            for job_idx, (inst_idx, S_subset, i) in enumerate(job_meta):
                value_of_per_instance[inst_idx][S_subset] = float(probs_pos[job_idx, i])

            for inst_idx, (valid, edge_id) in enumerate(instance_info):
                nv = len(valid)
                value_of = value_of_per_instance[inst_idx]
                phi = shapley_values(value_of, nv)

                full_set = frozenset(range(nv))
                empty_set = frozenset()
                gap = float(phi.sum() - (value_of[full_set] - value_of[empty_set]))
                efficiency_gaps.append(gap)

                for fi, (direction, hop, off, j) in enumerate(valid):
                    records[direction][hop].append((abs(float(phi[fi])), edge_id))
                n_instances += 1

    max_gap = max((abs(g) for g in efficiency_gaps), default=0.0)
    print(f"  n_instances={n_instances:,}  n_skipped(no context)={n_skipped_no_context:,}  "
          f"max efficiency-property gap={max_gap:.2e} (should be ~0, float roundoff only)  "
          f"elapsed={time.time() - t0:.1f}s")

    summary_rows = []
    for direction in ("fwd", "bwd"):
        for hop in (1, 2):
            vals_ids = records[direction][hop]
            if not vals_ids:
                continue
            vals = np.array([v for v, _ in vals_ids])
            eids = np.array([e for _, e in vals_ids])
            mean_shap = float(vals.mean())
            # one-way cluster-robust SE: collapse to per-edge means, then SE across those
            uniq, inv = np.unique(eids, return_inverse=True)
            cluster_means = np.bincount(inv, weights=vals) / np.bincount(inv)
            n_clusters = len(uniq)
            cluster_se = float(cluster_means.std(ddof=1) / np.sqrt(n_clusters)) if n_clusters > 1 else float("nan")
            summary_rows.append({
                "dataset": ds_name, "direction": direction, "hop": hop,
                "mean_abs_shap": mean_shap, "cluster_se": cluster_se,
                "n_obs": len(vals), "n_clusters": n_clusters,
            })
            print(f"    {direction} hop{hop}: mean|shap|={mean_shap:.4f} ± {cluster_se:.4f} "
                  f"(n={len(vals):,}, n_edges={n_clusters:,})")

    result = {
        "ds_name": ds_name, "n_instances": n_instances,
        "n_skipped_no_context": n_skipped_no_context, "max_efficiency_gap": max_gap,
        "records": records, "summary_rows": summary_rows,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"shap_directionality_{ds_name}_result.pkl"), "wb") as f:
        pickle.dump(result, f)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="all")
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=str, default=OUT_DIR_DEFAULT)
    args = parser.parse_args()

    datasets = ALL_DATASETS if args.datasets == "all" else args.datasets.split(",")

    computed = {}
    for ds_name in datasets:
        if ds_name not in ALL_DATASETS:
            print(f"Unknown dataset: {ds_name}")
            continue
        result = analyse_dataset(ds_name, args.out_dir, max_samples=args.max_samples,
                                  batch_size=args.batch_size, device=args.device)
        if result is not None:
            computed[ds_name] = result

    # Merge with any previously-computed per-dataset .pkl results not in this run,
    # so parallel single-dataset invocations (each writing only their own .pkl) don't
    # clobber each other's rows in the combined summary CSV.
    all_rows = []
    for ds_name in ALL_DATASETS:
        if ds_name in computed:
            all_rows.extend(computed[ds_name]["summary_rows"])
            continue
        pkl_path = os.path.join(args.out_dir, f"shap_directionality_{ds_name}_result.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                all_rows.extend(pickle.load(f)["summary_rows"])

    if all_rows:
        out_csv = os.path.join(args.out_dir, "shap_directionality_summary.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["dataset", "direction", "hop", "mean_abs_shap",
                                               "cluster_se", "n_obs", "n_clusters"])
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nwrote {out_csv} ({len(all_rows)} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
