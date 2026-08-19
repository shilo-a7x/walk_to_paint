"""Exact-Shapley vertex-vs-edge causal contribution (companion to shap_edge_directionality.py).

Question: Panel C of Figure 3 measures raw attention MASS split between vertex tokens and
edge tokens, but the paper already knows (Panel D, direction case) that attention mass and
causal contribution can disagree -- so a "prediction" claim about vertex/edge role needs its
own causal check, not a reuse of the direction Shapley or the raw attention numbers.

Scope change from shap_edge_directionality.py's original 2026-08-10 decision: that script
deliberately excluded vertex tokens as Shapley players ("edge-tokens-only, per the user's
call"). This script reverses that scope for a specific new question -- vertex-vs-edge causal
role -- and is a separate script, not a modification of the original (which still backs
Panel D exactly as before).

Feature set: within the local attention window, each side has three maskable slots instead
of the original two -- the hop-1 edge (i+-2), the "outer" vertex at the far end of that edge
(i+-3, a genuinely new vertex, not the target edge's own u/v which sit at i+-1 and are never
masked), and the hop-2 edge (i+-4):
    bwd edge h2 (i-4)  bwd vertex (i-3)  bwd edge h1 (i-2)  [target i]  fwd edge h1 (i+2)  fwd vertex (i+3)  fwd edge h2 (i+4)
n<=6 valid slots per instance -> exact Shapley needs up to 2^6=64 forward evaluations per
instance (vs. 16 for the direction-only version) -- same masking scheme (model's own <MASK>
token), same exact-Shapley weight, same cluster-robust SE, same efficiency-property check.

Aggregation: mean |shap| per (dataset, role) bucket, where role in {vertex, edge}, pooling
both directions and both edge hops into "edge" (matches Panel C's own vertex/edge grouping,
not the hop-level granularity of the direction-only script) -- plus a (dataset, direction,
role) breakdown for a finer look if needed.

Usage
-----
  python scripts/shap_edge_vertex_role.py [--datasets all] [--max-samples 20000] [--device cuda:0]
  python scripts/shap_edge_vertex_role.py --datasets bitcoin-alpha --max-samples 200  # smoke test
"""

import argparse
import csv
import itertools
import math
import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.dataset_cache import load_dataset_cache
from scripts.attention_directionality import load_model_and_dataset, ALL_DATASETS

OUT_DIR_DEFAULT = os.path.join(ROOT, "outputs", "shap_edge_vertex_role")
MAX_SAMPLES_DEFAULT = 20000

# (direction, role, offset) -- role in {"edge", "vertex"}
FEATURES = [
    ("bwd", "edge", -4), ("bwd", "vertex", -3), ("bwd", "edge", -2),
    ("fwd", "edge", 2), ("fwd", "vertex", 3), ("fwd", "edge", 4),
]


def shapley_values(value_of, n):
    """Exact Shapley value per feature index, weight = 1/(n * C(n-1, s))."""
    phi = np.zeros(n)
    all_idx = list(range(n))
    for i in all_idx:
        others = [j for j in all_idx if j != i]
        total = 0.0
        for s in range(len(others) + 1):
            weight = 1.0 / (n * math.comb(n - 1, s))
            for combo in itertools.combinations(others, s):
                S = frozenset(combo)
                S_plus_i = S | {i}
                total += weight * (value_of[S_plus_i] - value_of[S])
        phi[i] = total
    return phi


def analyse_dataset(ds_name, out_dir, max_samples=MAX_SAMPLES_DEFAULT, batch_size=32, device="cpu"):
    print(f"\n{'=' * 80}\nDATASET: {ds_name}  [SHAP vertex/edge role, LocalAttn4]\n{'=' * 80}")
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

    # records[role] -> list of (|shap|, edge_id); records_dir[(direction, role)] -> same
    records = {"vertex": [], "edge": []}
    records_dir = defaultdict(list)
    n_instances = 0
    n_skipped_no_context = 0
    efficiency_gaps = []

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

            job_rows, job_variants, job_meta = [], [], []
            instance_info = []

            for k in range(rows.numel()):
                r = int(rows[k]); i = int(cols[k])
                valid = []
                for direction, role, off in FEATURES:
                    j = i + off
                    if 0 <= j < S and bool(attention_mask[r, j]):
                        valid.append((direction, role, off, j))
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
                    for fi, (direction, role, off, j) in enumerate(valid):
                        if fi not in S_subset:
                            variant[j] = mask_id
                    job_rows.append(r)
                    job_variants.append(variant)
                    job_meta.append((inst_idx, S_subset, i))

            if not job_variants:
                continue

            big_batch = torch.stack(job_variants, dim=0)
            am_expand = attention_mask[job_rows]
            logits = model(big_batch, attention_mask=am_expand)
            probs_pos = torch.softmax(logits, dim=-1)[:, :, 1]

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

                for fi, (direction, role, off, j) in enumerate(valid):
                    records[role].append((abs(float(phi[fi])), edge_id))
                    records_dir[(direction, role)].append((abs(float(phi[fi])), edge_id))
                n_instances += 1

    max_gap = max((abs(g) for g in efficiency_gaps), default=0.0)
    print(f"  n_instances={n_instances:,}  n_skipped(no context)={n_skipped_no_context:,}  "
          f"max efficiency-property gap={max_gap:.2e} (should be ~0, float roundoff only)  "
          f"elapsed={time.time() - t0:.1f}s")

    def _summarize(vals_ids, extra):
        vals = np.array([v for v, _ in vals_ids])
        eids = np.array([e for _, e in vals_ids])
        mean_shap = float(vals.mean())
        uniq, inv = np.unique(eids, return_inverse=True)
        cluster_means = np.bincount(inv, weights=vals) / np.bincount(inv)
        n_clusters = len(uniq)
        cluster_se = float(cluster_means.std(ddof=1) / np.sqrt(n_clusters)) if n_clusters > 1 else float("nan")
        row = {"dataset": ds_name, "mean_abs_shap": mean_shap, "cluster_se": cluster_se,
               "n_obs": len(vals), "n_clusters": n_clusters}
        row.update(extra)
        return row

    summary_rows = []
    for role in ("vertex", "edge"):
        if not records[role]:
            continue
        row = _summarize(records[role], {"role": role})
        summary_rows.append(row)
        print(f"    {role}: mean|shap|={row['mean_abs_shap']:.4f} +- {row['cluster_se']:.4f} "
              f"(n={row['n_obs']:,}, n_edges={row['n_clusters']:,})")

    dir_summary_rows = []
    for (direction, role), vals_ids in records_dir.items():
        if not vals_ids:
            continue
        row = _summarize(vals_ids, {"direction": direction, "role": role})
        dir_summary_rows.append(row)

    result = {
        "ds_name": ds_name, "n_instances": n_instances,
        "n_skipped_no_context": n_skipped_no_context, "max_efficiency_gap": max_gap,
        "records": records, "records_dir": dict(records_dir),
        "summary_rows": summary_rows, "dir_summary_rows": dir_summary_rows,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"shap_role_{ds_name}_result.pkl"), "wb") as f:
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

    all_rows = []
    for ds_name in ALL_DATASETS:
        if ds_name in computed:
            all_rows.extend(computed[ds_name]["summary_rows"])
            continue
        pkl_path = os.path.join(args.out_dir, f"shap_role_{ds_name}_result.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                all_rows.extend(pickle.load(f)["summary_rows"])

    if all_rows:
        os.makedirs(args.out_dir, exist_ok=True)
        out_csv = os.path.join(args.out_dir, "shap_role_summary.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["dataset", "role", "mean_abs_shap", "cluster_se",
                                               "n_obs", "n_clusters"])
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nsaved {out_csv}")


if __name__ == "__main__":
    main()
