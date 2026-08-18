"""Cluster-robust SE for the attention forward/backward split (Panel B) AND the
vertex/edge split (Panel C) -- companion to attention_directionality.py, same spirit as
shap_edge_directionality.py.

Motivation: attention_directionality.py's saved pickle only keeps the AVERAGED
forward_total/backward_total/node_total/edge_total per (layer, head) -- the per-instance
values are summed away during the batch loop and never written to disk, so no error bar
can be computed from what's already there. This script reruns inference (same
LOCAL_RUN_INFO checkpoints, same max_samples=20000 cap, same 42-seeded subsampling as
attention_directionality.py) but retains the PER-INSTANCE layer-0 forward/backward AND
node/edge mass fractions together with each instance's target edge_id, so a
cluster-robust SE (cluster = target edge_id, matching shap_edge_directionality.py's
convention -- one edge can occur in several walks) can be computed for both splits from
a single inference pass.

Caching, so this expensive step (a real inference pass) is never re-paid just to
recompute either SE: raw per-instance arrays are saved to
outputs/attention_directionality/<ds>_local_panelBC_perinstance.pkl per dataset. Delete
a file to force a recompute for that dataset (same "safe to delete/regenerate"
convention as the other cached figure-generation intermediates in this project). Both
SEs are a cheap final step, computed fresh each run from whatever's cached on disk.

Layer 0 only, local variant only -- matches what Section 6.4 of the paper actually
reports ("attention mass at layer 0").

Verified 2026-08-18: forward/backward split is significant (cluster-robust CI excludes
zero, confirmed by Wilcoxon on the same cluster means) on all 6 datasets.

Usage
-----
  .venv/bin/python scripts/attention_directionality_panelB_se.py [--datasets all] [--device cuda:0]
"""
import argparse
import csv
import os
import pickle
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)

torch.backends.mha.set_fastpath_enabled(False)

from scripts.attention_directionality import load_model_and_dataset, ALL_DATASETS, MAX_SAMPLES_DEFAULT  # noqa: E402

OUT_DIR = os.path.join(ROOT, "outputs", "attention_directionality")
LAYER = 0


def perinstance_path(ds_name):
    return os.path.join(OUT_DIR, f"{ds_name}_local_panelBC_perinstance.pkl")


def run_inference(ds_name, max_samples=MAX_SAMPLES_DEFAULT, batch_size=64, device="cpu"):
    print(f"\n{'=' * 80}\nDATASET: {ds_name}  [local, layer {LAYER}]\n{'=' * 80}")
    t0 = time.time()

    bundle = load_model_and_dataset(ds_name, "local", stage="test")
    if bundle is None:
        return None

    model = bundle["model"].to(device).eval()
    ds = bundle["dataset"]
    ignore_index = bundle["ignore_index"]

    n = len(ds)
    if max_samples is not None and n > max_samples:
        rng = np.random.default_rng(42)  # same seed as attention_directionality.py's own subsampling
        idx = rng.choice(n, max_samples, replace=False).tolist()
        ds_run = torch.utils.data.Subset(ds, idx)
    else:
        ds_run = ds

    loader = torch.utils.data.DataLoader(
        ds_run, batch_size=batch_size, shuffle=False, collate_fn=bundle["collate"]
    )
    print(f"  N={n:,} test walks (using {len(ds_run):,}), ckpt={os.path.basename(bundle['ckpt_path'])}")

    edge_id_list = []
    fwd_frac_list = []
    bwd_frac_list = []
    node_frac_list = []
    edge_frac_list = []

    with torch.no_grad():
        for batch in loader:
            input_ids, labels, attention_mask, metadata = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            target_mask = labels != ignore_index
            rows, cols = target_mask.nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue

            _ = model(input_ids, attention_mask=attention_mask)

            rows_d = rows.to(device)
            cols_d = cols.to(device)
            S = input_ids.shape[1]
            am = attention_mask.float()
            j_idx = torch.arange(S, device=device)
            is_edge_pos = (j_idx % 2 == 1)  # odd position = edge token, matching attention_directionality.py

            dist = (j_idx.unsqueeze(0) - cols_d.unsqueeze(1))  # [M, S]
            fwd_mask = (dist > 0).float()
            bwd_mask = (dist < 0).float()
            not_self = (dist != 0).float()  # exclude self-attention (d=0), matching
            # attention_directionality.py's convention -- node_total/edge_total there are
            # built only from the fwd/bwd masked sums, never from the self position.
            edge_j = is_edge_pos.unsqueeze(0).expand_as(dist).float() * not_self  # [M, S]
            node_j = (1.0 - is_edge_pos.unsqueeze(0).expand_as(dist).float()) * not_self

            layer = model.transformer.layers[LAYER]
            attn = layer.last_attn_weights  # [B, nhead, S, S]
            sel = attn[rows_d, :, cols_d, :]  # [M, nhead, S]
            valid = sel * am[rows_d].unsqueeze(1)  # [M, nhead, S]
            valid_mean_heads = valid.mean(dim=1)  # [M, S] -- mean over heads, matching the paper's "mean over heads"

            fwd_frac = (valid_mean_heads * fwd_mask).sum(dim=1)  # [M]
            bwd_frac = (valid_mean_heads * bwd_mask).sum(dim=1)  # [M]
            node_frac = (valid_mean_heads * node_j).sum(dim=1)  # [M]
            edge_frac = (valid_mean_heads * edge_j).sum(dim=1)  # [M]

            target_edge_ids = metadata["edge_ids"][rows, cols]  # [M]

            edge_id_list.append(target_edge_ids.cpu().numpy())
            fwd_frac_list.append(fwd_frac.cpu().numpy())
            bwd_frac_list.append(bwd_frac.cpu().numpy())
            node_frac_list.append(node_frac.cpu().numpy())
            edge_frac_list.append(edge_frac.cpu().numpy())

    edge_ids = np.concatenate(edge_id_list)
    fwd_frac = np.concatenate(fwd_frac_list)
    bwd_frac = np.concatenate(bwd_frac_list)
    node_frac = np.concatenate(node_frac_list)
    edge_frac = np.concatenate(edge_frac_list)

    print(f"  n_targets={len(edge_ids):,}  n_distinct_edges={len(np.unique(edge_ids)):,}  "
          f"elapsed={time.time() - t0:.1f}s")
    print(f"  mean forward={fwd_frac.mean():.4f}  mean backward={bwd_frac.mean():.4f}")
    print(f"  mean node={node_frac.mean():.4f}  mean edge={edge_frac.mean():.4f}")

    result = {
        "ds_name": ds_name, "variant": "local", "layer": LAYER,
        "edge_ids": edge_ids, "fwd_frac": fwd_frac, "bwd_frac": bwd_frac,
        "node_frac": node_frac, "edge_frac": edge_frac,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = perinstance_path(ds_name)
    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"  wrote {out_path}")
    return result


def cluster_robust_se_diff(edge_ids, a_frac, b_frac):
    """One-way cluster-robust SE (cluster = target edge_id) on the (a-minus-b)
    difference -- generic over which two mass fractions are compared (forward/backward
    for Panel B, node/edge for Panel C). Same recipe as shap_edge_directionality.py:
    collapse to per-edge means, then SE across those cluster means. Also runs a
    Wilcoxon signed-rank test on the same per-edge cluster means as a distribution-free
    robustness check (the CI test above is a z-test on cluster means, which assumes
    those means are reasonably well-behaved -- attention-mass fractions are bounded in
    [0,1] and likely skewed, so Wilcoxon, which makes no such assumption, is worth
    checking agrees). Returns both; on every Panel B dataset checked 2026-08-18 the two
    methods agreed on direction and significance by a wide margin (Wilcoxon p ranging
    1.4e-44 to machine-precision-zero)."""
    diff = a_frac - b_frac
    order = np.argsort(edge_ids, kind="stable")
    eids_s, diff_s = edge_ids[order], diff[order]
    uniq, inv = np.unique(eids_s, return_inverse=True)
    cluster_means = np.bincount(inv, weights=diff_s) / np.bincount(inv)
    n_clusters = len(uniq)
    mean_diff = float(cluster_means.mean())
    se = float(cluster_means.std(ddof=1) / np.sqrt(n_clusters)) if n_clusters > 1 else float("nan")
    ci_lo, ci_hi = mean_diff - 1.96 * se, mean_diff + 1.96 * se

    from scipy import stats
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(cluster_means)

    return mean_diff, se, ci_lo, ci_hi, n_clusters, float(wilcoxon_p)


def cluster_robust_se_single(edge_ids, frac):
    """Cluster-robust mean + SE (cluster = target edge_id) for a single quantity (not a
    difference) -- for putting an error bar on one bar of a grouped bar chart, e.g.
    Panel B's forward bar alone. Same cluster-mean-then-SE-across-clusters recipe as
    cluster_robust_se_diff, just for one array instead of a difference of two."""
    order = np.argsort(edge_ids, kind="stable")
    eids_s, frac_s = edge_ids[order], frac[order]
    uniq, inv = np.unique(eids_s, return_inverse=True)
    cluster_means = np.bincount(inv, weights=frac_s) / np.bincount(inv)
    n_clusters = len(uniq)
    mean_val = float(cluster_means.mean())
    se = float(cluster_means.std(ddof=1) / np.sqrt(n_clusters)) if n_clusters > 1 else float("nan")
    return mean_val, se


SE_CSV = os.path.join(ROOT, "aaai2027", "figure_data", "attndir_panelBC_se.csv")


def write_se_csv(all_results):
    """Per-dataset, per-category cluster-robust mean+SE for all 4 mass categories
    (forward, backward, node, edge) -- consumed by plot_attndir_panelB_direction.py /
    plot_attndir_panelC_nodeedge.py to render error bars on each bar."""
    rows = []
    for ds_name, result in all_results.items():
        row = {"dataset": ds_name}
        for cat, arr in [("forward", result["fwd_frac"]), ("backward", result["bwd_frac"]),
                          ("node", result["node_frac"]), ("edge", result["edge_frac"])]:
            mean_val, se = cluster_robust_se_single(result["edge_ids"], arr)
            row[f"{cat}_mean"] = mean_val
            row[f"{cat}_se"] = se
        rows.append(row)

    os.makedirs(os.path.dirname(SE_CSV), exist_ok=True)
    fieldnames = ["dataset", "forward_mean", "forward_se", "backward_mean", "backward_se",
                  "node_mean", "node_se", "edge_mean", "edge_se"]
    with open(SE_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {SE_CSV} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["all"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--force", action="store_true", help="recompute even if cached perinstance pkl exists")
    args = ap.parse_args()

    device = args.device
    if device != "cpu" and not torch.cuda.is_available():
        print("  CUDA not available, falling back to cpu")
        device = "cpu"
    if device != "cpu" and device.isdigit():
        device = f"cuda:{device}"

    datasets = ALL_DATASETS if args.datasets == ["all"] else args.datasets

    print(f"\n{'#' * 80}\nSUMMARY: cluster-robust SE, Panel B (fwd-bwd) and Panel C (node-edge), layer {LAYER}\n{'#' * 80}")
    all_results = {}
    for ds_name in datasets:
        out_path = perinstance_path(ds_name)
        if os.path.exists(out_path) and not args.force:
            print(f"\n{ds_name}: using cached {out_path}")
            with open(out_path, "rb") as f:
                result = pickle.load(f)
        else:
            result = run_inference(ds_name, batch_size=args.batch_size, device=device)
            if result is None:
                continue
        all_results[ds_name] = result

        fb_mean, fb_se, fb_lo, fb_hi, n_clusters, fb_wp = cluster_robust_se_diff(
            result["edge_ids"], result["fwd_frac"], result["bwd_frac"])
        fb_sig = "significant" if (fb_lo > 0 or fb_hi < 0) else "not significant"
        print(f"{ds_name:16s} fwd-bwd ={fb_mean:+.4f}  SE={fb_se:.4f}  95% CI=[{fb_lo:+.4f},{fb_hi:+.4f}]  "
              f"({fb_sig}, n_edges={n_clusters:,})  Wilcoxon p={fb_wp:.3e}")

        ne_mean, ne_se, ne_lo, ne_hi, n_clusters2, ne_wp = cluster_robust_se_diff(
            result["edge_ids"], result["node_frac"], result["edge_frac"])
        ne_sig = "significant" if (ne_lo > 0 or ne_hi < 0) else "not significant"
        print(f"{ds_name:16s} node-edge={ne_mean:+.4f}  SE={ne_se:.4f}  95% CI=[{ne_lo:+.4f},{ne_hi:+.4f}]  "
              f"({ne_sig}, n_edges={n_clusters2:,})  Wilcoxon p={ne_wp:.3e}")

    write_se_csv(all_results)


if __name__ == "__main__":
    main()
