"""
Lead 2 (GNN bottleneck) -- Step 1b: per-edge bottleneck sensitivity.

Complements lead2_gnn_bottleneck_mi.py's node-level MI measurement (a
population statistic: does v's embedding *on average* lose neighbor-sign
info?) with a literal per-edge score: does removing ONE specific edge (v,q)
change v's own h_v^(1), relative to v's other incident edges? Exact and O(1)
per edge -- no per-edge re-forward-pass -- by exploiting each layer's
additive (GINEConv, aggr='add') or mean (SignedConv, aggr='mean') aggregation
structure directly: for a sum, leave-one-out is just subtraction; for a mean,
it's (sum - term) / (count - 1).

Two scores per edge (v,q), both LOW for "this edge got swamped by v's other
neighbors" (the bottleneck signature):
  contribution_share(v,q) = ||message(v,q)|| / sum_q' ||message(v,q')||
      -- pre-MLP, cheapest, "what fraction of v's aggregated input came from q"
  leave_one_out_delta(v,q) = ||h_v^(1) - h_v^(1)_without_(v,q)||
      -- post-MLP, captures nonlinear interactions

Usage
-----
  python scripts/lead2_edge_sensitivity.py --datasets bitcoin-alpha --models GINEConv
  (must run with the directed_gnn conda env -- needs torch_geometric)
"""
import os, sys, pickle, argparse, math
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "baselines", "GINEConv"))
sys.path.insert(0, os.path.join(ROOT, "baselines", "CSG"))

ALL_DATASETS = ["bitcoin-alpha", "bitcoin-otc", "epinions",
                "wiki-elec", "wiki-rfa", "slashdot090221"]
N_BUCKETS = 4


def load_splits(ds_name: str):
    splits = torch.load(os.path.join(ROOT, "baselines", "splits", f"{ds_name}.pt"),
                         weights_only=False)
    ei = splits["edge_index"]
    ew = splits["edge_weight"]
    trn_mask = splits["trn_mask"]
    num_nodes = int(ei.max().item()) + 1
    return ei[:, trn_mask], ew[trn_mask].float(), num_nodes


def reconstruct_x(num_nodes: int, dim: int, seed: int = 42) -> torch.Tensor:
    """Deterministic reconstruction of the random input features used during
    training -- matches run_with_our_splits.py's `torch_geometric.seed_everything
    (seed); np.random.seed(seed); x = np.random.rand(...)` sequence exactly
    (x only depends on numpy's RNG state right after np.random.seed(seed))."""
    torch_geometric.seed_everything(seed)
    np.random.seed(seed)
    return torch.from_numpy(np.random.rand(num_nodes, dim).astype(np.float32))


def quantile_buckets(values: np.ndarray, n_buckets: int = N_BUCKETS):
    edges = np.quantile(values, np.linspace(0, 1, n_buckets + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return np.digitize(values, edges[1:-1], right=True)


# ── GINEConv: exact O(1)-per-edge sensitivity via additive aggregation ───────

def gineconv_sensitivity(ds_name: str, art_dir: str = "seed42"):
    from model import GINEConvNet

    art_path = os.path.join(ROOT, "baselines", "GINEConv", "results_our_splits",
                             ds_name, "GINEConv", art_dir, "best_epoch_artifacts.pkl")
    with open(art_path, "rb") as f:
        art = pickle.load(f)

    edge_index, edge_weight, num_nodes = load_splits(ds_name)
    x = reconstruct_x(num_nodes, 64)
    edge_attr = edge_weight.unsqueeze(-1)

    model = GINEConvNet(in_dim=64, hid_dim=64, num_layers=2, edge_dim=1, aggr='add')
    model.load_state_dict(art["state_dict"])
    model.eval()

    conv1 = model.convs[0]
    captured = {}
    orig_message = conv1.message
    def hooked_message(x_j, edge_attr):
        out = orig_message(x_j, edge_attr)
        captured["msg"] = out.detach()
        return out
    conv1.message = hooked_message

    with torch.no_grad():
        layers = model.encode(x, edge_index, edge_attr, return_all_layers=True)
    conv1.message = orig_message
    h1_cached = layers[1]

    # sanity: does our reconstruction match the cached artifact's h_v^(1)?
    h1_artifact = torch.from_numpy(np.asarray(art["layer_embeddings"][1]))
    max_abs_diff = (h1_cached - h1_artifact).abs().max().item()

    msg = captured["msg"]                      # (E_train, 64), aligned with edge_index columns
    dst = edge_index[1]                         # destination node per edge (aggregation target)
    msg_norm = msg.norm(dim=1)                  # (E_train,)

    sum_msg = torch.zeros(num_nodes, 64)
    sum_msg.index_add_(0, dst, msg)
    sum_norm = torch.zeros(num_nodes)
    sum_norm.index_add_(0, dst, msg_norm)

    degree = torch.zeros(num_nodes)
    degree.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))

    contribution_share = (msg_norm / sum_norm[dst].clamp(min=1e-12)).numpy()

    eps = conv1.eps.item()
    pre_nn_full = (1 + eps) * x + sum_msg
    pre_nn_without = pre_nn_full[dst] - msg     # leave-one-out, batched over all edges
    with torch.no_grad():
        h1_without = model.convs[0].nn(pre_nn_without)
    leave_one_out_delta = (h1_cached[dst] - h1_without).norm(dim=1).numpy()
    # Raw delta grows with degree for a different reason than dilution: sum
    # aggregation makes high-degree nodes' pre-MLP magnitude itself larger,
    # so removing any one term shifts the (unbounded, non-scale-invariant)
    # MLP output more in absolute terms even as each edge's RELATIVE share
    # shrinks. Normalize by the embedding's own magnitude to get a
    # degree-comparable score, same spirit as contribution_share.
    h1_norm = h1_cached[dst].norm(dim=1).clamp(min=1e-12)
    leave_one_out_delta_relative = (leave_one_out_delta / h1_norm.numpy())

    return {
        "dataset": ds_name, "model": "GINEConv",
        "n_edges": edge_index.shape[1], "n_nodes": num_nodes,
        "h1_reconstruction_max_abs_diff": max_abs_diff,
        "contribution_share": contribution_share,
        "leave_one_out_delta": leave_one_out_delta,
        "leave_one_out_delta_relative": leave_one_out_delta_relative,
        "degree": degree[dst].numpy(),
    }


# ── SignedGCN (CSG): exact O(1)-per-edge sensitivity via mean aggregation ───

def signedgcn_sensitivity(ds_name: str, art_dir: str = "seed42"):
    from torch_geometric.nn import SignedGCN

    art_path = os.path.join(ROOT, "baselines", "CSG", "results_our_splits",
                             ds_name, "CSG", art_dir, "best_epoch_artifacts.pkl")
    with open(art_path, "rb") as f:
        art = pickle.load(f)
    if "state_dict" not in art:
        print(f"  ✗ {ds_name}/CSG artifact has no state_dict (retrain with the "
              f"updated run_with_our_splits.py) -- skipping")
        return None

    splits = torch.load(os.path.join(ROOT, "baselines", "splits", f"{ds_name}.pt"),
                         weights_only=False)
    ei, ew, trn_mask = splits["edge_index"], splits["edge_weight"], splits["trn_mask"]
    num_nodes = int(ei.max().item()) + 1
    pos_ei = ei[:, trn_mask & (ew > 0)]
    neg_ei = ei[:, trn_mask & (ew < 0)]
    x = reconstruct_x(num_nodes, 64)

    model = SignedGCN(64, 64, num_layers=2, lamb=5)
    model.load_state_dict(art["state_dict"])
    model.eval()
    conv1 = model.conv1

    with torch.no_grad():
        h1_cached = conv1(x, pos_ei, neg_ei)  # (N, 64) = cat([out_pos, out_neg])

    h1_artifact = torch.from_numpy(np.asarray(art["layer1_embedding"]))
    max_abs_diff = (h1_cached - h1_artifact).abs().max().item()

    # SignedConv.message(x_j) = x_j (identity); aggr='mean' over each channel
    # separately. Leave-one-out for a mean: new_mean = (sum - x_q)/(count-1).
    def loo_mean_aggregation(channel_ei, x_src, num_nodes):
        """Returns (mean_full[dst], sum_full[dst], count[dst], src, dst) for
        every edge in channel_ei, i.e. the per-edge view of an aggr='mean'
        propagate() call -- everything needed for O(1) leave-one-out."""
        src, dst = channel_ei[0], channel_ei[1]
        count = torch.zeros(num_nodes)
        count.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
        sum_x = torch.zeros(num_nodes, x_src.shape[1])
        sum_x.index_add_(0, dst, x_src[src])
        mean_x = sum_x / count.clamp(min=1).unsqueeze(-1)
        return mean_x, sum_x, count, src, dst

    mean_pos, sum_pos, cnt_pos, src_pos, dst_pos = loo_mean_aggregation(pos_ei, x, num_nodes)
    mean_neg, sum_neg, cnt_neg, src_neg, dst_neg = loo_mean_aggregation(neg_ei, x, num_nodes)

    def full_h1(mean_pos, mean_neg, x_dst):
        out_pos = conv1.lin_pos_l(mean_pos) + conv1.lin_pos_r(x_dst)
        out_neg = conv1.lin_neg_l(mean_neg) + conv1.lin_neg_r(x_dst)
        return torch.cat([out_pos, out_neg], dim=-1)

    # leave-one-out for every positive-channel edge
    with torch.no_grad():
        cnt_pos_loo = (cnt_pos[dst_pos] - 1).clamp(min=1)
        mean_pos_loo = (sum_pos[dst_pos] - x[src_pos]) / cnt_pos_loo.unsqueeze(-1)
        h1_pos_loo = full_h1(mean_pos_loo, mean_neg[dst_pos], x[dst_pos])
        cnt_neg_loo = (cnt_neg[dst_neg] - 1).clamp(min=1)
        mean_neg_loo = (sum_neg[dst_neg] - x[src_neg]) / cnt_neg_loo.unsqueeze(-1)
        h1_neg_loo = full_h1(mean_pos[dst_neg], mean_neg_loo, x[dst_neg])

    delta_pos = (h1_cached[dst_pos] - h1_pos_loo).norm(dim=1).numpy()
    delta_neg = (h1_cached[dst_neg] - h1_neg_loo).norm(dim=1).numpy()
    leave_one_out_delta = np.concatenate([delta_pos, delta_neg])
    norm_pos = h1_cached[dst_pos].norm(dim=1).clamp(min=1e-12).numpy()
    norm_neg = h1_cached[dst_neg].norm(dim=1).clamp(min=1e-12).numpy()
    leave_one_out_delta_relative = np.concatenate([delta_pos / norm_pos, delta_neg / norm_neg])
    degree = np.concatenate([cnt_pos[dst_pos].numpy(), cnt_neg[dst_neg].numpy()])

    # contribution_share via mean (not sum): a single neighbor's share of a
    # mean is 1/count regardless of magnitude (message(x_j)=x_j, no edge_attr
    # weighting) -- report ||x_q|| / (count * ||mean||) as the magnitude-aware
    # analogue.
    share_pos = (x[src_pos].norm(dim=1) / (cnt_pos[dst_pos] * mean_pos[dst_pos].norm(dim=1)).clamp(min=1e-12)).numpy()
    share_neg = (x[src_neg].norm(dim=1) / (cnt_neg[dst_neg] * mean_neg[dst_neg].norm(dim=1)).clamp(min=1e-12)).numpy()
    contribution_share = np.concatenate([share_pos, share_neg])

    return {
        "dataset": ds_name, "model": "CSG",
        "n_edges": pos_ei.shape[1] + neg_ei.shape[1], "n_nodes": num_nodes,
        "h1_reconstruction_max_abs_diff": max_abs_diff,
        "contribution_share": contribution_share,
        "leave_one_out_delta": leave_one_out_delta,
        "leave_one_out_delta_relative": leave_one_out_delta_relative,
        "degree": degree,
    }


def summarize_by_degree_bucket(res: dict) -> list:
    buckets = quantile_buckets(res["degree"])
    rows = []
    for b in range(N_BUCKETS):
        mask = buckets == b
        if mask.sum() == 0:
            continue
        rows.append({
            "bucket": b, "n_edges": int(mask.sum()),
            "degree_min": float(res["degree"][mask].min()),
            "degree_max": float(res["degree"][mask].max()),
            "mean_share": float(res["contribution_share"][mask].mean()),
            "mean_delta": float(res["leave_one_out_delta"][mask].mean()),
            "mean_delta_relative": float(res["leave_one_out_delta_relative"][mask].mean()),
        })
    return rows


def write_report(all_results: list, out_dir: str):
    lines = [
        "=" * 96,
        "  LEAD 2 STEP 1b -- per-edge bottleneck sensitivity (degree-stratified)",
        "=" * 96,
        "",
        "contribution_share / mean_delta_relative LOW for an edge (v,q) means q's",
        "signal got swamped by v's other neighbors in the aggregate -- the literal",
        "edge-level bottleneck signature. Bottleneck dilution confirmed if both",
        "columns DECREASE monotonically as degree (bucket) increases. mean_delta",
        "(raw, unnormalized) is included for reference only -- it grows with",
        "degree for a different reason (sum-aggregation inflates the embedding's",
        "own magnitude at high degree), not directly comparable across buckets.",
        "",
    ]
    for r in all_results:
        lines += [
            f"{'─'*96}",
            f"  {r['dataset']} x {r['model']}   "
            f"(n_edges={r['n_edges']:,}, h1 reconstruction max|diff|={r['h1_reconstruction_max_abs_diff']:.2e})",
            f"{'─'*96}",
            f"  {'bucket':<8}{'n_edges':>10}{'degree range':>16}{'mean_share':>12}"
            f"{'mean_delta_rel':>16}{'mean_delta(raw)':>16}",
        ]
        for row in r["by_degree_bucket"]:
            deg_range = f"[{row['degree_min']:.0f},{row['degree_max']:.0f}]"
            lines.append(
                f"  {row['bucket']:<8}{row['n_edges']:>10,}"
                f"{deg_range:>16}"
                f"{row['mean_share']:>12.5f}{row['mean_delta_relative']:>16.5f}{row['mean_delta']:>16.5f}"
            )
        lines.append("")

    path = os.path.join(out_dir, "edge_sensitivity_report.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n✓ Report written to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--models", nargs="+", default=["GINEConv", "CSG"])
    parser.add_argument("--out", default="outputs/lead2_gnn_bottleneck")
    args = parser.parse_args()

    datasets = ALL_DATASETS if args.datasets == ["all"] else args.datasets
    out_dir = os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)

    all_results = []
    for ds in datasets:
        for model_name in args.models:
            print(f"\n── {ds} x {model_name} ──")
            try:
                if model_name == "GINEConv":
                    res = gineconv_sensitivity(ds)
                elif model_name == "CSG":
                    res = signedgcn_sensitivity(ds)
                else:
                    raise ValueError(model_name)
            except FileNotFoundError as e:
                print(f"  ✗ missing artifact ({e}) -- train it first")
                continue
            if res is None:
                continue
            res["by_degree_bucket"] = summarize_by_degree_bucket(res)
            print(f"  h1 reconstruction max|diff|={res['h1_reconstruction_max_abs_diff']:.2e}")
            for row in res["by_degree_bucket"]:
                print(f"    bucket {row['bucket']} deg[{row['degree_min']:.0f},{row['degree_max']:.0f}] "
                      f"n={row['n_edges']:,} mean_share={row['mean_share']:.5f} "
                      f"mean_delta_rel={row['mean_delta_relative']:.5f} mean_delta_raw={row['mean_delta']:.5f}")
            all_results.append(res)
            save_path = os.path.join(out_dir, f"{ds}_{model_name}_edge_sensitivity.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(res, f)

    seen = {(r["dataset"], r["model"]) for r in all_results}
    for ds in ALL_DATASETS:
        for model_name in ["GINEConv", "CSG"]:
            if (ds, model_name) in seen:
                continue
            pkl_path = os.path.join(out_dir, f"{ds}_{model_name}_edge_sensitivity.pkl")
            if os.path.exists(pkl_path):
                with open(pkl_path, "rb") as f:
                    all_results.append(pickle.load(f))

    if all_results:
        all_results.sort(key=lambda r: (ALL_DATASETS.index(r["dataset"]), r["model"]))
        write_report(all_results, out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
