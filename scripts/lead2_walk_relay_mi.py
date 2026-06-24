"""
Lead 2 (GNN bottleneck) -- Step 2: walk-transformer relay-token hidden-state
MI. The walk-side analogue of lead2_gnn_bottleneck_mi.py's h_v^(1) measurement.

Token layout (alternating node/edge): pos 0=N_u0, 1=E_s1, 2=N_u1, 3=E_s2, ...
For a masked target edge at token position i (always odd -- edges sit at odd
positions), the immediately adjacent node token (i-1 or i+1) is one of the
masked edge's own two endpoints -- call it the relay R. The next edge+node
out from R (positions i+2,i+3, or i-2,i-3 on the other side) is the walk
model's analogue of "v's own out-edge" in Step 1's GNN measurement: does R's
TRANSFORMER HIDDEN STATE at its own token position retain the sign of that
next edge, the same question as Step 1, but for a model with no forced
compression bottleneck (every token stays directly attention-addressable)?

No retraining -- probes an existing checkpoint. Hidden states are captured
via a plain register_forward_hook on each nn.TransformerEncoderLayer (we want
the layer's full output, not its internal attention weights, so this is
simpler than attention_analysis.py's _sa_block-overriding subclass).

BUG FIX (user-caught): the original version trusted walk-TOKEN-position
offset (i+/-3) as a stand-in for true graph-hop distance. The R-W edge
itself is always real (the walk only steps along real edges), but random
walks can backtrack -- the walk sampler (src/data/walk_sampler.py) only
follows directed out-edges, and several of these datasets have genuine
bidirectional edge pairs (u->v AND v->u both present, confirmed during
Lead 2 Step 4's synthetic-graph work) -- so W (the node 3 positions past the
masked edge) can land back on the masked edge's OTHER endpoint `u`, or on
one of u's direct neighbors, instead of a fresh 2-hop frontier node. In that
case the measurement partly tests "does R's hidden state retain the edge we
just masked" (near-trivial -- that's the model's primary training
objective) rather than genuine 2-hop retention, inflating NMI. Fixed by
verifying true distance via the real graph adjacency
(`baselines/splits/<ds>.pt`'s `edge_index`, confirmed same dense node-id
space as the walk-transformer's tokenizer): a sample is only kept if W is
neither `u` itself nor one of u's direct (symmetrized) neighbors.

Usage
-----
  python scripts/lead2_walk_relay_mi.py --datasets bitcoin-alpha
  python scripts/lead2_walk_relay_mi.py --datasets all
"""
import os, sys, argparse, pickle
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from node_mi_structural_embedding import DATASET_CONFIGS, load_dataset_cfg  # noqa: E402
from attention_analysis import load_model_and_dataset  # noqa: E402
from mi_pca_binning_utils import mi_pca_bins  # noqa: E402

MAX_SAMPLES_DEFAULT = 20000


def build_neighbor_sets(ds_name: str) -> dict:
    """node -> set of direct (symmetrized) graph neighbors, from the real
    canonical splits.pt edge_index -- same dense node-id space as the walk-
    transformer's tokenizer (verified: both report identical num_nodes).
    Used to verify a walk-token-offset candidate is a genuinely fresh 2-hop
    frontier node, not a backtrack onto the masked edge's other endpoint or
    one of its direct neighbors."""
    splits_path = os.path.join(ROOT, "baselines", "splits", f"{ds_name}.pt")
    splits = torch.load(splits_path, map_location="cpu", weights_only=False)
    src, dst = splits["edge_index"][0].numpy(), splits["edge_index"][1].numpy()
    nbrs = {}
    for s, d in zip(src.tolist(), dst.tolist()):
        nbrs.setdefault(s, set()).add(d)
        nbrs.setdefault(d, set()).add(s)
    return nbrs


def build_token_lookup(cache_data):
    """node_id_of[token_id] -> int node id (only for N_* tokens);
    sign_of[token_id] -> int raw sign (only for E_* tokens, e.g. -1/+1)."""
    id2token = cache_data["tokenizer"]["id2token"]
    node_id_of, sign_of = {}, {}
    for tid, tok in id2token.items():
        if tok.startswith("N_"):
            try:
                node_id_of[tid] = int(tok.split("_", 1)[1])
            except ValueError:
                pass
        elif tok.startswith("E_"):
            try:
                sign_of[tid] = int(tok.split("_", 1)[1])
            except ValueError:
                pass
    return node_id_of, sign_of


def analyse_dataset(ds_name: str, cfg: dict, n_pca: int = 5, n_bins: int = 5,
                     max_samples: int = MAX_SAMPLES_DEFAULT, batch_size: int = 64,
                     device: str = "cpu"):
    bundle = load_model_and_dataset(ds_name, cfg, stage="test")
    if bundle is None:
        return None
    model = bundle["model"].to(device).eval()
    ds = bundle["dataset"]
    ignore_index = bundle["ignore_index"]
    nlayers = bundle["nlayers"]

    dscfg = load_dataset_cfg(cfg["ds_name"])
    cache_path = os.path.join(ROOT, dscfg.dataset.data_dir, "dataset_cache.pt")
    cache_data = torch.load(cache_path, map_location="cpu", weights_only=False)
    node_id_of, sign_of = build_token_lookup(cache_data)
    neighbor_sets = build_neighbor_sets(cfg["ds_name"])

    n = len(ds)
    if max_samples is not None and n > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, max_samples, replace=False).tolist()
        ds_run = torch.utils.data.Subset(ds, idx)
    else:
        ds_run = ds
    loader = torch.utils.data.DataLoader(
        ds_run, batch_size=batch_size, shuffle=False, collate_fn=bundle["collate"])

    captured = {}
    def make_hook(l):
        def hook(module, inp, out):
            captured[l] = out.detach()
        return hook
    hooks = [layer.register_forward_hook(make_hook(l))
             for l, layer in enumerate(model.transformer.layers)]

    anchors_per_layer = [[] for _ in range(nlayers)]
    contexts = []
    n_kept, n_filtered_backtrack = 0, 0

    with torch.no_grad():
        for batch in loader:
            input_ids, labels, attention_mask, metadata = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            _ = model(input_ids, attention_mask=attention_mask)

            ids_np = input_ids.cpu().numpy()
            am_np = attention_mask.cpu().numpy()
            labels_np = labels.numpy()
            S = ids_np.shape[1]

            rows, cols = np.nonzero(labels_np != ignore_index)
            for row, i in zip(rows.tolist(), cols.tolist()):
                for direction in (1, -1):
                    rp, ep, np_ = i + direction, i + 2 * direction, i + 3 * direction
                    u_pos = i - direction
                    if rp < 0 or np_ < 0 or rp >= S or np_ >= S or u_pos < 0 or u_pos >= S:
                        continue
                    if am_np[row, rp] == 0 or am_np[row, ep] == 0 or am_np[row, np_] == 0 \
                            or am_np[row, u_pos] == 0:
                        continue
                    rtok, etok = int(ids_np[row, rp]), int(ids_np[row, ep])
                    wtok, utok = int(ids_np[row, np_]), int(ids_np[row, u_pos])
                    if rtok not in node_id_of or etok not in sign_of \
                            or wtok not in node_id_of or utok not in node_id_of:
                        continue
                    # true-BFS-distance check (the Step 2 bug fix): W must be
                    # a fresh 2-hop frontier node from u, not a backtrack onto
                    # u itself or one of u's direct neighbors.
                    u_node, w_node = node_id_of[utok], node_id_of[wtok]
                    if w_node == u_node or w_node in neighbor_sets.get(u_node, ()):
                        n_filtered_backtrack += 1
                        continue
                    n_kept += 1
                    for l in range(nlayers):
                        anchors_per_layer[l].append(captured[l][row, rp, :].cpu().numpy())
                    contexts.append(sign_of[etok])

    for h in hooks:
        h.remove()

    contexts = np.asarray(contexts, dtype=np.float64)
    results_per_layer = []
    for l in range(nlayers):
        anchors = (np.stack(anchors_per_layer[l]) if anchors_per_layer[l]
                   else np.zeros((0, 1)))
        results_per_layer.append(mi_pca_bins(anchors, contexts, n_pca=n_pca, n_bins=n_bins))

    print(f"  [backtrack filter] kept {n_kept:,}, filtered {n_filtered_backtrack:,} "
          f"({100 * n_filtered_backtrack / max(1, n_kept + n_filtered_backtrack):.1f}%)")

    return {"dataset": ds_name, "n_pairs": len(contexts), "nlayers": nlayers,
            "n_kept": n_kept, "n_filtered_backtrack": n_filtered_backtrack,
            "results_per_layer": results_per_layer}


def write_report(all_results: list, out_dir: str):
    lines = [
        "=" * 88,
        "  LEAD 2 STEP 2 -- walk-transformer relay-token hidden-state MI",
        "=" * 88,
        "",
        "Same pairing as Step 1's h_v^(1) measurement (relay's own representation",
        "vs. the sign of the relay's own next-hop edge), applied to the walk-",
        "transformer's per-layer hidden state at the relay token's position.",
        "Ceiling: H(sign) ~= 1 bit. 'nmi' is already MI/H(context).",
        "",
        "n_filtered_backtrack = candidates dropped because the walk backtracked",
        "(W landed back on the masked edge's other endpoint or one of its direct",
        "neighbors, verified via the real graph adjacency) instead of reaching a",
        "genuinely fresh 2-hop frontier node -- see module docstring bug-fix note.",
        "",
    ]
    for r in all_results:
        n_kept, n_filt = r.get("n_kept", r["n_pairs"]), r.get("n_filtered_backtrack", 0)
        pct = 100 * n_filt / max(1, n_kept + n_filt)
        lines += [
            f"{'─'*88}",
            f"  {r['dataset']}   (n_pairs={r['n_pairs']:,}, nlayers={r['nlayers']}, "
            f"n_filtered_backtrack={n_filt:,} [{pct:.1f}%])",
            f"{'─'*88}",
            f"  {'layer':<8}{'mi(bits)':>12}{'nmi':>10}{'best_pc':>10}{'n_components':>14}",
        ]
        for l, res in enumerate(r["results_per_layer"]):
            lines.append(
                f"  {l:<8}{res['mi']:>12.6f}{res['nmi']:>10.4f}"
                f"{res['best_component']:>10}{res['n_components']:>14}"
            )
        lines.append("")

    path = os.path.join(out_dir, "relay_mi_report.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n✓ Report written to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--out", default="outputs/lead2_walk_relay_mi")
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES_DEFAULT)
    args = parser.parse_args()

    datasets = list(DATASET_CONFIGS.keys()) if args.datasets == ["all"] else args.datasets
    out_dir = os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)

    all_results = []
    for ds in datasets:
        if ds not in DATASET_CONFIGS:
            print(f"Unknown dataset: {ds}"); continue
        print(f"\n── {ds} ──")
        res = analyse_dataset(ds, DATASET_CONFIGS[ds], max_samples=args.max_samples)
        if res is None:
            print("  skipped (checkpoint/cache not found)")
            continue
        for l, layer_res in enumerate(res["results_per_layer"]):
            print(f"  layer {l}: mi={layer_res['mi']:.6f} bits  nmi={layer_res['nmi']:.4f}")
        all_results.append(res)
        with open(os.path.join(out_dir, f"{ds}_relay_mi.pkl"), "wb") as f:
            pickle.dump(res, f)

    seen = {r["dataset"] for r in all_results}
    for ds in DATASET_CONFIGS:
        if ds in seen:
            continue
        pkl_path = os.path.join(out_dir, f"{ds}_relay_mi.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                all_results.append(pickle.load(f))

    if all_results:
        ordered = {k: None for k in DATASET_CONFIGS}
        for r in all_results:
            ordered[r["dataset"]] = r
        write_report([v for v in ordered.values() if v is not None], out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
