"""Roadmap items 1-3 (HARDNESS_MINER_ROADMAP.md): entropy-based hardness candidates.

1. Recompute Lead4c's src_out/tgt_in node entropy from TRAIN+MASK edges only (their
   original computation used the full graph incl. VAL/TEST -- fine for their post-hoc
   analysis of a frozen checkpoint, NOT fine to feed into a hardness map used during
   training, where it would leak test-edge signs).
2. Predictive-validity screen: entropy-alone vs. entropy blended with the learned
   miner's accuracy hardness vs. blended with the (redundant) structural proxy, against
   BOTH LocalAttn4 and full-attention checkpoints' real per-node test error. Zero
   training needed -- this is pure correlation against already-trained checkpoints.
3. Also reports the edge-level "ceiling" check using Lead4c's exact directional pairing
   (H_out(u) + H_in(v), one per edge) against real per-edge error -- quantifies how much
   a role-aware reweighting formula (roadmap item 7) could add on top of a symmetric
   per-node map, without committing to that formula change yet.

No training. All correlations are Spearman rho vs. real held-out TEST-split error.
"""
import sys, os, pickle, glob
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import torch
import numpy as np
from scipy.stats import spearmanr, rankdata
from scripts.balance_theory_paths import load_edges_canonical
from scripts.hardness_predictive_validity import DATASETS, load_test_predictions, MIN_OCC

ATTN_VARIANTS = {
    "localattn4": "E17_HARDNODE_KCOVER_REMINE_LOCALATTN4_*",
    "full": "E17_HARDNODE_KCOVER_REMINE_full_*",
}


def find_run_dir(ds, pattern):
    cands = sorted(glob.glob(f"outputs/{ds}/{pattern}"))
    return cands[0] if cands else None


def shannon_h(p):
    """Binary Shannon entropy in bits, 0 at p in {0,1}. p: np.ndarray, nan-safe."""
    out = np.zeros_like(p, dtype=np.float64)
    valid = ~np.isnan(p) & (p > 0) & (p < 1)
    pv = p[valid]
    out[valid] = -pv * np.log2(pv) - (1 - pv) * np.log2(1 - pv)
    out[np.isnan(p)] = np.nan
    return out


def node_entropy_train_mask(train_mask_edges, n_nodes):
    """H_out(n), H_in(n) computed from TRAIN+MASK edges only (leakage-safe)."""
    out_pos = np.zeros(n_nodes, dtype=np.float64)
    out_tot = np.zeros(n_nodes, dtype=np.float64)
    in_pos = np.zeros(n_nodes, dtype=np.float64)
    in_tot = np.zeros(n_nodes, dtype=np.float64)
    for u, v, label in train_mask_edges:
        out_tot[u] += 1.0
        in_tot[v] += 1.0
        if label == 1:
            out_pos[u] += 1.0
            in_pos[v] += 1.0
    p_out = np.full(n_nodes, np.nan)
    p_in = np.full(n_nodes, np.nan)
    has_out = out_tot > 0
    has_in = in_tot > 0
    p_out[has_out] = out_pos[has_out] / out_tot[has_out]
    p_in[has_in] = in_pos[has_in] / in_tot[has_in]
    return shannon_h(p_out), shannon_h(p_in)


def build_node_error(pred, edges, min_occ=MIN_OCC):
    # edges is a Python list of (u, v, label) tuples; converting to a numpy array once
    # and vectorized-indexing it is ~50x faster than a per-occurrence Python list
    # comprehension (measured: 3.2s -> 0.07s on epinions' 2.6M test occurrences).
    edges_arr = np.asarray(edges, dtype=np.int64)
    edge_ids = pred["edge_ids"]
    correct = pred["correct"].astype(np.float64)
    n_nodes = max(max(u, v) for u, v, _ in edges) + 1
    wrong_sum = np.zeros(n_nodes)
    total_cnt = np.zeros(n_nodes)
    valid = (edge_ids >= 0) & (edge_ids < len(edges))
    eids = edge_ids[valid].astype(np.int64)
    corr = correct[valid]
    us = edges_arr[eids, 0]
    vs = edges_arr[eids, 1]
    np.add.at(wrong_sum, us, 1.0 - corr)
    np.add.at(total_cnt, us, 1.0)
    np.add.at(wrong_sum, vs, 1.0 - corr)
    np.add.at(total_cnt, vs, 1.0)
    node_error = np.full(n_nodes, np.nan)
    mask_stable = total_cnt >= min_occ
    node_error[mask_stable] = wrong_sum[mask_stable] / total_cnt[mask_stable]
    return node_error, mask_stable, (us, vs, valid, eids, corr)


def rank_blend(*arrs):
    """Average-rank blend over the intersection of finite entries."""
    stacked = np.vstack(arrs)
    valid = np.all(np.isfinite(stacked), axis=0)
    ranks = np.full(stacked.shape[1], np.nan)
    if valid.sum() > 0:
        rs = np.vstack([rankdata(a[valid]) for a in arrs])
        ranks[valid] = rs.mean(axis=0)
    return ranks


def main():
    edge_level_rows = {}
    node_level_rows = {}

    for ds, (data_dir_name, cache_file) in DATASETS.items():
        edges = load_edges_canonical(ds)
        n_nodes = max(max(u, v) for u, v, _ in edges) + 1
        cache = torch.load(f"data/{data_dir_name}/{cache_file}", map_location="cpu", weights_only=False)
        token2id = cache["tokenizer"]["token2id"]
        train_mask_edges = cache["splits"]["train"] + cache["splits"]["mask"]
        h_out, h_in = node_entropy_train_mask(train_mask_edges, n_nodes)

        # Existing static-recipe miner accuracy hardness (E21, keyed by TOKEN id)
        vpath = f"outputs/{ds}/E21_HARDNODE_METRIC_SCREEN/hardness_map.pt.variants.pt"
        miner_acc_by_tok = torch.load(vpath, weights_only=False)["accuracy"].numpy() if os.path.exists(vpath) else None

        edge_row = {}
        node_row = {}
        for attn_name, pattern in ATTN_VARIANTS.items():
            run_dir = find_run_dir(ds, pattern)
            if run_dir is None:
                continue
            pred = load_test_predictions(run_dir, ds)
            if pred is None:
                continue
            node_error, mask_stable, (us, vs, valid, eids, corr) = build_node_error(pred, edges)
            raw_nodes = np.nonzero(mask_stable)[0]

            # --- (3) Edge-level ceiling: Lead4c's exact directional pairing ---
            edge_err = 1.0 - corr
            edge_cand = h_out[us] + h_in[vs]
            ok = np.isfinite(edge_cand)
            if ok.sum() > 10:
                rho, p = spearmanr(edge_cand[ok], edge_err[ok])
                edge_row[f"{attn_name}_src_out_plus_tgt_in"] = rho

            # --- (2) Node-level symmetric candidates (what the CURRENT pipeline could use) ---
            # rank_blend (not a raw sum with 0-fill for missing) -- a naive nan_to_num(...,0)
            # sum treats "no in-edges at all" as "perfectly consistent" (hardness 0), which is
            # wrong (undefined != easy) and was corrupting datasets with many one-sided nodes.
            node_entropy_sum = rank_blend(h_out, h_in)

            def corr_node(cand):
                errs, hs = [], []
                for n in raw_nodes:
                    if not np.isfinite(cand[n]):
                        continue
                    errs.append(node_error[n])
                    hs.append(cand[n])
                if len(errs) < 10:
                    return None
                rho, _ = spearmanr(errs, hs)
                return rho

            node_row[f"{attn_name}_entropy_out_only"] = corr_node(h_out)
            node_row[f"{attn_name}_entropy_in_only"] = corr_node(h_in)
            node_row[f"{attn_name}_entropy_out_plus_in"] = corr_node(node_entropy_sum)

            if miner_acc_by_tok is not None:
                # Map miner accuracy (token-indexed) to raw-node-indexed array for blending.
                miner_acc_by_node = np.full(n_nodes, np.nan)
                for n in range(n_nodes):
                    tok = token2id.get(f"N_{n}")
                    if tok is not None and tok < len(miner_acc_by_tok):
                        miner_acc_by_node[n] = miner_acc_by_tok[tok]
                node_row[f"{attn_name}_entropy_plus_miner"] = corr_node(
                    rank_blend(node_entropy_sum, miner_acc_by_node)
                )
                node_row[f"{attn_name}_entropy_out_plus_miner"] = corr_node(
                    rank_blend(h_out, miner_acc_by_node)
                )

        edge_level_rows[ds] = edge_row
        node_level_rows[ds] = node_row

    print("\n=== Edge-level ceiling: H_out(u) + H_in(v), Lead4c's exact pairing (TRAIN+MASK-only, leakage-fixed) ===")
    cols = ["localattn4_src_out_plus_tgt_in", "full_src_out_plus_tgt_in"]
    print(f"{'dataset':<16}" + "".join(f"{c:>28}" for c in cols))
    for ds, row in edge_level_rows.items():
        print(f"{ds:<16}" + "".join(f"{row.get(c, float('nan')):>28.4f}" for c in cols))

    print("\n=== Node-level (symmetric, usable by CURRENT pipeline without code changes) ===")
    ncols = [
        "localattn4_entropy_out_only", "localattn4_entropy_in_only", "localattn4_entropy_out_plus_in",
        "localattn4_entropy_plus_miner", "localattn4_entropy_out_plus_miner",
        "full_entropy_out_only", "full_entropy_in_only", "full_entropy_out_plus_in",
        "full_entropy_plus_miner", "full_entropy_out_plus_miner",
    ]
    print(f"{'dataset':<16}" + "".join(f"{c:>32}" for c in ncols))
    for ds, row in node_level_rows.items():
        cells = [f"{row.get(c):.4f}" if row.get(c) is not None else "NA" for c in ncols]
        print(f"{ds:<16}" + "".join(f"{c:>32}" for c in cells))

    return edge_level_rows, node_level_rows


if __name__ == "__main__":
    main()
