"""Roadmap item 4: build the final candidate hardness_map.pt (E22), picking per-dataset
between two already-screened zero-training candidates:
  - structural (src_out only): sign-ratio extremity of outgoing edges, 1 - |2p-1|
  - entropy (src_out + tgt_in): mean of binary Shannon entropy H_out(n), H_in(n),
    BOTH-defined nodes only
Both naturally bounded in [0,1] -- no percentile-rank post-processing needed (an earlier
version pre-ranked before correlating, which breaks Spearman's internal tie-handling for
values with many exact ties -- e.g. many nodes sit at extremity=0 -- and biased the
structural candidate's apparent score down; fixed by correlating raw values throughout).
Both computed from TRAIN+MASK edges only (leakage-safe -- test signs never touched).
No training -- pure graph statistics, verified against real checkpoint predictions.
"""
import sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import torch
import numpy as np
from scipy.stats import spearmanr
from scripts.balance_theory_paths import load_edges_canonical
from scripts.hardness_entropy_screen import node_entropy_train_mask, ATTN_VARIANTS, find_run_dir, build_node_error
from scripts.hardness_predictive_validity import DATASETS, load_test_predictions

OUT_TAG = "E22_HARDNODE_ENTROPY"


def structural_out_only(train_mask_edges, n_nodes):
    """Raw sign-ratio extremity of outgoing edges only, in [0,1]. NaN if no out-edges."""
    out_pos = np.zeros(n_nodes, dtype=np.float64)
    out_tot = np.zeros(n_nodes, dtype=np.float64)
    for u, v, label in train_mask_edges:
        out_tot[u] += 1.0
        if label == 1:
            out_pos[u] += 1.0
    p = np.full(n_nodes, np.nan)
    has = out_tot > 0
    p[has] = out_pos[has] / out_tot[has]
    extremity = np.full(n_nodes, np.nan)
    extremity[has] = 1.0 - np.abs(2.0 * p[has] - 1.0)
    return extremity


def correlate_cand(cand_by_raw_node, token2id, ds, edges):
    """Correlate a raw-node-indexed candidate array against real per-node test error,
    averaged across LocalAttn4 and full attention. cand values must be raw (not
    pre-ranked) -- spearmanr handles tie-averaging correctly on its own."""
    rhos = []
    for attn_name, pattern in ATTN_VARIANTS.items():
        run_dir = find_run_dir(ds, pattern)
        if run_dir is None:
            continue
        pred = load_test_predictions(run_dir, ds)
        if pred is None:
            continue
        node_error, mask_stable, _ = build_node_error(pred, edges)
        raw_nodes = np.nonzero(mask_stable)[0]
        errs, hs = [], []
        for n in raw_nodes:
            if n >= len(cand_by_raw_node) or not np.isfinite(cand_by_raw_node[n]):
                continue
            errs.append(node_error[n])
            hs.append(cand_by_raw_node[n])
        if len(errs) < 10:
            continue
        rho, _ = spearmanr(errs, hs)
        rhos.append(rho)
    return float(np.mean(rhos)) if rhos else float("nan")


def to_tensor(cand, n_nodes, token2id, vocab_size):
    t = torch.zeros(vocab_size, dtype=torch.float32)
    n_set = 0
    for n in range(n_nodes):
        if not np.isfinite(cand[n]):
            continue
        tok = token2id.get(f"N_{n}")
        if tok is None or tok >= vocab_size:
            continue
        t[tok] = float(cand[n])
        n_set += 1
    return t, n_set


def main():
    print(f"{'dataset':<16}{'structural':>12}{'entropy':>12}{'winner':>12}{'n_scored':>10}")
    for ds, (data_dir_name, cache_file) in DATASETS.items():
        edges = load_edges_canonical(ds)
        n_nodes = max(max(u, v) for u, v, _ in edges) + 1
        cache = torch.load(f"data/{data_dir_name}/{cache_file}", map_location="cpu", weights_only=False)
        token2id = cache["tokenizer"]["token2id"]
        vocab_size = cache["metadata"]["vocab_size"]
        train_mask_edges = cache["splits"]["train"] + cache["splits"]["mask"]

        r_struct = structural_out_only(train_mask_edges, n_nodes)

        h_out, h_in = node_entropy_train_mask(train_mask_edges, n_nodes)
        both = np.isfinite(h_out) & np.isfinite(h_in)
        r_entropy = np.full(n_nodes, np.nan)
        r_entropy[both] = (h_out[both] + h_in[both]) / 2.0

        rho_struct = correlate_cand(r_struct, token2id, ds, edges)
        rho_entropy = correlate_cand(r_entropy, token2id, ds, edges)

        if rho_entropy > rho_struct:
            winner_cand, winner = r_entropy, "entropy"
        else:
            winner_cand, winner = r_struct, "structural"

        final, n_set = to_tensor(winner_cand, n_nodes, token2id, vocab_size)

        out_dir = Path(f"outputs/{ds}/{OUT_TAG}")
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(final, str(out_dir / "hardness_map.pt"))
        print(f"{ds:<16}{rho_struct:>12.4f}{rho_entropy:>12.4f}{winner:>12}{n_set:>10}")


if __name__ == "__main__":
    main()
