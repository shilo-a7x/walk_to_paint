"""
Lead 2 (GNN bottleneck) -- Step 4: synthetic "inverted fog" graph generator.

Builds a synthetic signed graph with the OPPOSITE MI-vs-distance profile from
every real dataset: MI(sign, d=1) ~= 0 but MI(sign, d=2) substantial. This is
a controlled stress test for the bottleneck hypothesis -- real datasets'
d>=2 signal is already so close to zero (per outputs/mi_vs_dist) that no
model, oracle or otherwise, can show a meaningful gain from better 2-hop
access there. A synthetic graph where 2-hop signal is the ONLY signal lets
us cleanly ask: does the baseline GNN specifically fail when it needs that
signal, while an oracle-GNN (Step 3) and the walk-transformer (no forced
compression) don't?

Generative procedure
---------------------
NOTE: an earlier version of this generator used "does c(u) match c(v)" (a
per-node diffused latent). That construction calibrated to ~0 MI at BOTH
d=1 and d=2 -- it's symmetric/XOR-like (sign(u,v) only reveals the
RELATIONSHIP between u and v's classes, not either node's class value in
isolation), so it doesn't propagate any node-identity information through a
chain of edges at all, regardless of distance. Fixed below by sharing an
explicit dependency between the anchor edge and a SPECIFIC one of the
destination's neighbors (its "representative neighbor") -- since that
neighbor is included in the 2-hop BFS frontier's context sum but not the
1-hop one, this creates correlation that lands specifically at d=2.

1. Take bitcoin-alpha's real topology (via load_edges_canonical) -- realistic
   degree distribution, ignore its original signs entirely.
2. Assign each node a binary latent b(v) ~ Bernoulli(0.5) i.i.d.
3. Precompute rep1(v) = one fixed representative direct neighbor of v (via
   compute_2hop_representative's same style of one-pass BFS, here truncated
   to 1 hop) -- shared by ALL edges into v, regardless of source.
4. sign(u,v) = +1 w.p. p_match if b(u)==b(rep1(v)) else p_mismatch, plus
   independent label-flip noise epsilon. Critically this depends on
   rep1(v), NOT v's own class b(v) -- so it does NOT correlate with v's own
   out-edges (which depend on rep1(w) for w=their own destinations, unrelated
   to rep1(v)) at d=1. But rep1(v) is itself one of v's neighbors, hence
   INSIDE the d=2 BFS frontier from u -- so when the d=2 frontier includes
   rep1(v), that frontier node's own out-edges (which depend on
   b(rep1(v)) vs b(rep1(rep1(v)))) DO share the b(rep1(v)) term with the
   anchor, landing the correlation specifically at d=2.
5. Write the SAME (u,v,sign) triples into both formats needed downstream:
   - data/synthetic-fog/synthetic_fog.csv (walk-transformer's load_bitcoin
     format: source,target,rating)
   - baselines/splits/synthetic-fog.pt (GNN baselines' format: edge_index,
     edge_weight, trn/val/tst masks, dense 0-indexed node ids)
   ...and asserts they agree on a sampled subset before proceeding (avoid
   the kind of silent cross-format mismatch lead1_degree_gap.py had to
   discover the hard way).

Calibration (built into this script, run automatically by main()):
NOTE this deliberately does NOT use edge_sign_mi_vs_distance_v3.py's pooled
BFS-frontier MI -- that script answers a different question (Lead 1's
node-personality homophily vs. distance: does u's own aggregate out-edge
sign-balance correlate with the *pooled sum* of that same personality scalar
across every node in the d-hop frontier). Our signal is a dependency on one
specific node, rep1(v), buried inside a frontier of dozens-to-hundreds of
unrelated nodes -- pooling dilutes it to numerical zero even though it's
genuinely there. Instead, `calibrate()` below directly tests the two
dependencies actually engineered, with exact discrete contingency-table MI
(everything is binary, so no PCA/binning needed -- reuses `mi_from_joint`
from node_mi_structural_embedding.py):
  1. MI(sign(u,v), majority-sign of v's own out-edges) -- the literal d=1
     analog; should be ~0.
  2. I(sign(u,v); b(rep1(v)) | b(u)) -- CONDITIONAL MI, not marginal. The
     match rule b(u)==b(rep1(v)) is an equality test between two
     independent uniform bits, exactly like Attempt 1's failed
     match(c(u),c(v)) -- an equality test's outcome is marginally
     independent of either input alone (P(match)=0.5 regardless of
     b(rep1(v))'s value, since b(u) is uniform), so the *unconditional*
     MI(sign, b(rep1(v))) is ~0 by construction, not a sign of weak
     signal. Conditioning on b(u) (always available -- it's the anchor
     node's own 0-hop feature) is what exposes the dependency: given
     b(u), sign becomes p_match/p_mismatch-determined by b(rep1(v)) alone.
If these don't land where expected, adjust p_match/p_mismatch/epsilon and
regenerate.

Usage
-----
  python scripts/generate_synthetic_fog_graph.py
  python scripts/generate_synthetic_fog_graph.py --p-match 0.9 --p-mismatch 0.5 \
      --k-seeds 40 --diffusion-steps 2 --epsilon 0.05
"""
import os, sys, argparse, gzip
import numpy as np
import torch
from scipy.sparse import csr_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.balance_theory_paths import load_edges_canonical
from scripts.node_mi_structural_embedding import mi_from_joint

OUT_CSV = os.path.join(ROOT, "data", "synthetic-fog", "synthetic_fog.csv")
OUT_SPLITS = os.path.join(ROOT, "baselines", "splits", "synthetic-fog.pt")
OUT_CALIBRATION = os.path.join(ROOT, "outputs", "lead2_gnn_bottleneck",
                                "synthetic_fog_calibration.txt")


def representative_neighbor(adj: csr_matrix, num_nodes: int,
                              rng: np.random.Generator) -> np.ndarray:
    """rep1[v] = one fixed, randomly-chosen direct neighbor of v (over the
    symmetrized topology), or -1 if v is isolated. Shared by ALL edges into
    v regardless of source -- this is what every edge (u,v)'s sign depends
    on (via b(rep1(v))), not v's own class."""
    indptr, indices = adj.indptr, adj.indices
    rep1 = np.full(num_nodes, -1, dtype=np.int64)
    for v in range(num_nodes):
        nbrs = indices[indptr[v]:indptr[v + 1]]
        if len(nbrs) > 0:
            rep1[v] = nbrs[rng.integers(len(nbrs))]
    return rep1


def generate(p_match: float, p_mismatch: float, epsilon: float, seed: int = 42):
    rng = np.random.default_rng(seed)

    raw_edges = load_edges_canonical("bitcoin-alpha")
    # Canonicalize to ONE undirected edge per node pair (smaller raw id
    # first) -- the topology has some pairs in both directions (e.g. u rated
    # v AND v rated u as distinct real edges); treating those as two
    # independent draws would let symmetrization later create the same
    # logical edge twice with two different, contradictory signs.
    seen = set()
    pairs = []
    for u, v, _ in raw_edges:
        key = (u, v) if u <= v else (v, u)
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    pairs = sorted(pairs)
    all_raw_ids = sorted({n for u, v in pairs for n in (u, v)})
    dense = {raw: i for i, raw in enumerate(all_raw_ids)}
    num_nodes = len(all_raw_ids)

    # Randomize each pair's direction -- canonicalizing as (smaller,larger)
    # would make the directed graph a DAG ordered by node id (the
    # largest-id node would have zero out-edges), badly skewing the
    # calibration BFS. Real directed graphs aren't ordered by id.
    pairs_arr = np.array(pairs, dtype=np.int64)
    swap = rng.random(len(pairs_arr)) < 0.5
    raw_u = np.where(swap, pairs_arr[:, 1], pairs_arr[:, 0])
    raw_v = np.where(swap, pairs_arr[:, 0], pairs_arr[:, 1])
    src = np.array([dense[u] for u in raw_u], dtype=np.int64)
    dst = np.array([dense[v] for v in raw_v], dtype=np.int64)

    # symmetrized adjacency, used only to compute each node's representative
    # neighbor over the undirected topology (not edge direction)
    sym_src = np.concatenate([src, dst])
    sym_dst = np.concatenate([dst, src])
    adj = csr_matrix((np.ones(len(sym_src), dtype=bool), (sym_src, sym_dst)),
                      shape=(num_nodes, num_nodes))

    b = (rng.random(num_nodes) < 0.5)
    rep1 = representative_neighbor(adj, num_nodes, rng)
    print(f"[INFO] {int((rep1 >= 0).sum())}/{num_nodes} nodes have a representative neighbor")

    # sign(u,v) depends on b(u) vs b(rep1(v)) -- NOT on b(v) itself, so it
    # does not correlate with v's own out-edges (d=1); rep1(v) IS one of
    # v's neighbors though, so it IS inside the d=2 BFS frontier from u.
    has_rep = rep1[dst] >= 0
    key_val = np.where(has_rep, b[np.where(has_rep, rep1[dst], 0)], rng.random(len(dst)) < 0.5)
    match = b[src] == key_val
    p_pos = np.where(match, p_match, p_mismatch)
    base_sign = np.where(rng.random(len(src)) < p_pos, 1, -1)
    flip = rng.random(len(src)) < epsilon
    sign = np.where(flip, -base_sign, base_sign)

    print(f"[INFO] {len(pairs):,} edges, {num_nodes:,} nodes, "
          f"match-fraction={match.mean():.3f}, sign balance={np.mean(sign > 0):.3f}")

    return raw_u, raw_v, sign, src, dst, num_nodes, b, rep1


def write_csv(raw_u, raw_v, sign):
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w") as f:
        for u, v, s in zip(raw_u, raw_v, sign):
            f.write(f"{u},{v},{int(s)}\n")
    print(f"[INFO] wrote {OUT_CSV} ({len(raw_u):,} rows)")


def write_splits(src, dst, sign, num_nodes, seed: int = 42):
    """80/10/10 train/val/test split, symmetrized to bidirectional rows
    (both (u,v) and (v,u), same sign) matching baselines/splits/*.pt's own
    convention for the existing real datasets."""
    rng = np.random.default_rng(seed)
    n = len(src)
    perm = rng.permutation(n)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    split_id = np.empty(n, dtype=np.int64)
    split_id[perm[:n_train]] = 0
    split_id[perm[n_train:n_train + n_val]] = 1
    split_id[perm[n_train + n_val:]] = 2

    full_src = np.concatenate([src, dst])
    full_dst = np.concatenate([dst, src])
    full_sign = np.concatenate([sign, sign])
    full_split = np.concatenate([split_id, split_id])

    edge_index = torch.from_numpy(np.stack([full_src, full_dst])).long()
    edge_weight = torch.from_numpy(full_sign).float()
    trn_mask = torch.from_numpy(full_split == 0)
    val_mask = torch.from_numpy(full_split == 1)
    tst_mask = torch.from_numpy(full_split == 2)

    os.makedirs(os.path.dirname(OUT_SPLITS), exist_ok=True)
    torch.save({
        "edge_index": edge_index, "edge_weight": edge_weight, "num_nodes": num_nodes,
        "trn_mask": trn_mask, "val_mask": val_mask, "tst_mask": tst_mask,
        "uni_trn_mask": trn_mask[:n], "uni_val_mask": val_mask[:n], "uni_tst_mask": tst_mask[:n],
    }, OUT_SPLITS)
    print(f"[INFO] wrote {OUT_SPLITS} ({edge_index.shape[1]:,} bidirectional rows, "
          f"{num_nodes:,} nodes)")
    return edge_index, edge_weight


def _joint_counts_binary(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """2x2 joint histogram for two boolean (or {0,1}-valued) arrays."""
    J = np.zeros((2, 2))
    np.add.at(J, (a.astype(np.int64), b.astype(np.int64)), 1)
    return J


def calibrate(src, dst, sign, b, rep1, num_nodes):
    """Exact discrete-MI check of the two dependencies actually engineered
    into sign(u,v) -- see module docstring for why this replaces
    edge_sign_mi_vs_distance_v3.py's pooled-frontier MI for this purpose."""
    sign_bin = (sign > 0)

    # 1. d=1 analog: sign(u,v) vs v's own out-edge majority sign. Nodes with
    # no out-edges or an exact tie have no defined majority -- excluded.
    sum_sign = np.zeros(num_nodes)
    np.add.at(sum_sign, src, sign)
    v_majority = np.sign(sum_sign)[dst]
    has_majority = v_majority != 0
    mi_d1 = mi_from_joint(_joint_counts_binary(
        sign_bin[has_majority], v_majority[has_majority] > 0))

    # 2. the literal engineered dependency: sign(u,v) vs b(rep1(v)),
    # CONDITIONED on b(u) -- see module docstring for why this must be
    # conditional MI, not marginal (match rule is an equality test between
    # two independent uniform bits; unconditionally it's ~0 by construction).
    has_rep = rep1[dst] >= 0
    s2, u2, r2 = sign_bin[has_rep], b[src[has_rep]], b[rep1[dst][has_rep]]
    n = len(s2)
    mi_d2_marginal = mi_from_joint(_joint_counts_binary(s2, r2))
    mi_d2 = 0.0
    for u_val in (False, True):
        m = u2 == u_val
        if m.sum() == 0:
            continue
        mi_d2 += (m.sum() / n) * mi_from_joint(_joint_counts_binary(s2[m], r2[m]))

    lines = [
        "Lead 2 Step 4 -- synthetic-fog targeted calibration (exact discrete MI)",
        "(see generate_synthetic_fog_graph.py module docstring for why this",
        " replaces edge_sign_mi_vs_distance_v3.py's pooled-frontier MI here)",
        "",
        f"MI(sign(u,v), majority-sign of v's own out-edges) [d=1 analog]      = {mi_d1:.6f} bits  (want ~0)",
        f"MI(sign(u,v), b(rep1(v)))  unconditional [expected ~0, see docstring] = {mi_d2_marginal:.6f} bits",
        f"I(sign(u,v); b(rep1(v)) | b(u))  [engineered d=2 dependency]        = {mi_d2:.6f} bits  (want substantial)",
    ]
    report = "\n".join(lines)
    print("\n" + report)
    os.makedirs(os.path.dirname(OUT_CALIBRATION), exist_ok=True)
    with open(OUT_CALIBRATION, "w") as f:
        f.write(report + "\n")
    print(f"\n[INFO] wrote {OUT_CALIBRATION}")
    return mi_d1, mi_d2


def assert_formats_agree(raw_u, raw_v, sign, edge_index, edge_weight, n_check: int = 500):
    """Cross-check the CSV (raw ids) and the .pt splits (dense ids) encode
    the same sign for the same logical edge, on a sampled subset --
    catching a node-id-space mismatch immediately rather than downstream."""
    all_raw_ids = sorted(set(raw_u.tolist()) | set(raw_v.tolist()))
    dense = {raw: i for i, raw in enumerate(all_raw_ids)}
    rng = np.random.default_rng(0)
    idx = rng.choice(len(raw_u), size=min(n_check, len(raw_u)), replace=False)

    ei0, ei1 = edge_index[0].numpy(), edge_index[1].numpy()
    ew = edge_weight.numpy()
    lookup = {}
    for k in range(len(ei0)):
        lookup[(int(ei0[k]), int(ei1[k]))] = ew[k]

    for i in idx:
        du, dv = dense[int(raw_u[i])], dense[int(raw_v[i])]
        assert (du, dv) in lookup, f"edge ({du},{dv}) missing from splits format"
        assert lookup[(du, dv)] == sign[i], (
            f"sign mismatch for edge ({raw_u[i]},{raw_v[i]}): "
            f"csv={sign[i]} splits={lookup[(du, dv)]}")
    print(f"[INFO] ✓ cross-format check passed on {len(idx)} sampled edges")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p-match", type=float, default=0.95)
    parser.add_argument("--p-mismatch", type=float, default=0.05)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_u, raw_v, sign, src, dst, num_nodes, b, rep1 = generate(
        args.p_match, args.p_mismatch, args.epsilon, args.seed)
    write_csv(raw_u, raw_v, sign)
    edge_index, edge_weight = write_splits(src, dst, sign, num_nodes, args.seed)
    assert_formats_agree(raw_u, raw_v, sign, edge_index, edge_weight)
    mi_d1, mi_d2 = calibrate(src, dst, sign, b, rep1, num_nodes)
    print("\nDone.", "Calibration looks correct" if (mi_d1 < 0.01 and mi_d2 > 0.05)
          else "Calibration off-target -- adjust p_match/p_mismatch/epsilon and regenerate.")


if __name__ == "__main__":
    main()
