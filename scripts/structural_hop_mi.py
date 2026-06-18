"""
Structural MI at true BFS graph-hop distance d.

Measures whether graph-structural features of nodes at BFS distance d
from a target edge's endpoints predict the target edge's sign.
This is DIFFERENT from C(k) in graph_signal_analysis.py: that measures
sign correlation at walk-step k; this measures at TRUE GRAPH HOP d.

Features computed at each hop d ∈ 1..D_MAX from a target edge (u, v):
  B1 – Sign features  (new vs existing C(k)):
       mean_sign_out_u_d  : mean sign of out-edges from d-hop out-nbrs of u
       mean_sign_in_v_d   : mean sign of in-edges  to  d-hop in-nbrs  of v
       frac_pos_out_u_d   : fraction positive out-edges at d hops from u
       frac_pos_in_v_d    : fraction positive in-edges  at d hops from v

  B2 – Structural / topological features:
       mean_out_deg_u_d   : mean out-degree of d-hop out-nbrs of u
       mean_in_deg_v_d    : mean in-degree  of d-hop in-nbrs  of v
       frac_dead_u_d      : fraction of d-hop out-nbrs of u with out_deg=0
       sign_entropy_u_d   : H(sign) of edges leaving d-hop out-nbrs of u
       signed_cc_u_d      : signed clustering coeff of d-hop out-nbrs of u
                            (fraction of closed directed triangles that are
                            balance-theory balanced, i.e. sign product = +1)
       overlap_d          : fraction overlap: |N_u_d ∩ N_v_d| / |N_u_d ∪ N_v_d|

For large graphs (epinions, slashdot) BFS neighbourhoods at d≥3 can be
enormous. We cap at MAX_NBRS nodes per hop and use reservoir sampling.

Usage
─────
  python scripts/structural_hop_mi.py [--datasets all] [--out outputs/structural_hop_mi]
"""

import os, sys, gzip, glob, pickle, argparse, math
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── Config & loaders (mirror graph_signal_stats.py) ──────────────────────────
D_MAX     = 5        # max hop distance to analyse
MAX_NBRS  = 500      # max nodes to track per hop (reservoir-sample beyond this)
MAX_EDGES = 80_000   # cap test edges for speed on large graphs
MAX_CC_NODES = 50    # nodes sampled for signed-CC per hop

DATASET_CONFIGS = {
    "bitcoin-alpha": {
        "edge_file":  "data/bitcoin-alpha/soc-sign-bitcoinalpha.csv.gz",
        "format":     "bitcoin",
        "exp_dir":    "outputs/transformer_incremental/bitcoin-alpha_seed42_nw5000000_mw80_bs1024_ep75_20260423-111407",
        "ds_name":    "bitcoin-alpha",
        "best_epoch": 24,
    },
    "bitcoin-otc": {
        "edge_file":  "data/bitcoin-otc/soc-sign-bitcoinotc.csv.gz",
        "format":     "bitcoin",
        "exp_dir":    "outputs/transformer_incremental/bitcoin-otc_seed42_nw500000_mw80_bs1024_ep50_20260419-133621",
        "ds_name":    "bitcoin-otc",
        "best_epoch": 60,
    },
    "epinions": {
        "edge_file":  "data/epinions/soc-sign-epinions.txt.gz",
        "format":     "simple",
        "exp_dir":    "outputs/transformer_incremental/epinions_seed42_nw500000_mw80_bs1024_ep50_20260416-125419",
        "ds_name":    "epinions",
        "best_epoch": 42,
    },
    "wiki-elec": {
        "edge_file":  "data/wiki-Elec/wikiElec.ElecBs3.txt.gz",
        "format":     "wiki_elec",
        "exp_dir":    "outputs/transformer_incremental/wiki-elec_seed42_nw500000_mw80_bs1024_ep50_20260419-133621",
        "ds_name":    "wiki-elec",
        "best_epoch": 36,
    },
    "wiki-rfa": {
        "edge_file":  "data/wiki-RfA/wiki-RfA.txt.gz",
        "format":     "wiki_rfa",
        "exp_dir":    "outputs/transformer_incremental/wiki-rfa_seed42_nw500000_mw80_bs1024_ep50_20260324-141920",
        "ds_name":    "wiki-rfa",
        "best_epoch": 25,
    },
    "slashdot": {
        "edge_file":  "data/slashdot090221/soc-sign-Slashdot090221.txt.gz",
        "format":     "simple",
        "exp_dir":    "outputs/transformer_incremental/slashdot090221_seed42_nw5000000_mw80_bs1024_ep50_20260420-134149",
        "ds_name":    "slashdot090221",
        "best_epoch": 20,
    },
}

# ── Edge loaders ──────────────────────────────────────────────────────────────

def _open_gz(path):
    p = os.path.join(ROOT, path)
    return gzip.open(p, "rt", encoding="utf-8", errors="replace") if p.endswith(".gz") \
           else open(p, "r", encoding="utf-8", errors="replace")


def load_edges_bitcoin(path):
    edges = []
    with _open_gz(path) as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            try:
                u, v, r = int(parts[0]), int(parts[1]), int(parts[2])
            except ValueError:
                continue
            if r > 0:
                edges.append((u, v, 1))
            elif r < 0:
                edges.append((u, v, -1))
    return edges


def load_edges_simple(path):
    edges = []
    with _open_gz(path) as fh:
        for line in fh:
            ln = line.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()
            if len(parts) < 3:
                parts = ln.split(",")
            if len(parts) < 3:
                continue
            try:
                u, v, s = int(parts[0]), int(parts[1]), int(parts[2])
            except ValueError:
                continue
            if s in (-1, 1):
                edges.append((u, v, s))
    return edges


def load_edges_wiki_elec(path):
    edges = []
    candidate_id = None
    with _open_gz(path) as fh:
        for line in fh:
            ln = line.rstrip("\n")
            if not ln.strip():
                candidate_id = None
                continue
            parts = ln.split()
            if not parts:
                continue
            if parts[0] == "U":
                try:
                    candidate_id = int(parts[1])
                except (IndexError, ValueError):
                    candidate_id = None
            elif parts[0] == "V" and candidate_id is not None:
                if len(parts) < 3:
                    continue
                try:
                    vote = int(parts[1])
                    voter_id = int(parts[2])
                except ValueError:
                    continue
                if vote in (-1, 1):
                    edges.append((voter_id, candidate_id, vote))
    return edges


def load_edges_wiki_rfa(path):
    import hashlib

    def _stable_id(s):
        h = hashlib.md5(s.encode("utf-8")).hexdigest()
        return int(h[:16], 16) % (10 ** 9)

    edges = []
    with _open_gz(path) as fh:
        block = {}
        for line in fh:
            ln = line.strip()
            if not ln:
                if block:
                    u_r = block.get("SRC")
                    v_r = block.get("TGT")
                    v_v = block.get("VOT")
                    if u_r and v_r and v_v is not None:
                        try:
                            u = int(u_r) if u_r.lstrip("-").isdigit() else _stable_id(u_r)
                            v = int(v_r) if v_r.lstrip("-").isdigit() else _stable_id(v_r)
                            vv = int(float(v_v))
                            if vv in (-1, 1):
                                edges.append((u, v, vv))
                        except (ValueError, TypeError):
                            pass
                    block = {}
                continue
            if ":" in ln:
                key, _, val = ln.partition(":")
                if key.strip().isupper():
                    block[key.strip()] = val.strip()
        if block:
            u_r = block.get("SRC")
            v_r = block.get("TGT")
            v_v = block.get("VOT")
            if u_r and v_r and v_v is not None:
                try:
                    u = int(u_r) if u_r.lstrip("-").isdigit() else _stable_id(u_r)
                    v = int(v_r) if v_r.lstrip("-").isdigit() else _stable_id(v_r)
                    vv = int(float(v_v))
                    if vv in (-1, 1):
                        edges.append((u, v, vv))
                except (ValueError, TypeError):
                    pass
    return edges


LOADERS = {
    "bitcoin":   load_edges_bitcoin,
    "simple":    load_edges_simple,
    "wiki_elec": load_edges_wiki_elec,
    "wiki_rfa":  load_edges_wiki_rfa,
}


# ── PKL loader (same pattern as graph_signal_stats.py) ───────────────────────

def load_pkl(cfg, split="test"):
    epoch   = cfg["best_epoch"]
    ds_name = cfg["ds_name"]
    exp_dir = os.path.join(ROOT, cfg["exp_dir"])
    pattern = os.path.join(
        exp_dir, "runs", ds_name, "E14_HARDNODE_L10", "checkpoints",
        f"{ds_name}_predictions", f"epoch_{epoch:03d}", f"{split}_predictions.pkl",
    )
    candidates = glob.glob(pattern)
    if not candidates:
        candidates = glob.glob(os.path.join(exp_dir, "**", f"{split}_predictions.pkl"), recursive=True)
        ep_str = f"epoch_{epoch:03d}"
        candidates = [c for c in candidates if ep_str in c]
    if not candidates:
        return None
    with open(candidates[0], "rb") as f:
        return pickle.load(f)


# ── Graph preprocessing ───────────────────────────────────────────────────────

def build_graph_dicts(edges):
    """Build adjacency and sign lookup structures."""
    out_nbrs  = defaultdict(list)   # node → [(nbr, sign)]
    in_nbrs   = defaultdict(list)   # node → [(nbr, sign)] (reversed)
    out_deg   = defaultdict(int)
    in_deg    = defaultdict(int)
    sign_dict = {}                   # (u,v) → sign

    for u, v, s in edges:
        out_nbrs[u].append((v, s))
        in_nbrs[v].append((u, s))
        out_deg[u] += 1
        in_deg[v]  += 1
        sign_dict[(u, v)] = s

    return out_nbrs, in_nbrs, out_deg, in_deg, sign_dict


# ── BFS hop neighbourhood (reservoir-sampled) ────────────────────────────────

def bfs_hop_neighbours(start_node, adj, max_hops, max_per_hop, rng):
    """Return dict: d → set of node ids at exactly hop d from start_node.

    Uses reservoir sampling when the frontier exceeds max_per_hop.
    adj: node → list of (neighbour, sign) tuples.
    """
    visited   = {start_node}
    frontier  = {start_node}
    hop_nodes = {}

    for d in range(1, max_hops + 1):
        next_frontier = set()
        for node in frontier:
            for (nbr, _s) in adj.get(node, []):
                if nbr not in visited:
                    next_frontier.add(nbr)
        if not next_frontier:
            break
        # Reservoir sample if too large
        if len(next_frontier) > max_per_hop:
            sampled = rng.choice(list(next_frontier), size=max_per_hop, replace=False)
            next_frontier = set(sampled.tolist())
        visited  |= next_frontier
        frontier  = next_frontier
        hop_nodes[d] = next_frontier

    return hop_nodes


# ── Feature extraction at hop d ───────────────────────────────────────────────

def features_at_hop(nodes, out_nbrs, in_nbrs, out_deg, in_deg, sign_dict, rng):
    """Compute structural and sign features for a set of nodes.

    Returns dict with keys:
      mean_sign_out, frac_pos_out, mean_out_deg, frac_dead, sign_entropy,
      signed_cc, n_nodes
    """
    if not nodes:
        return None

    out_signs = []
    all_out_degs = []
    dead_count = 0
    total_pos = 0
    total_neg = 0

    for n in nodes:
        nbrs = out_nbrs.get(n, [])
        d = out_deg.get(n, 0)
        all_out_degs.append(d)
        if d == 0:
            dead_count += 1
        for (_, s) in nbrs:
            out_signs.append(s)
            if s > 0:
                total_pos += 1
            else:
                total_neg += 1

    n_nodes = len(nodes)
    n_out   = len(out_signs)

    mean_sign_out  = float(np.mean(out_signs))  if out_signs  else 0.0
    frac_pos_out   = total_pos / (total_pos + total_neg) if (total_pos + total_neg) > 0 else 0.5
    mean_out_deg   = float(np.mean(all_out_degs)) if all_out_degs else 0.0
    frac_dead      = dead_count / n_nodes if n_nodes > 0 else 0.0

    # Sign entropy H(S) = -p log p - (1-p) log (1-p)
    p = frac_pos_out
    if 0 < p < 1:
        sign_entropy = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
    else:
        sign_entropy = 0.0

    # Signed clustering coefficient (sample up to MAX_CC_NODES nodes)
    cc_nodes = list(nodes)
    if len(cc_nodes) > MAX_CC_NODES:
        cc_nodes = rng.choice(cc_nodes, size=MAX_CC_NODES, replace=False).tolist()

    n_triangles   = 0
    n_balanced    = 0
    for n in cc_nodes:
        nbs = out_nbrs.get(n, [])
        for i, (u, s_nu) in enumerate(nbs):
            for j, (w, s_nw) in enumerate(nbs):
                if i >= j:
                    continue
                # check edge u→w or w→u
                if (u, w) in sign_dict:
                    s_uw = sign_dict[(u, w)]
                elif (w, u) in sign_dict:
                    s_uw = sign_dict[(w, u)]
                else:
                    continue
                n_triangles += 1
                if s_nu * s_nw * s_uw > 0:
                    n_balanced += 1

    signed_cc = n_balanced / n_triangles if n_triangles > 0 else float("nan")

    return {
        "mean_sign_out": mean_sign_out,
        "frac_pos_out":  frac_pos_out,
        "mean_out_deg":  mean_out_deg,
        "frac_dead":     frac_dead,
        "sign_entropy":  sign_entropy,
        "signed_cc":     signed_cc,
        "n_nodes":       n_nodes,
    }


# ── MI & correlation helpers ──────────────────────────────────────────────────

def _nan_mask(x):
    return ~np.isnan(x)


def pearson_r(x, y):
    m = _nan_mask(x) & _nan_mask(y)
    if m.sum() < 10:
        return float("nan")
    xm, ym = x[m], y[m].astype(float)
    if xm.std() < 1e-12 or ym.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(xm, ym)[0, 1])


def mi_bits(x, y, n_bins=5):
    """Discretised mutual information in bits.

    x: continuous feature (may contain NaN)
    y: binary label {0, 1}
    """
    m = _nan_mask(x)
    if m.sum() < 20:
        return float("nan")
    xv = x[m]
    yv = y[m].astype(int)
    # percentile-based binning
    edges = np.percentile(xv, np.linspace(0, 100, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 2:
        return 0.0
    xd = np.digitize(xv, edges[1:-1]).reshape(-1, 1)
    # mutual_info_classif expects 2D X
    mi = mutual_info_classif(xd, yv, discrete_features=True, random_state=42)[0]
    return float(mi / math.log(2))  # nats → bits


def auc_from_feature(x, y):
    """One-sided AUC of x predicting y (max of AUC and 1-AUC)."""
    m = _nan_mask(x)
    if m.sum() < 10 or len(np.unique(y[m])) < 2:
        return float("nan")
    try:
        a = roc_auc_score(y[m].astype(int), x[m])
        return max(a, 1 - a)
    except Exception:
        return float("nan")


# ── Per-dataset analysis ──────────────────────────────────────────────────────

FEATURE_NAMES_U = [
    "mean_sign_out",
    "frac_pos_out",
    "mean_out_deg",
    "frac_dead",
    "sign_entropy",
    "signed_cc",
]


def analyse_dataset(ds_name, cfg, out_dir):
    print(f"\n{'─'*60}")
    print(f"  {ds_name}")
    print(f"{'─'*60}")

    # ── Load edges & build graph ──────────────────────────────────────────────
    edges = LOADERS[cfg["format"]](cfg["edge_file"])
    print(f"  Loaded {len(edges)} edges")
    out_nbrs, in_nbrs, out_deg, in_deg, sign_dict = build_graph_dicts(edges)

    # ── Load test PKL to get test edge set + labels ───────────────────────────
    pkl = load_pkl(cfg)
    if pkl is None:
        print("  ✗ PKL not found")
        return None

    # Edge-level: unique (edge_id, target) pairs
    edge_ids_arr = pkl["edge_ids"]
    targets_arr  = pkl["targets"]
    # Build unique edge_id → label
    edge_label = {}
    for eid, tgt in zip(edge_ids_arr.tolist(), targets_arr.tolist()):
        if eid >= 0:
            edge_label[int(eid)] = int(tgt)

    # Map edge_id back to (u,v,sign).
    # edges is ordered; edge_id == index in edges list.
    edge_list = edges  # [(u,v,sign)]
    test_edges_uvs = []
    for eid, lbl in edge_label.items():
        if 0 <= eid < len(edge_list):
            u, v, _ = edge_list[eid]
            test_edges_uvs.append((u, v, lbl))

    if len(test_edges_uvs) > MAX_EDGES:
        rng_sub = np.random.default_rng(42)
        idx = rng_sub.choice(len(test_edges_uvs), MAX_EDGES, replace=False)
        test_edges_uvs = [test_edges_uvs[i] for i in idx]

    print(f"  Test edges to analyse: {len(test_edges_uvs)}")

    # ── Per-hop feature extraction ────────────────────────────────────────────
    rng = np.random.default_rng(42)

    # Containers: feature_name → hop_d → list of values; labels per edge
    feat_data = {fname: {d: [] for d in range(1, D_MAX + 1)} for fname in FEATURE_NAMES_U}
    # Additional: overlap between u-side and v-side neighbourhoods
    feat_data["overlap"] = {d: [] for d in range(1, D_MAX + 1)}
    feat_data["n_nodes_u"] = {d: [] for d in range(1, D_MAX + 1)}
    feat_data["n_nodes_v"] = {d: [] for d in range(1, D_MAX + 1)}
    labels_list = []

    for u, v, label in test_edges_uvs:
        labels_list.append(label)

        # BFS from u (out-edges)
        hop_u = bfs_hop_neighbours(u, out_nbrs, D_MAX, MAX_NBRS, rng)
        # BFS from v going backwards (in-edges)
        hop_v = bfs_hop_neighbours(v, in_nbrs, D_MAX, MAX_NBRS, rng)

        for d in range(1, D_MAX + 1):
            nodes_u = hop_u.get(d, set())
            nodes_v = hop_v.get(d, set())

            # B1 + B2 features on the u-side (out-neighbours of u)
            feat_u = features_at_hop(nodes_u, out_nbrs, in_nbrs, out_deg, in_deg,
                                     sign_dict, rng) if nodes_u else None

            for fname in FEATURE_NAMES_U:
                val = feat_u[fname] if feat_u is not None else float("nan")
                feat_data[fname][d].append(val)

            feat_data["n_nodes_u"][d].append(len(nodes_u))
            feat_data["n_nodes_v"][d].append(len(nodes_v))

            # Overlap between u-out and v-in neighbourhoods at this hop
            if nodes_u and nodes_v:
                inter = len(nodes_u & nodes_v)
                union = len(nodes_u | nodes_v)
                feat_data["overlap"][d].append(inter / union if union > 0 else 0.0)
            else:
                feat_data["overlap"][d].append(float("nan"))

    labels_np = np.array(labels_list, dtype=float)
    # Convert label: assume label ∈ {-1, 1} → {0, 1}
    if labels_np.min() < 0:
        labels_bin = ((labels_np + 1) / 2).astype(int)
    else:
        labels_bin = labels_np.astype(int)

    # ── Compute MI, correlation, AUC per feature per hop d ────────────────────
    all_features = list(FEATURE_NAMES_U) + ["overlap"]
    results = {}  # (feature, d) → {mi, r, auc}

    for fname in all_features:
        results[fname] = {}
        for d in range(1, D_MAX + 1):
            x = np.array(feat_data[fname][d], dtype=float)
            results[fname][d] = {
                "mi":  mi_bits(x, labels_bin),
                "r":   pearson_r(x, labels_bin.astype(float)),
                "auc": auc_from_feature(x, labels_bin),
            }

    # ── Plots ─────────────────────────────────────────────────────────────────
    hops = list(range(1, D_MAX + 1))
    n_feat = len(all_features)
    fig, axes = plt.subplots(3, n_feat, figsize=(4 * n_feat, 9))
    if n_feat == 1:
        axes = axes.reshape(3, 1)

    metric_names = ["mi", "r", "auc"]
    metric_labels = ["MI (bits)", "Pearson r", "AUC"]

    for fi, fname in enumerate(all_features):
        for mi_idx, mname in enumerate(metric_names):
            ax = axes[mi_idx, fi]
            vals = [results[fname][d][mname] for d in hops]
            valid = [(h, v) for h, v in zip(hops, vals) if not (isinstance(v, float) and math.isnan(v))]
            if valid:
                hx, vy = zip(*valid)
                ax.plot(hx, vy, marker="o", linewidth=2)
            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
            ax.set_xlabel("BFS hop d")
            if fi == 0:
                ax.set_ylabel(metric_labels[mi_idx])
            if mi_idx == 0:
                ax.set_title(fname.replace("_", "\n"), fontsize=9)
            ax.set_xticks(hops)

    fig.suptitle(f"{ds_name}: Structural features at BFS hop d vs target sign", fontsize=12)
    fig.tight_layout()
    save_path = os.path.join(out_dir, f"shmi_{ds_name}_features.png")
    fig.savefig(save_path, dpi=110)
    plt.close(fig)
    print(f"  ✓ Saved {os.path.basename(save_path)}")

    # ── Summary: MI vs hop (one line per feature) ─────────────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    sign_feats  = ["mean_sign_out", "frac_pos_out"]
    struct_feats = ["mean_out_deg", "frac_dead", "sign_entropy", "signed_cc"]
    other_feats = ["overlap"]
    colors_s  = plt.cm.Blues(np.linspace(0.5, 0.9, len(sign_feats)))
    colors_st = plt.cm.Oranges(np.linspace(0.5, 0.9, len(struct_feats)))
    colors_o  = ["green"]

    for ax_idx, (feat_group, colors) in enumerate([
        (sign_feats, colors_s), (struct_feats, colors_st), (other_feats, colors_o)
    ]):
        ax = axes2[ax_idx]
        for fi, fname in enumerate(feat_group):
            mi_vals = [results[fname][d]["mi"] for d in hops]
            ax.plot(hops, mi_vals, marker="o", label=fname, color=colors[fi % len(colors)])
        ax.set_xlabel("BFS hop d")
        ax.set_ylabel("MI (bits)")
        titles = ["Sign features", "Structural features", "Overlap"]
        ax.set_title(titles[ax_idx])
        ax.legend(fontsize=7)
        ax.set_xticks(hops)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    fig2.suptitle(f"{ds_name}: MI with target sign at each BFS hop d", fontsize=12)
    fig2.tight_layout()
    save_path2 = os.path.join(out_dir, f"shmi_{ds_name}_mi_summary.png")
    fig2.savefig(save_path2, dpi=110)
    plt.close(fig2)
    print(f"  ✓ Saved {os.path.basename(save_path2)}")

    return results


# ── Report ────────────────────────────────────────────────────────────────────

def write_report(all_results, out_dir):
    lines = [
        "=" * 72,
        "  STRUCTURAL HOP-MI REPORT",
        "=" * 72,
        "",
        "MI (bits) between target edge sign and structural features",
        "at TRUE BFS graph-hop distance d from the target edge endpoints.",
        "",
        "This is DIFFERENT from C(k) in graph_signal_analysis.py:",
        "  C(k) = sign correlation at WALK step k (confounded by cycling)",
        "  This   = MI at GRAPH hop d (unconfounded by walk sampling)",
        "",
        "Key: if MI at d≥3 is significant → long-range graph structure",
        "     predicts target sign → deep GNN / far-walking model can help.",
        "     If MI drops to 0 at d=2 → local model is sufficient.",
        "",
        "Features:",
        "  B1 sign: mean_sign_out, frac_pos_out",
        "  B2 struct: mean_out_deg, frac_dead, sign_entropy,",
        "             signed_cc (fraction balanced triangles)",
        "  B2+: overlap (fraction of u-side and v-side d-hop sets that coincide)",
        "",
    ]

    for ds_name, results in all_results.items():
        lines += [
            f"{'─'*72}",
            f"  DATASET: {ds_name}",
            f"{'─'*72}",
        ]
        all_features = list(FEATURE_NAMES_U) + ["overlap"]
        header = f"  {'feature':<20}" + "".join(f"  d={d}" for d in range(1, D_MAX + 1))
        lines.append(header + "  (MI in bits)")
        for fname in all_features:
            row = f"  {fname:<20}"
            for d in range(1, D_MAX + 1):
                mi = results[fname][d]["mi"]
                if isinstance(mi, float) and math.isnan(mi):
                    row += "   nan"
                else:
                    row += f"  {mi:.4f}"
            lines.append(row)
        lines.append("")
        lines.append("  Pearson r:")
        for fname in all_features:
            row = f"  {fname:<20}"
            for d in range(1, D_MAX + 1):
                r = results[fname][d]["r"]
                if isinstance(r, float) and math.isnan(r):
                    row += "    nan"
                else:
                    row += f"  {r:+.3f}"
            lines.append(row)
        lines.append("")

    path = os.path.join(out_dir, "structural_hop_mi_report.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n✓ Report written to {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--out", default="outputs/structural_hop_mi")
    args = parser.parse_args()

    if args.datasets == ["all"]:
        datasets = list(DATASET_CONFIGS.keys())
    else:
        datasets = args.datasets

    out_dir = os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}
    for ds_name in datasets:
        if ds_name not in DATASET_CONFIGS:
            print(f"Unknown dataset: {ds_name}")
            continue
        result = analyse_dataset(ds_name, DATASET_CONFIGS[ds_name], out_dir)
        if result is not None:
            all_results[ds_name] = result

    if all_results:
        write_report(all_results, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
