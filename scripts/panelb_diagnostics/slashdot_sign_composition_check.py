import os
import sys
import numpy as np

ROOT = "/home/eng/shilo_avital/yolo_lab/walk_to_paint"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from scripts.balance_theory_paths import load_edges_canonical

DS_NAME = "slashdot090221"

rows = np.load("outputs/panelb_diagnostics/panelb_raw_rows.npy")
# columns: line_dist, anchor_sign, context_sign, context_edge_id, pair_key
print("rows shape:", rows.shape, "line_dist range:", rows[:, 0].min(), rows[:, 0].max())

edges = load_edges_canonical(DS_NAME)
nodes = sorted({n for u, v, s in edges for n in (u, v)})
n2i = {n: i for i, n in enumerate(nodes)}
N, E = len(nodes), len(edges)
edge_x = np.empty(E, np.int32)
edge_y = np.empty(E, np.int32)
edge_s = np.empty(E, np.int8)
deg = np.zeros(N, np.int32)
for eid, (u, v, s) in enumerate(edges):
    ui, vi = n2i[u], n2i[v]
    edge_x[eid], edge_y[eid], edge_s[eid] = ui, vi, s
    deg[ui] += 1
    deg[vi] += 1

baseline_pos = float((edge_s.astype(np.float64) > 0).mean())
baseline_deg = float(deg.mean())
print(f"\ndataset-wide: %positive={baseline_pos:.1%}  mean total-degree={baseline_deg:.1f} max={deg.max()}")

for d in sorted(set(rows[:, 0].tolist())):
    mask = rows[:, 0] == d
    sub = rows[mask]
    ctx_signs = sub[:, 2]
    ctx_eids = sub[:, 3]
    pos_frac_pairweighted = float((ctx_signs > 0).mean())
    uniq_eids = np.unique(ctx_eids)
    uniq_signs = edge_s[uniq_eids]
    pos_frac_distinct = float((uniq_signs > 0).mean())
    uniq_deg = deg[edge_x[uniq_eids]].astype(np.float64) + deg[edge_y[uniq_eids]].astype(np.float64)
    uniq_deg_mean = float(uniq_deg.mean() / 2)  # avg endpoint degree
    n_pairs = mask.sum()
    n_distinct = len(uniq_eids)
    top10_share = float(np.sort(np.bincount(ctx_eids.astype(np.int64)))[::-1][:10].sum() / n_pairs) if n_pairs else float("nan")
    print(f"d={d}: n_pairs={n_pairs:,} n_distinct={n_distinct:,} "
          f"top10_share={top10_share:.1%} "
          f"%pos(pairweighted)={pos_frac_pairweighted:.1%} %pos(distinct)={pos_frac_distinct:.1%} "
          f"mean_endpoint_deg(distinct)={uniq_deg_mean:.1f}")
