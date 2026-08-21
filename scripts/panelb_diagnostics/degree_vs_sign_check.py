import os, sys
import numpy as np
from scipy.stats import spearmanr

ROOT = "/home/eng/shilo_avital/yolo_lab/walk_to_paint"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from scripts.balance_theory_paths import load_edges_canonical

for ds_name, directed in [("slashdot090221", False), ("wiki-rfa", True)]:
    edges = load_edges_canonical(ds_name)
    nodes = sorted({n for u, v, s in edges for n in (u, v)})
    n2i = {n: i for i, n in enumerate(nodes)}
    N, E = len(nodes), len(edges)
    edge_x = np.array([n2i[u] for u, v, s in edges], dtype=np.int32)
    edge_y = np.array([n2i[v] for u, v, s in edges], dtype=np.int32)
    edge_s = np.array([1 if s > 0 else 0 for u, v, s in edges], dtype=np.int8)
    outdeg = np.bincount(edge_x, minlength=N)
    indeg = np.bincount(edge_y, minlength=N)
    totdeg = outdeg + indeg

    if directed:
        # edge-level: outdeg of source node vs whether this edge is positive
        deg_of_edge = outdeg[edge_x].astype(np.float64)
        label = "outdeg(source u)"
    else:
        deg_of_edge = (totdeg[edge_x] + totdeg[edge_y]).astype(np.float64) / 2
        label = "mean endpoint total-degree"

    rho, p = spearmanr(deg_of_edge, edge_s)
    print(f"{ds_name} ({'directed' if directed else 'undirected'}): "
          f"Spearman(edge's {label}, edge positive) rho={rho:.4f} p={p:.2e}  n={E:,}")

    # bucket by degree decile, report % positive per decile
    order = np.argsort(deg_of_edge)
    deciles = np.array_split(order, 10)
    print("  decile (low->high degree): % positive")
    for i, idx in enumerate(deciles):
        print(f"    d{i}: n={len(idx):,} meandeg={deg_of_edge[idx].mean():.1f} %pos={edge_s[idx].mean():.1%}")
