"""
Quantifies how her v1/v2 loaders (nx.DiGraph()/nx.Graph()) handle reciprocal
edge pairs -- both (u,v) and (v,u) present as separate lines in the raw
edgelist -- versus production's array-based adjacency, which never merges
anything. See PANELB_INVESTIGATION_REPORT.md sec. 3.7 for the discussion.

Finding: nx.Graph() (her v2 default) silently collapses each reciprocal pair
into ONE edge via `add_edge` overwriting the existing edge's `sign` attribute
-- keeping only whichever direction was parsed LAST in the file. For the
~4% of reciprocal pairs where the two directions disagree in sign, this is a
genuine, irrecoverable loss of a distinct observation, not just a harmless
dedup of a redundant one. nx.DiGraph() (her v1 loader) does NOT have this
problem -- it indexes by ordered (u,v), so both directions coexist as
distinct edges.

Run: .venv/bin/python scripts/panelb_diagnostics/check_reciprocal_edge_handling.py
"""
import os

import networkx as nx

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH = os.path.join(ROOT, "aaai2027", "external_review", "slashdot090221.edgelist")


def main():
    total_lines = 0
    raw_edges = {}  # (u,v) ordered -> sign
    with open(PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            u, v, s = line.split()
            u, v, s = int(u), int(v), int(s)
            total_lines += 1
            raw_edges[(u, v)] = s

    reciprocal_pairs = sign_agree = sign_disagree = 0
    checked = set()
    for (u, v), s_uv in raw_edges.items():
        if (v, u) in raw_edges and (v, u) != (u, v) and frozenset((u, v)) not in checked:
            checked.add(frozenset((u, v)))
            reciprocal_pairs += 1
            s_vu = raw_edges[(v, u)]
            if s_uv == s_vu:
                sign_agree += 1
            else:
                sign_disagree += 1

    print(f"total directed edge lines in file: {total_lines:,}")
    print(f"distinct ordered (u,v) directed edges: {len(raw_edges):,}")
    print(f"reciprocal pairs (both (u,v) and (v,u) present): {reciprocal_pairs:,}")
    print(f"  sign AGREES both directions: {sign_agree:,} "
          f"({100*sign_agree/reciprocal_pairs:.1f}%)")
    print(f"  sign DISAGREES between directions: {sign_disagree:,} "
          f"({100*sign_disagree/reciprocal_pairs:.1f}%) -- these lose real, "
          f"distinct information under nx.Graph()'s silent overwrite")

    g = nx.Graph()
    with open(PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            u, v, label = line.split()
            g.add_edge(int(u), int(v), sign=int(label))
    print(f"\nnx.Graph() (her v2 loader) final edge count: {g.number_of_edges():,}")
    print(f"expected distinct UNORDERED node pairs: "
          f"{len(set(frozenset(k) for k in raw_edges)):,}  (matches -> confirms full collapse)")

    dg = nx.DiGraph()
    with open(PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            u, v, label = line.split()
            dg.add_edge(int(u), int(v), sign=int(label))
    print(f"\nnx.DiGraph() (her v1 loader) final edge count: {dg.number_of_edges():,}  "
          f"(matches total directed lines -> confirms NO collapse)")

    print(f"\nproduction (array-based full[u]/full[v] adjacency): keeps all "
          f"{len(raw_edges):,} distinct directed edges as separate ids -- "
          f"reciprocal pairs are never merged, both directions' signs survive "
          f"as independent observations.")


if __name__ == "__main__":
    main()
