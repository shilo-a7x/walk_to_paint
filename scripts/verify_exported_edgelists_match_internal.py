"""Independent cross-check that the exported edgelists in
aaai2027/dataset_edgelists_for_review.zip are not just a lossless encoding
of the internal edge lists (export_dataset_edgelists.py already verifies
that byte-for-byte), but would reproduce OUR OWN PUBLISHED NUMBERS if an
external reviewer ran their own analysis directly on the exported file --
i.e. no artifact from node/edge relabeling, parsing convention, or the 0/1
vs -1/1 label convention.

Method: this script does NOT import anything from src/data or scripts/
balance_theory_paths.py -- it parses the plain .edgelist file exactly as an
external reviewer would (three whitespace-separated ints per line, nothing
else), then independently recomputes the line_dist=1 NMI and phi coefficient
(anchor-edge sign vs. sign of every edge sharing an endpoint) using the
IDENTICAL sampling procedure (seed=42, 20,000 anchors, np.random.default_rng
.choice over edge INDEX, matching extract_empconf_panelB_mi_decay_linegraph.py
and extract_empconf_panelB_correlation_check.py exactly) and compares the
result against the values already published in
aaai2027/figure_data/empconf_panelB_mi_decay_linegraph.csv /
empconf_panelB_correlation_check.csv.

Line-graph distance 1 only needs each edge's own two endpoints (no BFS) --
two edges sharing a node are always at line-graph distance exactly 1 -- so
this is a fast, exact, from-scratch recomputation, not an approximation.

Why this is a stronger guarantee than the export script's round-trip check:
that check proves "decode(encode(edges)) == edges" using OUR OWN decode
logic. This script proves "an independent implementation, working only from
the deliverable file with no access to our node-id mapping or loader code,
converges on the exact number we already reported" -- i.e. node relabeling
is a graph isomorphism and every graph-topology statistic (degree,
connectivity, this MI/phi curve, etc.) is invariant under it, and this is
the empirical demonstration of that claim, not just the theoretical
argument for it.
"""
import csv
import math
import sys

import numpy as np

EDGELIST_DIR = "aaai2027/external_review"
MI_CSV = "aaai2027/figure_data/empconf_panelB_mi_decay_linegraph.csv"
PHI_CSV = "aaai2027/figure_data/empconf_panelB_correlation_check.csv"
SEED = 42
MAX_ANCHORS = 20000

DATASETS = ["bitcoin-alpha", "bitcoin-otc", "epinions", "wiki-elec", "wiki-rfa", "slashdot090221"]


def parse_edgelist(path):
    """Pure external-style parser: three whitespace-separated ints per line,
    no dependency on any internal loader/module."""
    us, vs, labels = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            u, v, lab = line.split()
            us.append(int(u))
            vs.append(int(v))
            labels.append(int(lab))
    return np.array(us), np.array(vs), np.array(labels)


def mi_from_cont(c):
    n = int(c.sum())
    p = c / n
    py = p.sum(axis=1)
    ps = p.sum(axis=0)
    mi = 0.0
    for i in range(2):
        for j in range(2):
            if p[i, j] > 0 and py[i] > 0 and ps[j] > 0:
                mi += p[i, j] * math.log2(p[i, j] / (py[i] * ps[j]))
    hy = -sum(q * math.log2(q) for q in py if q > 0)
    nmi = mi / hy if hy > 1e-12 else float("nan")
    return mi, nmi, n


def phi_from_cont(c):
    n = int(c.sum())
    n00, n01 = c[0, 0], c[0, 1]
    n10, n11 = c[1, 0], c[1, 1]
    row0, row1 = n00 + n01, n10 + n11
    col0, col1 = n00 + n10, n01 + n11
    denom = math.sqrt(float(row0) * row1 * col0 * col1)
    if denom == 0:
        return float("nan")
    return (n11 * n00 - n10 * n01) / denom


def analyse(ds_name):
    path = f"{EDGELIST_DIR}/{ds_name}.edgelist"
    us, vs, labels = parse_edgelist(path)  # labels already 0/1
    E = len(us)
    N = int(max(us.max(), vs.max())) + 1

    # node -> list of incident edge ids (undirected adjacency for line-graph
    # distance purposes -- matches _process_anchor's treatment exactly)
    incident = [[] for _ in range(N)]
    for eid in range(E):
        incident[us[eid]].append(eid)
        incident[vs[eid]].append(eid)

    rng = np.random.default_rng(SEED)
    anchor_ids = rng.choice(E, min(MAX_ANCHORS, E), replace=False)

    cont = np.zeros((2, 2), dtype=np.int64)
    for aid in anchor_ids:
        y = int(labels[aid])
        u, v = int(us[aid]), int(vs[aid])
        seen = set()
        for w in (u, v):
            for eid2 in incident[w]:
                if eid2 == aid or eid2 in seen:
                    continue
                seen.add(eid2)
                s2 = int(labels[eid2])
                cont[y, s2] += 1

    mi, nmi, n = mi_from_cont(cont)
    phi = phi_from_cont(cont)
    return N, E, n, mi, nmi, phi


# the published CSVs were written by extract_empconf_panelB_*.py, which key
# rows by scripts/balance_theory_paths.py's DATASET_CONFIGS shorthand
# ("slashdot"), not the real dataset.name ("slashdot090221") used everywhere
# else in this script -- lookup-only alias, not a data discrepancy.
CSV_KEY_ALIAS = {"slashdot090221": "slashdot"}


def load_published(csv_path, ds_name, value_cols):
    key = CSV_KEY_ALIAS.get(ds_name, ds_name)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["dataset"] == key and int(row["line_dist"]) == 1:
                return {c: float(row[c]) for c in value_cols}
    return None


def main():
    print(f"{'dataset':16} {'N':>10} {'E':>10} {'n_pairs':>14} {'NMI(recomputed)':>16} "
          f"{'NMI(published)':>15} {'match':>6}  {'phi(recomputed)':>16} {'phi(published)':>15} {'match':>6}")
    all_ok = True
    for ds in DATASETS:
        N, E, n, mi, nmi, phi = analyse(ds)
        pub_mi = load_published(MI_CSV, ds, ["nmi"])
        pub_phi = load_published(PHI_CSV, ds, ["phi"])
        nmi_pub = pub_mi["nmi"] if pub_mi else float("nan")
        phi_pub = pub_phi["phi"] if pub_phi else float("nan")
        nmi_match = math.isclose(nmi, nmi_pub, rel_tol=1e-9, abs_tol=1e-12)
        phi_match = math.isclose(phi, phi_pub, rel_tol=1e-9, abs_tol=1e-12)
        all_ok &= nmi_match and phi_match
        print(f"{ds:16} {N:>10,} {E:>10,} {n:>14,} {nmi:>16.10f} {nmi_pub:>15.10f} "
              f"{'OK' if nmi_match else 'FAIL':>6}  {phi:>16.10f} {phi_pub:>15.10f} "
              f"{'OK' if phi_match else 'FAIL':>6}")

    print()
    if all_ok:
        print("ALL DATASETS MATCH EXACTLY: an independent recomputation using only the "
              "exported edgelist file (no internal loader, no node-id mapping) reproduces "
              "the already-published line_dist=1 NMI and phi values exactly.")
    else:
        print("MISMATCH DETECTED -- see FAIL rows above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
