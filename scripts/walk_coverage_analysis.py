"""Phase 0 walk-coverage analysis (read-only).

Reads each dataset's production CSR walk cache (data/<ds>/dataset_cache.pt) and
computes, faithfully from the exact walks the SOTA run used:

  0.B  per-split edge coverage (train/mask/val/test)
  0.C  node coverage (appears-as-token vs has-an-incident-covered-edge)
  0.D  saturation: per-edge visit-count distribution, overall + per split + per sign
       + degree-stratified coverage

No GPU, no training, writes only under outputs/walk_coverage_analysis/.

Walk token layout is strictly alternating N E N E ... so every edge token at
flat index i has its endpoints at i-1 (u) and i+1 (v), both node tokens in the
same walk (an edge token is never first/last). SplitID: TRAIN0 MASK1 VAL2 TEST3
BAD-1; node-token positions carry split_mask==-1 and edge_id==-1.
"""
import os, json, sys
import numpy as np
import torch

DATASETS = ["bitcoin-alpha", "bitcoin-otc", "epinions",
            "wiki-elec", "wiki-rfa", "slashdot090221"]
SPLIT_NAMES = {0: "train", 1: "mask", 2: "val", 3: "test"}
OUT = "outputs/walk_coverage_analysis"
DATA_DIRS = {  # data_dir casing differs from dataset.name for some
    "bitcoin-alpha": "data/bitcoin-alpha",
    "bitcoin-otc": "data/bitcoin-otc",
    "epinions": "data/epinions",
    "wiki-elec": "data/wiki-Elec",
    "wiki-rfa": "data/wiki-RfA",
    "slashdot090221": "data/slashdot090221",
}


def build_token_lookups(id2token):
    """Return arrays id2node, id2label indexed by token id (-1 where N/A)."""
    max_id = max(int(i) for i in id2token)
    id2node = np.full(max_id + 1, -1, dtype=np.int64)
    id2label = np.full(max_id + 1, -1, dtype=np.int64)
    for tid, tok in id2token.items():
        tid = int(tid)
        if tok.startswith("N_"):
            id2node[tid] = int(tok[2:])
        elif tok.startswith("E_"):
            id2label[tid] = int(tok[2:])
    return id2node, id2label


def hist_summary(counts):
    """counts = per-edge visit counts for a set of edges (>=0). Return summary."""
    counts = np.asarray(counts)
    if counts.size == 0:
        return {}
    nz = counts[counts > 0]
    return {
        "n_edges": int(counts.size),
        "n_covered": int(nz.size),
        "coverage_frac": float(nz.size / counts.size),
        "mean_visits": float(counts.mean()),
        "median_visits": float(np.median(counts)),
        "p10_visits": float(np.percentile(counts, 10)),
        "p90_visits": float(np.percentile(counts, 90)),
        "max_visits": int(counts.max()),
        "pct_occ0": float((counts == 0).mean()),
        "pct_occ_lt5": float((counts < 5).mean()),
        "pct_occ_lt1of_covered": None,
        "redundancy_ratio": float(counts.sum() / max(nz.size, 1)),
    }


def analyze(ds):
    cache = os.path.join(DATA_DIRS[ds], "dataset_cache.pt")
    d = torch.load(cache, weights_only=False)
    enc = d["encoded"]
    iid = enc["flat_input_ids"].numpy()
    sm = enc["flat_split_mask"].numpy()
    eid = enc["flat_edge_ids"].numpy()
    offsets = enc["offsets"].numpy()
    splits = d["splits"]  # {name: set of (u,v,label)}
    id2node, id2label = build_token_lookups(d["tokenizer"]["id2token"])

    n_walks = len(offsets) - 1
    em = sm != -1  # edge-token positions
    idx = np.nonzero(em)[0]
    smv = sm[idx]
    eidv = eid[idx]

    # sanity: neighbors of edge tokens are node tokens (check a sample for speed)
    chk = idx[: min(idx.size, 1_000_000)]
    assert (sm[chk - 1] == -1).all() and (sm[chk + 1] == -1).all(), "layout broken"

    # ---- nominal splits: endpoints, signs, full node/edge sets ----
    all_nodes = set()
    split_triple_sets = {}
    deg = {}
    for name, st in splits.items():
        s = set(map(tuple, st))
        split_triple_sets[name] = s
        for (a, b, c) in s:
            all_nodes.add(a); all_nodes.add(b)
            deg[a] = deg.get(a, 0) + 1
            deg[b] = deg.get(b, 0) + 1
    n_nodes = len(all_nodes)

    # ---- per-edge-id: first-occurrence (u,v,label,split) + visit count ----
    # (decode only unique edges, not all ~10^8 edge-token positions)
    uniq_eid, first = np.unique(eidv, return_index=True)        # first idx in idx-space
    flatpos = idx[first]                                        # flat positions
    eu = id2node[iid[flatpos - 1]]
    ev = id2node[iid[flatpos + 1]]
    el = id2label[iid[flatpos]]
    es = smv[first]
    visit = np.bincount(eidv)                                   # indexed by eid value
    vis_u = visit[uniq_eid]                                     # per-unique-edge visit count

    res = {"dataset": ds, "n_walks": int(n_walks),
           "n_edge_tokens": int(idx.size), "n_nodes": n_nodes,
           "n_unique_covered_edges": int(uniq_eid.size)}

    covered_triples = list(zip(eu.tolist(), ev.tolist(), el.tolist()))  # unique edges only

    # 0.B + per-sign coverage per split
    per_split = {}
    covered_all = set(covered_triples)
    for sval, sname in SPLIT_NAMES.items():
        m = es == sval
        cov = set(zip(eu[m].tolist(), ev[m].tolist(), el[m].tolist()))
        nominal = split_triple_sets.get(sname, set())
        cov_in = cov & nominal
        by_sign = {}
        for sg, sname2 in ((1, "pos"), (-1, "neg")):
            nom_s = sum(1 for t in nominal if t[2] == sg)
            cov_s = sum(1 for t in cov_in if t[2] == sg)
            by_sign[sname2] = {"nominal": nom_s, "covered": cov_s,
                               "coverage": (cov_s / nom_s) if nom_s else None}
        per_split[sname] = {
            "nominal": len(nominal), "covered": len(cov_in),
            "coverage": (len(cov_in) / len(nominal)) if nominal else None,
            "by_sign": by_sign,
        }
    res["per_split_edge_coverage"] = per_split

    # ---- 0.C node coverage ----
    start_nodes = id2node[iid[offsets[:-1]]]
    nodes_as_token = set(eu.tolist()) | set(ev.tolist()) | set(start_nodes.tolist())
    nodes_as_token.discard(-1)
    nodes_with_cov_edge = set(eu.tolist()) | set(ev.tolist())
    nodes_with_cov_edge.discard(-1)
    res["node_coverage"] = {
        "n_nodes": n_nodes,
        "frac_appear_as_token": len(nodes_as_token & all_nodes) / n_nodes if n_nodes else None,
        "frac_with_incident_covered_edge": len(nodes_with_cov_edge & all_nodes) / n_nodes if n_nodes else None,
        "n_zero_incident_covered": n_nodes - len(nodes_with_cov_edge & all_nodes),
    }

    # ---- 0.D saturation over covered edges, overall + per split ----
    res["saturation_overall_covered_edges"] = hist_summary(vis_u)
    sat_split = {}
    for sval, sname in SPLIT_NAMES.items():
        sub = vis_u[es == sval]
        if sub.size:
            sat_split[sname] = hist_summary(sub)
    res["saturation_per_split_covered"] = sat_split

    # ---- degree-stratified TEST coverage ----
    test_nom = split_triple_sets.get("test", set())
    test_cov = covered_all & test_nom
    bins = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 10 ** 9)]
    deg_strat = []
    for lo, hi in bins:
        nom = [t for t in test_nom if lo <= min(deg.get(t[0], 0), deg.get(t[1], 0)) <= hi]
        cov = [t for t in nom if t in test_cov]
        deg_strat.append({"min_deg_bin": f"{lo}-{hi if hi < 10**9 else 'inf'}",
                          "nominal": len(nom), "covered": len(cov),
                          "coverage": (len(cov) / len(nom)) if nom else None})
    res["test_coverage_by_min_degree"] = deg_strat
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    only = sys.argv[1:] or DATASETS
    allres = {}
    for ds in only:
        print(f"=== {ds} ===", flush=True)
        r = analyze(ds)
        allres[ds] = r
        tc = r["per_split_edge_coverage"]["test"]
        print(f"  walks={r['n_walks']:,} test cov={tc['coverage']:.4f} "
              f"({tc['covered']}/{tc['nominal']})  "
              f"node_token={r['node_coverage']['frac_appear_as_token']:.4f} "
              f"node_incident={r['node_coverage']['frac_with_incident_covered_edge']:.4f}",
              flush=True)
    with open(os.path.join(OUT, "coverage_phase0.json"), "w") as f:
        json.dump(allres, f, indent=2)
    print(f"\nwrote {OUT}/coverage_phase0.json")


if __name__ == "__main__":
    main()
