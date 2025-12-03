#!/usr/bin/env python3
"""Quick test for dataset postprocessing (self-loops and multiedges).

Usage:
  python scripts/test_postprocess.py --config configs/wiki-rfa.yaml
  python scripts/test_postprocess.py --config configs/epinions.yaml
"""
import argparse
from collections import Counter, defaultdict
import gzip
import os
from src.utils.config import load_config
from src.data import datasets


def read_epinions(path):
    edges = []
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()
            if len(parts) < 3:
                parts = ln.split(",")
                if len(parts) < 3:
                    continue
            try:
                u = int(parts[0])
                v = int(parts[1])
                s = int(parts[2])
            except Exception:
                continue
            edges.append((u, v, s))
    return edges


def read_wiki_rfa(path):
    # minimal replicate of loader parsing but keep raw tuples (u,v,label,dat,yea)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        blocks = []
        cur = []
        for ln in f:
            if ln.strip() == "":
                if cur:
                    blocks.append(cur)
                    cur = []
                continue
            cur.append(ln.rstrip("\n"))
        if cur:
            blocks.append(cur)

    parsed = []
    for block in blocks:
        data = {}
        last = None
        for ln in block:
            if ":" in ln and ln.split(":", 1)[0].isupper():
                k, v = ln.split(":", 1)
                data[k.strip()] = v.lstrip()
                last = k.strip()
            else:
                if last:
                    data[last] = data.get(last, "") + "\n" + ln

        src = data.get("SRC")
        tgt = data.get("TGT")
        vot = data.get("VOT")
        dat = data.get("DAT")
        yea = data.get("YEA")
        if not src or not tgt or vot is None:
            continue
        try:
            u = int(src)
        except Exception:
            u = abs(hash(src)) % (10**9)
        try:
            v = int(tgt)
        except Exception:
            v = abs(hash(tgt)) % (10**9)
        try:
            lab = int(float(vot))
        except Exception:
            vr = vot.strip().lower()
            if vr in ("support", "for", "yes", "+", "+1"):
                lab = 1
            elif vr in ("oppose", "against", "no", "-", "-1"):
                lab = -1
            else:
                lab = 0
        parsed.append((u, v, lab, dat, yea))
    return parsed


def stats_before_after(raw_edges, processed_edges):
    # raw_edges: list of tuples (u,v,label,...) where first 3 elements are u,v,label
    tot_raw = len(raw_edges)
    tot_proc = len(processed_edges)
    self_raw = sum(1 for e in raw_edges if e[0] == e[1])
    self_proc = sum(1 for e in processed_edges if e[0] == e[1])
    # multiedges raw
    pairs_raw = Counter((e[0], e[1]) for e in raw_edges)
    mult_raw = sum(1 for c in pairs_raw.values() if c > 1)
    max_mult_raw = max(pairs_raw.values()) if pairs_raw else 0
    pairs_proc = Counter((e[0], e[1]) for e in processed_edges)
    mult_proc = sum(1 for c in pairs_proc.values() if c > 1)
    max_mult_proc = max(pairs_proc.values()) if pairs_proc else 0
    print(f"Total raw edges: {tot_raw}, processed edges: {tot_proc}")
    print(f"Self-loops raw: {self_raw}, processed: {self_proc}")
    print(
        f"Unique pairs raw: {len(pairs_raw)}, multiedge pairs raw: {mult_raw}, max multiplicity raw: {max_mult_raw}"
    )
    print(
        f"Unique pairs proc: {len(pairs_proc)}, multiedge pairs proc: {mult_proc}, max multiplicity proc: {max_mult_proc}"
    )
    # show top multiplicities
    print("Top 5 multiplicities (raw):")
    for p, c in pairs_raw.most_common(5):
        print(f"  {p}: {c}")
    print("Top 5 multiplicities (proc):")
    for p, c in pairs_proc.most_common(5):
        print(f"  {p}: {c}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    name = cfg.dataset.name
    data_dir = cfg.dataset.data_dir
    edge_file = os.path.join(data_dir, cfg.dataset.edge_list_file)
    print(f"Dataset: {name}, file: {edge_file}")

    if name.startswith("wiki"):
        raw = read_wiki_rfa(edge_file)
        # normalize raw to (u,v,label,ts)
        raw_norm = [
            (
                u,
                v,
                lab,
                dat if dat is not None else (str(yea) if yea is not None else None),
            )
            for (u, v, lab, dat, yea) in raw
        ]
        # call postprocess_edges to get processed
        proc = datasets.postprocess_edges(cfg, raw_norm)
        stats_before_after(raw_norm, proc)
    else:
        raw = read_epinions(edge_file)
        raw_norm = [(u, v, lab) for (u, v, lab) in raw]
        proc = datasets.postprocess_edges(cfg, raw_norm)
        stats_before_after(raw_norm, proc)


if __name__ == "__main__":
    main()
