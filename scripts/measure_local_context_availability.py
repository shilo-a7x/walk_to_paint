"""
Measures whether a target edge (this epoch's dynamically-masked supervision target)
actually has a real TRAIN/MASK-pool neighbor edge visible as context nearby -- either
within a LocalAttn4-style +-window token radius, or anywhere in the whole walk (pass
a large --window to get the whole-walk/full-attention-equivalent number).

Replicates the exact class-balanced target sampling in
LitEdgeClassifier._sample_epoch_targets (src/model/lit_model.py) against a real
cached walk file, read-only -- no training involved.

Background: naively truncating dataset.max_walk_length to match LocalAttn4's +-2-hop
window (L=2) starves most target instances of any labeled context edge at all (a
2-edge walk rarely has slack for both a target and a neighbor). Windowed attention on
a long walk gets the same +-2-hop locality without that starvation, because the walk
still wanders through the real graph instead of being cut off. See CLAUDE.md's
"Attention variant: full vs. local" section for the numbers this produced and how
they settled the LocalAttn4-vs-short-walk-truncation question (2026-07-20).

Usage: .venv/bin/python scripts/measure_local_context_availability.py <cache.pt> [ratio] [window]
"""

import sys

import torch


def main():
    path = sys.argv[1]
    ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
    window = int(sys.argv[3]) if len(sys.argv) > 3 else 4  # tokens; 4 = +-2 graph hops

    d = torch.load(path, map_location="cpu", weights_only=False)
    enc = d["encoded"]
    offsets = enc["offsets"].long()
    sm = enc["flat_split_mask"].long()
    ii = enc["flat_input_ids"].long()
    eid = enc["flat_edge_ids"].long()
    tok = d["tokenizer"]["token2id"]
    e_neg, e_pos = tok["E_-1"], tok["E_1"]

    nw = offsets.numel() - 1
    edge_pos = eid >= 0
    pool_pos = edge_pos & ((sm == 0) | (sm == 1))  # TRAIN or MASK: the only edges ever
    # visible/attended in a train-stage walk (VAL/TEST are attention_mask=0'd entirely,
    # see StageViewDataset._getitem_ragged in src/data/stage_dataset.py)

    pool_eid = eid[pool_pos]
    pool_tok = ii[pool_pos]
    max_eid = int(eid[edge_pos].max().item())
    edge_class = torch.full((max_eid + 1,), -1, dtype=torch.long)
    edge_class[pool_eid] = torch.where(pool_tok == e_pos, 1, 0)

    unique_pool_eid = torch.unique(pool_eid)
    unique_pool_cls = edge_class[unique_pool_eid]

    gen = torch.Generator(device="cpu")
    gen.manual_seed(42)
    selected_chunks = []
    for c in range(2):
        cids = unique_pool_eid[unique_pool_cls == c]
        count = int(cids.numel())
        if count == 0:
            continue
        k = int(round(count * ratio))
        if ratio > 0.0 and k == 0:
            k = 1
        k = min(k, count)
        if k == count:
            selected_chunks.append(cids)
        elif k > 0:
            perm = torch.randperm(count, generator=gen)
            selected_chunks.append(cids[perm[:k]])
    selected = (
        torch.unique(torch.cat(selected_chunks))
        if selected_chunks
        else torch.empty(0, dtype=torch.long)
    )
    is_target_by_eid = torch.zeros(max_eid + 1, dtype=torch.bool)
    is_target_by_eid[selected] = True

    n_local_ctx = n_walk_ctx_only = n_neither = n_target_occ = n_empty_walks = 0

    for w in range(nw):
        s, e = int(offsets[w]), int(offsets[w + 1])
        length = e - s
        pp = pool_pos[s:e]
        if not pp.any():
            continue
        ids_local = eid[s:e]
        tgt = torch.zeros(length, dtype=torch.bool)
        tgt[pp] = is_target_by_eid[ids_local[pp].clamp(min=0)]
        ctx = pp & (~tgt)
        if not tgt.any():
            n_empty_walks += 1
            continue
        has_any_ctx = ctx.any().item()
        for i in tgt.nonzero(as_tuple=True)[0].tolist():
            n_target_occ += 1
            lo, hi = max(0, i - window), min(length, i + window + 1)
            if ctx[lo:hi].any().item():
                n_local_ctx += 1
            elif has_any_ctx:
                n_walk_ctx_only += 1
            else:
                n_neither += 1

    print(f"file={path.split('/')[-1]} ratio={ratio} window={window}")
    print(f"empty-target walk rate: {n_empty_walks / nw * 100:.1f}%")
    print(f"total target occurrences examined: {n_target_occ}")
    print(f"  local (+-window) context available: {n_local_ctx / n_target_occ * 100:.1f}%")
    print(
        f"  context exists elsewhere in walk but NOT in local window: "
        f"{n_walk_ctx_only / n_target_occ * 100:.1f}%"
    )
    print(f"  no context anywhere in the whole walk: {n_neither / n_target_occ * 100:.1f}%")


if __name__ == "__main__":
    main()
