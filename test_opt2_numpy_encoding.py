"""
Opt-2 correctness + timing test: numpy encoding + deferred trivials.

Changes tested
--------------
1. encode_walks now returns 3 lists (input_ids, edge_split_masks, edge_ids)
   as numpy int64 arrays instead of 6 lists of torch tensors.
   - positions / walk_ids / walk_lengths are deferred — reconstructed in
     pad_and_build_stage_tensors from the padding mask.
   - np.array() per walk is ~5x faster than torch.tensor() for small lists.

2. pad_and_build_stage_tensors uses numpy fill + torch.from_numpy instead
   of pad_sequence (6 calls), which is ~10x faster.
   - positions / walk_ids / walk_lengths are reconstructed vectorially.

This test:
  A) Runs the old code path manually (torch.tensor + pad_sequence × 6).
  B) Runs the new code path (numpy + vectorial reconstruction).
  C) Asserts every output tensor is bit-for-bit identical.
  D) Reports timing for both.
"""

import sys
import time
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf

from src.utils.config import load_config, get_seed
from src.data.prepare_data import (
    get_edge_list,
    split_edges,
    get_walks,
    get_tokenizer,
    encode_walks,
    pad_and_build_stage_tensors,
    SplitID,
)


def make_cfg():
    cfg = load_config("config_toy.yaml")
    cfg.training.num_workers = 0
    cfg.training.persistent_workers = False
    cfg.training.pin_memory = False
    cfg.training.prefetch_factor = None
    return cfg


def build_raw(cfg):
    edges = get_edge_list(cfg)
    train_s, mask_s, val_s, test_s = split_edges(cfg, edges)
    walks = get_walks(cfg, edges)
    tokenizer = get_tokenizer(cfg, walks, edges)
    cfg.model.vocab_size   = tokenizer.vocab_size
    cfg.model.num_classes  = tokenizer.num_edge_tokens
    cfg.model.pad_id       = tokenizer.PAD_ID
    cfg.model.ignore_index = tokenizer.UNK_LABEL_ID
    cfg.model.unk_id       = tokenizer.UNK_ID
    cfg.model.mask_id      = tokenizer.MASK_ID
    return tokenizer, walks, edges, train_s, mask_s, val_s, test_s


# ── OLD reference implementation ──────────────────────────────────────────────

def old_encode_walks(walks, tokenizer, edges, train_set, mask_set, val_set, test_set):
    """Original: 6 lists of torch.tensor per walk."""
    split_lookup = {}
    split_lookup.update({t: SplitID.TEST  for t in test_set})
    split_lookup.update({t: SplitID.VAL   for t in val_set})
    split_lookup.update({t: SplitID.MASK  for t in mask_set})
    split_lookup.update({t: SplitID.TRAIN for t in train_set})
    split_lookup_get = split_lookup.get
    is_edge          = tokenizer.is_edge
    parse_node       = tokenizer.parse_node
    parse_edge_label = tokenizer.parse_edge_label
    unk_id           = tokenizer.UNK_ID
    token2id_get     = tokenizer.token2id.get
    edge_to_id = {(int(u), int(v), int(l)): i for i, (u, v, l) in enumerate(edges)}
    BAD = SplitID.BAD

    inp, spl, eid, wid, pos, wlen = [], [], [], [], [], []
    for walk_idx, walk in enumerate(walks):
        L = len(walk)
        x = [0]*L; sm = [BAD]*L; ei = [-1]*L
        for i in range(L):
            t = walk[i]
            x[i] = token2id_get(t, unk_id)
            if is_edge(t):
                u = parse_node(walk[i-1]) if i > 0 else None
                v = parse_node(walk[i+1]) if i < L-1 else None
                lbl = parse_edge_label(t)
                sm[i] = split_lookup_get((u, v, lbl), BAD)
                if u is not None and v is not None and lbl is not None:
                    ei[i] = edge_to_id.get((int(u), int(v), int(lbl)), -1)
        inp.append(torch.tensor(x,  dtype=torch.long))
        spl.append(torch.tensor(sm, dtype=torch.long))
        eid.append(torch.tensor(ei, dtype=torch.long))
        wid.append(torch.tensor([walk_idx]*L, dtype=torch.long))
        pos.append(torch.tensor(list(range(L)), dtype=torch.long))
        wlen.append(torch.tensor([L]*L, dtype=torch.long))
    return inp, spl, eid, wid, pos, wlen


def old_pad(inp, spl, eid, wid, pos, wlen, pad_id):
    """Original: 6× pad_sequence."""
    input_ids       = pad_sequence(inp,  batch_first=True, padding_value=pad_id).long()
    edge_split_mask = pad_sequence(spl,  batch_first=True, padding_value=int(SplitID.BAD)).long()
    edge_ids        = pad_sequence(eid,  batch_first=True, padding_value=-1).long()
    walk_ids        = pad_sequence(wid,  batch_first=True, padding_value=-1).long()
    positions       = pad_sequence(pos,  batch_first=True, padding_value=-1).long()
    walk_lengths    = pad_sequence(wlen, batch_first=True, padding_value=-1).long()
    attention_base  = (input_ids != pad_id).long()
    return dict(input_ids=input_ids, edge_split_mask=edge_split_mask,
                attention_base=attention_base, edge_ids=edge_ids,
                walk_ids=walk_ids, positions=positions, walk_lengths=walk_lengths)


# ── helpers ────────────────────────────────────────────────────────────────────

def eq(name, a, b):
    if not torch.equal(a, b):
        diff = (a != b).sum().item()
        print(f"  FAIL  {name}: {diff} elements differ  shapes {a.shape} vs {b.shape}")
        return False
    print(f"  OK    {name}")
    return True


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = make_cfg()
    print("Building raw pipeline data...")
    tokenizer, walks, edges, train_s, mask_s, val_s, test_s = build_raw(cfg)
    REPS = 5

    # ── time OLD ──────────────────────────────────────────────────────────────
    t_old = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        il, sl, el, wl, pl, wll = old_encode_walks(walks, tokenizer, edges, train_s, mask_s, val_s, test_s)
        old_base = old_pad(il, sl, el, wl, pl, wll, int(tokenizer.PAD_ID))
        t_old.append(time.perf_counter() - t0)

    # ── time NEW ──────────────────────────────────────────────────────────────
    t_new = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        il2, sl2, el2 = encode_walks(walks, tokenizer, edges, train_s, mask_s, val_s, test_s)
        new_base = pad_and_build_stage_tensors(cfg, il2, sl2, el2, tokenizer)
        t_new.append(time.perf_counter() - t0)

    # ── correctness ───────────────────────────────────────────────────────────
    print("\nCORRECTNESS CHECK")
    print("=" * 50)
    all_ok = True
    for key in old_base:
        all_ok &= eq(key, old_base[key], new_base[key])

    if all_ok:
        print("\n✅ ALL TENSORS IDENTICAL")
    else:
        print("\n❌ MISMATCH — do NOT commit")
        sys.exit(1)

    # ── timing ────────────────────────────────────────────────────────────────
    import statistics
    m_old = statistics.mean(t_old)
    m_new = statistics.mean(t_new)
    print(f"\nTIMING (mean of {REPS} reps)")
    print("=" * 50)
    print(f"  Old (torch.tensor × 6 + pad_sequence × 6):  {m_old*1000:.1f} ms")
    print(f"  New (np.array × 3 + numpy fill + vectorial): {m_new*1000:.1f} ms")
    print(f"  Speedup: {m_old/m_new:.1f}x   saved {(m_old-m_new)*1000:.1f} ms/run")


if __name__ == "__main__":
    main()
