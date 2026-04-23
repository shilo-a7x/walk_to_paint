"""
Issues 1+6 correctness + timing test.

What this tests
---------------
val_pack and test_pack were being built inside pad_and_build_stage_tensors
via _stage_views_from_base, then immediately discarded in prepare_data.
compute_class_weights_from_train consumed only train_pack[1] (train labels),
which is a tensor that can be derived directly from the base edge_split_mask
and input_ids without cloning any tensor for stage views.

Fix applied:
  - pad_and_build_stage_tensors no longer calls _stage_views_from_base.
  - compute_class_weights_from_base reads MASK positions directly.

This test:
  A) Builds pipeline up to base tensors (input_ids, edge_split_mask).
  B) OLD: calls _stage_views_from_base to get train_pack, then
          compute_class_weights_from_train(train_pack, ...).
  C) NEW: calls compute_class_weights_from_base directly on base tensors.
  D) Asserts class weights are numerically identical.
  E) Reports wall-clock timing for both paths.
"""

import sys
import time
import torch
from omegaconf import OmegaConf

from src.utils.config import load_config, get_seed
from src.data.prepare_data import (
    get_edge_list,
    split_edges,
    get_walks,
    get_tokenizer,
    encode_walks,
    pad_and_build_stage_tensors,
    _stage_views_from_base,
    compute_class_weights_from_train,
    compute_class_weights_from_base,
    SplitID,
)
from torch.nn.utils.rnn import pad_sequence


# ── helpers ───────────────────────────────────────────────────────────────────

def make_cfg():
    cfg = load_config("config_toy.yaml")
    cfg.training.num_workers = 0
    cfg.training.persistent_workers = False
    cfg.training.pin_memory = False
    cfg.training.prefetch_factor = None
    return cfg


def build_base(cfg):
    """Run pipeline up to padded base tensors; return base_tensors + tokenizer."""
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

    input_lists, split_lists, eid_list, wid_list, pos_list, wlen_list = encode_walks(
        walks, tokenizer, edges, train_s, mask_s, val_s, test_s
    )

    # pad manually (mirrors what pad_and_build does internally)
    pad_id = int(tokenizer.PAD_ID)
    input_ids    = pad_sequence(input_lists,  batch_first=True, padding_value=pad_id).long()
    edge_split   = pad_sequence(split_lists,  batch_first=True, padding_value=SplitID.BAD).long()
    edge_ids     = pad_sequence(eid_list,     batch_first=True, padding_value=-1).long()
    walk_ids     = pad_sequence(wid_list,     batch_first=True, padding_value=-1).long()
    positions    = pad_sequence(pos_list,     batch_first=True, padding_value=-1).long()
    walk_lengths = pad_sequence(wlen_list,    batch_first=True, padding_value=-1).long()

    return tokenizer, input_ids, edge_split, edge_ids, walk_ids, positions, walk_lengths


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = make_cfg()
    print("Building pipeline up to base tensors...")
    tokenizer, input_ids, edge_split, edge_ids, walk_ids, positions, walk_lengths = build_base(cfg)

    REPS = 5

    # ── OLD: _stage_views_from_base → compute_class_weights_from_train ──────
    t_old = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        train_pack, val_pack, test_pack = _stage_views_from_base(
            input_ids, edge_split, tokenizer,
            edge_ids, walk_ids, positions, walk_lengths,
        )
        old_weights = compute_class_weights_from_train(
            train_pack, cfg.model.ignore_index, cfg.model.num_classes
        )
        t_old.append(time.perf_counter() - t0)

    # ── NEW: compute_class_weights_from_base ─────────────────────────────────
    t_new = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        new_weights = compute_class_weights_from_base(
            input_ids, edge_split, tokenizer, cfg.model.num_classes, cfg.model.ignore_index
        )
        t_new.append(time.perf_counter() - t0)

    # ── correctness ───────────────────────────────────────────────────────────
    print("\nCORRECTNESS CHECK")
    print("=" * 50)
    all_ok = True

    if len(old_weights) != len(new_weights):
        print(f"  FAIL  len mismatch: old={len(old_weights)} new={len(new_weights)}")
        all_ok = False
    else:
        for i, (ow, nw) in enumerate(zip(old_weights, new_weights)):
            match = abs(ow - nw) < 1e-9
            status = "OK  " if match else "FAIL"
            print(f"  {status}  class[{i}]: old={ow:.8f}  new={nw:.8f}  diff={abs(ow-nw):.2e}")
            if not match:
                all_ok = False

    if all_ok:
        print("\n✅ CLASS WEIGHTS IDENTICAL")
    else:
        print("\n❌ MISMATCH DETECTED — do NOT apply this change")
        sys.exit(1)

    # ── also verify pad_and_build_stage_tensors API ───────────────────────────
    print("\nAPI CHECK: pad_and_build_stage_tensors returns dict (not tuple)")
    print("=" * 50)
    base = pad_and_build_stage_tensors(
        cfg,
        # rebuild lists from cfg — just check return type
        *([None] * 7),  # placeholder — we call it via a mini-rebuild instead
        tokenizer,
    ) if False else None  # skip actual call; check via isinstance below

    # verify the real call via benchmark API
    from src.utils.config import load_config as _lc
    _cfg2 = make_cfg()
    from src.data.prepare_data import get_edge_list, split_edges, get_walks, encode_walks
    _edges = get_edge_list(_cfg2)
    _ts, _ms, _vs, _tes = split_edges(_cfg2, _edges)
    _walks = get_walks(_cfg2, _edges)
    _tok2 = get_tokenizer(_cfg2, _walks, _edges)
    _cfg2.model.vocab_size = _tok2.vocab_size; _cfg2.model.num_classes = _tok2.num_edge_tokens
    _cfg2.model.pad_id = _tok2.PAD_ID; _cfg2.model.ignore_index = _tok2.UNK_LABEL_ID
    _cfg2.model.unk_id = _tok2.UNK_ID; _cfg2.model.mask_id = _tok2.MASK_ID
    il, sl, el, wl, pl, wll = encode_walks(_walks, _tok2, _edges, _ts, _ms, _vs, _tes)
    result = pad_and_build_stage_tensors(_cfg2, il, sl, el, wl, pl, wll, _tok2)

    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    expected_keys = {"input_ids", "edge_split_mask", "attention_base", "edge_ids", "walk_ids", "positions", "walk_lengths"}
    assert set(result.keys()) == expected_keys, f"Key mismatch: {set(result.keys())}"
    print("  OK    pad_and_build_stage_tensors returns dict with correct keys")
    print("\n✅ API CHECK PASSED")

    # ── timing ────────────────────────────────────────────────────────────────
    import statistics
    m_old = statistics.mean(t_old)
    m_new = statistics.mean(t_new)
    speedup = m_old / m_new if m_new > 0 else float("inf")

    print("\nTIMING (mean of {} reps)".format(REPS))
    print("=" * 50)
    print(f"  Old path (_stage_views_from_base + compute_class_weights_from_train):  {m_old*1000:.1f} ms")
    print(f"  New path (compute_class_weights_from_base only):                        {m_new*1000:.1f} ms")
    print(f"  Speedup:  {speedup:.1f}x")
    print(f"  Time saved per run: {(m_old - m_new)*1000:.1f} ms")


if __name__ == "__main__":
    main()
