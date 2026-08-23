"""Transform an existing production dataset cache into the edge-identity-token variant.

Current production scheme: every edge shares one of 2 tokens (E_-1 / E_1) -- the model
can't distinguish one positive edge from another, only its sign class. This experiment
gives every edge its own identity token (like vertices already have), with sign added
separately as an additive embedding at that position (see model.py), so identity and
sign become two independent signals instead of one conflated token.

This script does a PURE POST-PROCESSING transform of an already-cached, already-sampled
walk corpus -- no resampling, no tokenizer/walk-sampler changes, so it's fully isolated
from the production data pipeline (experiments/ isolation convention, see CLAUDE.md's
"Complexity claim" section for the precedent). At every edge position (flat_edge_ids>=0):
  - flat_input_ids: replaced with a NEW per-edge token id = old_vocab_size + edge_id
    (one id per distinct edge in the graph; old E_-1/E_1 ids become unused/dead).
  - flat_sign_ids (new array): the edge's true sign class (0/1), from the *old*
    input_ids via the original tokenizer's id2class mapping -- captured before the
    input_ids overwrite above.
At node positions (flat_edge_ids == -1): flat_input_ids unchanged; flat_sign_ids = 2
("not applicable" -- a third sign_embedding class, distinct from 0/1).

Masking design (confirmed with the user 2026-08-23): at a hidden/masked edge position,
identity stays VISIBLE (input_ids keeps the edge's own new identity token) -- only sign
is hidden (sign_ids forced to class 2). This is implemented downstream in dataset.py's
_getitem_ragged override, not here -- this script only builds the static transformed
arrays; the masking-per-occurrence logic runs at __getitem__ time same as production.

Usage:
  .venv/bin/python experiments/edge_identity_tokens/build_cache.py \
      --in data/bitcoin-alpha/dataset_cache__edge_cover_nw120930_mw80_seed42.pt \
      --out experiments/edge_identity_tokens/cache/bitcoin-alpha_eid.pt
"""
import argparse
import os

import torch


def build(in_path, out_path):
    d = torch.load(in_path, map_location="cpu", weights_only=False)
    enc = d["encoded"]
    tok = d["tokenizer"]

    flat_input_ids = enc["flat_input_ids"].clone()
    flat_edge_ids = enc["flat_edge_ids"]
    is_edge = flat_edge_ids >= 0

    # True sign class per position, from the ORIGINAL token ids (before any
    # rewrite) -- id2class-equivalent lookup built the same way StageViewDataset does.
    id2class = torch.full((tok["vocab_size"],), -1, dtype=torch.long)
    for class_id, edge_tok in tok["id2edge_label"].items():
        tok_id = tok["token2id"].get(edge_tok)
        if tok_id is not None:
            id2class[tok_id] = int(class_id)
    sign_class = id2class[flat_input_ids.long()]
    assert (sign_class[is_edge] >= 0).all(), "found an edge position with no resolvable sign class"

    old_vocab_size = int(tok["vocab_size"])
    num_edges = int(flat_edge_ids[is_edge].max().item()) + 1

    # New per-edge identity token ids: old_vocab_size + edge_id. New vocab size grows
    # by num_edges (old E_-1/E_1 ids become dead/unused, left in place rather than
    # reclaimed -- simplest, avoids renumbering anything else).
    flat_input_ids_new = flat_input_ids.clone().long()
    flat_input_ids_new[is_edge] = old_vocab_size + flat_edge_ids[is_edge].long()
    new_vocab_size = old_vocab_size + num_edges

    flat_sign_ids = torch.full_like(flat_input_ids_new, 2)  # 2 = "not applicable" (node position)
    flat_sign_ids[is_edge] = sign_class[is_edge]

    print(f"old vocab_size={old_vocab_size}  num_edges={num_edges}  new vocab_size={new_vocab_size}")
    print(f"sign_ids value counts: neg={int((flat_sign_ids==0).sum())} "
          f"pos={int((flat_sign_ids==1).sum())} n/a={int((flat_sign_ids==2).sum())}")

    # edge_id -> sign lookup (num_edges,), for building a full-vocab id2class array
    # downstream (eid_src/data/stage_dataset.py) without re-deriving sign per position.
    # Sign is a fixed property of the edge, so any one occurrence's value is correct --
    # scatter is safe (every position for a given edge_id agrees).
    edge_sign_lookup = torch.full((num_edges,), -1, dtype=torch.int8)
    edge_sign_lookup[flat_edge_ids[is_edge].long()] = sign_class[is_edge].to(torch.int8)
    assert (edge_sign_lookup >= 0).all(), "found an edge with no resolvable sign in edge_sign_lookup"

    new_tok = dict(tok)
    new_tok["vocab_size"] = new_vocab_size
    new_tok["old_vocab_size"] = old_vocab_size  # first new-edge-identity-token id
    new_tok["num_edges"] = num_edges
    new_tok["edge_sign_lookup"] = edge_sign_lookup

    new_enc = dict(enc)
    new_enc["flat_input_ids"] = flat_input_ids_new.to(torch.int32)
    new_enc["flat_sign_ids"] = flat_sign_ids.to(torch.int8)

    out = dict(d)
    out["tokenizer"] = new_tok
    out["encoded"] = new_enc
    # metadata["vocab_size"] is a separate copy of the old tokenizer.vocab_size, read
    # directly by src/data/prepare_data.py's cache-loading path (cfg.model.vocab_size =
    # cache_data["metadata"]["vocab_size"]) -- must be bumped too, or any code path that
    # loads this cache the "normal" way (not through eid_src's own loader) would silently
    # undersize the embedding table.
    if "metadata" in out and isinstance(out["metadata"], dict) and "vocab_size" in out["metadata"]:
        new_metadata = dict(out["metadata"])
        new_metadata["vocab_size"] = new_vocab_size
        out["metadata"] = new_metadata

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(out, out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    args = ap.parse_args()
    build(args.in_path, args.out_path)
