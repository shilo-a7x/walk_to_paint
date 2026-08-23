"""Single-split pilot: does giving edges their own identity token (sign added
separately, like positional encoding) change anything vs. the production
2-shared-sign-token scheme?

Bitcoin-alpha only, seed 42 only (smallest dataset, single split, per user request
2026-08-23). Self-contained training loop (plain PyTorch, not LitEdgeClassifier) --
this pilot doesn't need dynamic resplit, hardness reweighting, or any of the other
production training machinery, and a standalone loop is easier to verify correct for
a first pass on a genuinely new architecture. Same hyperparameters as
configs/bitcoin-alpha.yaml (embedding_dim=64, hidden_dim=128, nhead=4, nlayers=3,
dropout=0.20, lr=0.0016, batch_size=1024, local_attention_window=4) for a fair
comparison against the production number (0.913 test AUC, walk-level aggregated to
edge-level via func_logit_power in production; this script reports both walk-level
AUC and a simple mean-pooled edge-level AUC, not the fitted aggregator -- close
enough for "does this change anything", not meant to reproduce Table 1 exactly).

Usage:
  .venv/bin/python experiments/edge_identity_tokens/train_pilot.py
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from experiments.edge_identity_tokens.dataset import EdgeIdentityStageViewDataset, eid_collate_fn
from experiments.edge_identity_tokens.model import EdgeIdentityTransformerModel

CACHE_IN = "data/bitcoin-alpha/dataset_cache__edge_cover_nw120930_mw80_seed42.pt"
CACHE_OUT = "experiments/edge_identity_tokens/cache/bitcoin-alpha_eid.pt"

EMBEDDING_DIM = 64
HIDDEN_DIM = 128
NHEAD = 4
NLAYERS = 3
DROPOUT = 0.20
LOCAL_ATTENTION_WINDOW = 4
BATCH_SIZE = 1024
LR = 0.0016
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 0.54
MAX_EPOCHS = 50
EARLY_STOP_PATIENCE = 15
EARLY_STOP_MIN_DELTA = 0.001
SEED = 42
DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"

# Edge-identity replacement regularizer -- the edge-token analogue of production's
# node_context_mode=replace / node_replace_prob (src/model/lit_model.py
# _maybe_apply_node_replacement). Off by default (0.0); pass --edge-replace-prob to
# enable. See apply_edge_identity_replacement() below for the exact mechanism.
EDGE_REPLACE_UNK_RATIO = 0.7


def apply_edge_identity_replacement(input_ids, old_vocab_size, unk_id, replace_prob, unk_ratio):
    """Training-only regularizer: randomly corrupt VISIBLE edge-identity tokens.

    Direct analogue of _maybe_apply_node_replacement for the new per-edge identity
    tokens minted by build_cache.py (ids >= old_vocab_size). A visible edge-identity
    position (real edge token, not <MASK>) is, independently with probability
    replace_prob, replaced by either the shared <UNK> id (probability unk_ratio) or a
    different edge's identity token drawn from elsewhere in the same batch
    (probability 1 - unk_ratio). sign_ids is untouched by this function -- only the
    "which edge is this" signal is corrupted, never the sign, so the model is pushed
    away from keying its prediction off a specific edge's own identity embedding and
    toward the sign/structural context instead (the same rationale D/R already serve
    for node identity in production).
    """
    if replace_prob <= 0.0:
        return input_ids

    x = input_ids.clone()
    candidates = x >= old_vocab_size  # real edge-identity tokens only (not <MASK>, not nodes)
    if not candidates.any():
        return x

    replace_mask = (torch.rand_like(candidates, dtype=torch.float) < replace_prob) & candidates
    if not replace_mask.any():
        return x

    edge_pool = x[candidates]
    selected = replace_mask.nonzero(as_tuple=False)
    use_unk = torch.rand(selected.size(0), device=x.device) < unk_ratio

    if use_unk.any():
        unk_positions = selected[use_unk]
        x[unk_positions[:, 0], unk_positions[:, 1]] = unk_id

    rand_count = int((~use_unk).sum().item())
    if rand_count > 0:
        rand_positions = selected[~use_unk]
        rand_idx = torch.randint(0, edge_pool.numel(), (rand_count,), device=x.device)
        x[rand_positions[:, 0], rand_positions[:, 1]] = edge_pool[rand_idx]

    return x


def edge_level_mean_auc(edge_ids, probs, targets):
    order = np.argsort(edge_ids, kind="stable")
    eids_s = edge_ids[order]
    probs_s = probs[order]
    y_s = targets[order]
    uniq, first_idx, cnts = np.unique(eids_s, return_index=True, return_counts=True)
    sums = np.add.reduceat(probs_s, first_idx)
    means = sums / cnts
    labels = y_s[first_idx]
    return roc_auc_score(labels, means)


def run_eval(model, loader, ignore_index):
    model.eval()
    all_probs, all_targets, all_edge_ids = [], [], []
    with torch.no_grad():
        for input_ids, labels, attention_mask, metadata in loader:
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            sign_ids = metadata["sign_ids"].to(DEVICE)
            logits = model(input_ids, sign_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=-1)[:, :, 1]

            mask = (labels != ignore_index)
            if not mask.any():
                continue
            all_probs.append(probs[mask].cpu().numpy())
            all_targets.append(labels[mask].cpu().numpy())
            all_edge_ids.append(metadata["edge_ids"][mask].numpy())

    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)
    edge_ids = np.concatenate(all_edge_ids)
    walk_auc = roc_auc_score(targets, probs)
    edge_auc = edge_level_mean_auc(edge_ids, probs, targets)
    return walk_auc, edge_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge-replace-prob", type=float, default=0.0,
                     help="Prob. of corrupting a visible edge-identity token per step (0=off, matches "
                          "production node_replace_prob=0.2 in spirit). Default off, for a clean "
                          "baseline-vs-regularized comparison.")
    ap.add_argument("--edge-replace-unk-ratio", type=float, default=EDGE_REPLACE_UNK_RATIO)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if not os.path.isfile(CACHE_OUT):
        from experiments.edge_identity_tokens.build_cache import build
        build(CACHE_IN, CACHE_OUT)

    cache_data = torch.load(CACHE_OUT, map_location="cpu", weights_only=False)
    tok = cache_data["tokenizer"]
    vocab_size = tok["vocab_size"]
    pad_id = tok["PAD_ID"]
    unk_id = tok["UNK_ID"]
    old_vocab_size = tok["old_vocab_size"]
    ignore_index = tok["UNK_LABEL_ID"]
    max_walk_length = 80

    print(f"vocab_size={vocab_size}  device={DEVICE}  "
          f"edge_replace_prob={args.edge_replace_prob}  edge_replace_unk_ratio={args.edge_replace_unk_ratio}")

    train_ds = EdgeIdentityStageViewDataset(cache_data, stage="train")
    val_ds = EdgeIdentityStageViewDataset(cache_data, stage="val")
    test_ds = EdgeIdentityStageViewDataset(cache_data, stage="test")
    collate = eid_collate_fn(pad_id, ignore_index)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    model = EdgeIdentityTransformerModel(
        vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM, nhead=NHEAD,
        hidden_dim=HIDDEN_DIM, dropout=DROPOUT, nlayers=NLAYERS, num_classes=2,
        max_walk_length=max_walk_length, pad_id=pad_id,
        local_attention_window=LOCAL_ATTENTION_WINDOW,
    ).to(DEVICE)

    class_weights = torch.tensor(cache_data["metadata"]["class_weights"], dtype=torch.float32).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_auc = -1.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        n_batches = 0
        for input_ids, labels, attention_mask, metadata in train_loader:
            input_ids = input_ids.to(DEVICE)
            labels = labels.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            sign_ids = metadata["sign_ids"].to(DEVICE)

            if torch.all(labels == ignore_index):
                continue

            input_ids = apply_edge_identity_replacement(
                input_ids, old_vocab_size, unk_id, args.edge_replace_prob, args.edge_replace_unk_ratio
            )

            logits = model(input_ids, sign_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(
                logits.view(-1, 2), labels.view(-1), weight=class_weights, ignore_index=ignore_index
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        val_walk_auc, val_edge_auc = run_eval(model, val_loader, ignore_index)
        elapsed = time.time() - t0
        print(f"epoch {epoch:3d}  train_loss={total_loss/max(n_batches,1):.4f}  "
              f"val_walk_auc={val_walk_auc:.4f}  val_edge_auc={val_edge_auc:.4f}  ({elapsed:.1f}s)")

        if val_walk_auc > best_val_auc + EARLY_STOP_MIN_DELTA:
            best_val_auc = val_walk_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"early stopping at epoch {epoch} (best val_walk_auc={best_val_auc:.4f})")
                break

    model.load_state_dict(best_state)
    test_walk_auc, test_edge_auc = run_eval(model, test_loader, ignore_index)
    print(f"\nFINAL: best_val_walk_auc={best_val_auc:.4f}  "
          f"test_walk_auc={test_walk_auc:.4f}  test_edge_auc(mean-pool)={test_edge_auc:.4f}")
    print("production baseline, Table 1 (local attention, edge-level, 10-seed mean): 0.9134")
    print("production baseline, THIS seed 42 checkpoint (edge-level, func_logit_power): 0.9188 (E32 ckpt)")


if __name__ == "__main__":
    main()
