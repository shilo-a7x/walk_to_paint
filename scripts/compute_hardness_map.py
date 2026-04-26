#!/usr/bin/env python3
"""Compute per-node hardness map using a tiny miner transformer.

Algorithm:
  1. Load cached dataset (.pt file already built by a previous run).
  2. Train a tiny transformer (emb=16, hid=16, nhead=2, nlayers=2) on the
     training split for N epochs.
  3. Run the miner on the full training set and for every masked edge record
       correct = (pred == true_label)
     and accumulate correct/total counts for the two adjacent node tokens
     (position-1 and position+1 relative to the mask position).
  4. hardness[node_token] = 1 - accuracy(node_token)   (0 if never seen)
  5. Save as a float32 tensor of shape [vocab_size] to --out path.

Usage::
    python scripts/compute_hardness_map.py \\
        --cache tmp/transformer_incremental/<suite>/dataset_cache.pt \\
        --out   outputs/.../E14_.../hardness_map.pt \\
        --device 1 --epochs 5
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.stage_dataset import StageViewDataset  # noqa: E402
from src.model.model import TransformerModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True, help="Path to dataset_cache.pt")
    p.add_argument("--out", required=True, help="Output path for hardness_map.pt")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-walk-edges",
        type=int,
        default=0,
        help="If >0, miner uses only samples whose target walk has <= this many edges",
    )
    p.add_argument(
        "--dynamic-pool",
        action="store_true",
        default=False,
        help="If set, miner trains with rotating targets over the full TRAIN+MASK pool each epoch "
             "(mirrors _sample_epoch_targets in lit_model.py). Eval collection always covers the full pool.",
    )
    return p.parse_args()


def _target_rows_by_max_walk_edges(labels, metadata, ignore_index: int, max_walk_edges: int):
    """Return row mask selecting only samples with short-enough target walks."""
    B = labels.size(0)
    keep_rows = torch.ones(B, dtype=torch.bool, device=labels.device)
    if max_walk_edges <= 0:
        return keep_rows

    target_mask = labels != ignore_index
    rows, cols = target_mask.nonzero(as_tuple=True)
    if rows.numel() == 0:
        return torch.zeros(B, dtype=torch.bool, device=labels.device)

    walk_lengths = metadata["walk_lengths"].to(labels.device)
    target_walk_token_len = walk_lengths[rows, cols].long().clamp(min=1)
    # Tokenized walk shape is [node, edge, node, ...], so edges=(tokens-1)//2.
    target_walk_edges = (target_walk_token_len - 1) // 2

    keep_rows = torch.zeros(B, dtype=torch.bool, device=labels.device)
    keep_rows[rows] = target_walk_edges <= int(max_walk_edges)
    return keep_rows


# ---------------------------------------------------------------------------
# Full-pool miner dataset
# ---------------------------------------------------------------------------

class FullPoolMinerDataset(torch.utils.data.Dataset):
    """Like StageViewDataset(stage='train') but exposes the entire TRAIN+MASK
    pool as potential targets.

    selected_edge_ids (CPU 1-D LongTensor or None):
      * None  → all pool edges are labeled (used in eval collection pass)
      * Tensor → only those edge IDs are masked/labeled (used during dynamic
                 training to rotate a ~40 % subset each epoch)
    """

    _TRAIN = 0
    _MASK = 1
    _BAD = -1

    def __init__(self, cache_data: dict, selected_edge_ids=None):
        enc = cache_data["encoded"]
        tok = cache_data["tokenizer"]

        self.input_ids = enc["input_ids"]
        self.edge_split_mask = enc["edge_split_mask"]
        self.attention_base = enc["attention_base"]
        self.edge_ids = enc.get("edge_ids")

        # Reconstruct walk_ids/positions/walk_lengths from attention_base when absent
        # (v1.2+ caches no longer store these redundant tensors).
        N, seq_len = self.attention_base.shape

        if enc.get("walk_ids") is not None:
            self.walk_ids = enc["walk_ids"]
        else:
            self.walk_ids = torch.arange(N, dtype=torch.long).unsqueeze(1).expand(N, seq_len)

        if enc.get("positions") is not None:
            self.positions = enc["positions"]
        else:
            self.positions = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(N, seq_len)

        if enc.get("walk_lengths") is not None:
            self.walk_lengths = enc["walk_lengths"]
        else:
            lengths = self.attention_base.sum(dim=1, dtype=torch.long)
            self.walk_lengths = lengths.unsqueeze(1).expand(N, seq_len)

        self.mask_id = tok["MASK_ID"]
        self.ignore_index = tok["UNK_LABEL_ID"]

        vocab_size = tok["vocab_size"]
        self.id2class = torch.full((vocab_size,), self.ignore_index, dtype=torch.long)
        for class_id, edge_tok in tok["id2edge_label"].items():
            tok_id = tok["token2id"].get(edge_tok)
            if tok_id is not None:
                self.id2class[int(tok_id)] = int(class_id)

        # _selected_lookup: bool tensor [max_edge_id+1], True = this edge is a target.
        # None means expose all pool edges (eval pass or non-dynamic mode).
        self._selected_lookup: torch.Tensor | None = None
        self.update_selected(selected_edge_ids)

    def update_selected(self, selected_edge_ids):
        """Precompute a bool lookup table so __getitem__ is O(S) not O(S*K)."""
        if selected_edge_ids is None or self.edge_ids is None:
            self._selected_lookup = None
        elif selected_edge_ids.numel() == 0:
            self._selected_lookup = torch.zeros(1, dtype=torch.bool)
        else:
            max_id = int(selected_edge_ids.max().item())
            lookup = torch.zeros(max_id + 1, dtype=torch.bool)
            lookup[selected_edge_ids] = True
            self._selected_lookup = lookup

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_ids = self.input_ids[idx].clone()
        split_mask = self.edge_split_mask[idx]
        attention_mask = self.attention_base[idx].clone()

        in_pool = (split_mask == self._TRAIN) | (split_mask == self._MASK)
        # Mask out val/test edges so contexts outside the pool stay silent
        disallowed = (split_mask != self._BAD) & (~in_pool)

        labels = torch.full_like(input_ids, self.ignore_index)

        if in_pool.any():
            target_pos = in_pool.clone()
            if self._selected_lookup is not None and self.edge_ids is not None:
                edge_ids_row = self.edge_ids[idx]  # [S]
                lookup = self._selected_lookup
                clipped = edge_ids_row.clamp(0, lookup.numel() - 1)
                selected_mask = lookup[clipped] & (edge_ids_row >= 0)
                target_pos = target_pos & selected_mask
            if target_pos.any():
                labels[target_pos] = self.id2class[input_ids[target_pos]]
                input_ids[target_pos] = self.mask_id

        input_ids[disallowed] = self.mask_id
        attention_mask[disallowed] = 0

        if (
            self.edge_ids is not None
            and self.walk_ids is not None
            and self.positions is not None
            and self.walk_lengths is not None
        ):
            metadata = {
                "edge_ids": self.edge_ids[idx],
                "walk_ids": self.walk_ids[idx],
                "positions": self.positions[idx],
                "walk_lengths": self.walk_lengths[idx],
                "edge_split_mask": split_mask,
                "edge_classes": self.id2class[self.input_ids[idx]],
            }
            return input_ids, labels, attention_mask, metadata
        return input_ids, labels, attention_mask


def _build_pool_unique_edges(
    cache_data: dict, ignore_index: int
):
    """Return (unique_edge_ids [P], unique_edge_classes [P]) for the full TRAIN+MASK pool."""
    enc = cache_data["encoded"]
    tok = cache_data["tokenizer"]
    edge_split = enc["edge_split_mask"]    # [N, S]
    edge_ids_t = enc.get("edge_ids")        # [N, S]
    input_ids_all = enc["input_ids"]        # [N, S]

    if edge_ids_t is None:
        raise RuntimeError("cache is missing edge_ids — cannot build dynamic pool")

    vocab_size = tok["vocab_size"]
    id2class = torch.full((vocab_size,), ignore_index, dtype=torch.long)
    for class_id, edge_tok in tok["id2edge_label"].items():
        tok_id = tok["token2id"].get(edge_tok)
        if tok_id is not None:
            id2class[int(tok_id)] = int(class_id)

    pool_mask = ((edge_split == 0) | (edge_split == 1)) & (edge_ids_t >= 0)  # TRAIN=0, MASK=1
    pool_eids = edge_ids_t[pool_mask].long()
    pool_cls = id2class[input_ids_all[pool_mask]]

    valid = pool_cls != ignore_index
    pool_eids = pool_eids[valid]
    pool_cls = pool_cls[valid]

    max_eid = int(pool_eids.max().item())
    class_map = torch.full((max_eid + 1,), ignore_index, dtype=torch.long)
    class_map[pool_eids] = pool_cls

    unique_eids = torch.unique(pool_eids)
    unique_cls = class_map[unique_eids]
    ok = unique_cls != ignore_index
    return unique_eids[ok], unique_cls[ok]


def _sample_dynamic_pool(
    unique_edge_ids: torch.Tensor,
    unique_edge_classes: torch.Tensor,
    num_classes: int,
    target_ratio: float,
    seed: int,
) -> torch.Tensor:
    """Per-class-balanced sample from the full pool.  Mirrors _sample_epoch_targets."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    chunks = []
    for class_id in range(num_classes):
        mask = unique_edge_classes == class_id
        class_ids = unique_edge_ids[mask]
        count = int(class_ids.numel())
        if count == 0:
            continue
        k = int(round(count * target_ratio))
        if target_ratio > 0.0 and k == 0:
            k = 1
        k = min(k, count)
        if k == count:
            chunks.append(class_ids)
        elif k > 0:
            perm = torch.randperm(count, generator=gen)
            chunks.append(class_ids[perm[:k]])
    if chunks:
        return torch.unique(torch.cat(chunks, dim=0))
    return torch.empty(0, dtype=torch.long)


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _build_tiny_cfg(meta: dict, max_walk_length: int) -> object:
    return OmegaConf.create(
        {
            "model": {
                "vocab_size": int(meta["vocab_size"]),
                "embedding_dim": 16,
                "hidden_dim": 16,
                "nhead": 2,
                "nlayers": 2,
                "num_classes": int(meta["num_classes"]),
                "dropout": 0.0,
                "pad_id": int(meta["pad_id"]),
                "ignore_index": int(meta["ignore_index"]),
                "node_context_mode": "none",
                "node_mask_prob": 0.0,
                "node_noise_sigma": 0.0,
            },
            "dataset": {"max_walk_length": max_walk_length},
        }
    )


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    )

    cache_path = Path(args.cache)
    if not cache_path.exists():
        print(f"ERROR: cache not found at {cache_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[{_ts()}][miner] Loading cache from {cache_path} ...", flush=True)
    cache_data = torch.load(str(cache_path), map_location="cpu", weights_only=False)

    meta = cache_data["metadata"]
    vocab_size = int(meta["vocab_size"])
    ignore_index = int(meta["ignore_index"])

    # Infer max_walk_length from sequence length: seq_len = 2*mwl + 1
    seq_len = int(cache_data["encoded"]["input_ids"].shape[1])
    max_walk_length = (seq_len - 1) // 2

    cfg = _build_tiny_cfg(meta, max_walk_length)
    model = TransformerModel(cfg).to(device)
    num_classes = int(meta["num_classes"])

    # -----------------------------------------------------------------
    # Select dataset and (if dynamic-pool) pre-compute the edge pool
    # -----------------------------------------------------------------
    if args.dynamic_pool:
        print(f"[{_ts()}][miner] --dynamic-pool: building full TRAIN+MASK pool for sampling ...", flush=True)
        pool_unique_eids, pool_unique_cls = _build_pool_unique_edges(cache_data, ignore_index)
        # target_ratio: fraction of pool that becomes masked each epoch
        #   = |MASK edges| / (|TRAIN| + |MASK|)  (mirrors _sample_epoch_targets)
        enc_split = cache_data["encoded"]["edge_split_mask"]
        n_train_pos = int((enc_split == 0).sum().item())
        n_mask_pos  = int((enc_split == 1).sum().item())
        target_ratio = n_mask_pos / max(1, n_train_pos + n_mask_pos)
        print(
            f"[{_ts()}][miner]   pool unique edges: {pool_unique_eids.numel()}, "
            f"target_ratio={target_ratio:.3f}",
            flush=True,
        )
        train_ds = FullPoolMinerDataset(cache_data, selected_edge_ids=None)
    else:
        train_ds = StageViewDataset(cache_data, stage="train", dynamic_train_masking=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"[{_ts()}][miner] Training tiny model for {args.epochs} epoch(s) ...", flush=True)
    model.train()
    for epoch in range(args.epochs):
        # Dynamic pool: rotate target subset to match this epoch
        if args.dynamic_pool:
            selected = _sample_dynamic_pool(
                pool_unique_eids, pool_unique_cls,
                num_classes=num_classes,
                target_ratio=target_ratio,
                seed=args.seed + epoch,
            )
            train_ds.update_selected(selected)

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            input_ids, labels, attention_mask, metadata = batch[0], batch[1], batch[2], batch[3]
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)

            if args.max_walk_edges > 0:
                keep_rows = _target_rows_by_max_walk_edges(
                    labels,
                    metadata,
                    ignore_index=ignore_index,
                    max_walk_edges=args.max_walk_edges,
                )
                if not keep_rows.any():
                    continue
                labels = labels.clone()
                labels[~keep_rows] = ignore_index
                if torch.all(labels == ignore_index):
                    continue

            logits = model(input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(
                logits.view(-1, cfg.model.num_classes),
                labels.view(-1),
                ignore_index=ignore_index,
            )
            if not torch.isfinite(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"[{_ts()}][miner]   epoch {epoch + 1}/{args.epochs}: loss={avg_loss:.4f}", flush=True)

    # -----------------------------------------------------------------
    # Collect per-edge predictions on the FULL pool (TRAIN+MASK always)
    # -----------------------------------------------------------------
    print(f"[{_ts()}][miner] Collecting predictions on full TRAIN+MASK pool ...", flush=True)
    model.eval()

    node_correct = torch.zeros(vocab_size, dtype=torch.long)
    node_total = torch.zeros(vocab_size, dtype=torch.long)

    if args.dynamic_pool:
        # Full-pool eval: expose every TRAIN+MASK edge; no selection filter
        eval_ds = FullPoolMinerDataset(cache_data, selected_edge_ids=None)
    else:
        eval_ds = train_ds  # unchanged: StageViewDataset(stage="train")

    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    with torch.no_grad():
        for batch in eval_loader:
            input_ids_cpu = batch[0]  # keep CPU copy for token-id lookup
            labels_cpu = batch[1]
            attention_mask = batch[2].to(device)
            metadata = batch[3]

            if args.max_walk_edges > 0:
                keep_rows = _target_rows_by_max_walk_edges(
                    labels_cpu,
                    metadata,
                    ignore_index=ignore_index,
                    max_walk_edges=args.max_walk_edges,
                )
                if not keep_rows.any():
                    continue
                labels_cpu = labels_cpu.clone()
                labels_cpu[~keep_rows.cpu()] = ignore_index
                if torch.all(labels_cpu == ignore_index):
                    continue

            logits = model(input_ids_cpu.to(device), attention_mask=attention_mask)
            preds_cpu = logits.argmax(dim=-1).cpu()  # [B, S]

            B, S = labels_cpu.shape
            target_mask = labels_cpu != ignore_index  # [B, S]
            target_rows, target_cols = target_mask.nonzero(as_tuple=True)

            if target_rows.numel() == 0:
                continue

            correct = (
                preds_cpu[target_rows, target_cols]
                == labels_cpu[target_rows, target_cols]
            ).long()

            # Left adjacent node (mask_pos - 1)
            valid_left = target_cols > 0
            if valid_left.any():
                left_toks = input_ids_cpu[
                    target_rows[valid_left], target_cols[valid_left] - 1
                ]
                node_correct.scatter_add_(0, left_toks, correct[valid_left])
                node_total.scatter_add_(
                    0, left_toks, torch.ones_like(left_toks)
                )

            # Right adjacent node (mask_pos + 1)
            valid_right = target_cols + 1 < S
            if valid_right.any():
                right_toks = input_ids_cpu[
                    target_rows[valid_right], target_cols[valid_right] + 1
                ]
                node_correct.scatter_add_(0, right_toks, correct[valid_right])
                node_total.scatter_add_(
                    0, right_toks, torch.ones_like(right_toks)
                )

    # -----------------------------------------------------------------
    # Compute hardness = 1 - accuracy  (0 for unseen nodes)
    # -----------------------------------------------------------------
    hardness = torch.zeros(vocab_size, dtype=torch.float32)
    with_data = node_total > 0
    hardness[with_data] = 1.0 - (
        node_correct[with_data].float() / node_total[with_data].float()
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(hardness, str(out_path))

    n_nodes = int(with_data.sum().item())
    avg_h = float(hardness[with_data].mean().item()) if n_nodes > 0 else 0.0
    print(
        f"[{_ts()}][miner] Saved hardness map \u2192 {out_path}  "
        f"({n_nodes} nodes, mean_hardness={avg_h:.4f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
