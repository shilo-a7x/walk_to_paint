"""Evaluate an already-trained EID checkpoint's test AUC directly, no retraining.

Used to get bitcoin-alpha's real test AUC for the Optuna study's best trial
(trial 94, val_auc=0.8978) without retraining it from scratch -- the checkpoint
already exists (ModelCheckpoint saved it during the search). Reuses
EIDLitEdgeClassifier.load_from_checkpoint, which recovers cfg from
save_hyperparameters() (inherited from production LitEdgeClassifier) the same
way run_posthoc.py does for production checkpoints.

Usage:
  .venv/bin/python experiments/edge_identity_tokens/eval_eid_checkpoint.py \
      <checkpoint_path> --device 0
"""
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import torch
from pytorch_lightning import Trainer

from experiments.edge_identity_tokens.eid_src.model.lit_model import EIDLitEdgeClassifier
from experiments.edge_identity_tokens.eid_src.data.prepare_eid_data import prepare_eid_data
from experiments.edge_identity_tokens.run_eid import EID_CACHE_PATH, ensure_eid_cache


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=str)
    p.add_argument("--device", type=int, default=0)
    args = p.parse_args()

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    model = EIDLitEdgeClassifier.load_from_checkpoint(args.checkpoint, map_location="cpu")
    cfg = model.cfg
    print(f"Loaded checkpoint for dataset={cfg.dataset.name}, "
          f"edge_embed_rank={getattr(cfg.model, 'edge_embed_rank', None)}")

    eid_cache_path = EID_CACHE_PATH.format(dataset=cfg.dataset.name, num_walks=int(cfg.dataset.num_walks))
    ensure_eid_cache(cfg, eid_cache_path)
    data_module = prepare_eid_data(cfg, eid_cache_path)

    trainer = Trainer(
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
    )
    results = trainer.test(model, data_module["test"])
    print("RESULT:", results)


if __name__ == "__main__":
    main()
