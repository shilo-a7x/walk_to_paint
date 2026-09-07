"""Track 2 item 4 diagnostic (no retraining): does a trained EID model's per-edge
embedding encode sign-predictive information BEYOND what its two endpoint vertices
already imply?

Loads an existing checkpoint's trained edge_embed_low/edge_proj and base_embed/
node_proj weights directly (via EIDLitEdgeClassifier.load_from_checkpoint), computes
every edge's (a) pure identity embedding (edge_proj(edge_embed_low(edge_id)) -- the
same content vector the transformer sees at that edge's position, before context) and
(b) a vertex-pair baseline (sum of the SAME trained node embeddings for its two
endpoints), then fits/evaluates a held-out logistic-regression probe from each
representation to the edge's true sign, plus a concat of both. All CPU, no GPU needed,
no retraining -- if (a) doesn't beat (b) by much, the trained embeddings aren't storing
real residual edge-specific signal, independent of what the end-to-end AUC numbers say.

Usage:
  .venv/bin/python experiments/edge_identity_tokens/probe_edge_embeddings.py <checkpoint>
"""
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from experiments.edge_identity_tokens.eid_src.model.lit_model import EIDLitEdgeClassifier
from experiments.edge_identity_tokens.eid_src.data.prepare_eid_data import prepare_eid_data
from experiments.edge_identity_tokens.run_eid import EID_CACHE_PATH, ensure_eid_cache


def held_out_edge_ids(loader, ignore_index):
    """Every edge_id that ever appears as a masked PREDICTION TARGET in this loader
    -- i.e. an edge whose own sign was never visible to the model, at training time,
    at the position being predicted (for val/test loaders specifically: production's
    split-exclusion also keeps these edges out of every OTHER walk's context, so their
    identity embedding never got a gradient nudge tied to their own true sign at all)."""
    ids = []
    for input_ids, labels, attention_mask, metadata in loader:
        target_mask = labels != ignore_index
        ids.append(metadata["edge_ids"][target_mask])
    return set(torch.cat(ids).tolist()) if ids else set()


def probe(Xtr, ytr, Xte, yte):
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=str)
    args = ap.parse_args()

    lit_model = EIDLitEdgeClassifier.load_from_checkpoint(args.checkpoint, map_location="cpu")
    model = lit_model.model
    cfg = lit_model.cfg
    print(f"dataset={cfg.dataset.name}  edge_embed_rank={model.edge_embed_rank}  "
          f"residual_baseline={getattr(model, 'edge_residual_baseline', False)}")
    assert model.edge_embed_rank > 0, "diagnostic needs a factorized (rank>0) checkpoint"

    eid_cache_path = EID_CACHE_PATH.format(dataset=cfg.dataset.name, num_walks=int(cfg.dataset.num_walks))
    cache_tok = torch.load(eid_cache_path, map_location="cpu", weights_only=False)["tokenizer"]
    edge_u_ids = cache_tok["edge_u_ids"].long()
    edge_v_ids = cache_tok["edge_v_ids"].long()
    y = cache_tok["edge_sign_lookup"].long().numpy()  # 0/1 per edge
    num_edges = edge_u_ids.shape[0]
    print(f"num_edges={num_edges}  positive_rate={y.mean():.4f}")

    model.eval()
    with torch.no_grad():
        edge_ids_0 = torch.arange(num_edges)
        edge_embed = model.edge_proj(model.edge_embed_low(edge_ids_0)).numpy()
        pair_baseline = (model._node_content(edge_u_ids) + model._node_content(edge_v_ids)).numpy()

    print(f"edge_embed dim={edge_embed.shape[1]}  pair_baseline dim={pair_baseline.shape[1]}")

    # Split-aware probe: "seen" edges (ever a supervised prediction target during EID
    # training -- train+mask pool) train the probe; "held_out" edges (val+test, whose
    # own sign was NEVER visible to the model at any position, per split-exclusion)
    # test it. A naive random 80/20 split of all edges would leak train-split edges
    # (whose edge_embed_low row could have memorized "my own identity -> my own sign"
    # exactly the pathway edge_replace_prob exists to fight) into the probe's test
    # fold, inflating the edge-alone number for the wrong reason.
    ensure_eid_cache(cfg, eid_cache_path)
    data_module = prepare_eid_data(cfg, eid_cache_path)
    ignore_index = int(cfg.model.ignore_index)
    held_out = held_out_edge_ids(data_module["val"], ignore_index) | held_out_edge_ids(data_module["test"], ignore_index)
    all_ids = set(range(num_edges))
    seen = sorted(all_ids - held_out)
    held_out = sorted(held_out)
    print(f"seen (train+mask pool): {len(seen)}   held_out (val+test, never gradient-exposed to own sign): {len(held_out)}")

    ytr, yte = y[seen], y[held_out]

    def split_probe(X):
        return probe(X[seen], ytr, X[held_out], yte)

    auc_edge = split_probe(edge_embed)
    auc_pair = split_probe(pair_baseline)
    auc_concat = split_probe(np.concatenate([edge_embed, pair_baseline], axis=1))

    # Chance/majority-class reference: with class_weight="balanced" logistic regression
    # on pure noise, AUC concentrates around 0.5 -- included as a sanity floor.
    rng = np.random.RandomState(0)
    auc_random = split_probe(rng.normal(size=edge_embed.shape))

    print("\n=== held-out linear-probe AUC (edge sign), leakage-free split ===")
    print(f"  random noise (floor)         : {auc_random:.4f}")
    print(f"  vertex-pair baseline alone   : {auc_pair:.4f}")
    print(f"  edge identity embedding alone: {auc_edge:.4f}")
    print(f"  concat(edge, pair)           : {auc_concat:.4f}")
    print(f"\n  edge embedding vs pair baseline delta: {(auc_edge - auc_pair)*100:+.2f}pp")
    print(f"  concat vs pair-alone delta           : {(auc_concat - auc_pair)*100:+.2f}pp")


if __name__ == "__main__":
    main()
