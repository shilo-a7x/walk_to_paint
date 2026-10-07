"""Enqueue the full thesis campaign (2026-10-07) for run_reg_campaign.py: the adopted EID
architecture H3 and every ablation, 6 datasets x seeds 42-51.

H3 = production's per-dataset config (configs/<ds>.yaml, untouched) + per-edge identity
(rank 16, added to the content vector), target identity always hidden, VAL/TEST edges hidden
and blocked during training, and shown as <UNK> identity + sign when they are context at test
time (eid_unseen_identity_unk).

Condition grid over what the model sees (N = node tokens, I = context-edge identity,
S = context-edge sign):
  H3        N I S   adopted model (no ablation)
  H3_NI     N I -   mask_context_sign_only
  H3_N      N - -   mask_context_edges
  H3_NS     N - S   eid_identity_off (production-equivalent inside EID)
  H3_IS     - I S   mask_node_tokens
  H3_S      - - S   mask_node_tokens + eid_identity_off
  H3_I      - I -   mask_node_tokens + mask_context_sign_only
  H3_0      - - -   mask_node_tokens + mask_context_edges (walk shape only)
Other ablations:
  H3_SS     scramble_edge_signs   (fixed ~50% of edges show the wrong sign)
  H3_DIR    randomize_walk_direction (fixed per-walk coin flip, whole walk reversed)
  H3_PAIR   pair_only_attention   (target sees only its two endpoint nodes)
  H3_IDSCR  scramble_edge_identity (context edges show a fixed wrong identity)

Usage: .venv/bin/python experiments/edge_identity_tokens/make_thesis_jobs.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.edge_identity_tokens.make_reg_jobs import enqueue, job  # noqa: E402

H3 = ["model.edge_embed_rank=16", "model.edge_sign_combine=add", "model.node_embed_dim=0",
      "model.edge_residual_baseline=false", "model.mask_target_identity=true",
      "model.edge_replace_prob=0.2", "model.edge_replace_unk_ratio=0.7", "model.edge_embed_weight_decay=0.03",
      "model.eid_reveal_holdout_identity=false", "model.eid_unseen_identity_unk=true"]

# production walk budgets (configs/<ds>.yaml); heavy first so the queue tail stays short
WALKS = {"slashdot090221": 1647606, "epinions": 840799, "wiki-rfa": 265817,
         "wiki-elec": 155534, "bitcoin-otc": 177960, "bitcoin-alpha": 120930}
SEEDS = range(42, 52)

MN, MCE, MCSO, IDOFF = ("model.mask_node_tokens=true", "model.mask_context_edges=true",
                        "model.mask_context_sign_only=true", "model.eid_identity_off=true")
CONDITIONS = [
    ("H3", []),
    ("H3_NI", [MCSO]), ("H3_N", [MCE]), ("H3_NS", [IDOFF]),
    ("H3_IS", [MN]), ("H3_S", [MN, IDOFF]), ("H3_I", [MN, MCSO]), ("H3_0", [MN, MCE]),
    ("H3_SS", ["model.scramble_edge_signs=true"]),
    ("H3_DIR", ["model.randomize_walk_direction=true"]),
    ("H3_PAIR", ["model.pair_only_attention=true"]),
    ("H3_IDSCR", ["model.scramble_edge_identity=true"]),
]


def main():
    for i, (name, extra) in enumerate(CONDITIONS):
        jobs = [job(name, ds, s, WALKS[ds], H3 + extra) for ds in WALKS for s in SEEDS]
        enqueue(f"t{i:02d}_{name}", jobs)


if __name__ == "__main__":
    main()
