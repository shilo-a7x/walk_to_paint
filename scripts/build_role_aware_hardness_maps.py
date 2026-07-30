"""Roadmap follow-up: build role-aware hardness maps (E22, revised) --
hardness_source.pt = H_out(n) (predicts error when n is the SOURCE of the masked edge)
hardness_target.pt = H_in(n)  (predicts error when n is the TARGET of the masked edge)
Both computed from TRAIN+MASK edges only (leakage-safe). No "both-sided" restriction --
verified (scripts/hardness_role_aware_check.py) that H_out alone is a much better
predictor of source-role error than the symmetric blend, and H_in alone a much better
predictor of target-role error, so each map only needs its OWN direction defined --
substantially better node coverage than the earlier symmetric E22 map (1.3x-4.9x more
nodes scored, see HARDNESS_MINER_ROADMAP.md). No training -- pure graph statistics.
"""
import sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import torch
import numpy as np
from scripts.balance_theory_paths import load_edges_canonical
from scripts.hardness_entropy_screen import node_entropy_train_mask
from scripts.hardness_predictive_validity import DATASETS

OUT_TAG = "E22_HARDNODE_ENTROPY"


def main():
    for ds, (data_dir_name, cache_file) in DATASETS.items():
        edges = load_edges_canonical(ds)
        n_nodes = max(max(u, v) for u, v, _ in edges) + 1
        cache = torch.load(f"data/{data_dir_name}/{cache_file}", map_location="cpu", weights_only=False)
        token2id = cache["tokenizer"]["token2id"]
        vocab_size = cache["metadata"]["vocab_size"]
        train_mask_edges = cache["splits"]["train"] + cache["splits"]["mask"]

        h_out, h_in = node_entropy_train_mask(train_mask_edges, n_nodes)

        def to_tensor(cand):
            t = torch.zeros(vocab_size, dtype=torch.float32)
            n_set = 0
            for n in range(n_nodes):
                if not np.isfinite(cand[n]):
                    continue
                tok = token2id.get(f"N_{n}")
                if tok is None or tok >= vocab_size:
                    continue
                t[tok] = float(cand[n])
                n_set += 1
            return t, n_set

        t_source, n_source = to_tensor(h_out)
        t_target, n_target = to_tensor(h_in)

        out_dir = Path(f"outputs/{ds}/{OUT_TAG}")
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(t_source, str(out_dir / "hardness_source.pt"))
        torch.save(t_target, str(out_dir / "hardness_target.pt"))
        print(
            f"{ds}: source={n_source} nodes (mean={float(t_source[t_source>0].mean()):.4f}), "
            f"target={n_target} nodes (mean={float(t_target[t_target>0].mean()):.4f})"
        )


if __name__ == "__main__":
    main()
