"""Per-head layer-0 attention profiles (panel A data) for every H3 model: 6 datasets x splits 42-51.
Used to choose which single model panel A shows. Same measurement as eid_attention_shap.attention_pass
(20,000 test walks, rng 42). Output: outputs/attn_shap_multiseed/h3_pmf/<ds>_s<seed>.npz (pmf [nlayers,
nhead, 2*max_dist+1], max_dist, nhead).

Usage: .venv/bin/python experiments/edge_identity_tokens/attn_headgrid_multiseed.py --device 0 --jobs ds:seed ...
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from experiments.edge_identity_tokens import eid_attention_shap as eas  # noqa: E402

OUT = os.path.join(ROOT, "outputs", "attn_shap_multiseed", "h3_pmf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="0")
    ap.add_argument("--jobs", nargs="+", required=True)
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.device}")
    os.makedirs(OUT, exist_ok=True)
    for job in args.jobs:
        ds, seed = job.rsplit(":", 1)
        path = os.path.join(OUT, f"{ds}_s{seed}.npz")
        if os.path.exists(path):
            continue
        lit, ds_test, idx, collate, _ = eas.load(ds, dev, seed=int(seed))
        a = eas.attention_pass(ds, lit, ds_test, idx, collate, dev)
        np.savez(path, pmf=a["pmf"], max_dist=a["max_dist"], nhead=a["nhead"])
        print(f"[DONE] {ds} s{seed}", flush=True)
        del lit
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
