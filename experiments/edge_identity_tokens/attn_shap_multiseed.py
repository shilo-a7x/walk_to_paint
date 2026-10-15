"""Seed-stability of the attention-direction / SHAP measurements: production vs H3, seeds 42-51.

The WSDM figure (and the first H3 port) used one checkpoint per dataset (seed 42). This reruns the
same per-instance measurements on all 10 seeds of both models, so the thesis can tell which
per-dataset patterns are stable and which are seed noise.

  prod  production LocalAttn4: seed 42 = E32_PY314_LOCALATTN4 (the WSDM figure's checkpoint, also the
        10-seed campaign's seed 42), seeds 43-51 = MULTISEED_s<seed>_local. Measured by the production
        scripts' OWN functions, unchanged (attention_directionality_panelB_se.run_inference,
        shap_edge_directionality.analyse_dataset) -- only the checkpoint pin is swapped per seed.
  h3    EIDREG_H3_<DS>_s<seed>, measured by eid_attention_shap.py (same conventions); SHAP twice:
        remove="both" (identity <MASK> + sign hidden) and remove="sign" (sign hidden, identity kept).

Per (model, seed, dataset) result pickles under outputs/attn_shap_multiseed/<model>/s<seed>/;
existing ones are skipped. Summarise with --summarise.

Usage: .venv/bin/python experiments/edge_identity_tokens/attn_shap_multiseed.py --model prod --seeds 42 43 --device 0
"""
import argparse
import glob
import os
import pickle
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import scripts.attention_directionality as ad  # noqa: E402
import scripts.attention_directionality_panelB_se as pb  # noqa: E402
import scripts.shap_edge_directionality as sh  # noqa: E402
from experiments.edge_identity_tokens import eid_attention_shap as eas  # noqa: E402

DATASETS = ["bitcoin-alpha", "bitcoin-otc", "epinions", "slashdot090221", "wiki-elec", "wiki-rfa"]
SEEDS = list(range(42, 52))
OUT = os.path.join(ROOT, "outputs", "attn_shap_multiseed")


def prod_pin(ds, seed):
    """(run_dir, epoch) of the production LocalAttn4 checkpoint for this seed."""
    if seed == 42:
        return ad.LOCAL_RUN_INFO_WSDM[ds]
    d = sorted(glob.glob(os.path.join(ROOT, "outputs", ds, f"MULTISEED_s{seed}_local_*")))[-1]
    ck = [c for c in glob.glob(os.path.join(d, "checkpoints", "*.ckpt")) if "last" not in c]
    assert len(ck) == 1, ck
    return os.path.basename(d), int(re.search(r"epoch=(\d+)", ck[0]).group(1))


def run_prod(ds, seed, device, out):
    ad.LOCAL_RUN_INFO[ds] = prod_pin(ds, seed)  # same dict object load_model_and_dataset reads
    pb.OUT_DIR = out
    att = pb.run_inference(ds, device=device)
    shp = sh.analyse_dataset(ds, out, device=device)
    return {"attention": att, "shap": {"both": shp}, "pin": ad.LOCAL_RUN_INFO[ds]}


def run_h3(ds, seed, device):
    import torch
    dev = torch.device(device)
    lit, ds_test, idx, collate, ckpt = eas.load(ds, dev, seed=seed)
    a = eas.attention_pass(ds, lit, ds_test, idx, collate, dev)
    att = {"edge_ids": a["edge_ids"], "fwd_frac": a["fwd"], "bwd_frac": a["bwd"],
           "node_frac": a["node"], "edge_frac": a["edge"]}
    shp = {m: eas.shap_pass(ds, lit, ds_test, idx, collate, dev, remove=m) for m in ("both", "sign")}
    return {"attention": att, "shap": shp, "pin": ckpt}


def summarise():
    import numpy as np
    import pandas as pd
    rows, shap_rows = [], []
    for f in sorted(glob.glob(os.path.join(OUT, "*", "s*", "result_*.pkl"))):
        model, seed, ds = f.split(os.sep)[-3], int(f.split(os.sep)[-2][1:]), os.path.basename(f)[7:-4]
        r = pickle.load(open(f, "rb"))
        a = r["attention"]
        row = {"model": model, "seed": seed, "dataset": ds}
        for x, y, name in (("fwd_frac", "bwd_frac", "fwd_minus_bwd"), ("node_frac", "edge_frac", "node_minus_edge")):
            m, s, lo, hi, nc, wp = pb.cluster_robust_se_diff(a["edge_ids"], a[x], a[y])
            row |= {x[:-5]: float(np.mean(a[x])), y[:-5]: float(np.mean(a[y])), name: m,
                    f"{name}_sig": int(lo > 0) - int(hi < 0)}  # +1 / -1 significant, 0 n.s.
        rows.append(row)
        for mode, res in r["shap"].items():
            for sr in res["summary_rows"]:
                shap_rows.append({"model": model, "shap_remove": mode, "seed": seed} | sr)
    att, shp = pd.DataFrame(rows), pd.DataFrame(shap_rows)
    att.to_csv(os.path.join(OUT, "attention_per_seed.csv"), index=False)
    shp.to_csv(os.path.join(OUT, "shap_per_seed.csv"), index=False)
    print(f"wrote {OUT}/attention_per_seed.csv ({len(att)} rows), shap_per_seed.csv ({len(shp)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["prod", "h3"])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--device", default="0")
    ap.add_argument("--summarise", action="store_true")
    args = ap.parse_args()
    if args.summarise:
        return summarise()
    ad.LOCAL_RUN_INFO_WSDM = dict(ad.LOCAL_RUN_INFO)
    device = f"cuda:{args.device}"
    for seed in args.seeds:
        for ds in args.datasets:
            out = os.path.join(OUT, args.model, f"s{seed}")
            path = os.path.join(out, f"result_{ds}.pkl")
            if os.path.exists(path):
                continue
            os.makedirs(out, exist_ok=True)
            t0 = time.time()
            res = run_prod(ds, seed, device, out) if args.model == "prod" else run_h3(ds, seed, device)
            pickle.dump(res, open(path, "wb"))
            print(f"[DONE] {args.model} s{seed} {ds} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
