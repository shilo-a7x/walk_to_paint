"""EID (H3) port of the WSDM attention-directionality figure data (Fig. attndir A-D).

Same measurements, same conventions as the production scripts, on the seed-42 H3 checkpoint
(production used its seed-42 LocalAttn4 checkpoint):
  - scripts/attention_directionality.py      per-(layer, head) signed attention mass around each
                                              masked test target (Panel A uses layer 0, bitcoin-alpha)
  - scripts/attention_directionality_panelB_se.py
                                              per-instance layer-0 forward/backward and node/edge mass
                                              (mean over heads, self excluded), cluster-robust SE with
                                              cluster = target edge id (Panels B, C)
  - scripts/shap_edge_directionality.py      exact Shapley value of each context edge at token offsets
                                              -4/-2/+2/+4 (hop 1-2 backward/forward) for the target's
                                              predicted P(positive) (Panel D)
Test split, 20,000 walks sampled with rng seed 42 (production's cap and seed). Inputs go through
eid_posthoc.apply_eval_transforms, i.e. exactly what H3 sees at test time. A context edge "removed"
for SHAP shows <MASK> identity and the hidden sign (production: <MASK> sign token) -- the model's
own placeholder for an unavailable edge, so every coalition stays in-distribution.

Outputs (experiments/edge_identity_tokens/thesis_figure_data/), same schemas as aaai2027/figure_data/:
  attndir_panelA_headgrid.csv, attndir_panelBC_summary.csv, attndir_panelBC_se.csv,
  shap_edge_directionality.csv; per-dataset caches under outputs/eid_attention_shap/.

Usage: .venv/bin/python experiments/edge_identity_tokens/eid_attention_shap.py --device 0 [--datasets ...]
"""
import argparse
import csv
import glob
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
torch.backends.mha.set_fastpath_enabled(False)

from run_posthoc import resolve_checkpoint  # noqa: E402
from scripts.attention_directionality import (  # noqa: E402
    AttentionRecordingEncoderLayer, LocalAttentionRecordingEncoderLayer)
from scripts.attention_directionality_panelB_se import cluster_robust_se_diff, cluster_robust_se_single  # noqa: E402
from scripts.shap_edge_directionality import FEATURES, shapley_values  # noqa: E402
from experiments.edge_identity_tokens.eid_posthoc import apply_eval_transforms  # noqa: E402
from experiments.edge_identity_tokens.eid_src.data.prepare_eid_data import prepare_eid_data  # noqa: E402
from experiments.edge_identity_tokens.eid_src.data.stage_dataset import SIGN_NA  # noqa: E402
from experiments.edge_identity_tokens.eid_src.model.lit_model import EIDLitEdgeClassifier  # noqa: E402
from experiments.edge_identity_tokens.run_eid import EID_CACHE_PATH  # noqa: E402

DATASETS = ["bitcoin-alpha", "bitcoin-otc", "epinions", "slashdot090221", "wiki-elec", "wiki-rfa"]
PANEL_A_DATASET = "bitcoin-alpha"
SEED = 42
MAX_SAMPLES = 20000
LAYER = 0
CACHE_DIR = os.path.join(ROOT, "outputs", "eid_attention_shap")
OUT_DIR = os.path.join(ROOT, "experiments", "edge_identity_tokens", "thesis_figure_data")


def load(ds, device, seed=SEED):
    tag = f"EIDREG_H3_{ds.upper().replace('-', '')}_s{seed}_"
    exp_dir = sorted(glob.glob(os.path.join(ROOT, "outputs", ds, f"{tag}*")))[-1]
    ckpt = resolve_checkpoint(Path(exp_dir) / "checkpoints", "best", None)
    cfg = OmegaConf.create(torch.load(str(ckpt), map_location="cpu", weights_only=False)["hyper_parameters"]["cfg"])
    cfg.training.num_workers = 0
    cfg.training.persistent_workers = False
    lit = EIDLitEdgeClassifier.load_from_checkpoint(str(ckpt), cfg=cfg, map_location="cpu").to(device).eval()
    local = lit.model.local_attention_window is not None
    for layer in lit.model.transformer.layers:
        layer.__class__ = LocalAttentionRecordingEncoderLayer if local else AttentionRecordingEncoderLayer
    loader = prepare_eid_data(cfg, EID_CACHE_PATH.format(dataset=ds, num_walks=int(cfg.dataset.num_walks), seed=seed))["test"]
    ds_test = loader.dataset
    n = len(ds_test)
    idx = np.random.default_rng(42).choice(n, MAX_SAMPLES, replace=False).tolist() if n > MAX_SAMPLES else list(range(n))
    return lit, ds_test, idx, loader.collate_fn, str(ckpt)


def batches(ds_test, idx, collate, batch_size, device):
    sub = torch.utils.data.Subset(ds_test, idx)
    for input_ids, labels, attention_mask, metadata in torch.utils.data.DataLoader(
            sub, batch_size=batch_size, shuffle=False, collate_fn=collate):
        yield (input_ids.to(device), labels.to(device), attention_mask.to(device),
               metadata["sign_ids"].to(device), metadata)


def attention_pass(ds, lit, ds_test, idx, collate, device):
    model = lit.model
    nlayers, nhead = len(model.transformer.layers), model.transformer.layers[0].self_attn.num_heads
    window = model.local_attention_window
    max_dist = int(ds_test.lengths.max().item()) - 1
    hist = np.zeros((nlayers, nhead, 2 * max_dist + 1))
    per = defaultdict(list)
    n_targets = 0
    with torch.no_grad():
        for input_ids, labels, am, sign_ids, meta in batches(ds_test, idx, collate, 64, device):
            rows, cols = (labels != lit.ignore_index).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            x, s = apply_eval_transforms(lit, input_ids, sign_ids, labels, meta, device, ds_test)
            model(x, s, attention_mask=am)
            S = x.shape[1]
            j = torch.arange(S, device=device)
            dist = j.unsqueeze(0) - cols.unsqueeze(1)
            amf = am.float()[rows]
            for l, layer in enumerate(model.transformer.layers):
                sel = layer.last_attn_weights[rows, :, cols, :] * amf.unsqueeze(1)  # [M, H, S]
                d_flat = (dist + max_dist).reshape(-1).cpu().numpy()
                for h in range(nhead):
                    hist[l, h] += np.bincount(d_flat, weights=sel[:, h, :].reshape(-1).cpu().numpy(),
                                              minlength=2 * max_dist + 1)
                if l == LAYER:
                    v = sel.mean(1)
                    not_self = (dist != 0).float()
                    is_edge = (j % 2 == 1).float().unsqueeze(0) * not_self
                    per["fwd"].append((v * (dist > 0).float()).sum(1).cpu().numpy())
                    per["bwd"].append((v * (dist < 0).float()).sum(1).cpu().numpy())
                    per["edge"].append((v * is_edge).sum(1).cpu().numpy())
                    per["node"].append((v * (not_self - is_edge)).sum(1).cpu().numpy())
                    per["edge_ids"].append(meta["edge_ids"][rows.cpu(), cols.cpu()].numpy())
            n_targets += rows.numel()
    return {"pmf": hist / max(n_targets, 1), "nhead": nhead, "window": window, "max_dist": max_dist,
            "n_targets": n_targets, **{k: np.concatenate(v) for k, v in per.items()}}


def shap_pass(ds, lit, ds_test, idx, collate, device, remove="both"):
    """remove="both": a removed context edge shows <MASK> identity + hidden sign (default, the
    figure); remove="sign": only its sign is hidden, identity stays -- removes exactly what
    production's sign-only edge token carried."""
    model = lit.model
    mask_id = int(getattr(lit.cfg.model, "mask_id", 1))
    records = {"bwd": {1: [], 2: []}, "fwd": {1: [], 2: []}}
    gaps, n_inst = [], 0
    with torch.no_grad():
        for input_ids, labels, am, sign_ids, meta in batches(ds_test, idx, collate, 32, device):
            rows, cols = (labels != lit.ignore_index).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            x, s = apply_eval_transforms(lit, input_ids, sign_ids, labels, meta, device, ds_test)
            S = x.shape[1]
            jobs_x, jobs_s, jobs_r, jobs_meta, insts = [], [], [], [], []
            for k in range(rows.numel()):
                r, i = int(rows[k]), int(cols[k])
                valid = [(dn, hop, i + off) for dn, hop, off in FEATURES if 0 <= i + off < S and bool(am[r, i + off])]
                if not valid:
                    continue
                inst = len(insts)
                insts.append((valid, int(meta["edge_ids"][r, i])))
                for bm in range(2 ** len(valid)):
                    sub = frozenset(f for f in range(len(valid)) if (bm >> f) & 1)
                    xv, sv = x[r].clone(), s[r].clone()
                    for f, (_, _, jpos) in enumerate(valid):
                        if f not in sub:
                            sv[jpos] = SIGN_NA
                            if remove == "both":
                                xv[jpos] = mask_id
                    jobs_x.append(xv); jobs_s.append(sv); jobs_r.append(r); jobs_meta.append((inst, sub, i))
            if not jobs_x:
                continue
            probs = torch.softmax(model(torch.stack(jobs_x), torch.stack(jobs_s), attention_mask=am[jobs_r]), -1)[:, :, 1]
            val = defaultdict(dict)
            for jdx, (inst, sub, i) in enumerate(jobs_meta):
                val[inst][sub] = float(probs[jdx, i])
            for inst, (valid, eid) in enumerate(insts):
                phi = shapley_values(val[inst], len(valid))
                gaps.append(phi.sum() - (val[inst][frozenset(range(len(valid)))] - val[inst][frozenset()]))
                for f, (dn, hop, _) in enumerate(valid):
                    records[dn][hop].append((abs(float(phi[f])), eid))
                n_inst += 1
    rows_out = []
    for dn in ("fwd", "bwd"):
        for hop in (1, 2):
            vals = np.array([v for v, _ in records[dn][hop]])
            eids = np.array([e for _, e in records[dn][hop]])
            uniq, inv = np.unique(eids, return_inverse=True)
            cm = np.bincount(inv, weights=vals) / np.bincount(inv)
            rows_out.append({"dataset": ds, "direction": dn, "hop": hop, "mean_abs_shap": float(vals.mean()),
                             "cluster_se": float(cm.std(ddof=1) / np.sqrt(len(uniq))), "n_obs": len(vals),
                             "n_clusters": len(uniq)})
    return {"summary_rows": rows_out, "n_instances": n_inst, "max_efficiency_gap": float(np.max(np.abs(gaps)))}


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    os.makedirs(CACHE_DIR, exist_ok=True)
    for ds in args.datasets:
        cache = os.path.join(CACHE_DIR, f"{ds}.pkl")
        if ds not in DATASETS or os.path.exists(cache):  # '--datasets none' = only rebuild the CSVs
            continue
        t0 = time.time()
        lit, ds_test, idx, collate, ckpt = load(ds, device)
        att = attention_pass(ds, lit, ds_test, idx, collate, device)
        shp = shap_pass(ds, lit, ds_test, idx, collate, device)
        pickle.dump({"attention": att, "shap": shp, "ckpt": ckpt}, open(cache, "wb"))
        print(f"{ds}: targets={att['n_targets']:,} shap instances={shp['n_instances']:,} "
              f"max efficiency gap={shp['max_efficiency_gap']:.1e} ({time.time() - t0:.0f}s)", flush=True)

    res = {ds: pickle.load(open(os.path.join(CACHE_DIR, f"{ds}.pkl"), "rb")) for ds in DATASETS
           if os.path.exists(os.path.join(CACHE_DIR, f"{ds}.pkl"))}
    os.makedirs(OUT_DIR, exist_ok=True)
    if PANEL_A_DATASET in res:
        a = res[PANEL_A_DATASET]["attention"]
        write_csv(os.path.join(OUT_DIR, "attndir_panelA_headgrid.csv"),
                  [{"dataset": PANEL_A_DATASET, "layer": LAYER, "head": h, "d": d,
                    "mass": float(a["pmf"][LAYER, h, d + a["max_dist"]])}
                   for h in range(a["nhead"]) for d in range(-a["window"], a["window"] + 1)])
    summ, se, shap_rows = [], [], []
    for ds in DATASETS:
        if ds not in res:
            continue
        a = res[ds]["attention"]
        summ.append({"dataset": ds, "layer": LAYER, "nhead": a["nhead"], "forward": float(a["fwd"].mean()),
                     "backward": float(a["bwd"].mean()), "node": float(a["node"].mean()),
                     "edge": float(a["edge"].mean()),
                     "self": float(1 - a["fwd"].mean() - a["bwd"].mean())})
        row = {"dataset": ds}
        for cat in ("forward", "backward", "node", "edge"):
            m, s = cluster_robust_se_single(a["edge_ids"], a[{"forward": "fwd", "backward": "bwd"}.get(cat, cat)])
            row[f"{cat}_mean"], row[f"{cat}_se"] = m, s
        se.append(row)
        for name, (x, y) in (("fwd-bwd", ("fwd", "bwd")), ("node-edge", ("node", "edge"))):
            m, s, lo, hi, nc, wp = cluster_robust_se_diff(a["edge_ids"], a[x], a[y])
            print(f"{ds:15s} {name:9s} {m:+.4f} 95% CI [{lo:+.4f},{hi:+.4f}] "
                  f"{'significant' if lo > 0 or hi < 0 else 'n.s.'} (Wilcoxon p={wp:.1e})")
        shap_rows += res[ds]["shap"]["summary_rows"]
    write_csv(os.path.join(OUT_DIR, "attndir_panelBC_summary.csv"), summ)
    write_csv(os.path.join(OUT_DIR, "attndir_panelBC_se.csv"), se)
    write_csv(os.path.join(OUT_DIR, "shap_edge_directionality.csv"), shap_rows)


if __name__ == "__main__":
    main()
