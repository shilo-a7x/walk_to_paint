"""Does layer-0 attention treat positive-sign and negative-sign context edges
equally, and does that change by direction (forward/backward) or distance
(hop 1 vs hop 2)?

Companion to scripts/attention_directionality.py (which splits attention mass by
node/edge role and forward/backward direction, but not by the edge's own sign) --
reuses its exact checkpoint pins, loading, and recording-layer machinery. Local
attention (LocalAttn4) production checkpoints only, per request -- these are what
Table 1 / Figure 4 already use.

For every masked target-edge occurrence, at layer 0 only: sum the attention mass
landing on POSITIVE-sign context edge tokens vs NEGATIVE-sign context edge tokens
(mean over heads), further split by forward (context position after the target)
vs backward (before it), and by hop distance (1 = immediate neighbor, 2 = second
neighbor within the +-2-hop LocalAttn4 window -- 2 token positions per hop, same
convention as ATTENTION_MATH.md / the Shapley-directionality figure).

Statistics: per-target-edge cluster means (not per-head/per-occurrence raw values)
for a paired one-sided Wilcoxon test on (positive mass - negative mass), same
"cluster = target edge" convention already used in scripts/attention_directionality.py's
own write-up.

Inference only -- no retraining, no checkpoint modification.

Usage:
  .venv/bin/python scripts/attention_by_edge_sign.py [--datasets all] [--max-samples 20000]
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(ROOT))

torch.backends.mha.set_fastpath_enabled(False)

from attention_directionality import load_model_and_dataset, LOCAL_RUN_INFO, ALL_DATASETS

MAX_SAMPLES_DEFAULT = 20000
OUT_CSV = os.path.join(os.path.dirname(ROOT), "aaai2027", "figure_data", "attention_by_edge_sign.csv")


def analyse_dataset(ds_name, max_samples, batch_size, device):
    print(f"\n{'=' * 80}\nDATASET: {ds_name}  [local, layer 0]\n{'=' * 80}")
    t0 = time.time()

    bundle = load_model_and_dataset(ds_name, "local", stage="test")
    if bundle is None:
        return None

    model = bundle["model"].to(device).eval()
    ds = bundle["dataset"]
    ignore_index = bundle["ignore_index"]
    id2class = ds.id2class  # [vocab_size], token_id -> sign class (0/1) or ignore_index

    n = len(ds)
    if max_samples is not None and n > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, max_samples, replace=False).tolist()
        ds_run = torch.utils.data.Subset(ds, idx)
    else:
        ds_run = ds

    loader = torch.utils.data.DataLoader(
        ds_run, batch_size=batch_size, shuffle=False, collate_fn=bundle["collate"]
    )

    print(f"  N={n:,} test walks (using {len(ds_run):,}), ckpt={os.path.basename(bundle['ckpt_path'])}")

    # Per-target-edge cluster accumulators (one row per masked-target occurrence)
    # columns: pos_fwd, pos_bwd, neg_fwd, neg_bwd, pos_hop1, pos_hop2, neg_hop1, neg_hop2
    rows_out = []

    with torch.no_grad():
        for batch in loader:
            input_ids, labels, attention_mask, metadata = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            target_mask = labels != ignore_index
            trows, tcols = target_mask.nonzero(as_tuple=True)
            if trows.numel() == 0:
                continue

            _ = model(input_ids, attention_mask=attention_mask)

            trows_d = trows.to(device)
            tcols_d = tcols.to(device)
            S = input_ids.shape[1]
            am = attention_mask.float()  # [B, S]
            j_idx = torch.arange(S, device=device)

            # per-position sign class, from the (unmasked-context) input ids
            id2class_dev = id2class.to(device)
            pos_class_per_batch = id2class_dev[input_ids]  # [B, S], class or ignore_index
            is_pos_sign = pos_class_per_batch == 1  # [B, S] bool
            is_neg_sign = pos_class_per_batch == 0  # [B, S] bool

            dist = (j_idx.unsqueeze(0) - tcols_d.unsqueeze(1))  # [M, S], d = j - i (target row's own dist)
            fwd = dist > 0
            bwd = dist < 0
            hop = torch.ceil(dist.abs().float() / 2.0)  # 2 token positions per hop
            is_hop1 = hop == 1
            is_hop2 = hop == 2

            pos_sign_M = is_pos_sign[trows_d]  # [M, S]
            neg_sign_M = is_neg_sign[trows_d]  # [M, S]

            layer0 = model.transformer.layers[0]
            attn = layer0.last_attn_weights  # [B, nhead, S, S]
            sel = attn[trows_d, :, tcols_d, :]  # [M, nhead, S]
            sel = sel.mean(dim=1)  # mean over heads -> [M, S]
            valid = sel * am[trows_d]  # [M, S]

            pos_fwd = (valid * (pos_sign_M & fwd).float()).sum(dim=1)
            pos_bwd = (valid * (pos_sign_M & bwd).float()).sum(dim=1)
            neg_fwd = (valid * (neg_sign_M & fwd).float()).sum(dim=1)
            neg_bwd = (valid * (neg_sign_M & bwd).float()).sum(dim=1)
            pos_hop1 = (valid * (pos_sign_M & is_hop1).float()).sum(dim=1)
            pos_hop2 = (valid * (pos_sign_M & is_hop2).float()).sum(dim=1)
            neg_hop1 = (valid * (neg_sign_M & is_hop1).float()).sum(dim=1)
            neg_hop2 = (valid * (neg_sign_M & is_hop2).float()).sum(dim=1)

            # Availability counts: how many attendable (am==1) positive/negative-sign
            # context positions actually exist for this target, so raw mass can be
            # normalized to a PER-TOKEN attention rate -- raw mass alone conflates
            # "attention favors positive edges" with "positive edges are just far more
            # numerous in context" (this dataset's train pool is ~94% positive).
            attendable = am[trows_d].bool()  # [M, S]
            n_pos_avail = (pos_sign_M & attendable).float().sum(dim=1)
            n_neg_avail = (neg_sign_M & attendable).float().sum(dim=1)

            stacked = torch.stack(
                [pos_fwd, pos_bwd, neg_fwd, neg_bwd, pos_hop1, pos_hop2, neg_hop1, neg_hop2,
                 n_pos_avail, n_neg_avail], dim=1
            ).cpu().numpy()
            rows_out.append(stacked)

    if not rows_out:
        print("  no target occurrences found")
        return None

    M = np.concatenate(rows_out, axis=0)  # [n_targets, 10]
    cols = ["pos_fwd", "pos_bwd", "neg_fwd", "neg_bwd", "pos_hop1", "pos_hop2", "neg_hop1", "neg_hop2",
            "n_pos_avail", "n_neg_avail"]
    means = M.mean(axis=0)
    pos_total = M[:, 0] + M[:, 1]  # pos_fwd + pos_bwd, per target
    neg_total = M[:, 2] + M[:, 3]
    n_pos_avail = M[:, 8]
    n_neg_avail = M[:, 9]

    # Raw mass (conflates "attention favors this sign" with "this sign is more common
    # in context") vs per-token rate (mass / count of that-sign tokens actually
    # available to attend to) -- only the second answers "does attention treat a
    # positive and a negative edge equally, token for token". Restrict the per-token
    # rate to targets that actually have >=1 of BOTH signs available, so the ratio is
    # well-defined and not an artifact of one-sided availability.
    both_avail = (n_pos_avail > 0) & (n_neg_avail > 0)
    rate_pos = np.divide(pos_total, n_pos_avail, out=np.full_like(pos_total, np.nan), where=n_pos_avail > 0)
    rate_neg = np.divide(neg_total, n_neg_avail, out=np.full_like(neg_total, np.nan), where=n_neg_avail > 0)

    diff_raw = pos_total - neg_total
    stat, p_raw = wilcoxon(diff_raw, alternative="two-sided") if np.any(diff_raw != 0) else (None, 1.0)

    diff_rate = (rate_pos - rate_neg)[both_avail]
    if diff_rate.size > 0 and np.any(diff_rate != 0):
        stat_r, p_rate = wilcoxon(diff_rate, alternative="two-sided")
    else:
        p_rate = 1.0

    print(f"  n_targets={len(M):,}  elapsed={time.time() - t0:.1f}s")
    print(f"  mean mass: pos_fwd={means[0]:.4f} pos_bwd={means[1]:.4f} "
          f"neg_fwd={means[2]:.4f} neg_bwd={means[3]:.4f}")
    print(f"  mean mass: pos_hop1={means[4]:.4f} pos_hop2={means[5]:.4f} "
          f"neg_hop1={means[6]:.4f} neg_hop2={means[7]:.4f}")
    print(f"  RAW total pos mass={pos_total.mean():.4f}  neg mass={neg_total.mean():.4f}  "
          f"diff={diff_raw.mean():+.4f}  p={p_raw:.4g}")
    print(f"  availability: mean #pos-sign tokens attendable={n_pos_avail.mean():.2f}  "
          f"#neg-sign tokens attendable={n_neg_avail.mean():.2f}  "
          f"(both-signs-present on {100 * both_avail.mean():.1f}% of targets)")
    print(f"  PER-TOKEN rate (mass/count, both-signs-present targets only): "
          f"pos={np.nanmean(rate_pos[both_avail]):.4f}  neg={np.nanmean(rate_neg[both_avail]):.4f}  "
          f"diff={np.nanmean(diff_rate):+.4f}  p={p_rate:.4g}")

    return {
        "dataset": ds_name, "n_targets": len(M),
        **{c: float(v) for c, v in zip(cols, means)},
        "pos_total_mean": float(pos_total.mean()), "neg_total_mean": float(neg_total.mean()),
        "raw_diff_pos_minus_neg": float(diff_raw.mean()), "raw_p_two_sided": float(p_raw),
        "n_pos_avail_mean": float(n_pos_avail.mean()), "n_neg_avail_mean": float(n_neg_avail.mean()),
        "frac_both_signs_present": float(both_avail.mean()),
        "per_token_rate_pos": float(np.nanmean(rate_pos[both_avail])) if both_avail.any() else float("nan"),
        "per_token_rate_neg": float(np.nanmean(rate_neg[both_avail])) if both_avail.any() else float("nan"),
        "per_token_rate_diff": float(np.nanmean(diff_rate)) if diff_rate.size > 0 else float("nan"),
        "per_token_p_two_sided": float(p_rate),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--max-samples", type=int, default=MAX_SAMPLES_DEFAULT)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    datasets = ALL_DATASETS if args.datasets == "all" else args.datasets.split(",")

    results = []
    for ds_name in datasets:
        r = analyse_dataset(ds_name, args.max_samples, args.batch_size, args.device)
        if r is not None:
            results.append(r)

    if results:
        os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
        with open(OUT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        print(f"\nsaved {OUT_CSV}")


if __name__ == "__main__":
    main()
