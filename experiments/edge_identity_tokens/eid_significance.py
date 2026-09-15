"""Phase 2 (+ reused for Phase 3a) of plan-eid-multiseed-thesis.md: paired significance
testing on EID's 10-seed multiseed results, modeled directly on
`scripts/paper_figures/table1_paired_significance.py` (one-sided paired Wilcoxon
signed-rank test, `scipy.stats.wilcoxon`, across the 10 shared seeds -- same statistical
convention used everywhere else in this project's significance claims).

Two use modes, both driven by the same underlying machinery:

  --mode vs_production (Phase 2, internal sanity check only, not a thesis deliverable):
    EID's 10-seed no-ablation numbers vs. production's own existing 10-seed local-attention
    numbers (outputs/<ds>/MULTISEED_s<seed>_local_*, seed 42 = E32_PY314_LOCALATTN4).
    Production's side needs zero new runs -- already on disk from the WSDM multiseed
    campaign.

  --mode ablation (Phase 3a): one of EID's own ablations (mask_context_edges,
    mask_context_sign_only) vs. EID's own no-ablation baseline, paired on the same seed --
    mirrors exactly how production's corrected abl:masknode/abl:maskedge were tested
    (aaai2027/PAPER_CLOSEOUT_LOG.md, 2026-08-23 entry).

Requires experiments/edge_identity_tokens/run_eid_multiseed.py's campaign to have produced
posthoc/multiseed_agg/aggregator/func_logit_power/summary.txt for every (dataset, condition,
seed) cell being compared -- missing seeds are reported, not silently dropped.

Usage:
  .venv/bin/python experiments/edge_identity_tokens/eid_significance.py --mode vs_production
  .venv/bin/python experiments/edge_identity_tokens/eid_significance.py --mode ablation \
      --ablation mask_context_edges
"""
import argparse
import csv
import glob
import os
import re
import sys

import numpy as np
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import experiments.edge_identity_tokens.run_eid_multiseed as m  # noqa: E402

DATASETS = m.DATASETS
SEEDS = [42] + m.NEW_SEEDS  # 42, 43..51

# Production's own 10-seed local-attention numbers (WSDM multiseed campaign) -- the
# internal-sanity-check comparator for --mode vs_production. Same tag/glob convention as
# scripts/paper_figures/table1_paired_significance.py's pewter_seed_auc(variant="local").
PRODUCTION_SEED42_TAG = "E32_PY314_LOCALATTN4"
_AUC_RE = re.compile(r"Test\s+AUC:\s+([0-9.]+)\s+\((\d+)\s+edges\)")


def _read_summary_auc(exp_dir, run_id):
    summaries = glob.glob(os.path.join(exp_dir, "posthoc", run_id, "aggregator",
                                        "func_logit_power", "summary.txt"))
    for path in sorted(summaries):
        m_ = _AUC_RE.search(open(path).read())
        if m_:
            return float(m_.group(1))
    return None


def production_seed_auc(ds, seed):
    if seed == 42:
        pattern = f"outputs/{ds}/{PRODUCTION_SEED42_TAG}_*"
    else:
        pattern = f"outputs/{ds}/MULTISEED_s{seed}_local_*"
    matches = sorted(glob.glob(pattern))
    for run_dir in reversed(matches):
        auc = _read_summary_auc_glob(run_dir)
        if auc is not None:
            return auc
    return None


def _read_summary_auc_glob(run_dir):
    for path in glob.glob(os.path.join(run_dir, "posthoc", "*", "aggregator",
                                        "func_logit_power", "summary.txt")):
        m_ = _AUC_RE.search(open(path).read())
        if m_:
            return float(m_.group(1))
    return None


def eid_exp_dir(ds, cond_tag, seed):
    """Resolve the on-disk exp_dir for an EID (dataset, condition, seed) cell, matching
    run_eid_multiseed.py's own resolution logic exactly (backfilled dirs for seed 42
    where one exists, EID_MULTISEED_s<seed>_<cond_tag>_* for everything else, including
    wiki-rfa's seed-42 no-ablation baseline which had no salvageable backfill dir)."""
    if seed == 42:
        if cond_tag == "noablation" and ds not in m.BASELINE_NO_BACKFILL:
            return m.BASELINE_BACKFILL_DIRS.get(ds)
        if cond_tag != "noablation":
            for cond_flag, tag in m.CONDITIONS:
                if tag == cond_tag and cond_flag is not None:
                    return m.ablation_backfill_dir(ds, cond_flag)
    return m.latest_exp_dir(ds, f"EID_MULTISEED_s{seed}_{cond_tag}")


def eid_seed_auc(ds, cond_tag, seed):
    """Every EID multiseed job (backfill or fresh train) is posthoc'd under the same
    fixed run_id (run_eid_multiseed.RUN_ID = "multiseed_agg"), so this never needs the
    glob-any-run_id fallback production_seed_auc uses."""
    exp_dir = eid_exp_dir(ds, cond_tag, seed)
    if exp_dir is None or not os.path.isdir(exp_dir):
        return None
    return _read_summary_auc(exp_dir, m.RUN_ID)


def paired_test(ds, arr_a, arr_b, label_a, label_b):
    diff = arr_a - arr_b
    stat, p = wilcoxon(diff, alternative="greater")
    print(f"{ds}: {label_a} mean={arr_a.mean():.4f}  {label_b} mean={arr_b.mean():.4f}  "
          f"median diff={np.median(diff):+.4f}  wins={int((diff > 0).sum())}/{len(diff)}  "
          f"one-sided Wilcoxon p={p:.4g}")
    return {"dataset": ds, f"{label_a}_mean": arr_a.mean(), f"{label_b}_mean": arr_b.mean(),
            "median_diff": np.median(diff), "n_wins": int((diff > 0).sum()), "p_value": p}


def run_vs_production():
    rows = []
    for ds in DATASETS:
        eid_vals = [eid_seed_auc(ds, "noablation", s) for s in SEEDS]
        prod_vals = [production_seed_auc(ds, s) for s in SEEDS]
        if any(v is None for v in eid_vals):
            missing = [s for s, v in zip(SEEDS, eid_vals) if v is None]
            print(f"{ds}: EID MISSING seeds {missing}")
            continue
        if any(v is None for v in prod_vals):
            missing = [s for s, v in zip(SEEDS, prod_vals) if v is None]
            print(f"{ds}: production MISSING seeds {missing}")
            continue
        rows.append(paired_test(ds, np.array(eid_vals), np.array(prod_vals), "eid", "production"))
    return rows, "eid_vs_production"


def run_ablation(ablation):
    valid_tags = [tag for flag, tag in m.CONDITIONS if flag == ablation]
    if not valid_tags:
        raise SystemExit(f"unknown ablation {ablation!r}; choose from "
                          f"{[f for f, t in m.CONDITIONS if f]}")
    cond_tag = valid_tags[0]
    rows = []
    for ds in DATASETS:
        base_vals = [eid_seed_auc(ds, "noablation", s) for s in SEEDS]
        abl_vals = [eid_seed_auc(ds, cond_tag, s) for s in SEEDS]
        if any(v is None for v in base_vals):
            missing = [s for s, v in zip(SEEDS, base_vals) if v is None]
            print(f"{ds}: noablation MISSING seeds {missing}")
            continue
        if any(v is None for v in abl_vals):
            missing = [s for s, v in zip(SEEDS, abl_vals) if v is None]
            print(f"{ds}: {ablation} MISSING seeds {missing}")
            continue
        # one-sided: baseline > ablation (ablation should hurt, if it's a real signal source)
        rows.append(paired_test(ds, np.array(base_vals), np.array(abl_vals), "noablation", ablation))
    return rows, f"noablation_vs_{ablation}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["vs_production", "ablation"], required=True)
    ap.add_argument("--ablation", choices=["mask_context_edges", "mask_context_sign_only"])
    args = ap.parse_args()

    if args.mode == "vs_production":
        rows, tag = run_vs_production()
    else:
        if not args.ablation:
            raise SystemExit("--mode ablation requires --ablation")
        rows, tag = run_ablation(args.ablation)

    if not rows:
        print("\nNo complete datasets -- nothing to summarize (campaign likely still running).")
        return

    out_csv = f"logs/eid_multiseed/significance_{tag}.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved {out_csv}")
    n_sig = sum(1 for r in rows if r["p_value"] < 0.05)
    print(f"significant (p<0.05) on {n_sig}/{len(rows)} datasets")


if __name__ == "__main__":
    main()
