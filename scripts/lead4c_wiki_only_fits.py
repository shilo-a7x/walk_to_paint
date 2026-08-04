"""
Lead 4c wiki-only companion fits: reruns the full 6-term atomic model and the
minimal 2-term (src_out + tgt_in) model restricted to wiki-elec + wiki-rfa only,
pooled together (2 dataset fixed effects instead of 6). Same underlying fit
functions as the main script (lead4c_entropy_logit_regression.py) -- no new
statistics, just a different `datasets` subset passed to the existing
run_atomic_fits / run_atomic_fits_zscored / run_srctgt_fits / run_srctgt_fits_zscored.

Reads an already-built joined_table.pkl (from `lead4c_entropy_logit_regression.py
--mode compute`) -- does not recompute entropy features or reload predictions.

Usage:
    .venv/bin/python scripts/lead4c_wiki_only_fits.py \
        --joined-table outputs/lead4c_entropy_logit_regression_e27/joined_table.pkl \
        --out-dir outputs/lead4c_entropy_logit_regression_e27/wiki_only
"""
import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.lead4c_entropy_logit_regression import (
    run_atomic_fits, run_atomic_fits_zscored, run_srctgt_fits, run_srctgt_fits_zscored,
    run_node4_fits, run_node4_fits_zscored,
)

WIKI_DATASETS = ["wiki-elec", "wiki-rfa"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joined-table", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    joined = pd.read_pickle(args.joined_table)

    df_a = run_atomic_fits(joined, WIKI_DATASETS)
    df_az = run_atomic_fits_zscored(joined, WIKI_DATASETS)
    df_st = run_srctgt_fits(joined, WIKI_DATASETS)
    df_stz = run_srctgt_fits_zscored(joined, WIKI_DATASETS)
    df_n4 = run_node4_fits(joined, WIKI_DATASETS)
    df_n4z = run_node4_fits_zscored(joined, WIKI_DATASETS)

    df = pd.concat([df_a, df_az, df_st, df_stz, df_n4, df_n4z], ignore_index=True)
    fit_pkl = os.path.join(args.out_dir, "fit_results_wiki_only.pkl")
    fit_csv = os.path.join(args.out_dir, "fit_results_wiki_only.csv")
    df.to_pickle(fit_pkl)
    df.to_csv(fit_csv, index=False)
    print(f"wrote {fit_csv} ({len(df)} rows: {len(df_a)} atomic + {len(df_az)} atomic_zscored + "
          f"{len(df_st)} srctgt2 + {len(df_stz)} srctgt2_zscored + "
          f"{len(df_n4)} node4 + {len(df_n4z)} node4_zscored)")


if __name__ == "__main__":
    main()
