"""
Reproduces every regression result in this package from the shipped
`../data/joined_table.pkl` (already-built per-edge feature table -- see README.md
for what that file contains and how it was built).

Two model families x two dataset scopes = 4 fits (each run twice, raw + z-scored):
  1. Full 6-term atomic model,  all 6 datasets pooled + per-dataset
  2. Minimal 2-term model,      all 6 datasets pooled + per-dataset
  3. Full 6-term atomic model,  wiki-elec + wiki-rfa only, pooled + per-dataset
  4. Minimal 2-term model,      wiki-elec + wiki-rfa only, pooled + per-dataset

Usage:
    python reproduce.py --joined-table ../data/joined_table.pkl --out-dir ../results_reproduced
"""
import argparse
import os

import pandas as pd

from regression_lib import (
    run_atomic_fits, run_atomic_fits_zscored, run_srctgt_fits, run_srctgt_fits_zscored,
    run_node4_fits, run_node4_fits_zscored,
)

ALL_6_DATASETS = ["bitcoin-alpha", "bitcoin-otc", "epinions", "wiki-elec", "wiki-rfa", "slashdot090221"]
WIKI_DATASETS = ["wiki-elec", "wiki-rfa"]


def run_scope(joined, datasets, tag, out_dir):
    df_a = run_atomic_fits(joined, datasets)
    df_az = run_atomic_fits_zscored(joined, datasets)
    df_st = run_srctgt_fits(joined, datasets)
    df_stz = run_srctgt_fits_zscored(joined, datasets)
    df_n4 = run_node4_fits(joined, datasets)
    df_n4z = run_node4_fits_zscored(joined, datasets)
    df = pd.concat([df_a, df_az, df_st, df_stz, df_n4, df_n4z], ignore_index=True)
    csv_path = os.path.join(out_dir, f"fit_results_{tag}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[{tag}] wrote {csv_path} ({len(df)} rows: {len(df_a)} atomic + {len(df_az)} atomic_zscored + "
          f"{len(df_st)} srctgt2 + {len(df_stz)} srctgt2_zscored + "
          f"{len(df_n4)} node4 + {len(df_n4z)} node4_zscored)")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joined-table", default="../data/joined_table.pkl")
    ap.add_argument("--out-dir", default="../results_reproduced")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    joined = pd.read_pickle(args.joined_table)

    run_scope(joined, ALL_6_DATASETS, "all6", args.out_dir)
    run_scope(joined, WIKI_DATASETS, "wiki_only", args.out_dir)

    print("\nDone. Compare against ../results/fit_results_all6.csv and "
          "../results/fit_results_wiki_only.csv -- should match to floating-point precision.")


if __name__ == "__main__":
    main()
