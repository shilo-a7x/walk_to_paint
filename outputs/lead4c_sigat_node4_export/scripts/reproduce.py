"""
Reproduces the PER-DATASET (not pooled) 4-term node-entropy regression for SiGAT
only, from the shipped `../data/joined_table_sigat.pkl`.

Fit, separately for EACH of the 6 datasets (6 independent logistic regressions,
no data pooled across datasets):

    logit(P(correct)) = const + b1*src_out + b2*src_in + b3*tgt_out + b4*tgt_in

See README.md for full context (what SiGAT is, what the terms mean, what
"correct" is, and the dataset patterns found in these coefficients).

Usage:
    python reproduce.py --joined-table ../data/joined_table_sigat.pkl --out-dir ../results
"""
import argparse
import os

import pandas as pd

from regression_lib import run_node4_fits, run_node4_fits_zscored

ALL_6_DATASETS = ["bitcoin-alpha", "bitcoin-otc", "epinions", "wiki-elec", "wiki-rfa", "slashdot090221"]
TERMS = ["const", "src_out", "src_in", "tgt_out", "tgt_in"]


def to_wide(df):
    piv = df.pivot_table(index=["dataset", "model"], columns="term", values="beta", aggfunc="first")
    piv = piv.reindex(columns=TERMS)
    return piv.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joined-table", default="../data/joined_table_sigat.pkl")
    ap.add_argument("--out-dir", default="../results")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    joined = pd.read_pickle(args.joined_table)

    df_raw = run_node4_fits(joined, ALL_6_DATASETS)
    df_raw = df_raw[df_raw["pooled"] == False]  # noqa: E712 -- per-dataset fits only

    df_z = run_node4_fits_zscored(joined, ALL_6_DATASETS)
    df_z = df_z[df_z["pooled"] == False]  # noqa: E712

    long_raw = os.path.join(args.out_dir, "sigat_node4_per_dataset_long.csv")
    long_z = os.path.join(args.out_dir, "sigat_node4_per_dataset_long_zscored.csv")
    df_raw.to_csv(long_raw, index=False)
    df_z.to_csv(long_z, index=False)

    wide_raw = to_wide(df_raw)
    wide_z = to_wide(df_z)
    wide_raw_path = os.path.join(args.out_dir, "sigat_node4_per_dataset_wide.csv")
    wide_z_path = os.path.join(args.out_dir, "sigat_node4_per_dataset_wide_zscored.csv")
    wide_raw.to_csv(wide_raw_path, index=False)
    wide_z.to_csv(wide_z_path, index=False)

    print(f"wrote {long_raw} / {long_z}")
    print(f"wrote {wide_raw_path} / {wide_z_path}")
    print(f"\n{len(df_raw) // len(TERMS)} datasets x {len(TERMS)} coefficients (intercept + 4 node-entropy terms).")


if __name__ == "__main__":
    main()
