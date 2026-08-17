"""
Reproduces the PER-DATASET (not pooled) 4-term node-entropy regression from the
shipped `../data/joined_table.pkl`.

Model fit, separately for EACH of the 6 datasets and EACH of the 4 models (24
independent logistic regressions total, no data pooled across datasets):

    logit(P(correct)) = const + b1*src_out + b2*src_in + b3*tgt_out + b4*tgt_in

So each (dataset, model) pair gets its own intercept + 4 slopes -- 6 separate sets
of coefficients per model, not one shared/pooled slope across datasets. See
README.md for full context (what the models/terms mean, what "correct" is, etc).

Usage:
    python reproduce.py --joined-table ../data/joined_table.pkl --out-dir ../results
"""
import argparse
import os

import pandas as pd

from regression_lib import run_node4_fits, run_node4_fits_zscored

ALL_6_DATASETS = ["bitcoin-alpha", "bitcoin-otc", "epinions", "wiki-elec", "wiki-rfa", "slashdot090221"]
TERMS = ["const", "src_out", "src_in", "tgt_out", "tgt_in"]


def to_wide(df):
    """Long (one row per dataset/model/term) -> wide (one row per dataset/model,
    one column per term) for easy at-a-glance reading."""
    piv = df.pivot_table(index=["dataset", "model"], columns="term", values="beta", aggfunc="first")
    piv = piv.reindex(columns=TERMS)
    piv = piv.reset_index()
    return piv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joined-table", default="../data/joined_table.pkl")
    ap.add_argument("--out-dir", default="../results")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    joined = pd.read_pickle(args.joined_table)

    df_raw = run_node4_fits(joined, ALL_6_DATASETS)
    df_raw = df_raw[df_raw["pooled"] == False]  # noqa: E712 -- per-dataset fits only, drop the pooled rows

    df_z = run_node4_fits_zscored(joined, ALL_6_DATASETS)
    df_z = df_z[df_z["pooled"] == False]  # noqa: E712

    long_raw = os.path.join(args.out_dir, "node4_per_dataset_long.csv")
    long_z = os.path.join(args.out_dir, "node4_per_dataset_long_zscored.csv")
    df_raw.to_csv(long_raw, index=False)
    df_z.to_csv(long_z, index=False)

    wide_raw = to_wide(df_raw)
    wide_z = to_wide(df_z)
    wide_raw_path = os.path.join(args.out_dir, "node4_per_dataset_wide.csv")
    wide_z_path = os.path.join(args.out_dir, "node4_per_dataset_wide_zscored.csv")
    wide_raw.to_csv(wide_raw_path, index=False)
    wide_z.to_csv(wide_z_path, index=False)

    print(f"wrote {long_raw} / {long_z} (full stats: beta, se_robust, p, p_fdr, n, ...)")
    print(f"wrote {wide_raw_path} / {wide_z_path} (compact: 24 rows x [dataset, model, {', '.join(TERMS)}])")
    print(f"\n{len(df_raw) // len(TERMS)} (dataset, model) fits, {len(TERMS)} coefficients each "
          f"(intercept + 4 node-entropy terms).")


if __name__ == "__main__":
    main()
