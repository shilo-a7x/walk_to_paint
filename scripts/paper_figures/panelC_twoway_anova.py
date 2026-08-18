"""Two-way ANOVA per dataset for Panel C (SiGAT AUC vs. source/target entropy) --
professor's C-ter ask: "please also have a statistical test per dataset on the effect
of source and target entropy and their joined effect (a simple two way ANOVA - you can
use the cross validations as multiple measurements)".

Design: for each dataset, response = AUC computed within one of Panel C's existing 4x4
entropy bins (src_out entropy x tgt_in entropy), one observation per (seed, cell) --
the 10 splits (seeds 42-51) ARE the repeated measurements the professor asked for,
exactly as instructed ("use the cross validations as multiple measurements"), not a
single point estimate per cell. Two-way ANOVA (Type II sums of squares):
    auc ~ C(src_bin) + C(tgt_bin) + C(src_bin):C(tgt_bin)
Reports the F-statistic and p-value for each of the 3 terms (source main effect, target
main effect, interaction), per dataset.

Reuses the entropy-join machinery already built for the multiseed SiGAT heatmap
(extract_multiseed_entropy_heatmaps.py's binned_auc_grid, same VARIANT="out_in" axes as
Panel C, same MIN_CELL_N=30 threshold) -- but pulls per-seed raw SiGAT predictions from
the cache already built for Panel D's rebuild
(outputs/cache/sigat_raw_predictions/<ds>__seed<N>.pkl, scripts/paper_figures/
extract_empconf_panelD_signagreement_auc.py) instead of refitting SiGAT's
LogisticRegression a second time -- the entropy join itself (collect_model_records) is
cheap, no fitting, so this whole script runs off already-cached data with no GPU/refit
cost at all.
"""
import csv
import os
import pickle
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "baselines"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

from scripts.paper_figures.extract_multiseed_entropy_heatmaps import binned_auc_grid  # noqa: E402
from scripts.lead4_entropy_heterogeneity import build_sign_dicts, collect_model_records  # noqa: E402
from scripts.node_mi_structural_embedding import DATASET_CONFIGS, load_edges_canonical  # noqa: E402

DATASETS = ["bitcoin-alpha", "bitcoin-otc", "epinions", "wiki-elec", "wiki-rfa", "slashdot090221"]
SEEDS = [42] + list(range(43, 52))
VARIANT = "out_in"  # same axes as Panel C: src_ent = H_out(u), tgt_ent = H_in(v)
N_BINS = 4
MIN_CELL_N = 30  # matches Panel C's own single-split threshold

RAW_CACHE_DIR = os.path.join(ROOT, "outputs", "cache", "sigat_raw_predictions")
OUT_CSV = "aaai2027/figure_data/panelC_twoway_anova.csv"


def _ds_key(ds_name):
    return "slashdot" if ds_name == "slashdot090221" else ds_name


def load_raw_seed(ds, seed):
    path = os.path.join(RAW_CACHE_DIR, f"{ds}__seed{seed}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    anova_rows = []
    for ds in DATASETS:
        edges_list = load_edges_canonical(DATASET_CONFIGS[_ds_key(ds)]["ds_name"])
        sign_dicts = build_sign_dicts(edges_list)

        long_rows = []
        for seed in SEEDS:
            rec = load_raw_seed(ds, seed)
            if rec is None or not len(rec["u"]):
                print(f"{ds} seed={seed}: MISSING raw predictions")
                continue
            test_uvyp = list(zip(rec["u"], rec["v"], rec["y"], rec["p"]))
            entry = collect_model_records(test_uvyp, sign_dicts, VARIANT, min_total=1)
            if entry is None:
                print(f"{ds} seed={seed}: too few usable edges after entropy join")
                continue
            auc_grid, n_grid, edges = binned_auc_grid(
                entry["src_ent"], entry["tgt_ent"], entry["y"], entry["p"], N_BINS, MIN_CELL_N)
            for i in range(N_BINS):
                for j in range(N_BINS):
                    if np.isfinite(auc_grid[i, j]):
                        long_rows.append({"seed": seed, "src_bin": i, "tgt_bin": j,
                                           "auc": auc_grid[i, j], "n": int(n_grid[i, j])})

        df = pd.DataFrame(long_rows)
        n_cells_present = df.groupby(["src_bin", "tgt_bin"]).size()
        print(f"\n{ds}: {len(df)} (seed, cell) observations, "
              f"{(n_cells_present == len(SEEDS)).sum()}/{N_BINS * N_BINS} cells complete across all 10 seeds")

        model = ols("auc ~ C(src_bin) + C(tgt_bin) + C(src_bin):C(tgt_bin)", data=df).fit()
        table = sm.stats.anova_lm(model, typ=2)

        row = {"dataset": ds, "n_obs": len(df)}
        for term, label in [("C(src_bin)", "src_F"), ("C(tgt_bin)", "tgt_F"),
                             ("C(src_bin):C(tgt_bin)", "interaction_F")]:
            row[label] = table.loc[term, "F"]
            row[label.replace("_F", "_p")] = table.loc[term, "PR(>F)"]
        anova_rows.append(row)
        print(f"  src main effect:  F={row['src_F']:.2f}  p={row['src_p']:.4g}")
        print(f"  tgt main effect:  F={row['tgt_F']:.2f}  p={row['tgt_p']:.4g}")
        print(f"  interaction:      F={row['interaction_F']:.2f}  p={row['interaction_p']:.4g}")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "n_obs", "src_F", "src_p", "tgt_F", "tgt_p",
                                           "interaction_F", "interaction_p"])
        w.writeheader()
        w.writerows(anova_rows)
    print(f"\nsaved {OUT_CSV}")


if __name__ == "__main__":
    main()
