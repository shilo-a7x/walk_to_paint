"""Extract step for the new Attention Directionality figure, Panel A -- example
per-head signed attention-mass grid, layer 0 only, LocalAttn4 variant.

Source: outputs/attention_directionality/attention_directionality_<ds>_local_result.pkl
(already computed by scripts/attention_directionality.py -- this just reslices the
cached pmf array, no new inference).

Dataset choice: bitcoin-alpha (2026-08-04 call) -- one of the three 6-dataset-suite
datasets with nhead=4 (bitcoin-alpha, bitcoin-otc, slashdot090221; epinions has 8,
wiki-elec/wiki-rfa have 2), and the dataset already used as the flagship example
throughout this investigation. Restricted to layer 0 only, matching the
"focus on layer 0" convention already established this session for interpretability.

Only writes d in [-window, window] -- Panel B's plot re-ranges the x-axis to the
window instead of showing a wider range with a dashed boundary line, so the
wider tail isn't needed here (see plot script for the rationale).
"""
import csv
import os
import pickle

DATASET = "bitcoin-alpha"
IN_PKL = f"outputs/attention_directionality/attention_directionality_{DATASET}_local_result.pkl"
OUT_CSV = "aaai2027/figure_data/attndir_panelA_headgrid.csv"
LAYER = 0


def main():
    with open(IN_PKL, "rb") as f:
        res = pickle.load(f)

    nhead = res["nhead"]
    window = res["window"]
    max_dist = res["max_dist"]
    pmf = res["pmf"]  # [nlayers, nhead, 2*max_dist+1]

    rows = []
    for h in range(nhead):
        for d in range(-window, window + 1):
            mass = float(pmf[LAYER, h, d + max_dist])
            rows.append({"dataset": DATASET, "layer": LAYER, "head": h, "d": d, "mass": mass})

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "layer", "head", "d", "mass"])
        w.writeheader()
        w.writerows(rows)
    print(f"dataset={DATASET} nhead={nhead} window={window} n_targets={res['n_targets']:,}")
    print(f"wrote {OUT_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
