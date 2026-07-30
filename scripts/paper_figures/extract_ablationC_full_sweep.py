"""Extract step for Ablation C -- full functional-aggregator sweep (all ~39
func_* forms registered in run_posthoc.py, NO lgbm/attention variants per your
call), computed fresh on the current production LocalAttn4 checkpoint
(E27) for all 6 datasets.

Unlike the earlier 2-bar weighted-vs-mean check (extract_ablationC_aggregator.py,
kept for provenance but superseded by this), this reads the real sweep run
via run_posthoc.py --artifacts aggregator --agg-models <all 39 func_ names>
--run-id funcsweep_20260727, parsing each model's saved summary.txt under
outputs/<ds>/<run>/posthoc/funcsweep_20260727/aggregator/<model>/summary.txt.

Purely a parsing/aggregation step -- the actual sweep was run once via
run_posthoc.py (CPU-only, no retraining, no GPU: aggregator fitting reads
already-cached val/test prediction pkls).
"""
import csv
import os
import re

OUT_CSV = "aaai2027/figure_data/ablationC_full_sweep.csv"

RUN_DIRS = {
    "bitcoin-alpha":   "E27_NOHARD_EDGECOVER_LOCALATTN4_20260719-121955",
    "bitcoin-otc":     "E27_NOHARD_EDGECOVER_LOCALATTN4_20260719-121955",
    "epinions":        "E27_NOHARD_EDGECOVER_LOCALATTN4_20260719-121955",
    "wiki-elec":       "E27_NOHARD_EDGECOVER_LOCALATTN4_20260719-122848",
    "wiki-rfa":        "E27_NOHARD_EDGECOVER_LOCALATTN4_20260719-123214",
    "slashdot090221":  "E27_NOHARD_EDGECOVER_LOCALATTN4_20260719-121955",
}
RUN_ID = "funcsweep_20260727"


def parse_summary(path):
    text = open(path).read()
    test_auc = float(re.search(r"Test\s+AUC:\s*([0-9.]+)", text).group(1))
    train_auc = float(re.search(r"Train AUC:\s*([0-9.]+)", text).group(1))
    return train_auc, test_auc


def main():
    rows = []
    for ds, run_dir in RUN_DIRS.items():
        agg_dir = f"outputs/{ds}/{run_dir}/posthoc/{RUN_ID}/aggregator"
        models = sorted(os.listdir(agg_dir))
        for model in models:
            summary_path = f"{agg_dir}/{model}/summary.txt"
            if not os.path.exists(summary_path):
                continue
            train_auc, test_auc = parse_summary(summary_path)
            rows.append({"dataset": ds, "model": model, "train_auc": train_auc, "test_auc": test_auc})
        best = max((r for r in rows if r["dataset"] == ds), key=lambda r: r["test_auc"])
        uniform = next(r for r in rows if r["dataset"] == ds and r["model"] == "func_uniform")
        print(f"{ds}: best={best['model']} ({best['test_auc']:.4f})  "
              f"uniform={uniform['test_auc']:.4f}  gain={best['test_auc']-uniform['test_auc']:+.4f}")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "model", "train_auc", "test_auc"])
        w.writeheader()
        w.writerows(rows)
    print(f"saved {OUT_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
