"""Extract step for Ablation B (paper enumeration; formerly "Ablation C" in code/
data before the paper's A/C gap was fixed to A/B) -- curated functional-aggregator
sweep, computed fresh on the post-migration LocalAttn4 checkpoint
(E32_PY314_LOCALATTN4).

**Updated 2026-08-04 (post-migration rebuild + candidate-set refinement, per the
user's call):** two changes from the original ~39-function sweep. (1) Checkpoint
repointed from E27 (pre-migration) to E32_PY314_LOCALATTN4 (post-migration,
correctly-configured LocalAttn4 -- E31 turned out to be full attention, see
CLAUDE.md "Current SOTA"). (2) Candidate set restricted to CURATED_MODELS below:
only functions depending on the walk's predicted probability q alone (none of
run_posthoc.py's position/length-dependent forms), theoretically simple and
easily explained (mean, log-probability, certainty, entropy, Fisher-information/
inverse-variance), with one-sided (1-q)-only forms dropped entirely rather than
just deprioritized. Includes two new registry entries added this session
(func_fisher_power, func_maxprob_power) -- see run_posthoc.py::_func_registry
"Group 12" for their definitions/rationale.

Reads the real sweep run via run_posthoc.py --artifacts aggregator --agg-models
<CURATED_MODELS> --run-id ablationB_e32, parsing each model's saved summary.txt
under outputs/<ds>/<run>/posthoc/ablationB_e32/aggregator/<model>/summary.txt
(that on-disk run-id/directory was itself renamed 2026-08-05 from ablationC_e32
to ablationB_e32 to match the paper's letter -- it's just a tag, not tied to any
particular checkpoint).

Purely a parsing/aggregation step -- the actual sweep was run once via
run_posthoc.py (aggregator-only, reusing already-cached val/test prediction
pkls from the func_logit_power posthoc pass, no GPU inference re-run needed).
"""
import csv
import os
import re

OUT_CSV = "aaai2027/figure_data/ablationB_full_sweep.csv"

RUN_DIRS = {
    "bitcoin-alpha":   "E32_PY314_LOCALATTN4_20260804-225948",
    "bitcoin-otc":     "E32_PY314_LOCALATTN4_20260804-231122",
    "epinions":        "E32_PY314_LOCALATTN4_20260804-225948",
    "wiki-elec":       "E32_PY314_LOCALATTN4_20260804-232140",
    "wiki-rfa":        "E32_PY314_LOCALATTN4_20260804-232239",
    "slashdot090221":  "E32_PY314_LOCALATTN4_20260804-225948",
}
RUN_ID = "ablationB_e32"
CURATED_MODELS = [
    "func_uniform", "func_conf_power", "func_conf_exp", "func_conf_cert",
    "func_conf_logit", "func_logq_power", "func_logit_power",
    "func_entropy_power", "func_entropy_exp", "func_fisher_power", "func_maxprob_power",
]


def parse_summary(path):
    text = open(path).read()
    test_auc = float(re.search(r"Test\s+AUC:\s*([0-9.]+)", text).group(1))
    train_auc = float(re.search(r"Train AUC:\s*([0-9.]+)", text).group(1))
    return train_auc, test_auc


def main():
    rows = []
    for ds, run_dir in RUN_DIRS.items():
        agg_dir = f"outputs/{ds}/{run_dir}/posthoc/{RUN_ID}/aggregator"
        for model in CURATED_MODELS:
            summary_path = f"{agg_dir}/{model}/summary.txt"
            if not os.path.exists(summary_path):
                print(f"  ✗ missing: {summary_path}")
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
