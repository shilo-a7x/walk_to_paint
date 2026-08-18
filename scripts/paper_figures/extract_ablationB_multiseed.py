"""Ablation B (tab:ablationB) multiseed rebuild -- test AUC (mean +- std over 10 splits)
for Pewter (local attention) under all 11 aggregator weight functions. Replaces the
current single-split table (seed 42 / E32_PY314_LOCALATTN4 only, no error bars).

Confirmed feasible without any new training/posthoc runs: run_multiseed_pewter.py's own
docstring states every multiseed posthoc run already computed all 11 registered
aggregator functions per seed (not just func_logit_power), and this is directly verified
on disk -- every MULTISEED_s<seed>_local_*/posthoc/multiseed_agg/aggregator/ directory
(seeds 43-51) and E32_PY314_LOCALATTN4's own posthoc/ablationB_e32/aggregator/ directory
(seed 42, the pre-migration multiseed backfill run) both have all 11 func_*/summary.txt
files, all 6 datasets. Pure aggregation of existing files, no computation.
"""
import csv
import glob
import os
import re

DATASETS = ["bitcoin-alpha", "bitcoin-otc", "epinions", "wiki-elec", "wiki-rfa", "slashdot090221"]
SEEDS = [42] + list(range(43, 52))
SEED42_TAG = "E32_PY314_LOCALATTN4"

# order matches run_multiseed_pewter.py's AGG_FUNCS and the tex table's row order exactly
AGG_FUNCS = [
    "func_uniform", "func_conf_power", "func_conf_exp", "func_conf_cert",
    "func_conf_logit", "func_logq_power", "func_logit_power",
    "func_entropy_power", "func_entropy_exp", "func_fisher_power", "func_maxprob_power",
]
FUNC_DISPLAY = {
    "func_uniform": "Uniform (mean)", "func_conf_power": "$q^b$", "func_conf_exp": "$\\exp(bq)$",
    "func_conf_cert": "$|q-0.5|^b$", "func_conf_logit": "$\\mathrm{sig}(b(q-0.5))$",
    "func_logq_power": "$(-\\log q)^b$", "func_logit_power": "$|\\mathrm{logit}(q)|^b$ (default)",
    "func_entropy_power": "$(\\ln2-H(q))^b$", "func_entropy_exp": "$\\exp(-aH(q))$",
    "func_fisher_power": "$(q(1-q))^{-b}$ (Fisher-info)", "func_maxprob_power": "$\\max(q,1-q)^b$",
}

OUT_CSV = "aaai2027/figure_data/ablationB_multiseed.csv"
_AUC_RE = re.compile(r"Test\s+AUC:\s+([0-9.]+)\s+\((\d+)\s+edges\)")


def find_summary(ds, seed, func):
    if seed == 42:
        run_glob = f"outputs/{ds}/{SEED42_TAG}_*"
        posthoc_id = "ablationB_e32"
    else:
        run_glob = f"outputs/{ds}/MULTISEED_s{seed}_local_*"
        posthoc_id = "multiseed_agg"
    run_dirs = sorted(glob.glob(run_glob))
    for run_dir in reversed(run_dirs):
        path = os.path.join(run_dir, "posthoc", posthoc_id, "aggregator", func, "summary.txt")
        if os.path.exists(path):
            return path
    return None


def seed_auc(ds, seed, func):
    path = find_summary(ds, seed, func)
    if path is None:
        return None
    m = _AUC_RE.search(open(path).read())
    return float(m.group(1)) if m else None


def main():
    import statistics

    results = {}  # (ds, func) -> (mean, std, n_seeds)
    for ds in DATASETS:
        for func in AGG_FUNCS:
            vals = []
            for seed in SEEDS:
                v = seed_auc(ds, seed, func)
                if v is None:
                    print(f"{ds} {func} seed={seed}: MISSING")
                    continue
                vals.append(v)
            if vals:
                mean = statistics.mean(vals)
                std = statistics.stdev(vals) if len(vals) > 1 else 0.0
                results[(ds, func)] = (mean, std, len(vals))
            else:
                results[(ds, func)] = (None, None, 0)

    rows_out = []
    for ds in DATASETS:
        means = {func: results[(ds, func)][0] for func in AGG_FUNCS if results[(ds, func)][0] is not None}
        ranked = sorted(means, key=lambda f: -means[f])
        best = ranked[0] if len(ranked) > 0 else None
        second = ranked[1] if len(ranked) > 1 else None
        for func in AGG_FUNCS:
            mean, std, n_seeds = results[(ds, func)]
            rows_out.append({
                "dataset": ds, "func": func, "display": FUNC_DISPLAY[func],
                "mean_auc": mean, "std_auc": std, "n_seeds": n_seeds,
                "rank": "best" if func == best else ("second" if func == second else ""),
            })

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "func", "display", "mean_auc", "std_auc",
                                           "n_seeds", "rank"])
        w.writeheader()
        w.writerows(rows_out)
    print(f"saved {OUT_CSV}")

    # spread check (matches the tex caption's "full spread under 0.0022 AUC" claim)
    for ds in DATASETS:
        vals = [results[(ds, func)][0] for func in AGG_FUNCS if results[(ds, func)][0] is not None]
        if vals:
            print(f"{ds}: spread = {max(vals) - min(vals):.4f}  (n_funcs={len(vals)})")


if __name__ == "__main__":
    main()
