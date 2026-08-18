"""Extract step for Empirical Confirmation Panel E -- per-dataset, multiseed (10-seed)
logistic-regression coefficients for SiGAT, relabeled with Panel A's position-index
notation. Rebuilt 2026-08-18 (per the professor's C-ter instruction, "In e, please do
stacked bar plots") to replace the earlier single-split pooled version.

Source: outputs/lead4c_sigat_multiseed_node4_export/results/aggregated_summary.csv -- one
row per (dataset, term, scale), already mean/std/significance-count across the 10 seeds
(42-51), z-scored slopes from a 4-term model fit independently per dataset per seed
(logit(P(correct)) ~ src_out + src_in + tgt_out + tgt_in, two-way cluster-robust SE,
BH-FDR within each fit's 4 terms). See that package's README.md for the full methodology.
No new computation here -- this only re-shapes and relabels an already-computed multiseed
result for plotting.

"Robust" (the statistical test backing each bar, per the same convention already used
throughout that export package and its pooled companion): a term is marked robust for a
given dataset if it is BH-FDR significant (q<0.05) in at least 8 of the 10 independently
fitted seeds, not just significant in the single aggregate. Non-robust terms are drawn
hatched in the plot -- this is a real per-seed significance count, not an eyeballed
point estimate.
"""
import csv
import os
import sys

IN_CSV = "outputs/lead4c_sigat_multiseed_node4_export/results/aggregated_summary.csv"
OUT_CSV = "aaai2027/figure_data/empconf_panelE_coefficients.csv"
ROBUST_THRESHOLD = 0.8  # matches the export package's own "robust" convention (>=8/10 seeds)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_style import DATASET_ORDER

TERM_DISPLAY = {
    "src_out": "H_out(-1)",
    "src_in": "H_in(-1)",
    "tgt_out": "H_out(1)",
    "tgt_in": "H_in(1)",
}
TERM_ORDER = ["src_out", "src_in", "tgt_out", "tgt_in"]


def main():
    rows = []
    with open(IN_CSV) as f:
        for r in csv.DictReader(f):
            if r["scale"] != "zscored" or r["term"] not in TERM_DISPLAY:
                continue
            if r["dataset"] not in DATASET_ORDER:
                continue
            frac_sig = float(r["frac_seeds_significant"])
            rows.append({
                "dataset": r["dataset"],
                "term": r["term"],
                "display_term": TERM_DISPLAY[r["term"]],
                "mean_beta": float(r["mean_beta"]),
                "std_beta": float(r["std_beta"]),
                "n_seeds": int(r["n_seeds"]),
                "n_seeds_significant": int(r["n_seeds_significant"]),
                "frac_seeds_significant": frac_sig,
                "robust": frac_sig >= ROBUST_THRESHOLD,
            })

    expected = len(DATASET_ORDER) * len(TERM_DISPLAY)
    assert len(rows) == expected, f"expected {expected} rows, got {len(rows)}"

    rows.sort(key=lambda r: (DATASET_ORDER.index(r["dataset"]), TERM_ORDER.index(r["term"])))

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["dataset", "term", "display_term", "mean_beta",
                                             "std_beta", "n_seeds", "n_seeds_significant",
                                             "frac_seeds_significant", "robust"])
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"saved {OUT_CSV} ({len(rows)} rows)")
    for r in rows:
        tag = "robust" if r["robust"] else f"{r['n_seeds_significant']}/{r['n_seeds']}"
        print(f"  {r['dataset']:<16}{r['display_term']:<12}{r['mean_beta']:>8.3f} ± {r['std_beta']:.3f}  {tag}")


if __name__ == "__main__":
    main()
