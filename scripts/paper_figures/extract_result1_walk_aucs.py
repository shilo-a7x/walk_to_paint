"""Extract step for Result 1's SOTA table (tab:result1) -- \\method's two rows
(full attention, LocalAttn4), post-migration.

Source: each dataset's already-completed posthoc run (func_logit_power aggregator,
CLAUDE.md "Posthoc aggregation") --
  full attn:  outputs/<ds>/E31_PY314_MIGRATION_*/posthoc/*/aggregator/func_logit_power/summary.txt
  LocalAttn4: outputs/<ds>/E32_PY314_LOCALATTN4_*/posthoc/*/aggregator/func_logit_power/summary.txt
No new posthoc runs triggered here -- this only reads existing summary.txt files (all 12 exist
except slashdot090221's E32 side, pending training; that cell is left blank/marked pending until
its posthoc completes, not fabricated).

SE via the shared Hanley-McNeil closed form (hanley_mcneil.py), using each dataset's real
test-set n_pos/n_neg from aaai2027/figure_data/test_set_counts.csv (same source Ablation A
already uses) -- not re-derived per run, since it's the same canonical test split every time.
"""
import csv
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hanley_mcneil import auc_se

DATASETS = ["bitcoin-alpha", "bitcoin-otc", "epinions", "wiki-elec", "wiki-rfa", "slashdot090221"]
TAG = {"full": "E31_PY314_MIGRATION", "local": "E32_PY314_LOCALATTN4"}
COUNTS_CSV = "aaai2027/figure_data/test_set_counts.csv"
OUT_CSV = "aaai2027/figure_data/result1_walk_aucs.csv"

_AUC_RE = re.compile(r"Test\s+AUC:\s+([0-9.]+)\s+\((\d+)\s+edges\)")


def find_summary(ds, tag):
    pattern = f"outputs/{ds}/{tag}_*/posthoc/*/aggregator/func_logit_power/summary.txt"
    matches = glob.glob(pattern)
    if not matches:
        return None
    return sorted(matches)[-1]


def parse_summary(path):
    text = open(path).read()
    m = _AUC_RE.search(text)
    if not m:
        return None
    return float(m.group(1)), int(m.group(2))


def main():
    counts = {r["dataset"]: r for r in csv.DictReader(open(COUNTS_CSV))}

    rows = []
    for ds in DATASETS:
        for variant, tag in TAG.items():
            path = find_summary(ds, tag)
            if path is None:
                print(f"{ds:16s} {variant:6s} -- PENDING (no posthoc summary yet)")
                rows.append({"dataset": ds, "variant": variant, "auc": "", "se": "", "n_edges": ""})
                continue
            parsed = parse_summary(path)
            if parsed is None:
                print(f"{ds:16s} {variant:6s} -- FAILED to parse {path}")
                continue
            auc, n_edges = parsed
            n_pos, n_neg = int(counts[ds]["n_pos"]), int(counts[ds]["n_neg"])
            assert n_pos + n_neg == n_edges, f"{ds}: n_edges {n_edges} != counts {n_pos + n_neg}"
            se = auc_se(auc, n_pos, n_neg)
            print(f"{ds:16s} {variant:6s} AUC={auc:.4f} SE={se:.4f}  ({path})")
            rows.append({"dataset": ds, "variant": variant, "auc": f"{auc:.4f}",
                         "se": f"{se:.4f}", "n_edges": n_edges})

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "variant", "auc", "se", "n_edges"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
