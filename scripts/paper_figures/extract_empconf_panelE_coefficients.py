"""Extract step for Empirical Confirmation Panel E -- pooled logistic-regression
coefficients (node4_zscored spec) for the two GNN baselines, relabeled with Panel A's
position-index notation.

Source: outputs/lead4c_srctgt_export/results/fit_results_all6.csv (unzipped once from
outputs/lead4c_srctgt_export.zip). Filters to spec=node4_zscored, dataset=POOLED,
model in {GINEConv, SiGAT} (raw SiGAT -- see CLAUDE.md's Baselines note), terms
{src_out, src_in, tgt_out, tgt_in} -- confirmed by direct inspection this session that
node4_zscored yields exactly these 4 terms (no twohop/other terms, those only appear
under the atomic spec).

Term relabeling matches Panel A's schematic: src_out -> H_out(-1), src_in -> H_in(-1),
tgt_out -> H_out(1), tgt_in -> H_in(1).
"""
import csv
import os

IN_CSV = "outputs/lead4c_srctgt_export/results/fit_results_all6.csv"
OUT_CSV = "aaai2027/figure_data/empconf_panelE_coefficients.csv"

MODELS = ["GINEConv", "SiGAT"]
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
            if r["spec"] != "node4_zscored" or r["dataset"] != "POOLED":
                continue
            if r["model"] not in MODELS or r["term"] not in TERM_DISPLAY:
                continue
            rows.append({
                "model": r["model"],
                "term": r["term"],
                "display_term": TERM_DISPLAY[r["term"]],
                "beta": float(r["beta"]),
                "se_robust": float(r["se_robust"]),
                "p_fdr": float(r["p_fdr"]),
                "significant": float(r["p_fdr"]) < 0.05,
                "n": int(r["n"]),
            })

    expected = len(MODELS) * len(TERM_DISPLAY)
    assert len(rows) == expected, f"expected {expected} rows, got {len(rows)}"

    rows.sort(key=lambda r: (TERM_ORDER.index(r["term"]), MODELS.index(r["model"])))

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["model", "term", "display_term", "beta",
                                             "se_robust", "p_fdr", "significant", "n"])
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"saved {OUT_CSV} ({len(rows)} rows)")
    for r in rows:
        sig = "*" if r["significant"] else "n.s."
        print(f"  {r['model']:<10}{r['display_term']:<12}{r['beta']:>8.3f}  {sig}")


if __name__ == "__main__":
    main()
