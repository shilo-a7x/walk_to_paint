"""Plot step for Empirical Confirmation Panel E -- pooled regression coefficients,
raw beta, grouped bars (GINEConv vs SiGAT) x 4 terms (Panel A's position-index
notation). Pure rendering: reads aaai2027/figure_data/empconf_panelE_coefficients.csv
(built by extract_empconf_panelE_coefficients.py). Edit THIS file freely for
color/style changes -- no recomputation needed.

Sign convention (per user's call): raw beta, not negated. Negative beta = higher
entropy at that position associated with LOWER P(correct); axis label states this
explicitly since the sign is otherwise easy to misread as "more positive = more
harmful". Non-significant bars (p_fdr >= 0.05) are drawn hatched/lighter.

SiGAT only (2026-08-04 call) -- GINEConv dropped from this figure, kept in the
appendix/baseline table instead; the extract CSV still has both models' rows
for that reuse.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IN_CSV = "aaai2027/figure_data/empconf_panelE_coefficients.csv"
OUT_PNG = "aaai2027/figures/empconf_panelE_coefficients.png"

MODELS = ["SiGAT"]  # GINEConv dropped from this figure 2026-08-04, kept in appendix/table only
MODEL_COLOR = {"GINEConv": "#4472c4", "SiGAT": "#c0392b"}
TERM_ORDER = ["H_out(-1)", "H_in(-1)", "H_out(1)", "H_in(1)"]


def main():
    rows = list(csv.DictReader(open(IN_CSV)))
    data = {(r["model"], r["display_term"]): r for r in rows}

    x = np.arange(len(TERM_ORDER))
    width = 0.5 if len(MODELS) == 1 else 0.35
    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    for i, model in enumerate(MODELS):
        betas = [float(data[(model, t)]["beta"]) for t in TERM_ORDER]
        ses = [float(data[(model, t)]["se_robust"]) for t in TERM_ORDER]
        sig = [data[(model, t)]["significant"] == "True" for t in TERM_ORDER]
        xpos = x + (i - (len(MODELS) - 1) / 2) * width
        colors = [MODEL_COLOR[model] if s else "none" for s in sig]
        edgecolors = [MODEL_COLOR[model]] * len(TERM_ORDER)
        hatches = [None if s else "///" for s in sig]
        bars = ax.bar(xpos, betas, width, yerr=ses, capsize=3,
                       color=colors, edgecolor=edgecolors, linewidth=1.3,
                       label=model, zorder=3)
        for bar, h in zip(bars, hatches):
            if h:
                bar.set_hatch(h)
                bar.set_facecolor("white")
                bar.set_alpha(0.85)

    ax.axhline(0, color="black", linewidth=0.9, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(TERM_ORDER, fontsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_ylabel("coefficient (z-scored, raw $\\beta$)\nneg. = higher entropy $\\to$ lower P(correct)", fontsize=9.5)
    ax.set_title("Which entropy term hurts SiGAT", fontsize=11)
    if len(MODELS) > 1:
        ax.legend()
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
