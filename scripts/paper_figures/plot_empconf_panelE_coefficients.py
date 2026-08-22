"""Plot step for Empirical Confirmation Panel E -- per-dataset, multiseed (10-seed) SiGAT
entropy-term coefficients, DIVERGING STACKED BARS. Rebuilt 2026-08-18 per the professor's
C-ter instruction ("In e, please do stacked bar plots") -- replaces the earlier pooled
grouped-bar version, which hid the per-dataset "which term dominates" pattern that is the
actual point of this panel (tgt_in bigger on the 4 smaller/sparser datasets, src_out bigger
on epinions/slashdot -- see the source export package's README section 4.3). Pure
rendering: reads aaai2027/figure_data/empconf_panelE_coefficients.csv (built by
extract_empconf_panelE_coefficients.py). Edit THIS file freely for color/style changes --
no recomputation needed.

**Layout, corrected same day per the user's direct catch** ("i thought that stacked
barplot still need to show same bars like before but devided to the datasets. aint it?
not 6 bars"): the x-axis is the SAME 4 term categories as the pre-rebuild panel
(H_out(-1), H_in(-1), H_out(1), H_in(1)) -- NOT one bar per dataset. Each term's single
bar is now divided into 6 stacked segments, one per dataset, so a reader compares the
same term across datasets within one bar instead of comparing terms within one dataset's
bar. This is also why dataset color belongs on the stack segments, not the x-tick text --
an earlier version colored the axis ticks when datasets were still the x-axis category;
now that datasets ARE the visual element being stacked, DATASET_COLORS finally has a
real role to play (segment fill color = dataset), which is what the user's original
"keep dataset colors consistent" ask was actually about.

Sign convention (unchanged from the prior version): raw z-scored beta, not negated.
Negative beta = higher entropy at that position associated with LOWER P(correct).
Stacking is diverging (negative segments stack downward from 0, positive stack upward)
since most (dataset, term) values are negative and H_out(1) hovers near zero for most
datasets -- a same-sign stack would misrepresent a near-zero segment as adding to the
harm.

Each segment carries a small error whisker (+-1 SD across the 10 seeds, centered on that
segment alone, not the cumulative stack) and is hatched if it is not "robust" (BH-FDR
significant in fewer than 8/10 seeds, per the extract script's own convention) -- the
statistical test backing this panel, same standard as the export package's own forest
plots.

Term order is the "standard" position-based order matching Panel A's schematic (-1/source
terms first, then +1/target terms; out before in within each position), not a
by-magnitude order.

**Gap fix, corrected same day (the first attempt was wrong).** An initial guess blamed a
border-rendering seam (double-drawn edges at each segment boundary) and removed the
segment borders -- that alone did NOT fix it, per the user's direct catch ("your no gap
fix didnt work"). The real cause, found by printing the actual computed bar boundaries:
`bottoms = np.where(betas >= 0, cum_pos, cum_neg + betas)` added `betas` to the negative
branch's bottom -- but matplotlib's `bar(bottom=B, height=H)` already spans
`[B, B+H]`, so `height=betas` alone carries the full descent; adding `betas` again to
`bottom` double-counted it, shifting every negative segment down by its own value.
Verified numerically before and after (see git history / CLAUDE.md): with the bug,
Bitcoin-alpha's `H_out(-1)` segment rendered as `[-1.96, -0.98]` instead of the correct
`[-0.98, 0.00]`; downstream segments then landed at arbitrary overlaps or gaps depending
on each pair's specific magnitudes -- which is also why only ONE gap was visible in the
first render (an overlap is invisible, since the later-drawn segment just paints over the
earlier one; only the one pair that happened to land as a genuine gap showed white space).
Fixed by removing the erroneous `+ betas`; the borders were also kept removed on solid
segments since that's a legitimate (if secondary) cleanup -- hatched ("not robust")
segments keep a thin edge since matplotlib needs `edgecolor` to draw the hatch pattern's
own lines.

**Hatch meaning, now stated explicitly** (user: "no explanation why some stacks are with
dashed-empty-fill"): added a legend entry via a proxy patch describing what the
diagonal-hatch/white-fill styling means (not robust: BH-FDR significant in fewer than
8 of the 10 seeds) -- it was previously only explained in this script's docstring and the
tex caption, not in the figure itself.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_style import DATASET_ORDER, DATASET_COLORS, DATASET_DISPLAY

IN_CSV = "aaai2027/figure_data/empconf_panelE_coefficients.csv"
OUT_PNG = "aaai2027/figures/empconf_panelE_coefficients.png"

# position-based order, matching Panel A's schematic: -1 (source) before +1 (target),
# out before in within each position.
TERM_ORDER = ["src_out", "src_in", "tgt_out", "tgt_in"]
TERM_DISPLAY = {"src_out": "H_out(-1)", "src_in": "H_in(-1)", "tgt_out": "H_out(1)", "tgt_in": "H_in(1)"}


def main():
    rows = list(csv.DictReader(open(IN_CSV)))
    data = {(r["dataset"], r["term"]): r for r in rows}

    x = np.arange(len(TERM_ORDER))
    width = 0.62
    fig, ax = plt.subplots(figsize=(7.4, 5.1))

    cum_pos = np.zeros(len(TERM_ORDER))
    cum_neg = np.zeros(len(TERM_ORDER))
    for ds in DATASET_ORDER:
        betas = np.array([float(data[(ds, term)]["mean_beta"]) for term in TERM_ORDER])
        ses = np.array([float(data[(ds, term)]["std_beta"]) for term in TERM_ORDER])
        robust = [data[(ds, term)]["robust"] == "True" for term in TERM_ORDER]
        # bottom = the cumulative total BEFORE this segment (matplotlib bar() spans
        # [bottom, bottom+height], so height=betas already carries the descent -- adding
        # betas here too double-counted it, shifting every negative segment down by its
        # own value and creating overlaps/gaps depending on the neighboring magnitudes.
        # This is the real bug behind the "gaps between stacks" the user caught -- the
        # earlier border-seam theory was wrong, verified numerically before this fix.
        bottoms = np.where(betas >= 0, cum_pos, cum_neg)
        color = DATASET_COLORS[ds]
        # no border on solid (robust) segments -- avoids the double-edge seam between
        # adjacent stacked segments; hatched (non-robust) segments keep a thin edge since
        # the hatch pattern itself is drawn in the edgecolor.
        colors = [color if r else "none" for r in robust]
        linewidths = [0.0 if r else 0.7 for r in robust]
        bars = ax.bar(x, betas, width, bottom=bottoms, color=colors, edgecolor=color,
                       linewidth=linewidths, label=DATASET_DISPLAY[ds], zorder=3)
        for i, (bar, r) in enumerate(zip(bars, robust)):
            if not r:
                bar.set_hatch("///")
                bar.set_facecolor("white")
            mid = bottoms[i] + betas[i] / 2
            ax.errorbar(x[i], mid, yerr=ses[i], color=color, capsize=2,
                         linewidth=0.8, zorder=4, fmt="none")
        cum_pos = np.where(betas >= 0, cum_pos + betas, cum_pos)
        cum_neg = np.where(betas < 0, cum_neg + betas, cum_neg)

    ax.axhline(0, color="black", linewidth=0.9, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([TERM_DISPLAY[t] for t in TERM_ORDER], fontsize=15)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylabel("stacked regression coefficients", fontsize=14)
    ax.set_title("Which entropy term hurts SiGAT, by dataset", fontsize=16)

    dataset_handles, dataset_labels = ax.get_legend_handles_labels()
    hatch_handle = Patch(facecolor="white", edgecolor="#555", hatch="///",
                          label="Not robust ($<$8/10 seeds sig.)")
    ax.legend(handles=dataset_handles + [hatch_handle], fontsize=12.5, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.52), frameon=False)
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
