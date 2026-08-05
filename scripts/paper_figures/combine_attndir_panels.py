"""Combine Panels A-C into ONE image -- the Attention Directionality figure
(2026-08-04, separate from Figure 1's Empirical Confirmation panels and Figure 2's
schematic).

Panel A: example per-head signed attention-mass grid (bitcoin-alpha, layer 0,
PEWTER, x-axis re-ranged to the window). Panel B: cross-dataset forward/backward
mass (PEWTER, layer 0). Panel C: cross-dataset node/edge mass (PEWTER, layer 0).

Panel D (PEWTER's own binned entropy-vs-AUC heatmap) was dropped from this figure
2026-08-05 (user call) -- its extract/plot scripts
(extract_attndir_panelD_pewter_entropy_heatmap.py,
plot_attndir_panelD_pewter_entropy_heatmap.py) are kept on disk but no longer
wired in here.

**Layout changed 2026-08-05 (user call): single-column figure, not figure*.**
This paper's single column is ~3.31in wide (aaai2027.sty: textwidth=7.0in,
columnsep=0.375in -> (7.0-0.375)/2). WIDTH_IN below matches that directly, so
\\includegraphics[width=\\linewidth] displays this PNG at its native size
instead of shrinking a wider image down. Panels B and C (previously
side-by-side) are now stacked full-width like A, one per row -- each panel
gets the full column width instead of half of it.

Pure combination step: loads the three already-rendered PNGs and lays them out in a
gridspec. Re-run this after re-plotting any of the three panels.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

PANEL_A = "aaai2027/figures/attndir_panelA_headgrid.png"
PANEL_B = "aaai2027/figures/attndir_panelB_direction.png"
PANEL_C = "aaai2027/figures/attndir_panelC_nodeedge.png"
OUT_PNG = "aaai2027/figures/attndir_panels_abc_combined.png"

WIDTH_IN = 3.31  # single-column width (aaai2027.sty: (7.0in - 0.375in) / 2)
PANELS = [(PANEL_A, "(a)"), (PANEL_B, "(b)"), (PANEL_C, "(c)")]


def _aspect(path):
    im = mpimg.imread(path)
    return im, im.shape[0] / im.shape[1]


def _add_panel(fig, gs_cell, path, label):
    img, _ = _aspect(path)
    ax = fig.add_subplot(gs_cell)
    ax.imshow(img)
    ax.axis("off")
    ax.text(0.0, 1.0, label, transform=ax.transAxes, fontsize=11, fontweight="bold",
             va="top", ha="left")
    return ax


def main():
    row_heights = []
    for path, _ in PANELS:
        _, ar = _aspect(path)
        row_heights.append(WIDTH_IN * ar)

    total_h = sum(row_heights)
    fig = plt.figure(figsize=(WIDTH_IN, total_h))
    gs = fig.add_gridspec(len(PANELS), 1, height_ratios=row_heights, hspace=0.08)

    for i, (path, label) in enumerate(PANELS):
        _add_panel(fig, gs[i], path, label)

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight", pad_inches=0.05)
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
