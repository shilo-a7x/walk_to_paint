"""Combine Panels A-E into ONE image -- the reworked 5-panel Empirical
Confirmation / Figure 1 (checklist #12, 2026-08-04 rework), superseding the
3-panel combine (combine_empconf_panels_abc.py).

Panel A: schematic (position-index notation). Panel B: directed+undirected NMI
decay. Panel C: GNN AUC-vs-entropy heatmap (reused unchanged). Panel D: 4-way
sign-agreement AUC bars. Panel E: pooled regression-coefficient bars.

**Layout changed 2026-08-05 (user call): single-column figure, not figure*.**
This paper's single column is ~3.31in wide (aaai2027.sty: textwidth=7.0in,
columnsep=0.375in -> (7.0-0.375)/2). WIDTH_IN below is set to match that
directly, so \\includegraphics[width=\\linewidth] in the tex displays this PNG
at its native size instead of shrinking a wider image down (which is what made
panels unreadably small before). All five panels are now stacked one-per-row
(no more A+B or D+E side-by-side pairs) so each panel gets the FULL column
width instead of half of it -- the "make it bigger without going full-width"
trick: trade width you don't have for height, which is free in a float.

Pure combination step: loads the five already-rendered PNGs and lays them out
in a gridspec so LaTeX treats them as a single float. Re-run this after
re-plotting any of the five panels.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

PANEL_A = "aaai2027/figures/empconf_panelA_schematic.png"
PANEL_B = "aaai2027/figures/empconf_panelB_mi_decay_linegraph.png"
PANEL_C = "aaai2027/figures/empconf_panelC_gnn_entropy_heatmap.png"
PANEL_D = "aaai2027/figures/empconf_panelD_signagreement_auc.png"
PANEL_E = "aaai2027/figures/empconf_panelE_coefficients.png"
OUT_PNG = "aaai2027/figures/empconf_panels_abcde_combined.png"

WIDTH_IN = 3.31  # single-column width (aaai2027.sty: (7.0in - 0.375in) / 2)
PANELS = [(PANEL_A, "(a)"), (PANEL_B, "(b)"), (PANEL_C, "(c)"), (PANEL_D, "(d)"), (PANEL_E, "(e)")]


def _aspect(path):
    im = mpimg.imread(path)
    return im, im.shape[0] / im.shape[1]  # height/width


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
