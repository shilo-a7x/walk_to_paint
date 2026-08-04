"""Combine Panels A-D into ONE image -- the Attention Directionality figure
(2026-08-04, separate from Figure 1's Empirical Confirmation panels and Figure 2's
schematic).

Panel A: example per-head signed attention-mass grid (bitcoin-alpha, layer 0,
PEWTER, x-axis re-ranged to the window). Panel B: cross-dataset forward/backward
mass (PEWTER, layer 0). Panel C: cross-dataset node/edge mass (PEWTER, layer 0).
Panel D: PEWTER's own binned entropy-vs-AUC heatmap (added 2026-08-04), directly
comparable to Empirical Confirmation Panel C's GNN version.

Layout: row 1 = A (full width, wide per-head grid); row 2 = B + C side by side;
row 3 = D (full width, 1x6 heatmap row).

Pure combination step: loads the four already-rendered PNGs and lays them out in a
gridspec. Re-run this after re-plotting any of the four panels.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

PANEL_A = "aaai2027/figures/attndir_panelA_headgrid.png"
PANEL_B = "aaai2027/figures/attndir_panelB_direction.png"
PANEL_C = "aaai2027/figures/attndir_panelC_nodeedge.png"
PANEL_D = "aaai2027/figures/attndir_panelD_pewter_entropy_heatmap.png"
OUT_PNG = "aaai2027/figures/attndir_panels_abcd_combined.png"

WIDTH_IN = 7.2


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
    _, a_ar = _aspect(PANEL_A)
    row1_h = WIDTH_IN * a_ar

    b_w = WIDTH_IN * 0.5
    c_w = WIDTH_IN * 0.5
    _, b_ar = _aspect(PANEL_B)
    _, c_ar = _aspect(PANEL_C)
    row2_h = max(b_w * b_ar, c_w * c_ar)

    _, d_ar = _aspect(PANEL_D)
    row3_h = WIDTH_IN * d_ar

    total_h = row1_h + row2_h + row3_h
    fig = plt.figure(figsize=(WIDTH_IN, total_h))
    gs = fig.add_gridspec(3, 1, height_ratios=[row1_h, row2_h, row3_h], hspace=0.05)

    _add_panel(fig, gs[0], PANEL_A, "(a)")

    gs_row2 = gs[1].subgridspec(1, 2, width_ratios=[0.5, 0.5], wspace=0.03)
    _add_panel(fig, gs_row2[0], PANEL_B, "(b)")
    _add_panel(fig, gs_row2[1], PANEL_C, "(c)")

    _add_panel(fig, gs[2], PANEL_D, "(d)")

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight", pad_inches=0.05)
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
