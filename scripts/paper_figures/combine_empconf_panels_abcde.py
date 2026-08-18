"""Combine Panels A-E into ONE image -- the reworked 5-panel Empirical
Confirmation / Figure 1 (checklist #12, 2026-08-04 rework), superseding the
3-panel combine (combine_empconf_panels_abc.py).

Panel A: schematic (position-index notation). Panel B: directed+undirected NMI
decay. Panel C: GNN AUC-vs-entropy heatmap (reused unchanged). Panel D: 4-way
sign-agreement AUC bars. Panel E: pooled regression-coefficient bars.

Layout: row 1 = A (small schematic) + B (line plot, wider); row 2 = C (full
width -- it's already 2 rows x 6 datasets); row 3 = D + E (both grouped bar
charts, side by side). Adjust ROW_HEIGHT_IN / column width ratios below if the
real panel proportions need rebalancing once viewed at actual paper width.

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
PANEL_D = "aaai2027/figures/empconf_panelD_signagreement_auc_perdataset.png"
PANEL_E = "aaai2027/figures/empconf_panelE_coefficients.png"
OUT_PNG = "aaai2027/figures/empconf_panels_abcde_combined.png"

WIDTH_IN = 7.2
# row heights are set relative to each row's tallest panel (by aspect ratio at
# that panel's share of WIDTH_IN) -- see main() for the actual computation.
ROW1_A_FRAC = 0.34    # fraction of WIDTH_IN given to panel A in row 1 (rest -> B)
ROW3_SPLIT = 0.5       # D/E even split in row 3


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
    a_w = WIDTH_IN * ROW1_A_FRAC
    b_w = WIDTH_IN * (1 - ROW1_A_FRAC)
    _, a_ar = _aspect(PANEL_A)
    _, b_ar = _aspect(PANEL_B)
    row1_h = max(a_w * a_ar, b_w * b_ar)

    _, c_ar = _aspect(PANEL_C)
    row2_h = WIDTH_IN * c_ar

    d_w = WIDTH_IN * ROW3_SPLIT
    e_w = WIDTH_IN * (1 - ROW3_SPLIT)
    _, d_ar = _aspect(PANEL_D)
    _, e_ar = _aspect(PANEL_E)
    row3_h = max(d_w * d_ar, e_w * e_ar)

    total_h = row1_h + row2_h + row3_h
    fig = plt.figure(figsize=(WIDTH_IN, total_h))
    gs = fig.add_gridspec(3, 1, height_ratios=[row1_h, row2_h, row3_h], hspace=0.06)

    gs_row1 = gs[0].subgridspec(1, 2, width_ratios=[ROW1_A_FRAC, 1 - ROW1_A_FRAC], wspace=0.03)
    _add_panel(fig, gs_row1[0], PANEL_A, "(A)")
    _add_panel(fig, gs_row1[1], PANEL_B, "(B)")

    _add_panel(fig, gs[1], PANEL_C, "(C)")

    gs_row3 = gs[2].subgridspec(1, 2, width_ratios=[ROW3_SPLIT, 1 - ROW3_SPLIT], wspace=0.06)
    _add_panel(fig, gs_row3[0], PANEL_D, "(D)")
    _add_panel(fig, gs_row3[1], PANEL_E, "(E)")

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight", pad_inches=0.05)
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
