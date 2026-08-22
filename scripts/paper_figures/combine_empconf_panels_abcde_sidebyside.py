"""Combine Panels A-E, D+E-SIDE-BY-SIDE variant -- candidate for comparison against the
primary combine_empconf_panels_abcde.py (D over E, each its own full-width row).

Context: the primary (stacked D/E) layout made the combined Figure 2 too tall for a
single ACM sigconf column (aspect ratio ~2.24 at WIDTH_IN=7.2, i.e. ~7.5in tall once
scaled down to \\linewidth in the tex -- overflows the page). Reverting D/E to share one
row (like the pre-2026-08-18 layout) is the single biggest height lever (saves ~4in
pre-crop), but that's what caused the "D/E look tiny" complaint in the first place --
diagnosed then as an imshow width-binding-constraint issue (halving the row's width caps
rendered size regardless of height given).

This variant tries to have it both ways: D and E share row 3 side-by-side (recovering
the height), but their own source PNGs (plot_empconf_panelD_signagreement_auc.py's
plot_perdataset(), plot_empconf_panelE_coefficients.py) were regenerated with ~1.5x
larger internal fonts/markers specifically so they stay legible once physically
rendered at half-width instead of full width. Panel C was also shrunk (row height
3.65->3.0 per row) and combine-script padding tightened, same as the primary variant --
both changes apply globally to the same source images either combine script reads, not
duplicated here.

Run AFTER combine_empconf_panels_abcde.py's own panel regeneration steps (same PANEL_*
source PNGs, no separate rendering needed). Pure combination step, safe to re-run.
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
OUT_PNG = "aaai2027/figures/empconf_panels_abcde_combined_sidebyside.png"

WIDTH_IN = 7.2
ROW1_A_FRAC = 0.34
ROW3_D_FRAC = 0.5  # D and E split row 3 evenly


def _aspect(path):
    im = mpimg.imread(path)
    return im, im.shape[0] / im.shape[1]  # height/width


def _add_panel(fig, gs_cell, path, label, label_y=1.0, label_va="top"):
    img, _ = _aspect(path)
    ax = fig.add_subplot(gs_cell)
    ax.imshow(img)
    ax.axis("off")
    ax.text(0.0, label_y, label, transform=ax.transAxes, fontsize=11, fontweight="bold",
             va=label_va, ha="left")
    return ax


def main():
    a_w = WIDTH_IN * ROW1_A_FRAC
    b_w = WIDTH_IN * (1 - ROW1_A_FRAC)
    _, a_ar = _aspect(PANEL_A)
    _, b_ar = _aspect(PANEL_B)
    row1_h = max(a_w * a_ar, b_w * b_ar)

    _, c_ar = _aspect(PANEL_C)
    row2_h = WIDTH_IN * c_ar

    d_w = WIDTH_IN * ROW3_D_FRAC
    e_w = WIDTH_IN * (1 - ROW3_D_FRAC)
    _, d_ar = _aspect(PANEL_D)
    _, e_ar = _aspect(PANEL_E)
    row3_h = max(d_w * d_ar, e_w * e_ar)

    total_h = row1_h + row2_h + row3_h
    fig = plt.figure(figsize=(WIDTH_IN, total_h))
    gs = fig.add_gridspec(3, 1, height_ratios=[row1_h, row2_h, row3_h], hspace=0.03)

    gs_row1 = gs[0].subgridspec(1, 2, width_ratios=[ROW1_A_FRAC, 1 - ROW1_A_FRAC], wspace=0.03)
    _add_panel(fig, gs_row1[0], PANEL_A, "(A)")
    _add_panel(fig, gs_row1[1], PANEL_B, "(B)")

    _add_panel(fig, gs[1], PANEL_C, "(C)")

    gs_row3 = gs[2].subgridspec(1, 2, width_ratios=[ROW3_D_FRAC, 1 - ROW3_D_FRAC], wspace=0.03)
    _add_panel(fig, gs_row3[0], PANEL_D, "(D)", label_y=1.03, label_va="bottom")
    _add_panel(fig, gs_row3[1], PANEL_E, "(E)", label_y=1.03, label_va="bottom")

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
