"""Combine Panel B (MI decay) and Panel C (GNN entropy heatmap) into ONE image
so LaTeX treats them as a single float and can't separate them on the page.

Pure combination step: loads the two already-rendered PNGs (from
plot_empconf_panelB_mi_decay_linegraph.py and
plot_empconf_panelC_gnn_entropy_heatmap.py) and stacks them vertically with
(a)/(b) labels. Re-run this after re-plotting either panel.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

PANEL_B = "aaai2027/figures/empconf_panelB_mi_decay_linegraph.png"
PANEL_C = "aaai2027/figures/empconf_panelC_gnn_entropy_heatmap.png"
OUT_PNG = "aaai2027/figures/empconf_panels_bc_combined.png"


def main():
    img_b = mpimg.imread(PANEL_B)
    img_c = mpimg.imread(PANEL_C)

    # common width in inches; each row's height follows from its own image's aspect ratio
    width_in = 7.0
    h_b = width_in * img_b.shape[0] / img_b.shape[1]
    h_c = width_in * img_c.shape[0] / img_c.shape[1]

    fig = plt.figure(figsize=(width_in, h_b + h_c))
    gs = fig.add_gridspec(2, 1, height_ratios=[h_b, h_c], hspace=0.03)

    ax_b = fig.add_subplot(gs[0])
    ax_b.imshow(img_b)
    ax_b.axis("off")
    ax_b.text(0.0, 1.0, "(a)", transform=ax_b.transAxes, fontsize=11, fontweight="bold",
              va="top", ha="left")

    ax_c = fig.add_subplot(gs[1])
    ax_c.imshow(img_c)
    ax_c.axis("off")
    ax_c.text(0.0, 1.0, "(b)", transform=ax_c.transAxes, fontsize=11, fontweight="bold",
              va="top", ha="left")

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
