"""Combine Panel A (entropy asymmetry boxplot), Panel B (MI decay), and Panel
C (GNN entropy heatmap) into ONE image -- the full 3-panel Empirical
Confirmation figure (checklist #12), superseding the 2-panel B+C-only combine
(combine_empconf_panels_bc.py) now that Panel A is a real plot instead of a
text-only deferred slot.

Pure combination step: loads the three already-rendered PNGs and stacks them
vertically with (a)/(b)/(c) labels, so LaTeX treats them as a single float.
Re-run this after re-plotting any of the three panels.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

PANEL_A = "aaai2027/figures/empconf_panelA_entropy_boxplot.png"
PANEL_B = "aaai2027/figures/empconf_panelB_mi_decay_linegraph.png"
PANEL_C = "aaai2027/figures/empconf_panelC_gnn_entropy_heatmap.png"
OUT_PNG = "aaai2027/figures/empconf_panels_abc_combined.png"


def main():
    imgs = [mpimg.imread(PANEL_A), mpimg.imread(PANEL_B), mpimg.imread(PANEL_C)]
    labels = ["(a)", "(b)", "(c)"]

    width_in = 7.0
    heights = [width_in * im.shape[0] / im.shape[1] for im in imgs]

    fig = plt.figure(figsize=(width_in, sum(heights)))
    gs = fig.add_gridspec(3, 1, height_ratios=heights, hspace=0.03)

    for i, (img, label, h) in enumerate(zip(imgs, labels, heights)):
        ax = fig.add_subplot(gs[i])
        ax.imshow(img)
        ax.axis("off")
        ax.text(0.0, 1.0, label, transform=ax.transAxes, fontsize=11, fontweight="bold",
                va="top", ha="left")

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight", pad_inches=0.05)
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
