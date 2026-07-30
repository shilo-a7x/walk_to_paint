"""Combine Ablation A (full vs. local attention) and Ablation C (weighted vs.
plain mean vote) into ONE figure -- same technique as
combine_empconf_panels_bc.py, so LaTeX treats them as a single float. Ablation
B (smart sampling) was dropped 2026-07-26 (not a meaningful ablation --
edge_cover's coverage guarantee is essential for fair evaluation, not a
performance lever), so this is a 2-panel, not 3-panel, composite.

Re-run this after re-plotting either panel.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

PANEL_A = "aaai2027/figures/ablationA_full_vs_local.png"
PANEL_C = "aaai2027/figures/ablationC_full_sweep.png"
OUT_PNG = "aaai2027/figures/ablations_ac_combined.png"


def main():
    img_a = mpimg.imread(PANEL_A)
    img_c = mpimg.imread(PANEL_C)

    width_in = 7.0
    h_a = width_in * img_a.shape[0] / img_a.shape[1]
    h_c = width_in * img_c.shape[0] / img_c.shape[1]

    fig = plt.figure(figsize=(width_in, h_a + h_c))
    gs = fig.add_gridspec(2, 1, height_ratios=[h_a, h_c], hspace=0.05)

    ax_a = fig.add_subplot(gs[0])
    ax_a.imshow(img_a)
    ax_a.axis("off")
    ax_a.text(0.0, 1.0, "(a)", transform=ax_a.transAxes, fontsize=11, fontweight="bold",
              va="top", ha="left")

    ax_c = fig.add_subplot(gs[1])
    ax_c.imshow(img_c)
    ax_c.axis("off")
    ax_c.text(0.0, 1.0, "(b)", transform=ax_c.transAxes, fontsize=11, fontweight="bold",
              va="top", ha="left")

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight", pad_inches=0.05)
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
