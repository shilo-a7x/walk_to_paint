"""Plot step for the phi-correlation diagnostic check (§Part 2 of
MI_ESTIMATOR_ELI5.md) -- NOT a paper figure yet, a diagnostic comparison
against Panel B's MI/NMI curve to check whether the post-minimum "bump" is a
real, signed effect or noise.

Pure rendering: reads aaai2027/figure_data/empconf_panelB_correlation_check.csv
(real, built by extract_empconf_panelB_correlation_check.py) and the matching
shuffle-signs null run. Edit THIS file freely for style changes -- no
recomputation needed (the null CSV path below points at the scratchpad run;
move it under aaai2027/figure_data/ if this plot gets promoted to a real
paper asset).

One panel: phi coefficient vs. line-graph distance, one line per dataset,
solid = real, thin dashed grey = shuffle-signs null, zero reference line
(phi is signed, unlike NMI, so the zero line is the whole point of this
chart -- it's what makes the sign flip visible at all).
"""
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_REAL_CSV = "aaai2027/figure_data/empconf_panelB_correlation_check.csv"
IN_NULL_CSV = "/tmp/claude-30743/-home-eng-shilo-avital-yolo-lab-walk-to-paint/22a506ad-31f1-432e-aba0-042627f162c3/scratchpad/corr_null_full.csv"
OUT_PNG = "aaai2027/figures/empconf_panelB_correlation_check.png"

DATASET_ORDER = ["bitcoin-alpha", "bitcoin-otc", "epinions", "slashdot", "wiki-elec", "wiki-rfa"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
DISPLAY_LABEL = {"slashdot": "slashdot"}


def load(path):
    data = {ds: [] for ds in DATASET_ORDER}
    with open(path) as f:
        for row in csv.DictReader(f):
            ds = row["dataset"]
            if ds not in data:
                continue
            phi = row["phi"]
            z = row["z"]
            if phi in ("", "nan") or z in ("", "nan"):
                continue
            data[ds].append((int(row["line_dist"]), float(phi), float(z)))
    for ds in data:
        data[ds].sort()
    return data


def main():
    real = load(IN_REAL_CSV)
    null = load(IN_NULL_CSV)

    fig, ax_phi = plt.subplots(1, 1, figsize=(5.6, 4.0))

    for ds, color, marker in zip(DATASET_ORDER, COLORS, MARKERS):
        pts = real[ds]
        if not pts:
            continue
        xs = [p[0] for p in pts]
        phis = [p[1] for p in pts]
        ax_phi.plot(xs, phis, color=color, marker=marker, markersize=5, linewidth=1.6,
                    label=DISPLAY_LABEL.get(ds, ds), zorder=3)

        npts = null[ds]
        if npts:
            nxs = [p[0] for p in npts]
            nphis = [p[1] for p in npts]
            ax_phi.plot(nxs, nphis, color=color, linewidth=1.0, linestyle="--", alpha=0.45,
                        zorder=2)

    ax_phi.axhline(0, color="#8a8a80", linewidth=1.0, zorder=1)
    ax_phi.set_ylabel(r"$\phi$ (edge-sign correlation)", fontsize=9)
    ax_phi.set_xlabel("edge-to-edge distance (line-graph hops)", fontsize=9)
    ax_phi.set_xticks(sorted({p[0] for pts in real.values() for p in pts}))
    ax_phi.text(0.98, 0.04, "dashed = shuffle-signs null", transform=ax_phi.transAxes,
                ha="right", va="bottom", fontsize=7, color="#666", style="italic")
    ax_phi.legend(frameon=False, fontsize=7.5, loc="upper right", ncol=2)

    for ax in (ax_phi,):
        ax.grid(True, which="major", axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#c3c2b7")
        ax.tick_params(colors="#52514e")

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
