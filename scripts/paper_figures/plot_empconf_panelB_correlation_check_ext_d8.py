"""Extended-range (d=8) variant of plot_empconf_panelB_correlation_check.py --
separate output, does NOT touch the original diagnostic plot/PNG.

Pure rendering: reads the same production correlation CSV
(aaai2027/figure_data/empconf_panelB_correlation_check.csv), which now has
d=7,8 rows appended (real BFS re-run at --d-max 7, NOT interpolated) alongside
the original d=1-6 rows, left untouched. Null CSV also extended to d=7,8 the
same way (same scratchpad path as before -- see plot_empconf_panelB_
correlation_check.py's docstring for why it isn't yet a permanent paper
asset path).
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_REAL_CSV = "aaai2027/figure_data/empconf_panelB_correlation_check.csv"
IN_NULL_CSV = "/tmp/claude-30743/-home-eng-shilo-avital-yolo-lab-walk-to-paint/22a506ad-31f1-432e-aba0-042627f162c3/scratchpad/corr_null_full.csv"
OUT_PNG = "aaai2027/figures/empconf_panelB_correlation_check_ext_d8.png"

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
    ax_phi.text(0.02, 0.96, "extended to d=8 (real BFS re-run, d=1-6 unchanged)",
                transform=ax_phi.transAxes, ha="left", va="top", fontsize=6.5, color="#888",
                style="italic")
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
