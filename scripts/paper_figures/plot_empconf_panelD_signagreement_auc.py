"""Plot step for Empirical Confirmation Panel D -- 4-way sign-agreement AUC bars.
Pure rendering: reads aaai2027/figure_data/empconf_panelD_signagreement_auc.csv
(built by extract_empconf_panelD_signagreement_auc.py). Edit THIS file freely for
color/style changes -- no recomputation needed.

Bucket sample sizes (n) are large (millions, pooled across 6 datasets) and highly
uneven across buckets -- annotated on each bar per the plan's verification step.

SiGAT only (2026-08-04 call) -- GINEConv dropped from this figure, kept in the
appendix/baseline table instead; the extract CSV still has both models' rows
for that reuse.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IN_CSV = "aaai2027/figure_data/empconf_panelD_signagreement_auc.csv"
OUT_PNG = "aaai2027/figures/empconf_panelD_signagreement_auc.png"

MODELS = ["SiGAT"]  # GINEConv dropped from this figure 2026-08-04, kept in appendix/table only
MODEL_COLOR = {"GINEConv": "#4472c4", "SiGAT": "#c0392b"}
BUCKET_ORDER = ["in_same", "in_diff", "out_same", "out_diff"]
BUCKET_LABEL = {
    "in_same": "in-same", "in_diff": "in-diff",
    "out_same": "out-same", "out_diff": "out-diff",
}


def fmt_n(n):
    if n >= 1_000_000:
        return f"n={n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"n={n/1_000:.0f}k"
    return f"n={n}"


def main():
    rows = list(csv.DictReader(open(IN_CSV)))
    data = {(r["model"], r["bucket"]): r for r in rows}

    x = np.arange(len(BUCKET_ORDER))
    width = 0.6 if len(MODELS) == 1 else 0.35
    fig, ax = plt.subplots(figsize=(6.6, 4.2))

    YMIN = 0.4
    ax.set_ylim(YMIN, 1.05)
    for i, model in enumerate(MODELS):
        aucs = [float(data[(model, b)]["auc"]) for b in BUCKET_ORDER]
        ns = [int(data[(model, b)]["n"]) for b in BUCKET_ORDER]
        xpos = x + (i - (len(MODELS) - 1) / 2) * width
        # bars are clipped to start at YMIN (bottom=YMIN) since sklearn AUC is >=0.5
        # here anyway -- bar() with bottom=YMIN avoids drawing off-axis bar bodies
        bars = ax.bar(xpos, [a - YMIN for a in aucs], width, bottom=YMIN,
                       color=MODEL_COLOR[model], label=model, zorder=3)
        for bar, auc, n in zip(bars, aucs, ns):
            ax.text(bar.get_x() + bar.get_width() / 2, auc + 0.012, f"{auc:.3f}",
                     ha="center", va="bottom", fontsize=10)
            ax.text(bar.get_x() + bar.get_width() / 2, YMIN + 0.012, fmt_n(n),
                     ha="center", va="bottom", fontsize=8, rotation=90, color="white")

    ax.set_xticks(x)
    ax.set_xticklabels([BUCKET_LABEL[b] for b in BUCKET_ORDER], fontsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_ylabel("Test AUC", fontsize=10)
    ax.set_title("SiGAT AUC by sign agreement", fontsize=11)
    if len(MODELS) > 1:
        ax.legend(loc="lower right")
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
