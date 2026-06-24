"""
Regenerate node-MI plots with limited y-axis from cached pickle results.
Loads pre-computed results to avoid expensive recalculation.
"""

import os, sys, pickle, argparse, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATASET_CONFIGS = {
    "bitcoin-alpha": {},
    "bitcoin-otc": {},
    "epinions": {},
    "wiki-elec": {},
    "wiki-rfa": {},
    "slashdot": {},
}

MI_Y_LIMIT = 0.30


def regenerate_plots(result_dir: str, ds_name: str, y_limit: float, output_suffix: str = "_limited"):
    """Load pickled result and regenerate plot with improved y-axis scaling."""
    pkl_path = os.path.join(result_dir, f"node_mi_{ds_name}_result.pkl")

    if not os.path.exists(pkl_path):
        print(f"  ✗ Result file not found: {pkl_path}")
        return False

    with open(pkl_path, "rb") as f:
        result = pickle.load(f)

    results = result["results"]
    feature_names = result["feature_names"]
    d_max = result["d_max"]
    n_anchors = result["n_anchors"]

    print(f"  Loaded {ds_name}: {len(feature_names)} features, d_max={d_max}, anchors={n_anchors:,}")

    # ── Regenerate plot with limited y-axis ──────────────────────────────────
    n_feat = len(feature_names)
    ncols = 4
    nrows = math.ceil(n_feat / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_2d(axes)
    hops = list(range(1, d_max + 1))

    for fi, fname in enumerate(feature_names):
        ax = axes[fi // ncols, fi % ncols]
        vals = [results[fname]["by_d"][d]["mi"] for d in hops]
        valid = [(h, v) for h, v in zip(hops, vals) if not (isinstance(v, float) and math.isnan(v))]
        if valid:
            hx, vy = zip(*valid)
            ax.plot(hx, vy, marker="o", label="MI(d)", color="tab:blue")
        h_val = results[fname]["H"]
        if not math.isnan(h_val):
            ax.axhline(h_val, color="gray", linewidth=0.7, linestyle="--", label="H(feature)")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.set_title(fname, fontsize=9)
        ax.set_xlabel("hop d")
        ax.set_ylabel("MI (bits)")
        ax.set_xticks(hops)
        ax.set_ylim(bottom=0, top=y_limit)
        if fi == 0:
            ax.legend(fontsize=7)

    for fi in range(n_feat, nrows * ncols):
        axes[fi // ncols, fi % ncols].axis("off")

    fig.suptitle(
        f"{ds_name}: node-pair feature MI(A,B) vs hop distance d "
        f"(anchors={n_anchors:,})",
        fontsize=11)
    fig.tight_layout()

    save_path = os.path.join(result_dir, f"node_mi_{ds_name}{output_suffix}.png")
    fig.savefig(save_path, dpi=110)
    plt.close(fig)
    print(f"  ✓ Saved {os.path.basename(save_path)}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate node-MI plots with limited y-axis from cached results")
    parser.add_argument("--result-dir", default="outputs/node_mi_structural",
                        help="Directory containing pickle result files")
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--y-limit", type=float, default=MI_Y_LIMIT,
                        help=f"Y-axis limit for MI plots (default: {MI_Y_LIMIT})")
    args = parser.parse_args()

    if args.datasets == ["all"]:
        datasets = list(DATASET_CONFIGS.keys())
    else:
        datasets = args.datasets

    result_dir = os.path.join(ROOT, args.result_dir)
    if not os.path.isdir(result_dir):
        print(f"Error: result directory not found: {result_dir}")
        sys.exit(1)

    print(f"\nRegenerating plots (y_limit={args.y_limit}) from {result_dir}\n")

    success_count = 0
    for ds_name in datasets:
        if regenerate_plots(result_dir, ds_name, args.y_limit):
            success_count += 1

    print(f"\n✓ Regenerated {success_count}/{len(datasets)} datasets")


if __name__ == "__main__":
    main()
