"""
Lead 2 follow-up: weight-vs-importance diagnostic, variant 1 (bucketed MI).

Loads the per-dataset pickles produced by
baselines/SGA/extract_attention_weights.py (literal GATConv attention
weights `alpha` for SiGAT's direct pos/neg out-edge channels, plus the
final post-MLP embedding z and each node's own out-edge majority sign
`own_sign`).

For each directed signed edge (i, j) in the probed channel, alpha_ji is the
attention weight node j's GATConv channel assigns to neighbor i. The
question: does that weight track how much real, predictive information
about i actually ends up in j's final embedding z_j? Bucket edges by alpha
(quantile, same N_BUCKETS as Step 1b's degree buckets), then within each
bucket compute MI(z_j, own_sign(i)) -- i's own out-edge majority sign is
the same "context" signal Step 1/Step 2 used (relay's own next-hop edge
sign). If attention tracks importance, MI should rise with the bucket
(low-alpha bucket -> low MI, high-alpha bucket -> high MI). A bucket that
breaks this (low alpha, high MI) is the "good info through a bad pipe"
signature the user is looking for.

Also reports variant 2's correlation summary (delta_relative_final vs
alpha, already computed in the extraction script) alongside the MI table,
so both diagnostics are read together per dataset.

Usage
-----
  python scripts/lead2_sigat_attention_weight_mi.py --datasets all
"""
import argparse
import os
import pickle
import sys

import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from mi_pca_binning_utils import mi_pca_bins  # noqa: E402

ALL_DATASETS = ["bitcoin-alpha", "bitcoin-otc", "epinions",
                 "wiki-elec", "wiki-rfa", "slashdot090221"]
N_BUCKETS = 4


def quantile_buckets(values: np.ndarray, n_buckets: int = N_BUCKETS):
    edges = np.quantile(values, np.linspace(0, 1, n_buckets + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return np.digitize(values, edges[1:-1], right=True)


def analyse_channel(ch: dict, z_full: np.ndarray, own_sign: dict, ch_name: str):
    edges, alpha = ch["edges"], ch["alpha"]
    i_idx, j_idx = edges[:, 0], edges[:, 1]

    # context = i's own out-edge majority sign; drop edges where i has no
    # train out-edges of its own (own_sign undefined, can't form context).
    has_own = np.array([i in own_sign and own_sign[i] != 0 for i in i_idx.tolist()])
    if has_own.sum() < 20:
        return None
    delta_rel = ch.get("delta_relative_final", np.zeros_like(alpha))[has_own]
    i_idx, j_idx, alpha = i_idx[has_own], j_idx[has_own], alpha[has_own]
    context = np.array([own_sign[i] for i in i_idx.tolist()], dtype=np.float64)
    z_j = z_full[j_idx]

    buckets = quantile_buckets(alpha)
    bucket_results = []
    for b in range(N_BUCKETS):
        mask = buckets == b
        n = mask.sum()
        if n < 20:
            bucket_results.append({"n": int(n), "mi": None, "nmi": None,
                                    "alpha_range": (np.nan, np.nan)})
            continue
        res = mi_pca_bins(z_j[mask], context[mask])
        bucket_results.append({
            "n": int(n), "mi": res["mi"], "nmi": res["nmi"],
            "alpha_range": (float(alpha[mask].min()), float(alpha[mask].max())),
        })

    rho, _ = spearmanr(alpha, delta_rel)
    return {"channel": ch_name, "n_edges": int(has_own.sum()),
            "buckets": bucket_results, "spearman_alpha_delta": rho}


def analyse_dataset(ds_name: str, attn_dir: str):
    path = os.path.join(attn_dir, f"{ds_name}_attention.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        d = pickle.load(f)

    results = []
    for ch_name in ("pos", "neg"):
        r = analyse_channel(d[ch_name], d["z_full"], d["own_sign"], ch_name)
        if r is not None:
            results.append(r)
    return {"dataset": ds_name, "channels": results}


def write_report(all_results: list, out_dir: str):
    lines = [
        "=" * 96,
        "  LEAD 2 FOLLOW-UP -- SiGAT attention-weight vs. true-importance diagnostic",
        "  Variant 1: MI(final embedding z_j, neighbor i's own out-edge sign),",
        "             bucketed by the attention weight alpha_ji the model assigns to i.",
        "  Variant 2 (cross-ref): Spearman(alpha, post-MLP leave-one-out delta) from",
        "             the extraction script -- repeated here for side-by-side reading.",
        "=" * 96,
        "",
        "If attention tracks true importance: MI should rise monotonically bucket 0->3",
        "(low alpha -> low MI, high alpha -> high MI). A bucket that breaks this",
        "pattern (low alpha, high MI) is the 'good info through a bad pipe' signature.",
        "",
    ]
    for r in all_results:
        lines += [f"{'─'*96}", f"  {r['dataset']}", f"{'─'*96}"]
        for ch in r["channels"]:
            lines.append(f"  channel={ch['channel']}  n_edges={ch['n_edges']:,}  "
                          f"spearman(alpha, delta_rel_final)={ch['spearman_alpha_delta']:.4f}")
            lines.append(f"  {'bucket':<8}{'n':>10}{'alpha_range':>22}{'mi(bits)':>12}{'nmi':>10}")
            for b, br in enumerate(ch["buckets"]):
                if br["mi"] is None:
                    lines.append(f"  {b:<8}{br['n']:>10}{'(too few)':>22}{'--':>12}{'--':>10}")
                else:
                    arange = f"[{br['alpha_range'][0]:.4f},{br['alpha_range'][1]:.4f}]"
                    lines.append(f"  {b:<8}{br['n']:>10}{arange:>22}{br['mi']:>12.6f}{br['nmi']:>10.4f}")
            lines.append("")

    path = os.path.join(out_dir, "attention_weight_mi_report.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport written to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--attn-dir", default="outputs/lead2_sigat_attention")
    parser.add_argument("--out", default="outputs/lead2_sigat_attention")
    args = parser.parse_args()

    datasets = ALL_DATASETS if args.datasets == ["all"] else args.datasets
    attn_dir = os.path.join(ROOT, args.attn_dir)
    out_dir = os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)

    all_results = []
    for ds in datasets:
        print(f"\n-- {ds} --")
        res = analyse_dataset(ds, attn_dir)
        if res is None:
            print("  skipped (attention pickle not found)")
            continue
        for ch in res["channels"]:
            print(f"  [{ch['channel']}] spearman(alpha,delta_rel_final)={ch['spearman_alpha_delta']:.4f}")
            for b, br in enumerate(ch["buckets"]):
                if br["mi"] is not None:
                    print(f"    bucket {b} (alpha in [{br['alpha_range'][0]:.4f},{br['alpha_range'][1]:.4f}], "
                          f"n={br['n']}): mi={br['mi']:.6f}  nmi={br['nmi']:.4f}")
        all_results.append(res)

    if all_results:
        write_report(all_results, out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
