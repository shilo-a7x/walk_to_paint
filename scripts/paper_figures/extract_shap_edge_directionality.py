"""Extract step for the SHAP edge directionality figure -- reformats the per-dataset
summary rows already computed by scripts/shap_edge_directionality.py (exact Shapley,
LocalAttn4, edge-tokens-only, local-window-restricted; see that script's docstring for
the full method) into the standard aaai2027/figure_data/ location. No new computation --
mean|SHAP| and cluster-robust SE per (dataset, direction, hop) are read straight from
outputs/shap_edge_directionality/shap_directionality_<ds>_result.pkl.
"""
import csv
import os
import pickle

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASETS = ["bitcoin-alpha", "bitcoin-otc", "epinions", "wiki-elec", "wiki-rfa", "slashdot090221"]
IN_DIR = os.path.join(ROOT, "outputs", "shap_edge_directionality")
OUT_CSV = os.path.join(ROOT, "aaai2027", "figure_data", "shap_edge_directionality.csv")


def main():
    all_rows = []
    for ds in DATASETS:
        pkl_path = os.path.join(IN_DIR, f"shap_directionality_{ds}_result.pkl")
        with open(pkl_path, "rb") as f:
            result = pickle.load(f)
        all_rows.extend(result["summary_rows"])
        print(f"{ds}: n_instances={result['n_instances']:,} "
              f"max_efficiency_gap={result['max_efficiency_gap']:.2e}")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "direction", "hop", "mean_abs_shap",
                                           "cluster_se", "n_obs", "n_clusters"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nwrote {OUT_CSV} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
