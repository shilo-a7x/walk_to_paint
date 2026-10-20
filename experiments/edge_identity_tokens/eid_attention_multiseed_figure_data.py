"""Thesis attention figure, panels B-D: mean over the 10 trained H3 models (splits 42-51), error bars =
std across models (decided 2026-10-20; WSDM used one model, split 42, with SE across target edges).
Panel A stays the split-42 Bitcoin-alpha per-head grid (heads are not comparable across models).

Inputs: outputs/attn_shap_multiseed/{attention,shap}_per_seed.csv (attn_shap_multiseed.py --summarise).
Writes, in the schemas the unchanged WSDM plot scripts read (thesis_figure_data/):
  attndir_panelBC_summary.csv, attndir_panelBC_se.csv, shap_edge_directionality.csv
The WSDM panel B/C plot scripts draw yerr = 1.96 * <x>_se, panel D draws yerr = cluster_se; to get
+-1 std across models we store <x>_se = std / 1.96 and cluster_se = std. The split-42 versions are kept
as *_s42.csv.

Usage: .venv/bin/python experiments/edge_identity_tokens/eid_attention_multiseed_figure_data.py
"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "outputs", "attn_shap_multiseed")
OUT = os.path.join(ROOT, "experiments", "edge_identity_tokens", "thesis_figure_data")
DATASETS = ["bitcoin-alpha", "bitcoin-otc", "epinions", "slashdot090221", "wiki-elec", "wiki-rfa"]


def main():
    a = pd.read_csv(os.path.join(SRC, "attention_per_seed.csv"))
    a = a[a.model == "h3"]
    nhead = pd.read_csv(os.path.join(OUT, "attndir_panelBC_summary_s42.csv")).set_index("dataset")["nhead"]
    summ, se = [], []
    for ds in DATASETS:
        d = a[a.dataset == ds]
        assert len(d) == 10, (ds, len(d))
        m, s = d[["fwd", "bwd", "node", "edge"]].mean(), d[["fwd", "bwd", "node", "edge"]].std()
        summ.append({"dataset": ds, "layer": 0, "nhead": int(nhead[ds]), "forward": m.fwd, "backward": m.bwd,
                     "node": m.node, "edge": m.edge, "self": 1 - m.fwd - m.bwd})
        row = {"dataset": ds}
        for name, col in (("forward", "fwd"), ("backward", "bwd"), ("node", "node"), ("edge", "edge")):
            row[f"{name}_mean"], row[f"{name}_se"] = m[col], s[col] / 1.96
        se.append(row)
    pd.DataFrame(summ).to_csv(os.path.join(OUT, "attndir_panelBC_summary.csv"), index=False)
    pd.DataFrame(se).to_csv(os.path.join(OUT, "attndir_panelBC_se.csv"), index=False)

    s = pd.read_csv(os.path.join(SRC, "shap_per_seed.csv"))
    s = s[(s.model == "h3") & (s.shap_remove == "both")]
    g = s.groupby(["dataset", "direction", "hop"])
    shap = g.agg(mean_abs_shap=("mean_abs_shap", "mean"), cluster_se=("mean_abs_shap", "std"),
                 n_obs=("n_obs", "sum"), n_clusters=("seed", "nunique")).reset_index()
    assert (shap.n_clusters == 10).all()
    order = {ds: i for i, ds in enumerate(DATASETS)}
    shap = shap.sort_values(["dataset", "direction", "hop"], key=lambda c: c.map(order) if c.name == "dataset" else c)
    shap.to_csv(os.path.join(OUT, "shap_edge_directionality.csv"), index=False)
    print(f"wrote {OUT}/attndir_panelBC_summary.csv, attndir_panelBC_se.csv, shap_edge_directionality.csv")


if __name__ == "__main__":
    main()
