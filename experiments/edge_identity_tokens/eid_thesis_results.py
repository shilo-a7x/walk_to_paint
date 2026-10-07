"""Collect every thesis number for the H3 campaign (make_thesis_jobs.py) from on-disk artifacts --
no training, no refitting.

Per run (condition x dataset x seed): val/test edge AUC (func_logit_power posthoc summary), and
walk/edge-level macro-F1 + accuracy recomputed from the saved test_predictions.pkl with the
already-fit theta* (same procedure as scripts/paper_figures/extract_ablation_f1.py).
Then: mean+-std per condition/dataset; each condition vs H3 (paired by seed, two-sided Wilcoxon);
H3 vs production (two-sided) and vs the best baseline (one-sided, Table 1 convention); the
K-walks curve for H3 (uniform mean of K sampled occurrences, extract_ablation_kwalks.py's
procedure); Ablation B (all 11 aggregators, posthoc run_id thesis_aggB) for H3 if present.

Outputs: experiments/edge_identity_tokens/thesis_figure_data/eid_thesis_*.csv and
experiments/edge_identity_tokens/EID_THESIS_RESULTS.md.

Usage: .venv/bin/python experiments/edge_identity_tokens/eid_thesis_results.py
"""
import glob
import json
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import experiments.edge_identity_tokens.eid_significance as sig  # noqa: E402
import experiments.edge_identity_tokens.eid_vs_baseline_significance as vb  # noqa: E402
from experiments.edge_identity_tokens.make_thesis_jobs import CONDITIONS, SEEDS, WALKS  # noqa: E402

OUT = ROOT / "experiments" / "edge_identity_tokens" / "thesis_figure_data"
REPORT = ROOT / "experiments" / "edge_identity_tokens" / "EID_THESIS_RESULTS.md"
EPS = 1e-6
K_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
AGG_FUNCS = ["func_uniform", "func_conf_power", "func_conf_exp", "func_conf_cert", "func_conf_logit",
             "func_logq_power", "func_logit_power", "func_entropy_power", "func_entropy_exp",
             "func_fisher_power", "func_maxprob_power"]
DATASETS = list(WALKS)


def exp_dir(name, ds, seed):
    tag = f"EIDREG_{name}_{ds.upper().replace('-', '')}_s{seed}_"
    dirs = sorted(glob.glob(str(ROOT / "outputs" / ds / f"{tag}*")))
    return Path(dirs[-1]) if dirs else None


def read_summary(d, run_id, model="func_logit_power"):
    p = d / "posthoc" / run_id / "aggregator" / model / "summary.txt"
    if not p.is_file():
        return None, None
    t = p.read_text()
    v = re.search(r"Train AUC:\s+([0-9.]+)", t)
    s = re.search(r"Test\s+AUC:\s+([0-9.]+)", t)
    return (float(v.group(1)) if v else None), (float(s.group(1)) if s else None)


def test_predictions(d, ds):
    for e in sorted(glob.glob(str(d / "checkpoints" / f"{ds}_predictions" / "epoch_*")), reverse=True):
        p = Path(e) / "test_predictions.pkl"
        if p.is_file():
            return pickle.load(open(p, "rb"))
    return None


def f1_metrics(d, ds):
    cfg = d / "posthoc" / "reg" / "aggregator" / "func_logit_power" / "model_config.json"
    pred = test_predictions(d, ds)
    if pred is None or not cfg.is_file():
        return {}
    theta = json.load(open(cfg))["theta_star"][0]
    eids = np.asarray(pred["edge_ids"]).astype(np.int64)
    q = pred["probabilities"][:, 1].astype(float)
    y = np.asarray(pred["targets"]).astype(int)
    w = np.maximum(np.power(np.abs(np.log(np.clip(q, EPS, 1 - EPS) / np.clip(1 - q, EPS, 1))) + EPS, theta), EPS)
    order = np.argsort(eids, kind="stable")
    _, inv, cnt = np.unique(eids[order], return_inverse=True, return_counts=True)
    score = np.bincount(inv, weights=w[order] * q[order]) / np.maximum(np.bincount(inv, weights=w[order]), EPS)
    lab = y[order][np.concatenate([[0], np.cumsum(cnt)[:-1]])]
    return {"walk_f1": f1_score(y, q >= 0.5, average="macro"), "walk_acc": accuracy_score(y, q >= 0.5),
            "edge_f1": f1_score(lab, score >= 0.5, average="macro"), "edge_acc": accuracy_score(lab, score >= 0.5),
            "edge_f1_pos": f1_score(lab, score >= 0.5), "n_test_edges": len(lab)}


def kwalks(pred, seed):
    eids = np.asarray(pred["edge_ids"]).astype(np.int64)
    q = pred["probabilities"][:, 1].astype(float)
    y = np.asarray(pred["targets"]).astype(int)
    order = np.argsort(eids, kind="stable")
    q_s, y_s = q[order], y[order]
    _, start, cnt = np.unique(eids[order], return_index=True, return_counts=True)
    rng = np.random.default_rng(seed)
    out = []
    for k in K_GRID:
        sc = np.array([q_s[rng.choice(np.arange(a, a + c), size=k, replace=False)].mean() if c > k
                       else q_s[a:a + c].mean() for a, c in zip(start, cnt)])
        out.append((k, roc_auc_score(y_s[start], sc), int(cnt.max())))
        if k >= cnt.max():
            break
    return out


def md(df, index=True):
    """DataFrame -> GitHub markdown table (no tabulate dependency)."""
    df = df.reset_index() if index else df
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |"
            for r in df.itertuples(index=False)]
    return "\n".join(out)


def paired(a, b, alternative="two-sided"):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 5:
        return np.nan, np.nan, int(ok.sum())
    diff = a[ok] - b[ok]
    p = wilcoxon(diff, alternative=alternative).pvalue if np.any(diff != 0) else 1.0
    return float(diff.mean()), float(p), int((diff > 0).sum())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, krows, brows = [], [], []
    for name, _ in CONDITIONS:
        for ds in DATASETS:
            for seed in SEEDS:
                d = exp_dir(name, ds, seed)
                if d is None:
                    rows.append(dict(condition=name, dataset=ds, seed=seed))
                    continue
                val, test = read_summary(d, "reg")
                rows.append(dict(condition=name, dataset=ds, seed=seed, val_auc=val, test_auc=test,
                                 exp_dir=str(d.relative_to(ROOT)), **f1_metrics(d, ds)))
                if name == "H3":
                    pred = test_predictions(d, ds)
                    if pred is not None:
                        krows += [dict(dataset=ds, seed=seed, k=k, auc=a, max_occ=m) for k, a, m in kwalks(pred, seed)]
                    for agg in AGG_FUNCS:
                        bv, bt = read_summary(d, "thesis_aggB", agg)
                        if bt is not None:
                            brows.append(dict(dataset=ds, seed=seed, aggregator=agg, val_auc=bv, test_auc=bt))
    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "eid_thesis_runs.csv", index=False)
    pd.DataFrame(krows).to_csv(OUT / "eid_thesis_kwalks.csv", index=False)
    pd.DataFrame(brows).to_csv(OUT / "eid_thesis_ablationB.csv", index=False)

    metrics = ["test_auc", "val_auc", "edge_f1", "edge_acc", "walk_f1"]
    summ = runs.groupby(["condition", "dataset"]).agg(
        n=("test_auc", "count"), **{f"{m}_{s}": (m, s) for m in metrics for s in ("mean", "std")}).reset_index()
    summ.to_csv(OUT / "eid_thesis_summary.csv", index=False)

    def per_seed(cond, ds, col="test_auc"):
        x = runs[(runs.condition == cond) & (runs.dataset == ds)].set_index("seed")[col]
        return [x.get(s, np.nan) for s in SEEDS]

    tests = []
    for ds in DATASETS:
        h3 = per_seed("H3", ds)
        for name, _ in CONDITIONS[1:]:
            m, p, w = paired(per_seed(name, ds), h3)
            tests.append(dict(comparison=f"{name} - H3", dataset=ds, mean_diff=m, n_greater=w, p_value=p, test="two-sided"))
        prod = [sig.production_seed_auc(ds, s) for s in SEEDS]
        m, p, w = paired(h3, [np.nan if v is None else v for v in prod])
        tests.append(dict(comparison="H3 - production", dataset=ds, mean_diff=m, n_greater=w, p_value=p, test="two-sided"))
        bl = [vb.baseline_seed_auc(vb.BEST_BASELINE[ds], ds, s) for s in SEEDS]
        m, p, w = paired(h3, [np.nan if v is None else v for v in bl], "greater")
        tests.append(dict(comparison=f"H3 - {vb.BEST_BASELINE[ds]}", dataset=ds, mean_diff=m, n_greater=w, p_value=p,
                          test="one-sided greater"))
    tests = pd.DataFrame(tests)
    tests.to_csv(OUT / "eid_thesis_paired_tests.csv", index=False)

    lines = ["# EID thesis campaign results (H3, seeds 42-51)", "",
             "Generated by `experiments/edge_identity_tokens/eid_thesis_results.py` -- do not hand-edit.", "",
             "## Test AUC, mean +- std (n seeds)", ""]
    t = summ.assign(cell=lambda x: x.apply(lambda r: f"{r.test_auc_mean:.4f} +- {r.test_auc_std:.4f} ({int(r.n)})", axis=1))
    order = [c for c, _ in CONDITIONS]
    lines.append(md(t.pivot(index="condition", columns="dataset", values="cell").reindex(order)))
    lines += ["", "## Edge-level macro-F1, mean", ""]
    lines.append(md(summ.pivot(index="condition", columns="dataset", values="edge_f1_mean").reindex(order).round(4)))
    lines += ["", "## Paired tests (mean diff in AUC, seeds where first > second, p)", ""]
    tt = tests.assign(cell=lambda x: x.apply(lambda r: f"{r.mean_diff * 100:+.2f}pp ({r.n_greater}/10, p={r.p_value:.3g})", axis=1))
    lines.append(md(tt.pivot(index="comparison", columns="dataset", values="cell")))
    if krows:
        k = pd.DataFrame(krows).groupby(["dataset", "k"]).auc.mean().unstack(0)
        lines += ["", "## K-walks (H3, mean test AUC over seeds, uniform mean of K occurrences)", "", md(k.round(4))]
    if brows:
        b = pd.DataFrame(brows).groupby(["aggregator", "dataset"]).test_auc.mean().unstack(1)
        lines += ["", "## Ablation B (H3, mean test AUC per aggregator)", "", md(b.round(4))]
    missing = runs[runs.test_auc.isna()][["condition", "dataset", "seed"]]
    lines += ["", f"## Missing runs: {len(missing)}", ""]
    if len(missing):
        lines.append(md(missing, index=False))
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"wrote {REPORT} and {OUT}/eid_thesis_*.csv ({runs.test_auc.notna().sum()} runs with results)")


if __name__ == "__main__":
    main()
