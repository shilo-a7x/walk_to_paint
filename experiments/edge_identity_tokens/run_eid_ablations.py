"""EID context/sign ablation campaign, 2026-09-15: single-seed, all 6 datasets.

Two ablations, both EID-native (`eid_src/model/lit_model.py`):
  - `model.mask_context_edges`: every edge position that is NOT the current
    prediction target gets BOTH input_ids (identity) and sign_ids (sign) blanked
    -- "do we need context-edge information at all". Already correctly excludes
    the target position and treats disallowed/holdout bystanders the same as
    ordinary context (uniformly blanked). No bug found here.
  - `model.scramble_edge_signs`: a fixed ~50% subset of edges show a
    deterministic WRONG sign at every visible context position, identity
    untouched -- "does the model get misled by wrong context". **Bug fixed
    2026-09-15** (see lit_model.py's updated docstring): the original had no
    guard excluding already-hidden (target/disallowed) positions, so ~63% of
    already-hidden positions got silently corrupted with a wrong-but-concrete
    sign in one verified test batch -- contaminating the exact position being
    predicted, not just context. Fixed by requiring `sign_ids != SIGN_NA` on
    the incoming value before flipping (mirrors production's own
    `input_ids != mask_id` visibility guard).

Base config per dataset = that dataset's already-established gap-closed
architecture + budget (best-of v1-vs-v2, same discipline used to finalize the
gap-closer numbers) -- no new arch search, no new budget sweep, single seed 42.
`model.eid_reveal_holdout_identity=true` is included in every run (canon, not
an ablation toggle -- confirmed with the user 2026-09-15) even though it has
no effect under mask_context_edges specifically (that flag unconditionally
re-blanks every non-target position, including disallowed ones, every batch --
whatever reveal_holdout_identity did gets wiped right back out). Included
anyway for config consistency with every other EID run.

Usage:
  .venv/bin/python experiments/edge_identity_tokens/run_eid_ablations.py \
      --datasets bitcoin-alpha,bitcoin-otc,wiki-elec,wiki-rfa,epinions,slashdot090221
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import experiments.edge_identity_tokens.run_gap_closer as rgc  # noqa: E402

REPO_ROOT = rgc.REPO_ROOT
VENV_PY = rgc.VENV_PY
LOG_DIR = rgc.LOG_DIR
VANILLA_AUC = rgc.VANILLA_AUC
N_GPUS = rgc.N_GPUS

STATE_V1 = json.loads((LOG_DIR / "state.json").read_text())
STATE_V2 = json.loads((LOG_DIR / "state_v2.json").read_text())
OUT_PATH = LOG_DIR / "ablation_results.json"

ABLATIONS = ["mask_context_edges", "scramble_edge_signs"]

# Defined in run_gap_closer.py (the base module every EID campaign driver ultimately
# imports) -- re-exported here so callers that only import run_eid_ablations (e.g.
# run_eid_multiseed.py, as `ra.EID_EPOCH_OVERRIDE`) don't need a second import.
EID_EPOCH_OVERRIDE = rgc.EID_EPOCH_OVERRIDE


def winning_entry(dataset):
    """Best-of-{v1,v2} entry for a dataset, same discipline used to finalize
    the gap-closer table (max test_auc wins; bitcoin-alpha only has v2,
    bitcoin-otc only has v1)."""
    candidates = []
    if dataset in STATE_V1 and "best" in STATE_V1[dataset]:
        e = STATE_V1[dataset]
        params = e["stage1"]["params"]
        candidates.append((e["best"]["test_auc"], e["best"]["num_walks"], params, "v1"))
    if dataset in STATE_V2 and "best" in STATE_V2[dataset]:
        e = STATE_V2[dataset]
        params = e["best"]["params"]
        candidates.append((e["best"]["test_auc"], e["best"]["num_walks"], params, "v2"))
    if not candidates:
        raise RuntimeError(f"no gap-closer result found for {dataset}")
    candidates.sort(key=lambda c: c[0], reverse=True)
    test_auc, num_walks, params, source = candidates[0]
    return {"test_auc": test_auc, "num_walks": num_walks, "params": params, "source": source}


def build_common_overrides(params):
    embedding_dim = int(params["model.nhead"]) * int(params["model.head_dim"])
    overrides = [
        f"model.nhead={params['model.nhead']}",
        f"model.hidden_dim={params['model.hidden_dim']}",
        f"model.nlayers={params['model.nlayers']}",
        f"model.embedding_dim={embedding_dim}",
        f"model.head_dim={params['model.head_dim']}",
        f"model.dropout={params['model.dropout']}",
        f"training.lr={params['training.lr']}",
        f"model.node_replace_prob={params['model.node_replace_prob']}",
        f"model.node_replace_unk_ratio={params['model.node_replace_unk_ratio']}",
        f"model.edge_embed_rank={params['model.edge_embed_rank']}",
        f"model.edge_replace_prob={params['model.edge_replace_prob']}",
        f"model.edge_replace_unk_ratio={params['model.edge_replace_unk_ratio']}",
        "model.eid_reveal_holdout_identity=true",
    ]
    if int(params["model.edge_embed_rank"]) > 0:
        overrides += [
            f"model.edge_sign_combine={params['model.edge_sign_combine']}",
            f"model.edge_residual_baseline={str(params['model.edge_residual_baseline']).lower()}",
            f"model.sign_embed_dim={params['model.sign_embed_dim']}",
            f"model.node_embed_dim={params['model.node_embed_dim']}",
            f"model.edge_embed_weight_decay={params['model.edge_embed_weight_decay']}",
        ]
    return overrides


def run_ablation_campaign(datasets):
    jobs = []
    winners = {}
    for ds in datasets:
        w = winning_entry(ds)
        winners[ds] = w
        for ablation in ABLATIONS:
            jobs.append((ds, ablation, w))

    # Load-balance across GPUs by walk budget (proxy for training cost),
    # heaviest first, round-robin onto 4 buckets -- same pattern as
    # run_gap_closer's budget_sweep.
    jobs.sort(key=lambda j: j[2]["num_walks"], reverse=True)
    buckets = [[] for _ in range(N_GPUS)]
    for i, job in enumerate(jobs):
        buckets[i % N_GPUS].append(job)

    scripts = []
    for gpu, points in enumerate(buckets):
        if not points:
            continue
        lines = ["#!/bin/bash", "set -e", f"cd {REPO_ROOT}"]
        for ds, ablation, w in points:
            exp = f"EID_ABL2_{ds.upper().replace('-', '')}_{ablation.upper()}"
            common_str = " ".join(build_common_overrides(w["params"]))
            lines.append(f'echo "=== [$(date +%T)] START {exp} ==="')
            # No training.epochs override (bug fix 2026-09-15, see run_gap_closer.py's matching
            # note) -- let epochs flow through from configs/<dataset>.yaml (50 for 4 datasets,
            # 75 for epinions/slashdot090221, production's own deliberate per-dataset values).
            lines.append(
                f"{VENV_PY} experiments/edge_identity_tokens/run_eid.py "
                f"dataset.name={ds} dataset.num_walks={w['num_walks']} training.exp_name={exp} "
                f"training.batch_size=1024 {common_str} "
                f"model.{ablation}=true "
                f"--device {gpu} > {LOG_DIR}/{exp}.train.log 2>&1"
            )
            lines.append(f'exp_dir=$(ls -dt outputs/{ds}/{exp}_* | head -1)')
            lines.append(
                f"{VENV_PY} experiments/edge_identity_tokens/eid_posthoc.py "
                f'--exp-dir "$exp_dir" --device {gpu} --run-id posthoc '
                f"> {LOG_DIR}/{exp}.posthoc.log 2>&1"
            )
            lines.append(f'auc=$(grep "Edge agg_tr AUC" {LOG_DIR}/{exp}.posthoc.log | tail -1)')
            lines.append(f'echo "=== [$(date +%T)] DONE {exp} -- $auc ==="')
        script_path = LOG_DIR / f"ablation2_gpu{gpu}.sh"
        script_path.write_text("\n".join(lines) + "\n")
        script_path.chmod(0o755)
        scripts.append(script_path)

    print(f"[{time.strftime('%T')}] ablation campaign: {len(jobs)} jobs across {len(scripts)} GPUs", flush=True)
    procs = [subprocess.Popen([str(s)], cwd=str(REPO_ROOT),
                               stdout=open(str(s) + ".out", "w"), stderr=subprocess.STDOUT)
             for s in scripts]
    for p in procs:
        p.wait()

    results = {}
    for ds in datasets:
        results[ds] = {"baseline_test_auc": winners[ds]["test_auc"],
                        "baseline_num_walks": winners[ds]["num_walks"],
                        "baseline_source": winners[ds]["source"],
                        "production_auc": VANILLA_AUC[ds]}
        for ablation in ABLATIONS:
            exp = f"EID_ABL2_{ds.upper().replace('-', '')}_{ablation.upper()}"
            posthoc_log = LOG_DIR / f"{exp}.posthoc.log"
            if not posthoc_log.exists():
                results[ds][ablation] = None
                continue
            text = posthoc_log.read_text()
            val_auc = test_auc = None
            for line in text.splitlines():
                if "Edge agg_tr AUC" in line and "Edge test AUC" in line:
                    val_auc = float(line.split("Edge agg_tr AUC=")[1].split()[0])
                    test_auc = float(line.split("Edge test AUC=")[1].split()[0])
            results[ds][ablation] = {"val_auc": val_auc, "test_auc": test_auc}

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n[{time.strftime('%T')}] ABLATION CAMPAIGN COMPLETE -- {OUT_PATH}")
    print(f"{'dataset':<16}{'EID(no abl)':<14}{'mask_context':<14}{'scramble_sign':<14}{'production':<12}")
    for ds in datasets:
        r = results[ds]
        mc = r["mask_context_edges"]["test_auc"] if r["mask_context_edges"] else float("nan")
        ss = r["scramble_edge_signs"]["test_auc"] if r["scramble_edge_signs"] else float("nan")
        print(f"{ds:<16}{r['baseline_test_auc']:<14.4f}{mc:<14.4f}{ss:<14.4f}{r['production_auc']:<12.4f}")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", type=str, required=True)
    args = p.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    run_ablation_campaign(datasets)


if __name__ == "__main__":
    main()
