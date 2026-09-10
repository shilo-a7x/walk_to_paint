"""No-idle-GPU pipeline: per-dataset EID architecture Optuna (seeded from that
dataset's own production config -- see optuna_eid.py's seed-trial logic, already
dataset-aware) -> budget sweep (selected by EDGE-LEVEL VAL AUC, i.e. posthoc's
"agg_tr" number -- NOT test AUC, see the 2026-09-10 discussion: budget affects the
final metric through both per-walk transformer quality AND aggregation/ensembling
variance reduction, and val_auc_epoch only sees the first) -> refinement Optuna
(architecture re-searched with budget FIXED at the sweep's winner, seeded with the
first stage's winning config as an explicit trial). Processes datasets sequentially
in priority order; within each stage, uses all --n-gpus GPUs in parallel so none
sit idle.

Usage:
  .venv/bin/python experiments/edge_identity_tokens/run_gap_closer.py

Resumable: writes progress to logs/eid_gap_closer/state.json after every stage:
rerunning the script skips any dataset/stage already recorded there. Delete a
dataset's entry (or the whole file) to force a redo.
"""
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VENV_PY = str(REPO_ROOT / ".venv" / "bin" / "python")
LOG_DIR = REPO_ROOT / "logs" / "eid_gap_closer"
LOG_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = LOG_DIR / "state.json"
N_GPUS = 4

sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import optuna
from optuna.storages import JournalStorage, JournalFileStorage

# Production 10-seed LocalAttn4 reference AUC (CLAUDE.md "Current SOTA"), used only
# for AbsoluteFloorPruning's floor -- not a target, just a safety net against
# obviously-bad trials.
VANILLA_AUC = {
    "bitcoin-alpha": 0.9134, "bitcoin-otc": 0.9317, "epinions": 0.9536,
    "wiki-elec": 0.9023, "wiki-rfa": 0.8914, "slashdot090221": 0.8968,
}
# |E| per dataset (CLAUDE.md "Walk sampler" table), used to build each budget grid.
EDGE_COUNT = {
    "bitcoin-alpha": 24186, "bitcoin-otc": 35592, "epinions": 840799,
    "wiki-elec": 103689, "wiki-rfa": 177211, "slashdot090221": 549202,
}
BUDGET_GRID_MULT = [1.0, 1.5, 3.0, 5.0, 8.0]

COMMON_ARCH_KEYS = [
    "model.edge_embed_rank", "model.edge_sign_combine", "model.edge_residual_baseline",
    "model.sign_embed_dim", "model.node_embed_dim", "model.edge_embed_weight_decay",
    "model.nhead", "model.hidden_dim", "model.nlayers", "model.head_dim",
    "model.edge_replace_prob", "model.edge_replace_unk_ratio",
    "model.node_replace_prob", "model.node_replace_unk_ratio",
    "training.lr", "model.dropout",
]


def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2))


def run(cmd, log_path):
    with open(log_path, "w") as f:
        return subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT)


def optuna_study_best(study_name):
    optuna_dir = REPO_ROOT / "outputs" / "optuna" / study_name / "optuna"
    # optuna_eid.py resolves this via resolve_outputs_dirs; find the real journal file.
    candidates = list((REPO_ROOT / "outputs").glob(f"*/{study_name}/optuna/optuna_eid_study.log"))
    if not candidates:
        candidates = list((REPO_ROOT / "outputs").glob(f"*/{study_name}_*/optuna/optuna_eid_study.log"))
    if not candidates:
        raise RuntimeError(f"Could not find journal storage for study {study_name}")
    storage = JournalStorage(JournalFileStorage(str(candidates[0])))
    study = optuna.load_study(study_name=f"eid_optuna_{study_name}", storage=storage)
    return study.best_params, study.best_value


def launch_optuna_workers(dataset, num_walks, study_name, n_trials_per_gpu, n_gpus,
                           seed_params=None):
    """Launch n_gpus parallel optuna_eid.py workers sharing one study. num_walks=None
    means "don't override -- let configs/<dataset>.yaml's own production default flow
    through" (used for stage 1, the architecture search). seed_params, if given, gets
    passed through --extra-seed-json to enqueue it as an extra seed trial (stage 3,
    refinement, seeded from stage 1's winner)."""
    procs = []
    for d in range(n_gpus):
        cmd = [
            VENV_PY, "experiments/edge_identity_tokens/optuna_eid.py",
            f"dataset.name={dataset}",
        ]
        if num_walks is not None:
            cmd.append(f"dataset.num_walks={num_walks}")
        cmd += [
            f"training.exp_name={study_name}", "--device", str(d),
            "--n-trials", str(n_trials_per_gpu),
            "--total-trials", str(n_trials_per_gpu * n_gpus),
            "--vanilla-auc", str(VANILLA_AUC[dataset]),
        ]
        if seed_params is not None:
            cmd += ["--extra-seed-json", json.dumps(seed_params)]
        log_path = LOG_DIR / f"{study_name}_gpu{d}.log"
        procs.append(subprocess.Popen(cmd, cwd=str(REPO_ROOT),
                                       stdout=open(log_path, "w"), stderr=subprocess.STDOUT))
    print(f"[{time.strftime('%T')}] {study_name}: {n_gpus} workers launched "
          f"(num_walks={num_walks}, {n_trials_per_gpu}/gpu, total target "
          f"{n_trials_per_gpu * n_gpus})", flush=True)
    for p in procs:
        p.wait()
    best_params, best_value = optuna_study_best(study_name)
    print(f"[{time.strftime('%T')}] {study_name}: DONE, best val_auc={best_value:.4f}", flush=True)
    return best_params, best_value


def arch_params_to_cli(params):
    """Map a searched-param dict (Optuna keys) -> full run_eid.py CLI override list."""
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


def budget_sweep(dataset, arch_params, exp_prefix, n_gpus):
    edge_count = EDGE_COUNT[dataset]
    grid = [(mult, int(round(edge_count * mult))) for mult in BUDGET_GRID_MULT]
    common = arch_params_to_cli(arch_params)
    common_str = " ".join(common)

    # Round-robin split grid points across n_gpus, one sequential script per GPU.
    buckets = [[] for _ in range(n_gpus)]
    for i, (mult, nw) in enumerate(grid):
        buckets[i % n_gpus].append((mult, nw))

    scripts = []
    for gpu, points in enumerate(buckets):
        if not points:
            continue
        lines = ["#!/bin/bash", "set -e", f"cd {REPO_ROOT}"]
        for mult, nw in points:
            tag = f"{mult}X".replace(".0X", "X")
            exp = f"{exp_prefix}_{tag}"
            lines.append(f'echo "=== [$(date +%T)] START train {exp} (num_walks={nw}) ==="')
            lines.append(
                f"{VENV_PY} experiments/edge_identity_tokens/run_eid.py "
                f"dataset.name={dataset} dataset.num_walks={nw} training.exp_name={exp} "
                f"training.epochs=50 training.batch_size=1024 {common_str} "
                f"--device {gpu} > {LOG_DIR}/{exp}.train.log 2>&1"
            )
            lines.append(f'exp_dir=$(ls -dt outputs/{dataset}/{exp}_* | head -1)')
            lines.append(
                f"{VENV_PY} experiments/edge_identity_tokens/eid_posthoc.py "
                f'--exp-dir "$exp_dir" --device {gpu} --run-id posthoc '
                f"> {LOG_DIR}/{exp}.posthoc.log 2>&1"
            )
            lines.append(
                f'auc=$(grep "Edge agg_tr AUC" {LOG_DIR}/{exp}.posthoc.log | tail -1)'
            )
            lines.append(f'echo "=== [$(date +%T)] DONE {exp} -- $auc ==="')
        script_path = LOG_DIR / f"{exp_prefix}_gpu{gpu}.sh"
        script_path.write_text("\n".join(lines) + "\n")
        os.chmod(script_path, 0o755)
        scripts.append(script_path)

    print(f"[{time.strftime('%T')}] {exp_prefix}: budget sweep launched, "
          f"{len(grid)} points across {len(scripts)} GPUs", flush=True)
    procs = [subprocess.Popen([str(s)], cwd=str(REPO_ROOT),
                               stdout=open(str(s) + ".out", "w"), stderr=subprocess.STDOUT)
             for s in scripts]
    for p in procs:
        p.wait()

    # Parse every posthoc log for this prefix, pick best by VAL (agg_tr) AUC.
    results = []
    for mult, nw in grid:
        tag = f"{mult}X".replace(".0X", "X")
        exp = f"{exp_prefix}_{tag}"
        posthoc_log = LOG_DIR / f"{exp}.posthoc.log"
        if not posthoc_log.exists():
            continue
        text = posthoc_log.read_text()
        for line in text.splitlines():
            if "Edge agg_tr AUC" in line and "Edge test AUC" in line:
                val_auc = float(line.split("Edge agg_tr AUC=")[1].split()[0])
                test_auc = float(line.split("Edge test AUC=")[1].split()[0])
                results.append({"mult": mult, "num_walks": nw, "val_auc": val_auc, "test_auc": test_auc})
    if not results:
        raise RuntimeError(f"{exp_prefix}: no budget sweep results parsed")
    best = max(results, key=lambda r: r["val_auc"])
    print(f"[{time.strftime('%T')}] {exp_prefix}: best budget = {best['num_walks']} "
          f"(val_auc={best['val_auc']:.4f}, test_auc={best['test_auc']:.4f})", flush=True)
    return best, results


def process_dataset(dataset, state, skip_arch_search=False, known_arch=None, known_arch_val=None):
    entry = state.setdefault(dataset, {})

    # Stage 1: architecture search (skipped for datasets with an already-known winner).
    if "stage1" not in entry:
        if skip_arch_search and known_arch is not None:
            entry["stage1"] = {"params": known_arch, "val_auc": known_arch_val, "source": "prior_search"}
            print(f"[{time.strftime('%T')}] {dataset} stage1: reusing prior search result "
                  f"(val_auc={known_arch_val:.4f})", flush=True)
        else:
            study_name = f"EID_GAP_{dataset.upper().replace('-', '')}_ARCH"
            params, val_auc = launch_optuna_workers(
                dataset, None,  # num_walks=None -> use configs/<dataset>.yaml's own production default
                study_name, n_trials_per_gpu=6, n_gpus=N_GPUS,
            )
            entry["stage1"] = {"params": params, "val_auc": val_auc, "study": study_name}
        save_state(state)

    # Stage 2: budget sweep, selected by VAL (agg_tr) AUC.
    if "stage2" not in entry:
        arch_params = entry["stage1"]["params"]
        exp_prefix = f"EID_GAP_{dataset.upper().replace('-', '')}_BUDGET"
        best, all_results = budget_sweep(dataset, arch_params, exp_prefix, n_gpus=N_GPUS)
        entry["stage2"] = {"best": best, "all": all_results}
        save_state(state)

    # Stage 3: refinement Optuna, budget fixed at stage2's winner, seeded with stage1's winner.
    if "stage3" not in entry:
        best_budget = entry["stage2"]["best"]["num_walks"]
        seed_params = entry["stage1"]["params"]
        study_name = f"EID_GAP_{dataset.upper().replace('-', '')}_REFINE"
        params, val_auc = launch_optuna_workers(
            dataset, best_budget, study_name, n_trials_per_gpu=4, n_gpus=N_GPUS,
            seed_params=seed_params,
        )
        entry["stage3"] = {"params": params, "val_auc": val_auc, "num_walks": best_budget}
        save_state(state)

    # Final validation: train+posthoc the stage3 winner at its budget, report real edge test AUC.
    if "final" not in entry:
        arch_params = entry["stage3"]["params"]
        num_walks = entry["stage3"]["num_walks"]
        overrides = arch_params_to_cli(arch_params)
        exp = f"EID_GAP_{dataset.upper().replace('-', '')}_FINAL"
        cmd = [VENV_PY, "experiments/edge_identity_tokens/run_eid.py",
               f"dataset.name={dataset}", f"dataset.num_walks={num_walks}",
               f"training.exp_name={exp}", "training.epochs=50", "training.batch_size=1024",
               *overrides, "--device", "0"]
        run(cmd, LOG_DIR / f"{exp}.train.log")
        exp_dir = sorted((REPO_ROOT / "outputs" / dataset).glob(f"{exp}_*"))[-1]
        posthoc_cmd = [VENV_PY, "experiments/edge_identity_tokens/eid_posthoc.py",
                        "--exp-dir", str(exp_dir), "--device", "0", "--run-id", "posthoc"]
        run(posthoc_cmd, LOG_DIR / f"{exp}.posthoc.log")
        text = (LOG_DIR / f"{exp}.posthoc.log").read_text()
        for line in text.splitlines():
            if "Edge agg_tr AUC" in line and "Edge test AUC" in line:
                val_auc = float(line.split("Edge agg_tr AUC=")[1].split()[0])
                test_auc = float(line.split("Edge test AUC=")[1].split()[0])
                entry["final"] = {"val_auc": val_auc, "test_auc": test_auc, "num_walks": num_walks}
        save_state(state)
        gap = VANILLA_AUC[dataset] - entry["final"]["test_auc"]
        print(f"[{time.strftime('%T')}] {dataset} FINAL: test_auc={entry['final']['test_auc']:.4f} "
              f"(production {VANILLA_AUC[dataset]:.4f}, gap {gap*100:+.2f}pp)", flush=True)


def main():
    state = load_state()

    # Wiki-elec/wiki-rfa: architecture already searched this session -- skip stage 1,
    # go straight to a budget sweep re-check (cheap, and the earlier one used the wrong
    # selection metric) + refinement at the correct budget.
    KNOWN = {
        "wiki-elec": (
            {"model.edge_embed_rank": 20, "model.edge_sign_combine": "concat",
             "model.edge_residual_baseline": False, "model.sign_embed_dim": 20,
             "model.node_embed_dim": 64, "model.edge_embed_weight_decay": 0.0003,
             "model.nhead": 2, "model.hidden_dim": 64, "model.nlayers": 5,
             "model.head_dim": 32, "model.edge_replace_prob": 0.5,
             "model.edge_replace_unk_ratio": 0.23, "model.node_replace_prob": 0.49,
             "model.node_replace_unk_ratio": 0.65, "training.lr": 0.0016135042002810406,
             "model.dropout": 0.14736239940786994},
            0.8517,
        ),
        "wiki-rfa": (
            {"model.edge_embed_rank": 26, "model.edge_sign_combine": "concat",
             "model.edge_residual_baseline": False, "model.sign_embed_dim": 12,
             "model.node_embed_dim": 64, "model.edge_embed_weight_decay": 0.00010981640364482443,
             "model.nhead": 2, "model.hidden_dim": 128, "model.nlayers": 4,
             "model.head_dim": 24, "model.edge_replace_prob": 0.4139291642556332,
             "model.edge_replace_unk_ratio": 0.4959839660772761, "model.node_replace_prob": 0.4611897499261957,
             "model.node_replace_unk_ratio": 0.7613063072310584, "training.lr": 0.0032976773541742344,
             "model.dropout": 0.1899347167631212},
            0.8766,
        ),
    }

    order = ["wiki-elec", "wiki-rfa", "bitcoin-otc", "epinions", "slashdot090221"]
    for dataset in order:
        print(f"\n{'='*70}\n{dataset}\n{'='*70}", flush=True)
        if dataset in KNOWN:
            arch, val = KNOWN[dataset]
            process_dataset(dataset, state, skip_arch_search=True, known_arch=arch, known_arch_val=val)
        else:
            process_dataset(dataset, state)

    print("\n" + "=" * 70)
    print("GAP CLOSER PIPELINE COMPLETE")
    print("=" * 70)
    for dataset in order:
        f = state.get(dataset, {}).get("final")
        if f:
            gap = VANILLA_AUC[dataset] - f["test_auc"]
            print(f"  {dataset}: test_auc={f['test_auc']:.4f}  gap={gap*100:+.2f}pp  num_walks={f['num_walks']}")


if __name__ == "__main__":
    main()
