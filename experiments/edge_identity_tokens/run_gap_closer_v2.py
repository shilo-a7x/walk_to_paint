"""Gap-closer round 2, 2026-09-14: cheap-only version of run_gap_closer.py.

Per dataset: architecture Optuna (stage1, seeded from that dataset's own production
config, using per-dataset widened search bounds -- see optuna_eid.py's
DATASET_RANGE_OVERRIDES) -> budget sweep (stage2, selected by edge-level VAL AUC,
i.e. posthoc's "agg_tr" number, never test AUC). That's it -- NO refinement stage
(stage3/final) this round: the original run_gap_closer.py's refinement made 4 of 5
datasets WORSE on the real edge-level metric (it re-searches architecture by the
transformer's own per-walk val_auc_epoch, a proxy that doesn't reliably transfer to
the aggregated edge-level metric actually reported) while being by far the most
expensive stage (13+ hours for slashdot090221 alone at its 8x budget) -- dropped
per explicit user instruction ("i want cheap runs").

Separate state file (state_v2.json) from run_gap_closer.py's state.json so the prior
run's results stay on disk as a historical record, untouched.

Usage:
  .venv/bin/python experiments/edge_identity_tokens/run_gap_closer_v2.py \
      --datasets bitcoin-alpha,wiki-elec,wiki-rfa

  .venv/bin/python experiments/edge_identity_tokens/run_gap_closer_v2.py \
      --datasets epinions,slashdot090221

Resumable the same way as v1: writes state_v2.json after every stage; rerunning
skips whatever's already recorded. Delete a dataset's entry to force a redo.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import experiments.edge_identity_tokens.run_gap_closer as rgc  # noqa: E402

REPO_ROOT = rgc.REPO_ROOT
LOG_DIR = rgc.LOG_DIR
VANILLA_AUC = rgc.VANILLA_AUC
EDGE_COUNT = rgc.EDGE_COUNT
N_GPUS = rgc.N_GPUS
STATE_PATH = LOG_DIR / "state_v2.json"

# Per-dataset budget grid: epinions/slashdot090221 drop the 8x point per explicit
# user instruction ("dont do x8 for epinions and slashdot it is too heavy") -- 8x
# was the single most expensive point (slashdot090221's own 8x refinement alone took
# 13h22min in the v1 run) for a small marginal AUC gain (+0.4pp over 5x on
# slashdot090221; 5x/8x/1.5x/3x were all within noise on epinions, see the 2026-09-14
# session's value-for-money table).
BUDGET_GRID = {
    "bitcoin-alpha": [1.0, 1.5, 3.0, 5.0, 8.0],
    "wiki-elec": [1.0, 1.5, 3.0, 5.0, 8.0],
    "wiki-rfa": [1.0, 1.5, 3.0, 5.0, 8.0],
    "epinions": [1.0, 1.5, 3.0, 5.0],
    "slashdot090221": [1.0, 1.5, 3.0, 5.0],
}

# Arch-search trial count: epinions gets more trials (14-dim search space, largest
# production budget among these, and the dataset the user specifically asked to push
# further -- "epinions can do a bit more work as the 0.95 num looked nice"). Everyone
# else keeps the original 24 (6/gpu x 4) for cost control -- explicitly NOT increased
# for slashdot090221 per the same cost-control instruction.
TRIALS_PER_GPU = {
    "bitcoin-alpha": 6, "wiki-elec": 6, "wiki-rfa": 6,
    "epinions": 10, "slashdot090221": 6,
}


def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2))


def budget_sweep(dataset, arch_params, exp_prefix, n_gpus, grid_mult):
    """Same as run_gap_closer.budget_sweep but with an explicit (per-dataset) grid,
    instead of that module's fixed global BUDGET_GRID_MULT."""
    edge_count = EDGE_COUNT[dataset]
    grid = [(mult, int(round(edge_count * mult))) for mult in grid_mult]
    common_str = " ".join(rgc.arch_params_to_cli(arch_params))

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
                f"{rgc.VENV_PY} experiments/edge_identity_tokens/run_eid.py "
                f"dataset.name={dataset} dataset.num_walks={nw} training.exp_name={exp} "
                f"training.epochs=50 training.batch_size=1024 {common_str} "
                f"--device {gpu} > {LOG_DIR}/{exp}.train.log 2>&1"
            )
            lines.append(f'exp_dir=$(ls -dt outputs/{dataset}/{exp}_* | head -1)')
            lines.append(
                f"{rgc.VENV_PY} experiments/edge_identity_tokens/eid_posthoc.py "
                f'--exp-dir "$exp_dir" --device {gpu} --run-id posthoc '
                f"> {LOG_DIR}/{exp}.posthoc.log 2>&1"
            )
            lines.append(f'auc=$(grep "Edge agg_tr AUC" {LOG_DIR}/{exp}.posthoc.log | tail -1)')
            lines.append(f'echo "=== [$(date +%T)] DONE {exp} -- $auc ==="')
        script_path = LOG_DIR / f"{exp_prefix}_gpu{gpu}.sh"
        script_path.write_text("\n".join(lines) + "\n")
        script_path.chmod(0o755)
        scripts.append(script_path)

    print(f"[{time.strftime('%T')}] {exp_prefix}: budget sweep launched, "
          f"{len(grid)} points across {len(scripts)} GPUs (grid={grid_mult})", flush=True)
    import subprocess
    procs = [subprocess.Popen([str(s)], cwd=str(REPO_ROOT),
                               stdout=open(str(s) + ".out", "w"), stderr=subprocess.STDOUT)
             for s in scripts]
    for p in procs:
        p.wait()

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


def process_dataset(dataset, state):
    entry = state.setdefault(dataset, {})

    if "stage1" not in entry:
        study_name = f"EID_GAP_{dataset.upper().replace('-', '')}_ARCH_V2"
        n_tpg = TRIALS_PER_GPU[dataset]
        params, val_auc = rgc.launch_optuna_workers(
            dataset, None, study_name, n_trials_per_gpu=n_tpg, n_gpus=N_GPUS,
        )
        entry["stage1"] = {"params": params, "val_auc": val_auc, "study": study_name}
        save_state(state)

    if "stage2" not in entry:
        arch_params = entry["stage1"]["params"]
        exp_prefix = f"EID_GAP_{dataset.upper().replace('-', '')}_BUDGET_V2"
        best, all_results = budget_sweep(dataset, arch_params, exp_prefix, N_GPUS, BUDGET_GRID[dataset])
        entry["stage2"] = {"best": best, "all": all_results}
        save_state(state)

    if "best" not in entry:
        s2 = entry["stage2"]["best"]
        entry["best"] = {"val_auc": s2["val_auc"], "test_auc": s2["test_auc"],
                          "num_walks": s2["num_walks"], "source": "stage2",
                          "params": entry["stage1"]["params"]}
        save_state(state)
        gap = VANILLA_AUC[dataset] - entry["best"]["test_auc"]
        print(f"[{time.strftime('%T')}] {dataset} BEST: test_auc={entry['best']['test_auc']:.4f} "
              f"(production {VANILLA_AUC[dataset]:.4f}, gap {gap*100:+.2f}pp)", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", type=str, required=True,
                    help="Comma-separated dataset order for this invocation, e.g. "
                         "bitcoin-alpha,wiki-elec,wiki-rfa")
    args = p.parse_args()
    order = [d.strip() for d in args.datasets.split(",") if d.strip()]

    state = load_state()
    for dataset in order:
        print(f"\n{'='*70}\n{dataset}\n{'='*70}", flush=True)
        process_dataset(dataset, state)

    print("\n" + "=" * 70)
    print("GAP CLOSER V2 BATCH COMPLETE")
    print("=" * 70)
    for dataset in order:
        b = state.get(dataset, {}).get("best")
        if b:
            gap = VANILLA_AUC[dataset] - b["test_auc"]
            print(f"  {dataset}: test_auc={b['test_auc']:.4f}  gap={gap*100:+.2f}pp  "
                  f"num_walks={b['num_walks']}")


if __name__ == "__main__":
    main()
