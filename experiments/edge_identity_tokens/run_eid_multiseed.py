"""Phase 1 of plan-eid-multiseed-thesis.md: EID 10-seed multiseed campaign.

3 conditions (locked in 2026-09-15, see plan's Phase 0 resolution -- `scramble_edge_signs`
dropped in favor of `mask_context_sign_only`, which ties-or-beats it on all 6 datasets at
seed 42 while giving a cleaner "identity alone vs. both channels gone" contrast against
`mask_context_edges`):
  - no-ablation EID (each dataset's gap-closed "best" architecture+budget)
  - model.mask_context_edges=true
  - model.mask_context_sign_only=true

10 seeds (42, 43-51) x 6 datasets x 3 conditions = 180 logical jobs. Seed 42 is backfilled
from checkpoints that already exist wherever a real one was found (posthoc-only, no
retraining) -- see BASELINE_BACKFILL_DIRS/ablation glob patterns below. Every other
(dataset, condition, seed) combination trains from scratch, architecture/budget fixed at
that dataset's gap-closer winner, only reproducibility.seed varying -- exactly production's
own multiseed convention (scripts/run_multiseed_pewter.py).

Driver pattern: shared Queue + one worker thread per GPU (NOT run_gap_closer's per-GPU-
bucket-then-wait-all pattern -- that leaves GPUs idle once a bucket drains early, confirmed
wasteful at even 12 jobs; genuinely bad at 180). Resumable: every job checks whether its
posthoc output already exists before running, so a killed/restarted driver just resumes.

Usage (launch detached, per this project's standing background-job convention):
  nohup .venv/bin/python experiments/edge_identity_tokens/run_eid_multiseed.py \
      > logs/eid_multiseed/driver.log 2>&1 &
  disown
"""
import glob
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Queue

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import experiments.edge_identity_tokens.run_eid_ablations as ra  # noqa: E402

VENV_PY = ra.VENV_PY
LOG_DIR = REPO_ROOT / "logs" / "eid_multiseed"
LOG_DIR.mkdir(parents=True, exist_ok=True)

GPUS = [0, 1, 2, 3]
DATASETS = ["bitcoin-alpha", "bitcoin-otc", "wiki-elec", "wiki-rfa", "epinions", "slashdot090221"]
NEW_SEEDS = list(range(43, 52))  # 43..51, 9 new seeds; 42 backfilled (mostly)

# condition -> (cli flag or None, short tag used in exp_name/run_id)
CONDITIONS = [
    (None, "noablation"),
    ("mask_context_edges", "maskcontextedges"),
    ("mask_context_sign_only", "signonly"),
]

RUN_ID = "multiseed_agg"

# Seed-42 no-ablation baseline checkpoints, resolved by matching each dataset's
# gap-closer winning test_auc (logs/eid_gap_closer/state.json / state_v2.json) against
# every EID_GAP_* posthoc summary on disk -- verified unique match for 5/6 datasets
# (2026-09-15). wiki-rfa's v1 budget-sweep winner (mult=3.0, num_walks=531633,
# test_auc=0.8863) has no surviving checkpoint dir -- its EID_GAP_WIKIRFA_FINAL_* dir is a
# DIFFERENT (worse, 0.8753) refined run, not the actual winner -- so wiki-rfa's seed-42
# baseline is trained fresh instead of backfilled (see BASELINE_NO_BACKFILL below).
BASELINE_BACKFILL_DIRS = {
    "bitcoin-alpha": "outputs/bitcoin-alpha/EID_GAP_BITCOINALPHA_BUDGET_V2_8X_20260914-125053",
    "bitcoin-otc": "outputs/bitcoin-otc/EID_GAP_BITCOINOTC_BUDGET_8X_20260910-174418",
    "epinions": "outputs/epinions/EID_GAP_EPINIONS_BUDGET_V2_1.5X_20260914-164504",
    "wiki-elec": "outputs/wiki-elec/EID_GAP_WIKIELEC_BUDGET_V2_8X_20260914-132219",
    "slashdot090221": "outputs/slashdot090221/EID_GAP_SLASHDOT090221_BUDGET_V2_3X_20260914-232958",
}
BASELINE_NO_BACKFILL = {"wiki-rfa"}

sys.path.insert(0, str(REPO_ROOT / "baselines"))

_log_lock = threading.Lock()


def driver_log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with _log_lock:
        print(line, flush=True)


def run_logged(cmd, log_path, cwd=None):
    with open(log_path, "w") as f:
        f.write(f"$ {' '.join(cmd)}\n\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    return proc.returncode


def posthoc_done(exp_dir, run_id=RUN_ID):
    summary = Path(exp_dir) / "posthoc" / run_id / "aggregator" / "func_logit_power" / "summary.txt"
    return summary.is_file()


def latest_exp_dir(dataset, exp_name_prefix):
    matches = sorted(glob.glob(str(REPO_ROOT / "outputs" / dataset / f"{exp_name_prefix}_*")))
    return matches[-1] if matches else None


def do_posthoc(exp_dir, gpu, log_tag, run_id=RUN_ID):
    cmd = [
        VENV_PY, "experiments/edge_identity_tokens/eid_posthoc.py",
        "--exp-dir", str(exp_dir),
        "--device", str(gpu),
        "--run-id", run_id,
        "--agg-models", "func_logit_power",
    ]
    log_path = LOG_DIR / f"{log_tag}.posthoc.log"
    return run_logged(cmd, log_path, cwd=REPO_ROOT)


def ablation_backfill_dir(dataset, cond_flag):
    ds_upper = dataset.upper().replace("-", "")
    if cond_flag == "mask_context_edges":
        pattern = f"outputs/{dataset}/EID_ABL2_{ds_upper}_MASK_CONTEXT_EDGES_*"
    elif cond_flag == "mask_context_sign_only":
        pattern = f"outputs/{dataset}/EID_PHASE0_SIGNONLY_PILOT_*"
    else:
        raise ValueError(cond_flag)
    matches = sorted(glob.glob(str(REPO_ROOT / pattern)))
    return matches[-1] if matches else None


def job_backfill_seed42(dataset, cond_flag, cond_tag, gpu):
    log_tag = f"{dataset}_{cond_tag}_s42_backfill"
    if cond_flag is None:
        if dataset in BASELINE_NO_BACKFILL:
            driver_log(f"[SKIP-BACKFILL] {dataset} noablation seed42: no resolvable checkpoint, "
                       f"will be trained fresh instead")
            return
        exp_dir = BASELINE_BACKFILL_DIRS.get(dataset)
        if exp_dir is None:
            driver_log(f"[FAILED] {dataset} noablation seed42 backfill: no dir configured")
            return
    else:
        exp_dir = ablation_backfill_dir(dataset, cond_flag)
        if exp_dir is None:
            driver_log(f"[FAILED] {dataset} {cond_tag} seed42 backfill: no checkpoint dir found")
            return

    if posthoc_done(exp_dir):
        driver_log(f"[SKIP] {dataset} {cond_tag} seed42 backfill: already done ({exp_dir})")
        return

    driver_log(f"[START] backfill posthoc {dataset} {cond_tag} seed42 on GPU {gpu} ({exp_dir})")
    rc = do_posthoc(exp_dir, gpu, log_tag)
    driver_log(f"[{'DONE' if rc == 0 else 'FAILED'}] backfill {dataset} {cond_tag} seed42 (rc={rc})")


def job_train_and_posthoc(dataset, cond_flag, cond_tag, seed, gpu):
    exp_name = f"EID_MULTISEED_s{seed}_{cond_tag}"
    log_tag = f"{dataset}_{cond_tag}_s{seed}"

    existing = latest_exp_dir(dataset, exp_name)
    if existing is not None and posthoc_done(existing):
        driver_log(f"[SKIP] {dataset} {cond_tag} seed={seed}: already complete")
        return

    w = ra.winning_entry(dataset)
    overrides = ra.build_common_overrides(w["params"])
    cmd = [
        VENV_PY, "experiments/edge_identity_tokens/run_eid.py",
        f"dataset.name={dataset}", f"dataset.num_walks={w['num_walks']}",
        f"training.exp_name={exp_name}",
        # No blanket training.epochs override (bug fix 2026-09-15) -- this used to hardcode
        # 50 for every dataset, silently undoing configs/epinions.yaml's and
        # configs/slashdot090221.yaml's own epochs=75. Let it flow through from
        # configs/<dataset>.yaml instead, same as production's own run.py invocations do --
        # EXCEPT epinions/slashdot090221, which get an explicit, intentional EID-only
        # epochs=50 override (see ra.EID_EPOCH_OVERRIDE's docstring: a real diagnostic
        # showed slashdot090221 scores worse at 75 epochs for EID specifically, and a full
        # re-sweep to find EID's true best epoch/budget combo was judged too costly).
        "training.batch_size=1024",
        *overrides,
        f"reproducibility.seed={seed}",
    ]
    if dataset in ra.EID_EPOCH_OVERRIDE:
        cmd.append(f"training.epochs={ra.EID_EPOCH_OVERRIDE[dataset]}")
    if cond_flag is not None:
        cmd.append(f"model.{cond_flag}=true")
    cmd += ["--device", str(gpu)]

    driver_log(f"[START] train {dataset} {cond_tag} seed={seed} on GPU {gpu}")
    rc = run_logged(cmd, LOG_DIR / f"{log_tag}.train.log", cwd=REPO_ROOT)
    if rc != 0:
        driver_log(f"[FAILED] train {dataset} {cond_tag} seed={seed} (rc={rc})")
        return
    driver_log(f"[DONE] train {dataset} {cond_tag} seed={seed}")

    exp_dir = latest_exp_dir(dataset, exp_name)
    if exp_dir is None:
        driver_log(f"[FAILED] {dataset} {cond_tag} seed={seed}: no exp_dir found after training")
        return

    driver_log(f"[START] posthoc {dataset} {cond_tag} seed={seed} on GPU {gpu}")
    rc = do_posthoc(exp_dir, gpu, log_tag)
    driver_log(f"[{'DONE' if rc == 0 else 'FAILED'}] posthoc {dataset} {cond_tag} seed={seed} (rc={rc})")


def build_queue():
    """Ordered so the `noablation` condition -- across ALL 6 datasets -- fully drains
    before any ablation job starts, not per-dataset (2026-09-15 reorder, per the user:
    noablation is what unblocks most of the thesis's paper-asset reproduction, so it
    should finish first; ablations are the fill-in work that keeps GPUs busy once
    noablation is exhausted, not something interleaved with it dataset-by-dataset).
    Workers pull from one shared queue, so this ordering alone is sufficient -- no
    separate scheduling logic needed."""
    jobs = []
    noablation = [c for c in CONDITIONS if c[1] == "noablation"]
    ablations = [c for c in CONDITIONS if c[1] != "noablation"]

    # 1) noablation seed-42 backfills first (posthoc-only, fast) -- all datasets.
    for ds in DATASETS:
        for cond_flag, cond_tag in noablation:
            jobs.append(("backfill42", ds, cond_flag, cond_tag))
    # 1b) wiki-rfa's un-backfillable noablation baseline: train it fresh at seed 42.
    jobs.append(("train", "wiki-rfa", None, "noablation", 42))
    # 2) noablation main sweep: all datasets, 9 new seeds each -- must fully drain
    #    before any ablation job below starts.
    for ds in DATASETS:
        for cond_flag, cond_tag in noablation:
            for seed in NEW_SEEDS:
                jobs.append(("train", ds, cond_flag, cond_tag, seed))

    # 3) ablation seed-42 backfills -- all datasets, both ablations.
    for ds in DATASETS:
        for cond_flag, cond_tag in ablations:
            jobs.append(("backfill42", ds, cond_flag, cond_tag))
    # 4) ablation main sweep: all datasets, both ablations, 9 new seeds each.
    for ds in DATASETS:
        for cond_flag, cond_tag in ablations:
            for seed in NEW_SEEDS:
                jobs.append(("train", ds, cond_flag, cond_tag, seed))
    return jobs


def worker(q, gpu):
    while True:
        job = q.get()
        if job is None:
            q.task_done()
            return
        try:
            if job[0] == "backfill42":
                _, ds, cond_flag, cond_tag = job
                job_backfill_seed42(ds, cond_flag, cond_tag, gpu)
            elif job[0] == "train":
                _, ds, cond_flag, cond_tag, seed = job
                job_train_and_posthoc(ds, cond_flag, cond_tag, seed, gpu)
        except Exception as e:
            driver_log(f"[EXCEPTION] job={job} gpu={gpu}: {e!r}")
        finally:
            q.task_done()


def main():
    jobs = build_queue()
    driver_log(f"Queue built: {len(jobs)} jobs total, {len(GPUS)} workers (GPUs {GPUS})")

    q = Queue()
    for j in jobs:
        q.put(j)

    threads = [threading.Thread(target=worker, args=(q, gpu), daemon=True) for gpu in GPUS]
    for t in threads:
        t.start()

    q.join()
    driver_log("All jobs drained from queue. Sending stop sentinels.")
    for _ in GPUS:
        q.put(None)
    for t in threads:
        t.join()

    driver_log("=== EID multiseed driver finished ===")


if __name__ == "__main__":
    main()
