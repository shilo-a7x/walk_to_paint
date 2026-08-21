"""Vertex/edge token-masking ablation campaign (mask variant only, per user sign-off
2026-08-21 after a single-seed bitcoin-alpha pilot confirmed both `mask_node_tokens`
and `mask_edge_tokens` train cleanly and produce sensible, distinct AUC drops).

Two ablations, 6 datasets, 10 seeds each = 120 train+posthoc jobs:
  - mask_node_tokens: every vertex token -> <UNK> (train and eval), isolates how much
    signal edge signs alone carry.
  - mask_edge_tokens: every edge token -> <MASK> (train and eval), isolates how much
    signal vertex identity alone carries.
Local attention only (LocalAttn4, the production default) -- no full-attention side.

Ordering, per explicit user instruction:
  - Ablation-major: all 60 mask_node_tokens jobs enqueued before any mask_edge_tokens
    job. Both blocks share one FIFO queue drained by 4 worker threads (one per GPU),
    so a GPU that frees up early may start pulling from the edge block while 1-2 other
    GPUs are still finishing the last (slowest) vertex-block jobs -- this is the only
    way to satisfy "vertex before edge" AND "no GPU ever idle" simultaneously; the
    overlap is at most a few jobs at the block boundary, not a real violation of the
    ordering intent.
  - Within each ablation, datasets ordered easy-to-heavy by *measured* wall-clock time
    (not walk-budget count, which does not track measured time -- e.g. bitcoin-alpha
    has fewer walks than wiki-elec but measured slower): wiki-elec (3.3 min/seed),
    bitcoin-alpha (4.7), bitcoin-otc (6.0), wiki-rfa (6.6), epinions (15.6),
    slashdot090221 (24.9). Source: logs/multiseed/<ds>_local_s43.* mtimes, 2026-08-06
    LocalAttn4 campaign.

No dataset cache is touched by either ablation -- both are pure input-rewrite flags
applied inside LitEdgeClassifier._step, so every run reuses the existing production
data/<dataset>/dataset_cache.pt unmodified (cache-hit), same as any other config sweep.

Comparison baseline: the already-existing Table 1 "Pewter (local attention)" 10-seed
numbers, func_logit_power aggregator -- no new baseline runs needed. Posthoc here
computes func_logit_power ONLY (not all 11 aggregator functions like the original
Ablation B campaign) since the question is "how much does removing this token type
cost", not aggregator-choice robustness; cuts posthoc cost ~11x for no loss of
information relevant to this ablation.

Run under nohup:
  nohup .venv/bin/python scripts/run_ablation_campaign.py \
      > logs/ablation_campaign/driver.log 2>&1 &
  disown

Resumable: every job checks whether its posthoc output already exists before
(re)running, so a killed/restarted driver just picks up where it left off.

Live extension (no restart needed): this driver never exits on its own once the
initial 120-job queue drains. A background watcher thread polls EXTRA_JOBS_DIR
every 30s for new *.json job-batch files (format: a list of
{"dataset":..., "tag":..., "flag":..., "seed":...} objects) and pushes their
contents straight onto the same live queue the 4 GPU workers are draining -- so a
job dropped there fires on the next free GPU within 30s, no restart, no idle gap.
Use scripts/enqueue_extra_ablation_jobs.py to drop a batch (e.g. once ablation A's
exact mechanism is agreed). The driver keeps polling indefinitely until a STOP file
is dropped at logs/ablation_campaign/STOP, at which point it finishes draining
whatever's queued and exits cleanly.
"""
import glob
import json
import os
import subprocess
import threading
import time
from queue import Queue

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PY = os.path.join(REPO_ROOT, ".venv", "bin", "python")
LOG_DIR = os.path.join(REPO_ROOT, "logs", "ablation_campaign")
EXTRA_JOBS_DIR = os.path.join(LOG_DIR, "extra_jobs")
EXTRA_JOBS_PROCESSED_DIR = os.path.join(EXTRA_JOBS_DIR, "_processed")
STOP_FILE = os.path.join(LOG_DIR, "STOP")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(EXTRA_JOBS_PROCESSED_DIR, exist_ok=True)

GPUS = [0, 1, 2, 3]

DATASETS_EASY_TO_HEAVY = [
    "wiki-elec",        # ~3.3 min/seed measured
    "bitcoin-alpha",    # ~4.7 min/seed
    "bitcoin-otc",      # ~6.0 min/seed
    "wiki-rfa",         # ~6.6 min/seed
    "epinions",         # ~15.6 min/seed
    "slashdot090221",   # ~24.9 min/seed
]
SEEDS = list(range(42, 52))  # 42..51 inclusive, 10 seeds -- all fresh, no backfill

ABLATIONS = [
    # (tag, model config flag)
    ("MASKNODE", "model.mask_node_tokens=true"),
    ("MASKEDGE", "model.mask_edge_tokens=true"),
]

AGG_FUNCS = ["func_logit_power"]
AGG_FUNCS_CSV = ",".join(AGG_FUNCS)

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


def latest_exp_dir(dataset, exp_name_prefix):
    matches = sorted(glob.glob(os.path.join(REPO_ROOT, "outputs", dataset, f"{exp_name_prefix}_*")))
    return matches[-1] if matches else None


def posthoc_done(exp_dir, run_id):
    agg_dir = os.path.join(exp_dir, "posthoc", run_id, "aggregator")
    if not os.path.isdir(agg_dir):
        return False
    for fn in AGG_FUNCS:
        if not os.path.isfile(os.path.join(agg_dir, fn, "summary.txt")):
            return False
    return True


def do_posthoc(exp_dir, dataset, run_id, gpu, log_tag):
    cmd = [
        VENV_PY, "run_posthoc.py",
        "--exp-dir", exp_dir + "/",
        "--artifacts", "predictions,aggregator",
        "--agg-models", AGG_FUNCS_CSV,
        "--device", str(gpu),
        "--run-id", run_id,
        f"dataset.name={dataset}",
    ]
    log_path = os.path.join(LOG_DIR, f"{log_tag}.posthoc.log")
    return run_logged(cmd, log_path, cwd=REPO_ROOT)


def job_train_and_posthoc(dataset, ablation_tag, ablation_flag, seed, gpu):
    exp_name = f"ABLATION_{ablation_tag}_s{seed}"
    log_tag = f"{dataset}_{ablation_tag}_s{seed}"
    run_id = "ablation_agg"

    existing = latest_exp_dir(dataset, exp_name)
    if existing is not None and posthoc_done(existing, run_id):
        driver_log(f"[SKIP] {dataset} {ablation_tag} seed={seed}: already complete")
        return

    cmd = [
        VENV_PY, "run.py",
        "--device", str(gpu),
        f"dataset.name={dataset}",
        f"training.exp_name={exp_name}",
        "model.local_attention_window=4",
        *ablation_flag.split(),  # supports multiple space-separated overrides, e.g.
                                  # "model.randomize_edge_direction=true model.edge_direction_randomize_prob=1.0"
        f"reproducibility.seed={seed}",
    ]
    driver_log(f"[START] train {dataset} {ablation_tag} seed={seed} on GPU {gpu}")
    rc = run_logged(cmd, os.path.join(LOG_DIR, f"{log_tag}.train.log"), cwd=REPO_ROOT)
    if rc != 0:
        driver_log(f"[FAILED] train {dataset} {ablation_tag} seed={seed} (rc={rc}) -- see {log_tag}.train.log")
        return
    driver_log(f"[DONE] train {dataset} {ablation_tag} seed={seed}")

    exp_dir = latest_exp_dir(dataset, exp_name)
    if exp_dir is None:
        driver_log(f"[FAILED] {dataset} {ablation_tag} seed={seed}: no exp_dir found after training")
        return

    driver_log(f"[START] posthoc {dataset} {ablation_tag} seed={seed} on GPU {gpu}")
    rc = do_posthoc(exp_dir, dataset, run_id, gpu, log_tag)
    driver_log(f"[{'DONE' if rc == 0 else 'FAILED'}] posthoc {dataset} {ablation_tag} seed={seed} (rc={rc})")


def build_queue():
    jobs = []
    for ablation_tag, ablation_flag in ABLATIONS:  # vertex block, then edge block
        for ds in DATASETS_EASY_TO_HEAVY:
            for seed in SEEDS:
                jobs.append((ds, ablation_tag, ablation_flag, seed))
    return jobs


def worker(q, gpu):
    while True:
        job = q.get()
        if job is None:
            q.task_done()
            return
        try:
            job_train_and_posthoc(*job, gpu=gpu)
        except Exception as e:
            driver_log(f"[EXCEPTION] job={job} gpu={gpu}: {e!r}")
        finally:
            q.task_done()


def watch_for_extra_jobs(q, poll_interval=30):
    """Poll EXTRA_JOBS_DIR for new *.json job-batch files and push their contents
    onto the live queue. Runs for the driver's whole lifetime; returns (stops
    watching) once STOP_FILE appears, letting the driver drain and exit."""
    while True:
        if os.path.exists(STOP_FILE):
            driver_log("STOP file detected -- watcher exiting, no more extra-job batches will be accepted.")
            return
        for path in sorted(glob.glob(os.path.join(EXTRA_JOBS_DIR, "*.json"))):
            try:
                with open(path) as f:
                    batch = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                driver_log(f"[WARN] could not read extra-job batch {path}: {e!r} (will retry next poll)")
                continue
            # Claim via atomic rename first, so a file mid-write by the enqueue
            # helper (which itself writes-then-renames) is never double-processed.
            dest = os.path.join(EXTRA_JOBS_PROCESSED_DIR, os.path.basename(path))
            try:
                os.rename(path, dest)
            except OSError:
                continue
            for spec in batch:
                q.put((spec["dataset"], spec["tag"], spec["flag"], int(spec["seed"])))
            driver_log(f"[EXTRA] enqueued {len(batch)} jobs from {os.path.basename(path)}")
        time.sleep(poll_interval)


def main():
    jobs = build_queue()
    driver_log(f"Queue built: {len(jobs)} jobs total, {len(GPUS)} workers (GPUs {GPUS})")
    driver_log(
        f"Watching {EXTRA_JOBS_DIR} for additional ablation job batches "
        f"(drop a *.json file there any time, e.g. via enqueue_extra_ablation_jobs.py; "
        f"a STOP file at {STOP_FILE} ends the watch once the current queue drains)."
    )

    q = Queue()
    for j in jobs:
        q.put(j)

    threads = [threading.Thread(target=worker, args=(q, gpu), daemon=True) for gpu in GPUS]
    for t in threads:
        t.start()

    watcher = threading.Thread(target=watch_for_extra_jobs, args=(q,), daemon=True)
    watcher.start()
    watcher.join()  # returns only once a STOP file is dropped

    q.join()
    driver_log("Queue drained after STOP. Sending stop sentinels.")
    for _ in GPUS:
        q.put(None)
    for t in threads:
        t.join()

    driver_log("=== Ablation campaign driver finished ===")


if __name__ == "__main__":
    main()
