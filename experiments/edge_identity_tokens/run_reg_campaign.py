"""Live-extendable EID regularization campaign driver (2026-09-24).

One worker thread per GPU pulls jobs from a shared queue. A watcher thread polls
logs/eid_reg/queue/*.json every 30s; each file is a JSON list of jobs, pushed onto the live
queue and then moved to queue/consumed/. Drop a new file to add work without restarting.
The driver exits once logs/eid_reg/STOP exists and the queue is drained.

Job schema: {"name": str, "dataset": str, "num_walks": int, "seed": int, "overrides": [str]}
exp_name = EIDREG_<name>_<DS>_s<seed>. Skipped if its posthoc summary already exists.
EID caches are built under a global lock (parallel builds of one cache broke a run 2026-09-16).
Every finished job appends a row to logs/eid_reg/results.csv (val = aggregator-train split =
the val edges; test = test edges; both func_logit_power edge AUC).

Usage:
  nohup .venv/bin/python experiments/edge_identity_tokens/run_reg_campaign.py \
      > logs/eid_reg/driver.log 2>&1 &
"""
import csv
import glob
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Queue

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.edge_identity_tokens.run_eid import EID_CACHE_PATH, ensure_eid_cache  # noqa: E402
from src.utils.config import load_config  # noqa: E402

VENV_PY = str(REPO_ROOT / ".venv" / "bin" / "python")
BASE = REPO_ROOT / "logs" / "eid_reg"
QUEUE_DIR = BASE / "queue"
CONSUMED = QUEUE_DIR / "consumed"
JOB_LOGS = BASE / "jobs"
RESULTS = BASE / "results.csv"
STOP = BASE / "STOP"
GPUS = [0, 1, 2, 3]
RUN_ID = "reg"
_SUMMARY = "posthoc/{run}/aggregator/func_logit_power/summary.txt"

_log_lock = threading.Lock()
_cache_lock = threading.Lock()
_results_lock = threading.Lock()


def log(msg):
    with _log_lock:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def exp_name(job):
    return f"EIDREG_{job['name']}_{job['dataset'].upper().replace('-', '')}_s{job['seed']}"


def read_aucs(exp_dir):
    path = Path(exp_dir) / _SUMMARY.format(run=RUN_ID)
    if not path.is_file():
        return None, None
    text = path.read_text()
    v = re.search(r"Train AUC:\s+([0-9.]+)", text)
    t = re.search(r"Test\s+AUC:\s+([0-9.]+)", text)
    return (float(v.group(1)) if v else None), (float(t.group(1)) if t else None)


def latest_dir(job):
    dirs = sorted(glob.glob(str(REPO_ROOT / "outputs" / job["dataset"] / f"{exp_name(job)}_*")))
    return dirs[-1] if dirs else None


def ensure_cache(job):
    path = EID_CACHE_PATH.format(dataset=job["dataset"], num_walks=job["num_walks"], seed=job["seed"])
    with _cache_lock:
        if not Path(path).is_file():
            log(f"[CACHE] building {path}")
            cfg = load_config("config.yaml", overrides=[f"dataset.name={job['dataset']}",
                                                        f"dataset.num_walks={job['num_walks']}",
                                                        f"reproducibility.seed={job['seed']}"])
            ensure_eid_cache(cfg, path)


def run_logged(cmd, log_path):
    with open(log_path, "w") as f:
        f.write(f"$ {' '.join(cmd)}\n\n")
        f.flush()
        return subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT).returncode


def record(job, val, test, status):
    with _results_lock:
        new = not RESULTS.is_file()
        with open(RESULTS, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "name", "dataset", "seed", "val_auc", "test_auc", "status", "overrides"])
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), job["name"], job["dataset"], job["seed"],
                        val, test, status, " ".join(job["overrides"])])


def run_job(job, gpu):
    name = exp_name(job)
    existing = latest_dir(job)
    if existing and read_aucs(existing)[1] is not None:
        log(f"[SKIP] {name}: already complete")
        return
    ensure_cache(job)
    cmd = [VENV_PY, "experiments/edge_identity_tokens/run_eid.py",
           f"dataset.name={job['dataset']}", f"dataset.num_walks={job['num_walks']}",
           f"training.exp_name={name}", f"reproducibility.seed={job['seed']}",
           *job["overrides"], "--device", str(gpu)]
    log(f"[START] {name} on GPU {gpu}")
    if run_logged(cmd, JOB_LOGS / f"{name}.train.log") != 0:
        log(f"[FAILED] train {name}")
        record(job, None, None, "train_failed")
        return
    exp_dir = latest_dir(job)
    rc = run_logged([VENV_PY, "experiments/edge_identity_tokens/eid_posthoc.py", "--exp-dir", exp_dir,
                     "--device", str(gpu), "--run-id", RUN_ID, "--agg-models", "func_logit_power"],
                    JOB_LOGS / f"{name}.posthoc.log")
    val, test = read_aucs(exp_dir)
    status = "ok" if rc == 0 and test is not None else "posthoc_failed"
    log(f"[DONE] {name} val={val} test={test} ({status})")
    record(job, val, test, status)


def worker(q, gpu):
    while True:
        job = q.get()
        if job is None:
            q.task_done()
            return
        try:
            run_job(job, gpu)
        except Exception as e:
            log(f"[EXCEPTION] {exp_name(job)} gpu={gpu}: {e!r}")
            record(job, None, None, f"exception: {e!r}")
        finally:
            q.task_done()


def watcher(q):
    while True:
        for path in sorted(QUEUE_DIR.glob("*.json")):
            try:
                jobs = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue  # file still being written; retry next poll
            shutil.move(str(path), str(CONSUMED / path.name))
            for job in jobs:
                q.put(job)
            log(f"[QUEUE] +{len(jobs)} jobs from {path.name}")
        time.sleep(30)


def main():
    for d in (QUEUE_DIR, CONSUMED, JOB_LOGS):
        d.mkdir(parents=True, exist_ok=True)
    q = Queue()
    threading.Thread(target=watcher, args=(q,), daemon=True).start()
    threads = [threading.Thread(target=worker, args=(q, g), daemon=True) for g in GPUS]
    for t in threads:
        t.start()
    log(f"reg campaign driver up, GPUs {GPUS}, watching {QUEUE_DIR}")
    while True:
        time.sleep(30)
        if STOP.is_file() and q.unfinished_tasks == 0 and not list(QUEUE_DIR.glob("*.json")):
            break
    for _ in GPUS:
        q.put(None)
    for t in threads:
        t.join()
    log("=== reg campaign driver finished ===")


if __name__ == "__main__":
    main()
