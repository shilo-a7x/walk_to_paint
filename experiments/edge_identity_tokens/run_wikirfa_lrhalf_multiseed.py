"""Follow-up (2026-09-16) to run_wikirfa_lr_diagnostic.py's LRHALF finding: at seed 42/50,
halving wiki-rfa's learning rate (0.00330 -> 0.001649) nearly eliminated the seed-42-vs-
seed-50 gap (12.6pp -> 0.28pp) at a small cost to seed 42's peak (0.8863 -> 0.8762). That's
promising enough on the true 10-seed mean (baseline's 0.8585 is dragged down hard by seed
50's 0.7602 collapse) to check properly rather than guess -- this script trains the
remaining 8 seeds (43-49, 51) under LRHALF to get a complete, directly comparable 10-seed
number against baseline's 0.8585 +/- 0.0335.

Usage:
  nohup .venv/bin/python experiments/edge_identity_tokens/run_wikirfa_lrhalf_multiseed.py \
      > logs/eid_gap_closer/wikirfa_lrhalf_multiseed_driver.log 2>&1 &
  disown
"""
import copy
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Queue

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import experiments.edge_identity_tokens.run_eid_ablations as ra  # noqa: E402

VENV_PY = ra.VENV_PY
LOG_DIR = REPO_ROOT / "logs" / "eid_gap_closer"
LOG_DIR.mkdir(parents=True, exist_ok=True)

GPUS = [0, 1, 2, 3]
DATASET = "wiki-rfa"
SEEDS = [43, 44, 45, 46, 47, 48, 49, 51]  # 42 and 50 already done in run_wikirfa_lr_diagnostic.py

_log_lock = threading.Lock()


def driver_log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with _log_lock:
        print(line, flush=True)


def run_logged(cmd, log_path):
    with open(log_path, "w") as f:
        f.write(f"$ {' '.join(cmd)}\n\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT)
    return proc.returncode


def build_jobs():
    w = ra.winning_entry(DATASET)
    base_lr = float(w["params"]["training.lr"])
    params = copy.deepcopy(w["params"])
    params["training.lr"] = base_lr * 0.5
    jobs = []
    for seed in SEEDS:
        jobs.append({
            "dataset": DATASET, "num_walks": w["num_walks"], "params": params,
            "seed": seed, "exp": f"EID_WIKIRFALR_LRHALF_s{seed}",
            "lr": params["training.lr"],
        })
    return jobs


def run_job(job, gpu):
    ds, nw, params, seed, exp = job["dataset"], job["num_walks"], job["params"], job["seed"], job["exp"]
    overrides = ra.build_common_overrides(params)
    cmd = [
        VENV_PY, "experiments/edge_identity_tokens/run_eid.py",
        f"dataset.name={ds}", f"dataset.num_walks={nw}",
        f"training.exp_name={exp}", "training.batch_size=1024",
        *overrides, f"reproducibility.seed={seed}", "--device", str(gpu),
    ]
    driver_log(f"[START] train {exp} (lr={job['lr']:.6f}, seed={seed}) on GPU {gpu}")
    rc = run_logged(cmd, LOG_DIR / f"{exp}.train.log")
    if rc != 0:
        driver_log(f"[FAILED] train {exp} (rc={rc})")
        return
    driver_log(f"[DONE] train {exp}")

    exp_dirs = sorted((REPO_ROOT / "outputs" / ds).glob(f"{exp}_*"))
    if not exp_dirs:
        driver_log(f"[FAILED] {exp}: no exp_dir found after training")
        return
    exp_dir = exp_dirs[-1]

    driver_log(f"[START] posthoc {exp} on GPU {gpu}")
    posthoc_cmd = [
        VENV_PY, "experiments/edge_identity_tokens/eid_posthoc.py",
        "--exp-dir", str(exp_dir), "--device", str(gpu), "--run-id", "wikirfalr",
    ]
    rc = run_logged(posthoc_cmd, LOG_DIR / f"{exp}.posthoc.log")
    driver_log(f"[{'DONE' if rc == 0 else 'FAILED'}] posthoc {exp} (rc={rc})")


def worker(q, gpu):
    while True:
        job = q.get()
        if job is None:
            q.task_done()
            return
        try:
            run_job(job, gpu)
        except Exception as e:
            driver_log(f"[EXCEPTION] job={job['exp']} gpu={gpu}: {e!r}")
        finally:
            q.task_done()


def main():
    jobs = build_jobs()
    driver_log(f"wiki-rfa LRHALF multiseed completion: {len(jobs)} jobs, {len(GPUS)} workers (GPUs {GPUS})")
    for j in jobs:
        driver_log(f"  queued: {j['exp']}")

    q = Queue()
    for j in jobs:
        q.put(j)

    threads = [threading.Thread(target=worker, args=(q, gpu), daemon=True) for gpu in GPUS]
    for t in threads:
        t.start()

    q.join()
    driver_log("All jobs drained. Sending stop sentinels.")
    for _ in GPUS:
        q.put(None)
    for t in threads:
        t.join()

    driver_log("=== wiki-rfa LRHALF multiseed completion finished ===")


if __name__ == "__main__":
    main()
