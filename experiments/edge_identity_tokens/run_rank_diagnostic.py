"""Cheap diagnostic (2026-09-16): does reducing edge_embed_rank (EID's per-edge identity
embedding capacity) close wiki-elec/wiki-rfa's gap to production?

Motivated by real evidence, not a guess:
  - wiki-elec (rank=20) and wiki-rfa (rank=26) are the two datasets with the LARGEST
    edge_embed_rank of all 6, despite being the two SMALLEST-edge-count datasets
    (103k/177k edges) -- epinions (only dataset EID beats production on) uses rank=11
    on 840k edges.
  - Train-val AUC gap at seed43 (noablation): epinions=0.020 (smallest), slashdot=-0.025
    (still underfitting), bitcoin-otc=0.048, wiki-elec=0.056, bitcoin-alpha=0.063,
    wiki-rfa=0.073 (LARGEST). wiki-rfa's catastrophic seed50 shows a 0.160 gap (train_auc
    0.92, val_auc 0.76) with a completely healthy/normal train-loss curve -- this is
    memorization, not an optimization/init failure.

Single-seed=42, same winning num_walks/architecture otherwise, only edge_embed_rank swept:
{0 (identity fully disabled, isolates whether the identity mechanism itself is the problem),
4, 8} vs the current winners (20/26) as reference (already have those numbers).

Usage:
  nohup .venv/bin/python experiments/edge_identity_tokens/run_rank_diagnostic.py \
      > logs/eid_gap_closer/rank_diagnostic_driver.log 2>&1 &
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
RANKS = [0, 4, 8]
DATASETS = ["wiki-elec", "wiki-rfa"]

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
    jobs = []
    for ds in DATASETS:
        w = ra.winning_entry(ds)
        for rank in RANKS:
            params = copy.deepcopy(w["params"])
            params["model.edge_embed_rank"] = rank
            jobs.append({
                "dataset": ds, "num_walks": w["num_walks"], "params": params,
                "exp": f"EID_RANKDIAG_{ds.upper().replace('-', '')}_RANK{rank}",
                "baseline_rank": w["params"]["model.edge_embed_rank"],
                "baseline_test_auc": w["test_auc"],
            })
    return jobs


def run_job(job, gpu):
    ds, nw, params, exp = job["dataset"], job["num_walks"], job["params"], job["exp"]
    overrides = ra.build_common_overrides(params)
    cmd = [
        VENV_PY, "experiments/edge_identity_tokens/run_eid.py",
        f"dataset.name={ds}", f"dataset.num_walks={nw}",
        f"training.exp_name={exp}", "training.batch_size=1024",
        *overrides, "--device", str(gpu),
    ]
    if ds in ra.EID_EPOCH_OVERRIDE:
        cmd.append(f"training.epochs={ra.EID_EPOCH_OVERRIDE[ds]}")
    driver_log(f"[START] train {exp} (rank={params['model.edge_embed_rank']}, baseline_rank={job['baseline_rank']}, "
               f"baseline_test_auc={job['baseline_test_auc']:.4f}) on GPU {gpu}")
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
        "--exp-dir", str(exp_dir), "--device", str(gpu), "--run-id", "rankdiag",
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
    driver_log(f"Rank diagnostic queue built: {len(jobs)} jobs, {len(GPUS)} workers (GPUs {GPUS})")
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

    driver_log("=== Rank diagnostic driver finished ===")


if __name__ == "__main__":
    main()
