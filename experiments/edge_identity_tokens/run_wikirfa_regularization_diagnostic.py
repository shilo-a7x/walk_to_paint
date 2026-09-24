"""Cheap diagnostic (2026-09-16): does stronger regularization stabilize wiki-rfa's
architecture, which shows catastrophic single-run failures on two independent axes
(seed 50's train/val gap of 0.160 vs. 0.073 at a good seed; edge_embed_rank=4 collapsing
to ~random, 0.5035) despite identity being genuinely load-bearing there (the
mask_context_edges ablation drops it 0.8585->0.7324, the biggest hit of any dataset --
see EID_OVERFITTING_INVESTIGATION_20260916.md).

Two regularization variants, tested at seed 42 (reference) and seed 50 (the known-bad
seed) -- if either variant lifts seed 50 well above its current 0.7602 without hurting
seed 42, that is real evidence worth expanding to a full re-sweep (needs sign-off before
committing further). Everything else fixed at wiki-rfa's current winning architecture.

  - A: edge_embed_weight_decay x10 (0.00011 -> 0.0011) -- direct L2 shrinkage on the
    identity table without reducing its rank (unlike the rank-diagnostic, which showed
    reducing rank costs real accuracy on wiki-rfa specifically).
  - B: edge_replace_prob boosted (0.414 -> 0.65, roughly matching wiki-elec/slashdot's
    higher values) -- more train-time identity corruption as a regularizer.

Usage:
  nohup .venv/bin/python experiments/edge_identity_tokens/run_wikirfa_regularization_diagnostic.py \
      > logs/eid_gap_closer/wikirfa_reg_diagnostic_driver.log 2>&1 &
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
SEEDS = [42, 50]

VARIANTS = {
    "WD10X": lambda p: {**p, "model.edge_embed_weight_decay": p["model.edge_embed_weight_decay"] * 10},
    "REPLACE065": lambda p: {**p, "model.edge_replace_prob": 0.65},
}

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
    jobs = []
    for variant_name, fn in VARIANTS.items():
        params = copy.deepcopy(fn(w["params"]))
        for seed in SEEDS:
            jobs.append({
                "dataset": DATASET, "num_walks": w["num_walks"], "params": params,
                "seed": seed, "exp": f"EID_WIKIRFAREG_{variant_name}_s{seed}",
                "variant": variant_name,
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
    driver_log(f"[START] train {exp} (variant={job['variant']}, seed={seed}) on GPU {gpu}")
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
        "--exp-dir", str(exp_dir), "--device", str(gpu), "--run-id", "wikirfareg",
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
    driver_log(f"wiki-rfa regularization diagnostic: {len(jobs)} jobs, {len(GPUS)} workers (GPUs {GPUS})")
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

    driver_log("=== wiki-rfa regularization diagnostic finished ===")


if __name__ == "__main__":
    main()
