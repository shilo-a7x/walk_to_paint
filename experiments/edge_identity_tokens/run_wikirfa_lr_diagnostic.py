"""Cheap diagnostic (2026-09-16), follow-up to run_wikirfa_regularization_diagnostic.py:
does a LOWER learning rate stabilize wiki-rfa's seed 50 collapse?

Both prior regularization ideas (10x edge_embed_weight_decay, boosted edge_replace_prob)
FAILED to fix seed 50 -- and weight_decay x10 made it markedly WORSE (0.5136 test AUC,
collapsed further toward random than the original 0.7602). That's a meaningful negative
result: MORE regularization pressure did not help and actively hurt, which argues against
a pure overfitting/undertrained-regularizer story and toward an optimization-instability
story instead (a too-large step early in training landing weights in a bad, poorly-
generalizing basin -- consistent with a perfectly smooth train_loss/train_auc curve, since
that failure mode doesn't require divergence or NaN, just an unlucky big step).

Supporting fact: wiki-rfa's winning architecture has the HIGHEST learning rate of any of
the 6 datasets (0.0033, vs 0.0022/0.0016/0.0016/0.00008/0.00039 for
alpha/otc/elec/epinions/slashdot) -- a classic lever for exactly this kind of seed-
dependent instability.

Two LR variants, seeds {42 (reference), 50 (known-bad)}, everything else unchanged:
  - HALF: lr x0.5 (0.00330 -> 0.00165)
  - QUARTER: lr x0.25 (0.00330 -> 0.000824)

Usage:
  nohup .venv/bin/python experiments/edge_identity_tokens/run_wikirfa_lr_diagnostic.py \
      > logs/eid_gap_closer/wikirfa_lr_diagnostic_driver.log 2>&1 &
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
    "LRHALF": 0.5,
    "LRQUARTER": 0.25,
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
    base_lr = float(w["params"]["training.lr"])
    jobs = []
    for variant_name, mult in VARIANTS.items():
        params = copy.deepcopy(w["params"])
        params["training.lr"] = base_lr * mult
        for seed in SEEDS:
            jobs.append({
                "dataset": DATASET, "num_walks": w["num_walks"], "params": params,
                "seed": seed, "exp": f"EID_WIKIRFALR_{variant_name}_s{seed}",
                "variant": variant_name, "lr": params["training.lr"],
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
    driver_log(f"[START] train {exp} (variant={job['variant']}, lr={job['lr']:.6f}, seed={seed}) on GPU {gpu}")
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
    driver_log(f"wiki-rfa LR diagnostic: {len(jobs)} jobs, {len(GPUS)} workers (GPUs {GPUS})")
    for j in jobs:
        driver_log(f"  queued: {j['exp']} (lr={j['lr']:.6f})")

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

    driver_log("=== wiki-rfa LR diagnostic finished ===")


if __name__ == "__main__":
    main()
