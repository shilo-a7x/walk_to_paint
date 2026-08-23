"""Multi-seed training driver: trains each (dataset, attention variant) combination
across 10 seeds and runs posthoc aggregation on each, so mean+-std AUC can be computed
across seeds instead of relying on a single split.

- 10 seeds per (dataset, attention variant): 42..51.
- Dataset order, smallest to largest by production num_walks budget:
  bitcoin-alpha, wiki-elec, bitcoin-otc, wiki-rfa, epinions, slashdot090221.
- Per dataset: all 10 full-attention seeds are queued before any of that dataset's
  local-attention seeds (strict per-dataset ordering, not global).
- Every posthoc run computes all 11 registered aggregator weight functions (not just
  the func_logit_power default), so all of them are available for every seed without
  a second pass later.

Run under nohup, e.g.:
  nohup .venv/bin/python scripts/run_multiseed.py \
      > logs/multiseed/driver.log 2>&1 &
  disown

Resumable: every job checks whether its output already exists before running, so a
killed/restarted driver just picks up where it left off.

Adjust GPUS below to whichever CUDA device indices are available on your machine.
"""
import glob
import os
import subprocess
import threading
import time
from queue import Queue

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PY = os.path.join(REPO_ROOT, ".venv", "bin", "python")
LOG_DIR = os.path.join(REPO_ROOT, "logs", "multiseed")
os.makedirs(LOG_DIR, exist_ok=True)

GPUS = [0, 1, 2, 3]
DATASETS_ORDER = [
    "bitcoin-alpha",   # 120,930 walks
    "wiki-elec",       # 155,534
    "bitcoin-otc",     # 177,960
    "wiki-rfa",        # 265,817
    "epinions",        # 840,799
    "slashdot090221",  # 1,647,606
]
SEEDS = list(range(42, 52))  # 42..51 inclusive, 10 seeds

AGG_FUNCS = [
    "func_uniform", "func_conf_power", "func_conf_exp", "func_conf_cert",
    "func_conf_logit", "func_logq_power", "func_logit_power",
    "func_entropy_power", "func_entropy_exp", "func_fisher_power",
    "func_maxprob_power",
]
AGG_FUNCS_CSV = ",".join(AGG_FUNCS)

_log_lock = threading.Lock()


def driver_log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with _log_lock:
        print(line, flush=True)


def run_logged(cmd, log_path, cwd=None, env=None):
    with open(log_path, "w") as f:
        f.write(f"$ {' '.join(cmd)}\n\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
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
    rc = run_logged(cmd, log_path, cwd=REPO_ROOT)
    return rc


def job_train_and_posthoc(dataset, variant, seed, gpu):
    exp_name = f"MULTISEED_s{seed}_{variant}"
    log_tag = f"{dataset}_{variant}_s{seed}"

    existing = latest_exp_dir(dataset, exp_name)
    run_id = "multiseed_agg"
    if existing is not None and posthoc_done(existing, run_id):
        driver_log(f"[SKIP] {dataset} {variant} seed={seed}: already complete")
        return

    # Always (re)train here rather than trusting a `last.ckpt` on disk as proof
    # training finished -- PyTorch Lightning writes last.ckpt every epoch, so a
    # driver restart mid-training would otherwise see a real-but-partial checkpoint
    # and skip straight to posthoc on an under-trained model, silently producing a
    # wrong AUC for that seed. Retraining is cheap enough at these dataset sizes
    # that this is the safe default; a fresh timestamped exp_dir is created each
    # time, so any stale partial run from an earlier interrupted attempt is just
    # left on disk (harmless) rather than resumed from.
    attn_override = "model.local_attention_window=null" if variant == "full" else "model.local_attention_window=4"
    cmd = [
        VENV_PY, "run.py",
        "--device", str(gpu),
        f"dataset.name={dataset}",
        f"training.exp_name={exp_name}",
        attn_override,
        f"reproducibility.seed={seed}",
    ]
    driver_log(f"[START] train {dataset} {variant} seed={seed} on GPU {gpu}")
    rc = run_logged(cmd, os.path.join(LOG_DIR, f"{log_tag}.train.log"), cwd=REPO_ROOT)
    if rc != 0:
        driver_log(f"[FAILED] train {dataset} {variant} seed={seed} (rc={rc}) -- see {log_tag}.train.log")
        return
    driver_log(f"[DONE] train {dataset} {variant} seed={seed}")

    exp_dir = latest_exp_dir(dataset, exp_name)
    if exp_dir is None:
        driver_log(f"[FAILED] {dataset} {variant} seed={seed}: no exp_dir found after training")
        return

    driver_log(f"[START] posthoc (11 funcs) {dataset} {variant} seed={seed} on GPU {gpu}")
    rc = do_posthoc(exp_dir, dataset, run_id, gpu, log_tag)
    driver_log(f"[{'DONE' if rc == 0 else 'FAILED'}] posthoc {dataset} {variant} seed={seed} (rc={rc})")


def build_queue():
    jobs = []
    # per dataset: full (10 seeds) then local (10 seeds)
    for ds in DATASETS_ORDER:
        for variant in ("full", "local"):
            for seed in SEEDS:
                jobs.append(("train", ds, variant, seed))
    return jobs


def worker(q, gpu):
    while True:
        job = q.get()
        if job is None:
            q.task_done()
            return
        try:
            if job[0] == "train":
                _, ds, variant, seed = job
                job_train_and_posthoc(ds, variant, seed, gpu)
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

    driver_log("=== Multi-seed driver finished ===")


if __name__ == "__main__":
    main()
