"""Multi-seed driver for SNEA and CopulaLSP -- same motivation as
scripts/run_multiseed_pewter.py: their published numbers have no error bars, so
we get real mean+-std ourselves instead (10 seeds: 42 reused + 43..51 new).

Both models are undirected/pair-level in our implementation (confirmed
2026-08-06): baselines/CopulaLSP/run_with_our_splits.py builds
`uni_edge_index = edge_index[:, edge_index[0] < edge_index[1]]` and trains on
`uni_trn/val/tst_mask` for both --model CopulaLSP and --model SNEA. This is not
a leak risk -- baselines/prepare_splits.py::build_canonical_split's uni_tst_mask
only marks a pair "clean test" if EVERY real direction of it is in the walk
model's own test split (asserted in _assert_canonical); anything with a
train-side direction goes to train instead. So the undirected pair-level split
is still derived from, and consistent with, the same frozen walk split every
other baseline uses -- not an independent/looser split.

Same underlying bug as GINEConv, fixed the same way: --seed only seeds model
init here, never picks a different split (the split file path is fixed per
dataset), and SPLITS_DIR defaults to the old non-canonical baselines/splits/
tree. Reuses baselines/prepare_splits.py::save_canonical_split_for_seed --
already built for GINEConv, same split file format, nothing new needed there.

Run under nohup, e.g.:
  nohup .venv/bin/python scripts/run_multiseed_snea_copulalsp.py \
      > logs/multiseed_snea_copulalsp/driver.log 2>&1 &
  disown

Resumable: every job checks for its own score.csv before running.
"""
import os
import subprocess
import sys
import threading
import time
from queue import Queue

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(REPO_ROOT, "logs", "multiseed_snea_copulalsp")
os.makedirs(LOG_DIR, exist_ok=True)

GPUS = [1, 2, 3]
DATASETS_ORDER = [
    "bitcoin-alpha", "wiki-elec", "bitcoin-otc", "wiki-rfa", "epinions", "slashdot090221",
]
NEW_SEEDS = list(range(43, 52))  # 43..51, 9 new; 42 already exists (canonical, our repro)
MODELS = ["SNEA", "CopulaLSP"]  # per dataset: all SNEA seeds, then all CopulaLSP seeds

COPULA_PY = "/home/dsi/shilo_avital/.conda/envs/copula_env/bin/python"
COPULA_DIR = os.path.join(REPO_ROOT, "baselines", "CopulaLSP")
RESULTS_ROOT = "results_our_splits_canonical"

sys.path.insert(0, os.path.join(REPO_ROOT, "baselines"))

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


def splits_dir_for_seed(dataset, seed):
    if seed == 42:
        return os.path.join(REPO_ROOT, "baselines", "splits_canonical")
    import prepare_splits  # baselines/prepare_splits.py
    out_path = prepare_splits.save_canonical_split_for_seed(dataset, seed)
    return os.path.dirname(out_path)


def job(dataset, model, seed, gpu):
    score_csv = os.path.join(COPULA_DIR, RESULTS_ROOT, dataset, model, f"seed{seed}", "score.csv")
    if os.path.isfile(score_csv):
        driver_log(f"[SKIP] {model} {dataset} seed={seed}: score.csv already exists")
        return
    try:
        splits_dir = splits_dir_for_seed(dataset, seed)
    except Exception as e:
        driver_log(f"[FAILED] {model} {dataset} seed={seed}: could not build canonical split: {e!r}")
        return
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["SPLITS_DIR"] = splits_dir
    cmd = [
        COPULA_PY, "run_with_our_splits.py",
        "--dataset", dataset, "--model", model, "--seed", str(seed),
        "--device", "cuda", "--out-dir", RESULTS_ROOT,
    ]
    log_path = os.path.join(LOG_DIR, f"{model}_{dataset}_s{seed}.log")
    driver_log(f"[START] {model} {dataset} seed={seed} on GPU {gpu} (SPLITS_DIR={splits_dir})")
    rc = run_logged(cmd, log_path, cwd=COPULA_DIR, env=env)
    driver_log(f"[{'DONE' if rc == 0 else 'FAILED'}] {model} {dataset} seed={seed} (rc={rc})")


def build_queue():
    jobs = []
    for ds in DATASETS_ORDER:
        for model in MODELS:
            for seed in NEW_SEEDS:
                jobs.append((ds, model, seed))
    return jobs


def worker(q, gpu):
    while True:
        item = q.get()
        if item is None:
            q.task_done()
            return
        try:
            job(*item, gpu)
        except Exception as e:
            driver_log(f"[EXCEPTION] job={item} gpu={gpu}: {e!r}")
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

    driver_log("=== SNEA/CopulaLSP multi-seed driver finished ===")


if __name__ == "__main__":
    main()
