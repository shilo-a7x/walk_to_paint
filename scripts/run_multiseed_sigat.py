"""Multi-seed driver for SiGAT (raw, not SGA-augmented) -- same motivation as
scripts/run_multiseed_pewter.py and scripts/run_multiseed_snea_copulalsp.py:
get real mean+-std AUC over 10 splits (seed 42 reused + 43..51 new) instead of
a single split, both for Table 1 and for averaging the fixed-entropy-bin AUC
heatmap (scripts/lead4_entropy_heterogeneity.py) across splits.

SiGAT is directed/edge-level in this codebase's implementation (unlike
SNEA/CopulaLSP) -- baselines/SGA/run_with_our_splits.py trains on trn_mask/
tst_mask directly, no uni_edge_index. Entropy bins used downstream
(src_ent/tgt_ent per node) are computed from the fixed real dense edge set,
independent of split, so per-split AUC grids can be averaged cell-by-cell
without re-deriving the binning.

Same underlying bug as GINEConv/SNEA/CopulaLSP, fixed the same way: --seed
only seeds model init here, never picks a different split (the split file
path is fixed per dataset), and SPLITS_DIR defaults to the old non-canonical
baselines/splits/ tree. Reuses baselines/prepare_splits.py::
save_canonical_split_for_seed -- already built for GINEConv/SNEA/CopulaLSP,
same split file format, nothing new needed there.

Run under nohup, e.g.:
  nohup .venv/bin/python scripts/run_multiseed_sigat.py \
      > logs/multiseed_sigat/driver.log 2>&1 &
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
LOG_DIR = os.path.join(REPO_ROOT, "logs", "multiseed_sigat")
os.makedirs(LOG_DIR, exist_ok=True)

GPUS = [1, 2, 3]
DATASETS_ORDER = [
    "bitcoin-alpha", "wiki-elec", "bitcoin-otc", "wiki-rfa", "epinions", "slashdot090221",
]
NEW_SEEDS = list(range(43, 52))  # 43..51, 9 new; 42 already exists (canonical, our repro)

SIGAT_PY = "/home/dsi/shilo_avital/.conda/envs/sga_env/bin/python"
SIGAT_DIR = os.path.join(REPO_ROOT, "baselines", "SGA")
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


def job(dataset, seed, gpu):
    score_csv = os.path.join(SIGAT_DIR, RESULTS_ROOT, dataset, "SiGAT", f"seed{seed}", "score.csv")
    if os.path.isfile(score_csv):
        driver_log(f"[SKIP] SiGAT {dataset} seed={seed}: score.csv already exists")
        return
    try:
        splits_dir = splits_dir_for_seed(dataset, seed)
    except Exception as e:
        driver_log(f"[FAILED] SiGAT {dataset} seed={seed}: could not build canonical split: {e!r}")
        return
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["SPLITS_DIR"] = splits_dir
    env["RESULTS_ROOT"] = RESULTS_ROOT
    cmd = [
        SIGAT_PY, "run_with_our_splits.py",
        "--dataset", dataset, "--seed", str(seed), "--device", "cuda",
    ]
    log_path = os.path.join(LOG_DIR, f"SiGAT_{dataset}_s{seed}.log")
    driver_log(f"[START] SiGAT {dataset} seed={seed} on GPU {gpu} (SPLITS_DIR={splits_dir})")
    rc = run_logged(cmd, log_path, cwd=SIGAT_DIR, env=env)
    driver_log(f"[{'DONE' if rc == 0 else 'FAILED'}] SiGAT {dataset} seed={seed} (rc={rc})")


def build_queue():
    jobs = []
    for ds in DATASETS_ORDER:
        for seed in NEW_SEEDS:
            jobs.append((ds, seed))
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

    driver_log("=== SiGAT multi-seed driver finished ===")


if __name__ == "__main__":
    main()
