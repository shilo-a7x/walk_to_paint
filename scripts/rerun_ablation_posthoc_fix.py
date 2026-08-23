"""One-off re-evaluation driver: after fixing the mask_node_tokens/mask_edge_tokens
posthoc-eval bug (src/training/callbacks.py, PerEpochPredictionSaver._extract_predictions
was not reapplying the token-masking ablation at inference time -- see
aaai2027/PAPER_CLOSEOUT_LOG.md's 2026-08-23 entry), re-run ONLY the posthoc
predictions+aggregator step (NOT training -- checkpoints are already correctly
trained) for all 120 existing MASKNODE/MASKEDGE checkpoints, overwriting their
posthoc/ablation_agg/ output with corrected numbers.

Reuses the exact same run_posthoc.py invocation shape as
scripts/run_ablation_campaign.py::do_posthoc, same run_id ("ablation_agg"), so the
downstream tex-writing step reads from the same path with no other change needed.
DIRFLIP is untouched -- confirmed unaffected by this bug (data-level implementation).
"""
import glob
import os
import subprocess
import threading
import time
from queue import Queue

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PY = os.path.join(REPO_ROOT, ".venv", "bin", "python")
LOG_DIR = os.path.join(REPO_ROOT, "logs", "ablation_posthoc_fix")
os.makedirs(LOG_DIR, exist_ok=True)

GPUS = [0, 1, 2, 3]
DATASETS = [
    "wiki-elec", "bitcoin-alpha", "bitcoin-otc", "wiki-rfa", "epinions", "slashdot090221",
]
SEEDS = list(range(42, 52))
ABLATIONS = ["MASKNODE", "MASKEDGE"]
AGG_FUNCS_CSV = "func_logit_power"

_log_lock = threading.Lock()


def driver_log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with _log_lock:
        print(line, flush=True)


def latest_exp_dir(dataset, exp_name_prefix):
    matches = sorted(glob.glob(os.path.join(REPO_ROOT, "outputs", dataset, f"{exp_name_prefix}_*")))
    return matches[-1] if matches else None


def run_logged(cmd, log_path):
    with open(log_path, "w") as f:
        f.write(f"$ {' '.join(cmd)}\n\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT)
    return proc.returncode


def do_posthoc(exp_dir, dataset, gpu, log_tag):
    cmd = [
        VENV_PY, "run_posthoc.py",
        "--exp-dir", exp_dir + "/",
        "--artifacts", "predictions,aggregator",
        "--agg-models", AGG_FUNCS_CSV,
        "--device", str(gpu),
        "--run-id", "ablation_agg",
        f"dataset.name={dataset}",
    ]
    log_path = os.path.join(LOG_DIR, f"{log_tag}.posthoc.log")
    return run_logged(cmd, log_path)


def build_jobs():
    jobs = []
    for ablation_tag in ABLATIONS:
        for dataset in DATASETS:
            for seed in SEEDS:
                exp_name = f"ABLATION_{ablation_tag}_s{seed}"
                exp_dir = latest_exp_dir(dataset, exp_name)
                if exp_dir is None:
                    driver_log(f"[MISSING] {dataset} {ablation_tag} seed={seed}: no exp dir found, skipping")
                    continue
                jobs.append((dataset, ablation_tag, seed, exp_dir))
    return jobs


def worker(gpu, q, done_counter, total):
    while True:
        try:
            dataset, ablation_tag, seed, exp_dir = q.get_nowait()
        except Exception:
            return
        log_tag = f"{dataset}_{ablation_tag}_s{seed}"
        driver_log(f"[GPU{gpu}] START {log_tag}")
        rc = do_posthoc(exp_dir, dataset, gpu, log_tag)
        with _log_lock:
            done_counter[0] += 1
        status = "OK" if rc == 0 else f"FAIL(rc={rc})"
        driver_log(f"[GPU{gpu}] DONE  {log_tag} -> {status}  ({done_counter[0]}/{total})")
        q.task_done()


def main():
    jobs = build_jobs()
    total = len(jobs)
    driver_log(f"Queued {total} posthoc re-eval jobs (expected 120)")
    q = Queue()
    for j in jobs:
        q.put(j)

    done_counter = [0]
    threads = []
    for gpu in GPUS:
        t = threading.Thread(target=worker, args=(gpu, q, done_counter, total), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    driver_log(f"All done: {done_counter[0]}/{total}")


if __name__ == "__main__":
    main()
