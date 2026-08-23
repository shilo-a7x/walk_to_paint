"""SIGNSCRAMBLE ablation campaign: 6 datasets x 10 seeds = 60 train+posthoc jobs.

New ablation, model.scramble_edge_signs=true (src/model/lit_model.py). Every VISIBLE
(non-masked-target, non-split-excluded) edge-sign token shows a fixed, independently-random
sign instead of its true one -- the flip decision is drawn once per edge id (seeded off
cfg.reproducibility.seed + a fixed offset), not per occurrence or per epoch, so a given edge
shows the same (possibly wrong) sign everywhere it appears, and the ablation is automatically
consistent with dynamic resplit (an epoch's freshly-sampled targets are already masked before
scrambling runs). Motivation: mask_edge_tokens (removing the edge token's identity entirely)
showed no significant AUC effect on any of 6 datasets once its own posthoc-eval bug was fixed
(see PAPER_CLOSEOUT_LOG.md / ABLATION_MASKNODE_MASKEDGE_TRUE_RESULTS.md) -- but that conflates
"does an edge token being present matter" with "does its specific sign value matter".
SIGNSCRAMBLE isolates the second question directly by keeping a real (but decorrelated) sign
value in place of the true one, instead of removing the token altogether.

Pilot-validated before launching this: 6 synthetic unit tests (determinism, target/split-
excluded exclusion, correct flip-vs-keep, cross-occurrence consistency, ~50% flip rate) all
passed; a real 3-epoch smoke run on Wiki-elec s42 trained cleanly (no NaN/crash, sane AUC
progression); a run_posthoc.py pass on that checkpoint reproduced the trainer's own
test_auc_epoch exactly (0.7163 both ways), confirming callbacks.py's posthoc-eval path
correctly reapplies the ablation -- fixed in from the start this time, not after the fact
(unlike the mask_node_tokens/mask_edge_tokens bug this ablation is following up on).

Local attention only (LocalAttn4, the production default) -- no full-attention side, same
scope as the MASKNODE/MASKEDGE campaign. No dataset cache is touched (pure input-rewrite
flag applied inside LitEdgeClassifier._step). Posthoc computes func_logit_power only.

Run under nohup:
  nohup .venv/bin/python scripts/run_signscramble_campaign.py \
      > logs/signscramble_campaign/driver.log 2>&1 &
  disown

Resumable: every job checks whether its posthoc output already exists before (re)running.
Not live-extendable (unlike run_ablation_campaign.py) -- this is a single fixed 60-job batch,
no need for the extra-jobs watcher machinery.
"""
import glob
import os
import subprocess
import threading
import time
from queue import Queue

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PY = os.path.join(REPO_ROOT, ".venv", "bin", "python")
LOG_DIR = os.path.join(REPO_ROOT, "logs", "signscramble_campaign")
os.makedirs(LOG_DIR, exist_ok=True)

GPUS = [0, 1, 2, 3]

# Easy-to-heavy by measured wall-clock time (same ordering/source as
# run_ablation_campaign.py's DATASETS_EASY_TO_HEAVY).
DATASETS_EASY_TO_HEAVY = [
    "wiki-elec",
    "bitcoin-alpha",
    "bitcoin-otc",
    "wiki-rfa",
    "epinions",
    "slashdot090221",
]
SEEDS = list(range(42, 52))

ABLATION_TAG = "SIGNSCRAMBLE"
ABLATION_FLAG = "model.scramble_edge_signs=true"

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


def job_train_and_posthoc(dataset, seed, gpu):
    exp_name = f"ABLATION_{ABLATION_TAG}_s{seed}"
    log_tag = f"{dataset}_{ABLATION_TAG}_s{seed}"
    run_id = "ablation_agg"

    existing = latest_exp_dir(dataset, exp_name)
    if existing is not None and posthoc_done(existing, run_id):
        driver_log(f"[SKIP] {dataset} seed={seed}: already complete")
        return

    cmd = [
        VENV_PY, "run.py",
        "--device", str(gpu),
        f"dataset.name={dataset}",
        f"training.exp_name={exp_name}",
        "model.local_attention_window=4",
        ABLATION_FLAG,
        f"reproducibility.seed={seed}",
    ]
    driver_log(f"[START] train {dataset} seed={seed} on GPU {gpu}")
    rc = run_logged(cmd, os.path.join(LOG_DIR, f"{log_tag}.train.log"), cwd=REPO_ROOT)
    if rc != 0:
        driver_log(f"[FAILED] train {dataset} seed={seed} (rc={rc}) -- see {log_tag}.train.log")
        return
    driver_log(f"[DONE] train {dataset} seed={seed}")

    exp_dir = latest_exp_dir(dataset, exp_name)
    if exp_dir is None:
        driver_log(f"[FAILED] {dataset} seed={seed}: no exp_dir found after training")
        return

    driver_log(f"[START] posthoc {dataset} seed={seed} on GPU {gpu}")
    rc = do_posthoc(exp_dir, dataset, run_id, gpu, log_tag)
    driver_log(f"[{'DONE' if rc == 0 else 'FAILED'}] posthoc {dataset} seed={seed} (rc={rc})")


def build_queue():
    jobs = []
    for ds in DATASETS_EASY_TO_HEAVY:
        for seed in SEEDS:
            jobs.append((ds, seed))
    return jobs


def worker(q, gpu):
    while True:
        try:
            job = q.get_nowait()
        except Exception:
            return
        try:
            job_train_and_posthoc(*job, gpu=gpu)
        except Exception as e:
            driver_log(f"[EXCEPTION] job={job} gpu={gpu}: {e!r}")
        finally:
            q.task_done()


def main():
    jobs = build_queue()
    total = len(jobs)
    driver_log(f"Queue built: {total} jobs total (expected 60), {len(GPUS)} workers (GPUs {GPUS})")

    q = Queue()
    for j in jobs:
        q.put(j)

    threads = [threading.Thread(target=worker, args=(q, gpu), daemon=True) for gpu in GPUS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    driver_log("=== SIGNSCRAMBLE campaign driver finished ===")


if __name__ == "__main__":
    main()
