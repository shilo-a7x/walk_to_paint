"""Identity controls screen (2026-09-24), seed 42, all 6 datasets, 12 jobs.

IDOFF   -- `model.eid_identity_off=true` on production's exact per-dataset config
           (configs/<ds>.yaml untouched: architecture, lr, num_walks, epochs), with
           edge_embed_rank=0, eid_reveal_holdout_identity=false, edge_replace_prob=0.
           Edge positions carry sign only, like production's two sign tokens. Expected
           to reproduce production's own seed-42 number (E32_PY314_LOCALATTN4). If it
           doesn't, EID's code path differs from production beyond identity.
MASKTGT -- `model.mask_target_identity=true` on EID's current winning config: the
           target renders as <MASK> (like production), context edges keep identity.
           Tests whether the target's own identity (label-bearing for train targets,
           never for test targets) is what drives EID's overfitting. Compared to EID's
           own seed-42 noablation number.

Missing EID caches are built one at a time in the main thread before any GPU worker
starts (3 parallel jobs racing to build the same cache broke a run on 2026-09-16).

Usage:
  nohup .venv/bin/python experiments/edge_identity_tokens/run_identity_controls.py \
      > logs/eid_gap_closer/identity_controls_driver.log 2>&1 &
"""
import csv
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Queue

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import experiments.edge_identity_tokens.run_eid_ablations as ra  # noqa: E402
import experiments.edge_identity_tokens.eid_significance as sig  # noqa: E402
from experiments.edge_identity_tokens.run_eid import EID_CACHE_PATH, ensure_eid_cache  # noqa: E402
from src.utils.config import load_config  # noqa: E402

VENV_PY = ra.VENV_PY
LOG_DIR = REPO_ROOT / "logs" / "eid_gap_closer"
OUT_CSV = LOG_DIR / "identity_controls_results.csv"
GPUS = [0, 1, 2, 3]
SEED = 42
RUN_ID = "idctl"
DATASETS = ["slashdot090221", "epinions", "wiki-rfa", "wiki-elec", "bitcoin-otc", "bitcoin-alpha"]

_log_lock = threading.Lock()


def log(msg):
    with _log_lock:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def build_jobs():
    jobs = []
    for ds in DATASETS:
        prod_nw = int(load_config("config.yaml", overrides=[f"dataset.name={ds}"]).dataset.num_walks)
        jobs.append({
            "mode": "IDOFF", "dataset": ds, "num_walks": prod_nw,
            "exp": f"EID_IDCTL_IDOFF_{ds.upper().replace('-', '')}_s{SEED}",
            "overrides": [
                "model.edge_embed_rank=0", "model.eid_identity_off=true",
                "model.eid_reveal_holdout_identity=false", "model.edge_replace_prob=0.0",
            ],
        })
        w = ra.winning_entry(ds)
        overrides = ra.build_common_overrides(w["params"]) + ["training.batch_size=1024",
                                                               "model.mask_target_identity=true"]
        if ds in ra.EID_EPOCH_OVERRIDE:
            overrides.append(f"training.epochs={ra.EID_EPOCH_OVERRIDE[ds]}")
        jobs.append({
            "mode": "MASKTGT", "dataset": ds, "num_walks": int(w["num_walks"]),
            "exp": f"EID_IDCTL_MASKTGT_{ds.upper().replace('-', '')}_s{SEED}",
            "overrides": overrides,
        })
    return jobs


def prebuild_caches(jobs):
    for ds, nw in sorted({(j["dataset"], j["num_walks"]) for j in jobs}):
        path = EID_CACHE_PATH.format(dataset=ds, num_walks=nw, seed=SEED)
        if Path(path).is_file():
            continue
        log(f"[CACHE] building {path}")
        cfg = load_config("config.yaml", overrides=[f"dataset.name={ds}", f"dataset.num_walks={nw}",
                                                    f"reproducibility.seed={SEED}"])
        ensure_eid_cache(cfg, path)
        log(f"[CACHE] done {path}")


def run_logged(cmd, log_path):
    with open(log_path, "w") as f:
        f.write(f"$ {' '.join(cmd)}\n\n")
        f.flush()
        return subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT).returncode


def run_job(job, gpu):
    ds, exp = job["dataset"], job["exp"]
    cmd = [VENV_PY, "experiments/edge_identity_tokens/run_eid.py",
           f"dataset.name={ds}", f"dataset.num_walks={job['num_walks']}",
           f"training.exp_name={exp}", f"reproducibility.seed={SEED}",
           *job["overrides"], "--device", str(gpu)]
    log(f"[START] train {exp} on GPU {gpu}")
    if run_logged(cmd, LOG_DIR / f"{exp}.train.log") != 0:
        log(f"[FAILED] train {exp}")
        return
    exp_dirs = sorted((REPO_ROOT / "outputs" / ds).glob(f"{exp}_*"))
    if not exp_dirs:
        log(f"[FAILED] {exp}: no exp_dir after training")
        return
    log(f"[START] posthoc {exp} on GPU {gpu}")
    rc = run_logged([VENV_PY, "experiments/edge_identity_tokens/eid_posthoc.py",
                     "--exp-dir", str(exp_dirs[-1]), "--device", str(gpu),
                     "--run-id", RUN_ID, "--agg-models", "func_logit_power"],
                    LOG_DIR / f"{exp}.posthoc.log")
    log(f"[{'DONE' if rc == 0 else 'FAILED'}] posthoc {exp}")


def worker(q, gpu):
    while True:
        job = q.get()
        if job is None:
            q.task_done()
            return
        try:
            run_job(job, gpu)
        except Exception as e:
            log(f"[EXCEPTION] {job['exp']} gpu={gpu}: {e!r}")
        finally:
            q.task_done()


def report(jobs):
    rows = []
    for j in jobs:
        dirs = sorted((REPO_ROOT / "outputs" / j["dataset"]).glob(f"{j['exp']}_*"))
        auc = sig._read_summary_auc(str(dirs[-1]), RUN_ID) if dirs else None
        if j["mode"] == "IDOFF":
            ref_name, ref = "production_s42", sig.production_seed_auc(j["dataset"], SEED)
        else:
            ref_name, ref = "eid_noablation_s42", sig.eid_seed_auc(j["dataset"], "noablation", SEED)
        delta = (auc - ref) if (auc is not None and ref is not None) else None
        rows.append({"mode": j["mode"], "dataset": j["dataset"], "test_auc": auc,
                     "reference": ref_name, "reference_auc": ref, "delta": delta})
        log(f"[RESULT] {j['mode']:8s} {j['dataset']:16s} test={auc} vs {ref_name}={ref} delta={delta}")
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log(f"wrote {OUT_CSV}")


def main():
    jobs = build_jobs()
    log(f"identity controls: {len(jobs)} jobs, GPUs {GPUS}")
    prebuild_caches(jobs)
    q = Queue()
    for j in jobs:
        q.put(j)
    threads = [threading.Thread(target=worker, args=(q, g), daemon=True) for g in GPUS]
    for t in threads:
        t.start()
    q.join()
    for _ in GPUS:
        q.put(None)
    for t in threads:
        t.join()
    report(jobs)
    log("=== identity controls finished ===")


if __name__ == "__main__":
    main()
