"""Gap-closing diagnostic (2026-09-15, user-approved plan): 5 cheap single-seed=42 jobs
to test the specific levers evidence pointed at for each dataset that hasn't closed the
gap to production, BEFORE committing to another full 10-seed Phase 1 campaign:

  - wiki-elec: budget sweep never saturated (monotonic through 8x, still climbing) --
    test 12x/16x/20x on the existing winning architecture.
  - slashdot090221: its whole gap-closer sweep was measured at the wrong epoch count
    (50 instead of configs/slashdot090221.yaml's 75 -- the bug fixed this session in
    run_gap_closer.py/run_eid_ablations.py/run_eid_multiseed.py). Re-run the current
    winning config at the now-correct epoch count before touching anything else.
  - wiki-rfa: NOT actually a bad architecture/budget pick (v1's winner, 3x/531633
    walks, test_auc=0.8863, is only 0.5pp behind production) -- reconfirm it
    reproduces through the now-fixed per-seed cache pipeline (the wiki-rfa "gap" seen
    in Phase 1 was 2 bad-outlier seeds, not this config).
  - epinions: no diagnostic needed (already >= production even at the wrong epoch
    count) -- not included here, lowest priority per the user.

Shared-queue pattern (CLAUDE.md convention), even though 5 jobs is at the edge of
"handful of jobs where the bucket pattern is fine" -- queue is simpler to reason about
here since job costs are uneven (slashdot090221 at 75 epochs vs wiki-rfa's cheap 531k
walks).

Usage:
  nohup .venv/bin/python experiments/edge_identity_tokens/run_gap_diagnostic.py \
      > logs/eid_gap_closer/gap_diagnostic_driver.log 2>&1 &
  disown
"""
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
    wiki_elec = ra.winning_entry("wiki-elec")
    wiki_rfa = ra.winning_entry("wiki-rfa")
    slashdot = ra.winning_entry("slashdot090221")

    jobs = []
    # wiki-elec: extend budget past 8x (829512), same winning architecture params.
    for mult, nw in [(12.0, 1244268), (16.0, 1659024), (20.0, 2073780)]:
        jobs.append({
            "dataset": "wiki-elec", "num_walks": nw, "params": wiki_elec["params"],
            "exp": f"EID_GAPDIAG_WIKIELEC_{mult}X",
        })
    # slashdot090221: same winning config, but now correctly gets epochs=75 from
    # configs/slashdot090221.yaml (no more hardcoded epochs=50 override).
    jobs.append({
        "dataset": "slashdot090221", "num_walks": slashdot["num_walks"], "params": slashdot["params"],
        "exp": "EID_GAPDIAG_SLASHDOT090221_75EPOCH",
    })
    # wiki-rfa: reconfirm v1's winner reproduces through the fixed per-seed cache path.
    jobs.append({
        "dataset": "wiki-rfa", "num_walks": wiki_rfa["num_walks"], "params": wiki_rfa["params"],
        "exp": "EID_GAPDIAG_WIKIRFA_RECONFIRM",
    })

    # Heaviest (by num_walks) first, for reasonable load-balance across the queue.
    jobs.sort(key=lambda j: j["num_walks"], reverse=True)
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
    driver_log(f"[START] train {exp} (dataset={ds}, num_walks={nw}) on GPU {gpu}")
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
        "--exp-dir", str(exp_dir), "--device", str(gpu), "--run-id", "gapdiag",
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
    driver_log(f"Gap diagnostic queue built: {len(jobs)} jobs, {len(GPUS)} workers (GPUs {GPUS})")
    for j in jobs:
        driver_log(f"  queued: {j['exp']} (dataset={j['dataset']}, num_walks={j['num_walks']})")

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

    driver_log("=== Gap diagnostic driver finished ===")


if __name__ == "__main__":
    main()
