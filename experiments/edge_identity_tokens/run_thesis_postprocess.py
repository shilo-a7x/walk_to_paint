"""Post-campaign step for the H3 thesis campaign: waits until every (condition, dataset, seed) of
make_thesis_jobs.py has a row in logs/eid_reg/results.csv (ok or failed), then
  1. Ablation B: re-aggregates every H3 run with all 11 aggregators from its saved predictions
     (eid_posthoc.py --splits "" --run-id thesis_aggB), one worker per GPU;
  2. runs eid_thesis_results.py (all tables, F1, K-walks, paired tests);
  3. writes logs/eid_reg/THESIS_POSTPROCESS_DONE.

Usage:
  nohup .venv/bin/python experiments/edge_identity_tokens/run_thesis_postprocess.py \
      > logs/eid_reg/postprocess.log 2>&1 &
"""
import csv
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Queue

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.edge_identity_tokens.eid_thesis_results import AGG_FUNCS, exp_dir  # noqa: E402
from experiments.edge_identity_tokens.make_thesis_jobs import CONDITIONS, SEEDS, WALKS  # noqa: E402

PY = str(ROOT / ".venv" / "bin" / "python")
RESULTS = ROOT / "logs" / "eid_reg" / "results.csv"
LOGS = ROOT / "logs" / "eid_reg" / "jobs"
DONE = ROOT / "logs" / "eid_reg" / "THESIS_POSTPROCESS_DONE"
NAMES = {n for n, _ in CONDITIONS}
EXPECTED = len(NAMES) * len(WALKS) * len(SEEDS)


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def finished():
    with open(RESULTS) as f:
        return {(r["name"], r["dataset"], r["seed"]) for r in csv.DictReader(f) if r["name"] in NAMES}


def worker(q, gpu):
    while True:
        d = q.get()
        if d is None:
            return
        rc = subprocess.run([PY, "experiments/edge_identity_tokens/eid_posthoc.py", "--exp-dir", str(d),
                             "--device", str(gpu), "--splits", "", "--run-id", "thesis_aggB",
                             "--agg-models", ",".join(AGG_FUNCS)], cwd=ROOT,
                            stdout=open(LOGS / f"{d.name}.aggB.log", "w"), stderr=subprocess.STDOUT).returncode
        log(f"aggB {d.name} rc={rc}")


def main():
    while len(finished()) < EXPECTED:
        log(f"waiting: {len(finished())}/{EXPECTED} runs recorded")
        time.sleep(900)
    log("all runs recorded -- Ablation B re-aggregation")
    q = Queue()
    for ds in WALKS:
        for s in SEEDS:
            d = exp_dir("H3", ds, s)
            if d is not None:
                q.put(d)
    threads = [threading.Thread(target=worker, args=(q, g)) for g in range(4)]
    for _ in threads:
        q.put(None)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log("collecting results")
    subprocess.run([PY, "experiments/edge_identity_tokens/eid_thesis_results.py"], cwd=ROOT)
    DONE.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    log("done")


if __name__ == "__main__":
    main()
