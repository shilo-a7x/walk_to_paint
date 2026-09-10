"""No-idle-GPU queue driver for EID training+posthoc jobs.

Mirrors this project's established 4-GPU queue-worker pattern (scripts/run_
multiseed_pewter.py, scripts/run_ablation_campaign.py): one worker thread per GPU,
each pulls the next job off a shared queue, runs run_eid.py (train) then
eid_posthoc.py (edge-level aggregation) synchronously, logs the result, then loops
-- zero idle GPU time as long as the queue has jobs left.

Not a generic framework -- job list is defined inline in JOBS below, edited
directly per campaign (same convention as run_ablation_campaign.py's own job-list
style). Common architecture overrides (matching the bitcoin-alpha reveal-identity
sweep) are factored into COMMON_MODEL_OVERRIDES; per-job overrides (dataset,
num_walks, exp_name, any ablation flag) are specified per entry.

Usage:
  .venv/bin/python experiments/edge_identity_tokens/run_eid_queue.py
"""
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VENV_PY = str(REPO_ROOT / ".venv" / "bin" / "python")
LOG_DIR = REPO_ROOT / "logs" / "eid_queue"
LOG_DIR.mkdir(parents=True, exist_ok=True)

COMMON_MODEL_OVERRIDES = [
    "training.epochs=50", "training.batch_size=1024",
    "model.embedding_dim=64", "model.hidden_dim=128", "model.dropout=0.13",
    "model.nhead=4", "model.nlayers=2",
    "model.node_replace_prob=0.37", "model.node_replace_unk_ratio=0.75",
    "model.edge_embed_rank=20", "model.edge_sign_combine=concat",
    "model.edge_residual_baseline=true",
    "model.head_dim=24", "model.sign_embed_dim=20", "model.node_embed_dim=64",
    "model.edge_replace_prob=0.15", "model.edge_replace_unk_ratio=0.23",
    "model.edge_embed_weight_decay=0.001",
]

# --- Job list -----------------------------------------------------------------
# Part A: single-dataset ablation triad on the flagship bitcoin-alpha/300000 point
# (all combined with eid_reveal_holdout_identity=true, matching the flagship run).
ABLATION_JOBS = [
    {
        "dataset": "bitcoin-alpha", "num_walks": 300000,
        "exp_name": "EID_ABL_MASKEDGE_300000",
        "extra": ["model.eid_reveal_holdout_identity=true", "model.mask_edge_tokens=true"],
    },
    {
        "dataset": "bitcoin-alpha", "num_walks": 300000,
        "exp_name": "EID_ABL_SCRAMBLESIGN_300000",
        "extra": ["model.eid_reveal_holdout_identity=true", "model.scramble_edge_signs=true"],
    },
    {
        "dataset": "bitcoin-alpha", "num_walks": 300000,
        "exp_name": "EID_ABL_SCRAMBLEID_300000",
        "extra": ["model.eid_reveal_holdout_identity=true", "model.scramble_edge_identity=true"],
    },
]

# Part B: single-split reveal-identity cross-dataset budget check, two points per
# dataset (production's own walk-budget pick, and 2x/0.5x it -- kept modest for
# the two much larger datasets, epinions/slashdot, per this project's cost lean).
CROSS_DATASET_JOBS = [
    {"dataset": "bitcoin-otc", "num_walks": 177960, "exp_name": "EID_REVEAL_XDATASET_177960"},
    {"dataset": "bitcoin-otc", "num_walks": 355920, "exp_name": "EID_REVEAL_XDATASET_355920"},
    {"dataset": "wiki-elec", "num_walks": 155534, "exp_name": "EID_REVEAL_XDATASET_155534"},
    {"dataset": "wiki-elec", "num_walks": 311068, "exp_name": "EID_REVEAL_XDATASET_311068"},
    {"dataset": "wiki-rfa", "num_walks": 265817, "exp_name": "EID_REVEAL_XDATASET_265817"},
    {"dataset": "wiki-rfa", "num_walks": 531634, "exp_name": "EID_REVEAL_XDATASET_531634"},
    {"dataset": "epinions", "num_walks": 420400, "exp_name": "EID_REVEAL_XDATASET_420400"},
    {"dataset": "epinions", "num_walks": 840799, "exp_name": "EID_REVEAL_XDATASET_840799"},
    {"dataset": "slashdot090221", "num_walks": 823803, "exp_name": "EID_REVEAL_XDATASET_823803"},
    {"dataset": "slashdot090221", "num_walks": 1647606, "exp_name": "EID_REVEAL_XDATASET_1647606"},
]
for j in CROSS_DATASET_JOBS:
    j["extra"] = ["model.eid_reveal_holdout_identity=true"]

# Order: cheap/fast datasets first (bitcoin-otc, wiki-elec, wiki-rfa), then the
# 3 ablation jobs (all cheap, bitcoin-alpha/300000), then the two expensive
# datasets (epinions, slashdot) last -- so the queue keeps all 4 GPUs busy on
# fast jobs early rather than 1 GPU stuck on a slow epinions/slashdot job while
# 3 others sit empty near the end.
JOBS = (
    [j for j in CROSS_DATASET_JOBS if j["dataset"] in ("bitcoin-otc", "wiki-elec", "wiki-rfa")]
    + ABLATION_JOBS
    + [j for j in CROSS_DATASET_JOBS if j["dataset"] in ("epinions", "slashdot090221")]
)

NUM_GPUS = 4
result_lock = threading.Lock()
results = []


def run_job(job, device):
    dataset = job["dataset"]
    num_walks = job["num_walks"]
    exp_name = job["exp_name"]
    extra = job.get("extra", [])
    log_path = LOG_DIR / f"{exp_name}.train.log"

    train_cmd = [
        VENV_PY, "experiments/edge_identity_tokens/run_eid.py",
        f"dataset.name={dataset}", f"dataset.num_walks={num_walks}",
        f"training.exp_name={exp_name}",
        *COMMON_MODEL_OVERRIDES, *extra,
        "--device", str(device),
    ]
    print(f"[GPU{device}] START train {exp_name} ({dataset}, nw={num_walks})", flush=True)
    t0 = time.time()
    with open(log_path, "w") as f:
        ret = subprocess.run(train_cmd, cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT)
    train_secs = time.time() - t0
    if ret.returncode != 0:
        print(f"[GPU{device}] FAILED train {exp_name} (exit {ret.returncode}, {train_secs:.0f}s) -- see {log_path}", flush=True)
        with result_lock:
            results.append({"exp_name": exp_name, "status": "train_failed", "train_secs": train_secs})
        return

    # find the exp_dir this run just created
    out_glob = sorted((REPO_ROOT / "outputs" / dataset).glob(f"{exp_name}_*"))
    if not out_glob:
        print(f"[GPU{device}] WARN: no output dir found for {exp_name}", flush=True)
        with result_lock:
            results.append({"exp_name": exp_name, "status": "no_output_dir", "train_secs": train_secs})
        return
    exp_dir = out_glob[-1]

    posthoc_log_path = LOG_DIR / f"{exp_name}.posthoc.log"
    posthoc_cmd = [
        VENV_PY, "experiments/edge_identity_tokens/eid_posthoc.py",
        "--exp-dir", str(exp_dir), "--device", str(device), "--run-id", "posthoc",
    ]
    print(f"[GPU{device}] START posthoc {exp_name} ({train_secs:.0f}s train)", flush=True)
    with open(posthoc_log_path, "w") as f:
        ret2 = subprocess.run(posthoc_cmd, cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT)

    test_auc = None
    if ret2.returncode == 0:
        text = posthoc_log_path.read_text()
        for line in text.splitlines():
            if "Edge agg_tr AUC" in line and "Edge test AUC" in line:
                try:
                    test_auc = float(line.split("Edge test AUC=")[1].strip())
                except Exception:
                    pass
    status = "done" if ret2.returncode == 0 else "posthoc_failed"
    print(f"[GPU{device}] {status.upper()} {exp_name} -- edge test AUC={test_auc}", flush=True)
    with result_lock:
        results.append({
            "exp_name": exp_name, "status": status, "train_secs": train_secs,
            "dataset": dataset, "num_walks": num_walks, "edge_test_auc": test_auc,
        })


def worker(device, job_queue):
    while True:
        try:
            job = job_queue.get_nowait()
        except queue.Empty:
            return
        try:
            run_job(job, device)
        finally:
            job_queue.task_done()


def main():
    job_queue = queue.Queue()
    for j in JOBS:
        job_queue.put(j)

    threads = [
        threading.Thread(target=worker, args=(gpu, job_queue), daemon=False)
        for gpu in range(NUM_GPUS)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("\n=== ALL JOBS DONE ===")
    for r in sorted(results, key=lambda r: r["exp_name"]):
        print(r)


if __name__ == "__main__":
    main()
