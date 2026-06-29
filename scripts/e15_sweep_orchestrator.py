"""E15 budget x attention sweep orchestrator (k_cover k=5).

Grid: per-dataset budgets x {full, local} attention. For each cell:
  1. ensure keyed cache built (once per (ds,nw), global lock -> no race; full+local share it)
  2. train (run.py, deterministic exp dir via paths.append_timestamp=false)
  3. posthoc (func_logit_power) -> parse raw + flp test AUC + covered-edge count
  4. append to results CSV (skip-done on restart)
One job per GPU at a time; GPUS configurable. Run under nohup.
"""
import os, sys, csv, time, threading, subprocess, glob, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv/bin/python")
GPUS = [1, 2, 3]
LOGDIR = os.path.join(ROOT, "logs/e15_coverage/sweep")
CSV = os.path.join(ROOT, "outputs/walk_coverage_analysis/sweep_results.csv")
os.makedirs(LOGDIR, exist_ok=True)

GRID = {
    "bitcoin-alpha":  [1000000, 2000000, 5000000],
    "bitcoin-otc":    [500000, 1000000, 2000000],
    "wiki-elec":      [500000, 1000000, 2000000],
    "wiki-rfa":       [1000000, 2000000],
    "epinions":       [1000000, 2000000, 3000000],
    "slashdot090221": [2000000, 3000000, 5000000],
}
ATTN = ["full", "local"]
TI = "outputs/transformer_incremental"
HM = {
 "bitcoin-alpha": f"{TI}/bitcoin-alpha_seed42_nw5000000_mw80_bs1024_ep75_20260423-111407/artifacts/E14_HARDNODE_L10/hardness_map.pt",
 "bitcoin-otc":   f"{TI}/bitcoin-otc_seed42_nw500000_mw80_bs1024_ep50_20260419-133621/artifacts/E14_HARDNODE_L10/hardness_map.pt",
 "epinions":      f"{TI}/epinions_seed42_nw500000_mw80_bs1024_ep50_20260416-125419/artifacts/E14_HARDNODE_L10/hardness_map.pt",
 "wiki-elec":     f"{TI}/wiki-elec_seed42_nw500000_mw80_bs1024_ep50_20260419-133621/artifacts/E14_HARDNODE_L10/hardness_map.pt",
 "wiki-rfa":      f"{TI}/wiki-rfa_seed42_nw500000_mw80_bs1024_ep50_20260324-141920/artifacts/E14_HARDNODE_L10/hardness_map.pt",
 "slashdot090221":f"{TI}/slashdot090221_seed42_nw5000000_mw80_bs1024_ep50_20260420-134149/artifacts/E14_HARDNODE_L10/hardness_map.pt",
}

DATA_DIRS = {"bitcoin-alpha": "data/bitcoin-alpha", "bitcoin-otc": "data/bitcoin-otc",
             "epinions": "data/epinions", "wiki-elec": "data/wiki-Elec",
             "wiki-rfa": "data/wiki-RfA", "slashdot090221": "data/slashdot090221"}

_build_lock = threading.Lock()
_csv_lock = threading.Lock()
_built = set()


def keyed_cache_path(ds, nw):
    return os.path.join(ROOT, DATA_DIRS[ds],
                        f"dataset_cache__k_cover_k5_nw{nw}_mw80_seed42.pt")


def done_cells():
    d = set()
    if os.path.exists(CSV):
        with open(CSV) as f:
            for row in csv.DictReader(f):
                d.add((row["dataset"], int(row["num_walks"]), row["attn"]))
    return d


def record(row):
    with _csv_lock:
        new = not os.path.exists(CSV)
        with open(CSV, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["dataset", "num_walks", "attn", "raw_test_auc",
                                              "flp_test_auc", "covered_edges", "exp_dir", "status"])
            if new:
                w.writeheader()
            w.writerow(row)


def walk_overrides(ds, nw):
    return [f"dataset.name={ds}", "dataset.walk_strategy=k_cover",
            "dataset.walk_k_min=5", f"dataset.num_walks={nw}"]


def ensure_cache(ds, nw):
    key = (ds, nw)
    with _build_lock:
        if key in _built or os.path.exists(keyed_cache_path(ds, nw)):
            _built.add(key)
            return
        cmd = [PY, "scripts/build_cache.py"] + walk_overrides(ds, nw)
        lg = os.path.join(LOGDIR, f"buildcache_{ds}_nw{nw}.log")
        with open(lg, "w") as f:
            r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            raise RuntimeError(f"cache build failed for {ds} nw{nw} (see {lg})")
        _built.add(key)


def parse_posthoc(logpath):
    raw = flp = cov = None
    if os.path.exists(logpath):
        txt = open(logpath, errors="ignore").read()
        m = re.search(r"Saved test predictions:.*?(\d+) unique edges, AUC=([0-9.]+)", txt)
        if m:
            cov = int(m.group(1)); raw = float(m.group(2))
        m = re.search(r"Edge test AUC=([0-9.]+)", txt)
        if m:
            flp = float(m.group(1))
    return raw, flp, cov


def run_cell(gpu, ds, nw, attn):
    exp = f"E15_SWEEP_k5_nw{nw}_{attn}"
    exp_dir = f"outputs/{ds}/{exp}"
    tlog = os.path.join(LOGDIR, f"{ds}_nw{nw}_{attn}_train.log")
    plog = os.path.join(LOGDIR, f"{ds}_nw{nw}_{attn}_posthoc.log")
    common = walk_overrides(ds, nw) + [
        f"model.hardness_map_path={HM[ds]}", "model.hardness_lambda=1.0",
        f"training.exp_name={exp}", "paths.append_timestamp=false",
    ]
    if attn == "local":
        common += ["model.local_attention_window=4"]
    try:
        ensure_cache(ds, nw)
    except Exception as e:
        record(dict(dataset=ds, num_walks=nw, attn=attn, raw_test_auc="", flp_test_auc="",
                    covered_edges="", exp_dir=exp_dir, status=f"cache_fail:{e}"))
        return
    # train
    with open(tlog, "w") as f:
        r = subprocess.run([PY, "run.py", "--device", str(gpu)] + common,
                           cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        record(dict(dataset=ds, num_walks=nw, attn=attn, raw_test_auc="", flp_test_auc="",
                    covered_edges="", exp_dir=exp_dir, status="train_fail"))
        return
    # posthoc
    with open(plog, "w") as f:
        r = subprocess.run([PY, "run_posthoc.py", "--exp-dir", f"{exp_dir}/",
                            "--artifacts", "predictions,aggregator", "--agg-models", "func_logit_power",
                            "--device", str(gpu), "--run-id", exp] + walk_overrides(ds, nw),
                           cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    raw, flp, cov = parse_posthoc(plog)
    record(dict(dataset=ds, num_walks=nw, attn=attn, raw_test_auc=raw, flp_test_auc=flp,
                covered_edges=cov, exp_dir=exp_dir, status="ok" if flp else "posthoc_parse_fail"))


def main():
    done = done_cells()
    jobs = [(ds, nw, attn) for ds in GRID for nw in GRID[ds] for attn in ATTN
            if (ds, nw, attn) not in done]
    # local cells last so full-attention builds the shared cache first
    jobs.sort(key=lambda c: (c[2] == "local", c[0], c[1]))
    print(f"{len(done)} cells already done; {len(jobs)} to run on GPUs {GPUS}", flush=True)
    # shared job queue (list + lock via pop is not atomic across threads -> use Lock)
    lock = threading.Lock()
    shared = list(jobs)

    def safe_worker(gpu):
        while True:
            with lock:
                if not shared:
                    return
                cell = shared.pop(0)
            print(f"[GPU{gpu}] start {cell}", flush=True)
            t0 = time.time()
            run_cell(gpu, *cell)
            print(f"[GPU{gpu}] done {cell} in {time.time()-t0:.0f}s", flush=True)

    threads = [threading.Thread(target=safe_worker, args=(g,)) for g in GPUS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("SWEEP COMPLETE", flush=True)


if __name__ == "__main__":
    main()
