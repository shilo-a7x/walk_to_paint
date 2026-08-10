"""Multi-seed retrain driver for the WSDM revision's mean+-std AUC requirement
(replaces the single-split Hanley-McNeil SE in Table 1 / Ablation A / Ablation B).

Confirmed plan (2026-08-06, user sign-off):
- 10 seeds per (dataset, attention variant): 42 (already trained, reused/backfilled)
  + 43..51 (9 new trainings each).
- Dataset order, smallest to largest by production num_walks budget:
  bitcoin-alpha, wiki-elec, bitcoin-otc, wiki-rfa, epinions, slashdot090221.
- Per dataset: all 9 new full-attention seeds are queued before any of that dataset's
  local-attention seeds (strict per-dataset ordering, not global).
- Every posthoc run computes all 11 registered aggregator weight functions (not just
  the production func_logit_power default), so Ablation B numbers are available for
  every seed without a second pass later. The full-attention side only had
  func_logit_power computed for seed 42 (E31_PY314_MIGRATION) -- backfilled here too
  (run-id ablationB_e31, mirroring the existing ablationB_e32 on the local side).
- GINEConv (the one baseline we run ourselves) gets the same 10-seed treatment, at the
  tail of the queue, lowest priority -- it already has its own skip-if-exists logic for
  seed 42.
- 3 GPUs used (1,2,3) -- GPU 0 left alone, occupied by another user's job.

Run under nohup, e.g.:
  nohup .venv/bin/python scripts/run_multiseed_pewter.py \
      > logs/multiseed/driver.log 2>&1 &
  disown

Resumable: every job checks whether its output already exists before running, so a
killed/restarted driver just picks up where it left off.
"""
import glob
import os
import subprocess
import sys
import threading
import time
from queue import Queue

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PY = os.path.join(REPO_ROOT, ".venv", "bin", "python")
LOG_DIR = os.path.join(REPO_ROOT, "logs", "multiseed")
os.makedirs(LOG_DIR, exist_ok=True)

GPUS = [1, 2, 3]
DATASETS_ORDER = [
    "bitcoin-alpha",   # 120,930 walks
    "wiki-elec",       # 155,534
    "bitcoin-otc",     # 177,960
    "wiki-rfa",        # 265,817
    "epinions",        # 840,799
    "slashdot090221",  # 1,647,606
]
NEW_SEEDS = list(range(43, 52))  # 43..51 inclusive, 9 new seeds; 42 already exists

AGG_FUNCS = [
    "func_uniform", "func_conf_power", "func_conf_exp", "func_conf_cert",
    "func_conf_logit", "func_logq_power", "func_logit_power",
    "func_entropy_power", "func_entropy_exp", "func_fisher_power",
    "func_maxprob_power",
]
AGG_FUNCS_CSV = ",".join(AGG_FUNCS)

GINECONV_PY = "/home/dsi/shilo_avital/.conda/envs/sesgformer_env/bin/python"
GINECONV_DIR = os.path.join(REPO_ROOT, "baselines", "GINEConv")
GINECONV_RESULTS_ROOT = "results_our_splits_canonical"  # matches the paper's existing GINEConv numbers
GINECONV_SEEDS = [42] + NEW_SEEDS  # 42 included: split-generation/skip logic below handles it specially

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


def job_backfill_seed42_full(dataset, gpu):
    exp_dir = latest_exp_dir(dataset, "E31_PY314_MIGRATION")
    if exp_dir is None:
        driver_log(f"[SKIP] backfill {dataset} seed42 full: no E31_PY314_MIGRATION dir found")
        return
    run_id = "ablationB_e31"
    if posthoc_done(exp_dir, run_id):
        driver_log(f"[SKIP] backfill {dataset} seed42 full: already done ({run_id})")
        return
    driver_log(f"[START] backfill {dataset} seed42 full (11-func posthoc) on GPU {gpu}")
    rc = do_posthoc(exp_dir, dataset, run_id, gpu, f"{dataset}_seed42_full_backfill")
    driver_log(f"[{'DONE' if rc == 0 else 'FAILED'}] backfill {dataset} seed42 full (rc={rc})")


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


def gineconv_splits_dir(dataset, seed):
    """Ensure a canonical (walk-derived) split exists for this seed and return its dir.

    seed=42 reuses the existing baselines/splits_canonical/ (already built, what the
    paper's current GINEConv numbers come from). Other seeds get their own isolated
    baselines/splits_canonical_seed{seed}/ dir, built on demand from that seed's own
    walk cache -- see baselines/prepare_splits.py::save_canonical_split_for_seed.
    Requires the corresponding PEWTER walk cache for (dataset, seed) to already
    exist, which is guaranteed here since GINEConv is queued after all PEWTER jobs.
    """
    if seed == 42:
        return os.path.join(REPO_ROOT, "baselines", "splits_canonical")
    import prepare_splits  # baselines/prepare_splits.py, path added at module load
    out_path = prepare_splits.save_canonical_split_for_seed(dataset, seed)
    return os.path.dirname(out_path)


def job_gineconv(dataset, seed, gpu):
    out_tag = f"seed{seed}"
    score_csv = os.path.join(GINECONV_DIR, GINECONV_RESULTS_ROOT, dataset, "GINEConv", out_tag, "score.csv")
    if os.path.isfile(score_csv):
        driver_log(f"[SKIP] GINEConv {dataset} seed={seed}: score.csv already exists")
        return
    try:
        splits_dir = gineconv_splits_dir(dataset, seed)
    except Exception as e:
        driver_log(f"[FAILED] GINEConv {dataset} seed={seed}: could not build canonical split: {e!r}")
        return
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["SPLITS_DIR"] = splits_dir
    env["RESULTS_ROOT"] = GINECONV_RESULTS_ROOT
    cmd = [
        GINECONV_PY, "run_with_our_splits.py",
        "--dataset", dataset, "--seed", str(seed), "--device", "cuda",
        "--out-tag", out_tag,  # the script only varies out-tag with --aggr/--num-layers by default, not --seed
    ]
    log_path = os.path.join(LOG_DIR, f"gineconv_{dataset}_s{seed}.log")
    driver_log(f"[START] GINEConv {dataset} seed={seed} on GPU {gpu} (SPLITS_DIR={splits_dir})")
    rc = run_logged(cmd, log_path, cwd=GINECONV_DIR, env=env)
    driver_log(f"[{'DONE' if rc == 0 else 'FAILED'}] GINEConv {dataset} seed={seed} (rc={rc})")


def build_queue():
    jobs = []
    # 1) cheap seed-42 full-attention backfill, all datasets, size order
    for ds in DATASETS_ORDER:
        jobs.append(("backfill42", ds))
    # 2) main sweep: per dataset, full (9 seeds) then local (9 seeds)
    for ds in DATASETS_ORDER:
        for variant in ("full", "local"):
            for seed in NEW_SEEDS:
                jobs.append(("train", ds, variant, seed))
    # 3) GINEConv, lowest priority, tail of queue
    for ds in DATASETS_ORDER:
        for seed in GINECONV_SEEDS:
            jobs.append(("gineconv", ds, seed))
    return jobs


def worker(q, gpu):
    while True:
        job = q.get()
        if job is None:
            q.task_done()
            return
        try:
            if job[0] == "backfill42":
                _, ds = job
                job_backfill_seed42_full(ds, gpu)
            elif job[0] == "train":
                _, ds, variant, seed = job
                job_train_and_posthoc(ds, variant, seed, gpu)
            elif job[0] == "gineconv":
                _, ds, seed = job
                job_gineconv(ds, seed, gpu)
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
