"""Job-file helpers for run_reg_campaign.py.

eid_base(ds, lr_mult=1.0) -> (num_walks, overrides) for a dataset's current EID gap-closer
winner (same construction run_eid_multiseed.py uses), optionally with a scaled LR.
enqueue(stage, jobs) writes logs/eid_reg/queue/<stage>.json for the live driver to pick up.
"""
import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import experiments.edge_identity_tokens.run_eid_ablations as ra  # noqa: E402

QUEUE_DIR = REPO_ROOT / "logs" / "eid_reg" / "queue"


def eid_base(ds, lr_mult=1.0):
    w = ra.winning_entry(ds)
    params = copy.deepcopy(w["params"])
    params["training.lr"] = float(params["training.lr"]) * lr_mult
    overrides = ra.build_common_overrides(params) + ["training.batch_size=1024"]
    if ds in ra.EID_EPOCH_OVERRIDE:
        overrides.append(f"training.epochs={ra.EID_EPOCH_OVERRIDE[ds]}")
    return int(w["num_walks"]), overrides


def job(name, ds, seed, num_walks, overrides):
    return {"name": name, "dataset": ds, "seed": seed, "num_walks": num_walks, "overrides": list(overrides)}


def enqueue(stage, jobs):
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = QUEUE_DIR / f".{stage}.json.tmp"
    tmp.write_text(json.dumps(jobs, indent=1))
    tmp.rename(QUEUE_DIR / f"{stage}.json")
    print(f"enqueued {len(jobs)} jobs -> {stage}.json")
