"""Append a batch of extra ablation jobs to a *running* run_ablation_campaign.py
driver, with no restart needed. The driver polls logs/ablation_campaign/extra_jobs/
every 30s for new *.json batch files and pushes their contents onto its live job
queue -- so a batch dropped here fires on the next free GPU within 30s.

Usage (example -- ablation A's exact flag/seed range TBD, fill in once agreed):
  .venv/bin/python scripts/enqueue_extra_ablation_jobs.py \
      --tag ABLATIONA \
      --flag "model.randomize_edge_direction=true model.edge_direction_randomize_prob=1.0" \
      --datasets wiki-elec,bitcoin-alpha,bitcoin-otc,wiki-rfa,epinions,slashdot090221 \
      --seeds 42-51

If the driver process is not currently running, the batch file is still written and
will be picked up automatically the next time the driver is (re)started.
"""
import argparse
import json
import os
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRA_JOBS_DIR = os.path.join(REPO_ROOT, "logs", "ablation_campaign", "extra_jobs")


def parse_seeds(spec):
    seeds = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="short exp_name tag, e.g. ABLATIONA")
    ap.add_argument(
        "--flag", required=True,
        help="model.* override(s) for run.py, space-separated if more than one",
    )
    ap.add_argument("--datasets", required=True, help="comma-separated, easy-to-heavy order")
    ap.add_argument("--seeds", required=True, help="e.g. 42-51 or 42,43,44")
    args = ap.parse_args()

    os.makedirs(EXTRA_JOBS_DIR, exist_ok=True)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = parse_seeds(args.seeds)

    batch = [
        {"dataset": ds, "tag": args.tag, "flag": args.flag, "seed": seed}
        for ds in datasets
        for seed in seeds
    ]

    ts = int(time.time())
    tmp_path = os.path.join(EXTRA_JOBS_DIR, f".tmp_{args.tag}_{ts}.json")
    final_path = os.path.join(EXTRA_JOBS_DIR, f"{args.tag}_{ts}.json")
    with open(tmp_path, "w") as f:
        json.dump(batch, f, indent=2)
    os.rename(tmp_path, final_path)  # atomic on the same filesystem

    print(f"Wrote {len(batch)} jobs ({len(datasets)} datasets x {len(seeds)} seeds) to {final_path}")
    print("A running driver picks this up within its next 30s poll. If no driver is "
          "running, it will be picked up the next time one is started.")


if __name__ == "__main__":
    main()
