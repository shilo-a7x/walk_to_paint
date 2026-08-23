"""Compare the EID pilot's train/val curves against the production checkpoint's own
curves, on the same axes. Two data sources:

  - Production (bitcoin-alpha, local attention, seed 42, E32_PY314_LOCALATTN4):
    train_loss / val_auc_epoch pulled from its TensorBoard event file via
    tensorboard's EventAccumulator (PyTorch Lightning's CSVLogger wasn't used for
    this run, only TensorBoardLogger).
  - EID pilot: parsed directly from its own stdout log (pilot.log), which prints one
    line per epoch (see train_pilot.py).

Usage:
  .venv/bin/python experiments/edge_identity_tokens/plot_curves.py \
      --pilot-log experiments/edge_identity_tokens/pilot.log \
      --out experiments/edge_identity_tokens/curves.png \
      [--pilot-label "EID (no reg)"]
"""
import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROD_TB_DIR = (
    "outputs/bitcoin-alpha/E32_PY314_LOCALATTN4_20260804-225948/"
    "logs/bitcoin-alpha-E32_PY314_LOCALATTN4/version_0"
)

LINE_RE = re.compile(
    r"epoch\s+(\d+)\s+train_loss=([\d.]+)\s+val_walk_auc=([\d.]+)\s+val_edge_auc=([\d.]+)"
)


def parse_pilot_log(path):
    epochs, train_loss, val_walk_auc, val_edge_auc = [], [], [], []
    with open(path) as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            epochs.append(int(m.group(1)))
            train_loss.append(float(m.group(2)))
            val_walk_auc.append(float(m.group(3)))
            val_edge_auc.append(float(m.group(4)))
    return epochs, train_loss, val_walk_auc, val_edge_auc


def load_production_curves():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(PROD_TB_DIR)
    ea.Reload()

    def series(tag):
        events = ea.Scalars(tag)
        return [e.step for e in events], [e.value for e in events]

    steps_tl, train_loss = series("train_loss")
    steps_va, val_auc = series("val_auc_epoch")
    return (steps_tl, train_loss), (steps_va, val_auc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-log", default="experiments/edge_identity_tokens/pilot.log")
    ap.add_argument("--out", default="experiments/edge_identity_tokens/curves.png")
    ap.add_argument("--pilot-label", default="EID pilot")
    ap.add_argument("--pilot-log2", default=None,
                     help="Optional second pilot log to overlay (e.g. with edge-replace regularization).")
    ap.add_argument("--pilot-label2", default="EID pilot (regularized)")
    args = ap.parse_args()

    p_epochs, p_loss, p_walk_auc, p_edge_auc = parse_pilot_log(args.pilot_log)
    (prod_loss_steps, prod_loss), (prod_auc_steps, prod_auc) = load_production_curves()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(prod_loss_steps, prod_loss, label="production (2-shared-sign-token)", color="tab:blue")
    ax.plot(p_epochs, p_loss, label=args.pilot_label, color="tab:orange")
    if args.pilot_log2:
        e2, l2, w2, ed2 = parse_pilot_log(args.pilot_log2)
        ax.plot(e2, l2, label=args.pilot_label2, color="tab:green")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss (log scale)")
    ax.set_title("Train loss")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(prod_auc_steps, prod_auc, label="production val AUC (edge-level agg)", color="tab:blue")
    ax.plot(p_epochs, p_walk_auc, label=f"{args.pilot_label} val walk-AUC", color="tab:orange", linestyle="--")
    ax.plot(p_epochs, p_edge_auc, label=f"{args.pilot_label} val edge-AUC", color="tab:orange")
    if args.pilot_log2:
        ax.plot(e2, w2, label=f"{args.pilot_label2} val walk-AUC", color="tab:green", linestyle="--")
        ax.plot(e2, ed2, label=f"{args.pilot_label2} val edge-AUC", color="tab:green")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val AUC")
    ax.set_title("Validation AUC")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Bitcoin-alpha, seed 42: production vs. edge-identity-token pilot")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")
    print(f"pilot epochs parsed: {len(p_epochs)}, production epochs parsed: {len(prod_loss_steps)}")


if __name__ == "__main__":
    main()
