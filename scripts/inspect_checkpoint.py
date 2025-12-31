"""Inspect what's saved in a PyTorch Lightning checkpoint."""
import sys
import torch
from pprint import pprint

if len(sys.argv) < 2:
    print("Usage: python inspect_checkpoint.py <checkpoint_path>")
    sys.exit(1)

ckpt_path = sys.argv[1]
print(f"Loading checkpoint: {ckpt_path}\n")

ckpt = torch.load(ckpt_path, map_location="cpu")

print("=" * 80)
print("CHECKPOINT KEYS:")
print("=" * 80)
for key in ckpt.keys():
    print(f"  - {key}")

print("\n" + "=" * 80)
print("HYPER_PARAMETERS (saved by save_hyperparameters):")
print("=" * 80)
if "hyper_parameters" in ckpt:
    pprint(ckpt["hyper_parameters"])
else:
    print("  None found")

print("\n" + "=" * 80)
print("STATE_DICT KEYS (model weights):")
print("=" * 80)
if "state_dict" in ckpt:
    print(f"  Total parameters: {len(ckpt['state_dict'])}")
    print("  Sample keys:")
    for i, key in enumerate(list(ckpt["state_dict"].keys())[:10]):
        print(f"    {key}")
    if len(ckpt["state_dict"]) > 10:
        print(f"    ... ({len(ckpt['state_dict']) - 10} more)")

print("\n" + "=" * 80)
print("CALLBACKS_STATE:")
print("=" * 80)
if "callbacks" in ckpt:
    pprint(ckpt["callbacks"])
else:
    print("  None found")

print("\n" + "=" * 80)
print("EPOCH / GLOBAL_STEP:")
print("=" * 80)
print(f"  epoch: {ckpt.get('epoch', 'N/A')}")
print(f"  global_step: {ckpt.get('global_step', 'N/A')}")

print("\n" + "=" * 80)
print("OPTIMIZER / LR_SCHEDULER STATE:")
print("=" * 80)
print(f"  optimizer_states: {'Present' if 'optimizer_states' in ckpt else 'Missing'}")
print(f"  lr_schedulers: {'Present' if 'lr_schedulers' in ckpt else 'Missing'}")
