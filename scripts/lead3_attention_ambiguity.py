"""
Lead 3 -- Step 3: walk-model attention conditioned on 1-hop ambiguity.

Tests the actual proposed swamping-avoidance mechanism in the walk model:
does attention shift toward 2-hop+ tokens specifically *when the 1-hop
evidence at v is weak/ambiguous*, rather than just generically attending far
(attention_analysis.py already showed mean effective distance 9-16 tokens,
even though average 2-hop+ MI is ~0)? Rising attention-mass-beyond-1-hop with
ambiguity = evidence of adaptive/selective compensation; flat = generic,
unconditional far-reach.

No retraining -- probes the existing E14_HARDNODE_L10 checkpoint via
attention_analysis.analyse_dataset(..., return_per_example=True), which now
retains per-masked-target eff_dist / frac_beyond_1hop (see that file's
recent refactor) plus each target's (u, v) node ids decoded from the
flanking node tokens.

1-hop ambiguity score(v): model-free -- agreement rate among v's own direct
out-edges in the TRAINING graph (excluding the masked target edge itself,
which is in val/test and therefore never in this set): max(frac_pos,
frac_neg) over those signs. 1.0 = all-same-sign (unambiguous local
evidence), 0.5 = perfectly mixed (maximally ambiguous). Targets where v has
no other training out-edges are dropped (no ambiguity signal available).

Usage:
    python scripts/lead3_attention_ambiguity.py --dataset bitcoin-alpha
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from node_mi_structural_embedding import DATASET_CONFIGS, load_dataset_cfg  # noqa: E402
from attention_analysis import load_model_and_dataset, analyse_dataset, MAX_SAMPLES_DEFAULT  # noqa: E402

N_AMBIGUITY_BINS = 5
OUT_DIR_DEFAULT = os.path.join(ROOT, "outputs", "lead3_swamping")


def build_token_lookup(cache_data):
    """node_id_of[token_id] -> int node id, for N_* tokens. Verbatim pattern from
    lead2_walk_relay_mi.py:build_token_lookup (sign_of half dropped, unused here)."""
    id2token = cache_data["tokenizer"]["id2token"]
    node_id_of = {}
    for tid, tok in id2token.items():
        if tok.startswith("N_"):
            try:
                node_id_of[tid] = int(tok.split("_", 1)[1])
            except ValueError:
                pass
    return node_id_of


def build_train_out_signs(cache_data) -> dict:
    """node -> list of signs (+-1) of its directed out-edges in the TRAINING split
    only -- the model's actual training-time context, and disjoint from any val/test
    target edge, so no leakage of "the answer" into the ambiguity score.

    Must read from dataset_cache.pt's own splits['train'], NOT
    baselines/splits/<ds>.pt: the tokenizer's N_<id> tokens use each dataset's raw
    original node ids, while baselines/splits/*.pt remaps node ids to a contiguous
    0..num_nodes-1 range (see baselines/CopulaLSP/loader.py:remap_node_id). These
    two id spaces are NOT the same -- they only coincidentally overlap in range for
    some datasets (silently pairing wrong nodes) and don't overlap at all for others
    (e.g. wiki-rfa, whose raw ids are large account ids -- caught this via a
    zero-targets-retained crash). dataset_cache.pt['splits']['train'] is already
    (src, dst, sign) triples in the same raw-id space as the tokenizer, so no
    remapping is needed here at all."""
    out_signs = {}
    for s, t, w in cache_data["splits"]["train"]:
        out_signs.setdefault(s, []).append(w)
    return out_signs


def ambiguity_score(signs):
    signs = np.asarray(signs)
    frac_pos = (signs > 0).mean()
    return max(frac_pos, 1.0 - frac_pos)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bitcoin-alpha")
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=OUT_DIR_DEFAULT)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = DATASET_CONFIGS[args.dataset]

    device = args.device
    if device != "cpu" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    if device != "cpu" and device.isdigit():
        device = f"cuda:{device}"
    args.device = device

    dscfg = load_dataset_cfg(cfg["ds_name"])
    cache_path = os.path.join(ROOT, dscfg.dataset.data_dir, "dataset_cache.pt")
    cache_data = torch.load(cache_path, map_location="cpu", weights_only=False)
    node_id_of = build_token_lookup(cache_data)
    out_signs = build_train_out_signs(cache_data)

    print(f"=== Lead 3 Step 3: attention-vs-ambiguity, {args.dataset} ===")
    result = analyse_dataset(
        args.dataset, cfg, args.out, stage="test", max_samples=args.max_samples,
        batch_size=args.batch_size, device=args.device,
        return_per_example=True, node_id_of=node_id_of,
    )
    if result is None:
        print("Failed to load model/dataset.")
        return

    per_example = result["per_example"]
    print(f"  {len(per_example):,} masked targets total")

    # ── Verification: re-aggregating per-example values must reproduce the
    # dataset-level eff_dist already reported by attention_analysis.py ──────────
    reagg_eff = np.mean([r["eff_dist"] for r in per_example])
    published_eff = result["eff_dist"].mean()
    print(f"  [verify] re-aggregated mean eff_dist = {reagg_eff:.4f} vs. "
          f"dataset-level eff_dist.mean() = {published_eff:.4f} "
          f"(diff={abs(reagg_eff - published_eff):.4f}, expect ~0)")

    ambiguity, frac_beyond = [], []
    n_dropped_no_context = 0
    for r in per_example:
        v = r["v"]
        # require >=2 out-edges: with exactly 1, ambiguity_score is degenerately
        # always 1.0 (trivially "unambiguous"), which would inflate the
        # least-ambiguous bin with cases carrying no real agreement signal.
        if v is None or len(out_signs.get(v, [])) < 2:
            n_dropped_no_context += 1
            continue
        ambiguity.append(ambiguity_score(out_signs[v]))
        frac_beyond.append(r["frac_beyond_1hop"])
    ambiguity = np.asarray(ambiguity)
    frac_beyond = np.asarray(frac_beyond)
    print(f"  {len(ambiguity):,} targets retained ({n_dropped_no_context:,} dropped: "
          f"v has no other training out-edges or token decode failed)")

    # bin by ambiguity score (low ambiguity_score = high agreement = low ambiguity;
    # report bins from most-ambiguous (score near 0.5) to least-ambiguous (near 1.0))
    edges = np.quantile(ambiguity, np.linspace(0, 1, N_AMBIGUITY_BINS + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    bin_idx = np.digitize(ambiguity, edges[1:-1], right=True)

    lines = [
        "=" * 88,
        f"  LEAD 3 STEP 3 -- attention-mass-beyond-1-hop vs. 1-hop ambiguity, {args.dataset}",
        "=" * 88,
        "",
        "ambiguity_score(v) = agreement rate among v's training-graph out-edge signs",
        "(1.0 = all same sign / unambiguous, 0.5 = perfectly mixed / maximally ambiguous).",
        "Rising mean_frac_beyond_1hop as ambiguity_score DECREASES (i.e. as ambiguity",
        "increases) = evidence attention adaptively compensates for weak 1-hop evidence.",
        "Flat trend = generic far-reach, unconditional on local ambiguity.",
        "",
        f"n_targets_used={len(ambiguity):,}, n_dropped={n_dropped_no_context:,}",
        f"verify: re-agg eff_dist={reagg_eff:.4f} vs published={published_eff:.4f}",
        "",
        f"{'bin':>4}{'ambig_range':>18}{'n':>8}{'mean_ambiguity':>16}{'mean_frac_beyond_1hop':>24}",
    ]
    print(lines[-1])
    for b in range(N_AMBIGUITY_BINS):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        rng = f"[{ambiguity[mask].min():.3f},{ambiguity[mask].max():.3f}]"
        row = (f"{b:>4}{rng:>18}{mask.sum():>8}{ambiguity[mask].mean():>16.4f}"
               f"{frac_beyond[mask].mean():>24.4f}")
        print(row)
        lines.append(row)

    corr = np.corrcoef(ambiguity, frac_beyond)[0, 1]
    spearman = np.corrcoef(
        np.argsort(np.argsort(ambiguity)), np.argsort(np.argsort(frac_beyond))
    )[0, 1]
    summary = (f"\nPearson corr(ambiguity_score, frac_beyond_1hop) = {corr:.4f}  "
               f"(negative = attention reaches further when ambiguity is HIGHER, "
               f"consistent with adaptive compensation)\n"
               f"Spearman corr = {spearman:.4f}")
    print(summary)
    lines.append(summary)

    out_path = os.path.join(args.out, f"attention_ambiguity_{args.dataset}.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
