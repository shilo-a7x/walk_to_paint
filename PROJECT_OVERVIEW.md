# Walk-to-Paint: Complete Project Overview

> **Purpose:** This document lets any colleague with repo access fully understand, reproduce, and extend this research. Read top-to-bottom once; use section headers to jump back later.

> **⚠️ STATUS UPDATE (2026-06-29) — read before trusting result tables below.** Two things in
> this overview are superseded:
> 1. **SOTA numbers.** Tables in §1 (Comparison), §9 (best aggregator), and §11 (Results) are
>    the **old uniform-sampler / E14_HARDNODE_L10** numbers. Current SOTA is the **E15
>    `k_cover` k=5** full-coverage model — see the SOTA table in `CLAUDE.md` and
>    `outputs/walk_coverage_analysis/E15_SWEEP_RESULTS.md`. The qualitative story (walk beats
>    every GNN on all 6) holds and is now *stronger* (full coverage + same-partition splits).
> 2. **The "walk-coverage gap" is RESOLVED**, not open: the E15 edge-anchored sampler gives
>    ~100% node+edge coverage on all 6 (`WALK_COVERAGE.md`). Treat every "walk coverage is
>    lower / open" note below as historical.
>
> Also note: **OWL (occurrence-weighted loss) is dead code** (out of scope, never adopted) —
> ignore the `occurrence_weight_path` / OWL references in §6 and §13.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Datasets](#2-datasets)
3. [Data Loading Pipeline](#3-data-loading-pipeline)
4. [Model Input Format](#4-model-input-format)
5. [Transformer Architecture](#5-transformer-architecture)
6. [Loss and Optimization](#6-loss-and-optimization)
7. [Advanced Training Mechanisms](#7-advanced-training-mechanisms)
8. [Training Pipeline — `run.py`](#8-training-pipeline--runpy)
9. [Post-Hoc Aggregation](#9-post-hoc-aggregation)
10. [Post-Hoc Pipeline — `run_posthoc.py`](#10-post-hoc-pipeline--run_posthocpy)
11. [Results](#11-results)
12. [Research Analyses](#12-research-analyses)
13. [Research Status and Open Questions](#13-research-status-and-open-questions)
14. [Quickstart Commands](#14-quickstart-commands)
15. [Repo Map](#15-repo-map)
16. [Config Cheat-Sheet](#16-config-cheat-sheet)
17. [Known Gotchas](#17-known-gotchas)

---

## 1. Problem Statement

### What we are solving

**Edge sign prediction in directed signed graphs.** Given a graph where edges carry a binary label — `+1` (trust / support / friendship) and `-1` (distrust / opposition / enemy) — predict the sign of a *held-out* edge from the structure and signs of the *observed* edges.

Real-world signed graphs arise in:
- **Bitcoin trust networks** — users rating each other as trustworthy or not before transacting
- **Wikipedia editor elections** — editors voting support or oppose on administrator nominations
- **Epinions / Slashdot** — web-of-trust and friend/foe tagging systems

### Why it is hard

1. **Severe class imbalance.** Positive edges dominate in all six datasets (77–94%), yet the negative edges are the informative ones for downstream applications (fraud detection, conflict prediction).
2. **No node features.** We have only graph topology and edge signs — no user profiles, timestamps, or content.
3. **Structural sparsity.** Even in the largest dataset (epinions, 841k edges), many nodes have low degree and appear rarely in training context.
4. **Walk-coverage gap.** Any single random walk visits a tiny fraction of all edges; the model must generalise from walk-sampled context to unseen edges. *(Note: the related evaluation-coverage gap — the walk model only scoring edges that appear in a sampled walk — is RESOLVED as of 2026-06-29 by the E15 k_cover sampler; see `WALK_COVERAGE.md`.)*

### Our approach

Convert the graph into **random walk sequences** of alternating node and edge tokens, then train a **Transformer encoder** to predict masked edge signs. At inference, aggregate predictions from hundreds of walk occurrences of each test edge. This is similar in spirit to the way BERT predicts masked words from sentence context — but applied to graph walks instead.

### Comparison to published baselines

**Table A — published numbers, different splits/preprocessing (approximate, as of 2025).** These come from each method's original paper, using *that paper's* train/test split and preprocessing — not ours. Useful as a rough sanity check, but not an apples-to-apples comparison:

| Dataset | SGCN | SiGAT | SDGNN | **Ours (edge-level)** |
|---------|------|-------|-------|----------------------|
| bitcoin-alpha | 0.852 | 0.871 | 0.889 | **0.913** |
| bitcoin-otc | 0.874 | 0.890 | 0.917 | **0.943** |
| epinions | ~0.87 | — | ~0.91 | **0.945** |
| wiki-rfa | ~0.80 | — | ~0.85 | **0.882** |
| slashdot | ~0.83 | — | ~0.88 | **0.895** |

**Table B — same-split, verified comparison.** `baselines/` contains independent re-implementations of several GNN/SGNN baselines (CSG, CSG-GSGNN, CopulaLSP, GSGNN+SGA, SiGAT+SGA, SE-SGformer, SNEA), trained and evaluated on our splits. The table below compares the best such baseline per dataset against our model's test AUC (from §11).

> **⚠️ This table is doubly superseded (2026-06-29):**
> 1. The "best GNN" column is from the **old `baselines/splits/*.pt`**, which turned out to be
>    generated *independently* of the walk split (~10% edge overlap, plus fabricated reverse
>    edges) — NOT the apples-to-apples comparison it claims to be. The corrected
>    **canonical-split** rerun is in `CANONICAL_RERUN_FINDINGS.md` / `baselines/all_results_canonical.csv`
>    (see also `SPLIT_PROVENANCE.md`, `FABRICATED_REVERSE_EDGES.md`).
> 2. The "Ours" column is the **old uniform-sampler** walk model; current SOTA is **E15
>    k_cover k=5** (CLAUDE.md). On identical, full-coverage, same-partition edges the walk
>    model beats **every** GNN on **all 6** datasets, both attention variants.
>
> The original (now-stale) table is retained below for provenance:

| Dataset | Best GNN/SGNN baseline (model, AUC) | Ours (test AUC) | Margin |
|---|---|---|---|
| bitcoin-alpha | GSGNN+SGA, 0.8804 | 0.9131 | +0.0327 |
| bitcoin-otc | GSGNN+SGA, 0.9086 | 0.9433 | +0.0347 |
| epinions | CSG-GSGNN, 0.9113 | 0.9446 (lgbm) | +0.0333 |
| wiki-elec | CSG-GSGNN, 0.8840 | 0.8928 | +0.0088 |
| wiki-rfa | SiGAT+SGA, 0.8676 | 0.8816 | +0.0140 |
| slashdot090221 | SNEA, 0.8845 | 0.8954 | +0.0109 |

We beat every GNN/SGNN baseline on every dataset on the same splits; wiki-elec is the closest margin (+0.0088). For wiki-rfa, SiGAT+SGA (0.8676) edges out the previous best CSG-GSGNN (0.8673) by +0.0003 — within noise, and obtained from a different augset split (`--num 2`, see below) — but it's the highest baseline number available so it's listed.

**Split-parity provenance.** `old_chats/baselines.json` (a prior development session transcript) documents that `baselines/` was purpose-built so "the data look exactly the same as ours but the model is different" — split parity was an explicit design goal, not an afterthought. CSG and CSG-GSGNN were added later specifically because the GSGNN+SGA paper's own reported numbers looked "suspiciously high," motivating an independent same-split re-run with additional models.

This is confirmed numerically. `baselines/splits/bitcoin-alpha.pt` stores `uni_trn_mask`/`uni_val_mask`/`uni_tst_mask` (undirected, deduplicated edges) with sizes `11299 / 1412 / 1413` out of `14124` total — ratios `0.7999 : 0.0999 : 0.1001 ≈ 0.8 : 0.1 : 0.1`. `baselines/splits/epinions.pt` (much larger, `711210` unique edges) gives `568968 / 71121 / 71121` — exactly `0.8 : 0.1 : 0.1`. Both match `config.yaml`'s canonical split ratios exactly: `train_ratio=0.48 + mask_ratio=0.32 = 0.80`, `val_ratio=0.10`, `test_ratio=0.10`, with `reproducibility.seed=42` — i.e. baseline `trn_mask` = our `train ∪ mask`, and `val_mask`/`tst_mask` = our `val`/`test`, generated via the same seeded stratified `train_test_split` sequence as `src/data/prepare_data.py::split_edges`.

**SiGAT (GAT-based) — now run.** `baselines/SGA/sigat_SGA.py` (SGA's data-augmentation pipeline applied to `SiGAT`, a GAT-based signed-graph model) has been run for all 6 datasets, selecting the `0.4-0.45-0.98-0.98` augset row exactly like the GSGNN+SGA rows: bitcoin-alpha 0.8483, bitcoin-otc 0.8899, epinions 0.9081, wiki-elec 0.8463, slashdot090221 0.8003, wiki-rfa 0.8676. These are folded into Table B above as "SiGAT+SGA" rows in `baselines/all_results.csv`. Note: `wiki-RfA/tests/wiki-RfA-test-1.csv` (the `--num 1` test file) contains corrupted node ids (max id ≈ 1e9, vs ≈1.1e4 for the train graph), causing a 74.5 GiB OOM in `sigat_SGA.py`'s `np.random.rand(num_nodes, channels)` allocation; wiki-rfa's SiGAT+SGA result was instead obtained with `--num 2` (same `0.4-0.45-0.98-0.98` augset, sane test-2 file).

---

## 2. Datasets

### Dataset table

| Dataset | Nodes | Edges (canonical) | Positive % | Format | Config |
|---------|------:|------------------:|-----------:|--------|--------|
| `bitcoin-alpha` | 3,783 | 24,186 | 93.7% | CSV gzip (source, target, rating, time) | `configs/bitcoin-alpha.yaml` |
| `bitcoin-otc` | 5,901 | 35,592 | 90.0% | CSV gzip (source, target, rating, time) | `configs/bitcoin-otc.yaml` |
| `epinions` | ~131k | 841,372 | 85.3% | Space-separated txt gzip | `configs/epinions.yaml` |
| `wiki-elec` | ~7.1k | 103,689 | 78.8% | Block-structured txt gzip (votes) | `configs/wiki-elec.yaml` |
| `wiki-rfa` | ~11.4k | 184,546 | 78.4% | Block-structured txt gzip (votes) | `configs/wiki-rfa.yaml` |
| `slashdot090221` | ~82k | 549,202 | 77.4% | Space-separated txt gzip | `configs/slashdot090221.yaml` |

> **"Canonical" edge counts** are after `binary=true` (neutrals dropped), self-loop removal, and `most_recent` multiedge handling. These match exactly what the model trains on.

### Data files location

```
data/
  bitcoin-alpha/   soc-sign-bitcoinalpha.csv.gz
  bitcoin-otc/     soc-sign-bitcoinotc.csv.gz
  epinions/        soc-sign-epinions.txt.gz
  wiki-Elec/       wikiElec.ElecBs3.txt.gz
  wiki-RfA/        wiki-RfA.txt.gz
  slashdot090221/  soc-sign-Slashdot090221.txt.gz
```

### Dataset-specific notes

- **bitcoin-alpha / bitcoin-otc:** `multiedge_handling: most_recent` — only the most recent rating between any (u,v) pair is kept.
- **wiki-elec:** Votes include a neutral class (0). `binary: true` drops these, reducing raw 107,080 edges to 103,689. Missing this step would contaminate training with meaningless neutral pairs.
- **epinions / slashdot:** `multiedge_handling: keep` — all occurrences are retained (no duplicate problem in these datasets after preprocessing).

---

## 3. Data Loading Pipeline

The full pipeline runs inside `run.py` → `src/data/prepare_data.py`.

### 3.1 Edge loading — `src/data/datasets.py`

Each dataset has a dedicated loader:

| Loader | Dataset(s) | Format |
|--------|-----------|--------|
| `load_bitcoin()` | bitcoin-alpha, bitcoin-otc | CSV: `(source, target, rating, timestamp)` |
| `load_wiki_rfa()` | wiki-rfa | Block text: `SRC: / TGT: / VOT: / RES:` fields |
| `load_wiki_elec()` | wiki-elec | Block text (similar format) |
| `load_epinions()` | epinions | Space-separated: `source target sign` |
| `load_slashdot()` | slashdot | Space-separated: `source target sign` |

All loaders return a list of `(source_id, target_id, label)` tuples with `label ∈ {-1, 0, +1}`.

**Preprocessing** (`postprocess_edges()`):
1. **Self-loop removal** — drop any edge where `u == v`
2. **Multiedge handling** — per config: `most_recent` keeps latest timestamp, `keep` keeps all, `aggregate_majority` sums and thresholds
3. **Binary mode** — drop all edges with `label == 0` (neutral)

The function that does this correctly, using the training pipeline's own loader, is `load_edges_canonical()` in `scripts/balance_theory_paths.py` — import this for any analysis script to guarantee consistency.

### 3.2 Stratified splitting — `src/data/prepare_data.py`

Default split ratios (`config.yaml`):

| Split | Ratio | Purpose |
|-------|------:|---------|
| `train` | 48% | Walk context (never a prediction target) |
| `mask` | 32% | Training targets (supervised, masked in input) |
| `val` | 10% | Validation during training |
| `test` | 10% | Final evaluation |

Splitting is **hierarchical stratified**: first separate train from the rest (stratify on label), then mask from (val+test), then val from test. This ensures each split has class balance within ±2% of the full dataset.

Result: four Python `frozenset`s of `(u, v, label)` tuples for O(1) membership testing.

### 3.3 Walk sampling — `src/data/walk_sampler.py`

```
sample_random_walks(edges, num_walks, max_walk_length, num_workers, seed)
```

**Algorithm:**
1. Build adjacency: `graph[u] → [(v1, s1), (v2, s2), ...]`
2. For each walk `i` (seeded deterministically as `base_seed + i`):
   - Pick a random start node
   - Follow outgoing edges at random until `max_walk_length` tokens or a dead end
3. Use multiprocessing (8 workers by default); results sorted by task ID for reproducibility

**Walk strategy** (`dataset.walk_strategy`): The config default is `uniform`. Many alternatives are implemented (`guaranteed`, `neg_emphasis`, `node2vec`, `edge_seeded`, etc.). **Current SOTA (2026-06-29) uses `k_cover` with `walk_k_min=5`** — an edge-anchored sampler that visits every edge ≥5× and drives node+edge coverage to ~100% on all 6 (matches/beats uniform AUC). See `WALK_COVERAGE.md`, `src/data/coverage_aware_sampler.py::k_cover_walks_fast`. `uniform` was the best-performing baseline among the older strategies but has the coverage gap on sparse graphs.

**Walk counts per dataset:**

| Dataset | num_walks | max_walk_length |
|---------|----------:|----------------:|
| bitcoin-alpha | 5,000,000 | 80 |
| bitcoin-otc | 5,000,000 | 80 |
| epinions | 5,000,000 | 80 |
| wiki-elec | 5,000,000 | 80 |
| wiki-rfa | 5,000,000 | 80 |
| slashdot090221 | 5,000,000 | 80 |

### 3.4 Tokenization — `src/data/tokenizer.py`

**Special tokens:**

| Token | ID | Meaning |
|-------|----|---------|
| `<PAD>` | 0 | Padding to batch length |
| `<MASK>` | 1 | Masked edge (prediction target) |
| `<UNK>` | 2 | Unknown / replacement token |

**Vocabulary:**
- Node tokens: `N_{node_id}` (e.g., `N_42`)
- Edge tokens: `E_1` (positive), `E_-1` (negative)
- Vocab size ranges from ~3,800 (bitcoin-alpha) to ~132,000 (epinions)

**Edge label encoding** (uniform across all datasets):
- Class 0 = distrust (`E_-1`, minority)
- Class 1 = trust (`E_1`, majority)
- `q_j = P(class 1) = P(trust)` — the number the model outputs

### 3.5 Walk encoding

Each walk token is converted to:
- `input_ids[i]` — integer token ID
- `edge_split_mask[i]` — which split this edge belongs to: `TRAIN=0, MASK=1, VAL=2, TEST=3, BAD=-1`
- `edge_ids[i]` — global edge ID (same (u,v,label) always gets the same ID)

### 3.6 CSR storage and caching

Encoded walks are stored in ragged CSR format:
- `offsets` — cumulative walk lengths (int32/int64)
- `flat_input_ids` — concatenated token IDs
- `flat_split_mask` — concatenated split masks (int8)
- `flat_edge_ids` — concatenated edge IDs

Saved to `dataset_cache.pt` in the output directory. On subsequent runs with `preprocess.use_cache: true`, this file is loaded directly, skipping walk sampling and encoding (saves 5–20 minutes for large datasets).

---

## 4. Model Input Format

### Walk sequence layout

A walk of length $L$ edges visits $L+1$ nodes. The token sequence is:

```
[N_u0,  E_s1,  N_u1,  E_s2,  N_u2,  ...  E_sL,  N_uL,  <PAD>, ...]
```

Position parity: even positions = node tokens, odd positions = edge tokens. Sequence length = `2 * num_edges_in_walk + 1`, padded to the batch maximum.

### During training

For edges in the `mask` or `train` split that are selected as targets this epoch:
- The edge token is replaced with `<MASK>` (ID=1)
- The model must predict the original sign from context

For all other positions: token IDs are passed through unchanged.

### At inference

All val/test edges are masked; the model predicts them. Train/context edges are left visible as context.

### Positional encoding

Sinusoidal (fixed, non-learned), standard Transformer formula:
$$PE_{(pos,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

Max sequence length = `2 * max_walk_length + 1`.

---

## 5. Transformer Architecture

### Model class: `TransformerModel` — `src/model/model.py`

```
Input token IDs → Embedding(vocab_size, d_model) + SinPE
→ TransformerEncoder(nhead, d_ffn, nlayers, dropout)
→ Linear(d_model, 2)
→ logits at every masked edge position
```

Only the **masked edge positions** contribute to the loss. The final hidden state at position `i` is projected to 2-class logits; the class with higher logit wins.

### Per-dataset hyperparameters

| Dataset | `embedding_dim` | `hidden_dim` | `nhead` | `nlayers` | `dropout` |
|---------|----------------:|-------------:|--------:|----------:|----------:|
| bitcoin-alpha | 64 | 128 | 4 | 3 | 0.20 |
| bitcoin-otc | 64 | 128 | 4 | 3 | 0.20 |
| wiki-rfa | 64 | 64 | 2 | 5 | 0.147 |
| wiki-elec | 64 | 64 | 2 | 5 | 0.147 |
| epinions | 32 | 32 | 8 | 3 | 0.001 |
| slashdot090221 | 128 | 256 | 4 | 4 | 0.006 |

These were found by Optuna hyperparameter search (50–100 trials per dataset).

### Training hyperparameters

| Dataset | `lr` | `weight_decay` | `batch_size` | `grad_clip` | `epochs` |
|---------|-----:|---------------:|-------------:|------------:|---------:|
| bitcoin-alpha | 1.6e-3 | 1e-4 | 1024 | 0.54 | 50 |
| bitcoin-otc | 1.6e-3 | 1e-4 | 1024 | 0.54 | 50 |
| wiki-rfa | 1.6e-3 | 3e-5 | 1024 | 0.54 | 50 |
| wiki-elec | 1.6e-3 | 3e-5 | 1024 | 0.54 | 50 |
| epinions | 2.9e-3 | 2.9e-4 | 1024 | 0.11 | 50 |
| slashdot090221 | 1.2e-4 | 4.5e-4 | 1024 | 0.95 | 50 |

---

## 6. Loss and Optimization

### Class weighting

Computed from the `mask` split (train targets):
$$w_c = \frac{1/\text{count}(c)}{\sum_{c'} 1/\text{count}(c')}$$

Applied to `F.cross_entropy(..., weight=class_weights)`. Example for bitcoin-alpha: `[0.103, 1.897]` — distrust (minority) gets ~18× the weight of trust.

### Optimizer and scheduler

- **AdamW** with per-dataset `lr` and `weight_decay` (see table above)
- **CosineAnnealingLR**: LR decays from `lr` to near 0 over `epochs` epochs
- **Gradient clipping**: `max_norm = gradient_clip_val` (per-dataset)
- **Early stopping**: patience = 15 epochs, monitor = `val_auc_epoch`

### Optional loss components

| Flag | What it does | Key param |
|------|-------------|-----------|
| `model.hardness_lambda > 0` | Per-sample reweighting by adjacent node hardness | `hardness_lambda: 1.0` |
| ~~`occurrence_weight_path`~~ | **DEAD CODE (OWL) — never adopted, out of scope.** Up-weighted low-coverage edges; superseded by the k_cover sampler addressing coverage at the source | — |

---

## 7. Advanced Training Mechanisms

All three mechanisms live in `src/model/lit_model.py` and are controlled by config flags.
They were systematically ablated in `scripts/run_transformer_incremental_experiments.py`.

### E10 — Node Token Replacement (`node_context_mode: replace`)

**Problem:** The model can over-rely on memorising specific node IDs rather than learning generalizable sign patterns.

**Mechanism:** During training, with probability `node_replace_prob` (default 0.2), each node token is replaced:
- 70% of the time → `<UNK>` (ID=2): forces the model to rely on edge-sign context alone
- 30% of the time → a random node from the batch: acts as a form of data augmentation

Applied only to node tokens (even positions in the walk), not edge tokens.

**Config:**
```yaml
model:
  node_context_mode: replace   # "none" to disable
  node_replace_prob: 0.2
  node_replace_unk_ratio: 0.7
```

### E12 — Dynamic Train Masking (`dynamic_train_masking: true`)

**Problem:** With a fixed mask/train split, the model can memorise *which* edges it is asked to predict, rather than learning to predict them from context.

**Mechanism:** At the start of each epoch, the boundary between `train` (context) and `mask` (target) is re-drawn at random — while keeping `val` and `test` frozen. Concretely, the union `train ∪ mask` is shuffled (stratified), and 40% is designated as that epoch's targets. The model never sees the same set of targets twice.

**Config:**
```yaml
model:
  dynamic_train_masking: true
  dynamic_train_mask_seed_offset: 0
```

### E14 — Node Hardness Map (`hardness_map_path: ...`)

**Problem:** Not all training examples are equally informative. Easy edges (always-positive nodes) waste gradient capacity; hard edges (sign-ambiguous structural contexts) should be emphasised.

**How the hardness map is built** (`scripts/compute_hardness_map.py`):
1. Train a **tiny miner transformer** (emb=16, hidden=16, nhead=2, nlayers=2, 5 epochs) on the same train+mask pool
2. After training, run inference on the train set. For each masked edge at position `i`, record whether the prediction was correct, and attribute this to the **two adjacent node tokens** (at positions `i-1` and `i+1`)
3. `hardness[node] = 1 − (correct_count / total_count)` — high when the miner consistently fails in that node's neighbourhood
4. Save as `hardness_map.pt` (float32 tensor of shape `[vocab_size]`)

**Short-walk filter — NOT part of the production recipe (corrected 2026-07-07):** an
earlier version of this doc described `--max-walk-edges=7` here as if it were standard.
Verified against `scripts/run_transformer_incremental_experiments.py`: the actual
`E14_HARDNODE_L10` config (behind every hardness map in production) leaves this at its
default of `0` (no filter). The filter was only used in two variants that both
underperformed the no-filter config (`E14_DRH_SHORT7_E15_L10`, `E15_DRH_DYNMINER_L10` —
see `old_chats/DRH.md` and `~/.claude/plans/plan-hardness-miner.md`). Combining the
filter with the miner's already-tiny capacity over-restricts data per node. Leave
`--max-walk-edges` at 0 unless deliberately testing it.

**How it is used during main training:**
- Load `hardness_map.pt` as a registered buffer
- For each masked edge at position `i`: `h = λ × mean(hardness[node_{i-1}], hardness[node_{i+1}])`
- Per-sample loss weight: `w = 1 + h` (hard contexts get higher weight)

**Config:**
```yaml
model:
  hardness_lambda: 1.0
  hardness_map_path: outputs/.../artifacts/E14_HARDNODE_L10/hardness_map.pt
```

**Key paths** for existing hardness maps:

| Dataset | hardness_map.pt path |
|---------|---------------------|
| bitcoin-alpha | `outputs/transformer_incremental/bitcoin-alpha_seed42_nw5000000_mw80_bs1024_ep75_20260423-111407/artifacts/E14_HARDNODE_L10/hardness_map.pt` |
| bitcoin-otc | `outputs/transformer_incremental/bitcoin-otc_seed42_nw500000_mw80_bs1024_ep50_20260419-133621/.../hardness_map.pt` |
| epinions | `outputs/transformer_incremental/epinions_seed42_nw500000_mw80_bs1024_ep50_20260416-125419/.../hardness_map.pt` |
| wiki-elec | `outputs/transformer_incremental/wiki-elec_seed42_nw500000_mw80_bs1024_ep50_20260419-133621/.../hardness_map.pt` |
| wiki-rfa | `outputs/transformer_incremental/wiki-rfa_seed42_nw500000_mw80_bs1024_ep50_20260324-141920/.../hardness_map.pt` |
| slashdot090221 | `outputs/transformer_incremental/slashdot090221_seed42_nw5000000_mw80_bs1024_ep50_20260420-134149/.../hardness_map.pt` |

---

## 8. Training Pipeline — `run.py`

### Command

```bash
python run.py [--config CONFIG] [--device N] [dotlist overrides...]
```

Examples:
```bash
# Train bitcoin-alpha on GPU 0 (uses configs/bitcoin-alpha.yaml automatically)
CUDA_VISIBLE_DEVICES=0 python run.py dataset.name=bitcoin-alpha

# Train with a specific hardness map
CUDA_VISIBLE_DEVICES=1 python run.py dataset.name=wiki-rfa \
    model.hardness_map_path=outputs/.../hardness_map.pt \
    model.hardness_lambda=1.0

# Resume from checkpoint
CUDA_VISIBLE_DEVICES=2 python run.py dataset.name=epinions \
    training.resume_from_checkpoint=checkpoints/last.ckpt
```

### Config hierarchy

1. Load `config.yaml` (base defaults)
2. Auto-merge `configs/<dataset.name>.yaml` (dataset overrides) — triggered by any `dataset.name=X` override
3. Apply CLI dotlist overrides (highest priority)

> **Critical:** Pass the dataset name as `dataset.name=X` (a dotlist override), **not** as `--config configs/X.yaml`. The latter is ignored if passed positionally.

### What happens during a run

1. Seed everything: `seed_everything(cfg.reproducibility.seed)` (default 42)
2. Auto-generate `exp_name` with timestamp if placeholder
3. Create output dir: `outputs/<dataset>/<exp_name>_<timestamp>/`
4. Data pipeline: load → preprocess → split → walk → tokenize → cache
5. Train with PyTorch Lightning `Trainer`:
   - `ModelCheckpoint`: saves all epochs to `<output_dir>/checkpoints/`
   - `EarlyStopping`: patience=15, monitors `val_auc_epoch`
   - TensorBoard logs to `logs/<dataset>-<exp_name>/`
6. All checkpoints named: `<dataset>-<exp_name>-epoch={E:02d}-val_auc_epoch={AUC:.4f}.ckpt`

### GPU convention

4 × NVIDIA L40S available (indices 0–3). Always set `CUDA_VISIBLE_DEVICES=N` before the python command.

---

## 9. Post-Hoc Aggregation

### Motivation

The Transformer outputs one probability `q_j = P(trust)` **per walk occurrence** of each edge. A single test edge typically appears in 300–2000 different walks. These per-walk predictions need to be aggregated into a single edge-level score.

The naive approach is a simple mean. But per-walk predictions vary in quality: an edge buried at the very end of a long walk has less context than one in the middle, and a high-confidence prediction (q near 0 or 1) is more informative than an uncertain one (q ≈ 0.5). Aggregation exploits this.

### Three aggregation families

#### 1. Functional aggregators (36 variants)

Parametric weighted means optimized on the validation set:
$$\hat{y}_e = \frac{\sum_j w_j \, q_j}{\sum_j w_j}$$

where $w_j$ is a function of per-walk features:
- $q_j$ — transformer's class-1 probability
- $l_j = d^s_j + d^e_j + 1$ — walk length
- $d^s_j$ — steps from walk start to edge position
- $d^e_j$ — steps from edge position to walk end
- $\rho_j = d^s_j / l_j$ — relative position within walk

**Weight function groups:**

| Group | Formula | Params |
|-------|---------|--------|
| Baseline | $w=1$ (uniform mean) | 0 |
| Length | $w = l^{-a}$ | 1 |
| Position | $w = \exp(a \cdot d^s)$ or $w = (d^s d^e)^a$ | 1 |
| Confidence | $w = |q-0.5|^a$ | 1 |
| Len × Pos | combined | 2 |
| Len × Conf | $w = l^{-a} \cdot (-\log q)^b$ | 2 |
| Pos × Conf | combined | 2 |
| Log-linear | multi-term | 3–6 |
| Log-odds | $w = |\log(q/(1-q))|^b$ | 1–2 |
| Trust-conf | $w = (1/q)^a$ | 1 |

Parameters $\theta^*$ are found by Nelder-Mead minimizing $-\text{AUC}_\text{val}$.

#### 2. LightGBM aggregator

Train a `LGBMClassifier` on val-set walk occurrences with features `[dist_from_start, walk_length, q_j]`. Then:
1. Run LightGBM on all walk occurrences to get refined scores $\hat{q}_j$
2. Pool: `ŷ_e = mean(ŷ_j)`

Variants: `lgbm` (mean), `lgbm_lse` (log-sum-exp), `lgbm_max`, `lgbm_attention` (neural attention gate), `lgbm_set_attention`.

#### 3. Simple baselines

`func_uniform` (unweighted mean), `func_max` (max pooling).

### Best results per dataset

| Dataset | Best aggregator | Test AUC |
|---------|----------------|----------|
| bitcoin-alpha | `func_len_logq` / `func_logit_power` | **0.913** |
| bitcoin-otc | `func_len_cert` | **0.943** |
| wiki-elec | nearly all (uniform mean wins) | **0.893** |
| wiki-rfa | `func_len_conf` / `func_conf_logit` | **0.882** |
| epinions | `lgbm` | **0.945** |
| slashdot | `func_logit_power` | **0.895** |

> **wiki-elec anomaly:** LightGBM is 0.010 *below* the unweighted mean here. Possibly because walk occurrences are so uniformly distributed in this dataset that all have similar quality.  
> **epinions anomaly:** LightGBM is 0.013 *above* the best functional form — the feature interactions in epinions require a non-linear aggregator.

---

## 10. Post-Hoc Pipeline — `run_posthoc.py`

### Command

```bash
python run_posthoc.py \
    --config config.yaml \
    --exp-dir outputs/<dataset>/<run_name> \
    --checkpoint-choice best \
    --splits train,val,test \
    --artifacts predictions,triplets,heatmaps,aggregator \
    --agg-models lgbm \
    --device 0
```

### Steps

1. **Resolve checkpoint** (`--checkpoint-choice best`): picks highest `val_auc` from filename; fallback to lowest `val_loss`, then newest
2. **Generate predictions**: run the model on all three splits, save `(edge_id, walk_id, prob, target, dist_from_start, dist_from_end, walk_length)` per occurrence
3. **Triplet extraction**: `(edge_id, position, walk_len, correct)` tuples for heatmap visualization
4. **Heatmap**: 2D accuracy grid over `(dist_from_start, walk_length)` — diagnoses where in a walk predictions are most reliable
5. **Aggregation**: train aggregator on val occurrences, predict test edge scores, compute AUC

Results saved to `<exp-dir>/<checkpoint_stem>_posthoc/`.

---

## 11. Results

> **⚠️ Superseded (2026-06-29).** The tables below are the old uniform-sampler
> `E14_HARDNODE_L10` results. Current SOTA is **E15 k_cover k=5** — see `CLAUDE.md` and
> `outputs/walk_coverage_analysis/E15_SWEEP_RESULTS.md`. Retained for provenance.

### Best test AUC (experiment tag: `E14_HARDNODE_L10`)

| Dataset | Walks | Transformer walk AUC | Best edge AUC | Best aggregator | Gain |
|---------|------:|---------------------:|--------------:|-----------------|-----:|
| bitcoin-alpha | 5M | 0.8993 | **0.9131** | `func_len_logq` | +0.014 |
| bitcoin-otc | 5M | 0.9229 | **0.9433** | `func_len_cert` | +0.020 |
| epinions | 5M | 0.9334 | **0.9446** | `lgbm` | +0.011 |
| wiki-elec | 5M | 0.8562 | **0.8928** | uniform mean | +0.037 |
| wiki-rfa | 5M | 0.8415 | **0.8816** | `func_len_conf` | +0.040 |
| slashdot090221 | 5M | 0.8818 | **0.8954** | `func_logit_power` | +0.014 |

Best checkpoints used for these results:

| Dataset | Best epoch | Val AUC |
|---------|----------:|--------:|
| bitcoin-alpha | 24 | 0.9342 |
| bitcoin-otc | 60 | 0.9247 |
| epinions | 42 | 0.9344 |
| wiki-elec | 36 | 0.8544 |
| wiki-rfa | 25 | 0.8630 |
| slashdot090221 | 20 | 0.9279 |

---

## 12. Research Analyses

### 12.1 Balance Theory Path Analysis — `scripts/balance_theory_paths.py`

**Question:** Does structural balance theory predict edge signs via path sign-products?

**Background:** Balance theory states that signed triangles tend toward "balance" (product of signs = +1). This predicts: the sign product along any path u→...→v approximates the sign of the direct edge u→v.

**Method:**
- For each test edge `(u→v, Y)`, find all directed paths of length d from u to v (d=2..5) via DFS
- Compute path sign product: `P = ∏ sign(each edge in path)`
- `path_balance_score = mean(P)` over all d-paths → positive means "balance theory predicts positive"
- Two modes:
  - **full**: use all edges including the reciprocal `v→u`
  - **debidir**: remove `v→u` before path search (the "de-bidirectional" confound correction)

**Key findings:**

| Dataset | Bidir% | Full d=2 sign_acc | Debidir d=2 sign_acc | Debidir d=2 coverage |
|---------|-------:|------------------:|---------------------:|---------------------:|
| bitcoin-alpha | ~83% | ~90% | ~55% | ~6% |
| bitcoin-otc | ~79% | 86% | 47% | 8% |
| epinions | ~31% | 76% | 65% | 43% |
| wiki-elec | ~6% | 65% | 63% | 80% |
| wiki-rfa | ~8% | 67% | 64% | 83% |
| slashdot | ~18% | 72% | 63% | 22% |

**Interpretation:**
- For bitcoin (high bidir%): the apparent balance theory signal is dominated by the **bidirectional confound** — most paths go through the reciprocal edge `v→u`, which almost perfectly reveals Y. After removing it, sign accuracy collapses to near chance.
- For wiki-elec/wiki-rfa (low bidir%): a weak genuine balance-theory signal exists (sign_acc ~63–64% after debidir correction, with 80–83% coverage).
- **Conclusion:** Balance theory contributes only weakly to sign predictability. The model's AUC of 0.88–0.94 is not explained by structural balance.

Results: `outputs/balance_theory/balance_theory_report.txt`, PNG plots `outputs/balance_theory/bt_<dataset>.png`

### 12.2 Edge Sign MI vs Directed Distance — `scripts/edge_sign_mi_vs_distance_v3.py`

**Question:** Does mutual information between an anchor edge's sign and a "context" edge's sign decay with directed shortest-path distance from source to source?

This answers: *how far away in the graph can context edges be while still being informative about the anchor sign?*

**Method (v3 — exact, streaming, no OOM):**
- **d=0** (same source node): exact formula, O(N)
- **d=1** (direct out-neighbours): exact formula, O(E)
- **d≥2**: per-unique-source-node BFS (not per edge); streaming 2×2 contingency table; hybrid CSR/matvec BFS frontier expansion; configurable D_MAX

**Key results (NMI = MI / H(Y)):**

| Dataset | d=0 NMI | d=1 NMI | d=2 NMI | d≥3 NMI |
|---------|--------:|--------:|--------:|---------|
| bitcoin-alpha | 0.045 | 0.003 | 0.00002 | ≈0 |
| bitcoin-otc | 0.077 | 0.015 | 0.000001 | ≈0 |
| epinions | **0.340** | 0.005 | 0.0002 | ≈0 |
| slashdot | **0.171** | **0.025** | 0.00008 | ≈0 |
| wiki-elec | 0.016 | 0.0002 | 0 | 0 |
| wiki-rfa | 0.018 | 0.0003 | 0 | 0 |

**Interpretation:**
- **d=0 dominates.** A node's own outgoing edges are the strongest predictors of each other — "ego-network behavioral consistency." Epinions and Slashdot have very strong ego-network effects (NMI=0.34 and 0.17).
- **d=1 is small but non-zero** in epinions and slashdot. Immediate out-neighbours carry a marginal signal.
- **d≥2 is essentially zero everywhere.** No recoverable structural sign signal exists beyond 2 hops. This rules out the hypothesis that our Transformer succeeds by learning long-range structural sign patterns.
- **Implication:** The model's useful signal comes from local context (same source, 1-hop neighbourhood) and the rich combination of many such signals across hundreds of walk occurrences — not from long-range graph structure.

Results: `outputs/mi_vs_dist/mi_vs_dist_report_v3.txt`, PNG plots `outputs/mi_vs_dist/mi_vs_dist_<dataset>.png`

---

## 13. Research Status and Open Questions

### What works and why

- **Random-walk Transformer** achieves state-of-the-art AUC on all 6 datasets
- **Post-hoc aggregation** adds 1–4 AUC points on top of walk-level predictions — consistently beneficial
- **E14 hardness-map reweighting** provides the current best results (E14_HARDNODE_L10 is the production tag)
- **Dynamic masking (E12)** prevents memorisation of target edge assignments
- **Node replacement (E10)** prevents node-ID memorisation

### What we understand

- Sign information is **local** (d=0, d=1 in MI analysis) — the model cannot leverage long-range structural patterns because they don't exist
- **Balance theory signal is mostly confounded by bidirectionality** — the model is not doing balance theory reasoning
- The model succeeds by **aggregating many weak local signals** across hundreds of walk occurrences per edge

### Open questions (ranked by likely impact)

1. **Walk length sweet spot**: The walk-length sweep (5 lengths × 6 datasets) is partially complete. Do longer walks help because they provide more context, or hurt because the edge of interest ends up far from context? Initial results suggest mw=40 is near-optimal for most datasets, but the relaunch for epinions/wiki/slashdot needs re-verification.

2. **Why does wiki-elec beat LightGBM with a simple mean?** This is surprising and suggests walk occurrences are unusually uniform in quality. Needs deeper investigation.

3. **Epinions LightGBM advantage (+1.3%):** Feature interactions matter here but not elsewhere. Hypothesis: epinions has more variation in per-walk context quality due to its larger graph and longer paths.

4. **Scaling with walks (5M vs 500K):** We know 5M > 500K for bitcoin-alpha and slashdot, but whether this gap is due to better coverage or simply more averaging is unknown.

5. **Per-source MI rise at d=3–7 for large graphs (epinions, bitcoin-otc):** The v3 exact computation shows NMI rising from d=2 back up at d=3–7 for some datasets. This is real (not sampling noise) — it may reflect structural effects like hub nodes that organise their neighbourhoods consistently. Not yet explained.

6. ~~**OWL (Occurrence-Weighted Loss):**~~ **CLOSED — dead code, dropped.** Coverage is now
   handled at the sampling source by the E15 k_cover sampler (~100% coverage), so up-weighting
   low-coverage edges in the loss is moot. Do not revive.

---

## 14. Quickstart Commands

### Train from scratch

```bash
# Activate environment
source .venv/bin/activate
cd /home/dsi/shilo_avital/yolo_lab/walk_to_paint

# GPU convention: set CUDA_VISIBLE_DEVICES before each command
CUDA_VISIBLE_DEVICES=0 python run.py dataset.name=bitcoin-alpha
CUDA_VISIBLE_DEVICES=1 python run.py dataset.name=bitcoin-otc
CUDA_VISIBLE_DEVICES=2 python run.py dataset.name=wiki-rfa
CUDA_VISIBLE_DEVICES=3 python run.py dataset.name=wiki-elec
CUDA_VISIBLE_DEVICES=0 python run.py dataset.name=epinions
CUDA_VISIBLE_DEVICES=1 python run.py dataset.name=slashdot090221
```

### Train with E14 hardness map

```bash
# Step 1: compute hardness map (runs tiny miner first)
python scripts/compute_hardness_map.py --dataset bitcoin-alpha --device 0

# Step 2: train main model with hardness reweighting
CUDA_VISIBLE_DEVICES=0 python run.py dataset.name=bitcoin-alpha \
    model.hardness_map_path=<path_to_hardness_map.pt> \
    model.hardness_lambda=1.0
```

### Run post-hoc aggregation

```bash
python run_posthoc.py \
    --exp-dir outputs/bitcoin-alpha/<run_name> \
    --checkpoint-choice best \
    --splits val,test \
    --artifacts aggregator \
    --agg-models lgbm \
    --device 0
```

### Run MI vs distance analysis (v3)

```bash
# All 6 datasets, D_MAX=12 (exact, no OOM)
nohup python scripts/edge_sign_mi_vs_distance_v3.py --d-max 12 \
    > outputs/mi_vs_dist/run_v3.log 2>&1 &

# Single dataset
python scripts/edge_sign_mi_vs_distance_v3.py --datasets bitcoin-alpha --d-max 10
```

### Run balance theory analysis

```bash
python scripts/balance_theory_paths.py --datasets all
```

### Optuna hyperparameter search

```bash
python optuna_run.py --dataset bitcoin-alpha --n-trials 100 --device 0
```

---

## 15. Repo Map

### Core pipeline

| File | Description |
|------|-------------|
| `run.py` | **Main training entry point.** Config → data → train → checkpoint. |
| `run_posthoc.py` | **Post-hoc aggregation pipeline.** Load checkpoint → aggregate walk predictions → edge-level AUC. |
| `config.yaml` | Base config with all defaults and documentation of every flag. |
| `configs/<dataset>.yaml` | Per-dataset overrides (hyperparams, data paths). Auto-merged when `dataset.name=X`. |

### Source code

| File | Description |
|------|-------------|
| `src/data/datasets.py` | Raw edge loaders for all 6 dataset formats + `postprocess_edges()`. |
| `src/data/prepare_data.py` | Full data pipeline: split → walks → tokenize → encode → cache. |
| `src/data/walk_sampler.py` | Random walk sampling (10+ strategies, multiprocessing, deterministic). |
| `src/data/tokenizer.py` | Vocab construction, token↔ID mapping, special tokens. |
| `src/data/dataset_cache.py` | Save/load `dataset_cache.pt` from disk. |
| `src/model/model.py` | `TransformerModel`: embedding + sinusoidal PE + encoder + head. |
| `src/model/lit_model.py` | PyTorch Lightning wrapper: masking, loss, class weights, hardness reweighting, OWL. |
| `src/training/train.py` | Trainer setup, callbacks (checkpoint, early stop, prediction saver). |
| `src/utils/config.py` | `load_config()`: hierarchical merge + auto-dataset-merge logic. |

### Analysis scripts

| File | Description |
|------|-------------|
| `scripts/balance_theory_paths.py` | Balance theory path-product analysis. Also contains `load_edges_canonical()` — **import this for all analysis scripts**. |
| `scripts/edge_sign_mi_vs_distance_v3.py` | MI(sign_anchor, sign_context) vs BFS distance. Exact, streaming, no OOM. |
| `scripts/validate_d2_mi.py` | Standalone d=2 MI approximation validator (exact vs approximate). |
| `scripts/run_transformer_incremental_experiments.py` | Systematic ablation orchestrator for E10/E12/E14 experiments. |
| `scripts/compute_hardness_map.py` | Train miner, evaluate on train set, output `hardness_map.pt`. |
| `scripts/launch_walk_length_sweep.sh` | Walk-length sweep launcher (6 datasets × 5 lengths). |
| `optuna_run.py` | Optuna hyperparameter search driver. |
| `plot_metrics.py` | Plot training curves from TensorBoard logs. |

### Output directories

| Directory | Contents |
|-----------|----------|
| `outputs/<dataset>/<run>/` | Per-run outputs: `checkpoints/`, `artifacts/`, `dataset_cache.pt` |
| `outputs/balance_theory/` | Balance theory report + plots |
| `outputs/mi_vs_dist/` | MI vs distance reports + plots (v2, v3) |
| `outputs/walk_length_sweep/` | Walk-length sweep results and logs |
| `outputs/transformer_incremental/` | E10/E12/E14 ablation experiment outputs |
| `logs/` | TensorBoard log directories |
| `checkpoints/` | Top-level checkpoint directory (legacy; prefer per-run `checkpoints/`) |

---

## 16. Config Cheat-Sheet

Most important knobs in `config.yaml` / `configs/<dataset>.yaml`:

| Key | Default | Tune? | Meaning |
|-----|---------|-------|---------|
| `dataset.num_walks` | 5M | Yes | More walks = better coverage but slower |
| `dataset.max_walk_length` | 80 | Yes | Longer walks = more context per walk |
| `dataset.walk_strategy` | `uniform` | No | **SOTA uses `k_cover` (+`walk_k_min=5`)** for ~100% coverage; `uniform` is the best older strategy but under-covers sparse graphs |
| `dataset.binary` | `true` | No | Always `true` — drops neutral edges |
| `dataset.multiedge_handling` | varies | No | `most_recent` for bitcoin; `keep` for others |
| `model.embedding_dim` | 64 | Yes | Tune with Optuna |
| `model.hidden_dim` | 128 | Yes | FFN intermediate size; tune with Optuna |
| `model.nhead` | 4 | Yes | Must divide `embedding_dim` evenly |
| `model.nlayers` | 3 | Yes | Tune with Optuna |
| `model.dropout` | 0.20 | Yes | Tune with Optuna |
| `model.dynamic_train_masking` | `true` | Rarely | Always enable — prevents target memorisation |
| `model.node_context_mode` | `replace` | Rarely | Always enable — prevents node-ID memorisation |
| `model.node_replace_prob` | 0.2 | No | 0.2 found optimal in ablations |
| `model.hardness_lambda` | 1.0 | Rarely | 1.0 found optimal; 0 disables |
| `model.hardness_map_path` | `null` | N/A | Path to `hardness_map.pt` from miner run |
| `training.lr` | varies | Yes | Most sensitive param; Optuna-tuned per dataset |
| `training.weight_decay` | varies | Yes | Tune with Optuna |
| `training.batch_size` | 1024 | No | 1024 near-optimal for all datasets |
| `training.gradient_clip_val` | 0.5 | Yes | Tune with Optuna; epinions needs 0.11 |
| `training.early_stopping_patience` | 15 | No | 15 is safe; reduce to 10 for fast experiments |
| `preprocess.use_cache` | `true` | No | Always `true` after first run |
| `reproducibility.seed` | 42 | No | Change for multi-seed evaluation |

---

## 17. Known Gotchas

### Data loading

1. **`dataset.name=X` is the correct override, not `--config configs/X.yaml`**. Passing a config file positionally to `run.py` is silently ignored. The `dataset.name` override triggers auto-merge of `configs/<X>.yaml` inside `load_config()`.

2. **Always use `load_edges_canonical(ds_name)` for analysis scripts.** Direct CSV parsing without `postprocess_edges()` produces different edge sets (e.g., wiki-elec has 107,080 raw edges vs 103,689 canonical). Any analysis inconsistency between scripts is likely caused by custom loaders.

3. **Wiki-elec and wiki-rfa edge counts are not what you expect.** Wiki-elec has 107,080 edges including neutral votes; after `binary=true`, it becomes 103,689. The subagent-reported numbers may differ depending on whether the raw or canonical count was used.

### Model

4. **`nhead` must divide `embedding_dim` evenly.** e.g., `nhead=8, embedding_dim=32` works; `nhead=8, embedding_dim=64` works; `nhead=3, embedding_dim=64` will crash.

5. **Class weights are computed from the `mask` split, not `train`.** This is correct — the mask split is what the model is actually trained to predict. Recomputing from `train` would give biased weights.

6. **`q_j = P(class 1) = P(trust)` universally across all datasets.** Class 1 is always the majority (trust/positive); class 0 is the minority (distrust/negative). This is verified and correct.

### Reproducibility

7. **Walk sampling is deterministic but sensitive to `num_walks` and `num_workers`.** Changing either value (even with the same seed) produces different walks. Cache invalidation is manual — delete `dataset_cache.pt` to force re-generation.

8. **Dynamic masking means val/test AUC can legitimately vary between epochs** even if the model hasn't changed much. This is expected — the training target set changes each epoch, slightly affecting the model's gradient and therefore its predictions. The variance is small (~0.001–0.003 AUC) but noticeable in tensorboard curves.

### Post-hoc

9. **Posthoc AUC is better than walk-level AUC by definition on the validation set** (the aggregator is fit to val). The interesting number is test AUC, which verifies that the optimized aggregator generalizes.

10. **`func_uniform` (simple mean) is a strong baseline.** Before tuning a fancy aggregator, check whether the uniform mean already achieves near-peak performance. For wiki-elec, it is actually the best aggregator.
