# Lead 1 Measurements — Technical Reference

Quick reference for reproducible analysis of GNN over-averaging hypothesis.

## Step 1: Degree-Stratified AUC Gap

**File:** `scripts/lead1_degree_gap.py`

**What it computes:**
- Target-node degree from baseline graph (canonical edge loader, node-ID-aligned)
- Per-edge test AUC for walk model and GINEConv
- Buckets by degree quartile, reports AUC gap (walk − GNN)

**Key functions:**
- `canonical_rank_map(ds)`: rank-remap from raw SNAP ids to dense baseline ids (see docstring for alignment verification: 100% edge match, 99.0% sign agreement)
- `baseline_degree_dict(ds)`: degree computed from `baselines/splits/{ds}.pt` edge set
- `walk_model_side(ds, deg, rank)`: extract walk-model per-edge predictions from `DATASET_CONFIGS[ds]` checkpoint
- `gnn_side(ds, deg)`: extract GINEConv per-edge predictions from `results_our_splits/{ds}/GINEConv/seed42/best_epoch_artifacts.pkl`

**Usage:** `python scripts/lead1_degree_gap.py --dataset bitcoin-alpha`

**Cross-dataset finding:** No consistent degree-gap shape; degree alone doesn't drive walk model's advantage uniformly.

---

## Step 2: Per-Layer Probe AUC

**File:** `scripts/lead1_layer_probe.py`

**What it computes:**
- For each layer k (k=0 random input, k=1 after layer 1, k=2 final), extract per-edge pair features `[z_u^(k), z_v^(k)]`
- Train 5-fold CV logistic regression on train/val split of test-set edge pairs
- Report cross-validated AUC ± std

**Key functions:**
- `probe_auc(emb, edge_index, y)`: StandardScaler + LogisticRegression with 5-fold CV

**Usage:** `python scripts/lead1_layer_probe.py --dataset bitcoin-alpha`

**Cross-dataset finding:** Layer 1 jumps from ~0.50 to ~0.83, layer 2 flat or +0.03; no progressive destruction of decodable signal.

---

## Step 3a: Aggregator Ablation

**File:** `baselines/GINEConv/run_with_our_splits.py` with `--aggr {add,mean,max}`

**Trained variants:**
- `results_our_splits/{ds}/GINEConv/seed42/score.csv` (add, layers=2, base)
- `results_our_splits/{ds}/GINEConv/seed42_aggrmean_layers2/score.csv` (mean, layers=2)
- `results_our_splits/{ds}/GINEConv/seed42_aggrmax_layers2/score.csv` (max, layers=2)

**Cross-dataset finding:**
- Sum wins on bitcoin-alpha, bitcoin-otc (small/sparse)
- Mean wins on epinions, slashdot090221, wiki-elec, wiki-rfa (large/dense)
- Max uniformly worst

---

## Step 3b: Cancellation-Ratio Metric

**File:** `scripts/lead1_cancellation.py`

**What it computes:**
- For each train-graph node i and layer k, compute:
  ```
  ratio_i = || sum_j msg_ji || / sum_j || msg_ji ||
  ```
- `msg_ji = ReLU(x_j + lin_k(edge_attr))` from trained GINEConv state_dict
- Correlate ratio against:
  - **in-degree:** total incoming edges
  - **sign-heterogeneity:** min(n_pos, n_neg) / (n_pos + n_neg) among in-neighbors

**Key functions:**
- `per_node_messages(x_in, edge_index, edge_attr, lin_weight, lin_bias)`: aggregate incoming messages per node
- `sign_heterogeneity(edge_index, edge_attr, n)`: compute minority-sign fraction per node

**Usage:** `python scripts/lead1_cancellation.py --dataset bitcoin-alpha`

**Cross-dataset finding:** Sign-heterogeneity correlates with cancellation at r≈−0.96–−0.97 (layer 0) across all 6 datasets (p=0). In-degree correlation is weak (−0.10 to −0.13). Magnitude modest: mean ratio 0.91–0.99 (1–9% norm loss).

---

## Step 4: Depth Sweep + MAD

**File:** `scripts/lead1_depth_mad.py`

**What it computes:**
- For each depth L ∈ {1,2,3,4,5}, read test AUC from `score.csv`
- Compute MAD (mean pairwise cosine dissimilarity) on final-layer embeddings (2000-node sample for efficiency)

**Key functions:**
- `mad(emb, sample_n=2000)`: sample n random nodes, compute mean (1 − cosine_similarity) pairwise

**Usage:** `python scripts/lead1_depth_mad.py --dataset bitcoin-alpha`

**Cross-dataset finding:**
- Best AUC at L=1 or L=2 on all datasets
- No dataset benefits from L>2
- MAD does not consistently track AUC degradation (oversmoothing not the primary mechanism)

---

## Node-ID Alignment (Critical for Steps 1, 3b)

**Problem:** Walk model's canonical edge loader (`load_edges_canonical`) uses raw external SNAP ids; `baselines/splits/{ds}.pt` uses dense sorted-rank ids.

**Solution:** 
```python
all_ids = sorted({n for u, v, s in canonical_edges for n in (u, v)})
rank = {ext_id: i for i, ext_id in enumerate(all_ids)}
```

**Verification (bitcoin-alpha):**
- 100% of canonical edges map to baseline edges under this remap
- 99.0% sign agreement (1.0% mismatch exactly matches documented reciprocal-edge-conflict rate from old_chats/baselines.json)

See `scripts/lead1_degree_gap.py::canonical_rank_map()` for implementation and docstring.

---

## Training Scripts

**Batch training all variants:**

GPU1 (aggregator ablation + depth sweep for 5 datasets):
```bash
CUDA_VISIBLE_DEVICES=1 nohup bash baselines/run_gineconv_sweep.sh > /tmp/gineconv_sweep.log 2>&1 &
```

GPU2 (base artifact refresh for 5 datasets, overwrites in-place):
```bash
CUDA_VISIBLE_DEVICES=2 nohup bash baselines/run_gineconv_base_refresh.sh > /tmp/gineconv_base_refresh.log 2>&1 &
```

See `baselines/run_gineconv_sweep.sh` and `baselines/run_gineconv_base_refresh.sh` for exact command sequences.

---

## Full Report

See `LEAD1_GNN_OVER_AVERAGING_REPORT.md` for complete findings, per-dataset breakdowns, and conclusions.
