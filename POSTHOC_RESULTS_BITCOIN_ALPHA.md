# Post-hoc Aggregator Results — bitcoin-alpha

> **⚠️ SUPERSEDED for absolute SOTA numbers (2026-06-29).** The `E14_HARDNODE_L10`
> backbone / epoch-24 / 5M-uniform numbers below are the old uniform-sampler results.
> Current full-coverage SOTA is **E15 k_cover k=5** (bitcoin-alpha 5M, full 0.9251 /
> LocalAttn4 0.9362) — see `outputs/walk_coverage_analysis/E15_SWEEP_RESULTS.md` and
> `CLAUDE.md`. Aggregator-comparison conclusions still hold; absolute AUCs are stale.

## Dataset & Experiment

| Property | Value |
|---|---|
| Dataset | bitcoin-alpha |
| Backbone model | Transformer (E14\_HARDNODE\_L10) |
| Checkpoint | epoch 24, val AUC = 0.9342 |
| Walk config | max\_walk\_length = 80, num\_walks = 5,000,000 |
| Backbone seed | 42 |

### Walk-occurrence statistics

| Split | Walk occurrences | % distrust walks (label=1) | Unique edges | % distrust edges (label=1) | Mean walk length | Max walk length |
|---|---|---|---|---|---|---|
| Val (used for optimization) | 1,799,304 | 5.00% | 2,419 | 6.32% | 55.6 | 80 |
| Test | 1,749,550 | 4.84% | 2,419 | 6.37% | 55.5 | 80 |

**Label encoding:** `tokenizer.edge_label2id = {'E_1': 0, 'E_-1': 1}` — artifact of insertion order in the vocabulary. Class **0 = trust** (positive sign, majority ~93.7%), class **1 = distrust** (negative sign, minority ~6.3%). All AUC scores use `roc_auc_score(y, P(class 1))`, i.e. class 1 (distrust) is the "positive class" in the sklearn sense. This is internally consistent throughout the pipeline — training loss weights `[0.103, 1.897]` correctly up-weight the minority class 1. AUC values are correct regardless of convention.  
Each edge appears on average ~744 times in the val set (1,799,304 / 2,419).

### Notation

Per-walk features: $q_j = P(\text{class 1}) = P(\text{distrust})$ = transformer's predicted distrust probability, $d^s_j$ = steps from walk start to edge, $d^e_j$ = steps from edge to walk end, $l_j = d^s_j + d^e_j + 1$ = walk length, $\rho_j = d^s_j / l_j \in [0,1)$ = relative position within walk.

The edge score is the weighted mean: $\hat{y}_e = \frac{\sum_j w_j q_j}{\sum_j w_j}$ where the sum is over all walk occurrences $j$ of edge $e$.

---

## Full Leaderboard (edge-level test AUC)

All AUCs are **edge-level** (walk probabilities aggregated to edge score, then AUC over edges).  
"Full-data" = Nelder-Mead optimized on the full 1.8M-occurrence val set.  
"Subsample" = Nelder-Mead optimized on 60K stratified (random) walk occurrences (for comparison).

| Rank | Model | Val AUC | Test AUC (full) | Test AUC (sub) | #params | Category |
|---|---|---|---|---|---|---|
| 1 | `lgbm_set_attention` | 0.9464 | **0.9134** | — | — | LightGBM + neural pool |
| 1 | `lgbm_attention` | 0.9445 | **0.9134** | — | — | LightGBM + neural pool |
| 3 | `func_len_logq` ★ | 0.9356 | **0.9131** | — | 2 | Len × log-conf |
| 3 | `func_logit_power` ★ | 0.9341 | **0.9131** | — | 1 | Log-odds conf |
| 5 | `lgbm` | 0.9429 | 0.9130 | — | — | LightGBM + mean pool |
| 6 | `lgbm_lse` | 0.9430 | 0.9129 | — | — | LightGBM + LSE pool |
| 7 | `func_log1mq_power` ★ | 0.9372 | 0.9129 | — | 1 | Trust log-conf |
| 8 | `func_conf_power` | 0.9373 | 0.9128 | 0.9132 | 1 | Confidence |
| 8 | `func_logq_power` ★ | 0.9354 | 0.9128 | — | 1 | Distrust log-conf |
| 10 | `func_len_conf` | 0.9375 | 0.9127 | 0.9135 | 2 | Len × Conf |
| 11 | `func_invq_exp` ★ | 0.9334 | 0.9116 | — | 1 | Trust conf (exp) |
| 11 | `func_conf_exp` | 0.9334 | 0.9116 | 0.9113 | 1 | Confidence |
| 13 | `func_pos_invq` ★ | 0.9344 | 0.9115 | — | 2 | Position × Trust conf |
| 14 | `func_pos_conf` | 0.9363 | 0.9114 | 0.9134 | 2 | Position × Conf |
| 15 | `func_len_cert` | 0.9320 | 0.9112 | 0.9096 | 2 | Len × Certainty |
| 16 | `func_len_invq` ★ | 0.9338 | 0.9110 | — | 2 | Len × Trust conf |
| 17 | `func_conf_cert` | 0.9316 | 0.9100 | 0.9099 | 1 | Certainty |
| 18 | `certainty` | 0.9316 | 0.9099 | — | 0 | Certainty (fixed) |
| 19 | `lgbm_self_attention` | 0.9488 | 0.9097 | — | — | LightGBM + neural pool |
| 20 | `func_invq_power` ★ | 0.9336 | 0.9107 | — | 1 | Trust conf (power) |
| 21 | `func_pos_expds` | 0.9315 | 0.9093 | 0.9086 | 1 | Position (raw ds) |
| 22 | `func_pos_exprel` | 0.9321 | 0.9092 | 0.9087 | 1 | Position (relative) |
| 22 | `func_pos_expde` | 0.9317 | 0.9092 | 0.9084 | 1 | Position (raw de) |
| 22 | `func_ds_de` | 0.9316 | 0.9092 | 0.9079 | 2 | Position (ds·de) |
| 25 | `func_pos_power` | 0.9316 | 0.9090 | 0.9091 | 1 | Position |
| 25 | `func_conf_logit` | 0.9334 | 0.9090 | 0.9104 | 1 | Confidence |
| 27 | `func_len_pos` | 0.9321 | 0.9089 | 0.9090 | 2 | Len × Position |
| 27 | `func_de_power` | 0.9314 | 0.9089 | 0.9088 | 1 | Position (de) |
| 29 | `func_uniform` | 0.9313 | 0.9088 | 0.9088 | 0 | Baseline |
| 29 | `func_log3` | 0.9316 | 0.9088 | 0.9086 | 3 | Log-linear |
| 29 | `func_log4` | 0.9316 | 0.9088 | 0.9086 | 4 | Log-linear |
| 29 | `func_log5` | 0.9315 | 0.9088 | 0.9086 | 5 | Log-linear |
| 29 | `func_log6` | 0.9316 | 0.9088 | 0.9086 | 6 | Log-linear |
| 29 | `func_len_power` | 0.9316 | 0.9088 | 0.9086 | 1 | Length |
| 29 | `func_dist_prod` | 0.9316 | 0.9088 | 0.9079 | 1 | Position (ds·de prod) |
| 29 | `func_dist_min` | 0.9313 | 0.9088 | 0.9084 | 1 | Position (min endpoint) |
| 29 | `func_centrality` | 0.9314 | 0.9088 | 0.9087 | 1 | Position (mid-peak) |
| 38 | `func_rp_sym` | 0.9314 | 0.9087 | 0.9085 | 1 | Position (symmetric) |
| 38 | `func_pos_middle` | 0.9312 | 0.9087 | 0.9087 | 1 | Position |
| 38 | `func_len_harmonic` | 0.9312 | 0.9087 | 0.9087 | 1 | Length |
| 38 | `func_len_exp` | 0.9312 | 0.9087 | 0.9087 | 1 | Length |
| 42 | `func_inv_len` | 0.9296 | 0.9079 | 0.9079 | 0 | Length (fixed) |
| 43 | `lgbm_stats` | 1.0000 | 0.9046 | — | — | Two-stage LightGBM (**overfit**) |
| 44 | `attention` | 0.9232 | 0.9008 | — | — | Neural MIL |
| 45 | `self_attention` | 0.9343 | 0.8963 | — | — | Neural MIL |
| 46 | `set_attention` | 0.9291 | 0.8903 | — | — | Neural MIL |
| 47 | `lgbm_max` | 0.9225 | 0.8889 | — | — | LightGBM + max pool |

★ = new models (Groups 8 & 9, added after label-encoding investigation).

---

## Method Descriptions

### LightGBM-based methods

All `lgbm*` variants use **three walk-level features**: $(d^s_j,\, l_j,\, q_j)$ where $q_j$ is the transformer's raw predicted probability. LightGBM is trained on the val set to produce a refined walk score $\hat{q}_j$.

| Model | Pooling over walks per edge | Notes |
|---|---|---|
| `lgbm` | Arithmetic mean $\bar{q}_e = \frac{1}{n_e}\sum_j \hat{q}_j$ | Simple baseline; competitive |
| `lgbm_lse` | Log-sum-exp: $\frac{1}{\beta}\ln\frac{1}{n_e}\sum_j e^{\beta \hat{q}_j}$, $\beta=0.035$ | Soft-max smoothing |
| `lgbm_attention` | Gated attention (Ilse & Tomczak 2018): $\sum_j a_j \hat{q}_j$ where $a_j \propto \tanh(W\hat{q}_j) \odot \sigma(U\hat{q}_j)$ | Learnable walk importance |
| `lgbm_self_attention` | 1-layer MHSA (d=32, h=4) over walk set, mean pool | High train AUC (0.9488) but overfit |
| `lgbm_set_attention` | PMA (Set Transformer, d=32, h=4): cross-attention with 1 seed vector | Most expressive; tied #1 |
| `lgbm_max` | $\max_j \hat{q}_j$ | Worst pooling — noisy single-walk estimate |
| `lgbm_stats` | Two-stage: LightGBM walk-level + 11 edge-level statistics → second LightGBM | Train AUC = 1.000 → **severe overfit** |

### Neural MIL methods (no LightGBM)

Raw features $(d^s_j, l_j, q_j)$ fed directly into neural network, trained end-to-end to predict edge label.

| Model | Architecture | Notes |
|---|---|---|
| `attention` | GatedAttention(3→32): $\tanh(Wx) \odot \sigma(Ux) \to$ softmax → weighted avg | Underperforms; small training set |
| `self_attention` | 1-layer MHSA(3→32, h=4), mean pool | Overfit |
| `set_attention` | SetTransformer PMA(3→32, cross-attn seed) | Worst among neural MIL |

**Why neural MIL underperforms:** Training on edge-level labels from ~2,400 edges is far too few for a neural network. The LightGBM pre-processing step (which leverages the much larger walk-level signal) is crucial.

### Certainty (no training, no parameters)

| Model | Score | Notes |
|---|---|---|
| `certainty` | $\phi_j = \lvert q_j - 0.5 \rvert$, weighted mean $\frac{\sum_j \phi_j q_j}{\sum_j \phi_j}$ | Down-weights uncertain walks; 0 parameters |

### Group 8: Trust-confidence variants (1−q substitution) ★

For bitcoin-alpha, class 0 = trust and $q_j = P(\text{distrust})$, so $1-q_j = P(\text{trust})$ is the graph-positive confidence. These replace $q$ with $1-q$ in the weight, testing whether trust-confidence is a better signal.

| Model | $w_j$ | Notes |
|---|---|---|
| `func_invq_power` | $(1-q_j)^b$ | Counterpart to `func_conf_power`; b>0 upweights trust-confident walks |
| `func_invq_exp` | $e^{b(1-q_j)}$ | Counterpart to `func_conf_exp` |
| `func_len_invq` | $l_j^{-a}(1-q_j)^b$ | Counterpart to `func_len_conf` |
| `func_pos_invq` | $e^{-a\rho_j}(1-q_j)^b$ | Counterpart to `func_pos_conf` |

**Finding:** 1-q variants are consistently **weaker** than their q-based counterparts (~0.0010–0.0021 lower test AUC). The optimizer found that downweighting high-distrust-probability walks (q^{-b} with b<0) is a better strategy than upweighting high-trust-probability walks. The representations are not equivalent in this data regime.

### Group 9: Log-probability weight variants ★

Use information content ($-\log q$ or $-\log(1-q)$) as the confidence signal instead of the raw probability.

| Model | $w_j$ | Notes |
|---|---|---|
| `func_logq_power` | $(-\log q_j)^b$ | Information content of distrust prediction; high weight for low-q (trust-confident) walks |
| `func_log1mq_power` | $(-\log(1-q_j))^b$ | Information content of trust prediction; high weight for high-q walks |
| `func_len_logq` | $l_j^{-a}(-\log q_j)^b$ | Combines length and log-distrust-info |
| `func_logit_power` | $\lvert\text{logit}(q_j)\rvert^b$ | Absolute log-odds; symmetric confidence signal |

**Finding:** Log-probability variants are the **best single/two-parameter functional models**, with `func_len_logq` and `func_logit_power` both reaching **0.9131** test AUC — outperforming `func_conf_power` (0.9128) and matching `lgbm` (0.9130). The key insight: $-\log q$ provides a nonlinear emphasis on extreme predictions (very low or very high $q$) that is more discriminative than a power transformation alone.

---

## Functional Aggregators — Detail Table

Optimization: Nelder-Mead, full val set (1,799,304 occurrences), pre-sorted + bincount, `maxiter=500`, `xatol=fatol=1e-4`.  
"Sub θ\*" = theta found on 60K random subsample.  
All `w_j > 0`; edge score = $\hat{y}_e = \sum_j w_j q_j / \sum_j w_j$.

### Group 0: No-parameter baselines

| Model | $w_j$ | θ\* | Val AUC | Test AUC |
|---|---|---|---|---|
| `func_uniform` | $1$ | — | 0.9313 | 0.9088 |
| `func_inv_len` | $1/l_j$ | — | 0.9296 | 0.9079 |

### Group 1: Length-only

| Model | $w_j$ | θ\* (full) | θ\* (sub) | Val AUC | Test AUC (full) | Test AUC (sub) |
|---|---|---|---|---|---|---|
| `func_len_power` | $l_j^{-a}$ | −0.494 | 0.495 | 0.9316 | 0.9088 | 0.9086 |
| `func_len_exp` | $e^{-a l_j}$ | 0.00987 | 0.0075 | 0.9312 | 0.9087 | 0.9087 |
| `func_len_harmonic` | $(1 + a l_j)^{-1}$ | 0.017 | 0.0157 | 0.9312 | 0.9087 | 0.9087 |
| `func_de_power` | $(d^e_j + 1)^{-a}$ | 0.400 | 0.492 | 0.9314 | 0.9089 | 0.9088 |

Note: `func_len_power` full-data θ\* = −0.494 (upweight *longer* walks) vs subsample θ\* = +0.495 (downweight longer). Both give nearly identical AUC — the AUC surface is nearly flat in length. The small θ\* values for `func_len_exp` and `func_len_harmonic` confirm length barely matters.

### Group 2: Position-only

| Model | $w_j$ | θ\* (full) | θ\* (sub) | Val AUC | Test AUC (full) | Test AUC (sub) |
|---|---|---|---|---|---|---|
| `func_pos_exprel` | $e^{-a \rho_j}$ | −6.125 | 1.063 | 0.9321 | 0.9092 | 0.9087 |
| `func_pos_power` | $(1 - \rho_j)^a$ | −0.301 | −0.398 | 0.9316 | 0.9090 | 0.9091 |
| `func_pos_expds` | $e^{-a d^s_j}$ | −0.028 | 0.0094 | 0.9315 | 0.9093 | 0.9086 |
| `func_pos_expde` | $e^{-a d^e_j}$ | 0.049 | −0.023 | 0.9317 | 0.9092 | 0.9084 |
| `func_pos_middle` | $e^{-a \lvert \rho_j - 0.5 \rvert}$ | 0.725 | 1.100 | 0.9312 | 0.9087 | 0.9087 |

Note: Large and sign-inconsistent θ\* across runs indicates a flat/multimodal AUC surface in position. Position alone provides only marginal signal (+0.0004 over uniform mean).

### Group 3: Confidence-only

| Model | $w_j$ | θ\* (full) | θ\* (sub) | Val AUC | Test AUC (full) | Test AUC (sub) |
|---|---|---|---|---|---|---|
| `func_conf_power` | $q_j^b$ | −1.181 | −0.823 | 0.9373 | **0.9128** | 0.9132 |
| `func_conf_exp` | $e^{b q_j}$ | −5.363 | −4.397 | 0.9334 | 0.9116 | 0.9113 |
| `func_conf_cert` | $\lvert q_j - 0.5 \rvert^b$ | 1.144 | 1.036 | 0.9316 | 0.9100 | 0.9099 |
| `func_conf_logit` | $\sigma(b(q_j - 0.5))$ | −107.1 | −9.225 | 0.9334 | 0.9090 | 0.9104 |

**Key finding:** All confidence-based weights use **negative** $b$ (except certainty weighting `func_conf_cert`), meaning the optimizer *down-weights high-confidence walks*. Interpretation: high-$q_j$ walks are noisy outliers; moderate-confidence walks carry the true signal.  
`func_conf_logit` converges to a step function ($b=-107$) — effectively thresholding $q_j$ at 0.5.

### Group 4: 2-parameter combinations

| Model | $w_j$ | θ\* (full) | θ\* (sub) | Val AUC | Test AUC (full) | Test AUC (sub) |
|---|---|---|---|---|---|---|
| `func_len_conf` | $l_j^{-a} \cdot q_j^b$ | [−0.845, −1.276] | [−0.772, −0.769] | 0.9375 | 0.9127 | **0.9135** |
| `func_len_pos` | $l_j^{-a} \cdot e^{-g \rho_j}$ | [0.358, −5.793] | [−1.418, 0.124] | 0.9321 | 0.9089 | 0.9090 |
| `func_pos_conf` | $e^{-a \rho_j} \cdot q_j^b$ | [2.312, −1.676] | [−2.479, −0.973] | 0.9363 | 0.9114 | 0.9134 |
| `func_ds_de` | $(d^s_j+1)^{-a} (d^e_j+1)^{-b}$ | [−0.261, 0.524] | [0.515, 0.514] | 0.9316 | 0.9092 | 0.9079 |
| `func_len_cert` | $l_j^{-a} \cdot \lvert q_j - 0.5 \rvert^b$ | [−0.668, 2.963] | [0.461, 1.048] | 0.9320 | 0.9112 | 0.9096 |

Note: `func_len_conf` and `func_pos_conf` both keep $b < 0$ for the confidence component, consistent with Group 3. Adding a second feature (length or position) gives marginal improvement over `func_conf_power` alone (+0.0000–0.0001 test AUC with full data).

### Group 5: Log-linear in features

$w_j = \exp\!\left(\sum_k b_k f_k(j)\right)$

| Model | Features $f_k$ | θ\* (full) | Val AUC | Test AUC (full) | Test AUC (sub) |
|---|---|---|---|---|---|
| `func_log3` | $q_j,\; \rho_j,\; \ln l_j$ | [0.0026, 0.0020, 0.498] | 0.9316 | 0.9088 | 0.9086 |
| `func_log4` | $1,\; q_j,\; \rho_j,\; \ln l_j$ | [0.0034, 0.0040, 0.0013, 0.486] | 0.9316 | 0.9088 | 0.9086 |
| `func_log5` | $q_j,\; \ln d^s_j,\; \ln d^e_j,\; \ln l_j,\; q_j \ln l_j$ | [−0.0005, 0.0044, 0.0008, 0.479, 0.0031] | 0.9315 | 0.9088 | 0.9086 |
| `func_log6` | $q_j,\; \rho_j,\; \ln l_j,\; q_j\rho_j,\; q_j \ln l_j,\; \rho_j \ln l_j$ | [0.0017, 0.0003, 0.484, 0.0014, 0.0001, 0.0048] | 0.9316 | 0.9088 | 0.9086 |

**Key finding:** All log-linear models converge to $b \approx 0$ for the $q$ and position components and $b \approx +0.49$ for $\ln l_j$. Since $e^{b \ln l} = l^b$ with $b \approx 0.49 \approx +0.5$, these models recover $w_j \approx l_j^{+0.5}$ — upweighting longer walks — identical to `func_len_power` with $a = -0.5$. The $q$ and position coefficients are essentially zero. This family fails to capture the confidence signal because the log-linear objective is dominated by the length gradient.

### Group 6: Geometric / dual-endpoint

| Model | $w_j$ | θ\* (full) | θ\* (sub) | Val AUC | Test AUC (full) | Test AUC (sub) |
|---|---|---|---|---|---|---|
| `func_dist_prod` | $\bigl((d^s_j+1)(d^e_j+1)\bigr)^{-a}$ | −0.494 | 0.515 | 0.9316 | 0.9088 | 0.9079 |
| `func_dist_min` | $\bigl(\min(d^s_j, d^e_j)+1\bigr)^{-a}$ | −0.178 | 0.494 | 0.9313 | 0.9088 | 0.9084 |
| `func_centrality` | $e^{-a(\rho_j - 0.5)^2}$ | −4.075 | 2.316 | 0.9314 | 0.9088 | 0.9087 |
| `func_rp_sym` | $\bigl(\rho_j(1-\rho_j)\bigr)^a$ | 0.119 | 0.509 | 0.9314 | 0.9087 | 0.9085 |

Sign-flips between full and subsample runs confirm a nearly flat AUC surface for position-only features.

### Group 8: Trust-confidence variants (1−q) ★

$w_j$ uses $(1-q_j) = P(\text{trust})$ instead of $q_j = P(\text{distrust})$. For bitcoin-alpha, class 0 = trust, so this tests whether trust-probability is a better weighting signal than distrust-probability.

| Model | $w_j$ | θ\* | Val AUC | Test AUC |
|---|---|---|---|---|
| `func_invq_power` | $(1-q_j)^b$ | b=3.488 | 0.9336 | 0.9107 |
| `func_invq_exp` | $e^{b(1-q_j)}$ | b=5.363 | 0.9334 | 0.9116 |
| `func_len_invq` | $l_j^{-a}(1-q_j)^b$ | [a=−0.815, b=3.281] | 0.9338 | 0.9110 |
| `func_pos_invq` | $e^{-a\rho_j}(1-q_j)^b$ | [a=−3.692, b=4.958] | 0.9344 | 0.9115 |

**Key finding:** All 1-q variants learn **positive** $b$ (upweight high-trust-confidence walks), the opposite sign convention from the $q$-based models. Despite this, they are 0.0001–0.0021 worse than their direct counterparts (`func_conf_power`, etc.). The model gains more information from the distrust signal than the trust signal, consistent with $q_j$ being the better discriminative axis.

### Group 9: Log-probability weight variants ★

Weights based on information content of the prediction.

| Model | $w_j$ | θ\* | Val AUC | Test AUC |
|---|---|---|---|---|
| `func_logq_power` | $(-\log q_j)^b$ | b=2.622 | 0.9354 | 0.9128 |
| `func_log1mq_power` | $(-\log(1-q_j))^b$ | b=−1.103 | 0.9372 | 0.9129 |
| `func_len_logq` | $l_j^{-a}(-\log q_j)^b$ | [a=−0.664, b=2.925] | 0.9356 | **0.9131** |
| `func_logit_power` | $\lvert\text{logit}(q_j)\rvert^b$ | b=3.009 | 0.9341 | **0.9131** |

**Key finding:** `func_len_logq` and `func_logit_power` both reach 0.9131 — the best among all functional models, outperforming `func_conf_power` (0.9128).  
`func_log1mq_power` with $b < 0$ effectively weights walks proportional to $(-\log(1-q))^{-1.1}$, which strongly downweights walks where the trust-probability is near zero (high distrust), recovering a similar effect to `func_conf_power`'s $q^{-1.18}$.  
The logit form $|\text{logit}(q)|^b$ is symmetric and penalizes uncertain walks from both sides — it is equivalent to certainty weighting but with a steeper, log-odds-derived schedule.

---

## Optimization Details

| Setting | Value |
|---|---|
| Optimizer | Nelder-Mead (scipy) |
| Data for optimization | Full val set (1,799,304 walk occurrences) |
| Speed trick | Pre-sorted arrays (argsort + np.unique once) + `np.bincount` (vs `np.add.at`) |
| Per-eval time | ~25 ms (vs ~270 ms naive, ~10× speedup) |
| `maxiter` | 500 |
| `xatol` / `fatol` | 1×10⁻⁴ |
| Max function evals | N×200 (scipy default; N = number of parameters) |
| Total runtime (28 models) | ~84 seconds |
| Final AUC evaluation | Full val/test set via `_wfn_edge_scores` |

---

## Key Observations

1. **Confidence weighting dominates.** `func_conf_power` ($w_j = q_j^b$, $b \approx -1.18$) achieves 0.9128 test AUC with a single parameter, within 0.0006 of the best model. All confidence-based models learn $b < 0$, meaning *down-weighting high-confidence walks improves edge-level AUC*.

2. **Log-probability weighting is the best functional form.** `func_len_logq` and `func_logit_power` both reach **0.9131** — matching `lgbm` (0.9130) with only 2 and 1 parameters respectively. The nonlinear emphasis on extreme predictions from $-\log q$ is more discriminative than the raw probability power transform.

3. **1-q (trust-confidence) variants are consistently weaker.** Despite bitcoin-alpha having class 0 = trust, using $1-q_j$ as the weight signal gives 0.0001–0.0021 lower test AUC than $q_j$-based counterparts. The distrust probability is the better discriminative axis.

4. **Length and position barely help.** Every length-only and position-only func model hovers within ±0.0001 of `func_uniform` (0.9088). The empirical weight surface diagnostics (from `diagnose_posthoc.py`) confirm that the primary structure in the data is along the confidence axis, not walk length or position.

5. **The log-linear family collapses to length weighting.** func\_log3 through func\_log6 all converge to $\approx l_j^{+0.5}$ weighting with near-zero coefficients for $q$ and position. This is because length has a clear gradient in this family but confidence does not interact with the log-linear parameterization as efficiently as `func_conf_power`.

6. **LightGBM adds ~0.003 AUC over func\_len\_logq.** The best lgbm variant (0.9134) vs best functional (0.9131) is a gap of 0.0003. LightGBM learns a nonlinear walk scoring function vs the transformer's raw $q_j$, providing a small but consistent benefit.

7. **lgbm\_stats severely overfits** (train = 1.000, test = 0.9046). The second-stage LightGBM memorizes the edge-level distribution from the 11 summary statistics, which are derived from the same val set used to evaluate it.

8. **Neural MIL underperforms** (0.8903–0.9008) despite higher model capacity. Root cause: only ~2,400 labeled edges are available for training, far too few for end-to-end deep learning.

9. **Subsample vs full-data:** For flat landscapes (length/position models), theta\* can flip sign between runs yet AUC is unchanged (e.g. `func_len_power`: full $a = -0.494$, sub $a = +0.495$, AUC difference ≤ 0.0002). For confidence models with steeper landscapes, theta\* is more stable and consistent between runs.
