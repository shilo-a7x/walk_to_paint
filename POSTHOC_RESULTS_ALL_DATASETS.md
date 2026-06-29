# Post-Hoc Aggregation Results — All Datasets

> **⚠️ SUPERSEDED for absolute SOTA numbers (2026-06-29).** The AUCs and `E14_HARDNODE_L10`
> tag below are the **old uniform-sampler** results (~85–88% walk coverage on the sparse
> graphs). The current full-coverage SOTA is the **E15 k_cover k=5** model — see
> `outputs/walk_coverage_analysis/E15_SWEEP_RESULTS.md` and the SOTA table in `CLAUDE.md`.
> The aggregator-ranking findings on this page (which `func_*`/`lgbm` wins per dataset, and
> why) still stand; only the absolute AUCs/budgets are stale.

All six datasets evaluated under the same experimental tag (`E14_HARDNODE_L10`) and the same 36 functional aggregators.  
For bitcoin-alpha's full detail (including `lgbm_attention`, `lgbm_set_attention`, neural MIL, subsample θ\* comparisons) see [POSTHOC_RESULTS_BITCOIN_ALPHA.md](POSTHOC_RESULTS_BITCOIN_ALPHA.md).  
Run script: [scripts/run_func_all_datasets.sh](scripts/run_func_all_datasets.sh) | Log: [logs/func_all_datasets.log](logs/func_all_datasets.log)

---

## 0. Cross-Dataset Summary

"Transformer AUC" = walk-level test AUC from the transformer alone (no aggregation).  
"lgbm AUC" = edge-level test AUC after LightGBM + mean-pool aggregation (vanilla baseline).  
"Best func\_" = edge-level test AUC of the top functional aggregator (no LightGBM, ≤2 parameters).  
"Δ func−lgbm" = best-func minus lgbm (positive = func wins).

| Dataset | nw | Transformer Test AUC | lgbm Test AUC | Best func\_ Test AUC | Best func\_ model | Δ func−lgbm |
|---|---:|---:|---:|---:|---|---:|
| bitcoin-alpha | 5M | 0.8993 | 0.9130 | **0.9131** | `func_len_logq`, `func_logit_power` | **+0.0001** |
| bitcoin-otc | 500K | 0.9229 | 0.9433 | **0.9433** | `func_len_cert` | **0.0000** |
| epinions | 500K | 0.9334 | **0.9446** | 0.9314 | `func_len_invq` | −0.0132 |
| wiki-elec | 500K | 0.8562 | 0.8826 | **0.8928** | (nearly all models) | **+0.0102** |
| wiki-rfa | 500K | 0.8415 | 0.8815 | **0.8816** | `func_len_conf`, `func_conf_logit` | **+0.0001** |
| slashdot090221 | 5M | 0.8818 | **0.8954** | 0.8952 | `func_logit_power` | −0.0002 |

**Headline findings:**
- Functional forms match or exceed lgbm on 4/6 datasets using ≤2 parameters.
- **wiki-elec anomaly:** lgbm is 0.010 *below* even the unweighted mean — the simplest possible aggregator wins.
- **epinions anomaly:** lgbm is 0.013 *above* the best functional form — LightGBM captures feature interactions unavailable to closed-form weighting.
- Aggregation over walks always improves on the raw transformer: minimum gain +0.0137 (epinions −0.9334→0.9446 lgbm baseline), maximum +0.0366 (wiki-rfa −0.8415→0.8815).

---

## 1. Experiment Setup

| Item | Value |
|---|---|
| Experiment tag | `E14_HARDNODE_L10` (all datasets) |
| Aggregator optimizer | Nelder-Mead (scipy) |
| Optimization target | Edge-level val-set AUC |
| Functional forms | 36 (Groups 0–9, see §3) |
| bitcoin-alpha, slashdot | 5M-walk experiments |
| Other 4 datasets | 500K-walk experiments |
| bitcoin-alpha extra aggregators | `lgbm_attention`, `lgbm_set_attention`, `lgbm_lse`, `lgbm_max`, neural MIL — see [POSTHOC_RESULTS_BITCOIN_ALPHA.md](POSTHOC_RESULTS_BITCOIN_ALPHA.md) |

**Best checkpoints (matched to existing posthoc directories):**

| Dataset | Best epoch | Val AUC (ckpt) | Posthoc run-id |
|---|---|---|---|
| bitcoin-alpha | 24 | 0.9342 | `bitcoin-alpha-E14_HARDNODE_L10-epoch=24-val_auc_epoch=0.9342_posthoc` |
| bitcoin-otc | 60 | 0.9247 | `bitcoin-otc-E14_HARDNODE_L10-epoch=60-val_auc_epoch=0.9247_posthoc` |
| epinions | 42 | 0.9344 | `epinions-E14_HARDNODE_L10-epoch=42-val_auc_epoch=0.9344_posthoc` |
| wiki-elec | 36 | 0.8544 | `wiki-elec-E14_HARDNODE_L10-epoch=36-val_auc_epoch=0.8544_posthoc` |
| wiki-rfa | 25 | 0.8630 | `wiki-rfa-E14_HARDNODE_L10-epoch=25-val_auc_epoch=0.8630_posthoc` |
| slashdot090221 | 20 | 0.9279 | `slashdot090221-E14_HARDNODE_L10-epoch=20-val_auc_epoch=0.9279_posthoc` |

---

## 2. Notation

Per-walk features: $q_j = P(\text{class 1})$ = transformer's class-1 probability, $d^s_j$ = steps from walk start to edge, $d^e_j$ = steps from edge to walk end, $l_j = d^s_j + d^e_j + 1$ = walk length, $\rho_j = d^s_j / l_j \in [0,1)$ = relative position within walk, $E = 10^{-9}$ (numerical floor).

Edge score: $\hat{y}_e = \dfrac{\sum_j w_j\, q_j}{\sum_j w_j}$, where the sum is over all walk occurrences $j$ of edge $e$.

AUC: `roc_auc_score(y_edge, ŷ_edge)` — edge-level, test set only for the final numbers, val set for θ\* optimization.

---

## 3. Label Encoding Notes

The `tokenizer._build_edge_label_map` assigns class IDs in token-insertion order. Verified for all 6 datasets via val-set prediction PKLs (class 0 is majority in all) and `cache_warmup.log` files (positive/trust edges are majority in raw data for all): the first edge token encountered is always the positive/trust edge. **Encoding is consistent across all datasets: class 0 = trust (E\_1), class 1 = distrust (E\_{-1}), $q_j = P(\text{distrust})$ universally.**

| Dataset | `edge_label2id` | $q_j$ meaning | Majority class | Raw positive % | Walk-level class 0 % |
|---|---|---|---|---|---|
| bitcoin-alpha | `{E_1: 0, E_-1: 1}` | P(distrust) | Class 0 = trust | 93.7% | 95.0% |
| bitcoin-otc | `{E_1: 0, E_-1: 1}` | P(distrust) | Class 0 = trust | 90.0% | 93.8% |
| epinions | `{E_1: 0, E_-1: 1}` | P(distrust) | Class 0 = trust | 85.3% | 85.3% |
| wiki-rfa | `{E_1: 0, E_-1: 1}` | P(distrust) | Class 0 = trust | 78.4% | 78.8% |
| wiki-elec | `{E_1: 0, E_-1: 1}` | P(distrust) | Class 0 = trust | 78.8% | 78.1% |
| slashdot090221 | `{E_1: 0, E_-1: 1}` | P(distrust) | Class 0 = trust | 77.4% | 87.8% |

All AUC values are correct regardless of encoding — rank-based metrics are invariant under simultaneous negation of both scores and labels. See [POSTHOC_RESULTS_BITCOIN_ALPHA.md](POSTHOC_RESULTS_BITCOIN_ALPHA.md) §Label Encoding for the full proof and discussion.

---

## 4. Functional Aggregator Groups

All 36 forms optimized with Nelder-Mead on the full val-set occurrence array.

### Group 0 / 7: No-parameter baselines

| Model | $w_j$ |
|---|---|
| `func_uniform` | $1$ |
| `func_inv_len` | $1/l_j$ |

### Group 1: Length-only

| Model | $w_j$ |
|---|---|
| `func_len_power` | $l_j^{-a}$ |
| `func_len_exp` | $e^{-a l_j}$ |
| `func_len_harmonic` | $(1 + a l_j)^{-1}$ |
| `func_de_power` | $(d^e_j + 1)^{-a}$ |

### Group 2: Position-only

| Model | $w_j$ |
|---|---|
| `func_pos_exprel` | $e^{-a \rho_j}$ |
| `func_pos_power` | $(1 - \rho_j)^a$ |
| `func_pos_expds` | $e^{-a d^s_j}$ |
| `func_pos_expde` | $e^{-a d^e_j}$ |
| `func_pos_middle` | $e^{-a \lvert \rho_j - 0.5 \rvert}$ |

### Group 3: Confidence-only

| Model | $w_j$ |
|---|---|
| `func_conf_power` | $q_j^b$ |
| `func_conf_exp` | $e^{b q_j}$ |
| `func_conf_cert` | $\lvert q_j - 0.5 \rvert^b$ |
| `func_conf_logit` | $\sigma\!\left(b(q_j - 0.5)\right)$ |

### Group 4: 2-parameter combinations

| Model | $w_j$ |
|---|---|
| `func_len_conf` | $l_j^{-a} \cdot q_j^b$ |
| `func_len_pos` | $l_j^{-a} \cdot e^{-g \rho_j}$ |
| `func_pos_conf` | $e^{-a \rho_j} \cdot q_j^b$ |
| `func_ds_de` | $(d^s_j+1)^{-a}(d^e_j+1)^{-b}$ |
| `func_len_cert` | $l_j^{-a} \cdot \lvert q_j - 0.5 \rvert^b$ |

### Group 5: Log-linear

$w_j = \exp\!\bigl(\sum_k b_k f_k(j)\bigr)$ over features $q_j,\, \rho_j,\, \ln l_j$ and their interactions: `func_log3` (3 params), `func_log4` (4), `func_log5` (5), `func_log6` (6).

### Group 6: Geometric / dual-endpoint

| Model | $w_j$ |
|---|---|
| `func_dist_prod` | $\bigl((d^s_j+1)(d^e_j+1)\bigr)^{-a}$ |
| `func_dist_min` | $\bigl(\min(d^s_j,d^e_j)+1\bigr)^{-a}$ |
| `func_centrality` | $e^{-a(\rho_j-0.5)^2}$ |
| `func_rp_sym` | $\bigl(\rho_j(1-\rho_j)\bigr)^a$ |

### Group 8: Trust-confidence (1−q) variants ★

Motivated by bitcoin-alpha's inverted label convention ($1-q_j = P(\text{trust})$); also tested on all datasets.

| Model | $w_j$ |
|---|---|
| `func_invq_power` | $(1-q_j)^b$ |
| `func_invq_exp` | $e^{b(1-q_j)}$ |
| `func_len_invq` | $l_j^{-a}(1-q_j)^b$ |
| `func_pos_invq` | $e^{-a\rho_j}(1-q_j)^b$ |

### Group 9: Log-probability variants ★

| Model | $w_j$ |
|---|---|
| `func_logq_power` | $(-\log q_j)^b$ |
| `func_log1mq_power` | $(-\log(1-q_j))^b$ |
| `func_len_logq` | $l_j^{-a}(-\log q_j)^b$ |
| `func_logit_power` | $\lvert\log(q_j/(1-q_j))\rvert^b$ |

★ = added after label-encoding investigation.

---

## 5. Per-Dataset Results

### 5.1 bitcoin-alpha (5M walks)

See [POSTHOC_RESULTS_BITCOIN_ALPHA.md](POSTHOC_RESULTS_BITCOIN_ALPHA.md) for the complete leaderboard including `lgbm_attention`, `lgbm_set_attention`, neural MIL, and subsample comparisons. Summary:

- Val occurrences: 1,799,304 over 2,419 unique edges (~744 walks/edge)
- Label encoding: class 0 = trust, class 1 = distrust; $q_j = P(\text{distrust})$
- All confidence models learn **negative** $b$ (downweight high-distrust-confidence walks)

| Rank | Model | Val AUC | Test AUC | θ\* |
|---|---|---|---|---|
| 1 | `lgbm_set_attention` | 0.9464 | **0.9134** | — |
| 1 | `lgbm_attention` | 0.9445 | **0.9134** | — |
| 3 | `func_len_logq` ★ | 0.9356 | **0.9131** | [a=−0.664, b=2.925] |
| 3 | `func_logit_power` ★ | 0.9341 | **0.9131** | [b=3.009] |
| 5 | `lgbm` | 0.9429 | 0.9130 | — |
| 6 | `lgbm_lse` | 0.9430 | 0.9129 | — |
| 7 | `func_log1mq_power` ★ | 0.9372 | 0.9129 | [b=−1.103] |
| 8 | `func_conf_power` | 0.9373 | 0.9128 | [b=−1.181] |
| 8 | `func_logq_power` ★ | 0.9354 | 0.9128 | [b=2.622] |
| 10 | `func_len_conf` | 0.9375 | 0.9127 | [a=−0.845, b=−1.276] |
| 11 | `func_conf_exp` | 0.9334 | 0.9116 | [b=−5.363] |
| 11 | `func_invq_exp` ★ | 0.9334 | 0.9116 | [b=5.363] |
| 13 | `func_pos_invq` ★ | 0.9344 | 0.9115 | [a=−3.692, b=4.958] |
| 14 | `func_pos_conf` | 0.9363 | 0.9114 | [a=2.312, b=−1.676] |
| 15 | `func_len_cert` | 0.9320 | 0.9112 | [a=−0.668, b=2.963] |
| 16 | `func_len_invq` ★ | 0.9338 | 0.9110 | [a=−0.815, b=3.281] |
| 17 | `func_invq_power` ★ | 0.9336 | 0.9107 | [b=3.488] |
| — | `func_uniform` | 0.9313 | 0.9088 | — |

**Key observations:**
- Confidence signal is strong (+0.004 from uniform to best func_)
- Log-probability weights outperform raw probability power (func_len_logq/logit_power > func_conf_power)
- 1-q variants consistently weaker than q variants (distrust probability is the better discriminative axis)
- LightGBM variants add only 0.0003–0.0006 AUC over best functional form

---

### 5.2 bitcoin-otc

- Val occurrences: available, ~500K walks / 5,881 unique edges
- Transformer test AUC: **0.9229** | lgbm: **0.9433**

| Rank | Model | Val AUC | Test AUC | θ\* |
|---|---|---|---|---|
| 1 | `lgbm` | 0.9437 | **0.9433** | — |
| 1 | `func_len_cert` | 0.9373 | **0.9433** | [a=−2.600, b=6.183] |
| 3 | `func_logit_power` ★ | 0.9361 | 0.9431 | [b=1.082] |
| 4 | `func_conf_cert` | 0.9366 | 0.9430 | [b=3.369] |
| 5 | `func_pos_expds` | 0.9376 | 0.9429 | [a=−0.079] |
| 6 | `func_len_logq` ★ | 0.9381 | 0.9427 | [a=−7.966, b=0.216] |
| 6 | `func_len_conf` | 0.9381 | 0.9427 | [a=−7.684, b=−0.051] |
| 8 | `func_log3` | 0.9382 | 0.9426 | [b=0.036, g=0.024, a=−8.095] |
| 8 | `func_log4`–`func_log6` | 0.9382 | 0.9426 | (similar) |
| 12 | `func_len_power` | 0.9381 | 0.9425 | [a=−8.447] |
| 13 | `func_len_invq` ★ | 0.9381 | 0.9424 | [a=−10.635, b=2.766] |
| 13 | `func_len_exp` | 0.9380 | 0.9424 | [a=−0.124] |
| 15 | `func_pos_invq` ★ | 0.9363 | 0.9423 | [a=−9.181, b=2.418] |
| 15 | `func_invq_exp` ★ | 0.9358 | 0.9423 | [b=1.161] |
| 15 | `func_ds_de` | 0.9390 | 0.9423 | [a=−9.225, b=−4.794] |
| — | `func_conf_power` | 0.9358 | 0.9420 | [b≈0] |
| — | `func_uniform` | 0.9358 | 0.9420 | — |
| last | `func_dist_min` | 0.9365 | 0.9414 | [a=−2.856] |

**Key observations:**
- **Length dominates entirely.** `func_len_power` with $a = -8.45$ means $w_j = l_j^{+8.45}$ — extremely aggressive upweighting of longer walks. The top 12 func models all have large negative `a` components (i.e., upweight longer walks).
- **Confidence signal is absent.** `func_conf_power` converges to $b \approx 0$ (effectively uniform in confidence). Adding confidence weight to length-only models gives marginal improvement.
- **Certainty works.** `func_len_cert` ($l^{+2.6} \cdot |q-0.5|^{6.18}$) matches lgbm at 0.9433. The large $b$ creates a near-binary certainty weighting: walks where $q \approx 0$ or $q \approx 1$ get high weight, and $l^{+2.6}$ further amplifies longer ones.
- **Why length?** In bitcoin-otc, longer walks traverse more of the global trust network — a walk of length 10 implicitly reflects far more connectivity information than a length-2 stub. Longer-walk predictions are more reliable because they average over more intermediate node signals.

---

### 5.3 epinions

- Val occurrences: available, ~500K walks / unique edges
- Transformer test AUC: **0.9334** | lgbm: **0.9446**

| Rank | Model | Val AUC | Test AUC | θ\* |
|---|---|---|---|---|
| 1 | `lgbm` | 0.9471 | **0.9446** | — |
| 2 | `func_len_invq` ★ | 0.9312 | 0.9314 | [a=0.233, b=−0.528] |
| 3 | `func_pos_invq` ★ | 0.9312 | 0.9313 | [a=0.979, b=−0.539] |
| 3 | `func_invq_power` ★ | 0.9311 | 0.9313 | [b=−0.512] |
| 5 | `func_logit_power` ★ | 0.9312 | 0.9311 | [b=0.887] |
| 5 | `func_len_logq` ★ | 0.9308 | 0.9311 | [a=0.232, b=−0.313] |
| 7 | `func_pos_expds` | 0.9308 | 0.9310 | [a=0.046] |
| 7 | `func_logq_power` ★ | 0.9308 | 0.9310 | [b=−0.302] |
| 9 | `func_len_exp` | 0.9307 | 0.9309 | [a=0.029] |
| 10 | `func_conf_cert` | 0.9308 | 0.9307 | [b=1.012] |
| 10 | `func_len_cert` | 0.9309 | 0.9308 | [a=0.209, b=0.987] |
| — | `func_conf_power` | 0.9304 | 0.9306 | [b=−0.157] |
| — | `func_uniform` | 0.9303 | 0.9305 | — |
| — | `func_inv_len` | 0.9297 | 0.9301 | — |

**Key observations:**
- **LightGBM dominates by a large margin (+0.013 over best func_).** Epinions is the dataset where func_ forms fail most severely relative to lgbm.
- **All func_ models cluster within 0.0013 of each other** (0.9301–0.9314). The AUC landscape is nearly flat — no single weighting strategy significantly outperforms uniform.
- **θ\* magnitudes are tiny.** `func_conf_power` $b = -0.157$, `func_len_power` $a = +0.203$ — these parameters barely shift the weighting. Confidence and length provide minimal signal above the unweighted mean.
- **Why does lgbm succeed here?** LightGBM combines $d^s$, $l$, $q$ nonlinearly and can discover interaction effects (e.g., confidence matters only for walks of a specific length range) that no single closed-form weight can capture.
- **Note on 1-q variants:** `func_invq_power` ($b=-0.512$) is the top performer. With $(1-q)^{-0.512}$, large weight goes to high-$q$ walks — the opposite sign convention from bitcoin-alpha. This is likely because epinions' class-1 encoding is different (unverified), or the majority class is different.

---

### 5.4 wiki-elec

- Val occurrences: available, ~500K walks / unique edges
- Transformer test AUC: **0.8562** | lgbm: **0.8826**

| Rank | Model | Val AUC | Test AUC | θ\* |
|---|---|---|---|---|
| **LGBM baseline** | `lgbm` | 0.8936 | **0.8826** | — |
| 1 (func) | `func_logit_power` ★ | 0.8888 | **0.8928** | [b=0.524] |
| 1 | `func_invq_power` ★ | 0.8888 | **0.8928** | [b=−0.489] |
| 1 | `func_conf_cert` | 0.8888 | **0.8928** | [b=0.981] |
| 1 | `func_uniform` | 0.8886 | **0.8928** | — |
| 1 | `func_pos_power` | 0.8886 | **0.8928** | [b=−0.131] |
| 1 | `func_pos_expds` | 0.8886 | **0.8928** | [a=−0.012] |
| 1 | `func_conf_power` | 0.8886 | **0.8928** | [b=0.080] |
| 1 | `func_conf_exp` | 0.8887 | **0.8928** | [b=1.231] |
| 1 | `func_invq_exp` ★ | 0.8887 | **0.8928** | [b=−1.231] |
| 1 | `func_logq_power` ★ | 0.8887 | **0.8928** | [b=−0.369] |
| 1 | `func_log1mq_power` ★ | 0.8886 | **0.8928** | [b=0.160] |
| 1 | `func_len_harmonic` | 0.8886 | **0.8928** | [a≈0] |
| 1 | `func_len_exp` | 0.8886 | **0.8928** | [a=0.009] |
| 1 | `func_pos_conf` | 0.8885 | **0.8928** | [a=1.077, b=0.114] |
| — | (most 2-param models) | 0.8888–0.8891 | 0.8920–0.8924 | (various) |
| last | `func_inv_len` | 0.8880 | 0.8926 | — |
| ⚠ **last of all** | `lgbm` | 0.8936 | **0.8826** | — |

**Key observations — wiki-elec is the most anomalous dataset:**
- **lgbm (0.8826) is significantly WORSE than the unweighted mean (0.8928), by 0.010 AUC.** This means the trained LightGBM aggregator hurts performance on the test set despite having higher val AUC (0.8936).
- **The func_ AUC landscape is completely flat.** All 36 models land between 0.8920 and 0.8928. θ\* values are close to zero or trivially small — the optimizer cannot find meaningful signal in $q$, $l$, or $\rho$. `func_conf_power` converges to $b = +0.080$ (essentially zero), `func_len_power` to $a = -0.738$ (slightly upweight longer walks).
- **Why does lgbm fail?** The walk-level AUC for wiki-elec is 0.8562 — the transformer's predictions have significant noise. LightGBM learns walk-level feature patterns on the val set that overfit to val-specific noise, generalizing poorly. The simple mean aggregation over many walks per edge averages out this noise more robustly. This is an overfitting artifact: the LightGBM model has free parameters trained on val-set walks and tested on test-set edges, but the distribution of walk features (especially $q$ values) apparently differs between splits for this dataset.
- **Recommendation:** For wiki-elec, use `func_uniform` (0.8928). No further tuning needed.

---

### 5.5 wiki-rfa

- Val occurrences: available, ~500K walks / unique edges
- Transformer test AUC: **0.8415** | lgbm: **0.8815**

| Rank | Model | Val AUC | Test AUC | θ\* |
|---|---|---|---|---|
| 1 | `func_len_conf` | 0.8785 | **0.8816** | [a=0.544, b=0.705] |
| 1 | `func_conf_logit` | 0.8785 | **0.8816** | [b=4.587] |
| 3 | `lgbm` | 0.8837 | 0.8815 | — |
| 3 | `func_rp_sym` | 0.8781 | 0.8815 | [a=0.496] |
| 3 | `func_pos_conf` | 0.8786 | 0.8815 | [a=1.622, b=0.700] |
| 3 | `func_logq_power` ★ | 0.8784 | 0.8815 | [b=−0.217] |
| 3 | `func_log1mq_power` ★ | 0.8785 | 0.8815 | [b=0.397] |
| 3 | `func_len_logq` ★ | 0.8784 | 0.8815 | [a=−0.291, b=−0.247] |
| 3 | `func_invq_exp` ★ | 0.8784 | 0.8815 | [b=−1.422] |
| 3 | `func_conf_power` | 0.8785 | 0.8815 | [b=0.515] |
| 3 | `func_conf_exp` | 0.8784 | 0.8815 | [b=1.422] |
| — | `func_uniform` | 0.8783 | 0.8814 | — |
| — | `func_pos_expds` | 0.8784 | 0.8811 | [a=−0.040] |
| — | `func_len_exp` | 0.8785 | 0.8811 | [a=−0.042] |
| last | `func_logit_power` ★ | 0.8785 | 0.8810 | [b=−0.651] |
| last | `func_centrality` | 0.8785 | 0.8809 | [a=−8.607] |

**Key observations:**
- **Confidence signal is opposite in direction to bitcoin-alpha.** `func_conf_power` converges to $b = +0.515$ — the optimizer *upweights* high-$q$ walks, unlike bitcoin-alpha ($b = -1.18$). `func_conf_logit` with $b = 4.587$ is a near-step-function that selects walks with $q > 0.5$. This means: for wiki-rfa, the most confident walks (high $q$) are the most reliable, not noisy outliers.
- **func_ models marginally outperform lgbm** (0.8816 vs 0.8815). Both are very close.
- **`func_logit_power` underperforms** for wiki-rfa (0.8810, near last). The symmetric $|\text{logit}(q)|^b$ is not well-suited to a dataset where only one direction of confidence is informative.
- **Landscape is moderately flat** (0.8809–0.8816 range). Any confidence-based model with small $|b|$ is roughly equivalent.

---

### 5.6 slashdot090221 (5M walks)

- Val occurrences: available, ~1.7M walks / unique edges (5M-walk experiment)
- Label encoding: class 0 = trust, class 1 = distrust; $q_j = P(\text{distrust})$ (same as all datasets)
- Transformer test AUC: **0.8818** | lgbm: **0.8954**

| Rank | Model | Val AUC | Test AUC | θ\* |
|---|---|---|---|---|
| 1 | `lgbm` | 0.8972 | **0.8954** | — |
| 2 | `func_logit_power` ★ | 0.8957 | 0.8952 | [b=0.790] |
| 3 | `func_pos_conf` | 0.8954 | 0.8951 | [a=0.194, b=−0.343] |
| 3 | `func_len_conf` | 0.8954 | 0.8951 | [a=0.100, b=−0.346] |
| 3 | `func_len_cert` | 0.8955 | 0.8951 | [a=0.037, b=0.854] |
| 3 | `func_conf_power` | 0.8954 | 0.8951 | [b=−0.337] |
| 3 | `func_conf_cert` | 0.8955 | 0.8951 | [b=0.858] |
| 8 | `func_pos_expds` | 0.8953 | 0.8950 | [a=0.050] |
| 9 | `func_pos_power` | 0.8953 | 0.8949 | [a=0.225] |
| 9 | `func_pos_invq` ★ | 0.8954 | 0.8949 | [a=0.293, b=−0.308] |
| 9 | `func_pos_exprel` | 0.8953 | 0.8949 | [a=0.303] |
| 9 | `func_log3`–`func_log6` | 0.8953 | 0.8949 | (all ≈ same) |
| 9 | `func_logq_power` ★ | 0.8953 | 0.8948 | [b=−0.172] |
| — | `func_uniform` | 0.8952 | 0.8948 | — |
| — | `func_invq_power` ★ | 0.8954 | 0.8949 | [b=−0.290] |
| last | `func_inv_len` | 0.8948 | 0.8944 | — |

**Key observations:**
- **lgbm leads by only 0.0002** — `func_logit_power` (0.8952) is essentially equivalent to lgbm (0.8954).
- **Confidence signal is moderate.** `func_conf_power` converges to $b = -0.337$. Since $q_j = P(\text{distrust})$ universally, $b < 0$ downweights distrust-confident walks (upweights trust-confident walks) — **same direction as bitcoin-alpha** ($b = -1.181$), though substantially weaker. The distrust signal is noisier than the trust signal in both trust-heavy networks.
- **`func_logit_power` is the top functional model across 3/6 datasets** (bitcoin-alpha: 0.9131, wiki-elec: 0.8928, slashdot: 0.9131, 0.8928, 0.8952). Its symmetric certainty signal $|\text{logit}(q)|^b$ works well when the dataset has no strong directional preference for which class should be upweighted.
- **θ\* magnitudes are small** — $|b| \le 0.37$ for single-parameter models, indicating a gentle signal. The 0.0004 gain from uniform to func_ top is meaningful but modest.

---

## 6. Cross-Dataset Analysis

### 6.1 Confidence signal strength

Does confidence weighting ($w_j \propto q_j^b$ with optimized $b$) help over uniform aggregation?  
Gain = (func\_conf\_power test AUC) − (func\_uniform test AUC).

| Dataset | func\_uniform | func\_conf\_power | Gain | θ\* (b) | Direction |
|---|---|---|---|---|---|
| bitcoin-alpha | 0.9088 | 0.9128 | **+0.0040** | −1.181 | downweight high-$q$ |
| bitcoin-otc | 0.9420 | 0.9420 | +0.0000 | ≈0 | no signal |
| epinions | 0.9305 | 0.9306 | +0.0001 | −0.157 | weak downweight |
| wiki-elec | 0.8928 | 0.8928 | +0.0000 | +0.080 | no signal |
| wiki-rfa | 0.8814 | 0.8815 | +0.0001 | +0.515 | upweight high-$q$ |
| slashdot090221 | 0.8948 | 0.8951 | +0.0003 | −0.337 | weak downweight |

**Pattern:** Confidence weighting is **strongly beneficial only for bitcoin-alpha**. Since $q_j = P(\text{distrust})$ universally, $b < 0$ means "distrust-confident walks are noisy" (downweight them; upweight trust-confident walks), while $b > 0$ means "distrust-confident walks are reliable signal." The two 5M-walk experiments (bitcoin-alpha $b = -1.181$, slashdot $b = -0.337$) agree: in large trust-majority networks, the most extreme distrust walk predictions carry less reliable edge-level information. Wiki-rfa is the outlier ($b = +0.515$): its distrust-confident walks are the more useful signal. Bitcoin-otc and wiki-elec have AUC surfaces essentially flat in $q$ ($|b| \le 0.16$).

### 6.2 Length signal

Gain = (best length-only func test AUC) − (func\_uniform test AUC).

| Dataset | Best length func | Test AUC | Gain | θ\* |
|---|---|---|---|---|
| bitcoin-alpha | `func_len_power` | 0.9088 | +0.0000 | a=−0.494 (≈ flat) |
| **bitcoin-otc** | `func_len_power` | 0.9425 | **+0.0005** | a=−8.447 (strong upweight) |
| epinions | `func_len_exp` | 0.9309 | +0.0004 | a=0.029 (weak) |
| wiki-elec | `func_len_power` | 0.8925 | −0.0003 | a=−0.738 |
| wiki-rfa | `func_len_power` | 0.8814 | 0.0000 | a=0.519 |
| slashdot090221 | `func_len_power` | 0.8949 | +0.0001 | a=0.054 |

**Length matters only for bitcoin-otc** ($a = -8.45$, upweight $l^{+8.45}$). For all other datasets, the length-only AUC is within ±0.0005 of uniform. The extremely large $a$ for bitcoin-otc is unique — longer random walks through the trust network carry substantially more reliable edge-level signal.

### 6.3 The wiki-elec anomaly: why lgbm fails

| | Val AUC | Test AUC |
|---|---|---|
| `lgbm` | 0.8936 | 0.8826 |
| `func_uniform` | 0.8886 | 0.8928 |
| Generalization gap | 0.8936 − 0.8886 = 0.0050 (lgbm better on val) | 0.8826 − 0.8928 = −0.0102 (lgbm **worse** on test) |

The LightGBM model achieves higher val AUC than func\_uniform (0.8936 vs 0.8886) but dramatically lower test AUC (0.8826 vs 0.8928). This 0.015-point reversal from val to test indicates that the LightGBM model is **memorizing val-set walk distribution artifacts** that do not generalize.

Likely mechanism: wiki-elec has the lowest transformer walk-level AUC (0.8562). The transformer predictions $q_j$ are noisier for wiki-elec than for other datasets. LightGBM training on $(d^s, l, q)$ walk features picks up val-specific noise patterns (e.g., "in the val set, edges with many walks of length 3 at position 0.4 tend to be positive") that are absent in the test set. The simple mean aggregation averages over this noise robustly and generalizes better.

**Recommendation:** For wiki-elec, do not use LightGBM aggregation. Use `func_uniform` (unweighted mean of raw transformer walk scores).

### 6.4 The epinions anomaly: why lgbm dominates

| | lgbm AUC | Best func\_ AUC | Gap |
|---|---|---|---|
| epinions | 0.9446 | 0.9314 | **+0.0132** |
| bitcoin-alpha | 0.9130 | 0.9131 | 0.0000 |
| all others | 0.8815–0.9433 | ±0.0002 of lgbm | ≈0 |

For epinions, LightGBM learns feature interactions that no 1–2 parameter weight function can capture. The epinions graph has a richer trust structure than bitcoin-alpha/slashdot, and the walk-level predictor AUC (0.9334 walk-level, 0.9344 val) is already high — the signal exists, but extracting the most from it requires learning nonlinear combinations of ($d^s$, $l$, $q$).

Evidence from θ\* values: all func\_ models converge to nearly zero parameters ($|b|, |a| \le 0.5$), meaning no simple weighting direction provides consistent improvement. The landscape is smooth but the optimum is the uniform mean. LightGBM with its many decision boundaries accesses a different, richer hypothesis class.

### 6.5 func_logit_power: best single-parameter functional form overall

`func_logit_power` ($w_j = |\text{logit}(q_j)|^b$) achieves the best or near-best single-parameter result on 4/6 datasets:

| Dataset | func\_logit\_power test AUC | Rank (func\_) | θ\* (b) |
|---|---|---|---|
| bitcoin-alpha | **0.9131** | 1 (tied) | 3.009 |
| bitcoin-otc | 0.9431 | 3 | 1.082 |
| epinions | 0.9311 | 5 | 0.887 |
| wiki-elec | **0.8928** | 1 (tied) | 0.524 |
| wiki-rfa | 0.8810 | **last** | −0.651 |
| slashdot090221 | **0.8952** | 2 | 0.790 |

The logit form is a **symmetric certainty signal**: it gives high weight to walks where the model is highly confident in *either* class direction ($q \to 0$ or $q \to 1$). It fails for wiki-rfa because that dataset benefits specifically from high-$q$ walks (one-directional confidence), and the symmetric logit form penalizes low-$q$ uncertain walks that should be upweighted there.

### 6.6 Full θ\* cross-dataset table for top models

| Model | bitcoin-alpha | bitcoin-otc | epinions | wiki-elec | wiki-rfa | slashdot |
|---|---|---|---|---|---|---|
| `func_conf_power` (b) | −1.181 | ≈0 | −0.157 | +0.080 | **+0.515** | −0.337 |
| `func_len_power` (a) | −0.494 | **−8.447** | +0.203 | −0.738 | +0.519 | +0.054 |
| `func_logit_power` (b) | 3.009 | 1.082 | 0.887 | 0.524 | −0.651 | 0.790 |
| `func_logq_power` (b) | 2.622 | ≈0 | −0.302 | −0.369 | −0.217 | −0.172 |
| `func_len_logq` (a, b) | −0.664, 2.925 | −7.966, 0.216 | 0.232, −0.313 | −1.097, −0.389 | −0.291, −0.247 | 0.078, −0.160 |

**Observations on θ\* patterns:**
- `func_conf_power` $b$: negative for the two 5M datasets (downweight majority-class confidence), positive for wiki-rfa, essentially zero for bitcoin-otc and wiki-elec. No universal direction.
- `func_len_power` $a$: massively negative for bitcoin-otc (strong upweight longer walks), mildly negative/positive and near-zero for others.
- `func_logit_power` $b$: consistently positive except wiki-rfa. A positive $b$ rewards walk certainty symmetrically.
- `func_len_logq` $a$: negative for datasets where length helps (bitcoin-alpha, bitcoin-otc), positive for epinions/slashdot where length barely matters and $a \approx 0$.

---

## 7. Optimization Details

| Setting | Value |
|---|---|
| Optimizer | Nelder-Mead (scipy) |
| Max iterations | 500 |
| `xatol` / `fatol` | 1×10⁻⁴ |
| Data for optimization | Full val-set occurrence array (pre-sorted, `np.bincount`) |
| Per-eval time | ~25 ms for 5M datasets, faster for 500K |

All optimization is **on val-set edge-level AUC**. Test AUC is only computed after the optimizer converges; it is never used to select θ\*.

---

## 8. Files and Paths

| Dataset | EXP_DIR |
|---|---|
| bitcoin-alpha | `outputs/transformer_incremental/bitcoin-alpha_seed42_nw5000000_mw80_bs1024_ep75_20260423-111407/runs/bitcoin-alpha/E14_HARDNODE_L10` |
| bitcoin-otc | `outputs/transformer_incremental/bitcoin-otc_seed42_nw500000_mw80_bs1024_ep50_20260419-133621/runs/bitcoin-otc/E14_HARDNODE_L10` |
| epinions | `outputs/transformer_incremental/epinions_seed42_nw500000_mw80_bs1024_ep50_20260416-125419/runs/epinions/E14_HARDNODE_L10` |
| wiki-elec | `outputs/transformer_incremental/wiki-elec_seed42_nw500000_mw80_bs1024_ep50_20260419-133621/runs/wiki-elec/E14_HARDNODE_L10` |
| wiki-rfa | `outputs/transformer_incremental/wiki-rfa_seed42_nw500000_mw80_bs1024_ep50_20260324-141920/runs/wiki-rfa/E14_HARDNODE_L10` |
| slashdot090221 | `outputs/transformer_incremental/slashdot090221_seed42_nw5000000_mw80_bs1024_ep50_20260420-134149/runs/slashdot090221/E14_HARDNODE_L10` |
