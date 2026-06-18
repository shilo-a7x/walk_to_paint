# Edge Sign MI vs Directed Distance: Full Analysis

## 1. What We Are Measuring

For a **signed directed graph** with edges labeled +1 (trust) or -1 (distrust):

> *Does knowing the sign of one edge tell us anything about the sign of a structurally nearby edge?*

We measure **Mutual Information**:

$$MI(Y;\, S) = \sum_{y,s \in \{-1,+1\}} P(y,s) \log_2 \frac{P(y,s)}{P(y)\,P(s)}$$

- **Y** = sign of **anchor edge** `u→v`
- **S** = sign of **context edge** `a→b`
- **distance d** = shortest directed BFS path from source `u` to source `a`

We report **NMI = MI / H(Y)** for comparability across datasets.

---

## 2. Motivation

Our transformer trains by predicting masked edge signs from surrounding walk context.
If MI(Y; S) drops to zero beyond d=1:
- Long walks carry no additional informational value — only variance reduction
- The model cannot be learning long-range structural sign patterns
- Post-hoc aggregation gains come purely from averaging more walk occurrences

**Finding: MI drops to near-zero at d=2 for all six datasets.**

---

## 3. Exactly What Pairs Are Counted

### d=0 — ordered pairs of out-edges from the same source node

```
n_pairs(d=0) = sum_u  k_out(u) * (k_out(u) - 1)
```

For each node `u` with >=2 out-edges: every ordered pair `(u->v_i, u->v_j)` with `i!=j`.
Equals `sum(k_out^2)`. Dominated entirely by out-degree hubs.

**Bidirectionality has zero effect on d=0.** The reciprocal `v->u` is an in-edge to u
and does not enter `k_out(u)`.

### d=1 — anchor `u->v`, context = out-edges of `v` (minus reciprocal `v->u`)

```
n_pairs(d=1) = sum_v  k_in(v) * k_out(v)  -  |reciprocal edges|
```

Depends on the **correlation between in-degree and out-degree**:
- corr≈1 (bitcoin, 83% bidir): `sum(k_in*k_out) ≈ sum(k_out^2)` → d=1 ≈ d=0
- corr≈0.3 (wiki, 6% bidir): high-out-deg voters have near-zero in-deg → d=0 >> d=1

### d>=2 — per-source BFS, context = all out-edges at that layer

```
n_pairs(d>=2) = sum_u  k_out(u) * sum_{w in L_d(u)} k_out(w)
```

`L_d(u)` = nodes first reached at exactly hop d (BFS visited set excludes closer nodes).

**Why d=2 >> d=1:** Layer1 ≈ k_out ≈ 6 nodes. Layer2 ≈ 100–500 nodes — a 40–200x larger
frontier, each contributing their own out-edges. That is why n_pairs jumps 200–2000x
going from d=1 to d=2.

---

## 4. Per-Dataset Analysis

---

### bitcoin-alpha — N=3,783  E=24,186  bidir=83%  corr(out,in)=0.97

**Out-degree:** mean=6.4, median=2, std=18.3, max=490

| Bin | Nodes |
|-----|------:|
| =0  |   497 |
| =1  | 1,180 |
| 2–5 | 1,258 |
| 6–20|   610 |
| 21–100 | 212 |
| >100 |  **26** |

sum(k^2) = 1,420,436   sum(k*(k-1)) = 1,396,250

**In-degree:** mean=6.4, median=2, std=16.4, max=398

| Bin | Nodes |
|-----|------:|
| =0  |    29 |
| =1  | 1,465 |
| 2–5 | 1,418 |
| 6–20|   647 |
| 21–100 | 202 |
| >100 |  **22** |

sum(k_in * k_out) = 1,256,332

| d | n_pairs | NMI | Notes |
|---|--------:|----:|-------|
| 0 | 1,396,250 | **0.045** | Hub k=490 → 239k pairs (17%). 2,106 nodes contribute. No cap needed. |
| 1 | 1,236,208 | **0.003** | −20,124 recip excl. Avg 51 pairs/edge. d=0/d=1 = 1.13× |
| 2 | 261,042,821 | 0.00002 | Avg L2=279 nodes (med=147, max=1,693). **208× vs d=1** |
| 3–12 | decreasing | <0.001 | noise floor |

d=0 ≈ d=1 because corr=0.97 → `sum(k_in*k_out) ≈ sum(k_out^2)`.

---

### bitcoin-otc — N=5,881  E=35,592  bidir=79%  corr(out,in)=0.95

**Out-degree:** mean=6.1, median=2, std=21.1, max=763. 38 nodes >100.  
**In-degree:** mean=6.1, median=2, std=17.7, max=535. 34 nodes >100.  
sum(k_in*k_out) = 2,301,858

| d | n_pairs | NMI | Notes |
|---|--------:|----:|-------|
| 0 | 2,794,766 | **0.077** | Hub k=763 → 581k pairs (21%). d=0/d=1 = 1.23× |
| 1 | 2,273,658 | **0.015** | −28,200 recip excl. Avg 64 pairs/edge. |
| 2 | 550,460,630 | 0.000001 | Avg L2=384 nodes (med=196, max=2,541). **242× vs d=1** |
| 3–12 | decreasing | <0.001 | |

---

### epinions — N=131,580  E=840,799  bidir=31%  corr(out,in)=0.60

**Out-degree:** mean=8.9, median=**1**, std=33.0, max=2,070

| Bin | Nodes |
|-----|------:|
| =0  | 36,541 **(28%)** |
| =1  | 52,632 |
| 2–5 | 24,971 |
| 6–20| 10,253 |
| 21–100 | 5,503 |
| **>100** | **1,680** |

sum(k_out^2) = 148,414,133. HUB_CAP=2000 clips top nodes slightly.

**In-degree:** mean=10.0, median=**1**, std=34.8, max=3,478

| Bin | Nodes |
|-----|------:|
| =0  | 47,276 **(36%)** |
| =1  | 39,363 |
| 2–5 | 26,569 |
| 6–20| 11,578 |
| 21–100 | 5,171 |
| **>100** | **1,623** |

sum(k_in*k_out) = 95,840,323

| d | n_pairs | NMI | Notes |
|---|--------:|----:|-------|
| 0 | 146,789,022 | **0.340** | HUB_CAP=2000 applied. 1,680 super-raters dominate. d=0/d=1 = 1.54× |
| 1 | 95,581,145 | **0.005** | −259,178 recip excl. |
| 2 | ~199 billion | 0.0002 | Avg L2=493 nodes (med=53, max=16,006). **~2,000× vs d=1** |
| 3–12 | hundreds of billions | <0.001 | |

**Strongest d=0 NMI (0.340).** Users have strong personal signing biases (optimists vs pessimists).
The 36% of nodes with zero out-degree are pure recipients who never rate anyone — invisible at d=0.

---

### wiki-elec — N=7,115  E=103,689  bidir=6%  corr(out,in)=0.32

**Near-bipartite.** Voters = high out-deg, low in-deg. Candidates = high in-deg, low out-deg.

**Out-degree:** mean=14.6, median=2, std=42.3, max=893

| Bin | Nodes |
|-----|------:|
| =0  | 1,005 |
| =1  | 2,382 |
| 2–5 | 1,546 |
| 6–20| 1,083 |
| 21–100 | 873 |
| **>100** | **226** |

sum(k_out^2) = 14,229,321

**In-degree:** mean=14.6, **median=0**, std=31.7, max=457

| Bin | Nodes |
|-----|------:|
| **=0** | **4,734 (67%)** |
| =1  |    78 |
| 2–5 |   190 |
| 6–20|   549 |
| 21–100 | 1,388 |
| >100 |  176 |

Only **2,381 nodes** have any in-edges (were ever nominated as candidates).  
sum(k_in*k_out) = 4,542,805

| d | n_pairs | NMI | Notes |
|---|--------:|----:|-------|
| 0 | 14,125,632 | **0.016** | Power-voter k=893 → 796k pairs (5.6%). 3,728 nodes contribute. |
| 1 | 4,536,951 | **0.0002** | d=0/d=1 = **3.11×** Avg target k_out (weighted by in-deg) = 43.8. |
| 2 | 3,174,746,549 | ≈0 | Avg L2=319 nodes (med=104, max=1,721). **700× vs d=1** |
| 3–12 | billions | ≈0 | |

**Why d=0 is 3× d=1:** A voter with k_out=893 but k_in=5 contributes 893×892=796k pairs
at d=0 but appears as a *target* (d=1 context source) only 5 times. High-out-deg nodes
have near-zero in-deg (corr=0.32), so `sum(k_in*k_out)` is suppressed.

---

### wiki-rfa — N=11,256  E=177,211  bidir=7%  corr(out,in)=0.40

Same bipartite-like structure as wiki-elec, larger.

**Out-degree:** mean=15.7, median=2, std=43.0, max=1,063. 384 nodes >100.  
**In-degree:** mean=15.7, **median=0**, std=37.1, max=684. **7,767 nodes (69%) have in_deg=0.**  
sum(k_in*k_out) = 9,899,592

| d | n_pairs | NMI | Notes |
|---|--------:|----:|-------|
| 0 | 23,423,844 | **0.018** | Hub k=1,063 → 1.1M pairs. d=0/d=1 = **2.37×** |
| 1 | 9,886,684 | **0.0003** | −12,908 recip excl. |
| 2 | ~9.2 trillion | ≈0 | Avg L2=359 nodes (med=138, max=2,198). **~930× vs d=1** |
| 3–12 | trillions | ≈0 | |

---

### slashdot090221 — N=82,140  E=549,202  bidir=18%  corr(out,in)=0.31

**Celebrity-asymmetric.** Power users are tagged by thousands (in_deg up to 2,543)
but only tag ~10 themselves. Out-deg bounded at ~400; in-deg reaches 2,543.

**Out-degree:** mean=6.7, median=1, std=23.6, max=428. 1,124 nodes >100.  
**In-degree:** mean=6.7, median=1, std=31.4, max=2,543. 726 nodes >100.  
sum(k_in*k_out) = 22,639,046

| d | n_pairs | NMI | Notes |
|---|--------:|----:|-------|
| 0 | 48,764,066 | **0.171** | No hub cap (max k_out=428 < 2000). d=0/d=1 = **2.16×** |
| 1 | 22,541,604 | **0.025** | −97,442 recip excl. 2nd strongest d=1 NMI. |
| 2 | ~45 trillion | 0.00008 | Avg L2=366 nodes (med=61, max=8,205). **~2,000× vs d=1** |
| 3–12 | trillions | <0.001 | |

---

## 5. Cross-Dataset Summary

| Dataset | d=0 NMI | d=1 NMI | d=2 NMI | d=0/d=1 pairs | d=2/d=1 pairs | corr(out,in) | bidir% |
|---------|--------:|--------:|--------:|--------------:|--------------:|-------------:|-------:|
| bitcoin-alpha | 0.045 | 0.003 | 0.00002 | 1.13× | 208× | 0.97 | 83% |
| bitcoin-otc   | 0.077 | 0.015 | 0.000001 | 1.23× | 242× | 0.95 | 79% |
| epinions      | **0.340** | 0.005 | 0.0002 | 1.54× | ~2,000× | 0.60 | 31% |
| wiki-elec     | 0.016 | 0.0002 | ≈0 | **3.11×** | 700× | 0.32 | 6% |
| wiki-rfa      | 0.018 | 0.0003 | ≈0 | **2.37×** | ~930× | 0.40 | 7% |
| slashdot      | 0.171 | **0.025** | 0.00008 | 2.16× | ~2,000× | 0.31 | 18% |

**Three structural regimes:**
1. **Bitcoin** (high bidir, high corr): d=0 ≈ d=1 pairs; moderate d=0 NMI; near-zero d=1+
2. **Epinions/slashdot** (medium scale, low bidir): strong d=0 from ego-network bias; real d=1; zero d=2+
3. **Wiki** (near-bipartite, corr≈0.3): very weak NMI everywhere; extreme degree asymmetry

---

## 6. File Locations

| File | Description |
|------|-------------|
| `outputs/mi_vs_dist/mi_vs_dist_report_v3.txt` | Main results: MI and NMI at d=0..12 for all 6 datasets |
| `outputs/mi_vs_dist/mi_vs_dist_<dataset>.png` | Per-dataset plots |
| `outputs/mi_vs_dist/run_v3.log` | Full v3 run log (timing, pair counts) |
| `outputs/mi_vs_dist/run_v2.log` | v2 run — bitcoin-alpha missing |
| `outputs/mi_vs_dist/d2_validation.txt` | Exact vs approximate d=2 comparison (OOM beyond bitcoin-otc) |
| `scripts/edge_sign_mi_vs_distance_v3.py` | Current production script |
| `scripts/edge_sign_mi_vs_distance.py` | Original v2 (kept for reference) |
| `scripts/validate_d2_mi.py` | Standalone d=2 exact vs approximate validator |

---

## 7. Open Questions and Proposed Experiments

### 7.1 Transpose graph

Run the identical MI analysis on A^T (reverse all edge directions). For wiki-elec/rfa the
reverse direction connects candidates back to their voters.

**Expected:** stronger d=1 signal in the transpose — a candidate who received mostly
negative votes should appear alongside other unpopular candidates in the reverse BFS.
This would show that the *recipient* perspective carries signal invisible in the sender direction.

**Results (run via `scripts/edge_sign_mi_vs_distance_v3.py --transpose --d-max 8`,
`outputs/mi_vs_dist/mi_vs_dist_transpose_report_v3.txt`).**

**d=1 is a dead end — turns out to be a mathematical identity, not a new signal.**
The d=1 MI (bits) is *exactly* identical between the forward and transpose graphs for
every dataset (to 8 decimal places): bitcoin-alpha 0.00078178, bitcoin-otc 0.00422282,
epinions 0.00193214, wiki-elec 0.00012842, wiki-rfa 0.00017450, slashdot 0.01824065.
This isn't a coincidence: forward-d1 sums `MI(sign(u→v) ; sign-distribution of v's
other out-edges)` over all edges, while transpose-d1 sums `MI(sign(u→v) ; sign-distribution
of u's other in-edges)` — over the *same* edge set, these two aggregate contingency tables
turn out to be transposes of each other, so MI (which doesn't care which axis is rows vs
columns) comes out identical. (NMI differs slightly because it normalises by H(context),
whose marginal *does* differ between "v's out-edges" and "u's in-edges".) **So the §7.1
hypothesis as originally framed (stronger d=1 in transpose) is mathematically impossible
to observe — the real "recipient perspective" effect, if any, has to show up at d=0.**

**d=0 — recipient-vs-sender "node prior" — the hypothesis holds, but only for
wiki-elec/wiki-rfa, and the opposite holds for epinions/slashdot:**

| Dataset | forward d=0 NMI (sender bias) | transpose d=0 NMI (recipient reputation) | ratio (transpose/forward) |
|---|---|---|---|
| bitcoin-alpha | 0.0447 | 0.0850 | 1.9x |
| bitcoin-otc | 0.0766 | 0.0773 | ~1.0x |
| epinions | **0.3404** | 0.0679 | 0.20x |
| wiki-elec | 0.0158 | **0.0438** | 2.8x |
| wiki-rfa | 0.0180 | **0.0329** | 1.8x |
| slashdot | **0.1708** | 0.0618 | 0.36x |

For wiki-elec/wiki-rfa — the low-bidirectional-edge, near-bipartite voter→candidate
graphs — the *recipient's* aggregate reputation (sign distribution of v's other
in-edges) is ~2-3x more informative about edge sign than the *sender's* own bias
(sign distribution of u's other out-edges), confirming the qualitative direction of
the §7.1 hypothesis, just at d=0 rather than d=1. For epinions and slashdot the
opposite holds — sender bias dominates by 3-5x — consistent with these being
denser, more reciprocal trust networks where "how generous is this rater" is a
stronger prior than "how is this target generally received."

**d≥2 (both directions): still negligible everywhere.** Transpose NMI at d=2..8
stays below ~0.005 for all datasets (same order of magnitude as the forward-graph
result already reported in §4-5), with the same noisy uptick at the largest d for
the smaller datasets (small `n_pairs`). The transpose run does **not** surface any
new long-range signal — it only reshuffles where the (already-known-to-be-small)
non-d0 signal sits.

**Takeaway for §8:** the transpose experiment doesn't change the central puzzle —
both directions agree that MI(sign, sign) ≈ 0 beyond d=1 (mod the d=1 identity above).
It does refine the d=0 picture: "node prior" is not a single number per dataset but
has a sender-side and a recipient-side component whose relative importance flips
between datasets, which is useful context for the §7.6 node-prior baseline (Step 2).

---

### 7.2 Node-level MI — structural features and model embeddings

**Structural features per node:** out-degree, in-degree, in/out ratio, signed out-ratio
(fraction positive), PageRank, HITS hub/authority scores, betweenness centrality.

**Model embedding experiments:**
- Extract node token embeddings from trained transformer's embedding matrix (indexed by `N_id`)
- Measure MI(embedding_u, sign(u→v)) — does the model encode signing tendency in node tokens?
- Cluster embeddings: do "trusters" and "distrusters" separate?
- Compare ||emb(u) - emb(v)|| vs sign(u→v): do mutually trusting nodes sit closer?

---

### 7.3 Attention map analysis — does the transformer actually look far?

**Experiment A — attention by distance:**
Hook into `nn.MultiheadAttention` during inference. For each masked edge at position `i`,
plot mean attention weight as a function of `|position - i|` per layer and per head.
Does any head attend beyond ±2 positions from the target?

**Experiment B — attention-capped ablation (most direct test):**
Add an attention mask restricting each token to attend only within ±K positions.
Train variants: K=1, 2, 4, 8, full (uncapped).
If K=2 ≈ K=full in test AUC, the model is only using 2-hop local context
and longer walks add only variance reduction, not new information.

**Experiment C — 3-token baseline:**
Train transformer on just `[left_node, MASK, right_node]` — 3 tokens, no walk context.
If this matches full-walk AUC, the entire walk sampling machinery is irrelevant and a
simple lookup of the two adjacent node embeddings would suffice.

---

### 7.4 Walk-to-Paint vs GNN: what is fundamentally different?

**GAT's structural limitation (your intuition formalised):**

In an L-layer GAT, the gradient of the target's representation w.r.t. a 2-hop neighbour
is proportional to the product of two attention weights:

    dh_v^(2) / dx_w^(0)  ∝  alpha_vx * alpha_xu * W^2

If the intermediate node x has low attention weight alpha_vx, the 2-hop signal is gated
regardless of how informative x→w would be. **A "boring" intermediate node blocks the path.**
In a GAT voting on whether u→v is positive, if the path u←h→x→v passes through a low-attention
hub h, the information at x is suppressed even if x is highly informative.

**Walk-transformer's advantage (horizontal vs forward attention):**

In a walk `[N_a, E_1, N_b, E_2, N_c, MASK, N_d]`, the masked position can attend
**directly** to `E_1` (3 positions away) with the same flat attention as attending to `N_c`
(1 position away). No multiplicative gating through intermediate nodes. Attention is parallel.

Think of it this way: GAT attends *horizontally* across a fixed-depth neighbourhood
(all nodes at hop k). The walk-transformer attends *forward along paths* — it sees a specific
sampled path through the graph, and any edge on that path can directly inform the target prediction.
A low-attention intermediate node in GAT cuts off the path; a "boring" intermediate node in a
walk just sits at an intermediate position and the transformer can simply skip over it by
attending to earlier positions in the walk.

Additionally: the walk transformer sees each test edge in **hundreds of different walk contexts**
(different entry points, different surrounding edges). A hub dominates GNN aggregation uniformly;
in walk sampling its contribution scales with how often walks actually pass through it,
which is naturally regularised.

**Experiment:** Train SGCN/GAT on same split. Compare test AUC per edge.
Find edges where walk-transformer wins. Analyse: are these edges where the 2-hop path
goes through a low-GAT-attention intermediate? This would directly validate the mechanism.

---

### 7.5 Is the 2-hop neighbourhood truly sufficient?

**k-hop subgraph sequences:**
Instead of random walks, for each anchor edge `u→v`, enumerate all edges within k directed
hops of `u` and `v`, linearise in BFS order, feed to transformer.
- k=1: only edges adjacent to u or v
- k=2: full 2-hop neighbourhood (exact, deterministic)
- Full walk: current approach

Compare AUC: k=1 vs k=2 vs full-walk. If k=2 ≈ full-walk, walking beyond 2 hops is useless.

**Walk coverage analysis:**
For each test edge compute the fraction of its 2-hop neighbourhood that appeared in
training walks. Does per-edge AUC correlate with this coverage?
This separates "inherently hard edges" from "under-sampled edges."

---

### 7.6 Decomposing d=0: node bias vs structural correlation

**d=0 NMI** mixes two effects:
1. **Node-level signing bias**: u mostly gives +1 (optimist) or -1 (pessimist)
2. **Community-level bias**: u's neighbours share its signing tendency

**Node-prior baseline:** Predict sign(u→v) = majority sign of all u's *other* out-edges.
Compute AUC. This is the maximum "free" AUC from d=0 signal alone.

**Conditional MI:** Compute MI(Y; S | node_u).
If ≈0 at d=0, the entire ego-network signal is node bias — balance theory and structural
correlation add nothing beyond a per-node intercept (equivalent to a logistic regression
with node dummy variables).

---

### 7.7 Does the tiny d=3+ signal have a real explanation?

Bitcoin-otc d=7 NMI=0.008, slashdot d=9 NMI=0.002 are above the statistical noise floor
(null expectation ≈ 1/(2*n*ln2) ≈ 0 for n in the billions).

**Hypothesis A — community structure:** BFS at d=3–7 exits the ego-network and enters
a broader signed community with consistent sign tendency.

**Hypothesis B — hub-mediated correlation:** BFS from u often passes through a high-degree
hub h at d=1–2. Since h's out-edges are correlated with its in-edges (corr=0.95 for bitcoin),
an indirect anchor↔context correlation persists even at d=3–7.

**Experiment:** Stratify d=3+ pairs by whether a hub (>100 degree) appears on the BFS path.
Compute MI separately for hub-mediated vs non-hub paths. If hub-mediated paths drive the
d=3+ signal, Hypothesis B is confirmed.

---

### 7.8 MI on model-accessible labels vs full ground truth

Current analysis uses full ground-truth labels. The model only sees 48% train-split context.

**Experiment:** Compute MI(Y_test; S_train at d=0,1,2) where S_train are only visible
(train-split) edge signs. If walk sampling over-represents high-degree edges (they appear
proportionally more often in walks), the accessible MI could differ from the population MI
in a non-uniform way that biases the model's learning signal.

---

### 7.9 Per-head attention specialisation by distance

**Hypothesis:** Different attention heads specialise on different positions in the walk.
Head 1 might focus locally (±1), head 2 on 2-hop context, etc.

**Experiment:**
- Compute each head's "effective distance" = mean |pos_attended - pos_target| weighted by
  attention weight across all inference examples
- Cluster heads by effective distance profile
- Ablate each head individually, measure AUC drop
- If "far-looking" heads have near-zero ablation impact, this confirms the model is not
  meaningfully using long-range attention even though it architecturally could

---

## 8. The Central Open Question

**Given that MI(d>=2) ≈ 0, why does our model substantially outperform simpler 1-hop baselines?**

Three candidate answers (not mutually exclusive):

1. **Variance reduction from aggregation.** Each walk gives one noisy local estimate.
   Averaging 500+ walk occurrences per edge gives a reliable prediction. A 1-hop GNN
   sees each edge exactly once — no averaging.

2. **Joint multi-edge patterns.** The transformer attends to multiple context edges
   simultaneously. The sequence `[E_+1, N_x, E_+1, N_y, MASK]` (two consecutive positive
   edges before the target) may be more informative than either alone — the transformer
   can model joint patterns that marginal MI would miss.

3. **Walk diversity over local context distribution.** The same 2-hop neighbourhood
   appears in different walks at different entry points. This Monte Carlo sampling is more
   robust to hub dominance than a single deterministic GNN aggregation pass.

**Discriminating experiments:**
- Attention-capped ablation (§7.3B): if K=2 ≈ K=full then only (1) matters
- 3-token baseline (§7.3C): tests whether (1) alone is sufficient
- k-hop subgraph (§7.5): tests whether (3) adds beyond a deterministic 2-hop neighbourhood
