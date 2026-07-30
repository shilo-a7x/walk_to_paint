# Lead 4c — asymmetric entropy sensitivity: what we know and what we don't

## Professor Q&A (2026-07-05) — his message in 3 blocks, with equations and results

His full message, split into the 3 claims/questions he raised, each answered directly.

### Block 1 — "entropy of incoming edges vs outgoing edges"

> "נסתכל על שתי קשתות עוקבות. האינפורמציה המשותפת של השניה על הראשונה או הראשונה על השניה היא
> זהה, לפי ההגדרה. אבל אם לראשונה יש הרבה יותר אנטרופיה מהשניה אז הראשונה תסביר את השניה הרבה
> יותר טוב מההפך. זה קל לבדוק... שך של יוצאות הרבה יותר נמוכה."

**Reasoning (information theory):** for two quantities A, B, mutual information is symmetric,
`I(A;B) = I(B;A)`, by definition:

```
I(A;B) = H(A) - H(A|B) = H(B) - H(B|A)
```

But the *fraction of uncertainty explained away* is not symmetric. If `H(A) ≫ H(B)`, then:

```
I(A;B)/H(B)  →  can approach 1   (B is almost fully explained by A)
I(A;B)/H(A)  →  stays small      (A is barely explained by B)
```

So whichever side has *lower* marginal entropy gets more explained by the higher-entropy side.
His bet: outgoing-edge sign entropy is the lower side.

**Equation used to test it — per-node binary entropy of the sign distribution:**

```
p_out(n) = fraction of n's outgoing edges that are positive
H_out(n) = -p_out(n)·log2(p_out(n)) - (1-p_out(n))·log2(1-p_out(n))     (0 if p_out ∈ {0,1})
```

and symmetrically `H_in(n)` using n's incoming edges. Computed on the full canonical graph
(all edges, `load_edges_canonical`), one value per node, restricted to nodes with both
in-degree≥1 and out-degree≥1 so `H_out(n)` and `H_in(n)` are both defined ("paired").

**Result: confirmed in 4/6 datasets, reversed in 2/6.** Full table and detail in
"Claim 1 result" below. Bottom line: bitcoin-alpha/bitcoin-otc/epinions/slashdot090221 — his bet
holds (p ≤ 5×10⁻⁴). wiki-elec/wiki-rfa — reverses (p=1.00 against his direction; these are
RfA/election-style graphs where a few admin-candidate nodes concentrate large, genuinely mixed
in-vote counts).

### Block 2 — "in-degree distribution broader than out-degree" + "think of a way to quantify learning from in vs out"

> "התפלגות דרגות הכניסה הרבה יותר רחבה מאשר דרגות היציאה... מה שאומר שמסלולים יכולים ללמוד יותר
> מהכניסה מאשר ביציאה. אני לא בטוח איך לכמת את זה במודלים השונים אבל אשמח שתחשוב על דרך לבדוק"

**Equation used to test the degree claim:**

```
outdeg(n) = |{edges leaving n}|,   indeg(n) = |{edges entering n}|
CV_out = std(outdeg) / mean(outdeg),   CV_in = std(indeg) / mean(indeg)
```

**Result: not confirmed, genuinely mixed.** Full table below. bitcoin-alpha/bitcoin-otc actually
have *more* relative spread on the out-degree side (opposite of his bet); wiki-elec/wiki-rfa have
much higher mean in-degree but *lower* relative spread (CV≈1.0 vs 2.6) due to admin-candidate
concentration; only epinions/slashdot lean his direction.

**Way to quantify "paths learn more from in than out" across models — run 2026-07-05.**
Added `log(outdeg(u))` and `log(indeg(v))` as two more z-scored covariates to the atomic
regression (`correct ~ src_out + src_in + tgt_out + tgt_in + twohop_in + twohop_out + log_outdeg_u
+ log_indeg_v`), refit per model with the same cluster-robust setup. Implemented as a fully
separate code path (`run_atomic_degree_fits`/`run_atomic_degree_fits_zscored`, spec
`atomic_degree`/`atomic_degree_zscored`) — the original 6-term atomic model/fits are byte-identical
before and after (verified, max diff 0.0).

**Result (pooled, z-scored):**

| Term | walk_full | walk_localattn4 | GINEConv | SiGAT |
|---|---|---|---|---|
| log_outdeg(u) | 0.383\*** | 0.374\*** | −0.017 (n.s., p=0.53) | 0.360\*** |
| log_indeg(v) | 0.144\*** | 0.100\*** | 0.126\*** | 0.171\*** |

**Not what he predicted — the ranking reverses.** His hypothesis implied the walk model's
in-degree coefficient should dominate. Instead, for the walk models (and SiGAT), **out-degree(u)
has the much bigger effect** (0.36–0.38 vs 0.10–0.17) — walk models benefit more from more evidence
of *u's own rating behavior*, not from v's in-neighborhood size.

**What is a clean, real finding: GINEConv is the only model with ~zero benefit from
out-degree(u)** (β=−0.017, not significant), while every model — including GINEConv — benefits
from in-degree(v). This is architecturally consistent with the earlier explanation: GINEConv's
node embedding for u is built purely from u's in-neighbors via message passing, so u's own
out-degree is structurally invisible to it; the walk model and SiGAT can both exploit "more
evidence of u's typical behavior" when u has more out-edges, which GINEConv simply cannot access.

The 6 entropy terms are essentially unchanged after adding these two degree covariates (e.g.
walk_full `src_out` −1.002 → −1.063, `tgt_in` −0.629 → −0.633) — the original entropy asymmetry is
not a degree/hubness artifact.

### Block 3 — "no drastic model difference, but that may be the ceiling given real differences"

> "לגבי ההשוואה בין מודלים. לא נראה שיש הבדל דרסטי אבל יכול להיות שההבדלים הקטנים הם מה שניתן
> להשיג בהינתן ההבדלים האמיתיים"

**Checked directly against the SOTA table** (best walk AUC vs best canonical-split GNN AUC, per
dataset — see `CLAUDE.md` SOTA table):

| Dataset | Best canon-GNN AUC | Best walk AUC | Gap (pp) | Claim 1 held? |
|---|---|---|---|---|
| wiki-rfa | 0.8831 | 0.8932 | **1.01** | reversed |
| wiki-elec | 0.8930 | 0.9038 | **1.08** | reversed |
| bitcoin-alpha | 0.9051 | 0.9362 | 3.11 | confirmed |
| epinions | 0.9146 | 0.9568 | 4.22 | confirmed |
| slashdot090221 | 0.8587 | 0.9012 | 4.25 | confirmed |
| bitcoin-otc | 0.8972 | 0.9427 | 4.55 | confirmed |

**Result:** the 2 datasets where Claim 1 reversed are exactly the 2 smallest walk-vs-GNN gaps —
clean separation, no overlap between the "confirmed" group (≥3.1pp) and the "reversed" group
(≤1.1pp). Spearman rank correlation between "how strongly Claim 1 holds" (fraction of non-tied
node pairs with H_out<H_in) and gap size ≈ 0.71–0.77 across the 6 datasets (n=6 — too small for a
real p-value, but the categorical split is clean on its own).

This supports his framing directly: where real, inherent directional asymmetry exists in the sign
data, the walk model has more of it to exploit and the SOTA gap is large; where it's weak or
reversed (the 2 vote-style datasets), the achievable advantage shrinks toward zero.

---

## The regression

Logistic regression: `P(correct prediction) ~ 6 directional entropy terms + dataset intercepts`.
Fit separately per model (walk_full, walk_localattn4, GINEConv, SiGAT).
All entropy inputs z-scored before fitting. Two-way cluster-robust SEs (edge endpoints u,v).
n=125,212 pooled across 6 datasets.

For edge (u→v), the 6 atomic terms are:
- `src_out`: entropy of signs on u's outgoing edges (how consistently does u rate others?)
- `src_in`: entropy of signs on u's incoming edges (how consistently is u rated by others?)
- `tgt_out`: entropy of signs on v's outgoing edges
- `tgt_in`: entropy of signs on v's incoming edges (how contested is v's reputation?)
- `twohop_in`, `twohop_out`: 2-hop path consistency terms (small effects, not the story)

Full coefficients: `lead4_coefficients.csv` / `lead4_coefficients.md`.

## Key finding: model families fail on opposite entropy directions

| Term | walk_full β | walk_localattn4 β | GINEConv β | SiGAT β |
|---|---|---|---|---|
| **src_out** | **-1.002** | **-1.015** | -0.499 | -1.013 |
| src_in | ≈0 | ≈0 | -0.220 | -0.094 |
| tgt_out | -0.043 | -0.047 | ≈0 | ≈0 |
| **tgt_in** | -0.629 | -0.638 | **-1.148** | **-0.826** |
| twohop_in | -0.108 | -0.123 | -0.200 | -0.096 |
| twohop_out | ≈0 | ≈0 | +0.054 | ≈0 |

The structural asymmetry (all β are log-odds per 1-SD increase in that entropy):

- **`src_out` hurts walk models ~2× more than GINEConv** (β ≈ -1.0 vs -0.5). Both walk variants and SiGAT are similarly hurt.
- **`tgt_in` hurts GINEConv ~1.8× more than walk** (β ≈ -1.15 vs -0.63). SiGAT is intermediate (-0.83).
- `src_in`/`tgt_out` are near-null for walk, mild for GNNs.

In words: the walk model is more sensitive to the *source node's rating inconsistency*; GNNs are more sensitive to the *target node's contested reputation*. Note that SiGAT (graph attention) behaves more like the walk model on `src_out` (similarly hurt) and intermediate on `tgt_in` — its attention mechanism may partially compensate for target ambiguity.

## What we checked and ruled out

**Hypothesis: forward-only walk sampling → more right context → asymmetric information access.**

The walk sampler only traverses outgoing edges, so a walk for edge (u→v) has:
- "Left context" (tokens before the masked edge) = path leading to u in that walk
- "Right context" (tokens after) = forward walk from v

Measured empirically (bitcoin-alpha, 5M-walk k_cover k=5, 17.5M test-edge occurrences across walks):

| | Left context tokens | Right context tokens |
|---|---|---|
| Mean | **55.6** | **55.5** |
| Median | 45 | 45 |

The context is **symmetric**. Anchor walks (where edge appears at position 1, left=1 token only) are 2.9% of total occurrences — negligible. 97% of the time the test edge appears deep in a walk and has substantial context on both sides.

Therefore: the `src_out` vs `tgt_in` asymmetry in the regression is **not explained by an imbalance in available context length**. The walk model has as much left context (leading to u's in-neighbors) as right context (v's out-neighbors) on average.

## Open question

Something about the walk model's *use* of context — not the quantity of it — makes it more sensitive to source-out entropy and less sensitive to target-in entropy than GNNs. And vice versa.

Possible angles (unverified):
- GNNs explicitly aggregate over v's in-neighbors in message passing (computing h_v) → the aggregation mechanism itself mixes conflicting signals when `tgt_in` entropy is high. The walk model has no such explicit aggregation step.
- The walk model's loss is computed on the *masked edge token* position, which may implicitly bias learning toward the source node's context (u is the immediately preceding token in many walks). Attention patterns may differ accordingly.
- Attention head analysis: does the walk model's attention for a masked edge position preferentially attend to source-node-adjacent tokens vs target-node-adjacent tokens? (Not yet measured without the BFS-distance bug fix in `scripts/attention_analysis.py`.)

What we do **not** know: whether this is a structural property of the architectures, a training artifact, or a consequence of the specific walk format.

## Follow-up: professor's information-theoretic hypothesis (checked directly, 2026-07-05)

Professor's reasoning (translated): for two adjacent edges sharing a node, mutual information is
symmetric by definition, but if one side has much higher marginal entropy than the other, the
higher-entropy side explains a much larger *fraction* of the lower-entropy side's uncertainty than
the reverse (I/H_low is large, I/H_high is small). He proposed two concrete, checkable claims:

**Claim 1 — a node's outgoing-edge sign entropy (H_out) is systematically lower than its
incoming-edge sign entropy (H_in).** If true, this is an inherent property of the sign data itself
(nothing to do with walk sampling) — a rater tends to be internally consistent (mostly-positive or
mostly-negative), while the ratings a node *receives* are a mix of many different raters' opinions,
so are naturally noisier.

**Claim 2 — in-degree distribution is usually much broader/more variable than out-degree
distribution.**

### Method

Computed directly from the canonical edge lists (`load_edges_canonical`, all 6 datasets, no
sampling involved — this is a property of the raw graphs). For Claim 1: per-node binary entropy of
outgoing vs incoming edge signs (`build_sign_dicts`/`entropy_lookup` from
`lead4_entropy_heterogeneity.py`), restricted to nodes with both an out- and in-degree ≥ 1 so the
comparison is paired. Reported the aggregate mean/median, plus (since both distributions are
heavily zero-inflated — most nodes are perfectly one-sided raters) the **tie rate** and, among
non-tied node pairs, the fraction where H_out < H_in with a one-sided Wilcoxon signed-rank test on
the non-zero differences.

### Claim 1 result — confirmed in 4/6, reversed in 2/6

| Dataset | mean H_out | mean H_in | tie rate | frac(H_out<H_in) among non-tied | p (one-sided) |
|---|---|---|---|---|---|
| bitcoin-alpha | 0.069 | 0.083 | 0.794 | 0.516 | 5.1e-04 |
| bitcoin-otc | 0.089 | 0.107 | 0.761 | 0.551 | 2.7e-11 |
| epinions | 0.085 | 0.136 | 0.623 | 0.623 | 3.7e-244 |
| slashdot090221 | 0.211 | 0.240 | 0.457 | 0.602 | 2.6e-218 |
| wiki-elec | 0.300 | 0.390 | 0.102 | **0.404** | **1.00** |
| wiki-rfa | 0.309 | 0.398 | 0.106 | **0.424** | **1.00** |

On **bitcoin-alpha, bitcoin-otc, epinions, slashdot090221**: the professor's bet holds, cleanly
and significantly — among nodes where H_out and H_in actually differ, a majority have the lower
entropy on the outgoing side, and the effect is highly significant in epinions/slashdot (huge n).

On **wiki-elec and wiki-rfa**: the pattern **reverses** — a majority of non-tied nodes have
H_out > H_in (one-sided test against the professor's direction fails completely, p=1.00). The
raw unconditional means still look consistent with the hypothesis (0.30 vs 0.39, 0.31 vs 0.40),
but that's an artifact of a right-skewed H_in distribution, not the node-level majority pattern —
these are RfA/election-style datasets where a handful of admin-candidate nodes receive very large,
genuinely mixed in-vote counts (skewing the mean up) while most ordinary nodes don't fit the
"consistent rater, noisy target" story the same way bitcoin/epinions/slashdot trust graphs do.

**Conclusion: Claim 1 is real but dataset-dependent — strong on the 4 trust/rating-style graphs,
absent (reversed) on the 2 vote-style graphs.**

### Claim 2 result — not confirmed, dataset-dependent both ways

| Dataset | mean out-deg | mean in-deg | std out-deg | std in-deg | CV out | CV in |
|---|---|---|---|---|---|---|
| bitcoin-alpha | 7.36 | 6.44 | 19.4 | 16.5 | 2.64 | 2.56 |
| bitcoin-otc | 7.39 | 6.08 | 23.1 | 17.7 | 3.12 | 2.91 |
| epinions | 8.85 | 9.97 | 38.5 | 43.0 | 4.35 | 4.31 |
| wiki-elec | 16.97 | 43.55 | 45.2 | 41.8 | 2.66 | **0.96** |
| wiki-rfa | 17.26 | 50.79 | 44.7 | 51.6 | 2.59 | **1.02** |
| slashdot090221 | 12.47 | 7.81 | 31.1 | 33.8 | 2.49 | 4.33 |

By raw std, in-degree is *not* consistently broader — bitcoin-alpha/otc actually have **more
spread on the out-degree side**. By coefficient of variation (std/mean, the more meaningful
"relative spread" measure since means differ a lot), wiki-elec/wiki-rfa show the *opposite* of
the claim: in-degree is far *less* relatively variable than out-degree (0.96–1.02 vs 2.6), because
those two datasets concentrate huge in-degree on a small number of admin-candidate nodes (high
mean, but a fairly regular/repeated pattern) while epinions and slashdot show in-degree modestly
*more* spread than out-degree, consistent with the claim.

**Conclusion: no universal winner. Degree-spread direction is genuinely dataset-specific — 2/6
support the claim (epinions weakly, slashdot), 2/6 contradict it outright (bitcoin-alpha/otc), 2/6
depend entirely on which spread measure you pick (wiki-elec/wiki-rfa: broader in raw std, narrower
in CV).**

### What this does and doesn't establish

- Claim 1 gives a genuine, inherent (non-sampling) directional asymmetry in 4/6 datasets: a node's
  own outgoing rating behavior is more predictable/consistent than the incoming opinions about it.
  This is a real candidate mechanism for *some* baseline directionality in the data, independent of
  any model.
- It is **not yet shown** that this specific node-level entropy asymmetry is *what causes* the
  regression's `src_out`-hurts-walk / `tgt_in`-hurts-GNN pattern. That would require a direct link:
  e.g., showing that within the `src_out` term, the walk model's degradation concentrates on edges
  where the source node's entropy is unusually high *relative to how consistent out-raters usually
  are* (a rare, surprising deviation) — versus the GNN's `tgt_in` degradation being smooth across
  the whole range of target in-entropy (a structural cost of aggregation, not a surprise effect).
  Both stories are plausible; only the raw entropy-asymmetry fact has been checked so far, not the
  causal link to model behavior.
- Given the dataset-dependence found here (both claims fail on at least one dataset), a
  single clean mechanistic story probably isn't going to explain all 6 datasets uniformly — the
  next useful step is probably per-dataset, not a pooled average.
