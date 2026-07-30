# Panel B Investigation Report — MI/phi vs. line-graph distance on slashdot090221

**Status: both threads below are resolved.** This doc consolidates a multi-session investigation
into two separate questions about Panel B (the "information decays, then bumps back up" MI/phi
curve used in the PEWTER paper's Empirical Confirmation section): (A) is the post-minimum bump at
distance 4–6 real, and (B) why does an external colleague's independent reimplementation of the
same idea report "close to zero" everywhere. Both questions turned out to have real, quantifiable
answers — no fabricated numbers, no hand-waving; every number below is either already in
`aaai2027/figure_data/`/`outputs/panelb_diagnostics/` or reproducible from the scripts in
`scripts/panelb_diagnostics/`.

**Read this before touching Panel B, Figure 4/ablations, or attention-map work** — several of the
methodological lessons here (cluster-robust CIs, unseeded randomness, direction/recording
conventions for BFS-based graph distance) generalize to other distance-based diagnostics in this
repo (Lead 2/3, `attention_analysis.py`, the walk-coverage sampler analysis), not just Panel B.

---

## 1. What Panel B is

Panel B of the paper's 3-panel Empirical Confirmation figure (`fig:empconf-panels`) plots MI
(bits) and NMI between an anchor edge's sign and a context edge's sign, as a function of
line-graph (edge-adjacency) distance between them, pooled over many anchors. The production
numbers (`aaai2027/figure_data/empconf_panelB_mi_decay_linegraph.csv`,
`scripts/paper_figures/extract_empconf_panelB_mi_decay_linegraph.py`) show the expected sharp
decay from d=1, bottoming out around d=3, and then a genuine **rise (the "bump") through d=4–6**
before decaying again — this shape is the thing both investigation threads below are about.

Production convention, throughout: **undirected** BFS (both edge directions), **all edges** among
every reached node recorded (not just BFS-discovery/tree edges), a **large seeded random** sample
of anchor edges (tens of thousands), and **no cap** on how many (anchor, context) pairs get pooled
into the per-distance contingency table — phi/MI are computed from a running 2×2 count, so there's
never a reason to materialize or subsample the raw pair list.

---

## 2. Thread A — Is the bump real, or an artifact?

### 2.1 Prior investigation (2026-07-28, documented in `CLAUDE.md` / `PEWTER_ASSETS_CHECKLIST.md` #12)

Already ruled out before this session: BFS-correctness (shell values matched `networkx` ground
truth exactly), numerical instability (integer counts, no overflow), and pure estimator noise (the
bump survives a `--shuffle-signs` null control by 2–3 orders of magnitude on most datasets, and
doesn't keep growing if `d_max` is extended to 9 — the graphs simply run out of reachable
structure). Two real, confirmed mechanisms were identified: **degree collapse at outer BFS shells**
(a correlate, not sufficient by itself — a direct degree-filter pilot on bitcoin-alpha left the
bump essentially unchanged) and **non-independence/clustering of pooled (anchor, context) pairs**
(a small number of "hub" context edges get hit by many different anchors' BFS at long distance —
measured: top-10 distinct edges account for ~40% of all pooled pairs at the farthest distance vs.
0.2% at distance 1). The clustering mechanism was flagged as the leading candidate but its fix
("dedup by distinct edge, or a cluster-aware bootstrap") was **not yet implemented** as of that
writeup.

### 2.2 This session's first attempt — retracted

Built a "fix": dedup the pooled pairs down to one row per distinct context edge, recompute
phi/MI on the deduped sample, and bootstrap **that deduped sample** to get a "cluster CI." This
looked like it roughly halved the bump's magnitude and widened its CI enough to look
non-significant at some distances.

**This was wrong, and was retracted in-session after direct pushback.** The error: bootstrapping a
smaller, deduped sample just reproduces the standard `1/√n` CI-widening from a smaller `n` — it
has nothing to do with genuinely correcting for clustering. It also silently changes the *point
estimate* by discarding real data, which is a second, separate problem: repeated encounters of a
hub edge are not obviously "bias" to be thrown away — they may be a real feature of how often a
walk/model would actually encounter that edge in production. Discarding them for the sake of
getting a CI is optimizing the wrong thing.

There was also a concrete, measured cost to the dedup step, validating a class-imbalance concern
raised directly in response to the first attempt: picking one random occurrence per distinct
context edge measurably shifts the sign composition of the pooled sample, not just its size —
**up to −6.23 percentage points** of "% positive" shift at some distances (full table below). Any
conclusion drawn from a deduped sample is drawn from a sample with a measurably different sign
balance than the real one, which is exactly the kind of distortion that should raise suspicion of
an analysis, not increase confidence in it.

### 2.3 The corrected method — proper (block) cluster bootstrap

The right fix keeps the two problems ("get a valid CI" vs. "the point estimate might be biased")
separate, and only solves the first one:

- **Point estimate: use ALL rows, nothing discarded.** This is the number that matters — repeated
  hub-edge encounters are real data, not noise to filter out.
- **CI: a proper block/cluster bootstrap.** Resample distinct **context edges** with replacement
  (not raw rows), but each resampled edge contributes **all of its original rows**, not just one.
  This changes the resampling *unit* from "row" to "edge" (correctly modeling the actual
  dependence structure — rows sharing a context edge are not independent draws) without discarding
  any information anywhere.

Implementation: `scripts/panelb_diagnostics/proper_cluster_bootstrap.py`. The core trick — turning
"pick `n_clusters` clusters with replacement, then concatenate every original row of each chosen
cluster" into a single vectorized numpy operation (no per-replicate Python loop over rows) — is a
sort-by-cluster + `np.unique(..., return_index=True, return_counts=True)` + a `np.repeat`/`np.cumsum`
"ragged repeat" construction. This is verified against a brute-force reference on synthetic data
in a `_self_test()` that runs automatically before touching real data.

Scaled-down real-data run (undirected BFS on slashdot090221, 3,000 random anchor edges, capped at
4,000 candidate context edges/anchor for tractability — **not** the full ~549K-anchor production
sweep, this experiment answers the methodological question, not a byte-for-byte reproduction of the
published CSV): `outputs/panelb_diagnostics/proper_cluster_bootstrap_output.txt`.

**Dedup cost / composition-shift table** (validates the class-imbalance concern directly):

| d | n_raw | n_distinct edges | % discarded by dedup | % positive (raw) | % positive (deduped) | shift |
|---|---|---|---|---|---|---|
| 1 | 7,198 | 7,073 | 1.7% | 79.66% | 79.51% | −0.15pp |
| 2 | 735,367 | 328,299 | 55.4% | 78.61% | 76.61% | −2.00pp |
| 3 | 8,513,452 | 542,828 | 93.6% | 76.26% | 77.18% | +0.92pp |
| 4 | 2,571,149 | 494,607 | 80.8% | 80.16% | 77.20% | **−2.96pp** |
| 5 | 161,531 | 89,056 | 44.9% | 87.22% | 80.99% | **−6.23pp** |
| 6 | 11,303 | 6,693 | 40.8% | 92.23% | 88.06% | **−4.17pp** |

**Proper cluster-bootstrap result** (the number that matters):

| d | phi (raw, nothing discarded) | n_raw | n_clusters | 95% cluster-bootstrap CI | excludes 0? |
|---|---|---|---|---|---|
| 1 | 0.33065 | 7,198 | 7,073 | [0.303, 0.357] | **YES** |
| 2 | 0.06124 | 735,367 | 328,299 | [0.059, 0.063] | **YES** |
| 3 | −0.00095 | 8,513,452 | 542,828 | [−0.0016, −0.0003] | **YES** |
| 4 | −0.02315 | 2,571,149 | 494,607 | [−0.0242, −0.0220] | **YES** |
| 5 | −0.08082 | 161,531 | 89,056 | [−0.0850, −0.0770] | **YES** |
| 6 | −0.07727 | 11,303 | 6,693 | [−0.0860, −0.0674] | **YES** |

**Conclusion: the bump is real at every tested distance, including d=3 and d=6, under a properly
constructed cluster-robust CI, at the full (undiscarded) point estimate.** The earlier "roughly
halved, overstated by 2×" conclusion from §2.2 is retracted in full. The clustering/pseudo-
replication concern from the 2026-07-28 investigation was a real thing to check, but the correct
response to it is a cluster-robust CI on the full data, not a dedup-then-bootstrap on a shrunk
sample.

**Naive (uncorrected) raw-row bootstrap, for direct comparison** — resampling raw rows directly
instead of clusters, i.e. the "textbook mistake" of pretending every pooled pair is an independent
draw:

| d | naive 95% CI (raw-row bootstrap) | proper cluster-bootstrap 95% CI |
|---|---|---|
| 1 | [0.30378, 0.35684] | [0.30336, 0.35725] |
| 2 | [0.05892, 0.06358] | [0.05888, 0.06343] |
| 3 | [−0.00156, −0.00028] | [−0.00159, −0.00026] |
| 4 | [−0.02436, −0.02200] | [−0.02422, −0.02198] |
| 5 | [−0.08411, −0.07713] | [−0.08499, −0.07696] |
| 6 | [−0.08577, −0.06804] | [−0.08598, −0.06741] |

**Surprising, worth flagging explicitly: the two CIs are nearly identical at every distance**,
despite dedup discard rates as high as 93.6% (d=3). The clustering-driven pseudo-replication
concern raised on 2026-07-28 was real and worth checking (heavy edge reuse is genuinely present,
confirmed above), but in this instance it turns out **not** to meaningfully inflate the true
variance relative to the naive estimate — the proper correction was necessary to *verify* this
safely, not skippable a priori, but the correction's answer is "the naive CI was already fine
here," not "the naive CI was overconfident." Both CIs agree: every distance's interval excludes
zero.

---

## 3. Thread B — Why does the colleague's method report "close to zero"?

### 3.1 Her script vs. production: 4 methodological gaps

A colleague independently wrote `load_slashdot.py` (original, unmodified copy: repo-root
`load_slashdot.py`; an instrumented copy with print statements and a corrected data path lives at
`scripts/paper_figures/load_slashdot.py` — verified byte-identical in all *logic*, diffed
line-by-line, only additive prints/return-values/path differ) implementing the same "correlate an
anchor edge's sign with nearby edges' signs" idea, and reported a curve close to zero everywhere —
apparently no long-range signal. Comparing her script line-by-line against the production
methodology surfaced **four** independent gaps, not one:

| # | Knob | Her convention | Production convention |
|---|---|---|---|
| 1 | **Direction** | directed BFS (successors only) | undirected BFS (both directions) |
| 2 | **Recording** | tree-discovery-edge only (one edge per newly-reached node) | all edges among every reached node |
| 3 | **Anchor sampling** | first 1,000 edges in file/loader order (deterministic) | 20,000 edges, seeded random draw |
| 4 | **Pair-cap** (found only via direct code diff, not originally suspected) | `MAX_PAIRS=10,000` per distance bucket, via **unseeded** `random.sample()` | no cap — running contingency table, full pool |

Knobs 1–3 were tested as a full 2×2×2 = 8-cell grid (`scripts/ablation_slashdot_colleague_vs_ours.py`,
results in `aaai2027/figure_data/ablation_slashdot_colleague_vs_ours.csv`) that reimplements her
exact BFS/counting logic once and swaps each factor independently, so every cell reports both NMI
and phi from the identical counting code. Knob 4 was only discovered afterward by diffing her
actual script byte-for-byte, and was never varied in the ablation grid at all — the reimplementation
always used the full, uncapped pool.

### 3.2 Full 8-cell grid (φ by distance; production = last row)

| variant | n_anchors | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 |
|---|---|---|---|---|---|---|---|---|---|
| directed/tree/small_nonrandom (**her exact method**) | 1,000 | 0.234 | 0.042 | -0.002 | -0.003 | -0.015 | -0.012 | 0.003 | -0.021 |
| directed/tree/large_random_tree | 3,000 | 0.379 | 0.054 | 0.006 | -0.004 | -0.018 | -0.016 | -0.040 | -0.018 |
| directed/tree/large_random (extra run, this session) | 10,000 | 0.385 | 0.049 | 0.007 | -0.003 | -0.017 | -0.031 | -0.022 | -0.032 |
| directed/all/small_nonrandom | 1,000 | 0.163 | 0.043 | -0.003 | -0.004 | -0.022 | -0.016 | 0.001 | -0.009 |
| directed/all/large_random_all | 20,000 | 0.319 | 0.072 | 0.001 | -0.004 | -0.018 | -0.038 | -0.049 | -0.053 |
| undirected/tree/small_nonrandom | 150 | 0.151 | 0.021 | -0.003 | -0.006 | -0.002 | -0.007 | nan | nan |
| undirected/tree/large_random_tree | 300 | 0.267 | 0.047 | 0.003 | -0.019 | **+0.019** | -0.065 | -0.024 | nan |
| undirected/all/small_nonrandom | 1,000 | 0.161 | 0.053 | -0.005 | -0.030 | -0.006 | **~0.000** | -0.018 | nan |
| **undirected/all/large_random_all (PRODUCTION)** | 20,000 | **0.321** | **0.061** | **-0.001** | **-0.022** | **-0.083** | **-0.088** | **-0.058** | **-0.028** |

Production is the only row with the real bump-and-decay shape (peaks at d6, decays by d7/d8).
Every other row is flat-to-noise from d3 onward — reproducing the "close to zero" report.

**Anchor counts are inconsistent across rows on purpose, not by oversight**: `directed`+`tree`
recording is cheap (~110s/1,000 anchors, single-threaded), affording 1,000–10,000 anchors.
`undirected`+`tree` recording is far more expensive (a bigger BFS frontier) — an early attempt at
1,000–3,000 anchors ran >87 minutes with no end in sight and had to be killed, so those two rows
are capped at 150/300 anchors, at a real cost in precision (visible in the sign-flip at d5 for
n=300, and the d6 zero-collapse for the n=1,000 `undirected/all/small_nonrandom` row). `all`-edges
recording is multiprocessed (matches the production extractor's 128-way fork pool) so it affords
the full 20,000 anchors regardless of direction.

### 3.3 Isolated per-factor influence (holding the other two factors fixed at production's values)

| Factor isolated | NMI recovery (of production's d6 value) | phi recovery (of production's d6 value) |
|---|---|---|
| **Direction** (recording=all, sampling=large_random/20k held fixed) | 11.4% → 100% (**8.75×**) | 43.1% → 100% (**2.3×**) |
| **Recording** (direction=directed, sampling=small/1000 held fixed) | 1.1% → 2.2% (**2×**) | 13.4% → 18.4% (**1.37×**) |
| **Anchor sampling** (direction=directed, recording=tree held fixed), n=1k→3k→10k | 1.1%→1.4%→6.1% | 13.4%→17.6%→34.7% |

**Direction is the dominant driver of the "fake null" by a wide margin**, under both statistics.
Every variant that keeps `direction=directed` caps out at ≤11.4% (NMI) / ≤43.1% (phi) of the real
bump regardless of which other factor is fixed to match production. The moment direction flips to
`undirected` — even with the wrong recording still in place and only 300 anchors
(`undirected/tree/large_random_tree`) — recovery jumps to 31.3% (NMI) / 74.1% (phi), the largest of
any non-production cell. There's a shape-level tell too: `directed/all/large_random_all` (direction
is the *only* wrong factor, full 20k anchors, no small-n excuse) has |phi| **still growing** through
d6→d7→d8 (0.038→0.049→0.053) instead of decaying like production (0.088→0.058→0.028) — directed
BFS doesn't just shrink the bump, it can't reproduce the peak-then-decay *shape* at all, because
successor-only reachability keeps discovering new context edges past the point where undirected
BFS has already saturated the local neighborhood.

Recording is a real but secondary factor (~1.4–2× effect). Anchor sampling/count matters, but only
at large multiples, and can't compensate for a wrong direction on its own (10,000 *directed*
anchors still lands at 6.1% NMI / 34.7% phi recovery — worse than 300 *undirected* anchors).

**NMI and phi mostly agree on ranking but disagree on magnitude**, because NMI is a nonlinear
(roughly squared-order, for small associations) function of the same underlying association while
phi is linear — the same absolute gap reads as an 8.75× fold-change in NMI but only 2.3× in phi.
Neither is "more correct"; reporting only one understates how the other reads. **Phi also carries
sign, which NMI structurally cannot** — `undirected/tree/large_random_tree` at d5 has phi=+0.019
against production's −0.083, a genuine sign disagreement that an NMI-only view (always
non-negative) would report merely as "small," hiding that it points the wrong way entirely. This is
a concrete argument for always reporting both statistics in this kind of ablation.

### 3.4 The 4th knob: her script is non-deterministic, independent of the above

Diffing her freshly-reuploaded original `load_slashdot.py` against the copy already run in this
investigation confirmed the algorithm is byte-identical (only prints/return-values/path differ) —
but a fresh run's plot didn't match a previously-saved `slashdot_sign_correlation_original.png`.
Root cause, confirmed by grep: **`random.sample()` (line 74) is called with no seed anywhere in the
file.** Whenever a distance bucket's pre-cap pair count exceeds `MAX_PAIRS=10,000` (true at every
distance except d8, where the pool is already smaller than the cap), a *different* random 10k
subsample gets drawn every run.

Direct proof (`scripts/panelb_diagnostics/demo_slashdot_cap_nondeterminism.py`): BFS/pair-collection
run once, then the capping+correlation step re-applied 5 times on the *identical* underlying pair
pool:

| d | pre-cap pairs | % kept | run0 | run1 | run2 | run3 | run4 |
|---|---|---|---|---|---|---|---|
| 1 | 226,192 | 4.4% | 0.235 | 0.232 | 0.241 | 0.220 | 0.217 |
| 2 | 5,578,021 | 0.18% | 0.057 | 0.043 | 0.050 | 0.038 | 0.043 |
| 3 | 29,438,147 | 0.034% | 0.0001 | 0.009 | 0.002 | **-0.002** | 0.005 |
| 4 | 25,036,185 | 0.04% | **0.013** | **-0.016** | **-0.016** | -0.005 | -0.012 |
| 5 | 5,975,714 | 0.17% | -0.021 | -0.017 | -0.007 | -0.022 | -0.036 |
| 6 | 730,725 | 1.4% | -0.008 | -0.012 | -0.014 | -0.016 | -0.003 |
| 7 | 69,959 | 14.3% | **0.020** | 0.007 | 0.005 | **-0.007** | 0.014 |
| 8 | 6,592 | **100% (no cap)** | -0.0214 | -0.0214 | -0.0214 | -0.0214 | -0.0214 |

d8 is bit-for-bit identical across all 5 runs (no cap ever applies there — nothing to randomize).
Every other distance moves around, with sign flips at d3/d4/d7 — the exact distances with the most
extreme discard rates. **This is a completely separate problem from §3.1–3.3**: it doesn't explain
*why* her number is small (that's the direction/recording/sampling gap), it explains why her number
is *unstable run-to-run* — a second, independent bug stacked on top of the first three.

The real fix isn't "pick a better cap size" or "seed it" (though seeding would at least make it
reproducible) — it's that **the cap only exists because her script materializes every raw pair into
a Python list before calling `numpy.corrcoef`.** Phi/MI are both computable from a running 2×2 count
table, updated incrementally as pairs are discovered, with O(1) memory regardless of how many
billions of pairs exist at a given distance (this is exactly what the production extractor and the
ablation reimplementation already do) — there is no statistical reason to ever subsample here at
all.

---

## 4. Consolidated recommendation: how to handle each knob so results stay meaningful

| Knob | Reliable choice | Why |
|---|---|---|
| **Direction** | Always undirected | "Distance" here means symmetric line-graph/shell distance. Directed-only BFS silently redefines the metric rather than just weakening it — a directed shell at nominal distance d is not the same object as an undirected shell at d. Use directed BFS only if the actual question is about directional information flow, and answer that with node-level in/out entropy (Lead 4/4c) instead of this knob. |
| **Recording** | Always "all edges among reached nodes" | Tree-discovery-only keeps exactly one sample per newly-discovered node, and which edge "wins" that slot depends on arbitrary BFS queue tie-breaking — discarding real data for a reason unrelated to the science. "All" captures every edge actually inside the distance-d shell. |
| **Anchor sampling** | Large (tens of thousands+) AND seeded random | Two requirements bundled here: size (thin anchor counts are visibly noisy at far distances — our own undirected+tree cells at n=150/300 flip sign from noise alone) and reproducibility (seeded random gets a representative sample that reruns identically; "first N in file order" risks an unmeasured selection bias tied to data-collection order). |
| **Pair pooling / cap** | Never subsample — use a running contingency table | Phi and MI are computable in O(1) memory from incremental 2×2 counts; there's no principled reason to ever materialize or cap the raw pair list. If a cap is truly unavoidable for some other reason, it must at minimum be seeded — but the better fix removes the need for a cap entirely. |
| **CI / significance on pooled pairs** | Cluster (block) bootstrap over distinct context edges, resampling whole clusters with replacement — never bootstrap a deduped/shrunk sample | Pooled (anchor, context) pairs are not independent draws — a hub context edge appears in many anchors' neighborhoods. The correct fix changes the resampling *unit* (row → cluster) without discarding any data; bootstrapping a deduped sample instead just reproduces ordinary 1/√n widening from a smaller n and additionally biases the point estimate (measured up to −6.23pp composition shift). |
| **Reporting** | Always report both phi (linear, signed) and NMI/MI (nonlinear, magnitude-only) | They agree on ranking here but disagree on magnitude, and phi catches genuine sign flips (real qualitative disagreements) that NMI's non-negativity structurally hides. |

---

## 5. Status

- **Thread A (is the bump real): RESOLVED.** Real at every tested distance (d1–d6, including the
  weakest point d3) under a properly constructed cluster-robust bootstrap on the full,
  undiscarded data. Supersedes and retracts an earlier flawed same-session conclusion.
  Checklist item #12 (`aaai2027/PEWTER_ASSETS_CHECKLIST.md`) should be updated to reflect this —
  its "fix NOT YET IMPLEMENTED" note is now stale.
- **Thread B (why did the colleague get near-zero): RESOLVED.** Fully explained by 3 systematic
  methodological gaps (direction dominant, recording secondary, anchor-count/sampling tertiary)
  plus one independent non-determinism bug (unseeded pair-cap) that doesn't bias the result but
  destroys run-to-run reproducibility.
- **Not done / possible follow-ups, not started, no user sign-off obtained**: (1) extending the
  2×2×2 grid or the cluster-bootstrap check to the other 5 datasets (only slashdot090221 was
  tested — this was a targeted debug of one external report, not a systematic sweep); (2)
  reflecting the cluster-bootstrap-confirmed bump, and/or the resolved colleague-discrepancy
  story, into `pewter_aaai.tex`'s Panel B paragraph (currently still has an inline `%%` comment
  describing the superseded degree-confound framing, per `CLAUDE.md`'s Figure 1 status note).

## 6. File map

- **Scripts** (durable, in-repo): `scripts/ablation_slashdot_colleague_vs_ours.py` (the 2×2×2 grid),
  `scripts/panelb_diagnostics/extract_panelb_raw_rows.py` (raw-pair extractor for the bootstrap
  check), `scripts/panelb_diagnostics/proper_cluster_bootstrap.py` (corrected CI method, with
  self-test), `scripts/panelb_diagnostics/demo_slashdot_cap_nondeterminism.py` (the 5-run
  non-determinism proof).
- **Data/outputs**: `aaai2027/figure_data/ablation_slashdot_colleague_vs_ours.csv` (full 8-cell
  grid), `outputs/panelb_diagnostics/` (raw pair cache `panelb_raw_rows.npy`, cluster-bootstrap
  output, her-original-script run log, the 10k-random-anchor extra run).
- **Her original script**: repo-root `load_slashdot.py` (untouched, user-provided original) and
  `scripts/paper_figures/load_slashdot.py` (instrumented copy, verified byte-identical logic).
- **Prior/background context**: `CLAUDE.md`'s "Panel B's post-minimum 'bump'" note (2026-07-28
  investigation this session's Thread A builds on), `aaai2027/PEWTER_ASSETS_CHECKLIST.md` item
  #12 (Panel B's checklist status — needs updating per §5 above).
