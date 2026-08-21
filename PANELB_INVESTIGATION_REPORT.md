# Panel B Investigation Report — MI/phi vs. line-graph distance on slashdot090221 and wiki-rfa

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

### 3.5 Her fixed ("v2") script: 2 of 4 gaps closed, 1 new gap found

The colleague produced a revised script (repo-root `load_slashdot (1).py`, untouched original;
instrumented copy `scripts/paper_figures/load_slashdot_v2.py`) that fixes 2 of the 4 gaps above —
`load_slashdot()` now builds `nx.Graph()` (undirected, fixing knob 1) and anchor selection is now
`random.sample(range(len(g.edges())), 10000)` (10,000 random anchors, fixing knob 3's count/order
problem) — while recording stays tree-only (knob 2 unfixed) and the pair-cap stays (knob 4, now at
least seeded via `random.seed(datetime.now().timestamp())`, though a *timestamp* seed is still
non-reproducible run-to-run; pinned to 42 in our copy per direct request).

**Two more bugs found before it would even run, unrelated to the seed change**, fixed only in the
copy: `depth_bar.close()` (line 67) references a variable never constructed in this version
(`NameError`), and `print(str(i+1) + " " + correlations[i])` concatenates a `str` with a `float`
(`TypeError`). Neither changes any computed result — both are pure crash-preventing fixes.

**Performance finding, not just a correctness one**: her (and v2's) inner loop calls
`list(g.edges(data=True))[edge_id]` *inside* the per-anchor for-loop — rebuilding the full O(E) edge
list from scratch on every one of the 10,000 iterations (~5 billion redundant tuple constructions
across a full run on this graph's 500,481 edges). A first sequential timing run measured ~1.4s/anchor
→ tqdm's own ETA was **~3h50m** for the full 10,000 anchors — this was assumed at the time to be
"undirected BFS is just expensive" (consistent with the earlier `undirected+tree` cost finding in
§3.2), but hoisting the edge-list construction out of the loop (a one-line, zero-semantic-change fix)
plus farming the now-cheap per-anchor BFS across a fork-based multiprocessing pool
(`scripts/paper_figures/load_slashdot_v2_parallel.py`, same globals-before-`Pool()` pattern as
`ablation_slashdot_colleague_vs_ours.py`) brought the full 10,000-anchor run down to **199 seconds**
— confirming the redundant list rebuild, not BFS cost, was the dominant factor. Randomness ordering
is preserved to match a sequential run: `random.seed(42)` and the anchor `random.sample()` call both
happen in the main process before any forking, and the `MAX_PAIRS` capping step also stays
sequential in the main process, after all workers finish (see caveat in §3.8).

### 3.6 Rigorous proof: distance definitions are identical once both are undirected

Formal argument: standard unweighted multi-source BFS has an order-independent invariant — every
node gets its true shortest-path distance from the source set at first discovery, regardless of
queue vs. level-synchronized traversal. For any discovery/tree edge `(a,b)` where `b` is newly
reached via already-visited `a`: her label is `d[b] = d[a]+1`; production's label is
`min(shell(a),shell(b))+1`. Since `a` was visited first, `shell(a) < shell(b)`, so
`min(shell(a),shell(b)) = shell(a)`, and both formulas reduce to `shell(a)+1 = shell(b)`. This holds
for *any* graph, *any* traversal order, and *any* choice of parent among ties — every valid
discovery parent of `b` is, by the same invariant, necessarily at exactly `shell(b)-1`.

Empirically confirmed at scale, `scripts/panelb_diagnostics/verify_distance_definitions.py`
(`outputs/panelb_diagnostics/verify_distance_definitions_output.txt`): (1) our own shell computation
vs. networkx's `single_source_shortest_path_length` ground truth — 1,232,099 node-shells checked
across 15 anchors, **0 mismatches**; (2) her per-edge label vs. `min(shell)+1` — **500 anchors,
41,068,944 recorded edges checked, 0 mismatches**, including 17,956,325 edges (44%) where the
discovered node had more than one tied possible discovery parent, directly confirming the proof's
claim that the label doesn't depend on which parent BFS happens to pick. **Conclusion: distance was
never a real source of disagreement, in either her v1 or v2 script, once direction matches.** The
remaining daylight is entirely about which edges get counted (recording, §3.1 knob 2) and the pair
cap (§3.4), not how distance is measured.

### 3.7 A 5th gap, found only when asked directly: reciprocal-edge handling

`nx.Graph()` (her v2 default loader) silently **merges** reciprocal edge pairs — both `(u,v)` and
`(v,u)` present as separate lines in the raw edgelist — into a single edge. `graph.add_edge(v2, v1,
sign=...)` on a pair that already has an edge `(v1,v2)` overwrites the existing edge's `sign`
attribute in place, keeping only whichever direction was parsed **last** in the file.

Quantified (`scripts/panelb_diagnostics/check_reciprocal_edge_handling.py`,
`outputs/panelb_diagnostics/check_reciprocal_edge_handling_output.txt`): of 549,202 raw directed
edges, 48,721 are reciprocal pairs (both directions present). `nx.Graph()` collapses these down to
exactly 500,481 edges (matching the count of distinct *unordered* node pairs — full collapse
confirmed). Of the 48,721 pairs, 96.0% (46,772) agree in sign — harmless merge there — but **4.0%
(1,949 pairs, 3,898 edges) disagree**, meaning one real, distinct sign observation is silently and
irrecoverably discarded at graph-construction time, before BFS or recording even runs.

This is genuinely a 5th independent axis, not reducible to direction/recording/sampling/cap:

| Loader | Reciprocal pairs merged? | Notes |
|---|---|---|
| Production (array-based `full[u]`/`full[v]` or `succ[u]` adjacency, either direction) | **Never** | Every one of the 549,202 directed edges keeps its own id and sign permanently; confirmed `dg.number_of_edges() == 549,202` for the directed-adjacency equivalent. For the undirected case both directions of a pair contribute two independent observations when encountered; for the directed case both are preserved but only discoverable if BFS reaches the edge's *source* endpoint. |
| Her v1 script, `nx.DiGraph()` | **Never** (indexes by ordered `(u,v)`) | Confirmed: 549,202 edges, matching the raw line count exactly. No merge-loss, but tree-recording (§3.1 knob 2) independently means most reciprocal edges — like most cross/back edges generally — never get sampled as context edges regardless. |
| Her v2 script (undirected), `nx.Graph()` | **Yes — the new finding** | 549,202 → 500,481 edges; 3,898 edges' worth of information (the disagreeing 4%) permanently lost at load time, stacked on top of the same tree-recording loss as v1. Two independent loss mechanisms, not one. |
| Her v2-directed variant (presumed `nx.DiGraph()` swap, her own follow-up experiment, §3.9) | **Never**, if `DiGraph()` was used | Same as v1 — avoids the merge loss that the *undirected* v2 default introduced as an unintended side effect of fixing knob 1. |

**Ironic finding worth flagging explicitly**: fixing the direction gap (switching `DiGraph()` →
`Graph()`) introduced this new gap as a side effect — `nx.Graph()`'s implicit "adding an edge that
already exists just updates its attributes" semantics silently drops information that `DiGraph()`
never touched, precisely because `DiGraph()` treats `(u,v)` and `(v,u)` as different edges by
construction.

### 3.8 Cross-machine reproducibility: same seed, different results, and why

The colleague ran her v2-family script (undirected and a directed variant) on her own machine with
what should be equivalent settings (seed=42, 10,000 random anchors, `MAX_PAIRS=10,000` cap) — her
numbers are close to, but not identical to, our parallelized run of the same nominal configuration
(§3.9 table). Likely cause, identified but not yet fixed: our parallel script uses
`pool.imap_unordered` for speed, which returns chunk results in **completion order**, not submission
order. The anchor *selection* is unaffected (that random draw happens in the main process before any
forking), but the **order** that (anchor,context) pairs get appended into each distance's list is
scheduler-dependent — and since `MAX_PAIRS`'s cap draws `random.sample(range(len(x)), 10000)` (an
*index* draw), the same seeded indices land on different actual pairs depending on append order.
This plausibly explains why the distances with the most extreme discard ratios (d3/d4/d7) diverge
the most between our run and hers, while d1/d5/d6/d8 (much less discarding) land close. **Not yet
fixed**: switching `imap_unordered` → `imap` (ordered) in
`scripts/paper_figures/load_slashdot_v2_parallel.py` would make chunk-merge order match a sequential
run's order exactly (chunks are dispatched in `edge_numbers` order and each chunk's own anchors are
processed in order internally), which should make our run byte-for-byte reproducible against any
sequential run using the same seed and edge-file line order. Flagged as a known, understood,
low-effort fix if exact cross-run reproducibility is ever needed again — not applied this session.

### 3.9 Master comparison across every variant tried

φ (phi) by distance, every configuration run in this investigation:

| variant | n_anchors | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 |
|---|---|---|---|---|---|---|---|---|---|
| **PRODUCTION** (undirected/all, no cap) | 20,000 | **0.321** | **0.061** | **-0.001** | **-0.022** | **-0.083** | **-0.088** | **-0.058** | **-0.028** |
| Production, directed only (all, no cap) | 20,000 | 0.319 | 0.072 | 0.001 | -0.004 | -0.018 | -0.038 | -0.049 | -0.053 |
| Her v1 exact (directed/tree, `DiGraph`) | 1,000 | 0.234 | 0.042 | -0.002 | -0.003 | -0.015 | -0.012 | 0.003 | -0.021 |
| Her v1, more anchors (directed/tree, `DiGraph`) | 3,000 | 0.379 | 0.054 | 0.006 | -0.004 | -0.018 | -0.016 | -0.040 | -0.018 |
| Her v1, more anchors still (directed/tree, `DiGraph`) | 10,000 | 0.385 | 0.049 | 0.007 | -0.004 | -0.017 | -0.031 | -0.022 | -0.032 |
| Her v2, undirected/tree, cap=10k, seed=42, `Graph` (**our** machine, parallel/`imap_unordered`) | 10,000 | 0.338 | 0.038 | 0.022 | -0.027 | -0.057 | -0.058 | -0.055 | -0.036 |
| Her v2, undirected/tree, cap=10k, seed=42, `Graph` (**her** machine) | 10,000 | 0.327 | 0.040 | 0.007 | -0.033 | -0.058 | -0.062 | -0.044 | -0.034 |
| Her v2-variant, directed/tree, cap=10k, presumed `DiGraph` (**her** machine) | 10,000 | 0.378 | 0.045 | 0.023 | 0.011 | -0.008 | -0.032 | -0.030 | -0.018 |

Reading, from most to least aligned with production: fixing direction+sampling (her v2 undirected,
both machines) gets close on d1/d5/d6/d8 and reproduces the right overall bump-then-decay shape, but
still sits at 60-70% of production's magnitude at d2/d5/d6 and disagrees in sign at d3 — attributable
to the still-unfixed recording gap (§3.1) plus residual cap-driven noise (§3.4/§3.8), not to distance
(§3.6, ruled out) or reciprocal-edge handling alone (§3.7; the undirected v2 variant's extra
reciprocal-merge loss is a real but second-order contributor next to recording).

**This entire thread (§3) is closed as of this update. No config, sampler, or Figure 1 change
resulted from it — production's existing undirected/all/no-cap convention remains canon,** exactly as
before this investigation started; the value of this thread was fully explaining an external
discrepancy report, not discovering anything wrong with production itself.

---

## 3b. Thread C (2026-08-20) — does a dataset property or a compositional-selection
mechanism explain the bump, for the two specific instances the user asked about:
slashdot090221 undirected (hops 4–8) and wiki-rfa directed (hops 7–8)

Threads A and B above establish that the bump is *real* (not a bug, not pure noise) and
explain one external near-zero *reproduction* failure. Neither one explains *why* the bump
happens mechanistically. This thread tries several candidate mechanisms for both requested
instances, honestly reporting what holds up and what doesn't — no forced narrative.

### 3b.1 Is wiki-rfa's directed d7–8 bump real in the first place? (never checked before this thread)

Every check in Threads A/B above was slashdot090221-only. Ran the same shuffle-signs null
control (`extract_empconf_panelB_mi_decay_linegraph.py --direction directed --shuffle-signs`,
full production scale, all 177,211 edges as anchors, d_max=7):

| d | real NMI | shuffled-null NMI | real / null |
|---|---|---|---|
| 7 | 0.002715 | 0.0000604 | **45×** |
| 8 | 0.026687 | 0.0000089 | **3003×** |

**Real, by a wide margin at both distances** — same order-of-magnitude-or-more-above-noise
standard Thread A used for slashdot. This is the first confirmation this specific bump
(wiki-rfa, directed, tail distances) isn't a sampling/estimator artifact.

### 3b.2 Hub-edge repetition: real for slashdot, NOT the story for wiki-rfa

Thread A found slashdot's outer shell dominated by a small number of repeated hub edges
(top-10 distinct edges ≈ 40% of all pooled pairs at the farthest distance). Ran the
equivalent measurement for wiki-rfa directed at d∈{7,8} (new script,
`scripts/panelb_diagnostics/wikirfa_directed_hub_check.py`, full 177,211-anchor sweep,
~4.2hr single-threaded — log: `outputs/panelb_diagnostics/wikirfa_directed_hub_check.log`):
69,288 distinct context edges hit, **top-10 account for only 2.3%** of the 271,341 pooled
pairs. **Wiki-rfa's tail is not hub-edge-repetition-dominated the way slashdot's is** — a
real, dataset-specific structural difference, not the same mechanism recurring.

### 3b.3 Sign-composition shift: real in both, opposite direction

New check (not attempted in Threads A/B): does the *sign mix* of outer-shell context edges
differ from the dataset's overall sign balance?

**Slashdot090221 (undirected), from the cached raw-pair extractor's d=1..6 data**
(`outputs/panelb_diagnostics/panelb_raw_rows.npy`, reused, no rerun needed):

| d | n_pairs | % positive (pair-weighted) | mean endpoint degree (distinct ctx. edges) |
|---|---|---|---|
| 1 | 7,198 | 79.7% | 394.0 |
| 4 | 2,571,149 | 80.2% | 140.3 |
| 5 | 161,531 | 87.2% | 89.9 |
| 6 | 11,303 | **92.2%** | 27.8 |

(dataset-wide baseline: 77.4% positive, mean degree 13.4)

Sign composition trends sharply **more positive** with distance through the bump range, in
lockstep with a sharp **drop in endpoint degree**.

**Wiki-rfa (directed), from the new full-scale hub-check above:**

| | % positive | mean outdeg(source) | mean indeg(target) |
|---|---|---|---|
| context edges at d∈{7,8} | **68.2%** (pair-wt.), 59.5% (distinct) | 177.9 | 75.8 |
| dataset-wide baseline | 78.4% | 15.7 | 15.7 |

Sign composition trends sharply **more negative** with distance (opposite direction from
slashdot), in lockstep with a sharp **rise** in source out-degree (opposite direction from
slashdot's degree trend too).

### 3b.4 Does raw degree alone explain either shift? Checked directly — no, not fully, in either case

The natural next question: is the sign shift just "degree correlates with sign, and the
outer shell is degree-selected"? Checked with a dataset-wide degree-decile breakdown of
%positive (`degree_vs_sign_check.py`, one-off, not yet moved into `scripts/`), independent of
the BFS-shell sampling:

- **Slashdot**: %positive vs. edge endpoint degree is **U-shaped**, not monotonic — 88.5%
  positive at the lowest-degree decile (mean deg 9.4), dropping to 69.8% at mid-degree (deg
  121.5), then **rising back to 86.4%** at the highest-degree decile (deg 622). At the outer
  shell's actual mean degree (~27.8, between decile 1 and 2), the dataset-wide baseline
  predicts only ~78–81% positive — the observed 92.2% is well above that. **Degree alone
  under-predicts the shift; something beyond raw degree is selecting for unusually positive
  edges at the outer shell.**
- **Wiki-rfa**: %positive vs. source out-degree is **weakly monotonically increasing**
  (77.7% at the lowest decile → 82.2% at the highest) — the *opposite* direction from the
  outer shell's actual 68.2%. Degree-matched edges (outdeg≈178, between decile 6 and 7)
  predict ~79–80% positive; the observed 68.2% is far below that, and on the wrong side of
  the dataset-wide baseline entirely. **Degree does not explain this shift either — if
  anything, it points the wrong way.**

This matches and sharpens Thread A's own earlier finding for slashdot ("a direct
degree-filter pilot on bitcoin-alpha left the bump essentially unchanged... hub effect is a
correlate, not sufficient by itself") — now confirmed with a cleaner, quantitative decile
comparison, and shown to hold (differently) for wiki-rfa's directed tail too.

### 3b.5 Honest conclusion for this thread

Both requested bump instances are **real** (not artifacts — confirmed for wiki-rfa here for
the first time, reconfirmed for slashdot via existing Thread A results) and both show a
**genuine, measurable sign-composition shift** at the bump distances relative to each
dataset's overall sign balance. But:

- The shift runs in **opposite directions** (slashdot's outer shell skews positive;
  wiki-rfa's skews negative) and is accompanied by **opposite degree trends** (slashdot's
  outer shell is unusually low-degree; wiki-rfa's is unusually high-out-degree) — there is no
  single shared mechanism across the two instances, only a shared *shape of finding*
  (distance-selects a compositionally distinct, non-representative sub-population of edges).
- Neither shift is fully explained by degree alone — both leave a real residual once degree
  is accounted for (slashdot: more positive than its low degree predicts; wiki-rfa: more
  negative than its high degree predicts). What that residual selection mechanism actually
  is (topological role — e.g. "pendant"/tree-like edges vs. cycle-heavy ones — a specific
  community-structure effect, or something dataset-specific like wiki-rfa's known
  admin-candidate concentration) is **not pinned down** by this thread.
- **No forced explanation is offered beyond this.** Per the standing instruction that a
  clean negative/partial result is an acceptable outcome here: this thread establishes *that*
  a real, non-degree-reducible compositional selection effect drives both bump instances, and
  *that* the two datasets' effects point in opposite directions — without claiming to have
  found the underlying generative mechanism for either.

**Not done / possible follow-ups, no sign-off obtained**: (1) a genuine causal test — filter
out the top out-degree/low-degree context edges and see whether the residual bump survives
(the direct analogue of Thread A's bitcoin-alpha degree-filter pilot, not yet run for either
requested instance); (2) a topological-role check (pendant/bridge edges vs. cycle-embedded
edges) as a candidate for the unexplained residual; (3) for wiki-rfa specifically, checking
whether the negative-skewed outer-shell edges concentrate on the known admin-candidate nodes
(CLAUDE.md's Lead4c section already documents these as unusually high-in-degree, genuinely
mixed-sign nodes) — plausible given the out/in-degree elevation already found, but not
directly tested. Any of these would need a fresh sign-off given the ~4hr single-threaded cost
of the wiki-rfa full-scale sweep already run in this thread (a parallelized version, mirroring
the production extractor's multiprocessing pattern, would very likely be far cheaper if this
is picked up again).

## 4. Consolidated recommendation: how to handle each knob so results stay meaningful

| Knob | Reliable choice | Why |
|---|---|---|
| **Direction** | Always undirected | "Distance" here means symmetric line-graph/shell distance. Directed-only BFS silently redefines the metric rather than just weakening it — a directed shell at nominal distance d is not the same object as an undirected shell at d. Use directed BFS only if the actual question is about directional information flow, and answer that with node-level in/out entropy (Lead 4/4c) instead of this knob. |
| **Recording** | Always "all edges among reached nodes" | Tree-discovery-only keeps exactly one sample per newly-discovered node, and which edge "wins" that slot depends on arbitrary BFS queue tie-breaking — discarding real data for a reason unrelated to the science. "All" captures every edge actually inside the distance-d shell. |
| **Anchor sampling** | Large (tens of thousands+) AND seeded random | Two requirements bundled here: size (thin anchor counts are visibly noisy at far distances — our own undirected+tree cells at n=150/300 flip sign from noise alone) and reproducibility (seeded random gets a representative sample that reruns identically; "first N in file order" risks an unmeasured selection bias tied to data-collection order). |
| **Pair pooling / cap** | Never subsample — use a running contingency table | Phi and MI are computable in O(1) memory from incremental 2×2 counts; there's no principled reason to ever materialize or cap the raw pair list. If a cap is truly unavoidable for some other reason, it must at minimum be seeded — but the better fix removes the need for a cap entirely. |
| **CI / significance on pooled pairs** | Cluster (block) bootstrap over distinct context edges, resampling whole clusters with replacement — never bootstrap a deduped/shrunk sample | Pooled (anchor, context) pairs are not independent draws — a hub context edge appears in many anchors' neighborhoods. The correct fix changes the resampling *unit* (row → cluster) without discarding any data; bootstrapping a deduped sample instead just reproduces ordinary 1/√n widening from a smaller n and additionally biases the point estimate (measured up to −6.23pp composition shift). |
| **Reporting** | Always report both phi (linear, signed) and NMI/MI (nonlinear, magnitude-only) | They agree on ranking here but disagree on magnitude, and phi catches genuine sign flips (real qualitative disagreements) that NMI's non-negativity structurally hides. |
| **Reciprocal edges** | Never merge — keep both `(u,v)` and `(v,u)` as independent, separately-signed observations | `nx.Graph()`'s implicit "adding an existing edge just updates it" semantics silently overwrites one direction's sign with the other's; confirmed 4.0% of real reciprocal pairs in this dataset disagree in sign, so this is a genuine, not merely redundant, information loss. Array-based adjacency (or `nx.DiGraph()`, or a `nx.MultiGraph()`) avoids this by construction. |

---

## 5. Status

- **Thread A (is the bump real): RESOLVED.** Real at every tested distance (d1–d6, including the
  weakest point d3) under a properly constructed cluster-robust bootstrap on the full,
  undiscarded data. Supersedes and retracts an earlier flawed same-session conclusion.
  Checklist item #12 (`aaai2027/PEWTER_ASSETS_CHECKLIST.md`) should be updated to reflect this —
  its "fix NOT YET IMPLEMENTED" note is now stale (checklist item renumbered/rewritten since;
  see `aaai2027/PEWTER_ASSETS_CHECKLIST.md`'s "Contributions, line 94" row).
- **Thread B (why did the colleague get near-zero): FULLY RESOLVED, including two follow-up rounds
  after the original 4-gap ablation.** Root-caused to 5 independent gaps total: direction (dominant),
  recording (secondary, still unfixed even in her revised script), anchor sampling (tertiary), an
  unseeded/non-reproducible pair-cap (§3.4/§3.8, orthogonal to the bias question — affects
  reproducibility, not the point estimate's direction), and a newly-found reciprocal-edge-merge bug
  specific to `nx.Graph()` (§3.7). Also proved rigorously (formal argument + 41M-edge empirical
  check, §3.6) that distance itself was never a real disagreement between the two methods once
  direction matches. Her revised ("v2") script fixes 2 of the 5 gaps (direction, sampling) and gets
  substantially closer to production as a result (§3.9), while introducing the reciprocal-edge issue
  as a side effect of its direction fix.
- **Explicitly confirmed with the user (this update): no config, sampler, or Figure 1 change results
  from any of this.** Production's undirected/all-edges/no-cap convention stays canon exactly as
  before the investigation started. This entire thread is closed — its value was fully explaining an
  external discrepancy report, not surfacing a defect in production.
- **Not done / possible follow-ups, not started, no user sign-off obtained**: (1) extending the
  2×2×2 grid or the cluster-bootstrap check to the other 5 datasets (only slashdot090221 was
  tested — this was a targeted debug of one external report, not a systematic sweep); (2)
  reflecting the cluster-bootstrap-confirmed bump, and/or the resolved colleague-discrepancy
  story, into `pewter_aaai.tex`'s Panel B paragraph (currently still has an inline `%%` comment
  describing the superseded degree-confound framing, per `CLAUDE.md`'s Figure 1 status note); (3)
  the `imap_unordered` → `imap` reproducibility fix in `load_slashdot_v2_parallel.py` (§3.8), low
  effort, not applied since exact cross-run matching wasn't required for the conclusions drawn.
- **Thread C (dataset-property / compositional-selection mechanism, both requested bump
  instances): CLOSED as a partial/honest-negative result, per explicit user permission that "it's
  ok if we don't find anything."** Both instances (slashdot090221 undirected d4–8, wiki-rfa
  directed d7–8) confirmed real (the latter for the first time, via shuffle-null, §3b.1); ruled
  out hub-edge-repetition as wiki-rfa's mechanism (only 2.3% top-10 share, vs. slashdot's ~40%,
  §3b.2); found real, opposite-direction sign-composition shifts in both (§3b.3) that are **not**
  fully explained by raw degree in either case (§3b.4) — a genuine residual left unexplained,
  not forced into a single narrative. See §3b.5 for the three concrete, not-yet-run follow-ups
  that could sharpen this further, none launched without a fresh sign-off given the ~4hr cost of
  the wiki-rfa full sweep.

## 6. File map

- **Scripts** (durable, in-repo): `scripts/ablation_slashdot_colleague_vs_ours.py` (the 2×2×2 grid),
  `scripts/panelb_diagnostics/extract_panelb_raw_rows.py` (raw-pair extractor for the bootstrap
  check), `scripts/panelb_diagnostics/proper_cluster_bootstrap.py` (corrected CI method, with
  self-test), `scripts/panelb_diagnostics/demo_slashdot_cap_nondeterminism.py` (the 5-run
  non-determinism proof), `scripts/panelb_diagnostics/verify_distance_definitions.py` (§3.6's
  formal-proof + 41M-edge empirical check), `scripts/panelb_diagnostics/check_reciprocal_edge_handling.py`
  (§3.7's reciprocal-merge quantification), `scripts/panelb_diagnostics/wikirfa_directed_hub_check.py`
  (§3b.2/§3b.3's full-scale wiki-rfa directed hub-concentration + sign/degree profile at
  d∈{7,8}, single-threaded, ~4.2hr — see its own log before rerunning),
  `scripts/panelb_diagnostics/slashdot_sign_composition_check.py` (§3b.3's slashdot sign/degree
  profile, reuses the cached `panelb_raw_rows.npy`, seconds to run),
  `scripts/panelb_diagnostics/degree_vs_sign_check.py` (§3b.4's dataset-wide degree-decile vs.
  sign check for both datasets, seconds to run).
- **Her scripts**: repo-root `load_slashdot.py` and `load_slashdot (1).py` (both untouched,
  user-provided originals — v1 and v2 respectively), `scripts/paper_figures/load_slashdot.py`
  (instrumented copy of v1, byte-identical logic), `scripts/paper_figures/load_slashdot_v2.py`
  (instrumented copy of v2 — seed pinned to 42, two crash bugs fixed, see §3.5),
  `scripts/paper_figures/load_slashdot_v2_parallel.py` (parallelized v2, 199s vs. the sequential
  version's ~4hr ETA — see §3.5 and the `imap_unordered` reproducibility caveat in §3.8).
- **Data/outputs**: `aaai2027/figure_data/ablation_slashdot_colleague_vs_ours.csv` (full 8-cell
  grid), `outputs/panelb_diagnostics/` (raw pair cache `panelb_raw_rows.npy`, cluster-bootstrap
  output, her-original-script run log, the 10k-random-anchor extra runs, `her_v2_seed42_parallel.csv`,
  `her_v2_her_machine_results.csv`, `verify_distance_definitions_output.txt`,
  `check_reciprocal_edge_handling_output.txt`, `wikirfa_directed_hub_check.log`,
  `wikirfa_directed_shuffled_nmi.csv`), repo-root `slashdot_sign_correlation_v2_seed42_parallel.png`.
- **Prior/background context**: `CLAUDE.md`'s "Panel B's post-minimum 'bump'" note (2026-07-28
  investigation this session's Thread A builds on), `aaai2027/PEWTER_ASSETS_CHECKLIST.md` item
  #12 (Panel B's checklist status — needs updating per §5 above).
