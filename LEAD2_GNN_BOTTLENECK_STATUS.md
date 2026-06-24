# Lead 2: GNN Bottleneck Investigation — Status Report (handoff)

**Status:** Original scope (Steps 0–6) COMPLETE. Steps 0–3 fully done, all 6
datasets, both model types. Step 4 (synthetic stress-test graph) was
calibrated successfully but Step 5's GINEConv trial run revealed a
fundamental confound (fixed-random vs. trainable node features) that the
user decided not to work around — Steps 4/5 abandoned by user decision on
2026-06-22, real-data findings (Steps 0–3) stand as the deliverable. Step 6
(consolidated summary) written: `outputs/lead2_gnn_bottleneck/SUMMARY.txt`.
Step 2's relay-MI numbers were later corrected for a walk-distance-vs-true-
BFS-distance bug (see Step 2 below) — table updated, narrative revised.

**NEW follow-up phase (in progress, started 2026-06-23):** user proposed a
weight-vs-importance diagnostic — do GNN-style attention/aggregation
weights actually track which neighbors carry genuinely important
information, or can a node get under-weighted despite mattering a lot
("good information flowing through bad pipes")? This required a literal
attention-weight baseline (GAT-style), so SiGAT was integrated both as (a) a
new comparable baseline on canonical splits and (b) the attention-weight
probe source. See "SiGAT baseline + attention-weight diagnostic" section
below for status.

**Governing plan document:** `~/.claude/plans/read-claude-plans-plan-research-leads-md-delegated-cook.md`
(the canonical, detailed plan — context, exact formulas, file-by-file design.
Read that first for full rationale; this file is the status/results layer on
top of it.) Parent context: `~/.claude/plans/plan-research-leads.md`.

**The question Lead 2 asks:** GNNs compress a 2-hop node's info through a
fixed-size embedding at the 1-hop intermediate node (`h_v^(1)`) before it can
reach the target edge's prediction. The walk-transformer has no such forced
compression (every token stays attention-visible). Does this bottleneck
explain part of the walk model's AUC advantage over GNN baselines?

**Standing instructions from the user (don't relitigate):**
- GNN testbeds: GINEConv + SignedGCN/CSG only (skip attention-based baselines
  — their aggregation already resembles attention, confounding the comparison).
- CSG here = `torch_geometric.nn.models.SignedGCN` (literally PyG's built-in
  class, `SignedConv` layers, mean aggregation) wrapped in the repo's existing
  curriculum-learning training schedule (`Graph.edge_score()` balance-theory
  sort + `train_scheduler`). The curriculum complexity is training-only and
  doesn't touch the encoder we hook into for `h_v^(1)` — confirmed fine to use
  as-is rather than writing a "plain SignedGCN" pipeline from scratch.
- MI metric: PCA(~5 comp) + 5-bin percentile binning + joint-histogram MI,
  reusing `node_mi_structural_embedding.py`'s method (NOT
  `edge_sign_mi_vs_distance_v3.py`'s contingency-table approach — that one is
  hardcoded to binary ±1 and measures a different, node-personality-level
  quantity; see Step 4 below for why this distinction matters).
- Every new measurement script must validate end-to-end on bitcoin-alpha
  first, then run (and be reported) across all 6 real datasets individually
  — never average across datasets, always show per-dataset rows.
- Oracle-GNN ablation (skip connection bypassing the 1-hop relay) was to be
  built in this same pass, not deferred.
- Synthetic "inverted fog" graph (user's idea): a graph with
  MI(sign,d=1)≈0 but MI(sign,d=2) substantial — opposite of every real
  dataset — as a stress test for whether baseline GNN specifically
  underperforms when 2-hop signal is the *only* signal.
- Step 1b (per-edge sensitivity) was added mid-session per the user's own
  question: "we measure mainly bottleneck-ness of nodes, but what about
  edges — is there a way to directly find if some edge is a bottleneck?"

**Environments (easy to get wrong, has bitten us twice):**
- `.venv/bin/python3` (this repo's own venv) — has numpy/sklearn/networkx/
  omegaconf/pytorch_lightning/torch 2.7.1. Use for walk-transformer scripts
  and the MI utility scripts (`lead2_gnn_bottleneck_mi.py`,
  `lead2_walk_relay_mi.py`, `generate_synthetic_fog_graph.py`,
  `edge_sign_mi_vs_distance_v3.py`).
- `/home/eng/shilo_avital/.conda/envs/directed_gnn/bin/python` — has
  `torch_geometric`, lacks omegaconf/pytorch_lightning. Use for ALL
  GNN-baseline scripts (`baselines/GINEConv/*.py`, `baselines/CSG/*.py`,
  `lead2_edge_sensitivity.py` — this one imports torch_geometric despite
  living in `scripts/`, easy to forget).

---

## Step 0 — MI utility (DONE)

`scripts/mi_pca_binning_utils.py` — `mi_pca_bins(anchor_embeddings,
context_values, n_pca=5, n_bins=5)` → `{"mi","nmi","n_pairs","best_component",
"mi_per_component","n_components"}`. Validated: recovers bitcoin-alpha's known
d=1 contingency-table MI (0.00078178 bits) within tolerance.

Key bugfix: percentile-binning degenerated to 1 bin on binary (±1) inputs.
Fixed via `_global_bins()`'s lossless exact-bin shortcut when
`len(unique values) <= n_bins`.

## Step 1 — `h_v^(1)` bottleneck MI vs. ceiling (DONE, all 6 × both models)

Script: `scripts/lead2_gnn_bottleneck_mi.py --datasets <...> --models GINEConv CSG`.

**Pairing (corrected mid-session, see plan doc's Context section for the full
reasoning):** for each node `v`, pair `h_v^(1)` against the signs of `v`'s own
direct out-edges — NOT `v`'s 2-hop frontier. Ceiling = `H(sign) ≈ 1 bit`.

Results (`outputs/lead2_gnn_bottleneck/h1_mi_report.txt`):

| dataset | model | mi (bits) | nmi | null_nmi |
|---|---|---|---|---|
| bitcoin-alpha | CSG | 0.0781 | 0.1661 | 0.0054 |
| bitcoin-alpha | GINEConv | 0.0916 | 0.1947 | 0.0034 |
| bitcoin-otc | CSG | 0.0918 | 0.1496 | 0.0026 |
| bitcoin-otc | GINEConv | 0.1647 | 0.2686 | 0.0026 |
| epinions | CSG | 0.0618 | 0.0938 | 0.0005 |
| epinions | GINEConv | 0.1368 | 0.2075 | 0.0012 |
| wiki-elec | CSG | 0.0129 | 0.0171 | 0.0003 |
| wiki-elec | GINEConv | 0.0919 | 0.1215 | 0.0006 |
| wiki-rfa | CSG | 0.0061 | 0.0080 | 0.0005 |
| wiki-rfa | GINEConv | 0.0991 | 0.1294 | 0.0002 |
| slashdot090221 | CSG | 0.0754 | 0.0950 | 0.0003 |
| slashdot090221 | GINEConv | 0.1358 | 0.1711 | 0.0004 |

**Intermediate conclusion:** both models retain real (well above
permutation-null), but far from lossless (NMI 0.008–0.27), neighbor-sign
information in `h_v^(1)`. Consistent direction across all 6 datasets for both
models → genuine bottleneck signature, not a single-dataset artifact. GINEConv
consistently retains more than CSG on every dataset (sum-aggregation +
nonlinear MLP apparently preserves more than CSG's mean-aggregation
`SignedConv`). wiki-elec/wiki-rfa show the weakest retention for both models.

## Step 1b — Per-edge sensitivity (DONE, all 6 × both models)

Script: `scripts/lead2_edge_sensitivity.py --datasets <...> --models GINEConv CSG`
(**must use the `directed_gnn` conda env**, not `.venv` — it imports
`torch_geometric`).

Two exact, O(1)-per-edge scores reusing Step 1's cached `h_v^(1)` checkpoints
(no retraining): `contribution_share` (pre-MLP, fraction of the aggregate sum
contributed by one edge's message) and `leave_one_out_delta_relative`
(post-MLP, `||h_v^(1) - h_v^(1) without this edge|| / ||h_v^(1)||` — the
"relative" normalization was added after discovering the raw delta grows with
degree due to sum-aggregation magnitude inflation, not dilution, which was
backwards from the hypothesis).

GINEConv result, all 6 datasets, in `outputs/lead2_gnn_bottleneck/edge_sensitivity_report.txt`:
**`contribution_share` decreases sharply/monotonically with out-degree bucket
on every dataset** (confirms dilution — a high-degree node's specific edge
contributes a shrinking share of the aggregate). `mean_delta_relative` stays
roughly flat or rises mildly with degree (the nonlinear MLP partially
counteracts the linear pre-MLP dilution) — also consistent across all 6.

**CSG result, all 6 datasets, found a real bug (not an environment issue):**
the rerun crashed on bitcoin-otc with a tensor shape mismatch inside
`full_h1()` (scripts/lead2_edge_sensitivity.py ~line 195) — the residual term
`conv1.lin_pos_r(x)` used the FULL node-feature matrix `x` even when called
with already-per-edge (`dst`-indexed) aggregate tensors, instead of `x[dst]`.
This only "worked" by accident when num_nodes happened to equal the edge
count being passed in (never, in general — it silently produced a shape
error the first time the script ran on a dataset where the mismatch was
exposed). Fixed by passing `x[dst_pos]`/`x[dst_neg]` explicitly. After the
fix, all 6 datasets confirmed the same monotonic dilution pattern as
GINEConv. Two CSG artifacts (epinions, slashdot090221) also turned out to be
missing `state_dict` (see Step 3 below for why) — retrained, then rerun
through the fixed script. Final results in
`outputs/lead2_gnn_bottleneck/edge_sensitivity_report.txt`, summarized in
`SUMMARY.txt`.

## Step 2 — Walk-transformer relay-token MI (DONE, bug fixed, all 6 rerun)

Script: `scripts/lead2_walk_relay_mi.py`. Plain `register_forward_hook` on
each `nn.TransformerEncoderLayer`; for a masked target edge at position `i`,
the 1-hop relay node sits at `i±1`, the far edge whose sign we test at `i±3`.

**Bug (user-caught):** `i±3` is a WALK-token-position offset, not a verified
true graph-BFS distance. The R→W edge itself (`i±2,i±3`) is always a real
graph edge of R (the walk only ever steps along real edges), so that part of
the pairing is sound — but random walks can backtrack/revisit, so W (the
node at `i±3`) is not guaranteed to be a *fresh* 2-hop frontier node from the
masked edge's far endpoint `u` (at `i∓1`). If the walk backtracks (W == u,
or any node already visited earlier in the walk), the model is effectively
being asked "does R's hidden state retain the very edge we just masked"
(near-trivial, since masked-edge-recovery is the model's primary training
objective) rather than "does R's hidden state retain genuinely new 2-hop
information" — this can inflate the reported NMI, and the effect should be
worse on graphs with more backtracking (denser/lower-diameter graphs).
Fixed: verify true BFS distance(u, W) == 2 via the real graph adjacency
(`baselines/splits/<ds>.pt`'s `edge_index` — confirmed same dense node-id
space as the walk-transformer's tokenizer, both report num_nodes=3783 on
bitcoin-alpha) before counting a sample, filtering out backtracked/stale
pairs (`build_neighbor_sets()` in `lead2_walk_relay_mi.py`). Reran all 6
datasets with the fix — see corrected table below.

**Audit — does this bug affect other Lead 2 steps? (user asked to check and
mark, not necessarily fix elsewhere):**
- Steps 0/1/1b (`lead2_gnn_bottleneck_mi.py`, `lead2_edge_sensitivity.py`):
  NOT affected — operate directly on real graph `edge_index`, no walk/token
  concept involved at all.
- Step 3 (`oracle_utils.py::compute_2hop_representative`): NOT affected —
  uses real sparse-matrix BFS (`A @ A`) and explicitly excludes `v` and v's
  1-hop neighbors from 2-hop candidates; genuinely 2-hop by construction.
- `edge_sign_mi_vs_distance_v3.py` / `node_mi_structural_embedding.py`
  (Lead 1-era, reused as context for Lead 2's Step 4 discussion): NOT
  affected — both explicitly do real per-source-node BFS on the graph
  adjacency, not walk-token offsets.
- **`scripts/attention_analysis.py` (Step 3 of the original
  Information/Understanding track, referenced throughout CLAUDE.md's
  "attends far despite empty signal" headline finding): LIKELY AFFECTED,
  same root cause — `effective_distance`/`frac_beyond_2hop` are computed
  from raw token-position offset `|i-j|` (1 graph-hop ≡ 2 token positions),
  with no backtrack/revisit verification against the real graph. This was
  NOT written or rerun during this Lead 2 session and has NOT been fixed —
  flagging only, per instruction. If revisited, the same true-BFS-distance
  verification approach should apply, and the existing "mean effective
  distance 9–16 tokens despite empty signal" claim in CLAUDE.md should be
  treated as provisional until re-verified.

bitcoin-alpha layer-by-layer NMI (corrected, post-fix): 0.0204 → 0.0333 →
0.2665 → 0.3343 (clean increasing pattern — later layers build up more
2-hop-relevant information at the relay position; very close to the
pre-fix numbers, since bitcoin-alpha's backtrack rate is moderate at 18.9%).

**Corrected final-layer NMI across all 6** (`outputs/lead2_walk_relay_mi/relay_mi_report.txt`),
alongside the fraction of candidate pairs dropped as backtracks:

| dataset | NMI (pre-fix, buggy) | NMI (post-fix, corrected) | % filtered as backtrack |
|---|---|---|---|
| bitcoin-alpha | 0.32 | 0.3343 | 18.9% |
| bitcoin-otc | 0.24 | 0.2356 | 15.3% |
| slashdot090221 | 0.28 | 0.1717 | 51.4% |
| epinions | 0.10 | 0.0505 | 31.2% |
| wiki-elec | 0.08 | 0.0797 | 2.9% |
| wiki-rfa | 0.02–0.05 | 0.0179 | 1.5% |

**Revised conclusion:** the fix changes the cross-dataset pattern materially.
slashdot090221 dropped from the "high-NMI" cluster (0.28, alongside
bitcoin-alpha/otc) to a middle position (0.1717, ≈40% relative drop) — over
half its original candidate pairs were backtracks, the largest filter rate
of any dataset, meaning the original measurement was substantially testing
"does R recall the very edge just masked" rather than genuine 2-hop
retention. epinions roughly halved (0.10→0.0505). wiki-elec/wiki-rfa were
barely affected (both have very low backtrack rates, 1.5–2.9%) and stay the
lowest-NMI datasets. Post-fix, only bitcoin-alpha/bitcoin-otc form a clear
"high-NMI" pair; slashdot090221 no longer co-clusters with them. This
weakens the original Step 2 narrative that linked high relay-MI to large
walk-model AUC margins — slashdot090221's AUC margin over baselines remains
large despite its corrected relay-MI being mid-pack, so the relay-MI-vs-
AUC-margin relationship across datasets is less clean than first reported.
Note for writeup: this measures whether the relay token's representation
*retains* 2-hop info, not whether the final prediction *uses* it via
attention (that's `attention_analysis.py`'s effective-distance metric, which
is itself flagged with the same unfixed bug — see audit above — report both
with that caveat, don't conflate).

## Step 3 — Oracle-GNN ablation (DONE, all 6 × both models)

New files: `baselines/GINEConv/model_oracle.py` + `run_oracle_with_our_splits.py`;
`baselines/CSG/oracle_discriminator.py` + `run_oracle_with_our_splits.py`;
shared `baselines/oracle_utils.py::compute_2hop_representative`.

Both oracle discriminators concatenate a precomputed representative 2-hop
neighbor's embedding alongside the existing `[z_u, z_v]`, bypassing the 1-hop
relay's compression for that one signal path. **Important convention match:**
CSG's `OracleDiscriminator.test()` must mirror the *original*
`SignedGCN.test()`'s exact convention (hard argmax label as the AUC score
input, `f1_score(..., average='binary')`) — an earlier draft used a
"better" continuous-probability + macro-F1 version which would have made
oracle/baseline numbers not directly comparable; this was caught and fixed.

**GINEConv oracle — complete, all 6 datasets** (verified against
`results_our_splits/<ds>/GINEConv_oracle/seed42/score.csv`, field 5 = `tst_auc`
— note an earlier comparison accidentally read field 4 = `val_auc` and looked
like a null result; corrected):

| dataset | baseline AUC | oracle AUC | Δ |
|---|---|---|---|
| bitcoin-alpha | 0.8740 | 0.8774 | +0.0034 |
| bitcoin-otc | 0.8970 | 0.8964 | −0.0007 |
| epinions | 0.8753 | 0.9075 | **+0.0321** |
| wiki-elec | 0.8638 | 0.8726 | +0.0088 |
| wiki-rfa | 0.8506 | 0.8576 | +0.0070 |
| slashdot090221 | 0.8567 | 0.8821 | **+0.0254** |

**CSG oracle — complete, all 6 datasets:**

| dataset | baseline AUC | oracle AUC | Δ |
|---|---|---|---|
| bitcoin-alpha | 0.7930 | 0.7883 | −0.0047 |
| bitcoin-otc | 0.8095 | 0.8074 | −0.0021 |
| epinions | 0.7627 | 0.7553 | −0.0074 |
| wiki-elec | 0.6906 | 0.6805 | −0.0101 |
| wiki-rfa | 0.7118 | 0.6603 | −0.0516 |
| slashdot090221 | 0.6838 | 0.7041 | +0.0203 |

Note: epinions and slashdot090221's plain (non-oracle) CSG artifacts were
found mid-session to be missing `state_dict` (needed by Step 1b) — root
cause: `run_with_our_splits.py` was edited at 11:51 to add `state_dict`
saving, but those two datasets' training runs had already started (loaded
the old script into memory) before the edit landed, so they finished without
it despite the file on disk having the fix. Retrained both with the current
script; CSG oracle's epinions/slashdot090221 numbers above were already
correct (oracle script wasn't affected by this race).

**Intermediate conclusion — a real, interesting tension worth keeping in the
final writeup:** GINEConv's oracle gain is large specifically on
epinions/slashdot (and the original retraining-fix history of this repo;
double check these aren't the same two datasets where CSG's gain is null/
negative — they are, except slashdot is CSG's one positive case). CSG's
oracle effect is flat-to-negative almost everywhere. This is the **opposite**
pattern from Step 2: epinions/wiki-elec/wiki-rfa are where the walk
transformer's relay MI was *lowest*, yet they're where GINEConv's oracle gains
the *most*. Don't average across model type when writing the final summary —
GINEConv and CSG disagree on whether oracle 2-hop access helps at all.

## Step 4 — Synthetic "inverted fog" graph (calibrated successfully, then ABANDONED at Step 5)

Goal: a graph with MI(sign,d=1)≈0, MI(sign,d=2) substantial — opposite of all
real datasets — to test whether baseline GNN specifically fails when 2-hop
signal is the *only* signal, while oracle-GNN/walk-transformer don't.

**Attempt 1 (diffusion-based, abandoned):** binary class diffused via BFS
from k seed nodes; sign = match(c(u),c(v)). Calibrated to ~0 MI at BOTH d=1
and d=2 — failed. Root cause: match/XOR-style signs are symmetric and don't
propagate node-identity information through a chain of edges at any distance.

**Two real bugs found and fixed along the way** (both in
`scripts/generate_synthetic_fog_graph.py`, kept since they're correctness
issues independent of the signal-design problem below):
1. Canonicalizing undirected pairs as `(min,max)` let bidirectional original
   edges get two independent sign draws, producing contradictory signs after
   symmetrization. Fixed by deduplicating to one undirected pair first.
2. Canonical (smaller,larger) direction made the directed graph a DAG ordered
   by node id (highest-id node = zero out-edges), badly skewing BFS. Fixed via
   random 50/50 direction swap per pair.
Both caught by the script's own `assert_formats_agree()` cross-format check
— keep that check, it's cheap and has already paid for itself twice.

**Attempt 2 (rep1-based, current code, signal-design problem found):**
each node gets `rep1(v)` = one fixed random direct neighbor. `b(v) ~
Bernoulli(0.5)` i.i.d. `sign(u,v) = +1 w.p. p_match if b(u)==b(rep1(v)) else
p_mismatch` (`p_match=0.95, p_mismatch=0.05, epsilon=0.02` label noise).
Rationale: `rep1(v)` is itself one of v's neighbors, so it's inside u's d=2
BFS frontier, without correlating with v's own out-edges (d=1).

Ran `edge_sign_mi_vs_distance_v3.py --datasets synthetic-fog --d-max 5` on
this construction:

```
d=0   MI=0.00021   (~0, fine)
d=1   MI=0.00000005 (~0, as intended)
d=2   MI=0.00000000  ← wanted "substantial", got zero
d=3-5 MI≈0
```

**Diagnosis (the actual finding, not just a failed run):** I read through
`edge_sign_mi_vs_distance_v3.py`'s d≥2 computation (lines ~176-239) and it
does NOT test "does this edge's sign depend on a specific node 2 hops away."
It tests **node-personality homophily**: does anchor source `u`'s own
aggregate out-edge sign-balance (a fixed scalar per node — "mostly positive"
vs "mostly negative") correlate with the *pooled sum* of that same
personality-scalar across **every node in the entire d-hop frontier**. Our
`rep1(v)` signal is a dependency on one specific buried node inside that
frontier; pooling it together with (typically) dozens-to-hundreds of unrelated
frontier nodes' personalities dilutes it to numerical zero. **The
construction's engineered signal is very likely still genuinely there — it's
the calibration tool that's measuring the wrong thing for this experiment.**

There's also a structural reason this is hard to fix by tuning parameters
rather than changing approach: node-personality is a fixed per-node scalar,
so any mechanism that makes it correlate at distance 2 tends to also leak
correlation to distance 1 (e.g. exact bipartite 2-coloring gives anti-
correlation at d=1, but MI doesn't care about sign of correlation — that
still fails the "≈0 at d=1" requirement). I don't think there's a parameter
sweep that fixes this; it needs a different calibration target.

**Resolution (user decision):** implement the targeted exact-MI calibration
directly in `generate_synthetic_fog_graph.py` (not the v3 script's
pooled-frontier metric). Done — `calibrate()` function added, reusing
`mi_from_joint` from `node_mi_structural_embedding.py`. Result:

```
MI(sign, majority-sign of v's own out-edges)        [d=1 analog]  = 0.000493 bits  (~0, good)
MI(sign, b(rep1(v)))  unconditional                                = 0.009646 bits  (see below)
I(sign; b(rep1(v)) | b(u))  [engineered d=2 dependency, CONDITIONAL] = 0.635026 bits (substantial, good)
```

**A second, more important catch found while implementing this:** the
originally-recommended calibration target — unconditional
`MI(sign(u,v), b(rep1(v)))` — is mathematically guaranteed to be ~0
regardless of construction strength, for the same structural reason Attempt 1
failed: `sign(u,v)` depends on whether `b(u) == b(rep1(v))`, an equality test
between two independent uniform bits. An equality test's outcome is
marginally independent of either input alone (P(match)=0.5 regardless of
b(rep1(v))'s value, since b(u) is uniform) — so the 0.0096 bits measured
above is finite-sample noise, not weak signal. The CONDITIONAL form,
`I(sign; b(rep1(v)) | b(u))`, is what actually exposes the dependency (since
u's own label is always available to a model as the 0-hop anchor) — this
came back at 0.635 bits, confirming the construction is sound. **If you
revisit this construction, calibrate on the conditional form, not the
marginal one — this is an easy trap to fall into again.**

## Step 5 — Train all 3 model types on synthetic-fog (ABANDONED — see below)

Infrastructure: `'synthetic-fog'` was missing from the plain (non-oracle)
`baselines/{GINEConv,CSG}/run_with_our_splits.py`'s argparse `choices=[...]`
(only the oracle scripts had it) — added. Trained GINEConv baseline + oracle
on the calibrated graph:

```
GINEConv baseline AUC = 0.5825
GINEConv oracle AUC   = 0.5740   (oracle WORSE than baseline, both ~chance)
```

**Diagnosis — a fundamental confound, not a tuning problem:** GNN baselines
(`run_with_our_splits.py`) initialize node features `x` as fixed random
vectors, `np.random.rand(...)`, never registered as a model parameter and
never updated by gradient descent. `b(v)`'s identity never enters the model
as a learnable input anywhere. This doesn't matter on real datasets because
the predictive signal there is generalizable structural balance (shared conv
weights learn triadic-balance patterns that transfer across nodes). But
synthetic-fog's signal requires recovering an arbitrary i.i.d. per-node
latent label with zero structural regularity — not learnable through fixed
random features and shared weights, independent of whether a bottleneck
exists. The oracle's non-positive result is exactly what this predicts: a
shortcut to a 2-hop representation that never had access to informative
signal can't help.

Separately and just as importantly: the walk-transformer
(`src/model/model.py:26`) uses a TRAINABLE `nn.Embedding` keyed by node id —
it has the capacity to memorize per-node idiosyncratic facts via gradient
descent, something the GNN baselines structurally cannot do. So any
walk-transformer "advantage" measured on this graph would be confounded with
"trainable node embedding vs. fixed random feature," not isolable as a pure
bottleneck effect — even if the GNN-feature issue above were fixed.

**User decision (2026-06-22):** abandon synthetic-fog rather than redesign
(e.g. toward a structurally-learnable d=2 signal analogous to real-data
triadic balance, which would need a fresh construction, not a parameter
tweak). The walk-transformer training run in progress (PID 2427063) was
killed. `data/synthetic-fog/` and `baselines/splits/synthetic-fog.pt` are
left on disk (calibrated, correct) in case this is revisited later, but nb:
any future attempt MUST address the node-feature confound above before the
comparison is meaningful.

## Step 6 — Consolidated report (DONE)

`outputs/lead2_gnn_bottleneck/SUMMARY.txt` — one row per real dataset (h1 MI
vs ceiling, edge-sensitivity dilution, walk relay MI, oracle AUC deltas for
both model types, explicitly not averaged), the synthetic-fog
calibration-and-abandonment writeup, then an explicit overall verdict.
Headline: the bottleneck is real and measurable (Steps 0–1b), but whether
bypassing it actually helps AUC (Step 3) is dataset- and architecture-
dependent — GINEConv gains on 6/6 (substantially on epinions/slashdot090221),
CSG gains on only 1/6 (slashdot090221) — and doesn't cleanly track the walk
model's own relay-MI pattern (Step 2). Overall verdict: bottleneck is a real
mechanism but not sufficient on its own to explain a uniform walk-model
advantage across datasets/architectures — see SUMMARY.txt's "OVERALL
VERDICT" section for the full statement.

---

## Already-registered infrastructure for synthetic-fog (done, reusable)

- `src/data/datasets.py`: `"synthetic-fog": load_bitcoin` in `DATASET_LOADERS`.
- `configs/synthetic-fog.yaml`: copy of bitcoin-alpha's config,
  `dataset.name/data_dir/edge_list_file` updated, `multiedge_handling: keep`.
- `scripts/balance_theory_paths.py`: `"synthetic-fog"` entry in
  `DATASET_CONFIGS` (`exp_dir: None, best_epoch: None` — unused by the
  MI-only calibration script).
- `baselines/*/run*_with_our_splits.py`: `synthetic-fog` already in both
  scripts' `--dataset` `choices=[...]` allowlists.
- Current data on disk: `data/synthetic-fog/synthetic_fog.csv` and
  `baselines/splits/synthetic-fog.pt` reflect the rep1-based Attempt 2
  construction (not yet known-good — see Step 4).

## SiGAT baseline + attention-weight diagnostic (NEW phase, IN PROGRESS)

**Motivation (user's idea):** Steps 0–1b measure GNN bottleneck dilution via
`contribution_share`/`leave_one_out_delta` — a proxy for "how much weight
does the GNN implicitly give this neighbor," derived from GINEConv/CSG's
sum/mean aggregation math. The user wants the same question asked with
LITERAL attention weights (GAT-style `alpha` coefficients), and to check
whether attention weight tracks actual neighbor *importance* — i.e. whether
some neighbors get a low attention weight from the model despite being
genuinely informative, meaning good signal is flowing through an
under-weighted ("bad") pipe.

**Why SiGAT:** the repo already has a GAT-style signed-graph baseline
(`baselines/SGA/sigat_SGA.py`, from the SGA paper's codebase) that was not
previously integrated into the canonical-splits comparison framework used by
GINEConv/CSG. User confirmed using it "both as probe and a baseline."
Architecture: 38 parallel `GATConv` aggregators (6 base pos/neg ×
edgelist/in/out + 32 triad-derived channels from `build_adj_lists()`), each
concatenated through an MLP (`torch_geometric_signed_directed.nn.signed.SiGAT`,
read at `.../sga_env/.../SiGAT.py`). Uses a **trainable**
`nn.Parameter(init_emb, requires_grad=True)` node embedding — unlike
GINEConv/CSG's fixed random `x` — an asymmetry worth keeping in mind (same
class of confound flagged in the abandoned synthetic-fog work above), since
it means SiGAT's comparison to GINEConv/CSG isn't purely about
attention-vs-sum aggregation.

**Scope decision (user):** "Bare SiGAT + curriculum only" — i.e. mirror how
GINEConv/CSG's `run_with_our_splits.py` scripts work: load our canonical
`baselines/splits/<ds>.pt`, keep SiGAT's own native curriculum
(`edgesBalanceDegree_sp` balance-degree-ordered edge schedule from
`balanceDegree.py`), track best-val-AUC checkpoint, report test AUC there.
Explicitly does NOT run SGA's separate data-augmentation pipeline
(`embeddingsGenerator`/`candidatesGenerator`/`dataAugmentation`) — declared
out of scope.

**New file:** `baselines/SGA/run_with_our_splits.py` (env:
`/home/eng/shilo_avital/.conda/envs/sga_env`, torch==1.12.0 — note
`torch.load()` there does NOT support `weights_only=`, omit the kwarg).
Validated end-to-end on bitcoin-alpha first, then run on all 6 (4 in
parallel across GPUs 0–3, epinions run last/sequentially with
`--eval-every 150` since its curriculum-rebuild-per-epoch loop is the
slowest of the 6).

**Status: DONE, all 6 complete.** Test AUC at best-val-AUC checkpoint
(`results_our_splits/<dataset>/SiGAT/seed42/score.csv`):

| dataset | SiGAT val AUC | SiGAT test AUC | test F1 |
|---|---|---|---|
| bitcoin-alpha | 0.8565 | 0.8449 | 0.9501 |
| bitcoin-otc | 0.8762 | 0.8700 | 0.9370 |
| epinions | 0.8711 | 0.8712 | 0.9262 |
| wiki-elec | 0.8598 | 0.8563 | 0.9033 |
| wiki-rfa | 0.8506 | 0.8422 | 0.8913 |
| slashdot090221 | 0.8418 | 0.8428 | 0.8831 |

For context against the project's existing SOTA table (CLAUDE.md): SiGAT
trails the walk-transformer on every dataset and trails the best GNN
baseline (GSGNN+SGA/CSG-GSGNN/SNEA, per dataset) on most — e.g.
bitcoin-alpha SiGAT 0.8449 vs. best-GNN-baseline 0.8804 vs. ours 0.9131;
epinions SiGAT 0.8712 vs. best-GNN-baseline 0.9113 vs. ours 0.9311. Not
surprising (SiGAT is an older/simpler architecture than the SOTA baselines
already in `baselines/all_results.csv`, and we deliberately skipped its
SGA data-augmentation) — its role here is as an attention-weight PROBE, not
as a new SOTA contender; the comparable-baseline framing is secondary.
Saved artifacts per dataset (`best_epoch_artifacts.pkl`): final node
embedding `z`, raw trainable input `x`, and full `state_dict` (needed to
reconstruct `GATConv` attention coefficients without retraining).

**Next: build the weight-vs-importance diagnostic itself — NOT YET STARTED
(design only, per user's "Both" answer on metric choice):**

1. **Bucketed MI by attention weight.** Extract literal `GATConv` attention
   coefficients via `return_attention_weights=True` (need to re-run forward
   passes from the saved `state_dict`+`x`, since the SiGAT training loop
   doesn't capture them) from the direct pos/neg out-edge channels
   (channels 1 `pos_out_edgelist` and 4 `neg_out_edgelist` in
   `build_adj_lists()`'s channel ordering — the direct analogues of
   GINEConv/CSG's `pos_ei`/`neg_ei` split used throughout Steps 0–1b).
   Bucket each (src, dst) edge by its attention weight (mirroring Step 1b's
   degree-bucket structure), then compute Step 1's
   MI(h_v^(SiGAT), sign of v's own out-edges) — or the SiGAT-equivalent
   final embedding `z` — separately within each attention-weight bucket. If
   attention weight tracked importance well, low-attention-weight buckets
   should show LOWER MI; the diagnostic looks for buckets where that breaks
   (low weight, high MI — info flowing through an under-weighted pipe).
2. **Leave-one-out delta vs. attention weight, correlated directly.** Reuse
   Step 1b's leave-one-out-delta machinery design pattern (recompute the
   aggregation/forward pass excluding one neighbor's message, measure the
   resulting embedding/prediction change) and directly correlate that delta
   (a more direct "how much did this neighbor actually matter" measure than
   bucketed MI) against the neighbor's own `GATConv` attention weight, per
   edge, across all 6 datasets. Low correlation, or a cluster of
   high-delta/low-weight points, is the direct signature of the user's
   "good info through bad pipes" hypothesis.

Per user instruction, build BOTH variants (not a choice between them).

**Status: DONE, both variants built and run on all 6 datasets.**

`baselines/SGA/extract_attention_weights.py` (sga_env): rebuilds each
dataset's trained SiGAT model from its saved checkpoint, extracts literal
`alpha` for the pos_out/neg_out channels via `return_attention_weights=True`,
and computes variant 2's ground-truth delta. Important correction made
mid-build: the FIRST version computed leave-one-out delta at the
PRE-MLP, single-channel output level, using the standard softmax
leave-one-out identity (`out_j' = (out_j - alpha_ji*Wx_i)/(1-alpha_ji)`).
Validating on bitcoin-alpha showed this is Spearman ≈0.99 correlated with
alpha itself — **nearly tautological**, since a GAT channel's own output is
mathematically determined by alpha (bigger weight → mathematically bigger
swing when removed). Fixed by pushing the leave-one-out-modified channel
output back through the model's full concat+MLP (`mlp_layer`, mixing all 38
channels nonlinearly) and measuring the delta in the FINAL embedding `z_j`
instead — this is the only version that can show the MLP doing something
attention alone didn't predict. Saved per dataset to
`outputs/lead2_sigat_attention/<ds>_attention.pkl`.

`scripts/lead2_sigat_attention_weight_mi.py` (.venv): variant 1 (bucket
edges by alpha, MI(z_j, neighbor i's own out-edge sign) per bucket) and
cross-references variant 2's Spearman(alpha, post-MLP delta) per channel.
Report: `outputs/lead2_sigat_attention/attention_weight_mi_report.txt`.

**Results — Variant 2 (Spearman(alpha, post-MLP leave-one-out delta)):**
even after fixing the tautology by going through the MLP, correlation stays
high but is no longer ≈1: pos channel 0.92–0.99 across all 6 datasets, neg
channel notably lower and more variable, 0.73–0.91. The MLP does add real
noise relative to alpha alone, more so for negative edges than positive.

**Results — Variant 1 (bucketed MI, 4 quantile buckets, low→high alpha):**

| dataset | pos channel (bucket 0→3 MI nmi) | neg channel (bucket 0→3 MI nmi) |
|---|---|---|
| bitcoin-alpha | 0.070→0.013→0.071→0.099 (mostly ↑) | 0.110→0.110→0.089→0.124 (mostly flat) |
| bitcoin-otc | 0.012→0.015→0.046→**0.190** (clean ↑) | 0.102→**0.137**→0.064→0.052 (↓ after bucket 1!) |
| epinions | 0.004→0.007→0.021→0.054 (clean ↑) | 0.086→**0.127**→0.076→0.101 (non-monotonic) |
| wiki-elec | 0.006→0.006→0.007→0.044 (clean ↑) | 0.052→0.066→0.066→0.095 (clean ↑) |
| wiki-rfa | 0.004→0.007→0.007→0.029 (clean ↑) | 0.022→0.025→0.056→0.107 (clean ↑) |
| slashdot090221 | 0.006→0.013→0.019→0.023 (clean ↑) | 0.010→0.009→0.010→0.038 (flat then ↑) |

**Finding (this is the headline result of the new follow-up phase):** for
the **positive**-edge channel, attention weight tracks true downstream
importance cleanly and monotonically on all 6 datasets — no "bad pipe"
cases. For the **negative**-edge channel, the picture is sign-asymmetric and
genuinely mixed: wiki-elec/wiki-rfa/slashdot090221 still show a clean (or
mostly-clean) increasing pattern, but **bitcoin-otc and epinions show a real
violation** — the highest-MI bucket is bucket 1 (low-mid attention), not
bucket 3 (highest attention): bitcoin-otc's bucket 1 (MI=0.1369) is more
than DOUBLE its bucket 3 (MI=0.0524), and epinions' bucket 1 (0.1271) also
exceeds its bucket 3 (0.1009). These are not small-sample artifacts — n per
bucket is 1,237 (bitcoin-otc) and 47,243 (epinions), the latter especially
solid. This is a genuine instance of the user's hypothesis: a meaningful
chunk of negative-edge neighbors carry MORE real predictive signal than
SiGAT's own attention weight credits them for — "good information through
an under-weighted (negative-edge) pipe" — specifically on bitcoin-otc and
epinions, not universally.

## Step 2 walk-distance bug — audit follow-up (marked, not fixed elsewhere)

`scripts/attention_analysis.py` was flagged during the Step 2 bug audit
(see Step 2 above) as LIKELY suffering the same walk-token-position-vs-true-
BFS-distance issue (`effective_distance`/`frac_beyond_2hop` use raw token
offsets with no backtrack verification). Per explicit user instruction this
was marked only, not fixed or rerun. CLAUDE.md's "attends far (mean 9–16
tokens) despite empty signal" headline finding rests on this script and
should be treated as provisional pending the same true-BFS-distance fix, if
this is ever revisited.

## Immediate next actions for whoever picks this up

Original Lead 2 scope (Steps 0–6) is complete — `outputs/lead2_gnn_bottleneck/
SUMMARY.txt` is that deliverable (read that first), now updated with
corrected Step 2 numbers. The active follow-up is the weight-vs-importance
diagnostic above: next concrete step is writing the attention-weight
extraction code (re-run SiGAT forward pass with
`return_attention_weights=True` from saved artifacts) before either MI- or
delta-correlation variant can be computed.
- If revisiting synthetic-fog (Step 4/5), the node-feature confound
  described above must be addressed first (either give the GNN baselines
  learnable per-node features, or redesign the d=2 signal to be
  structurally-recoverable like real-data triadic balance) — do not just
  rerun the existing construction expecting different results.
- Lead 3 (first-hop masking / "fog of war") is also complete per CLAUDE.md —
  this SiGAT/attention-weight diagnostic is a new sub-thread of Lead 2, not
  blocking on Lead 3.
