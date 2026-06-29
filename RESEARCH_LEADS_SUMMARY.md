# Research Leads Summary: Why Does the Walk-Transformer Beat GNN Baselines?

**Prepared:** 2026-06-23, **last updated:** 2026-06-28
**Audience:** supervisor / progress report
**Status:** Leads 1–3 complete. Lead 2 has an active follow-up sub-thread
(SiGAT attention-weight diagnostic, complete). Lead 4/4b (entropy
heterogeneity) complete and re-derived on the canonical split, but the
effect is entropy-variant-dependent (strong for `in_in`, weak for `out_in`)
and still no final causal verdict. See "Open threads" at the end for
candidate next steps.

> ### ⚠️ 2026-06-26 CANONICAL-SPLIT RERUN — read `CANONICAL_RERUN_FINDINGS.md`
>
> All Leads below were originally computed with the walk model and the GNN
> baselines scored on **~independent edge samples** (~10% overlap — see
> `SPLIT_PROVENANCE.md`). Every baseline has now been re-run on the unified
> canonical split and every affected Lead re-derived on **shared ground truth**.
> Net effect:
> - **Headline strengthens.** On identical shared test edges the walk model
>   beats every GNN on all 6 datasets (it was never a real comparison before).
> - **Leads 1, 2, 3 hold** qualitatively (over-averaging not primary; bottleneck
>   real, NMI 0.001–0.108; swamping severe in theory).
> - **Leads 4 / 4b: variant-dependent, revised 2026-06-28 (see
>   `CANONICAL_RERUN_FINDINGS.md` §2.1 for full numbers).** The original retraction
>   only checked the `out_in` entropy definition (H(out-signs of u) × H(in-signs of
>   v)), where the effect is genuinely weak/mixed (on epinions/wiki-rfa GINEConv
>   degrades *less* than walk). But the `in_in` definition (H(in-signs of u) ×
>   H(in-signs of v)) shows a **strong, robust** effect: GINEConv's AUC drop from
>   low- to high-entropy is **2–10× the walk model's drop, on 5/6 datasets, at every
>   bucket size tested (b2–b32)**. Lead 4b (2-hop path consistency) does **not**
>   corroborate this — it stays noisy/mixed regardless of variant. Interpretive
>   wrinkle: `in_in` is mechanistically the *less* obvious variant for predicting
>   `sign(u→v)` (it's `out_in` that structural-balance theory would motivate), so
>   the strongest effect lives in the variant with the weaker causal story —
>   possibly a proxy for degree/hubness rather than heterogeneity per se. Also
>   still open: whether walk coverage (`WALK_COVERAGE.md`, active workstream)
>   correlates with `in_in` entropy, which would affect how representative the
>   high-entropy buckets are.

> ### ✅ 2026-06-29 E15 FULL-COVERAGE RERUN — the coverage caveat above is RESOLVED
>
> The walk coverage workstream shipped: the `k_cover` k=5 sampler drives walk
> coverage to **~100% of every dataset's test set** (was ~85–88% on the sparse
> graphs). `predictions_raw_canonical.pkl` was rebuilt from the new full-coverage
> walk runs (GNNs unchanged) and Leads 1/4/4b rerun on **identical, complete**
> shared edges. Result: the variant-dependent picture **HOLDS and slightly
> STRENGTHENS** — `in_in` differential degradation is positive on all 6 (best-GNN
> drop − walk drop = +0.015..+0.34, ≥ the ~88%-coverage values),
> `out_out`/`inout_inout` positive 5/6 (wiki-elec ~null), `out_in` still weak.
> The previously-uncovered (harder, peripheral) edges did **not** wash out the
> effect — so it is NOT a different-edges artifact, and the walk advantage
> concentrates in heterogeneous in-anchored neighborhoods, not a uniform offset.
> Full numbers: `outputs/lead4_entropy_heterogeneity/E15_FULLCOVERAGE_RERUN_NOTE.md`;
> sampler/SOTA story: `outputs/walk_coverage_analysis/E15_SWEEP_RESULTS.md`.

## The central question

The walk-Transformer beats every GNN/SGNN baseline on all 6 signed-graph
datasets (bitcoin-alpha, bitcoin-otc, epinions, wiki-elec, wiki-rfa,
slashdot090221) by 3–5 percentage points test AUC, on identical canonical
splits. This is puzzling given a prior finding: **edge-sign mutual
information collapses 10–1000× between 1-hop and 2-hop neighbors**, on every
dataset. If almost no sign-relevant information exists beyond the immediate
neighborhood, why would an architecture that can "see" further (the walk
model's full-sequence attention) have any advantage over a 2-layer GNN that
mostly only needs 1–2 hops anyway?

Three hypotheses (the user's organizing intuition: GNNs do BFS-like
horizontal aggregation per layer; walks give the Transformer a single
sequence axis with independently-weighted attention per position) were
investigated in order:

1. **Lead 1 — over-averaging:** GNN sum/mean aggregation collapses diverse
   per-neighbor evidence into one vector, losing information a non-naive
   aggregator (like the walk model's posthoc `func_logit_power`) could use.
2. **Lead 2 — bottleneck:** a 2-hop node's information must pass through a
   compressed, fixed-size 1-hop embedding before reaching the target edge —
   information loss from forced compression, independent of averaging.
3. **Lead 3 — fog of war / swamping:** the strong 1-hop signal numerically
   drowns out the much weaker 2-hop signal when both are forced to share an
   aggregation slot (a magnitude/SNR effect, distinct from both of the above).

**Bottom line up front:** all three mechanisms are real and measurable to
some degree, but **none of them — alone or combined — cleanly explains a
uniform advantage across all 6 datasets and both GNN architectures tested.**
Each lead found a genuine GNN failure mode, but bypassing it (via ablation)
either didn't move AUC much, or helped inconsistently across
datasets/architectures. This is itself a meaningful finding — it suggests
either the explanation is multi-causal (several small effects stacking) or
there's a still-undiscovered 4th mechanism (see "Open threads").

---

## Lead 1 — GNN Over-Averaging (sum/mean aggregation loses information)

**Method:** 4 measurements on bitcoin-alpha → expanded to all 6 datasets,
using GINEConv (sum-aggregator GNN) as the clean testbed.

| Measurement | Finding |
|---|---|
| Degree-stratified AUC gap | **No consistent shape across datasets** — inverted-U, flat, U-shaped, or monotonic depending on dataset. Only wiki-elec matches the "gap widens with degree" prediction. |
| Per-layer sign-decodability probe | Sign goes from random (~0.50 AUC) to highly decodable (~0.82–0.86) after just 1 layer, and **stays flat or improves slightly** at layer 2 — no progressive information destruction across layers. |
| Aggregator ablation (sum vs. mean vs. max) | **Dataset-dependent**: sum wins on the 2 small/sparse datasets, mean wins on the 4 large/dense ones. Mean's advantage does NOT concentrate at high degree (where cancellation should be worst) — looks like an optimization effect, not cancellation relief. Max is uniformly worst. |
| Cancellation-ratio metric (vector cancellation from mixed-sign neighbors) | **Strongest, most universal finding of this lead**: sign-heterogeneity correlates with vector cancellation at r≈−0.96 to −0.97 across all 6 datasets — extremely clean, dataset-independent. **But the effect is small**: only 1–9% norm loss on average. |
| Depth sweep (1–5 layers) + oversmoothing (MAD) | Depth beyond 2 layers never helps; 4/6 datasets actually peak at 1 layer. But MAD (the standard oversmoothing metric) does **not** consistently track the AUC degradation — oversmoothing isn't the operating mechanism; it's more likely that there's simply no real signal past 1–2 hops for deeper layers to exploit (consistent with the MI-collapse finding), so depth just adds noise/optimization instability.

**Verdict: ruled out as the primary driver.** Cancellation is real and
universal but modest (~5–9%). No other sub-mechanism in this lead shows a
consistent cross-dataset pattern. Full detail: `LEAD1_GNN_OVER_AVERAGING_REPORT.md`.

---

## Lead 2 — GNN Bottleneck (forced compression through the 1-hop embedding)

**Method:** measured whether a GNN's 1-hop embedding `h_v^(1)` genuinely
loses 2-hop-relevant information relative to a theoretical ceiling, then
tested whether bypassing that bottleneck (an "oracle" skip-connection giving
the discriminator direct access to a 2-hop node) recovers AUC. Two GNN
testbeds: GINEConv (sum-aggregator) and SignedGCN/CSG (mean-aggregator).

| Step | Finding |
|---|---|
| `h_v^(1)` MI vs. ceiling (all 6 datasets × 2 models) | Both models retain real, above-chance, but **far from lossless** neighbor-sign information (NMI 0.008–0.27 out of a ceiling of ~1 bit). Consistent direction across all 6 datasets/both models → genuine bottleneck signature. GINEConv (sum) consistently retains more than CSG (mean). |
| Per-edge bottleneck sensitivity | A specific edge's contribution to its node's aggregate **shrinks monotonically with out-degree**, confirmed on all 6 datasets, both models — direct confirmation of dilution at the per-edge level, not just on average. |
| Walk-model's own relay-token MI (does the walk model itself retain 2-hop info at the analogous position?) — **corrected 2026-06-24, see note below** | Final-layer NMI, post-fix: bitcoin-alpha 0.33, bitcoin-otc 0.24, slashdot090221 0.17, epinions 0.05, wiki-elec 0.08, wiki-rfa 0.02. |
| **Oracle-GNN ablation (the key causal test)** | **GINEConv**: oracle 2-hop access improves AUC on 6/6 datasets, substantially on epinions (+0.032) and slashdot (+0.025), modestly elsewhere. **CSG**: oracle access helps on only 1/6 datasets (slashdot, +0.020) and is flat-to-negative everywhere else (notably −0.052 on wiki-rfa). |
| Synthetic "inverted fog" stress graph (engineered to have ≈0 MI at 1-hop, substantial MI at 2-hop — opposite of every real dataset) | Successfully calibrated after two failed attempts, but training revealed a **fundamental confound**: GNN baselines use fixed, non-trainable random node features, so they structurally cannot learn an arbitrary per-node latent signal regardless of any bottleneck — while the walk-transformer's trainable per-node embeddings could trivially "solve" it by memorization, confounding any comparison. **Abandoned by decision, not by running out of leads.** |

**Bug correction (2026-06-24, affects the relay-MI row above):** the
original relay-MI measurement identified the "2-hop" comparison node by a
fixed walk-token-position offset, but random walks can backtrack — so that
token isn't always a genuinely fresh 2-hop node (it can be the very node the
masked edge already revealed). After filtering out backtracked pairs via
true graph-BFS distance, NMI dropped substantially on the two datasets with
high backtrack rates: **slashdot090221 0.28→0.17** (51% of original pairs
were backtracks) and **epinions 0.10→0.05** (31% backtracks); bitcoin-alpha/
otc/wiki-elec/wiki-rfa were only mildly affected (backtrack rates 1.5–19%).
This weakens the original cross-dataset story that linked high relay-MI to
large walk-model AUC margins — slashdot090221's AUC margin stays large
despite its corrected relay-MI now being mid-pack, not top-tier.

**Follow-up sub-thread — SiGAT literal-attention-weight diagnostic (new,
complete):** integrated SiGAT (GAT-style, trainable node features, 38
parallel attention channels) as both (a) a new canonical-splits baseline
(test AUC 0.84–0.87 across all 6 datasets — trails the walk-transformer on
every dataset and trails most "best GNN baseline" entries above too) and (b)
a literal-attention-weight probe, asking whether `GATConv`'s own attention
coefficients track genuine neighbor importance (bucketed MI between
attention-weight quartile and the neighbor's real out-edge-sign information).
**Finding:** on the positive-edge channel, attention weight tracks
importance cleanly and monotonically on all 6 datasets. On the
**negative**-edge channel, **bitcoin-otc and epinions** show a genuine
violation — a low-mid-attention bucket carries *more* real signal than the
highest-attention bucket (bitcoin-otc 0.137 vs. 0.052 NMI; epinions 0.127 vs.
0.101 NMI, both large-n) — "good information through an under-weighted
(negative-edge) pipe," sign-asymmetric and dataset-specific, not universal.

**Verdict: a real mechanism, but not sufficient alone, and architecture- and
dataset-dependent.** The bottleneck demonstrably loses information (Steps
0–1b), but whether removing it actually helps (Step 3) is split: it matters
a lot for GINEConv, barely at all for CSG, and doesn't track the walk
model's own (corrected) relay-MI pattern (the datasets where GINEConv's
oracle gain is biggest are *not* the datasets where the walk model retains
the most 2-hop information). The attention-weight follow-up adds a second,
narrower finding: even where the model uses literal attention, negative-edge
importance can be systematically under-weighted on specific datasets.
Full detail: `LEAD2_GNN_BOTTLENECK_STATUS.md`.

---

## Lead 3 — Fog of War / Signal Swamping

**Method:** tests a narrower, more specific mechanism than Lead 2 — not
"information is compressed away" but "a strong signal numerically drowns
out a weak one when both are forced into the same averaged slot."

| Step | Finding |
|---|---|
| Synthetic SNR test (model-free, no real data) | Swamping is **real and severe** in the idealized case: mean-aggregated recoverability of a weak target signal degrades to chance as the number/magnitude of diluting "1-hop-like" signals grows, while concatenation (mimicking attention's independent per-token slots) stays flat. Sanity checks confirmed the effect is specifically magnitude-driven, not just dilution-by-count. |
| Real-GNN magnitude-ratio check (reusing Lead 2's measured dilution levels, no retraining) | Plugging real datasets' measured dilution into the synthetic swamping curve predicts that **every degree bucket except the lowest is in the "destroyed" regime** on all 6 datasets, both architectures — i.e., if the swamping mechanism applies as modeled, it should be severe almost everywhere in practice. |
| Walk-model attention vs. 1-hop ambiguity (the direct test of *adaptive* compensation) | Tests whether attention shifts toward 2-hop tokens specifically when 1-hop evidence is locally ambiguous. **Result: no evidence of this** — correlation between ambiguity and attention-mass-beyond-1-hop is small-to-moderate and **positive** on all 6 datasets (opposite of the predicted sign). The model attends further when 1-hop evidence is already clear, not when it's weak. |

**Verdict: swamping is a real risk for GNN aggregation, and the walk
model structurally avoids it (concatenation/independent attention slots
never force a shared sum) — but this avoidance is a structural byproduct of
the architecture, not a learned, adaptive compensation mechanism.** The
model isn't "smart" about it; it just never has the problem in the first
place. Full detail: `LEAD3_FOG_OF_WAR_REPORT.md`.

---

## Lead 4 / 4b — Node Sign-Entropy Heterogeneity vs. AUC

**Method:** distinct from Leads 1–3 — instead of probing a GNN internal
mechanism, bins test edges by how *sign-heterogeneous* (unpredictable) their
endpoints' neighborhoods are, and compares per-bin AUC across `walk_full`,
`walk_localattn4`, GINEConv, SiGAT, on all 6 datasets, no retraining (reuses
existing prediction artifacts + the project's standard `func_logit_power`
posthoc aggregator, verified to exactly reproduce the SOTA table above).
- **Lead 4**: 2D grid — source-node sign-entropy × target-node sign-entropy
  (4 directional variants: out/out, in/in, out/in, combined).
- **Lead 4b**: a single edge-level score — entropy of the target node's own
  2-hop out-path sign-consistency (does a `v→m→k` path tend to be `+/+` or
  `-/-`, vs. unpredictable) — a different heterogeneity definition, same
  spirit, reuses Lead 4's cached predictions (no model reloading).

**Finding, revised 2026-06-28 after the canonical-split rerun (see
`CANONICAL_RERUN_FINDINGS.md` §2.1 for the full per-variant numbers — this
supersedes the original pre-canonical write-up that used to follow this
line):** the original cross-dataset description below was measured on
**independent edge samples** for walk vs. GNN (the pre-canonical-split bug,
`SPLIT_PROVENANCE.md`). On the corrected shared-edge data, the effect turned
out to be **entropy-variant-dependent**, not uniform:
- `out_in` (the mechanistically natural variant — u's own outgoing tendency ×
  v's own incoming tendency): **weak/mixed** — on epinions/wiki-rfa GINEConv
  actually degrades *less* than the walk model.
- `in_in` (u's incoming-sign entropy × v's incoming-sign entropy): **strong
  and robust** — GINEConv's low→high-entropy AUC drop is 2–10× the walk
  model's drop, on 5/6 datasets, at every bucket size from b2 to b32 (not a
  single-bucket-size artifact). This is one of the cleanest cross-dataset
  patterns found anywhere in Leads 1–4b.
- `out_out`/`inout_inout`: mixed, closer to `out_in`.
- **Lead 4b (2-hop path consistency) does not corroborate the `in_in`
  story** — it stays noisy/sign-flipping across datasets regardless of
  variant; the clean differential survives only on epinions, as originally
  reported.

**Open interpretive question:** `in_in` is *not* the variant structural-
balance theory would predict should matter most for `sign(u→v)` (that's
`out_in`) — so the cleanest empirical effect lives in the variant with the
weakest causal story, suggesting `in_in` entropy may be acting as a proxy for
something else (in-degree/hub-ness is the leading suspect) rather than
"sign-unpredictability" directly. Also unresolved: whether walk coverage
(`WALK_COVERAGE.md`) correlates with `in_in` entropy, which would bear on how
representative the (smaller) high-entropy buckets actually are. Full detail,
per-dataset tables/heatmaps: `outputs/lead4_entropy_heterogeneity/report.md`,
`outputs/lead4_twohop_path_consistency/report.md`,
`CANONICAL_RERUN_FINDINGS.md` §2.1.

---

## Synthesizing the leads

| | Real & measurable? | Universal across 6 datasets? | Closes the AUC gap when bypassed? |
|---|---|---|---|
| Over-averaging/cancellation (Lead 1) | Yes, but small (~5–9%) | Direction yes, magnitude flat | N/A — too small to test causally |
| Bottleneck (Lead 2) | Yes (NMI well above null) | Yes (direction), magnitude varies | Only for GINEConv, only some datasets |
| Swamping (Lead 3) | Yes, severe in theory | Yes (predicted from real dilution) | No adaptive compensation found; avoidance is structural not learned |
| Entropy-heterogeneity gap (Lead 4/4b) | Yes, but only under the `in_in` entropy definition | `in_in`: yes, 5/6 datasets, all bucket sizes. `out_in`/other variants: no | N/A — descriptive correlation, no ablation performed yet; causal driver (heterogeneity vs. degree/hub-ness proxy) unresolved |

The honest picture for your supervisor: **we have ruled out the simplest
single-cause story** ("GNNs lose information through averaging/bottleneck and
the walk model doesn't"). All three mechanistic leads (1–3) exist in GNNs
and are individually measurable, but none of the corresponding ablations
(aggregator swap, oracle bypass, attention-adaptivity check) produces a
clean, dataset-uniform recovery of the gap. Lead 4/4b adds a fourth,
still-correlational data point — GNNs specifically struggle on `in_in`-defined
heterogeneous neighborhoods (robust, 5/6 datasets) but not clearly on other
entropy definitions — that hasn't yet been tied to one of Leads 1–3's specific
mechanisms, shown via ablation, or disentangled from a possible degree/hub-ness
confound. The walk model's advantage
appears to be **structural and somewhat overdetermined** — multiple small,
real GNN handicaps stack, rather than one dominant mechanism explaining
everything — or there's a contributing factor outside the leads
investigated so far (see below).

---

## Open threads worth flagging (not yet investigated)

A few candidate explanations were named in the original plan or surfaced
during this work but never executed. **Note: these are unrelated to, and
should not be confused with, the existing Lead 4/4b above** — numbering here
restarts at the next free slot (Lead 5) to avoid collision:

1. **Lead 5 — Per-edge walk-prediction variance / "ensemble effect" (the
   original "cheap, do first" cross-cutting measurement).** The very first
   item in the original plan — compute the variance of walk-level
   predictions for a single target edge across the many walks that pass
   through it — was never actually run. High variance would mean each walk
   carries distinct evidence that the non-naive posthoc aggregator
   (`func_logit_power`) can exploit, which a single deterministic GNN
   forward pass structurally cannot replicate. This is a 5th candidate
   mechanism (call it "ensemble effect") distinct
   from all three leads tested, and it's the cheapest of all of them to
   check (no training needed — existing posthoc artifacts already have the
   per-walk predictions).
2. **Lead 6 — Training-regime confound + capacity mismatch (merged, same
   root cause).** The walk model's SOTA configuration uses dynamic
   re-masking (D), node-token replacement (R), and hard-node reweighting (H)
   — training-time regularization tricks with no GNN-baseline analogue. None
   of Leads 1–3 controlled for this; the entire gap measured throughout was
   "walk model with all its training tricks" vs. "plain GNN," not
   "architecture vs. architecture" in isolation. Separately, no check was
   done on whether the walk-Transformer simply has more trainable
   parameters/capacity than the small 2-layer GNN baselines. **Scoping this
   plan surfaced that both threads trace to one root cause:**
   `baselines/GINEConv/run_with_our_splits.py:87` (and the same pattern in
   CSG/GSGNN/SGCN/SNEA) initializes node features as `np.random.rand(...)`,
   **never registered as a trainable parameter** — so there's nothing for an
   R-analogue to regularize, and it's most of why these baselines have far
   fewer trainable parameters than the walk model's `num_nodes ×
   embedding_dim` embedding table. **A second, independent confound found
   during scoping:** every `torch_geometric`/`torch_geometric_signed_directed`
   baseline (CSG, SGCN, SNEA, SiGAT) has a library-intended default of real
   *spectral* input features (`create_spectral_features()`, TruncatedSVD of
   the signed adjacency) that this repo's wrapper scripts consistently
   bypass in favor of pure random noise — likely an oversight, not a
   documented choice. **Tempering evidence already in hand:** SiGAT (the
   one baseline in this repo that already has trainable, if randomly
   initialized, features) still only gets test AUC 0.84–0.87 across all 6
   datasets — trailing the walk-transformer by a wide margin — so trainable
   features alone are unlikely to be a full explanation, though the
   spectral-feature variant hasn't been tested yet. See the dedicated plan
   for the full investigation and experiment design:
   `~/.claude/plans/` (Lead 5/6 subplans, see plan index once split).

I'd recommend treating these as the natural next phase if you want to
continue this line — a concrete plan for Lead 5 and Lead 6 already exists
(see plan files).
