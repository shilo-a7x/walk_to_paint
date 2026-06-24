# Research Leads Summary: Why Does the Walk-Transformer Beat GNN Baselines?

**Prepared:** 2026-06-23
**Audience:** supervisor / progress report
**Status:** Leads 1–3 complete. See "Open threads" at the end for candidate next steps.

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
| Walk-model's own relay-token MI (does the walk model itself retain 2-hop info at the analogous position?) | Highly mixed: 0.32/0.24/0.28 NMI on bitcoin-alpha/otc/slashdot vs. only 0.02–0.10 on epinions/wiki-elec/wiki-rfa. |
| **Oracle-GNN ablation (the key causal test)** | **GINEConv**: oracle 2-hop access improves AUC on 6/6 datasets, substantially on epinions (+0.032) and slashdot (+0.025), modestly elsewhere. **CSG**: oracle access helps on only 1/6 datasets (slashdot, +0.020) and is flat-to-negative everywhere else (notably −0.052 on wiki-rfa). |
| Synthetic "inverted fog" stress graph (engineered to have ≈0 MI at 1-hop, substantial MI at 2-hop — opposite of every real dataset) | Successfully calibrated after two failed attempts, but training revealed a **fundamental confound**: GNN baselines use fixed, non-trainable random node features, so they structurally cannot learn an arbitrary per-node latent signal regardless of any bottleneck — while the walk-transformer's trainable per-node embeddings could trivially "solve" it by memorization, confounding any comparison. **Abandoned by decision, not by running out of leads.** |

**Verdict: a real mechanism, but not sufficient alone, and architecture- and
dataset-dependent.** The bottleneck demonstrably loses information (Steps
0–1b), but whether removing it actually helps (Step 3) is split: it matters
a lot for GINEConv, barely at all for CSG, and doesn't track the walk
model's own relay-MI pattern (the datasets where GINEConv's oracle gain is
biggest are *not* the datasets where the walk model retains the most 2-hop
information). Full detail: `LEAD2_GNN_BOTTLENECK_STATUS.md`.

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

## Synthesizing the three leads

| | Real & measurable? | Universal across 6 datasets? | Closes the AUC gap when bypassed? |
|---|---|---|---|
| Over-averaging/cancellation (Lead 1) | Yes, but small (~5–9%) | Direction yes, magnitude flat | N/A — too small to test causally |
| Bottleneck (Lead 2) | Yes (NMI well above null) | Yes (direction), magnitude varies | Only for GINEConv, only some datasets |
| Swamping (Lead 3) | Yes, severe in theory | Yes (predicted from real dilution) | No adaptive compensation found; avoidance is structural not learned |

The honest picture for your supervisor: **we have ruled out the simplest
single-cause story** ("GNNs lose information through averaging/bottleneck and
the walk model doesn't"). All three candidate loss mechanisms exist in GNNs
and are individually measurable, but none of the corresponding ablations
(aggregator swap, oracle bypass, attention-adaptivity check) produces a
clean, dataset-uniform recovery of the gap. The walk model's advantage
appears to be **structural and somewhat overdetermined** — multiple small,
real GNN handicaps stack, rather than one dominant mechanism explaining
everything — or there's a contributing factor outside the three leads
investigated so far (see below).

---

## Open threads worth flagging (not yet investigated)

A few candidate explanations were named in the original plan or surfaced
during this work but never executed:

1. **Per-edge walk-prediction variance (the original "cheap, do first"
   cross-cutting measurement).** The very first item in the plan — compute
   the variance of walk-level predictions for a single target edge across
   the many walks that pass through it — was never actually run. High
   variance would mean each walk carries distinct evidence that the
   non-naive posthoc aggregator (`func_logit_power`) can exploit, which a
   single deterministic GNN forward pass structurally cannot replicate. This
   is really a 4th candidate mechanism (call it "ensemble effect") distinct
   from all three leads tested, and it's the cheapest of all of them to
   check (no training needed — existing posthoc artifacts already have the
   per-walk predictions).
2. **Training-regime confound.** The walk model's SOTA configuration uses
   dynamic re-masking, node-token replacement, and hard-node reweighting —
   training-time regularization tricks with no GNN-baseline analogue. None
   of Leads 1–3 controlled for this; the entire gap measured throughout was
   "walk model with all its training tricks" vs. "plain GNN," not
   "architecture vs. architecture" in isolation.
3. **Capacity mismatch.** The GNN baselines are small 2-layer networks;
   no check was done on whether the walk-Transformer simply has more
   trainable parameters/capacity, independent of its architecture's access
   pattern.

I'd recommend treating these as the natural next phase if you want to
continue this line — happy to scope a concrete plan for any of them.
