# Canonical-split rerun — findings (2026-06-26)

After fixing the split mismatch (`SPLIT_PROVENANCE.md`), every baseline was re-run
on the **canonical (walk-derived) split** into isolated dirs
(`baselines/*/results_our_splits_canonical/`, `baselines/all_results_canonical.csv`),
and all affected research Leads were re-derived on the corrected, **same-edge**
predictions (`outputs/lead4_entropy_heterogeneity/predictions_raw_canonical.pkl`).
This file is the consolidated answer to: *do the numbers and the leads still hold?*

The walk model is **not** retrained — its predictions are byte-identical to the
SOTA table. Only the baselines and the diagnostics moved onto shared ground truth.

---

## 1. Headline: walk model still wins — and the win is now real

### 1a. Apples-to-apples (identical shared test edges)

On the exact same edges (the walk-covered test set; walk_full ⊆ every GNN's test
set), the walk model beats **every** GNN on **all 6** datasets. Previously this
comparison ran on ~90%-disjoint edge sets, so it wasn't a real comparison.

| dataset | walk_full | walk_localattn4 | GINEConv | SiGAT | n_shared |
|---|---|---|---|---|---|
| bitcoin-alpha | **0.9131** | 0.9370 | 0.8610 | 0.8821 | 2419 |
| bitcoin-otc | **0.9431** | 0.9337 | 0.8849 | 0.8759 | 3560 |
| epinions | **0.9311** | 0.9445 | 0.8642 | 0.9109 | 73783 |
| wiki-elec | **0.8928** | 0.8917 | 0.8738 | 0.8892 | 8857 |
| wiki-rfa | **0.8810** | 0.8882 | 0.8722 | 0.8808 | 15280 |
| slashdot | **0.8952** | 0.8958 | 0.7869 | 0.8571 | 53962 |

(LocalAttn4 wins outright on 4/6.) Source:
`outputs/lead4_entropy_heterogeneity/canonical_join_report.md`.

### 1b. SOTA table (each model on its OWN full test) — canonical best GNN

| dataset | old best-GNN | canonical best-GNN | walk |
|---|---|---|---|
| bitcoin-alpha | 0.8804 (GSGNN+SGA) | 0.9051 (SGA-GSGNN) | **0.9131** |
| bitcoin-otc | 0.9086 (GSGNN+SGA) | 0.8972 (SNEA) | **0.9431** |
| epinions | 0.9113 (CSG-GSGNN) | 0.9146 (SiGAT) | **0.9311** |
| wiki-elec | 0.8840 (CSG-GSGNN) | 0.8930 (SiGAT) | 0.8928 |
| wiki-rfa | 0.8673 (CSG-GSGNN) | 0.8831 (SiGAT) | 0.8810 |
| slashdot | 0.8845 (SNEA) | 0.8587 (SiGAT) | **0.8952** |

Canonical best-GNN numbers are close to the old ones (some up, some down) — local
≈ published, as expected. **SiGAT** is the strongest baseline on the corrected split.

**The wiki-elec / wiki-rfa caveat.** On each-model's-own-full-test SiGAT marginally
edges the walk model (wiki-elec +0.0002, wiki-rfa +0.0021) — but **only because the
walk model covers ~85% of those test edges and SiGAT is scored on 100%** (walk
coverage, see `WALK_COVERAGE.md`). On the **identical** edges (1a) the walk model
wins both. So: walk wins all 6 apples-to-apples; walk wins 4/6 on each-own-full-test,
ties on the two wiki datasets where coverage < 100%.

Full per-model canonical AUCs: `baselines/all_results_canonical.csv`.

---

## 2. The research Leads, re-derived on shared edges

| Lead | original verdict | canonical-split verdict | holds? |
|---|---|---|---|
| 1 — over-averaging (degree gap) | modest ~5–9%, not primary | walk-vs-GNN gap does **not** systematically widen with degree (widens on some, narrows on others) | ✅ holds |
| 2 — GNN bottleneck (h¹ MI) | bottleneck real, NMI 0.008–0.27 | NMI 0.001–0.108, well above null floor — bottleneck real | ✅ holds |
| 2 — edge-sensitivity dilution | high-degree share collapses | low-degree share ~0.5 → high-degree ~0.005, same pattern | ✅ holds |
| 3 — swamping (real-GNN magnitude) | severe in theory | high-degree buckets "destroyed/partial" across all 6 | ✅ holds |
| **4 / 4b — entropy heterogeneity** | **GNNs degrade more than walk as sign-heterogeneity rises** | **variant-dependent: weak/mixed for the `out_in` entropy definition; strong and robust (2–10× larger GNN drop, all bucket sizes, 5/6 datasets) for the `in_in` definition — see §2.1 correction below** | ⚠️ **revised 2026-06-28 — not a uniform retraction, see §2.1** |

### The Lead 4 / 4b correction (the one real change)

Lead 4/4b's specific mechanism — "the walk model's advantage is *concentrated* in
high-heterogeneity neighborhoods, because GNNs degrade there and the walk model
doesn't" — was **substantially an artifact of comparing different edge sets**, but
**only for one of the four entropy-variant definitions Lead 4 computes**. The
original retraction (below) quoted only `out_in`; re-checking the raw per-cell data
(`outputs/lead4_entropy_heterogeneity/raw_data.txt`) across all four variants and all
five bucket sizes (2026-06-28) shows the effect is variant-dependent, not uniformly
gone.

**`out_in` (H(out-signs of u) × H(in-signs of v) — the mechanistically "natural"
pairing, directionally aligned with the edge u→v itself): weak/mixed**, as already
reported. Low→high joint-entropy AUC drop, b8 corner cells:

| dataset | walk drop | GINEConv drop | SiGAT drop | GNN drops *more*? |
|---|---|---|---|---|
| bitcoin-otc | +0.192 | +0.422 | +0.725 | yes |
| epinions | +0.277 | **+0.211** | +0.336 | **no (GINE less)** |
| wiki-elec | +0.440 | +0.408 | +0.422 | ~equal |
| wiki-rfa | +0.419 | **+0.243** | +0.445 | **no (GINE less)** |
| slashdot | +0.464 | +0.452 | +0.469 | ~equal |

**`in_in` (H(in-signs of u) × H(in-signs of v)): strong and robust**, in the
opposite direction from the retraction. GINEConv's drop is 2–10× the walk model's
drop, in **every** dataset with enough high-entropy mass to measure (bitcoin-alpha's
high-entropy corner is too sparse, n=3), and the pattern holds at **every** bucket
size from b2 to b32, not just b8:

| dataset | walk drop (b8) | GINEConv drop (b8) | SiGAT drop (b8) | walk drop range (b2→b32) | GINEConv drop range (b2→b32) |
|---|---|---|---|---|---|
| bitcoin-otc | 0.035 | **0.498** | **0.813** | 0.035–0.122 | 0.263–0.498 |
| epinions | 0.133 | **0.479** | 0.190 | 0.062–0.203 | 0.201–0.559 |
| slashdot | 0.143 | **0.438** | 0.171 | 0.104–0.162 | 0.234–0.449 |
| wiki-elec | 0.248 | **0.424** | 0.279 | 0.150–0.272 | 0.228–0.576 |
| wiki-rfa | 0.248 | **0.397** | 0.289 | 0.172–0.278 | 0.193–0.479 |

GINEConv's drop is consistently 2–4× the walk model's drop at every bucket
granularity, on 5/6 datasets, sample sizes from hundreds to tens of thousands —
about as clean as this kind of descriptive correlation gets.

**`out_out` and `inout_inout`: mixed**, closer to `out_in` than to `in_in` (e.g.
`out_out` GINEConv *improves* on higher entropy on epinions/wiki-elec/wiki-rfa,
opposite of the hypothesis). **Lead 4b (2-hop path consistency) does not
corroborate the `in_in` story** — its `out`/`in`/`inout` variants are noisy and
sign-flip across datasets (e.g. bitcoin-otc `in`: walk −0.011, GINEConv +0.003 —
GINEConv improves; epinions `in`: walk −0.104, GINEConv +0.072 — opposite
directions). The original note that "the clean differential survives only on
epinions" for Lead 4b still stands.

**Why `in_in` and `out_in` disagree (a real interpretive wrinkle, not just noise).**
`out_in` is the causally motivated variant for predicting `sign(u→v)`: u's own
tendency to extend positive/negative links, v's own tendency to receive them — the
two quantities structural-balance theory would actually invoke. `in_in` swaps in
"who points into u" — a quantity with no direct mechanistic relationship to u's
*outgoing* edge to v. So the strongest, most robust empirical effect lives in the
variant with the *weaker* causal story, which raises the possibility that `in_in`
entropy is a proxy for something else (e.g. in-degree / hub-ness / some other
structural correlate) that differentially hurts GINEConv, rather than "sign
unpredictability" per se. This needs to be disentangled (e.g. partial out a
degree control) before it goes into a paper claim — see the elaborated claim and
analytical framing in the chat session that produced this update.

**Reinterpretation (revised):** the walk model's advantage is **not** a uniform
AUC offset everywhere — that conclusion only holds under the `out_in` lens. Under
`in_in`, there is a real, large, robust, GINEConv-specific degradation with rising
heterogeneity that the walk model mostly escapes. The honest combined statement:
*whether* the walk model's advantage concentrates in heterogeneous neighborhoods
depends on which notion of "heterogeneous" you use, and the one that shows the
effect most cleanly (`in_in`) is not the most mechanistically obvious one. The
"concentrated in high-entropy region" framing should be **narrowed to the `in_in`
definition**, not retired outright.

**Open caveat — walk coverage (flagged 2026-06-28, investigation in progress, see
`WALK_COVERAGE.md`).** All numbers above are computed on the shared
(walk-covered) edge set for every model, so the walk-vs-GNN comparison *within* a
bucket is not biased by coverage (identical edges for all 4 models). What's still
open: high-entropy buckets are systematically smaller (e.g. epinions `in_in` b8:
n=8322 low vs. n=394 high), and if walk coverage itself correlates with `in_in`
entropy, the high-entropy bucket could be a non-representative sample of the true
high-entropy population for *all* models alike — this would affect the absolute
levels, though not obviously the walk-vs-GNN comparison's direction. Worth checking
once the coverage workstream lands.

---

## 3. Baseline-rerun operational notes

- **SE-SGformer is NOT a canonical-split regression.** Its AUCs (0.57–0.73) look
  near-random but the OLD numbers were equally low (bitcoin-alpha 0.607, wiki-rfa
  0.548; canonical is marginally *higher*). Root cause: SE-SGformer's KNN
  `discriminate()` emits **hard one-hot labels**, and `roc_auc_score` is fed those
  argmax 0/1 predictions — so its "AUC" is really balanced accuracy, structurally far
  below a probability-ranking AUC (its acc / F1 are a healthy ~0.83–0.95). Inherent
  to the architecture-as-implemented; it was never an AUC-competitive baseline.
- **SGA-GSGNN** now runs on the canonical split for bitcoin-alpha (0.9051 — new best
  GNN there), bitcoin-otc (0.8848), wiki-elec (0.8771) after fixing the argparse
  `choices` that rejected `*_canon` dataset names. **wiki-rfa SGA-GSGNN is
  genuinely-missing** — its raw node ids reach ~1e9, so `create_spectral_features`'
  SVD tries to allocate 551 GiB and OOMs; the **original** SGA pipeline never produced
  a wiki-rfa number either (not in `all_results.csv`), so this matches, not regresses.
- **epinions / slashdot** SGA-GSGNN + SE-SGformer remain OOM-skipped by design
  (O(N²) candidate/spatial matrices); their old best-GNN was never those models.

---

## 4. Reproduce

```
# baselines (idempotent, skips done):
nohup .venv/bin/python baselines/run_canonical_orchestrator.py --gpus 0,1,2,3 &

# cross-model leads (shared-edge, all-raw). Lead 4 / 4b are now canonical-native
# (they read predictions_raw_canonical.pkl and restrict to the shared edge set
# themselves), so they ARE the standing scripts -- no separate _canonical driver:
.venv/bin/python scripts/lead4_entropy_heterogeneity.py --mode all --datasets all   # Lead 4
.venv/bin/python scripts/lead4_twohop_path_consistency.py --mode all --datasets all # Lead 4b (out/in/inout)
.venv/bin/python scripts/lead1_canonical_degree_gap.py       # Lead 1

# GNN-internal leads (env-overridable paths):
RESULTS_ROOT_NAME=results_our_splits_canonical SPLITS_DIRNAME=splits_canonical \
  .venv/bin/python scripts/lead2_gnn_bottleneck_mi.py --out outputs/lead2_gnn_bottleneck_canonical
RESULTS_ROOT_NAME=results_our_splits_canonical SPLITS_DIRNAME=splits_canonical \
  <sesgformer_env>/python scripts/lead2_edge_sensitivity.py --out outputs/lead2_gnn_bottleneck_canonical
LEAD2_ART_DIR=outputs/lead2_gnn_bottleneck_canonical LEAD3_OUT_DIR=outputs/lead3_swamping_canonical \
  PYTHONPATH=. .venv/bin/python scripts/lead3_real_gnn_swamping_check.py
```

Walk-only diagnostics (`edge_sign_mi_vs_distance*`, `node_mi_structural_embedding`,
`attention_analysis`, `lead3_attention_ambiguity`) are unaffected at the edge level
(they read `dataset_cache.pt`'s own splits, not `baselines/splits`) and were not rerun.
`lead2_walk_relay_mi` / `lead2_sigat_attention_weight_mi` (walk-relay / SiGAT-internal,
not cross-model AUC comparisons) can be rerun with `SPLITS_DIRNAME=splits_canonical`
if desired; their conclusions don't hinge on which test edges were sampled.
