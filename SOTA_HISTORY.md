# SOTA / results history (archived detail)

Archived out of `CLAUDE.md` on 2026-08-18 to keep the live file short. Nothing here is
current guidance — it's provenance for numbers/decisions CLAUDE.md now only summarizes.
If anything below conflicts with CLAUDE.md, CLAUDE.md wins.

## Old single-split SOTA table (E25/E26/E31/E32) — superseded 2026-08-10

The table below is single-split, kept for history only — the WSDM paper's Table 1
(`aaai2027/WSDM_format_revised.tex`) no longer uses it; it uses the 10-seed mean±std
table in CLAUDE.md's "Current SOTA" section instead.

**Table below reflects the adopted `edge_cover` sampler** at each dataset's production
`num_walks` budget (see "Walk sampler" in CLAUDE.md) — confirmed reproducible via a plain
`dataset.name=<ds>` run, since `edge_cover` and its adopted budget are already the default
in every `configs/<dataset>.yaml`. Old uniform-E14 numbers in parentheses (had the ~85–88%
coverage caveat on the sparse graphs). The baseline column shows the **canonical-split**
best GNN (re-run on the unified walk-derived split, `baselines/all_results_canonical.csv`);
pre-canonical best-GNN in its own parentheses. **On the identical shared test edges the walk
model beats EVERY GNN on all 6 datasets, both attention variants** (apples-to-apples; full
matched table in `outputs/walk_coverage_analysis/E15_SWEEP_RESULTS.md`, older canonical
detail in `CANONICAL_RERUN_FINDINGS.md` §1a). With full coverage the walk now also wins on
each-model's-own-full-test on all 6 — the prior wiki-elec/wiki-rfa "SiGAT marginally higher"
exception was purely a coverage artifact and is gone.

| Dataset         | Canon best-GNN (old)              | Full attn, no-H (E25/E26) | LocalAttn4, no-H (E32) — **current default** |
|-----------------|-----------------------------------|-----------------------------|------------------------------------------------|
| bitcoin-alpha   | 0.9051 SGA-GSGNN (0.8804)         | **0.9219**                  | 0.9188                                          |
| bitcoin-otc     | 0.8972 SNEA (0.9086)              | 0.9311                      | **0.9358**                                      |
| epinions        | 0.9146 SiGAT (0.9113)             | 0.9527                      | **0.9535**                                      |
| wiki-elec       | 0.8930 SiGAT (0.8840)             | 0.9036                      | **0.9075**                                      |
| wiki-rfa        | 0.8831 SiGAT (0.8673)             | 0.8930                      | **0.8976**                                      |
| slashdot090221  | 0.8587 SiGAT (0.8845)             | **0.9007**                  | 0.8981                                          |

**Corrected 2026-08-04/2026-08-05: `E31_PY314_MIGRATION` is full attention, not
LocalAttn4.** Root cause: `config.yaml`'s `model.local_attention_window` default was
`null`, not `4`, so the plain documented training command silently trained full
attention — confirmed by inspecting the checkpoint config directly
(`local_attention_window=None` on all 6) and by the AUCs matching the Full-attn
(E25/E26) column almost exactly. Fixed at the root: `config.yaml`'s default is now
`local_attention_window: 4` (full attention now requires an explicit
`model.local_attention_window=null`). `E31_PY314_MIGRATION` checkpoints are kept on
disk (`outputs/<ds>/E31_PY314_MIGRATION_*`) as an incidental extra full-attention data
point under the Python 3.14 stack — not adopted into this table's Full-attn column,
which stays on its original Python-3.9-stack (E25/E26) numbers; do not use E31 as
LocalAttn4 for anything.

**`E32_PY314_LOCALATTN4`** is the corrected, properly-configured post-migration
LocalAttn4 retrain (verified via direct checkpoint inspection to carry
`local_attention_window=4` on all 6 datasets) and is what the LocalAttn4 column above
reflects. Deltas vs. the old Python-3.9-stack E27 numbers: alpha +0.62pp, otc -0.32pp,
epinions +0.02pp, wiki-elec +0.14pp, wiki-rfa +0.17pp, slashdot090221 0.00pp — all
within the established noise band; Local-vs-Full keeps the same 4/6-win split as
before (otc, epinions, wiki-elec, wiki-rfa win for Local; alpha, slashdot090221 win
for Full).

Entropy-hardness numbers (E28, no longer recommended) for provenance: alpha 0.9184,
otc 0.9284, epinions 0.9535, wiki-elec 0.9064, wiki-rfa 0.8966, slashdot 0.8976 — flat-
to-negative vs. no-H on 5/6.

Apples-to-apples (shared edges) best GNN is always lower still — e.g. epinions GINE
0.8642 / SiGAT 0.9109, slashdot GINE 0.7869 / SiGAT 0.8571. SE-SGformer excluded from
"best GNN": its KNN discriminator emits hard labels, so its AUC is really balanced
accuracy (~0.57-0.73).

Experiment tags: `E25_BUDGET_SWEEP`/`E26_WIKI_SWEEP` (full attention, Python-3.9-stack
— this table's canonical Full-attn column) and `E32_PY314_LOCALATTN4` (LocalAttn4,
Python-3.14-stack, canonical as of 2026-08-05), both on the `edge_cover` sampler at
each dataset's production `num_walks` (isolated keyed caches
`data/<ds>/dataset_cache__edge_cover_nw<nw>_mw80_seed42.pt`, environment-independent,
reused across the migration). Prior LocalAttn4 tag `E27_NOHARD_EDGECOVER_LOCALATTN4`
(Python-3.9-stack) and prior `k_cover`-sampler SOTA (E15/E14) kept for provenance only
— `git log` this file or `HARDNESS_MINER_ROADMAP.md`'s history.

**Paper figures/tables rebuilt 2026-08-05** against the corrected checkpoints (full
attn = `E31_PY314_MIGRATION`, LocalAttn4 = `E32_PY314_LOCALATTN4`) — Result 1 table,
Result 2 heatmap, Attention Directionality figure, Ablations A and B (formerly C). See
`aaai2027/PEWTER_ASSETS_CHECKLIST.md` rows #20-25 for scripts/data pointers.

## 10-seed migration and Table 1 baseline history (2026-08-10)

**WSDM Table 1 baseline rows added 2026-08-10:** added GCN, GAT, SGCN, GSGNN, SGA,
and **SiGAT** using numbers the user supplied from external publications (not reproduced
locally for GCN/GAT — no local run exists at all; SGCN/GSGNN are distinct from this
repo's curriculum-augmented CSG/CSG-GSGNN reproductions, not treated as equivalent per
explicit user call, "for now we dont include the curriculum"). **SiGAT correction
(2026-08-10, same day):** Table 1's SiGAT row initially showed our own single-split
canonical reproduction (0.882/0.876/0.915/0.893/0.883/0.859) instead of the published
numbers the user had already supplied in the same message as GCN/GAT/SGCN/GSGNN/SGA —
caught by the user ("why do you need SiGAT? i already gave the nums from published"),
fixed by switching the row to the published values (0.855/0.883/0.891/0.880/0.871/0.846,
±SE as given). Our own local SiGAT reproduction is still used elsewhere and stays on
disk — it's the per-edge-prediction source for the entropy-vs-AUC analyses (Empirical
Confirmation panels, Attention Directionality) in the WSDM paper, which need real
per-edge predictions a published aggregate AUC can't provide; it's just no longer what
Table 1 itself reports for SiGAT, matching the other published-only rows. CSG and
CSG-GSGNN removed from the table (not deleted from the repo — `baselines/CSG/
results_our_splits_canonical/` still has the single-split numbers) pending their own
10-split rerun, for consistency with the same standard now applied to PEWTER/GINEConv.
SNEA/CopulaLSP remain single-split for now; a 10-split campaign for both was launched the
same day (`scripts/run_multiseed_snea_copulalsp.py`, reuses `baselines/prepare_splits.py`'s
new `save_canonical_split_for_seed()` — same per-seed canonical-split fix built for
GINEConv's multiseed run). Both SNEA and CopulaLSP are undirected/pair-level models in
this codebase's implementation (`run_with_our_splits.py` builds `uni_edge_index` for
both), confirmed not a leakage/apples-to-apples problem: `prepare_splits.py`'s
`uni_tst_mask` only marks a canonical pair "clean test" if every real direction of it is
in the walk model's own test split, asserted in `_assert_canonical`.

**Sanity check against published numbers (2026-08-10):** the one clean same-model match
available, raw SiGAT (our canonical reproduction vs. the user's supplied published
SiGAT), came back within 0.7–3.7pp on all 6 datasets — no red flags. No such comparison
was attempted for SGCN/GSGNN vs. CSG/CSG-GSGNN (different models, curriculum added) or
for GCN/GAT (no local run exists).

**SiGAT 10-split campaign (launched 2026-08-10, DONE same day, 54/54 jobs, 0 failed).**
Motivation: the entropy-vs-AUC binned heatmaps (`scripts/paper_figures/
extract_empconf_panelC_gnn_entropy_heatmap.py`, `extract_attndir_panelD_pewter_entropy_
heatmap.py`) need real 10-split SiGAT predictions, not just a 10-split aggregate AUC, so
the SiGAT reproduction had to be re-run regardless of what Table 1 showed. Driver:
`scripts/run_multiseed_sigat.py` (54 new jobs: 6 datasets × 9 new seeds, seed 42 reuses
the existing canonical reproduction), same `save_canonical_split_for_seed()` infra as
SNEA/CopulaLSP, `baselines/SGA/run_with_our_splits.py` (env `sga_env`). Notably higher
than the published numbers on 5/6 datasets (epinions +1.8pp) — not flagged as a red flag,
just a real split/hyperparameter difference between our canonical-split reproduction and
the original paper's own split.

## Entropy-heatmap multi-split methodology (decided 2026-08-10)

Two ways to combine the 10 SiGAT/PEWTER splits into one binned heatmap were discussed:
(1) pool all 10 splits' test predictions per cell then compute one AUC (uses ~10x the
edges per cell, the only option that can de-noise the naturally-thin high-entropy "hard
node" corner cells, but mixes predictions from 10 different trained model instances);
(2) compute each split's cell AUC independently on its own ~10% slice, then average —
matches the mean±std convention already used everywhere else in this paper (Table 1,
GINEConv), but doesn't fix small-N noise in rare cells. **User picked (2) explicitly,
"much simpler and reliable."** The fixed 4×4 entropy bins (`empconf_panelC`) are
bin-edges-on-entropy-value, not percentile-based, and the per-node entropy values
(`src_ent`/`tgt_ent` in `lead4_entropy_heterogeneity.py`) are computed once from the fixed
real dense edge set, independent of split — so averaging the per-cell AUC across the 10
splits is a straightforward loop-and-average over the existing binning code.

Extract: `scripts/paper_figures/extract_multiseed_entropy_heatmaps.py` — pulls fresh
per-seed predictions for both models (SiGAT: `sigat_raw_seed()`, fits a fresh
LogisticRegression per seed; PEWTER local: `walk_raw_seed()`, mean-prob aggregation over
each edge's walk occurrences). Seed 42 needed special-casing (predates the `MULTISEED_*`
naming, pulls from `attention_directionality.py`'s `LOCAL_RUN_INFO` pins instead) — a
first pass silently found 0 matches for it and averaged over only 9/10 splits before this
was caught. Output: `aaai2027/figure_data/empconf_panelC_sigat_10split.csv`,
`aaai2027/figure_data/pewter_sigat_delta_heatmap.csv`. Plot:
`scripts/paper_figures/plot_multiseed_entropy_heatmaps.py` → diverging `RdBu` (blue =
PEWTER higher, red = SiGAT higher — first render used `RdBu_r`, caught and flipped).
**Finding:** PEWTER (local) beats SiGAT in nearly every cell on 5/6 datasets
(bitcoin-otc's high-entropy corner is the largest gap, +0.26 AUC); bitcoin-alpha is mixed
— PEWTER wins the low-entropy corner (+0.16) but loses (low-src, mid-tgt) (-0.12).

Figure 1's Panel C was updated in place with this 10-split SiGAT data (2026-08-10, no new
figures, no tex changes) — GINEConv's cells stayed single-split (already dropped from
this panel's plot on 2026-07-27). Rerun order: extract → `plot_empconf_panelC_gnn_
entropy_heatmap.py` → `combine_empconf_panels_abcde.py`.

## Readability pass (2026-08-10)

Several panel scripts had long in-image titles duplicating the external LaTeX caption,
eating vertical space that forced small fonts once compressed into the combined grid.
Shortened titles / raised font sizes in `plot_empconf_panelC_gnn_entropy_heatmap.py`,
`plot_empconf_panelD_signagreement_auc.py`, `plot_empconf_panelE_coefficients.py`,
`plot_attndir_panelA_headgrid.py`, `plot_attndir_panelB_direction.py`,
`plot_attndir_panelC_nodeedge.py`. Panel C needed the biggest bump (title 12→17pt, cell
text 9.5→13pt) since it's a 6-column-wide panel compressed harder than narrower ones.

## SHAP edge directionality figure (2026-08-10)

`scripts/shap_edge_directionality.py`, `scripts/paper_figures/{extract,plot}_shap_edge_
directionality.py`, `aaai2027/figures/shap_edge_directionality.png`. Companion to the
Attention Directionality figure — measures actual causal contribution (exact Shapley, not
raw attention weight) of each context edge to a masked target edge's predicted
P(positive), on the LocalAttn4 checkpoint, restricted to the local attention window (two
hops each side = 4 context-edge features, ≤16 subsets per instance, exact not sampled).
Masking reuses the model's own `<MASK>` token. Efficiency-property check
(`sum(shap) == value(full) − value(empty)`) passed to float precision (~1e-16) on all 6
datasets. **Finding:** mean |SHAP| decays from hop 1 to hop 2 on all 6/6 datasets, both
directions, well outside cluster-robust SEs. Direction asymmetry splits into two groups:
bitcoin-alpha/bitcoin-otc/epinions show forward≈backward; slashdot090221/wiki-elec/wiki-rfa
show forward > backward at hop 1. Raw results:
`outputs/shap_edge_directionality/shap_directionality_<ds>_result.pkl`; summary:
`aaai2027/figure_data/shap_edge_directionality.csv`. Placement in the paper not yet
decided.

## Attention variant full-vs-local — E30 pilot detail (2026-07-19/20)

Full derivation behind the verdict now stated compactly in CLAUDE.md's "Attention variant"
section.

**E30 pilot** tested the paper's own proposed cleaner ablation — literally shortening
`dataset.max_walk_length` instead of masking attention within a long walk — on
bitcoin-alpha + epinions, full attention, no hardness, `max_walk_length` ∈ {2, 4, 8, 16}:

| dataset | L=2 | L=4 | L=8 | L=16 | L=80 (reference, E25/E26) |
|---|---|---|---|---|---|
| bitcoin-alpha | 0.9112 | 0.9196 | 0.9200 | **0.9305** | 0.9219 |
| epinions | 0.9480 | 0.9515 | **0.9528** | 0.9491 | 0.9527 |

L=8/L=16 looked like a clean win for short-walk truncation over LocalAttn4 (matches or
beats L=80 at 1/10th the length). But digging into *why* L=2 underperforms found the
walk-length axis is confounded at the short end, and the confound gets worse — not better
— the closer you push toward LocalAttn4's own ±2-hop span.

**The confound:** dynamic masking selects this epoch's supervised targets *globally* per
edge_id (`_sample_epoch_targets`, `src/model/lit_model.py:237`), not per-walk — so a 2-edge
walk has a real chance none of its edges are selected this epoch (measured: 43.9%/45.3% of
L=2 walks on alpha/epinions contribute zero gradient). Raising `model.dynamic_target_ratio`
(an override added for this investigation, decoupled from `dataset.mask_ratio`/`train_ratio`)
fixes that for free — and L=2 bitcoin-alpha jumps from 0.9112 to **0.9251** (ratio=0.7),
beating the L=80 reference.

**But that "fix" trades one problem for a worse one.** Measured directly against the real
cached walk data (`scripts/measure_local_context_availability.py`): a 2-edge walk
essentially never has room for both a target *and* a labeled neighbor edge to condition on.
At the untouched baseline (ratio=0.4), a target edge already has zero labeled context 43.2%
of the time on bitcoin-alpha (54.6% on epinions) purely because there usually isn't a
second pool edge in such a short walk — raising the ratio to fix the empty-walk waste makes
this *strictly worse* (67% zero-context at ratio=0.7, 100% at ratio=1.0). So an
L=2-truncated walk cannot cleanly demonstrate "the model conditions on nearby labeled
edges." L=8/L=16 are NOT affected by this (only 0.5–2.1% of walks have zero context
anywhere at those lengths) — their sufficiency result stands unconfounded; it's specifically
the short end that breaks, which happens to be exactly the length range that would make the
cleanest paper story.

Full context-availability table (`scripts/measure_local_context_availability.py`,
ratio=0.4 default throughout):

| L | bitcoin-alpha: local-window context available | bitcoin-alpha: zero context anywhere | epinions: local-window context | epinions: zero context anywhere |
|---|---|---|---|---|
| 2 | 56.8% | 43.2% | 45.4% | 54.6% |
| 4 | 78.2% | 15.5% | — | — |
| 8 | 84.8% | 2.1% | 82.3% | 5.9% |
| 16 | 88.0% | 0.5% | — | — |
| 80 | 90.6% | 0.1% | — | — |

## Hardness map paths (historical — H is scrapped, not used anywhere as of 2026-07-19)

Kept for provenance only. Do not point any new run at these.

`outputs/<dataset>/E22_HARDNODE_ENTROPY/hardness_{source,target}.pt` — role-aware entropy
maps, the best-quality map found (`HARDNESS_MINER_ROADMAP.md`), used in the final E28/E29
ablation that led to the scrap decision.
`outputs/<dataset>/E17_HARDNODE_KCOVER_REMINE/hardness_map.pt` — old learned-miner map,
re-mined on the (now superseded) `k_cover` sampler.
`outputs/transformer_incremental/bitcoin-alpha_.../artifacts/E14_HARDNODE_L10/hardness_map.pt`
— original pre-`k_cover` miner map (others: same structure under each dataset's run dir).
