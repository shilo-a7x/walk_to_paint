# Thesis plan and full EID results — 2026-10-13

Numbers: 10 seeds (42-51), test AUC, `func_logit_power`. Sources:
`experiments/edge_identity_tokens/EID_THESIS_RESULTS.md`, `thesis_figure_data/eid_thesis_*.csv`,
`thesis_figure_data/eid_entropy_heatmaps.csv`.

## 1. Status
- Thesis campaign (12 conditions x 6 datasets x 10 seeds = 720 runs): done 2026-10-09, 0 failures.
  Post-processing (Ablation B, F1, K-walks, paired tests) done.
- Missing ablation found 2026-10-13: full vs. local attention (WSDM abl:proximal). H3_FULL queued
  (60 runs, ~8h).
- Entropy heatmaps (H3 vs production vs SiGAT): data done, not plotted.
- Attention-direction + SHAP figure: not ported to EID yet.

## 2. Results

### 2a. Visibility grid (N = nodes, I = context-edge identity, S = context-edge sign)
| cond | N | I | S | b-alpha | b-otc | epinions | slashdot | wiki-elec | wiki-rfa |
|---|---|---|---|---|---|---|---|---|---|
| H3 | y | y | y | 0.9187 | 0.9355 | 0.9535 | 0.8962 | 0.8976 | 0.8921 |
| H3_NS | y | - | y | 0.9196 | 0.9356 | 0.9541 | 0.8963 | 0.8994 | 0.8909 |
| H3_NI | y | y | - | 0.9160 | 0.9312 | 0.9528 | 0.8964 | 0.8987 | 0.8914 |
| H3_N | y | - | - | 0.9165 | 0.9328 | 0.9530 | 0.8966 | 0.8991 | 0.8915 |
| H3_IS | - | y | y | 0.8920 | 0.9078 | 0.9122 | 0.8353 | 0.6523 | 0.7820 |
| H3_I | - | y | - | 0.8887 | 0.8994 | 0.9102 | 0.8390 | 0.5958 | 0.7366 |
| H3_S | - | - | y | 0.7294 | 0.7968 | 0.6820 | 0.6736 | 0.5683 | 0.5903 |
| H3_0 | - | - | - | 0.5832 | 0.5890 | 0.6472 | 0.5212 | 0.5483 | 0.5599 |
| prod. node-masking (= S only) | | | | 0.7382 | 0.7979 | 0.6819 | 0.6730 | 0.5697 | 0.5902 |

Paired vs H3 (pp): NS/NI/N within +-0.43 everywhere (only b-otc NI -0.43 p=.006); IS -2.7..-24.5,
I -3.0..-30.2, S -13.9..-32.9, 0 -30.6..-37.5 (all 0/10, p=.002).

### 2b. Other ablations
| cond | b-alpha | b-otc | epinions | slashdot | wiki-elec | wiki-rfa |
|---|---|---|---|---|---|---|
| sign scramble | -0.16 | -0.32* | -0.02 | +0.06 | +0.10 | -0.11 |
| identity scramble | -0.05 | -0.65 | +0.02 | +0.08 | +0.12 | -0.28 |
| direction flip | -0.52 | -0.65* | -0.69* | -2.06* | -2.89* | -3.44* |
| pair-only | -1.84* | -2.02* | -0.81* | -3.15* | -0.65* | -0.59* |
(* p<0.05 paired Wilcoxon vs H3.) Production direction flip: -0.9..-5.3pp.

### 2c. Headline
| | b-alpha | b-otc | epinions | slashdot | wiki-elec | wiki-rfa |
|---|---|---|---|---|---|---|
| H3 | 0.9187 | 0.9355 | 0.9535 | 0.8962 | 0.8976 | 0.8921 |
| production | 0.9134 | 0.9318 | 0.9536 | 0.8968 | 0.9023 | 0.8914 |
| H3 - prod | +0.53 | +0.38* | -0.02 | -0.06 | -0.47* | +0.07 |
| H3 - best baseline (10/10, p=.001 all) | +4.82 | +4.56 | +4.47 | +3.76 | +0.92 | +1.41 |
| pair-only - best baseline | +2.98* | +2.54* | +3.66* | +0.60* | +0.27 (p=.08) | +0.81* |
| H3 edge macro-F1 | 0.756 | 0.813 | 0.850 | 0.773 | 0.771 | 0.759 |
| H3 K=1 walk | 0.9056 | 0.9274 | 0.9475 | 0.8875 | 0.8892 | 0.8853 |
| aggregator spread (11 funcs) | 0.12pp | 0.07pp | 0.24pp | 0.02pp | 0.04pp | 0.05pp |
Mean gain over best baseline: 3.32pp (WSDM: 3.3pp).

Entropy heatmap (4x4, >=50 edges/cell): H3 > SiGAT in 4/8, 10/15, 16/16, 15/16, 8/16, 11/16 cells
(mean +2.8, +6.1, +5.1, +2.5, +0.5, +0.7pp); H3 - production mean cell diff within +-1.4pp.

## 3. What the results say
1. Identity is the dominant signal: nodes alone ~ full model; without nodes, edge identity
   alone recovers most of it (IS/I), signs alone recover much less (S).
2. Node and edge identity are redundant with each other; context signs are redundant given
   either identity, but carry real signal alone (S - 0: +14.6, +20.8, +3.5, +15.2, +2.0, +3.0pp).
3. Edge identity adds no accuracy to the full model (H3 ~ H3_NS on all 6).
4. Local neighbourhood beyond the pair matters (pair-only -0.6..-3.2pp, all significant), and
   direction matters (flip -0.5..-3.4pp) -> local directed structure is used.
5. Yet pair-only already beats the best baseline on 5/6 -> most of the gain over GNNs comes from
   how vertex representations are learned (free, edge-supervised through local masked-sign
   prediction, not WL-bound), with local context adding the rest.
6. EID's code reproduces production: S-only = production node-masking (within 0.1pp on 5/6).

## 4. Framing options
- A. Same paper, EID as the model: keep WSDM claims, swap Table 1 to H3, replace the
  commented-out node/edge ablations with the grid. Must soften "conditions on signed edges".
- B. Production stays the headline model; EID is the analysis instrument: Table 1 unchanged
  (production 10-seed), new chapter "What Pewter uses" built on the EID grid. Most honest given
  finding 3 (identity adds no accuracy) and the smallest rewrite.
- C. Bold reframe "identity and local structure, not signs" as the thesis centre.
Recommendation: B (or A if the user wants EID as the single model); finding 5 must be reported
either way, and the contribution wording shifts from "edge-local signed context" to
"edge-supervised representations + local directed context".

## 5. Forecast WSDM -> thesis diff (section by section)
- Abstract/Intro/Contributions: add the grid finding; soften signed-context claim; keep 3.3pp.
- Related work: + per-edge/entity identity embeddings, transductive embeddings vs WL.
- Theory (Props, proofs): unchanged.
- Methods: tokenizer paragraph (identity channel + separate sign channel) and holdout paragraph
  (H3 policy) if A; unchanged if B (EID described in the analysis chapter). Pipeline schematic
  redrawn under A.
- Empirical evidence (Fig 1 A-E): unchanged (baseline-side only).
- Table 1: H3 numbers (A) or unchanged (B); + F1 column.
- Delta heatmap: H3 vs SiGAT (A) or unchanged (B).
- Ablations: proximal (H3_FULL, running), aggregator, single-walk, direction flip re-based on H3;
  new: visibility grid, pair-only, sign/identity scramble.
- Attention section: re-extract on H3 (A) or unchanged (B).
- New chapter/subsection: what Pewter uses (grid, redundancy, pair-only, direction).
- Discussion/limitations: identity adds no accuracy; wiki-elec gap; transductive-only.

## 6. Adopted EID version: H3
Production per-dataset config untouched; rank-16 per-edge identity table + shared Linear(16->d)
added to the content vector; sign as a separate 3-value channel (neg/pos/NA) embedded and added;
target: identity <MASK>, sign NA, always; dynamic resplit as production; edge-identity replacement
p=0.2 (70% <UNK>); identity-table weight decay 0.03; VAL/TEST edges hidden and blocked from
attention while not yet allowed; when allowed as context (VAL edges at the test stage) shown as
<UNK> identity + real sign.

## 7. Making EID canonical (plan, not done)
- Tokenizer/vocab: specials + node tokens + per-edge identity ids (edge id = index into the edge
  list); no E_+/E_- tokens and no old_vocab_size offset concept.
- Data prep: fold build_cache.py into src/data/prepare_data.py -> one cache per
  (dataset, budget, seed) with flat_node_or_edge_ids, flat_sign_ids, flat_edge_ids,
  flat_split_mask. Walks and splits unchanged (same sampler and seed).
- Dataset: one visibility function implementing the H3 policy per stage.
- Model: separate node table and edge-identity table (rank r, r=0 => no identity = production
  semantics), sign embedding; drop dead options (concat, node_embed_dim, residual baseline,
  reveal_holdout*, mask_target_identity since always on).
- Lit model + posthoc: one shared `apply_input_transforms(batch, stage, cfg)` used by both
  training and posthoc (removes the bug class that hit us twice: ablations not reapplied in
  posthoc).
- Ablation flags renamed to the grid: hide_nodes, hide_context_identity, hide_context_sign,
  scramble_signs, scramble_identity, randomize_walk_direction, pair_only_attention.
- Equivalence tests: new code reproduces H3 per seed within run-to-run noise, and r=0
  reproduces production; old caches kept until verified.

## 8. Figures (300 dpi)
All 4 WSDM PNGs verified 300 dpi. Thesis: Fig 1 A-E reuse; delta heatmap plot (data ready);
attention+SHAP figure needs EID port; new grid figure; pipeline schematic redraw (A only).

## 9. Left to do
1. H3_FULL (running). 2. Port attention+SHAP extraction to EID and run. 3. Thesis plot scripts
(300 dpi). 4. Pick framing (user + professor). 5. Create thesis/ and apply the diff.
6. Canonical refactor (can follow the writing). 7. Optional: topology bump dedup test.

## 10. Update 2026-10-13 (evening): all runs done
- H3_FULL 60/60 ok; collector rerun (EID_THESIS_RESULTS.md, eid_thesis_*.csv include H3_FULL).
  Test AUC full / local (H3): b-alpha .9155/.9187, b-otc .9309/.9355*, epinions .9499/.9536*,
  slashdot .8989/.8962* (full better), wiki-elec .8972/.8976, wiki-rfa .8928/.8921 (* p<.01 paired).
  Production for reference full/local: .9146/.9134 .9315/.9317 .9523/.9536 .8989/.8968 .9008/.9023 .8923/.8914.
- Attention/SHAP on H3 (seed 42; SHAP removal = identity <MASK> + sign hidden, option (a), pending
  user sign-off) done for all 6; thesis/figures/attndir_panels_abcd_combined.png rendered, 300 dpi.
  fwd/bwd mass: b-alpha .426/.443, b-otc .511/.393, epinions .498/.386, slashdot .270/.243,
  wiki-elec .435/.377, wiki-rfa .457/.394 -> 5 forward-dominant, 1 backward (production: 2 fwd, 4 bwd).
  node/edge mass: node-dominant on all 6 (production: 4 node, 2 edge).
  SHAP: hop1 > hop2 on 6/6 both directions (WSDM claim holds); hop-1 fwd > bwd on all 6, so the
  WSDM "attention and SHAP disagree on the wikis" sentence no longer applies (they now agree).
  -> the WSDM attention paragraphs need their dataset lists rewritten, not just numbers.
- Bump dedup: see outputs/topology_mi_vs_distance/FINDINGS.md (slashdot-undirected and
  wiki-directed bumps are real, not reuse; bitcoin far hops are artifacts).

## 11. Note 2026-10-20: identity-scramble ablation is uninformative by construction
`model.scramble_edge_identity` maps every edge to a fixed substitute identity (a derangement) consistently
in training AND evaluation -> a pure relabeling of identity-table rows, no information removed; its null
result is guaranteed. A valid test would scramble at evaluation only, or re-randomize per occurrence.
Sign scramble (fixed 50% liars, train+eval) makes the presented sign independent of the true sign ->
equivalent to hiding context signs (grid row "vertices + identities"). Both left out of the thesis.
