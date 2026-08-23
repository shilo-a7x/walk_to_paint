# abl:masknode / abl:maskedge — true (bug-fixed) numbers, held out of the paper pending SIGNSCRAMBLE

**Status: the paper currently states the pre-fix numbers deliberately** (`aaai2027/WSDM_format_revised.tex`'s `abl:maskedge` paragraph), pending one more ablation
(SIGNSCRAMBLE — see below) before revisiting the "does edge-sign context help" conclusion.
This file is the correct, verified record of what the numbers actually are, kept separate
so it doesn't get lost.

## The numbers

Root cause and fix: `src/training/callbacks.py`'s `PerEpochPredictionSaver._extract_predictions`
never reapplied `mask_node_tokens`/`mask_edge_tokens` before the forward pass at posthoc-eval
time (training was always correct; only evaluation was blind to the ablation). Fixed, all 120
checkpoints (6 datasets × 10 seeds × 2 ablations) re-evaluated (posthoc only, no retraining).
Full bug narrative: `PAPER_CLOSEOUT_LOG.md`'s 2026-08-23 correction entry. Extraction script:
`scripts/paper_figures/extract_ablation_masknode_maskedge_significance.py`. Data:
`aaai2027/figure_data/ablation_masknode_maskedge_significance.csv`.

| dataset | local AUC | masknode AUC | masknode Δ (pp) | masknode p (one-sided) | maskedge AUC | maskedge Δ (pp) | maskedge p (one-sided) |
|---|---|---|---|---|---|---|---|
| Bitcoin-alpha | 0.9134 | 0.7382 | −17.52 | 0.00098 | 0.9133 | −0.02 | 0.216 |
| Bitcoin-otc | 0.9318 | 0.7979 | −13.39 | 0.00098 | 0.9319 | +0.01 | 0.237 |
| Epinions | 0.9536 | 0.6819 | −27.17 | 0.00098 | 0.9530 | −0.06 | 0.238 |
| Wiki-elec | 0.9023 | 0.5697 | −33.26 | 0.00098 | 0.9026 | +0.02 | 0.688 |
| Wiki-RfA | 0.8914 | 0.5902 | −30.13 | 0.00098 | 0.8916 | +0.02 | 0.423 |
| Slashdot | 0.8968 | 0.6730 | −22.38 | 0.00098 | 0.8963 | −0.05 | 0.069 |

- **`abl:masknode`**: unchanged conclusion from the paper's current text (vertex identity carries
  the overwhelming majority of the signal), magnitudes shift somewhat (13.4–33.3pp true vs.
  14.6–33.7pp as currently stated in the tex) since the bug's impact on masknode wasn't uniformly
  negligible across datasets — only the one seed-42/wiki-elec spot check done mid-session
  suggested "negligible," which did not generalize.
- **`abl:maskedge`**: real change. Paper currently states −1.0 to −1.6pp, "significant on all
  six, $p\le0.032$." True numbers: every delta is within ±0.06pp and **none reach $p<0.05$** —
  no dataset shows a statistically detectable effect from removing edge-sign context, once
  evaluated correctly.

## Why this isn't in the paper yet

Per explicit decision (2026-08-23): keep the paper's current "edge context helps a bit" framing
(the pre-fix numbers) until a new ablation — **SIGNSCRAMBLE** (`model.scramble_edge_signs`,
implemented in `src/model/lit_model.py`, see design/launch status in
`PAPER_CLOSEOUT_LOG.md`) — gives a second, independent read on whether edge-sign *correctness*
matters. `mask_edge_tokens` removes the edge token's identity entirely (replaced with
`<MASK>`), which conflates two different questions: "does an edge token being present at all
matter" (structural/positional signal) vs. "does its specific sign value matter." SIGNSCRAMBLE
isolates the second question directly by keeping a real (but decorrelated-from-truth) sign
value in place of the true one. Once SIGNSCRAMBLE's numbers land, revisit both this file and
the tex's `abl:maskedge` paragraph together, informed by both ablations rather than one.
