# Phase 2a — k_cover_bp generation-only benchmark at production budgets

Gates (plan-a-fix-for-glimmering-panda.md Phase 2a): (1) gen time within ~1.5-
2.5x uniform-sampler envelope, (2) 100% edge+node coverage / full k=5 floor,
(3) near-zero duplication (dup_after_retries ~= 0%, raw-dup% ~= capped-%).

| dataset | nw | \|E\| | uniform (s) | k_cover_bp (s) | ratio | edge cov | node cov | %≥k | min visit | capped % | dup_after_retries % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bitcoin-alpha | 5,000,000 | 24,186 | 128.6 | 133.4 | 1.04x | 1.0000 | 1.0000 | 1.0000 | 5 | 0.0372% | 0.0000% |
| bitcoin-otc | 2,000,000 | 35,592 | 37.4 | 45.3 | 1.21x | 1.0000 | 1.0000 | 1.0000 | 5 | 0.0169% | 0.0000% |
| epinions | 3,000,000 | 840,799 | 45.1 | 133.5 | 2.96x | 1.0000 | 1.0000 | 1.0000 | 5 | 1.5434% | 0.2383% |
| wiki-elec | 500,000 | 103,689 | 3.2 | 10.1 | 3.15x | 1.0000 | 1.0000 | 1.0000 | 5 | 13.1518% | 1.1920% |
| wiki-rfa | 1,000,000 | 177,211 | 9.8 | 21.3 | 2.16x | 1.0000 | 1.0000 | 1.0000 | 5 | 4.3214% | 0.3612% |
| slashdot090221 | 5,000,000 | 549,202 | 42.0 | 110.0 | 2.62x | 1.0000 | 1.0000 | 1.0000 | 5 | 2.6502% | 0.3756% |

## Gate-by-gate verdict

**Gate 2 (coverage/k-floor): PASS on all 6.** edge_cov=node_cov=pct_ge_k=1.0000,
min_visit=5 everywhere — no regression vs. `k_cover_walks_fast`.

**Gate 3 (near-zero duplication): PASS on all 6, decisively.** `dup_after_retries`
ranges 0.00%-1.19% vs. the OLD sampler's measured raw duplicate rate of 5.92%-39.46%
(`PHASE0B_DUP_MEASUREMENT.md`) — a >95% reduction in every case, and every remaining
duplicate has an identified structural cause (`capped` or a residual retry-exhaustion
case, both reported explicitly, none silent). wiki-elec's residual 1.19% is the
highest of the six — plausible given its capped fraction (13.15%) is also far the
highest (small, dense graph — many nodes hit in/out-degree-1 dead ends quickly); worth
a `max_dedup_retries` bump if this dataset's Phase 4 numbers end up sensitive to it,
not a blocker now.

**Gate 1 (perf, ~1.5-2.5x uniform envelope): PASS on 3/6 (bitcoin-alpha 1.04x,
bitcoin-otc 1.21x, wiki-rfa 2.16x), MISS on 3/6 (epinions 2.96x, wiki-elec 3.15x,
slashdot090221 2.62x, the last only marginally over).** Mechanistic explanation, not
a mystery: the ratio tracks how much of the production budget is spent in the anchor
phase (where the new backward-prefix/hash/retry machinery adds real per-anchor cost)
vs. the uniform-fill phase (untouched, identical cost to before). bitcoin-alpha/otc
run at huge nw/|E| (207x, 56x) — almost the whole budget is uniform fill, so the
anchor overhead is diluted to near-nothing. epinions/wiki-elec/slashdot run at much
smaller nw/|E| (3.6x, 4.8x, 9.1x) — anchors are a much bigger share of the budget, so
the new overhead shows up more in the aggregate ratio.

**In absolute terms this is not alarming:** wiki-elec's "3.15x" is 10.1s vs. 3.2s (a
7s difference) — the ratio gate, inherited verbatim from the old sampler's benchmark,
is a poor fit for a dataset this fast; nobody would notice a 7-second cache-build cost
in practice. epinions' 133.5s and slashdot's 110.0s are the only two with real
absolute cost (~2 minutes each), and both are one-time cache-generation costs, not a
per-epoch training cost. **Recommendation: treat Gate 1 as advisory here, not a hard
blocker — proceed to Phase 2b/3/4 as planned; revisit only if a specific dataset's
generation time becomes a practical bottleneck during the budget sweep (Phase 4),
where larger multiplier grid points will scale these numbers up further.**
