# Walk-coverage Phase 0 — read-only analysis findings (2026-06-28)

All numbers computed faithfully from the exact production CSR walk caches
(`data/<ds>/dataset_cache.pt`) the SOTA runs used, plus the canonical GNN
artifacts. Scripts: `scripts/walk_coverage_analysis.py` (0.A–0.D),
`scripts/walk_coverage_bias.py` (0.E), `scripts/walk_variance_probe.py` (0.F).
Raw JSON: `coverage_phase0.json`, `bias_phase0E.json`.

## 0.A — Actual budgets (confirmed)
bitcoin-alpha = **5,000,000**, slashdot090221 = **5,000,000**; bitcoin-otc,
epinions, wiki-elec, wiki-rfa = **500,000**. All `max_walk_length = 80`.
(Matches the user's statement; configs' `# was N` comments are stale.)

## 0.B — Edge coverage per split (reproduces WALK_COVERAGE.md exactly)
Coverage is essentially identical across train/mask/val/test within a dataset.

| dataset | budget | test cov | +cov | -cov | sign gap |
|---|---|---|---|---|---|
| bitcoin-alpha | 5M | 1.000 | 1.000 | 1.000 | 0.0 |
| bitcoin-otc | 500k | 1.000 | 1.000 | 1.000 | 0.0 |
| epinions | 500k | 0.878 | 0.869 | 0.929 | **−6.0pp** (neg better) |
| wiki-elec | 500k | 0.854 | 0.849 | 0.873 | −2.4pp (neg better) |
| wiki-rfa | 500k | 0.862 | 0.864 | 0.856 | +0.8pp |
| slashdot | 5M | 0.983 | 0.986 | 0.970 | +1.6pp |

**Sign finding:** the prior session's "11.9pp negative-under-coverage gap" was a
500k artifact (and was computed with a neg=label-0 bug). At production budget the
sign gap is small and *inconsistent in direction* — on epinions/wiki-elec
negatives are actually BETTER covered. Sign bias is not a real concern.

## 0.C — Node coverage (new)
`frac_appear_as_token` == `frac_with_incident_covered_edge` on every dataset (no
dead-end-start-node gap). Key contrast with edge coverage:

| dataset | node cov | edge cov | reading |
|---|---|---|---|
| epinions | 0.972 | 0.878 | 2.8% of nodes never visited (≈ isolated/tiny-component) + edge under-sampling |
| wiki-elec | 0.9987 | 0.854 | **nodes ~fully visited, edges only 85%** → pure edge under-sampling between visited nodes |
| wiki-rfa | 0.9997 | 0.862 | same — under-sampling, not unreachability |
| slashdot | 0.9954 | 0.983 | 0.46% nodes unvisited |

**Implication:** for wiki the coverage gap is NOT a reachability problem — the
nodes are all visited, specific edges between them just aren't traversed. This is
exactly what edge-anchored sampling (k_cover) fixes; raising node coverage / LCC
would do nothing here.

## 0.D — Saturation (visits per covered edge)
| dataset | budget | mean | median | p10 | %edges <5 visits |
|---|---|---|---|---|---|
| bitcoin-alpha | 5M | 7302 | 6856 | 4217 | 0% (absurd over-saturation) |
| bitcoin-otc | 500k | 390 | 340 | 190 | 0% |
| epinions | 500k | 9.9 | 4 | 1 | **50.3%** |
| wiki-elec | 500k | 18.7 | 7 | 1 | 37.1% |
| wiki-rfa | 500k | 31.3 | 9 | 1 | 33.4% |
| slashdot | 5M | 57.2 | 14 | 3 | 19.2% |

Degree-stratified TEST coverage (counter-intuitive, replicates prior session at
production budget): **low-degree edges are BETTER covered than hub-incident edges**
(wiki-elec deg1=0.995 vs deg32+=0.824; wiki-rfa 0.997 vs 0.840; epinions similar).
Uniform sampling under-covers high-degree-node edges (a hub visit picks only 1 of
its many edges). slashdot at 5M is ~flat (~0.98 all bins).

**Implication:** budgets are wildly mis-sized — bitcoin graphs are over-saturated
hundreds–thousands× while the 500k datasets have half their edges at <5 visits.

## 0.E — Is the coverage gap benign? (GNN AUC on covered vs uncovered test edges)
| dataset | model | cov AUC | unc AUC | Δ(cov−unc) | cov negR | unc negR |
|---|---|---|---|---|---|---|
| wiki-elec | GINEConv | 0.874 | 0.882 | −0.008 | 0.217 | 0.184 |
| wiki-elec | SiGAT | 0.889 | 0.917 | −0.027 | | |
| wiki-rfa | GINEConv | 0.872 | 0.855 | +0.017 | 0.214 | 0.225 |
| wiki-rfa | SiGAT | 0.881 | 0.897 | −0.016 | | |
| slashdot | GINEConv | 0.787 | 0.756 | +0.031 | 0.223 | **0.387** |
| slashdot | SiGAT | 0.857 | 0.907 | **−0.050** | | |
| epinions | GINEConv | 0.864 | 0.821 | **+0.043** | 0.156 | 0.086 |
| epinions | SiGAT | 0.911 | 0.922 | −0.012 | | |

**Not fully benign.** Uncovered edges differ from covered ones by up to
~0.03–0.05 AUC (model-dependent: uncovered are HARDER for GINEConv, EASIER for
SiGAT) AND have systematically different sign composition (slashdot uncovered
38.7% neg vs 22.3% covered). On the wiki near-ties (SiGAT beats walk by
0.0002/0.0021 on its full test) SiGAT is notably stronger on the uncovered edges
the walk never scores — so part of SiGAT's edge is an artifact of scoring 100%.
This justifies the fairness fix (push walk coverage to ~100%).

## 0.F — Does walk variance / saturation contribute to AUC? (YES, modestly, fast plateau)
AUC vs #occurrences-per-edge aggregated (mean-prob proxy for func_logit_power):

| dataset | k=1 | k=3 | k=5 | k=8 | all | gain (all−1) | plateau |
|---|---|---|---|---|---|---|---|
| epinions | 0.9220 | 0.9292 | 0.9300 | 0.9305 | 0.9305 | +0.0085 | ~k=3–5 |
| wiki-elec | 0.8893 | 0.8914 | 0.8915 | 0.8921 | 0.8928 | +0.0035 | ~k=8 |
| slashdot | 0.8878 | 0.8927 | 0.8939 | 0.8942 | 0.8948 | +0.0070 | ~k=8 |
| bitcoin-alpha | 0.9013 | 0.9103 | — | 0.9078 | 0.9088 | +0.0075 | noisy/saturated |

Proxy AUCs land within ~0.0006 of SOTA → trustworthy. **Saturation helps
+0.003–0.009 AUC and is fully captured by ~5–8 visits/edge; beyond that gives
nothing.** So: (a) the premise is real — go for saturation, not bare coverage;
(b) the target floor is k≈5–8, NOT higher; (c) the 500k datasets (half their edges
<5 visits) are leaving a little AUC on the table.

## Net implications for Phase 1/2
1. **Fix is justified**: coverage gap is unfair (0.E) and the 500k datasets are
   both under-covered (0.B) and under-saturated (0.D/0.F).
2. **Target**: ~100% node+edge coverage with a per-edge saturation floor **k≈5–8**.
3. **LCC irrelevant** (0.C): wiki gaps are under-sampling between fully-visited
   nodes, not reachability. Confirms dropping LCC.
4. **Per-dataset budget tuning is essential** (0.D): bitcoin graphs are
   over-saturated by 100–1000×; the 500k datasets need more / smarter sampling.
   An edge-anchored k_cover (k≈5) directly fixes the hub-edge under-coverage (0.D)
   and the saturation floor at once.
5. **Known sampler bug to fix in Phase 2**: `neg_emphasis_walks` treats negatives
   as label 0, but negatives are −1 (tokens E_-1/E_1) — that strategy is currently
   a no-op/uniform fallback. Not our chosen strategy, but note it.
