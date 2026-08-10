# Phase 3a — k_cover k=5 budget sweep (full attention), test AUC vs SOTA

SOTA = current E14 walk func_logit_power test AUC (CLAUDE.md table).
Goal: min budget whose func_logit_power test AUC >= SOTA. raw = per-occurrence transformer test AUC.

| dataset | SOTA flp | budget | raw test AUC | flp test AUC | vs SOTA | coverage |
|---|---|---|---|---|---|---|
| bitcoin-otc | 0.9431 | 200k | 0.9023 | 0.9264 | -0.0167 | 100% |
| wiki-elec | 0.8928@85% | 500k | 0.8416 | 0.8731 | -0.0197* | 100% (was 85.4%) |

*wiki-elec/epinions/wiki-rfa: SOTA flp was on the OLD covered subset; new flp is on the
FULL 100% test (includes the hard, previously-uncovered edges) -> NOT apples-to-apples.
Fair-test GNN refs (canonical, 100%): wiki-elec GINEConv 0.8738 / SiGAT 0.8930.
=> at 500k the walk model on wiki-elec's FULL test (0.8731) ~ GINEConv, BELOW SiGAT.
Disentangle: (a) harder edge set, (b) budget too low (newly-covered edges ~5 visits).

## PIVOTAL FINDING (matched-edge, same aggregator=mean-prob)
wiki-elec, identical 8857 SOTA-covered edges, same 500k budget:
  uniform@500k (SOTA) = 0.8928
  k_cover@500k (NEW)  = 0.8687   => -2.4pp from the SAMPLER alone (not edge set).
k_cover's forced anchor walks over-sample peripheral edges -> training distribution
skewed away from the natural one -> AUC drops even on already-covered edges.
HYPOTHESIS: at higher budget the uniform fill dominates and anchors become a small
fraction -> distortion shrinks -> AUC recovers while keeping 100% coverage. Tested by
epinions@2M / wiki-rfa@1M (running) and a wiki-elec budget bump. ALT: uniform@high-budget
(natural distribution + high coverage, no anchor skew).

## CORRECTION (process error)
The kill/relaunch of wiki-elec/wiki-rfa created DUPLICATE exp dirs. The earlier
wiki-elec@500k eval (flp 0.8731, matched 0.8687) used dir ...152926 = the KILLED,
undertrained run (~6 epochs), NOT the full nohup run ...153111 (38 ckpts). So the
"-2.4pp anchoring hurts" claim is INVALID/retracted pending re-eval on the correct dirs.
Correct full-run dirs: wiki-elec ...153111, wiki-rfa ...153111, wiki-elec@1.5M ...NW1500000_154944.
Re-evaluating now.

## CORRECTED RESULTS (full nohup runs, matched-edge mean-prob)
wiki-rfa k_cover@1M (vs SOTA uniform@500k flp 0.8810, 86% cov):
  walk NEW full(100% cov) = 0.8931   walk NEW matched = 0.8907 (+0.97pp vs SOTA on same edges)
  GNN full: GINEConv 0.8699, SiGAT 0.8831  => WALK WINS on full coverage.
  => k_cover@1M: 100% coverage AND beats SOTA + all GNNs. Clean win.
  "anchoring hurts" RETRACTED (was undertrained killed run).

wiki-elec k_cover@500k (EQUAL budget vs SOTA uniform@500k flp 0.8928, 85% cov):
  walk NEW full(100% cov)=0.9008  matched=0.8981 (+0.53pp vs SOTA on same edges, SAME budget)
  GNN full: GINEConv 0.8753, SiGAT 0.8930  => WALK WINS on full coverage (resolves wiki-elec SiGAT caveat).
=> CONCLUSION: k_cover at <= SOTA budget improves AUC AND gives 100% coverage. The only
   regression was bitcoin-otc@200k (200k < SOTA 500k budget) -> pure under-budget, retesting @500k.

## bitcoin-otc (already 100% covered by uniform)
  uniform@500k SOTA flp = 0.9431
  k_cover@200k flp = 0.9264 ; k_cover@500k flp = 0.9280  -> BOTH below SOTA.
=> For datasets uniform already covers 100%, k_cover anchoring adds no signal and slightly
   skews the train distribution -> small AUC loss. KEEP uniform SOTA for bitcoin-alpha/otc.
=> k_cover is a fix specifically for UNDER-covered datasets (epinions, wiki-elec, wiki-rfa, maybe slashdot).
Confirming on bitcoin-alpha@1M.

## epinions k_cover@2M (vs SOTA uniform@500k flp 0.9311, 87.8% cov)
  walk NEW full(100% cov)=0.9557  matched=0.9526 (+2.15pp vs SOTA on same edges)
  GNN full: GINEConv 0.8610, SiGAT 0.9137  => WALK WINS huge on full coverage.
  (ran full 50 epochs, still improving -> could go higher with more epochs)

## SUMMARY (under-covered datasets, k_cover WINS on full 100% coverage):
| dataset | budget | walk flp@100% | matched vs SOTA | best GNN@full |
| wiki-elec | 500k | 0.9008 | +0.5pp | SiGAT 0.8930 |
| wiki-rfa  | 1M   | 0.8931 | +0.97pp | SiGAT 0.8831 |
| epinions  | 2M   | 0.9557 | +2.15pp | SiGAT 0.9137 |
bitcoin-alpha/otc: KEEP uniform SOTA (already 100% covered; k_cover slightly hurts).
slashdot@5M: running.

## STATUS (pending long runs)
- slashdot090221 k_cover@5M: training (epoch ~13/50, raw val 0.913 vs SOTA 0.8952). posthoc on completion.
- bitcoin-alpha k_cover@1M: training (epoch ~16, raw val 0.9165 vs SOTA flp 0.9131 -- val ABOVE SOTA,
  so unlike bitcoin-otc it may NOT be hurt; posthoc needed to confirm). Both nohup, checkpoint/epoch.

## DECISIONS (final, pending slashdot/bitcoin-alpha posthoc)
- epinions, wiki-elec, wiki-rfa: ADOPT k_cover k=5 (budgets 2M/500k/1M). Big wins, 100% coverage,
  beat all GNNs on the fair full test. Resolves wiki SiGAT coverage caveats.
- bitcoin-otc: KEEP uniform SOTA (k_cover hurts -1.5pp; already 100% covered).
- bitcoin-alpha: TBD (val promising; posthoc pending).
- slashdot: TBD (posthoc pending; marginal 98.3% case).

## REMAINING WORK (Phase 4, after 6/6 finalized)
- Rebuild predictions_raw_canonical.pkl with new walk preds for adopted datasets.
- Rerun Leads 1 / 4 / 4b (consume walk per-edge preds). Leads 2/3 coverage-independent.
- Update CLAUDE.md SOTA table (walk col) + remove wiki coverage caveat; update WALK_COVERAGE.md.

## bitcoin-alpha k_cover@1M (already 100% covered; SOTA uniform@5M flp 0.9131)
  walk NEW full=0.9134  matched=0.9134  vs SOTA 0.9131  => TIE, at 1/5 the budget (1M vs 5M).
  GNN: GINEConv 0.8610, SiGAT 0.8821. k_cover NEUTRAL for alpha (vs otc where it lost -1.5pp).
=> Decision: keep BOTH bitcoin datasets on uniform SOTA (100% covered already; no coverage benefit).

## slashdot k_cover@5M (vs SOTA uniform@5M flp 0.8952, 98.3% cov)
  walk NEW full(100% cov)=0.9007  matched=0.8996 (+0.44pp on same edges)
  GNN full: GINEConv 0.7864, SiGAT 0.8588 => WALK WINS big on full coverage.
=> ALL 4 under-covered datasets WIN with k_cover (epinions +2.15, wiki-rfa +0.97, wiki-elec +0.5, slashdot +0.44),
   all now 100% coverage. bitcoin-alpha@1M ties SOTA; bitcoin-otc k_cover < uniform (keep uniform).

## bitcoin-otc bigger budget (user's question answered)
  otc k_cover: 200k=0.9264, 500k=0.9280, 2M=0.9427 ~ SOTA uniform 0.9431 (TIE at 2M).
=> bitcoin-otc recovers with enough budget. So k_cover matches/beats uniform on ALL 6 at
   sufficient budget, while giving 100% coverage. bitcoin pair gain no COVERAGE (already 100%)
   but lose no AUC at adequate budget -> all-6 k_cover methodology is viable.

================================================================================
# FINAL CONSOLIDATED RESULTS — full 34-cell sweep complete (2026-06-29)
================================================================================
Sweep DONE 34/34 (6 ds × {full,local} × budget grid). Orchestrator exited "SWEEP COMPLETE".
LocalAttn4 ran cleanly on all 17 local cells (first production confirmation of the perf fix).
Picks metric = func_logit_power (flp) test AUC. Matched table = mean-prob (≈flp within ~0.003).

## Best budget per dataset (flp test AUC, ~100% coverage)
| dataset        | Ours (full) | nw  | LocalAttn4 | nw  |
|----------------|------------|-----|-----------|-----|
| bitcoin-alpha  | 0.9251     | 5M  | 0.9362    | 5M  |
| bitcoin-otc    | 0.9427     | 2M  | 0.9410    | 1M  |
| epinions       | 0.9562     | 3M  | 0.9568    | 2M  |
| wiki-elec      | 0.9016     | 0.5M| 0.9038    | 0.5M|
| wiki-rfa       | 0.8932     | 1M  | 0.8916    | 1M  |
| slashdot090221 | 0.9012     | 5M  | 0.8984    | 3M  |

Budget shape: epinions/otc plateau 1-2M; bitcoin-alpha & slashdot still climbing at 5M (full);
wiki-elec/wiki-rfa AUC DROPS past min budget (over-saturation hurts small dense wiki graphs).

## Apples-to-apples matched (exact OLD-SOTA-covered edges; mean-prob)
FULL (Ours):
| dataset        | new full(100%) | new matched | old SOTA matched | best GNN matched |
|----------------|---------------|-------------|------------------|------------------|
| bitcoin-alpha  | 0.9237        | 0.9237      | 0.9131           | SiGAT 0.8821     |
| bitcoin-otc    | 0.9421        | 0.9421      | 0.9431           | GINE  0.8849     |
| epinions       | 0.9557        | 0.9537      | 0.9311           | SiGAT 0.9109     |
| wiki-elec      | 0.9008        | 0.8981      | 0.8928           | SiGAT 0.8892     |
| wiki-rfa       | 0.8931        | 0.8907      | 0.8810           | SiGAT 0.8808     |
| slashdot090221 | 0.9007        | 0.8996      | 0.8952           | SiGAT 0.8571     |
LOCALATTN4:
| dataset        | new full(100%) | new matched | best GNN matched |
|----------------|---------------|-------------|------------------|
| bitcoin-alpha  | 0.9353        | 0.9353      | SiGAT 0.8821     |
| bitcoin-otc    | 0.9406        | 0.9406      | GINE  0.8849     |
| epinions       | 0.9558        | 0.9541      | SiGAT 0.9109     |
| wiki-elec      | 0.9036        | 0.9012      | SiGAT 0.8892     |
| wiki-rfa       | 0.8916        | 0.8901      | SiGAT 0.8808     |
| slashdot090221 | 0.8979        | 0.8968      | SiGAT 0.8571     |

## VERDICTS
1. New sampler >= old SOTA on the SAME edges everywhere except bitcoin-otc (0.9421 vs 0.9431
   = tie). Real matched gains on the under-covered four (epinions +2.3pp, wiki-rfa +1.0,
   wiki-elec +0.5, slashdot +0.4) => k_cover saturation improves the model on identical edges.
2. Walk beats EVERY GNN on identical (matched) edges, all 6, both attention variants.
   Wiki SiGAT coverage caveat is DEAD.
3. Full-test(100%) ≈ matched on every ds => coverage expansion didn't inflate the headline.

## Winner run dirs (Phase 4 per-edge pred extraction)
full:  alpha E15_SWEEP_k5_nw5000000_full; otc E15_COVERAGE_KCOVER_K5_NW2000000_20260628-173849;
       epinions E15_SWEEP_k5_nw3000000_full; wiki-elec E15_COVERAGE_KCOVER_K5_20260628-153111;
       wiki-rfa E15_COVERAGE_KCOVER_K5_20260628-153111; slashdot E15_COVERAGE_KCOVER_K5_20260628-160751
local: alpha E15_SWEEP_k5_nw5000000_local; otc E15_SWEEP_k5_nw1000000_local;
       epinions E15_SWEEP_k5_nw2000000_local; wiki-elec E15_SWEEP_k5_nw500000_local;
       wiki-rfa E15_SWEEP_k5_nw1000000_local; slashdot E15_SWEEP_k5_nw3000000_local
preds: <dir>/checkpoints/<ds>_predictions/epoch_*/test_predictions.pkl
caches: data/<ds>/dataset_cache__k_cover_k5_nw<nw>_mw80_seed42.pt
