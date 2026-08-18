# Statistical Tests Audit — PEWTER (WSDM)

**Purpose.** This is not part of the paper. It is a working reference so you don't have to
remember what every test is called or does. For every empirical "A is better/worse/different
from B" claim anywhere in `WSDM_format_revised.tex`, it records: where the claim lives, what
test (if any) currently backs it, a plain-language explanation of that test, whether multiple-
comparison correction applies, and a recommendation for whether/how it belongs in the paper
given WSDM's page limit.

**How to read the recommendations.** Three buckets, used consistently below:
- **Keep as-is** — the test is real, correctly used, and already stated at the right level of
  detail for the space it takes.
- **Fix/add cheaply** — a real gap, but closeable with light computation from data already on
  disk (no new training runs).
- **State it plainly, no test** — the underlying quantity is a description (a percentage, a
  point estimate, an ablation table) that doesn't need — and would be over-dressed by — a
  formal significance test; recommend saying so as a description, not implying inferential
  weight it doesn't have.

Regenerate this doc's "not yet run" items by hand as they get computed; it's a snapshot, not a
generated artifact like the `figure_data/*.csv` files.

---

## Summary table

| # | Claim (short) | Location | Current test | Status | Recommendation |
|---|---|---|---|---|---|
| 1 | Pewter beats the best baseline, all 6 datasets, +3.3pp mean | Abstract, 6.2, 6.3, Conclusion | mean±std over 10 splits only | **done 2026-08-18** — paired Wilcoxon, 10/10 splits win on all 6 datasets, p=0.00098 throughout | Ready to state in 6.2's rewrite as a compact parenthetical; not yet inserted (6.2 prose doesn't exist yet) |
| 2 | Edge-vs-vertex / "in some graphs most of the prediction is from edges, in others from vertices" | Abstract, Contributions #5, 6.1(D) | none — investigation not started | **needs investigation + a test**, see the WSDM closeout plan | Do not state as a headline empirical claim until real data backs it (see Section D of the plan file) |
| 3 | MI decay 178×–73,000× between distance 1 and 2–3 | Contributions #2, 6.1(A) | none (point ratios, real vs. null-shuffle control exists in the underlying script but isn't quoted) | adequate as a descriptive statistic | **State it plainly, no test** — it's a magnitude claim, not a comparison needing p-values |
| 4 | Baseline AUC falls as endpoint entropy rises (bottleneck confirmation) | 6.1(B), Panel C | per-dataset 2-way ANOVA (src×tgt entropy bin, 10 seeds as replicates) | **done 2026-08-18** — both main effects sig. on all 6, interaction sig. on 5/6 (bitcoin-alpha underpowered) | Ready to cite in 6.1(B)'s rewrite; bitcoin-alpha's interaction result needs the sparsity caveat if quoted individually |
| 5 | Entropy asymmetry: source lower than target on 4/6, reversed on 2/6 | 6.1(C) | one-sided Wilcoxon signed-rank per dataset + DerSimonian-Laird pooled random-effects | correct test, no FDR correction across the 6 per-dataset tests | **Keep as-is**, note FDR non-issue given p-value magnitudes (see below) |
| 6 | Panel D (Fig. 2): sign-agreement bucketed AUC | 6.1(D)-adjacent / Figure 2 panel (D) caption | per-dataset dots, mean±SD over 10 seeds; paired Wilcoxon (same vs. diff) | **done 2026-08-18** — dots-per-dataset wired into the combined figure; paired test now done too: same>diff on all 6 datasets, both directions, p=0.001 throughout | Ready to state in the caption as a compact parenthetical — proposed wording pending your confirmation |
| 7 | Panel E (Fig. 2): entropy-term coefficients, which term dominates error, per dataset | Figure 2 panel (E) caption | per-dataset: two-way cluster-robust SE + BH-FDR per seed, "robust" = significant in ≥8/10 seeds | **done 2026-08-18** — rebuilt as a per-dataset stacked bar (10-seed refit), citation gap still open | Keep new per-dataset test as-is; citation gap (`cameron2011robust`, `benjamini1995controlling`) still needs your sign-off before re-adding |
| 8 | Ablation A: local attention matches/beats full on 4/6 datasets | 6.5(A) | "within reported standard errors" — informal eyeball | **needs a test added**, AND the "4/6" number itself is stale (real recompute shows 3/6, see plan Phase 4) | Fix/add cheaply — paired bootstrap across the 10 splits; fix the stale count in the same pass |
| 9 | Ablation B: aggregator choice barely matters (spread < 0.0018 AUC) | 6.5(B), Table 2 | mean±std over 10 splits, spread vs. std comparison | **done 2026-08-18** — table rebuilt multiseed, spread tightened to 0.0018, an order of magnitude below the std | Formal paired-bootstrap on the 10-seed data is a nice-to-have, not needed — the spread-vs-std comparison already makes the point |
| 10 | K-ablation: is the gain "just ensembling"? | 6.5(C) | mean±std AUC per K across 10 seeds | **done 2026-08-18** | K=1 already beats the best baseline on all 6 datasets; gap to saturated ceiling is <1pp everywhere — representation, not ensembling, carries the result. Script: `scripts/paper_figures/extract_ablation_kwalks.py` |
| 11 | Attention split: forward- vs. backward-dominant per dataset | 6.4, Panel B/C of Fig. 4 | one-way cluster-robust SE (cluster = target edge) + Wilcoxon confirmation | **done 2026-08-18** — significant on all 6 datasets, both the fwd/bwd split (Panel B) and the node/edge split (Panel C) | Error bars added to both plots; 6.4 prose updated. Script: `scripts/attention_directionality_panelB_se.py`, cached per-instance data at `outputs/attention_directionality/<ds>_local_panelBC_perinstance.pkl` |
| 12 | Shapley causal contribution: hop-1 forward-dominant on 3/6 datasets | 6.4, Panel D of Fig. 3 | one-way cluster-robust SE (cluster = target edge), CI-exclude-zero test | **done 2026-08-18** — test now named in-prose | Kept as-is |
| 13 | Attention-split vs. entropy-asymmetry correlation (6 points) | 6.4 | Spearman ρ=−0.71, p=0.11, n=6 | **done 2026-08-18** — not significant, stated as such | Kept as descriptive; TODO removed from tex |
| 14 | Vertex-vs-edge attention share vs. AUC boost over best GNN (6 points) | not yet in tex, feeds the edge-vs-vertex investigation (item #2) | Spearman ρ=−0.32, p=0.54, n=6 | **done 2026-08-18** — null result | Does not support "edge-leaning attention → bigger GNN advantage" as an alternative theory; do not cite this correlation as evidence for item #2's claim |
| 15 | Figure 3 (delta heatmap): "gain concentrates where the bound bites" | 6.3 | cluster-robust regression, delta ~ src/tgt entropy, per dataset | **done 2026-08-18 — result is MIXED**, only 1-2/6 datasets show the claimed pattern, 1 reverses | **Decided 2026-08-18**: compact in-text sentence naming only the significant cases (Bitcoin-otc both axes, Epinions source axis), now in the tex; other 4 datasets not itemized in-text |
| 16 | Datasets are heavily imbalanced (77–94% positive) | Datasets paragraph, Ethics | none needed | N/A | **State it plainly, no test** — a prevalence statistic, not a comparison |

---

## 1. Pewter beats the best baseline on all six datasets

**Where:** Abstract ("mean +3.3 AUC points... range +1.4 to +4.5"), Section 6.2 ("Pewter has a
higher test AUC than the best baseline on all six datasets..."), Conclusion ("improves AUC over
eight baselines").

**Current test: none.** The claim is backed by two point estimates per dataset (Pewter's
mean±std over 10 splits, vs. the best baseline's own mean±std over its own 10 splits, per
Table 1) with no formal comparison between them. Eyeballing whether the std bars are far apart
is not a significance test — it doesn't account for the fact that both models are evaluated on
the *same* 10 test splits (a paired setting, which has much more power than treating the two
sets of 10 numbers as independent samples).

**What a real test looks like here.** Two candidates, in order of preference:
- **Paired Wilcoxon signed-rank test across the 10 splits**: for a given dataset, take
  (Pewter AUC − best-baseline AUC) on each of the 10 shared splits, test whether the median
  difference is positive. Non-parametric, robust to the small sample size (n=10), and matches
  the paired-test convention already used elsewhere in this paper (6.1(C)'s entropy-asymmetry
  test).
- **Paired bootstrap over test edges** (finer-grained, higher power): resample the shared test
  edges with replacement, recompute both models' AUC on the same resample, repeat ~3000 times,
  report the 95% CI of the AUC difference. This is the same recipe already implemented in
  `scripts/paper_figures/ablationB_paired_significance.py` for a different comparison (Ablation
  B's aggregators) — the code pattern ports directly, just swapping which two score arrays are
  compared.

**Feasibility, checked directly (this is the part that actually needs deciding):** per-seed
data exists in different shapes for different baselines —
- **SiGAT** (`baselines/SGA/results_our_splits_canonical/<ds>/SiGAT/seed{42..51}/score.csv`,
  66 files, real 10-seed campaign) — usable directly.
- **node2vec** (`baselines/node2vec/results_canonical/<ds>/seed{42..51}/`, 60 files) — usable.
- **SNEA / CopulaLSP** — 10-seed campaigns exist per `CLAUDE.md`'s "Current SOTA" note
  (`scripts/run_multiseed_snea_copulalsp.py`) — usable.
- **GSGNN** (via CSG) — 30 `score.csv` files found under `baselines/CSG/`; **not yet confirmed
  whether this is a genuine 10-seed-per-dataset campaign or a partial one** — check before
  relying on it (CLAUDE.md's "Current SOTA" section separately notes CSG/CSG-GSGNN were pulled
  from Table 1 pending a 10-split rerun in an earlier snapshot of this project; the Table 1 row
  currently in the tex is labeled plain "GSGNN," not "CSG-GSGNN," so confirm which pipeline
  produced today's numbers before trusting the count).
- **SGCN** — need to check; not verified in this pass.
- **GCN, GAT** — **no local per-seed data exists at all.** Per `CLAUDE.md`: "using numbers the
  user supplied from external publications (not reproduced locally for GCN/GAT — no local run
  exists at all)." A paired test against GCN/GAT is **not possible** with what's on disk — only
  a single externally-reported number per dataset, no per-seed variance, no shared splits.

**Done 2026-08-18.** Turned out simpler than the open feasibility question suggested: the best
non-Pewter baseline per dataset is always either SiGAT or SNEA (SNEA on bitcoin-alpha/
bitcoin-otc, SiGAT on the other 4) — both already confirmed with real 10-seed data, so GS-GNN's
uncertain per-seed status never had to be resolved. Per dataset, picked whichever Pewter variant
(full/local) has the higher 10-seed mean (this matched Table 1's existing bold marks exactly, a
good consistency check) and ran a one-sided paired Wilcoxon signed-rank test against the best
baseline across the 10 shared seeds:

| dataset | Pewter variant | Pewter mean | baseline | baseline mean | wins | $p$ |
|---|---|---|---|---|---|---|
| bitcoin-alpha | full | 0.9146 | SNEA | 0.8705 | 10/10 | 0.00098 |
| bitcoin-otc | local | 0.9318 | SNEA | 0.8899 | 10/10 | 0.00098 |
| epinions | local | 0.9536 | SiGAT | 0.9088 | 10/10 | 0.00098 |
| wiki-elec | local | 0.9023 | SiGAT | 0.8884 | 10/10 | 0.00098 |
| wiki-rfa | full | 0.8923 | SiGAT | 0.8780 | 10/10 | 0.00098 |
| slashdot090221 | full | 0.8989 | SiGAT | 0.8586 | 10/10 | 0.00098 |

Pewter's winning variant beats the best baseline on all 10/10 individual splits, on all 6
datasets (60/60 total) — $p=0.00098$ is the minimum achievable one-sided Wilcoxon $p$-value at
$n=10$ (a perfect sweep), so this is as strong a result as this test can report; no FDR
correction concern since every one of the 6 tests independently clears $p<0.001$, nowhere near
a borderline call. Script: `scripts/paper_figures/table1_paired_significance.py`, data:
`aaai2027/figure_data/table1_paired_significance.csv`. **Do not claim the paired test covers
GCN/GAT** — those stay point-estimate comparisons only (no local per-seed data exists), and the
prose should say so if it names them specifically rather than implying every row in Table 1 got
the same statistical treatment.

**Recommendation:** state the result as a compact parenthetical wherever Table 1 is described,
e.g. "(paired Wilcoxon signed-rank test across the 10 splits, Pewter's better-of-full/local
variant beats the strongest baseline on all 10/10 splits on all six datasets, $p<0.001$
throughout)". This is exactly the level of detail the professor's 6.2 prose ask wants — hold
for that rewrite rather than inserting ad hoc, since 6.2 doesn't exist yet as real prose.

---

## 2. Edge-vs-vertex / directionality "alternative theories" claim

**Where:** Abstract's new ending, Contributions bullet 5 (currently a placeholder), Section
6.1's stub paragraph (D).

**Current test: none — the underlying data source for this claim hasn't even been identified
yet.** This is flagged in detail in the plan file's Section D ("Edge-vs-vertex / directionality
claims changed") — Panel (c)'s existing vertex-vs-edge *attention mass* split may or may not be
the right evidence for a claim about where the *prediction* comes from (a stronger claim than
attention mass, closer to an ablation — masking vertex vs. edge tokens and measuring the AUC
drop). Until that's resolved, there is no test to audit here.

**Recommendation:** do not let this claim ship with an implied statistical backing it doesn't
have. Once the investigation lands, whatever test is designed (candidates: a masking ablation's
paired AUC-drop test, or a correlation as in items 13/14 below) needs its own entry here before
the paper states it as fact.

---

## 3. MI decay with distance (178×–73,000×)

**Where:** Contributions bullet 2, Section 6.1(A).

**Current test:** none — and none is needed. This is a magnitude/ratio claim (NMI at distance 1
divided by NMI at distance 2–3), not a claim that two things differ significantly — a "how much"
statement, not a "whether" statement. The underlying extraction script (per prior work in this
project, see `CLAUDE.md`'s Panel B "bump" investigation) does have a `--shuffle-signs` null
control available for a *different*, adjacent claim (whether the bump at distance 4–6 is real),
but that's not what's being stated in 6.1(A) — 6.1(A) just reports the ratio.

**Recommendation:** state it plainly, no test needed. If you want to preempt a reviewer asking
"is this ratio itself noisy," the null-control machinery already exists and could produce a CI
on the ratio cheaply — but that's a nice-to-have, not a gap, given WSDM's page budget.

---

## 4. Baseline AUC falls as endpoint entropy rises (Panel C / paragraph B)

**Where:** Section 6.1(B), Figure 2 (`fig:empconf-panels`) panel (c).

**Current test: none.** The claim ("AUC is highest where... unanimous and falls toward chance
as entropy rises") is read directly off the heatmap's monotonic-looking pattern across bins — no
formal test that the pattern is more than bin-to-bin noise.

**The professor's request (already in the plan): a per-dataset two-way ANOVA**, factors =
source-entropy bin × target-entropy bin (the same 4×4 binning Panel C already uses), response =
AUC, using the 10 cross-validation splits as repeated measurements per cell instead of a single
point estimate.

**Done 2026-08-18.** Turned out even cheaper than the feasibility note expected: reused the raw
per-seed SiGAT predictions already cached for Panel D's rebuild
(`outputs/cache/sigat_raw_predictions/`), joined with entropy (`collect_model_records`, no
fitting) and binned into Panel C's own 4×4 grid per (seed, cell) — one AUC observation per
(seed, cell), up to 160 per dataset (10 seeds × 16 cells; fewer where a cell falls below the
`n≥30` threshold in some seeds). Two-way ANOVA (Type II SS):
`auc ~ C(src_bin) + C(tgt_bin) + C(src_bin):C(tgt_bin)`.

| dataset | cells complete (of 16) | src main effect | tgt main effect | interaction |
|---|---|---|---|---|
| bitcoin-alpha | 6/16 | F=11.5, p=0.0010 | F=16.6, p=9.7e-5 | F=0.45, p=0.87 (n.s.) |
| bitcoin-otc | 15/16 | F=75.2, p=3.0e-29 | F=65.1, p=1.4e-26 | F=6.9, p=3.3e-8 |
| epinions | 16/16 | F=232.5, p=5.5e-55 | F=380.7, p=3.1e-68 | F=11.2, p=3.4e-13 |
| wiki-elec | 16/16 | F=39.4, p=1.2e-18 | F=420.2, p=5.4e-71 | F=13.1, p=3.8e-15 |
| wiki-rfa | 16/16 | F=99.2, p=7.1e-35 | F=685.2, p=5.3e-85 | F=29.6, p=1.2e-28 |
| slashdot090221 | 16/16 | F=1852.2, p=9.0e-115 | F=303.4, p=5.1e-62 | F=28.5, p=7.0e-28 |

Both main effects are significant on all 6 datasets, confirming Proposition 1's prediction that
AUC depends on both source and target entropy, not just one. The interaction term is
significant on 5/6 datasets (all but bitcoin-alpha) — the two entropy axes don't act purely
additively; **caveat: bitcoin-alpha's own result is the least reliable of the six** (only 6/16
cells have data in all 10 seeds — its ~2,300-edge test set is too small to fill the sparser
corner bins every seed, matching Panel C's own single-split heatmap already showing two `n/a`
cells for this dataset — and the fit returns a rank-deficiency warning as a result); treat
bitcoin-alpha's non-significant interaction as underpowered, not necessarily a genuine null, if
this table is cited claim-by-claim rather than as an aggregate pattern. Script:
`scripts/paper_figures/panelC_twoway_anova.py`, data:
`aaai2027/figure_data/panelC_twoway_anova.csv`.

**Caveat to flag before running it:** ANOVA's response variable here would be per-cell AUC
(one number per seed per cell), not a raw per-edge outcome — this is testing "does the *cell
AUC* vary systematically by bin," which is the right question, but note some cells may have
`MIN_CELL_N`-masked (too few edges) values in some seeds; those need to drop out of the ANOVA
design rather than being treated as 0 or imputed.

**Recommendation:** fix/add cheaply — this is genuinely a same-day task given the existing
per-seed data. Report per-dataset ANOVA F-statistics/p-values for the two main effects and the
interaction; given WSDM's space, a compressed form ("both main effects significant at
$p<0.05$ on N/6 datasets; interaction significant on M/6") is the right level of detail, full
ANOVA tables belong in the appendix if anywhere.

---

## 5. Entropy asymmetry (paragraph C) — the one test already done right

**Where:** Section 6.1(C).

**Current test: one-sided Wilcoxon signed-rank test, per dataset, plus a DerSimonian-Laird
random-effects meta-analysis pooled across the four confirming datasets.** Verified against
`scripts/lead4c_directionality_answers.py` — this is real, already-computed machinery, not
just prose.

**Plain-language explanation, for future reference:**
- **Wilcoxon signed-rank test** is the non-parametric analogue of a paired $t$-test: it takes
  each vertex's $(\Hh_\outdeg(v) - \Hh_\indeg(v))$ difference and tests whether the *median*
  difference is zero, without assuming the differences are normally distributed. This matters
  here because entropy values are bounded in $[0,1]$ and heavily clustered at exactly 0 (any
  vertex with unanimous incident signs has zero entropy on that side) — a paired $t$-test's
  normality assumption would be shaky on a distribution with a spike at a boundary, while
  Wilcoxon only needs the differences to be symmetric around their median under the null, a much
  weaker requirement. "One-sided" (`alternative="less"` / `"greater"`) means the test checks a
  specific direction (source lower than target, or the reverse) rather than just "different,"
  which is the right choice here since the paper has a directional hypothesis to confirm or
  reject per dataset, not just "is there any difference."
- **DerSimonian-Laird random-effects meta-analysis** pools an effect size (here, the proportion
  of non-tied vertex pairs favoring lower source entropy) across multiple independent studies
  (here, the 4 confirming datasets) into one combined estimate with a confidence interval,
  *while allowing for genuine between-dataset heterogeneity* (as opposed to a fixed-effects
  pool, which assumes every dataset is measuring the exact same true effect and only differs by
  sampling noise — an assumption this paper's own data contradicts, since 2 of the 6 datasets
  reverse the direction entirely). This is the right tool given the explicit two-family
  structure (4 datasets confirming, 2 reversing) already established.

**Multiple-comparison correction: not applied, and worth a one-line justification rather than
a fix.** Six per-dataset Wilcoxon tests are run (one per dataset), with no Bonferroni/FDR
correction across them. In most settings, 6 simultaneous tests would warrant correction. Here
it's a non-issue in practice: the smallest p-values are $5.1\times10^{-4}$ to
$3.7\times10^{-244}$ (the 4 "confirming" datasets) and $1.7\times10^{-10}$/$3.4\times10^{-14}$
(the 2 "reversed" datasets) — even a very conservative Bonferroni correction (÷6) leaves every
one of these significant by dozens of orders of magnitude. **Recommendation: keep as-is**, but
consider adding one clause noting no correction was needed given the p-value magnitudes, so a
reviewer doesn't have to do that arithmetic themselves.

---

## 6. Panel D (Figure 2): sign-agreement bucketed AUC

**Where:** Figure 2 (`fig:empconf-panels`) panel (d) caption; body prose is currently the
duplicate-of-(C) stub in paragraph (D), not yet describing this panel specifically.

**Current test: none.** Panel (d) reports one AUC point per bucket (same/diff × in/out), pooled
across all 6 datasets, no error bar, no test that the buckets differ from each other.

**Pooled multiseed rebuild — done 2026-08-18.** Panel D now reports mean $\pm$ 1 SD across
10 splits per bucket (`aaai2027/figure_data/empconf_panelD_signagreement_auc.csv`), reusing
`sigat_raw_seed()` the same way Panel C's SiGAT cells were rebuilt — real error bars, not a
single-split point estimate. Numbers barely moved from the old single-split values (e.g.
in-same 0.940→0.939, out-same 0.983→0.981), a good sanity check that the old single-split
number wasn't an outlier. Script: `scripts/paper_figures/extract_empconf_panelD_signagreement_auc.py`
(now caches each (dataset, seed)'s raw SiGAT prediction to
`outputs/cache/sigat_raw_predictions/` — the LogisticRegression refit is the expensive part,
~60 fits; the cache means any future rebuild of this panel, or any other analysis needing
raw per-seed SiGAT predictions, doesn't redo it), `plot_empconf_panelD_signagreement_auc.py`.

**Per-dataset slice — data computed, visualization NOT yet decided, paused per the user's
explicit request** ("we will revisit this panel in future as i want to see how it looks...").
Per-(dataset, bucket) mean±SD across the 10 seeds is saved
(`aaai2027/figure_data/empconf_panelD_signagreement_auc_perdataset.csv`) and a draft
dot-plot rendering exists (`plot_perdataset()` in the same plot script, not yet run/wired
into any combined figure) — but the final visual form (dots, a 6-wide bucket slice, a
boxplot, or something else) is explicitly still open. **The statistical test itself is
already settled regardless of which visual is chosen**: mean $\pm$ 1 SD (equivalently, a
95% CI) across the 10 independently-trained-and-refit seeds per (dataset, bucket) — the same
"Option 2" convention (per-split point, then average) used everywhere else in this project's
multiseed work, not pooled raw predictions across seeds.

**Paired significance test — done 2026-08-18.** Per the professor's C-ter ask ("please have
statistical tests for D and e"). Claim being tested: AUC is higher when a target edge's
neighbor edges agree with its own sign (\emph{same}) than when they disagree (\emph{diff}).
One-sided paired Wilcoxon signed-rank test across the 10 seeds (AUC$_\text{same}$ $-$
AUC$_\text{diff}$ $>0$), per dataset, in and out directions separately — same test family
already used for Table 1 and the entropy-asymmetry claim in 6.1(C), not a new convention.
Reused the already-cached per-seed raw SiGAT predictions (`outputs/cache/
sigat_raw_predictions/`, no refit) and the extract script's own bucketing logic — pure
reduction of existing data. Result: **same $>$ diff on all 6 datasets, both directions
(12/12), $p=0.001$ throughout** (the minimum achievable one-sided Wilcoxon $p$ at $n{=}10$ —
a perfect sweep, same pattern as Table 1's paired test):

| dataset | in: same | in: diff | in: $\Delta$ | out: same | out: diff | out: $\Delta$ |
|---|---|---|---|---|---|---|
| bitcoin-alpha | 0.964 | 0.619 | +0.344 | 0.894 | 0.776 | +0.118 |
| bitcoin-otc | 0.948 | 0.638 | +0.310 | 0.949 | 0.827 | +0.122 |
| epinions | 0.944 | 0.747 | +0.197 | 0.992 | 0.817 | +0.175 |
| wiki-elec | 0.911 | 0.656 | +0.255 | 0.935 | 0.880 | +0.054 |
| wiki-rfa | 0.890 | 0.658 | +0.232 | 0.926 | 0.862 | +0.064 |
| slashdot090221 | 0.921 | 0.750 | +0.170 | 0.952 | 0.600 | +0.352 |

All $p=0.0009766$. Script: `scripts/paper_figures/panelD_paired_significance.py`, data:
`aaai2027/figure_data/panelD_paired_significance.csv`.

**Recommendation:** pooled version — keep as-is, done. Paired test — clean, unambiguous
result, ready to state in the caption as a compact parenthetical (proposed wording pending
user confirmation). Per-dataset visualization — do not decide placement or exact chart type
until the user reviews a candidate render (this is now moot for Figure 2 itself, since the
per-dataset dots view was already adopted into the combined figure on 2026-08-18 — kept here
as history of the decision process).

---

## 7. Panel E (Figure 2): entropy-term coefficients

**Where:** Figure 2 (`fig:empconf-panels`) panel (e) caption: "error bars are cluster-robust
standard errors, hatched bars are not significant after FDR correction ($q<0.05$)."

**Current test: two-way cluster-robust standard errors (Cameron–Gelbach–Miller 2011) + BH-FDR
correction across the coefficients within a model.** Verified directly in
`scripts/lead4c_entropy_logit_regression.py` (`cluster_robust_2way`, lines ~309–330) — this is
real, correctly-implemented machinery, not a shortcut.

**Plain-language explanation:**
- **Cluster-robust standard errors** widen a coefficient's naive standard error to account for
  correlated errors within groups — here, all edges sharing the same source vertex $u$, or the
  same target vertex $v$, are not independent observations (a single "hard" vertex contributes
  many correlated edge-level errors), so treating every edge as an independent draw understates
  uncertainty. **Two-way** clustering (Cameron–Gelbach–Miller) handles the fact that an edge
  belongs to *two* overlapping groups simultaneously (its source's cluster and its target's
  cluster) — the formula is $V = V_u + V_v - V_{u,v}$, adding the two one-way cluster variance
  estimates and subtracting their intersection to avoid double-counting.
- **Benjamini–Hochberg (BH) FDR correction** controls the *false discovery rate* — of all the
  coefficients declared "significant," what fraction are expected to be false positives — rather
  than the stricter (and more conservative) family-wise error rate a Bonferroni correction would
  target. FDR is the right choice when testing several related coefficients from the same
  regression (as here, 4–6 entropy terms per model) and some real effects are expected among
  them; it's less punishing than Bonferroni while still controlling for the fact that testing
  multiple coefficients inflates the chance of a spurious "significant" result.

**Real gap found: the citations for both methods are currently missing from the bib.** Per the
"MAJOR RESYNC" audit of the professor's bib rewrite, `cameron2011robust` and
`benjamini1995controlling` were both present earlier this session and are now gone from
`pewter_references.bib`, and the tex's Figure 2 caption states both methods with no citation at
all (the earlier-session convention of not putting `\citep` inside a caption is also a factor —
even if the bib entries existed, they'd need to be cited in the body prose that describes this
figure, not the caption itself). **Per the standing rule from this session, this needs your
sign-off before either bib entry is re-added** — flagging it here rather than adding it
unilaterally.

**Per-dataset stacked-barplot rebuild — done 2026-08-18.** Panel E now shows SiGAT's 4
entropy-term coefficients ($z$-scored) stacked per dataset (diverging: negative terms stack
downward), refit independently on each of the 10 splits (42–51) rather than pooled across
datasets — the per-dataset breakdown the pooled version was hiding (e.g. `tgt_in` dominates on
the 4 smaller/sparser datasets, `src_out` dominates on epinions/slashdot090221, matching the
per-dataset multiseed export's own README finding). **The new test, designed for this rebuild:
a term is marked "robust" for a given dataset if it is BH-FDR significant ($q<0.05$) in at
least 8 of the 10 independently-fitted seeds** (same convention already used by both multiseed
SiGAT export packages' own forest plots, `≥8/10` = "robust"), not just significant in a single
aggregate fit — non-robust segments are drawn hatched. Error bars are $\pm1$ SD of each term's
own coefficient across the 10 seeds. Source: `outputs/lead4c_sigat_multiseed_node4_export/
results/aggregated_summary.csv` (already-computed 10-seed per-dataset regression, no new
fitting needed). Scripts: `scripts/paper_figures/extract_empconf_panelE_coefficients.py`,
`plot_empconf_panelE_coefficients.py`.

**Recommendation:** the per-dataset test above is now real and adequate — **keep as-is**.
Citation gap for cluster-robust SE / BH-FDR (used by the underlying per-seed fits, not just the
old pooled version) is unchanged and still pending your sign-off on which citation, if any, to
re-add.

---

## 8. Ablation A: local attention matches/beats full on 4/6 (stale) datasets

**Where:** Section 6.5(A), paragraph "(A) Proximal attention is sufficient."

**Current test: none — "within the reported standard errors" is an informal eyeball**, and (a
separate, more urgent problem) **the underlying numbers are stale.** The tex currently reads
"matches or beats full attention on four of six datasets (bitcoin-otc +0.02pp, epinions
+0.07pp, wiki-elec +0.30pp, wiki-rfa +0.38pp)... trails on the other two (bitcoin-alpha
−0.27pp, slashdot090221 −0.21pp)." The closeout plan (Phase 4) already recomputed these deltas
directly from Table 1's real 10-seed numbers and found a **3-3 split, not 4-2**: bitcoin-otc
+0.02pp, epinions +0.13pp, wiki-elec +0.15pp win for local; bitcoin-alpha −0.12pp, wiki-rfa
−0.09pp (sign flips from the stale text!), slashdot −0.21pp trail. This numeric fix has not yet
been applied to the tex as of this audit — **flagging it here again since it's exactly the kind
of claim this audit exists to catch, but it's tracked as its own Phase 4 action item, not
newly discovered here.**

**What a real test looks like:** the full/local comparison is paired (same 10 splits, same
architecture, only the attention window differs), so a paired Wilcoxon or paired bootstrap
across the 10 splits per dataset is the right tool — same pattern as item #1 above, and the
per-seed data already exists (it's the same checkpoints backing Table 1).

**Recommendation:** fix/add cheaply, and do the numeric correction and the test addition in the
*same* edit (per the plan's own "Verification" section warning: fixing the numbers without
re-reading whether the surrounding sentence's claim still holds is exactly the failure mode to
avoid — "four of six" printed next to three winners would be a self-evidently broken sentence if
only the numbers were swapped and not the count).

---

## 9. Ablation B: aggregator choice barely matters

**Where:** Section 6.5(B), Table 2 (`tab:ablationB`).

**Current test: not stated in the tex.** A real paired-bootstrap significance script already
exists (`scripts/paper_figures/ablationB_paired_significance.py`) — it resamples test edges and
compares pairs of aggregator functions' AUC on the same resample, correctly accounting for the
fact that every aggregator is scored on the identical set of edges (so naive independent
standard errors would be far too narrow). **But it's single-split** (the `E27_...` checkpoints,
pre-dating the multiseed campaign) and isn't currently referenced anywhere in the tex or cited
as backing the "full spread under 0.0022 AUC" claim.

**Multiseed rebuild — done 2026-08-18.** Confirmed all 11 aggregator functions' posthoc
summaries already existed per seed (all 6 datasets, all 10 seeds, `func_logit_power` through
`func_maxprob_power`) — pure aggregation, no new training/posthoc runs needed. Table now reports
mean±std over the 10 splits per (function, dataset) cell instead of a single-split point.
**The real spread tightened**: max cross-function spread on any dataset is now $0.0018$ AUC (was
stated as $<0.0022$ from the single split) — about an order of magnitude below the ~0.005–0.018
split-to-split std shown alongside it, which is a stronger, more legible way to make the "barely
matters" point than the bare spread number alone (a reader can see directly that the
between-function differences are smaller than the noise). Script:
`scripts/paper_figures/extract_ablationB_multiseed.py`, data:
`aaai2027/figure_data/ablationB_multiseed.csv`.

**Not done, optional**: porting the existing single-split paired-bootstrap script
(`ablationB_paired_significance.py`) to the 10-seed data for a formal per-pair significance
test. Given the spread-vs-std comparison already makes the point clearly and directly, and this
project's stated lean toward the cheaper option absent a clear gain, treat this as a
nice-to-have, not a gap — revisit only if a reviewer specifically asks for a formal test rather
than the descriptive comparison.

---

## 10. K-ablation (planned, not yet in the tex)

**Where:** Section 6.5(B) tail, `\ph{TODO}` — not yet written.

**Current test: not applicable yet — the whole analysis hasn't been run.** Per the plan
(confirmed cheap, "minutes, CPU-only, reuses already-saved per-walk predictions"), this sweeps
$K$ (number of walk occurrences aggregated per edge) from 1 upward and reports AUC per $(dataset,
seed, K)$.

**What counts as "answering the question" here, without needing a fancy test:** the whole point
of this ablation is to see *where* AUC saturates as a function of $K$ — mean±std across the 10
seeds at each $K$ is itself the answer (does $K=1$ already compete with the best GNN baseline,
or does the gain only appear at large $K$?). A formal significance test between adjacent $K$
values isn't necessary for the paper's argument; the shape of the curve (and where its
mean±std band stops moving) is self-interpreting.

**Recommendation:** no test needed beyond mean±std per $(dataset, K)$ across the 10 seeds — this
is a "state it plainly" case once the sweep is run, not a "needs a test added" case.

---

## 11. Attention forward/backward split, per dataset (Section 6.4, Panel B)

**Where:** Section 6.4 body: "Two datasets are forward-dominant (bitcoin-alpha $0.443$ forward
vs. $0.411$ backward...)... four are backward-dominant..."; Figure 3 panel (b).

**Current test: none.** These are raw point percentages with no error bar and no stated test
that, say, bitcoin-alpha's $0.443$ vs. $0.411$ split is more than noise — contrast this with the
*same figure's* panel (d) (Shapley), which does report cluster-robust SEs and explicit
significance language ("significantly forward-dominant"). The asymmetry in how rigorously the
two panels of the same figure are described is itself worth fixing for consistency, not just
because item #11 is individually a gap.

**Feasibility:** cheap. The same one-way cluster-robust SE machinery already used for panel (d)
(cluster = target edge; `scripts/shap_edge_directionality.py`'s pattern) applies directly to
attention-mass values — attention mass at layer 0 is already computed per (edge, direction) the
same way Shapley values are, just a different quantity being averaged.

**Recommendation:** fix/add cheaply — add cluster-robust SEs to the attention-mass split the
same way panel (d) already has them, and state which datasets' splits are significant vs. which
aren't (probably not all — some of the listed splits look close, e.g. bitcoin-otc's $0.403$ vs.
$0.445$).

---

## 12. Shapley causal contribution: hop-1 direction split

**Where:** Section 6.4, second paragraph; Figure 3 panel (d) caption.

**Current test: one-way cluster-robust SE (cluster = target edge_id), used to support
"significantly forward-dominant" language.** Verified in `scripts/shap_edge_directionality.py`
lines ~213–224 — real, correctly-scoped machinery (collapse to per-edge means, SE across those
means, standard one-way cluster-robust recipe).

**Gap: the exact test isn't named in prose.** The tex says "significantly forward-dominant"
without stating whether that means a 95% CI on the forward-minus-backward difference excludes
zero, a $z$-test using the cluster-robust SE, or something else. These are typically equivalent
constructions (CI-exclusion and a $z$-test off the same SE give the same accept/reject
decision), so this is a documentation gap, not a methodology gap.

**Recommendation:** state it plainly — add one clause naming the test explicitly, e.g.
"(cluster-robust 95\% CI on the forward−backward difference excludes zero)" the first time this
kind of claim appears in 6.4, so the reader doesn't have to infer the mechanism from "cluster-
robust SE" alone. No code change needed, this is a one-sentence fix to already-correct work.

---

## 13. Attention-split vs. entropy-asymmetry correlation (existing `\ph{TODO}`)

**Where:** Section 6.4, `\ph{TODO}` currently reads: "does the per-dataset forward-minus-
backward attention mass track the per-dataset source-minus-target entropy gap from Section
5(A)?" (note: also has the stale "Section 5(A)" cross-reference bug flagged separately in the
closeout plan's Phase 1).

**Current test: not yet run.** Both quantities are already computed and quoted elsewhere in the
paper (6.4's own attention-split numbers; 6.1(C)'s entropy-asymmetry numbers) — this is a
6-point Spearman correlation, cheap once both are pulled into one table.

**Plain-language note on why Spearman, not Pearson, if you're deciding:** Spearman correlates
the *ranks* of the two quantities rather than their raw values, which is the safer choice with
only 6 data points — it doesn't assume a linear relationship or nicely-behaved (e.g.
homoscedastic, roughly normal) residuals the way Pearson's correlation does, both of which are
hard to justify with n=6.

**Recommendation:** fix/add cheaply — compute it, report ρ and whether it's significant at n=6
(critical value for two-sided $\alpha=0.05$ at n=6 is $|\rho|\ge0.886$, i.e. this correlation
needs to be quite strong to read as more than suggestive — the tex's own existing TODO already
acknowledges this: "with six points this is only suggestive"). Report honestly either way.

---

## 14. Vertex-vs-edge attention share vs. AUC boost over best GNN (new, planned)

**Where:** not yet in the tex — planned addition to Section 6.4 per the closeout plan's Phase 6.

**Current test: not yet run**, same shape as item #13 (6-point Spearman, both inputs likely
already computable from existing extract-script outputs — Panel C's "nodeedge" split for the
attention share, Table 1 for the AUC-boost side).

**Recommendation:** fix/add cheaply, same caveat as #13 about n=6 needing a strong correlation
to read as more than suggestive. If this correlation is found and holds up, per the plan it's a
candidate "alternative theory" for the edge-vs-vertex claim in item #2 — but per item #2's own
recommendation, don't let a suggestive 6-point correlation alone carry a headline abstract claim
without being explicit about how thin that evidentiary base is.

---

## 15. Figure 3 delta heatmap — "gain concentrates where the bound bites"

**Where:** Section 6.3, Figure 3 (`fig:delta-heatmap`) caption.

**Current test: none** — the claim is read directly off the heatmap's visual pattern (larger
Pewter-minus-SiGAT deltas in high-entropy cells) with no formal statistic.

**This is exactly what the professor's new Section 6.2 prose ask (item 4 in the closeout plan's
C-ter section) requests**: "a statistical claim on the difference in general and the
contribution of the entropy to the difference." Two candidate designs:
- **Correlate per-cell AUC delta against the cell's (source, target) entropy midpoint** — a
  simple regression of delta on entropy, reporting the slope and whether it's significant.
  Straightforward, but entropy bins are 2-D (source × target), so this needs a choice of how to
  reduce them to one axis (e.g. sum, or a 2-variable regression instead of a single correlation).
- **A 2-variable regression, delta $\sim$ source-entropy-bin + target-entropy-bin**, directly
  analogous to the two-way ANOVA already requested for Panel C (item #4 above) — this is
  probably the cleaner match to "the contribution of entropy to the difference," since it
  produces a coefficient per entropy axis rather than one pooled correlation.

**Feasibility:** the delta heatmap's underlying per-cell, per-seed data should already exist
from the same 10-seed SiGAT + PEWTER-local pipeline used to build the heatmap itself — check
`aaai2027/figure_data/pewter_sigat_delta_heatmap.csv`'s generation script for whether it kept
per-seed cell values or only the final mean, per this project's own "cache the expensive step
separately" convention (if it only kept the mean, the per-seed step needs rerunning, but that's
a cache-miss, not a new computation).

**Done 2026-08-18 — result is real but MIXED, complicates the current caption's claim rather
than confirming it.** Ran a per-dataset regression, `delta ~ src_entropy_mid + tgt_entropy_mid`
(cluster-robust SE, cluster=seed), one observation per (seed, cell), same `MIN_CELL_N_DELTA=50`
threshold and entropy axes as the existing heatmap:

| dataset | n obs | src coef | src p | tgt coef | tgt p |
|---|---|---|---|---|---|
| bitcoin-alpha | 76 | −0.115 | 0.227 (n.s.) | +0.115 | 0.114 (n.s.) |
| bitcoin-otc | 125 | **+0.131** | **0.0007** | **+0.194** | **0.0007** |
| epinions | 160 | **+0.054** | **0.0029** | +0.003 | 0.552 (n.s.) |
| wiki-elec | 160 | +0.000 | 0.986 (n.s.) | +0.007 | 0.383 (n.s.) |
| wiki-rfa | 160 | +0.013 | 0.111 (n.s.) | −0.004 | 0.448 (n.s.) |
| slashdot090221 | 160 | +0.005 | 0.125 (n.s.) | **−0.021** | **0.017** |

A positive coefficient means the delta (Pewter's AUC advantage) genuinely grows with that
entropy axis — the "gain concentrates where the bound bites" reading. **Only bitcoin-otc shows
this cleanly on both axes; epinions shows it on the source axis only; the other three (bitcoin-
alpha, wiki-elec, wiki-rfa) are flat/null on both axes; slashdot090221's target-entropy
coefficient is significant in the OPPOSITE direction** (higher target entropy → smaller Pewter
advantage there, not bigger). This is a real, computed result, not a bug — re-verified the
script's ROOT-path computation after an initial run silently produced zero usable observations
on every dataset (a missing `os.path.dirname()` level pointed it at `scripts/` instead of the
repo root; fixed, then reran and got the above). Script:
`scripts/paper_figures/delta_heatmap_entropy_regression.py`, data:
`aaai2027/figure_data/delta_heatmap_entropy_regression.csv`.

**Decided 2026-08-18, user picked option (a), compact form — now in the tex.** Rather than
stating "gain concentrates where the bound bites" as a uniform 6-dataset finding, Section 6.3's
prose (right after the Figure 3 paragraph) now names only the two datasets where the regression
is actually significant, in plain language with no statistical jargon (no "cluster-robust,"
no explicit "not significant on the others" clause — toned down per direct user request): "Using
a per-dataset regression of this delta on the source- and target-entropy bin midpoints, the
effect is significant on both axes for Bitcoin-otc and on the source-entropy axis for Epinions
(both $p<0.01$)." Per the user's explicit instruction, this stays compact and does not itemize
the other four datasets' individual coefficients/p-values (those remain in this audit doc's
table above and in `aaai2027/figure_data/delta_heatmap_entropy_regression.csv` if needed for a
supplementary table or a conversation with the professor) — slashdot090221's reversal on the
target axis is likewise not called out in the main text. Options (b)/(c) from the original three
(drop the framing entirely / investigate the bitcoin-otc-epinions split further) were not taken.

---

## 16. Dataset imbalance, prevalence statistics

**Where:** Datasets paragraph (77–94% positive), Ethical Considerations (same numbers,
restated).

**Current test: none, and none needed** — these are description of the raw data, not
comparisons between models or conditions. **State it plainly, no test.**

---

## Notes on scope and what's deliberately not covered here

- **Wilcoxon signed-rank test, logistic regression, cross-entropy loss** are used throughout
  the paper as standard, canonical tools and are not separately audited here as "claims" —
  per the earlier citation sweep this session, these are also deliberately left uncited (matches
  established ML/stats paper convention for naming a well-known test/method without a citation).
- **The Complexity section (6.6)** makes no empirical statistical claims (it's a theoretical
  asymptotic-complexity argument) and is out of scope for this audit.
- **Proposition/Lemma/Corollary proofs** are mathematical claims, not empirical statistical
  claims, and are being verified separately (see the closeout plan's math-verification task) —
  not duplicated here.
