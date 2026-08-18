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
| 1 | Pewter beats the best baseline, all 6 datasets, +3.3pp mean | Abstract, 6.2, 6.3, Conclusion | mean±std over 10 splits only | **needs a test added** | Add a paired test (Wilcoxon or paired bootstrap) per dataset across the 10 splits, vs. the strongest per-dataset baseline that has per-seed data; state as a compact parenthetical |
| 2 | Edge-vs-vertex / "in some graphs most of the prediction is from edges, in others from vertices" | Abstract, Contributions #5, 6.1(D) | none — investigation not started | **needs investigation + a test**, see the WSDM closeout plan | Do not state as a headline empirical claim until real data backs it (see Section D of the plan file) |
| 3 | MI decay 178×–73,000× between distance 1 and 2–3 | Contributions #2, 6.1(A) | none (point ratios, real vs. null-shuffle control exists in the underlying script but isn't quoted) | adequate as a descriptive statistic | **State it plainly, no test** — it's a magnitude claim, not a comparison needing p-values |
| 4 | Baseline AUC falls as endpoint entropy rises (bottleneck confirmation) | 6.1(B), Panel C | none — visual heatmap monotonicity only | **needs a test added** (2-way ANOVA requested) | Fix/add cheaply — data (`sigat_raw_seed()`) already supports it |
| 5 | Entropy asymmetry: source lower than target on 4/6, reversed on 2/6 | 6.1(C) | one-sided Wilcoxon signed-rank per dataset + DerSimonian-Laird pooled random-effects | correct test, no FDR correction across the 6 per-dataset tests | **Keep as-is**, note FDR non-issue given p-value magnitudes (see below) |
| 6 | Panel D (Fig. 2): sign-agreement bucketed AUC | 6.1(D)-adjacent / Figure 2 panel (d) caption | none — point AUC per bucket | **needs a test added**, plus multiseed rebuild (already planned) | Fix/add cheaply once the multiseed rebuild lands |
| 7 | Panel E (Fig. 2): entropy-term coefficients, which term dominates error | Figure 2 panel (e) caption | two-way cluster-robust SE (Cameron–Gelbach–Miller) + BH-FDR | correct test, but **citations missing from bib** (`cameron2011robust`, `benjamini1995controlling` both dropped) | **Fix/add cheaply** — re-add citations (needs your sign-off, see note below); new per-dataset stacked-bar view needs its own new test (not yet designed) |
| 8 | Ablation A: local attention matches/beats full on 4/6 datasets | 6.5(A) | "within reported standard errors" — informal eyeball | **needs a test added**, AND the "4/6" number itself is stale (real recompute shows 3/6, see plan Phase 4) | Fix/add cheaply — paired bootstrap across the 10 splits; fix the stale count in the same pass |
| 9 | Ablation B: aggregator choice barely matters (spread < 0.0022 AUC) | 6.5(B), Table 2 | none in the tex; a single-split paired-bootstrap script exists (`ablationB_paired_significance.py`) but isn't cited/rerun for the multiseed table | **needs updating**, not fundamentally missing | Fix/add cheaply once Ablation B's multiseed rebuild lands — rerun the existing bootstrap script against the new per-seed data |
| 10 | K-ablation: is the gain "just ensembling"? | 6.5(C) | mean±std AUC per K across 10 seeds | **done 2026-08-18** | K=1 already beats the best baseline on all 6 datasets; gap to saturated ceiling is <1pp everywhere — representation, not ensembling, carries the result. Script: `scripts/paper_figures/extract_ablation_kwalks.py` |
| 11 | Attention split: forward- vs. backward-dominant per dataset | 6.4, Panel B/C of Fig. 4 | one-way cluster-robust SE (cluster = target edge) + Wilcoxon confirmation | **done 2026-08-18** — significant on all 6 datasets, both the fwd/bwd split (Panel B) and the node/edge split (Panel C) | Error bars added to both plots; 6.4 prose updated. Script: `scripts/attention_directionality_panelB_se.py`, cached per-instance data at `outputs/attention_directionality/<ds>_local_panelBC_perinstance.pkl` |
| 12 | Shapley causal contribution: hop-1 forward-dominant on 3/6 datasets | 6.4, Panel D of Fig. 3 | one-way cluster-robust SE (cluster = target edge), CI-exclude-zero test | **done 2026-08-18** — test now named in-prose | Kept as-is |
| 13 | Attention-split vs. entropy-asymmetry correlation (6 points) | 6.4 | Spearman ρ=−0.71, p=0.11, n=6 | **done 2026-08-18** — not significant, stated as such | Kept as descriptive; TODO removed from tex |
| 14 | Vertex-vs-edge attention share vs. AUC boost over best GNN (6 points) | not yet in tex, feeds the edge-vs-vertex investigation (item #2) | Spearman ρ=−0.32, p=0.54, n=6 | **done 2026-08-18** — null result | Does not support "edge-leaning attention → bigger GNN advantage" as an alternative theory; do not cite this correlation as evidence for item #2's claim |
| 15 | Figure 3 (delta heatmap): "gain concentrates where the bound bites" | 6.3 | none — visual trend only | **needs a test added** | Fix/add cheaply — regress per-bin AUC delta on source/target entropy, or correlate |
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

**Recommendation:** run the paired test against whichever baseline is actually the
runner-up per dataset among those with real per-seed data (SiGAT is the runner-up on 5/6
datasets per Table 1's current numbers — check GSGNN's per-seed status first since it's the
runner-up on bitcoin-otc). State the result as a compact parenthetical in 6.2, e.g. "(paired
Wilcoxon across the 10 splits, $p<0.05$ on N/6 datasets)" — this is exactly the level of detail
the professor's 6.2 prose ask wants. **Do not claim the paired test covers GCN/GAT** — those
stay point-estimate comparisons only, and the prose should say so if it names them specifically
rather than implying every row in Table 1 got the same statistical treatment.

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

**Feasibility, checked directly:** yes, cheap. `scripts/paper_figures/
extract_multiseed_entropy_heatmaps.py`'s `sigat_raw_seed()` already loads per-seed, per-edge
SiGAT predictions with their entropy-bin assignment for all 10 seeds — exactly the granularity
a two-way ANOVA needs (one AUC-contributing observation per (seed, cell), 10 seeds × 16 cells
per dataset). This is a `scipy.stats`/`statsmodels` two-way ANOVA call once the per-(seed,cell)
AUCs are tabulated — no new predictions need to be generated.

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

**Already planned (Phase 6 of the closeout plan): rebuild as multiseed (10 seeds) with error
bars**, reusing `sigat_raw_seed()` the same way Panel C's SiGAT cells were rebuilt. Once that
lands, the natural test is the same paired-bootstrap-across-seeds pattern used elsewhere: is the
same-bucket AUC significantly different from the diff-bucket AUC, per direction (in/out), using
the 10 seeds as the resampling unit (mean±SD across seeds, "Option 2" convention already
established in this project for exactly this kind of multi-split combination).

**Recommendation:** fix/add cheaply, but only after the multiseed rebuild itself lands — testing
against single-split point estimates isn't worth doing twice.

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

**New gap, not yet designed:** the professor's ask to rebuild Panel E as a per-dataset stacked
barplot (rather than the current pooled bars) needs its own new significance test — e.g., is a
given entropy term's per-dataset coefficient significantly different from zero (or from the
pooled estimate)? Not yet specified; do this alongside the stacked-barplot rebuild itself.

**Recommendation:** fix/add cheaply for the citation gap (pending your sign-off on which
citation, if any, to re-add); the new per-dataset test needs to be designed as part of the
Panel E rebuild task, not bolted on after.

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

**Recommendation:** fix/add cheaply, but sequence it after Ablation B's own planned multiseed
rebuild (Phase 4) — rerunning a paired-bootstrap script against soon-to-be-replaced single-split
predictions is wasted work. Once the 10-seed per-function predictions exist, port the existing
script's resampling logic to run per-seed and pool, or run it once per seed and report how many
of the 10 seeds agree on which aggregator wins (a simple, honest way to show the "barely
matters" claim is not an artifact of one split).

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

**Recommendation:** fix/add cheaply, prefer the 2-variable regression framing (consistent with
item #4's ANOVA) over a single pooled correlation — report the entropy coefficients' sign,
magnitude, and significance in one sentence in the 6.2/6.3 prose.

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
