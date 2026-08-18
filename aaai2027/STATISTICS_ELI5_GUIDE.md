# Statistics ELI5 Guide — PEWTER (WSDM)

**Purpose.** Not part of the paper, and not a catalog of which test backs which claim
(that's `STATISTICAL_TESTS_AUDIT.md`). This is a plain-language reference for the
statistical machinery this project actually uses, written for someone who doesn't
remember what each test is called or why you'd reach for it. Read this when a term in
the audit doc, the tex, or a script docstring doesn't ring a bell.

---

## Quick-reference: "I want to..." → "...use this"

| I want to... | Use | Section below |
|---|---|---|
| Compare two things measured on the **same** units (e.g. each vertex's own out-entropy vs. its own in-entropy) | Paired test — Wilcoxon signed-rank (non-normal) or paired $t$-test (normal-ish) | [Comparing two related things](#comparing-two-related-things-paired-tests) |
| Compare two things measured on **different** units (e.g. Pewter's 10 splits vs. a totally different model's 10 splits, no shared pairing) | Independent-samples test, or better: a paired bootstrap if there's a natural pairing (same splits) | [Comparing two related things](#comparing-two-related-things-paired-tests) |
| Know how uncertain a mean is | Standard error (SE), then a confidence interval (CI) | [Mean, SD, SE, CI](#mean-standard-deviation-se-and-ci-the-four-things-people-mix-up) |
| Say "X is significantly different from Y" | A test with a p-value or a CI that excludes zero — not just eyeballing whether two SE bars overlap | [p-values and significance](#p-values-and-significance-what-they-actually-mean) |
| Test the effect of two factors at once (e.g. source entropy AND target entropy on AUC) | Two-way ANOVA | [ANOVA](#two-way-anova-testing-two-factors-at-once) |
| Check if two things move together (e.g. attention split vs. entropy asymmetry, across 6 datasets) | Correlation — Spearman (rank-based, safer with few points) over Pearson | [Correlation](#correlation-spearman-vs-pearson) |
| Run many tests at once and not fool yourself with false positives | Multiple-comparison correction — FDR/BH (or the stricter Bonferroni) | [Multiple comparisons](#multiple-comparisons-why-testing-many-things-at-once-is-dangerous) |
| Get an honest standard error when observations aren't independent (e.g. many edges share the same vertex) | Cluster-robust standard errors (one-way or two-way) | [Cluster-robust SE](#cluster-robust-standard-errors-when-observations-arent-independent) |
| Combine a result measured separately on 4-6 datasets into one number | Random-effects meta-analysis (DerSimonian-Laird) | [Meta-analysis](#combining-results-across-datasets-random-effects-meta-analysis) |
| Say which of several inputs actually caused a prediction, not just correlated with it | Exact Shapley value | [Shapley values](#shapley-values-what-actually-caused-this-prediction) |
| Report a metric across many random splits so it's not a fluke of one split | Mean ± std across seeds | [Multi-seed](#multi-seed--multi-split-evaluation-why-10-different-splits) |

---

## Mean, standard deviation, SE, and CI: the four things people mix up

- **Mean**: the average. Not interesting on its own without knowing how much it varies.
- **Standard deviation (SD)**: how spread out the individual *data points* are around
  the mean. If bitcoin-alpha's Pewter AUC across 10 splits is $0.9146 \pm 0.0102$ std,
  that "$\pm0.0102$" says: if you picked one of those 10 splits at random, its AUC
  typically sits about 0.01 away from the mean. This is a property of the data, and
  doesn't shrink just because you collected more splits.
- **Standard error (SE)**: how uncertain the *mean itself* is — i.e. if you reran the
  whole 10-split experiment again, how much would the reported mean wobble?
  $\text{SE} = \text{SD}/\sqrt{n}$. Unlike SD, SE *does* shrink as you add more splits
  (more data → more confident about where the true average is), which is why SE is the
  right quantity for a significance test and SD is the right quantity for "how variable
  is a single run."
- **Confidence interval (CI)**: a range built from the SE that's meant to contain the
  true value most of the time. A "95% CI" is (roughly) the range you'd expect to cover
  the true value 95 times out of 100 if you repeated the whole experiment over and over.
  **The practical shortcut used throughout this project**: if a 95% CI on a *difference*
  (e.g. forward-minus-backward attention mass) excludes zero, that difference is
  "significant" at the conventional 5% threshold — this is mathematically the same
  conclusion a p-value < 0.05 would give you, just read off an interval instead of a
  single number.

**Which one is reported where in this paper**: Table 1 reports mean ± **std** (so you
know how noisy a single split's AUC is). The cluster-robust-SE claims (Figure 2 panel
(e), Figure 3 panel (d)) report **SE**-based CIs (so you know whether the *average*
effect is distinguishable from zero).

---

## p-values and significance: what they actually mean

A **p-value** answers a narrow, specific question: *if there were truly no effect (the
"null hypothesis" — e.g. "source entropy and target entropy are really the same"), how
surprising would data this extreme be, just from random noise?* A small p-value (by
convention, $p<0.05$) means "data this extreme would be rare under pure chance," which
is usually read as evidence the null hypothesis is false.

**What it does NOT mean** (common trip-ups, worth knowing before quoting one):
- It is **not** "the probability the null hypothesis is true." That's a different
  quantity (needs Bayesian reasoning, not what these tests compute).
- It is **not** a measure of *how big* the effect is. A p-value of $10^{-244}$ (as seen
  in this paper's 6.1(C) entropy-asymmetry test on Epinions) doesn't mean the effect is
  244 times bigger than one with $p=0.01$ — it mostly reflects sample size (Epinions has
  tens of thousands of vertex pairs, so even a small, consistent effect produces an
  extreme p-value). **Effect size and significance are different axes** — see
  [rank-biserial correlation](#effect-size-how-big-not-just-whether) below for the
  "how big" half of this project's entropy-asymmetry claim.
- "Not significant" does **not** mean "no effect" — it means the data couldn't
  distinguish the effect from noise at this sample size. This project's own item #13
  correlation (attention split vs. entropy asymmetry, $\rho=-0.71$, $p=0.11$) is a good
  example: a fairly strong-looking correlation that isn't "significant" only because
  $n=6$ is a very small sample to test with — six data points can only produce a
  significant Spearman correlation if $|\rho|\ge0.886$, so a $-0.71$ genuinely can't
  clear that bar even though it looks substantial.

---

## Comparing two related things (paired tests)

**The core idea**: if you're comparing two measurements that come from *the same
underlying unit* (the same vertex, the same edge, the same train/test split), you get a
much more powerful, more honest test by looking at the **differences** per unit, rather
than treating the two groups as unrelated.

**Why pairing matters, concretely**: suppose Pewter and SiGAT are both evaluated on the
same 10 splits. Split #3 might just be an "easier" split for everyone (better
train/test overlap by chance) — that shared per-split noise cancels out if you look at
(Pewter's AUC on split 3) − (SiGAT's AUC on split 3), but it doesn't cancel if you
naively compare "Pewter's 10 numbers" against "SiGAT's 10 numbers" as if they were two
unrelated piles of data.

- **Paired $t$-test**: the classic tool — tests whether the *average* per-unit
  difference is zero, assuming those differences are roughly normally distributed
  (bell-curve-shaped).
- **Wilcoxon signed-rank test** (used in this paper's 6.1(C), the entropy-asymmetry
  claim): the non-parametric cousin of the paired $t$-test. Instead of assuming a bell
  curve, it just ranks the differences and checks whether positive differences tend to
  outrank negative ones. **Why this paper uses Wilcoxon instead of a paired
  $t$-test**: entropy values are bounded in $[0,1]$ and pile up heavily at exactly 0
  (any vertex with unanimous incident signs has zero entropy) — a real spike at a
  boundary, not a bell curve, so the paired $t$-test's normality assumption would be on
  shaky ground here.
- **Paired bootstrap** (used in `ablationB_paired_significance.py`, and the
  audit-recommended fix for Table 1's "Pewter beats every baseline" claim): instead of
  a formula, you *simulate* the uncertainty directly — resample the paired units (e.g.
  test edges) with replacement thousands of times, recompute the difference each time
  on the *same* resample for both models, and look at the spread of those simulated
  differences. This works even when the theory behind a formula-based test would be
  shaky, at the cost of needing a computer to do the resampling.
- **One-sided vs. two-sided**: a "one-sided" test (used in 6.1(C)) checks a specific
  predicted direction ("is source entropy *lower* than target entropy?") rather than
  just "are they different at all" (two-sided). One-sided is the right choice only when
  you had a specific direction in mind *before* looking at the data — using it to
  "get" a smaller p-value after the fact would be a red flag.

---

## Two-way ANOVA: testing two factors at once

**ANOVA** (ANalysis Of VAriance) tests whether grouping your data by some factor(s)
explains more of the variation in an outcome than you'd expect from chance alone.
**Two-way** ANOVA does this for two factors simultaneously — e.g. this project's
planned Panel B/C test: does *source*-entropy bin and *target*-entropy bin (two
separate factors) each affect AUC, and do they interact (does the effect of one factor
depend on the level of the other)?

A two-way ANOVA reports (at minimum) three things:
1. **Main effect of factor 1** (source entropy): does AUC vary across source-entropy
   bins, averaging over target-entropy bins?
2. **Main effect of factor 2** (target entropy): same question for target entropy.
3. **Interaction**: does source entropy's effect on AUC change depending on which
   target-entropy bin you're in? (E.g. maybe source entropy matters a lot when target
   entropy is low, but barely at all when target entropy is already high.)

**Using cross-validation splits as "repeated measurements"** (the professor's
suggestion for this project): instead of having just one AUC value per (source-bin,
target-bin) cell, use each of the 10 splits' own cell AUC as a separate observation —
this gives the ANOVA real within-cell variability to work with (10 numbers per cell
instead of 1), rather than treating a single point estimate as if it had no
uncertainty.

---

## Correlation: Spearman vs. Pearson

Both measure "do these two things move together," on a scale from $-1$ (perfectly
opposite) to $+1$ (perfectly together), $0$ = no relationship.

- **Pearson correlation**: measures a *linear* relationship between the raw values.
  Sensitive to outliers and assumes the relationship, if any, is roughly a straight
  line.
- **Spearman correlation**: converts both variables to **ranks** first (1st smallest,
  2nd smallest, ...), then correlates the ranks. This only requires the relationship to
  be *monotonic* (consistently increasing or decreasing, not necessarily a straight
  line), and is far less sensitive to a single extreme outlier skewing the result.

**Why this project uses Spearman everywhere it correlates things** (attention split vs.
entropy asymmetry, vertex/edge attention share vs. AUC boost): both of those
correlations only have **6 data points** (one per dataset). With that few points, a
single unusual dataset (e.g. wiki-rfa's very backward-heavy attention split) could
distort a Pearson correlation a lot; Spearman is the safer default at small $n$.

**A rule of thumb worth remembering for this project**: with only 6 points, a two-sided
Spearman correlation needs $|\rho|\ge0.886$ to be "significant" at the conventional
5% threshold. That's a high bar — most correlations computed on 6 datasets in this
project will read as "suggestive, not significant," and that's an honest reflection of
how little data 6 points really is, not a flaw in the test.

---

## Multiple comparisons: why testing many things at once is dangerous

If you run 20 independent tests, each with a 5% false-positive rate, you'd expect about
**one of them to look "significant" purely by chance**, even if nothing real is going
on anywhere. Running many tests and reporting only the "significant" ones without
correcting for this is a classic way papers accidentally report noise as a finding.

Two standard fixes:

- **Bonferroni correction**: the strict, simple fix — divide your significance
  threshold by the number of tests (e.g. testing at $p<0.05/20=0.0025$ instead of
  $p<0.05$). Very conservative — controls the chance of *even one* false positive
  across all your tests (the "family-wise error rate"), which gets very hard to
  achieve as the number of tests grows.
- **Benjamini-Hochberg (BH) / FDR correction** (used in this paper, Figure 2 panel (e)
  and Figure 1 panel (e)'s coefficient significance stars): a less strict alternative
  that controls the **false discovery rate** — of all the results you call
  "significant," what fraction are expected to be false positives, rather than trying
  to avoid a single false positive entirely. This is the standard choice when testing
  several related things from one analysis (e.g. several coefficients in the same
  regression) and you expect some of them to be real — FDR lets real, smaller effects
  survive that Bonferroni would likely kill. **"$q$-value"** is BH's version of a
  p-value — a $q<0.05$ means "if you called everything with $q<0.05$ significant, you'd
  expect at most 5% of those calls to be wrong," as opposed to a raw p-value which
  makes no such promise once you're testing many things at once.

**When this project doesn't bother with correction (and why that's fine, not
sloppy)**: 6.1(C)'s six per-dataset Wilcoxon tests aren't FDR/Bonferroni-corrected, but
the smallest p-values there are astronomically small ($10^{-10}$ to $10^{-244}$) — even
dividing by 6 (Bonferroni) or applying FDR leaves every one of them significant by many
orders of magnitude. Correction matters when p-values are *close* to the threshold;
it's a non-issue when they're nowhere near it.

---

## Cluster-robust standard errors: when observations aren't independent

Most standard error formulas assume every data point is an **independent** draw. That
assumption breaks when your data has natural groups that share hidden correlated noise
— e.g. many edges share the same source vertex $u$; if $u$ happens to be a genuinely
"hard" or unusual vertex, *every* edge touching $u$ will tend to err in a correlated
way, not independently. Treating each edge as fully independent in that case
understates your true uncertainty (a naive SE would be too small, making effects look
more "significant" than they really are).

**Cluster-robust SE** fixes this: instead of pretending every edge is independent, it
computes the outcome averaged **within each cluster** (e.g. within each vertex's edges)
first, then measures how much those *cluster-level* averages vary. Since there are
fewer clusters than raw data points, and clusters absorb the correlated noise, this
gives an honest (usually larger, more conservative) standard error.

- **One-way clustering** (used in `shap_edge_directionality.py` and the new attention
  forward/backward-split SE): cluster by a single grouping variable — here, target
  edge_id, since one edge can appear as a token in many different sampled walks, and
  those occurrences aren't independent measurements of "how forward/backward-leaning is
  this edge."
- **Two-way clustering** (Cameron-Gelbach-Miller 2011, used in Figure 2 panel (e)'s
  entropy-term regression): used when a single observation belongs to **two**
  overlapping groups at once — an edge belongs to both its source vertex's cluster
  *and* its target vertex's cluster. The formula combines both one-way variance
  estimates and subtracts their overlap: $V = V_u + V_v - V_{u,v}$ (add each one-way
  cluster's contribution, then subtract the part that would otherwise be double-counted
  where the two groupings overlap).

---

## Combining results across datasets: random-effects meta-analysis

**The problem**: this paper's entropy-asymmetry finding holds (in the same direction)
on 4 of 6 datasets. How do you report one combined "how strong is this effect" number
across those 4 datasets, given each dataset is a different size and might have a
genuinely different true effect size (not just noisier measurements of the same true
number)?

- **Fixed-effects pooling** (the naive approach): assumes every dataset is measuring
  *the exact same* true effect, and any difference between datasets is purely sampling
  noise. Wrong assumption here — this paper's own data shows real heterogeneity (2 of 6
  datasets reverse direction entirely, so datasets clearly don't all share one true
  effect).
- **Random-effects meta-analysis (DerSimonian-Laird, used in 6.1(C))**: allows each
  dataset to have its own true effect, drawn from some distribution of "plausible true
  effects" — it pools the datasets' estimates while inflating the combined uncertainty
  to account for the fact that datasets genuinely disagree with each other, not just
  due to sampling noise. This is why the paper's pooled estimate ($58.1\%$, 95% CI
  $[55.3\%,60.8\%]$) is only computed across the 4 *confirming* datasets, not all 6 —
  pooling all 6 together (with 2 pointing the opposite direction) correctly produces a
  non-significant, null-spanning interval, since there's no one shared true effect to
  estimate across a group that disagrees on direction.

---

## Shapley values: what actually caused this prediction?

**The problem attention weight doesn't solve**: high attention on some context token
tells you the model is *looking* at it, but not necessarily that the token actually
*changed* the prediction. A head could allocate a lot of raw attention mass to a token
whose presence or absence barely moves the output.

**Shapley values** (from cooperative game theory — originally about fairly splitting a
group payoff among contributors) answer the causal version of this question directly:
for a specific prediction, systematically try every possible subset of the candidate
context features present-vs-masked, and measure how much the prediction's output
changes on average when a given feature is added to a subset, averaged over *every*
possible subset it could be added to (so the credit assigned to one feature properly
accounts for its interactions with every other feature, not just its effect in
isolation). With few enough candidate features (this paper's Shapley analysis has at
most 4: forward/backward × hop-1/hop-2), this can be computed **exactly** — trying all
$2^4=16$ subsets — with no approximation needed, unlike most real-world Shapley
applications (e.g. SHAP on image models) which need to *sample* subsets because there
are too many to try them all.

**Why this project found a case (wiki-rfa) where attention weight and Shapley
disagree**: wiki-rfa's raw attention mass is the *most* backward-heavy of all 6
datasets, yet its Shapley-measured causal contribution is significantly *forward*-
dominant. This is exactly the failure mode Shapley is meant to catch — a head can point
a lot of raw weight backward while the backward tokens' actual marginal effect on the
output is smaller than the forward tokens'.

---

## Multi-seed / multi-split evaluation: why 10 different splits

Any single train/test split is one particular random draw — a model's reported AUC on
that one split reflects both genuine model quality *and* that split's own luck (which
edges happened to land in test, which happened to be easy or hard). Reporting a single
split's number risks reporting noise as if it were a stable property of the model.

**Running the same experiment on 10 independent splits (seeds 42-51) and reporting
mean ± std** answers two different questions at once: the **mean** estimates the
model's true average performance (averaging out any one split's luck), and the **std**
tells you how much that performance actually varies from split to split — which, per
this project's own finding, turned out to be noticeably *wider* than an
analytically-computed standard error would have suggested (the old Hanley-McNeil
analytic SE only captures within-split sampling noise, not the extra split-to-split
variance that comes from which edges happen to land in which split) — a real,
measured example of why re-running on multiple real splits catches something a formula
alone would have understated.

---

## Effect size: how big, not just whether

A significance test (p-value, CI) answers "is this real or just noise" — it does not
answer "how big is this, in a way I can compare across different measurements." Two
common effect-size measures that show up in this project:

- **Rank-biserial correlation $r$** (used in the entropy-asymmetry `claim1_table.csv`):
  a correlation-like number, from $-1$ to $+1$, summarizing how consistently one group
  of paired values beats the other, on a scale that doesn't depend on the units of the
  original measurement (unlike, say, a raw mean difference) — useful for comparing
  "how strong is this effect" across datasets whose entropy scales differ.
- **AUC** itself is technically an effect-size-like metric before it's ever a
  significance test: it is the probability that a random positive example is scored
  higher than a random negative example, which is why AUC=0.5 means "no better than a
  coin flip" and AUC=1.0 means "perfect separation" — an AUC difference of, say,
  $+0.03$ is a magnitude claim (how much better), separate from whatever paired test
  you'd run to ask whether that $+0.03$ is distinguishable from noise.

**The general lesson worth keeping in mind reading this paper**: a huge dataset (like
Epinions, tens of thousands of vertex pairs) can produce an astronomically small
p-value for an effect that is, in absolute terms, quite small — and a small dataset can
produce a large, real effect that never clears significance. Always ask both questions
("is it real?" and "how big is it?") separately; neither one substitutes for the other.
