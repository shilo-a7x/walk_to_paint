# Mutual information, ELI5 — how we compute it, why it's biased, how to fix it

This is the companion doc for Empirical Confirmation Panel B (the "information decay with
distance" line plot) and the investigation into its post-minimum "bump" at distance 4-6
(see `CLAUDE.md`'s Figure 1 status note for the full investigation writeup; this doc is
just "what is this number, and what can go wrong with it," explained from zero).

---

## 1. What are we even measuring?

We want to know: **if I already know the sign of one edge, does that tell me anything about
the sign of a *different* edge sitting some distance away in the graph?**

"Tells me anything" has a precise meaning: **mutual information**, $I(Y; S)$, where:
- $Y$ = the sign (+/-) of an "anchor" edge
- $S$ = the sign (+/-) of a "context" edge sitting at some fixed line-graph distance from it

If knowing $S$ never changes your guess about $Y$ — the two are statistically independent —
then $I(Y;S) = 0$. If knowing $S$ pins $Y$ down exactly, $I(Y;S)$ is as large as it can be
(equal to $H(Y)$, the total uncertainty in $Y$ on its own). Everything in between is a
partial, probabilistic hint.

We report **NMI** = $I(Y;S) / H(Y)$ — mutual information as a *fraction* of the anchor's own
total uncertainty. This puts every dataset on the same 0-to-1 "how much of the mystery did
the context edge solve" scale, regardless of how imbalanced that dataset's signs are.

---

## 2. How we actually build the number: the contingency table

For a fixed distance $d$ (say, "distance 3"), we scan a large number of (anchor edge,
context edge) pairs that are exactly $d$ apart, and for each pair we just look at two coin
flips: was the anchor $+$ or $-$? Was the context edge $+$ or $-$? Every pair falls into
exactly one of 4 buckets:

| | context $S=-$ | context $S=+$ |
|---|---|---|
| **anchor $Y=-$** | $n_{--}$ | $n_{-+}$ |
| **anchor $Y=+$** | $n_{+-}$ | $n_{++}$ |

That's it — that 2x2 table of counts *is* the entire dataset, as far as this calculation is
concerned. On bitcoin-alpha at distance 1, for example, this table has about 4.2 million
pairs spread across its 4 cells; at distance 3, about 292 million.

---

## 3. The formula (the "plug-in" estimator)

Turn every count into a fraction of the total ($\hat p_{ij} = n_{ij}/N$, same for the row
sums $\hat p_{i\cdot}$ and column sums $\hat p_{\cdot j}$), then:

$$\hat I(Y;S) = \sum_{i,j} \hat p_{ij} \log_2\!\left(\frac{\hat p_{ij}}{\hat p_{i\cdot}\,\hat p_{\cdot j}}\right)$$

In plain words, for each of the 4 cells: **"is this cell more or less common than you'd
expect if $Y$ and $S$ were totally unrelated?"** If a cell's real frequency matches what
independence would predict, that cell contributes 0. If it's way more (or less) common than
independence predicts, that cell contributes a positive amount. Sum the 4 cells, and you
get $\hat I(Y;S)$ in bits. This is called the **plug-in estimator** because we just plug the
*observed* frequencies straight into the textbook MI formula, as if they were the *true*
probabilities.

**A tiny worked example.** Say we flip two fair, totally independent coins 100 times each
and tally them:

| | $S=-$ | $S=+$ |
|---|---|---|
| $Y=-$ | 24 | 26 |
| $Y=+$ | 27 | 23 |

True answer: $I(Y;S) = 0$ exactly (they're independent by construction). But plug in these
numbers and you get $\hat I \approx 0.0009$ bits — small, but *not* zero. That's the whole
problem in miniature, scaled up: **pure noise never averages out to exactly the textbook
answer; it averages out to something slightly *above* it.**

---

## 4. Why the plug-in estimator is always biased *upward*

Three ways to see why, from most to least intuitive:

1. **Finite samples always show *some* accidental pattern.** Flip two independent fair
   coins 100 times — you will basically never get *exactly* 25/25/25/25. Random
   fluctuation alone always creates a *little* apparent association. The plug-in formula
   has no way to tell "this is a real pattern" from "this is just 100 coin flips wiggling
   around" — it reports whatever's in front of it as if it were the truth.
2. **The direction of the noise only ever pushes one way.** $\hat I(Y;S) \ge 0$ always,
   by construction (it's a discrete KL divergence, which can never be negative). So
   sampling noise can push the estimate *up* from the true value, but it structurally
   can't push it *down* past 0. Average that over many repeated experiments and the
   *expected* value of $\hat I$ is strictly above the true $I$ whenever the true $I$ is
   small — there's a floor at 0 but no ceiling on the noise.
3. **It gets worse as $N$ shrinks.** With more coin flips, chance patterns average out
   and the plug-in estimate creeps closer to the truth. With fewer flips, chance patterns
   are relatively larger, so the upward bias is relatively larger too. This is exactly why
   the bump in Panel B shows up specifically at the distances where the *sample size*
   collapses (from ~300 million pairs at distance 3 down to ~180 thousand at distance 6 on
   bitcoin-alpha) — smaller $N$, bigger upward bias, and the *true* signal out there is
   already close to 0, so the bias is no longer a rounding error, it's the whole story.

---

## 5. A formula for the bias: the Miller-Madow correction

For a table with $r$ rows and $c$ columns and $N$ total samples, the *expected size* of this
upward bias is approximately:

$$\text{bias} \approx \frac{(r-1)(c-1)}{2N\ln 2} \text{ bits}$$

For our 2x2 tables, $(r-1)(c-1) = 1$, so it simplifies to $\dfrac{1}{2N\ln 2}$. Plug in
bitcoin-alpha's distance-6 sample size ($N \approx 180{,}672$): predicted bias
$\approx 4\times 10^{-6}$ bits. When we actually ran a **shuffle-signs control**
(randomly scramble which edges are $+$/$-$, keeping the graph itself untouched, so the
*true* MI is exactly 0 by construction) at that exact distance, the measured "MI" came out
to $4.27\times10^{-6}$ bits — matching the formula almost exactly. That's a satisfying
confirmation that this part of the story is textbook small-sample bias, not a mystery.

**But** — and this is the twist we found — the *real* (non-shuffled) signal at that same
distance is roughly 90-3000x bigger than this bias floor on most datasets. So Miller-Madow
bias is real, it's measurable, and it matches theory beautifully... but it's not big enough
to explain the actual bump. Something else is going on too (see §7).

---

## 6. How to "unbias" it — three ways, from quick-and-dirty to fully rigorous

**(a) Subtract the Miller-Madow formula.** Fast, no extra computation, but it's an
*asymptotic approximation* — it assumes samples are independent draws, which (see §7) they
turn out not to be here, so it under-corrects in exactly the regime we care about most.

**(b) Subtract an empirical null (what we actually did).** Instead of trusting a formula,
run the *exact same* computation on **the exact same graph** but with signs randomly
shuffled first. Since the true MI is 0 by construction, whatever number comes out *is* the
bias (from sampling noise, AND from any other systematic wrinkle in how the pairs were
counted — a strictly more honest estimate than the formula, since it doesn't assume anything
about independence). Subtract that from the real measurement:
$\text{debiased NMI} = \text{NMI}_{\text{real}} - \text{NMI}_{\text{shuffled}}$. This is what
let us confirm the bump is mostly real (survives subtraction by 2-3 orders of magnitude at
distance 5-6 on most datasets) rather than pure bias.

**(c) Fix the actual root cause: the same real edge gets counted many times over.** Both
(a) and (b) assume you're just correcting a noisy-but-honest count of independent
observations. We found that's not quite true here — see §7 for exactly what this means and
why it isn't a data bug.

---

## 7. "Duplicate edges" — what that actually means (it's not a data bug)

To be very clear up front: **there are no duplicate edges in the dataset.** Every edge in
the graph appears exactly once. What gets reused is not the *data* — it's how many times
the *same single real edge* ends up as a row in our contingency table.

**A real measurement, not a made-up example.** I sampled 3,000 anchor edges on bitcoin-alpha
and asked, at the farthest tested distance (line-distance 6): which single context edge got
reached by the most different anchors? Answer: **edge #23712** (node 2665 ↔ node 1697,
sign `+`) — reached by **1,856 of the 3,000 anchors, 62% of them.** Five of those anchors,
picked at random, have nothing to do with each other:

| anchor edge | anchor's own nodes | anchor's sign | paired with edge #23712's sign |
|---|---|---|---|
| #10601 | 241 ↔ 55 | + | + |
| #21917 | 1116 ↔ 619 | + | + |
| #7837 | 28 ↔ 3690 | + | + |
| #5469 | 14 ↔ 1872 | + | + |
| #23611 | 1567 ↔ 3773 | − | + |

That's 5 rows in the table — out of 1,856 total for this one edge — all reporting
"context sign = +", because they all bottom out at the same single real edge. There is
exactly *one* fact here ("edge #23712 is +"), but the table records it as if it were 1,856
separate coin flips, all landing on the same side.

**Why this one edge, and why so many anchors hit it.** Note edge #23712's own endpoints have
degree 2 and 4 — nothing special, not a "hub" in any obvious sense. The real reason is
simpler: **bitcoin-alpha is a small graph (3,783 nodes) with a small diameter.** Starting
from almost *any* anchor and expanding outward 5-6 hops sweeps up nearly the whole graph.
By the time you're 5-6 hops out, only a handful of nodes are left that haven't already been
swept into somebody's search — and since nearly every anchor's search has already covered
almost everything else, that small remaining "still not yet reached" pool ends up being
nearly the *same* small pool for every anchor. It's not that edge #23712 is special — it's
that in a graph this size, there's only a tiny sliver of "still far away" left once you've
gone this many hops, and that sliver is shared by almost everyone.

**Why this only shows up far from the anchor.** Near the anchor (distance 1-2), the set of
"nearby edges" is almost always genuinely different from one anchor to the next — two
random anchors are rarely close enough to share neighbors, so each row really is close to a
fresh, independent fact. Far from the anchor, the graph is running out of anything left to
reach at all (recall: wiki-elec/wiki-rfa hit *zero* new edges past distance 6 — the graph
is fully exhausted by then), so what remains gets rediscovered over and over. Measured
directly on bitcoin-alpha: at distance 1, the 10 most-reused context edges account for only
**0.2%** of all table rows (no problem — plenty of genuinely distinct edges). At the
farthest distance, the 10 most-reused edges account for **~40%** of all rows — nearly half
of what looks like "hundreds of thousands of independent observations" is really a handful
of real edges re-typed into the table hundreds of times each.

**Why that breaks the MI calculation.** The MI formula (§3) assumes every row is an
independent draw. If 40% of the rows are really "the same fact about edge $X$,
copy-pasted," then (i) the table looks far more confident/certain than the underlying
evidence actually is — the effective sample size is much smaller than the row count
suggests, so even the shuffle-null correction in (b) above is itself estimated on a noisier
basis than its huge nominal $N$ implies; and (ii) if $X$'s sign happens, by pure chance, to
line up even slightly with something about *which* anchors reach it, that one-edge
coincidence gets amplified into what looks like a broad, dataset-wide correlation — purely
because it got counted hundreds of times instead of once. This is the same problem as
polling "1,000 students" for an opinion when you really only sampled 10 classrooms and
asked everyone in each — you don't have 1,000 independent opinions, you have 10.

**The fix ("dedup").** Instead of adding one row per (anchor, context-edge) *occurrence*,
add one row per **distinct** context edge — so edge $X$ contributes exactly once to the
table no matter how many different anchors happened to rediscover it. Not yet implemented
(would require reworking the contingency-table accumulation from "one increment per anchor
that reached this edge" to "one increment per unique edge id"); see `CLAUDE.md`'s Figure 1
status note for where this stands.

---

## Part 2: trying a completely different statistic — phi / Spearman correlation

MI has one annoying property that makes it hard to fully trust at the tail: it can **only
ever be zero or positive**, never negative (see §4). That means when we see an elevated
number at distance 4-6, we can't tell from MI alone whether that's "real correlation" or
"one-sided noise inflating a true value of zero" — both look like *some positive number*.
A plain old **correlation coefficient** doesn't have this problem: under pure noise, its
expected value is exactly 0, and it can land above or below that with equal ease. So we
reran the *exact same* pipeline (same datasets, same distances, same 2×2 tables) and
computed a correlation coefficient from them instead, to see if it agrees with the MI
picture or reveals something MI structurally can't show.

## 8. What is "phi," and how is it calculated?

**Phi ($\phi$) is just the ordinary correlation coefficient (the same "Pearson $r$" from an
intro stats class), specialized to two yes/no variables.** Same question as always — "when
one edge is $+$, does the other tend to be $+$ too, or tend to be $-$?" — just answered with
a number that can be negative, unlike MI.

We compute it directly from the **same four counts** already built for MI (§2):

| | context $S=-$ | context $S=+$ |
|---|---|---|
| **anchor $Y=-$** | $n_{--}$ | $n_{-+}$ |
| **anchor $Y=+$** | $n_{+-}$ | $n_{++}$ |

$$\phi = \frac{n_{++}\,n_{--} \;-\; n_{+-}\,n_{-+}}{\sqrt{(n_{+-}{+}n_{++})(n_{--}{+}n_{-+})(n_{--}{+}n_{+-})(n_{-+}{+}n_{++})}}$$

That looks like a mouthful, but the numerator is the intuitive part: it's comparing "how
often did the two edges *agree*" ($n_{++}\cdot n_{--}$, both same-sign) against "how often
did they *disagree*" ($n_{+-}\cdot n_{-+}$, opposite-sign). If agreement dominates, $\phi>0$.
If disagreement dominates, $\phi<0$. If they're evenly split, $\phi=0$. The square-root part
in the denominator just rescales the answer to always land between $-1$ and $+1$, the same
convention as any correlation coefficient, regardless of how imbalanced the two variables
are individually.

**A tiny worked example.** Say 20 (anchor, context) pairs give this table:

| | $S=-$ | $S=+$ | row total |
|---|---|---|---|
| $Y=-$ | 3 | 7 | 10 |
| $Y=+$ | 8 | 2 | 10 |
| col total | 11 | 9 | 20 |

$$\phi = \frac{(2)(3) - (8)(7)}{\sqrt{10\cdot10\cdot11\cdot9}} = \frac{6-56}{\sqrt{9900}} = \frac{-50}{99.5} \approx -0.50$$

Reading the table by eye confirms it: when the anchor is $-$, the context is usually $+$ (7
out of 10); when the anchor is $+$, the context is usually $-$ (8 out of 10) — a real,
fairly strong *anti*-correlation, and $\phi\approx-0.50$ says exactly that. This is a number
MI could never have produced with a minus sign in front.

## 9. Why "phi," "Pearson $r$," and "Spearman $\rho$" are all the same thing here

These normally sound like three different statistics with different assumptions, but for
two yes/no variables they collapse into exactly the same number. Here's why, worked through
concretely:

- **Phi = Pearson $r$**: if you code $-$ as 0 and $+$ as 1 and run the textbook Pearson
  correlation formula ($r = \text{covariance}(Y,S)/(\text{std}(Y)\cdot\text{std}(S))$) on
  that 0/1-coded data, you get exactly the $\phi$ formula above — it's not a coincidence or
  an approximation, they're algebraically identical. (This is a standard, known result, not
  something specific to this analysis — "phi coefficient" is just the traditional name for
  "Pearson correlation of two binary variables.")
- **Pearson $r$ = Spearman $\rho$, for binary data specifically**: Spearman is defined as
  "Pearson correlation, but computed on the *ranks* of the data instead of the raw values."
  For a variable that only ever takes 2 distinct values, converting to ranks doesn't do
  anything interesting — it just relabels one group with one fixed number and the other
  group with another fixed number. Concretely: take 5 context signs $[-,-,+,+,+]$, i.e.
  $[0,0,1,1,1]$. Their ranks (using the standard "average rank among ties" rule) are
  $[1.5, 1.5, 4, 4, 4]$ — the two 0's tie for ranks 1 and 2 (average 1.5), the three 1's tie
  for ranks 3, 4, 5 (average 4). Notice $1.5 = 2.5\times 0 + 1.5$ and $4 = 2.5\times 1 + 1.5$
  — going from raw values to ranks was **just a fixed stretch-and-shift** ($2.5\times$, then
  $+1.5$), nothing more. And correlation is *provably unaffected* by stretching or shifting
  either variable by a fixed amount (that's true for any two variables, not just binary
  ones — it's the whole reason correlation is reported on a fixed $-1$ to $+1$ scale
  regardless of the original units). So ranking a binary variable changes nothing about its
  correlation with anything else — Spearman on binary data **is** Pearson on binary data
  **is** $\phi$.

**Practical upshot:** we don't need to call `scipy.stats.spearmanr` on millions of raw pairs
(slow, and needs the full pair list in memory) — the $\phi$ formula above, computed directly
from the same small 2×2 count table already being accumulated for MI, gives the exact same
answer instantly.

## 10. What the z-score means, and how it's calculated

A raw $\phi$ value on its own doesn't tell you whether it's a *real* effect or just noise —
$\phi=-0.015$ could be a genuine (if small) pattern, or it could be what 180,000 pure coin
flips look like by chance. The **z-score** answers exactly that: *"how many standard errors
away from zero is this $\phi$, if the true answer were exactly zero (no real
correlation)?"*

The formula used here is the standard large-sample approximation for a correlation
coefficient's standard error under the null hypothesis of *zero* correlation:

$$\text{SE}_{\text{null}} \approx \frac{1}{\sqrt{n-1}} \qquad z = \frac{\phi}{\text{SE}_{\text{null}}}$$

**Worked with our own numbers.** On bitcoin-alpha, distance 1 has $n=4{,}187{,}709$ pairs
and $\phi=+0.134$:

$$\text{SE}_{\text{null}} = \frac{1}{\sqrt{4{,}187{,}709 - 1}} \approx 0.000488 \qquad
  z = \frac{0.134}{0.000488} \approx 274$$

A $z$ of 274 means: *if there were truly zero correlation, seeing $\phi$ this far from 0
would basically never happen by chance* (converted to a p-value via the normal
distribution, it's indistinguishable from 0). Intuition for why $z$ can get so enormous:
with millions of pairs, the "noise floor" ($\text{SE}_{\text{null}}$) becomes tiny — the law
of large numbers means random wiggles average out more and more the more coin flips you
have — so even a fairly small true tilt ($\phi=0.13$, i.e. "13% more agreement than
disagreement") becomes overwhelmingly, unmistakably detectable.

At distance 6, $n$ has collapsed to only $180{,}672$: $\text{SE}_{\text{null}} \approx
0.00235$, and $\phi=-0.0153$ gives $z\approx -6.5$ — a much smaller z than distance 1's 274,
simply because there's 20x less data to average the noise away, but still enormously beyond
chance (a $z$ of 6.5 corresponds to a p-value of about $8\times10^{-11}$).

**One caveat, carried over from Part 1.** This $\text{SE}_{\text{null}}$ formula, like the
Miller-Madow correction, technically assumes every row is an independent draw — which §7
already showed isn't quite true at the far distances (the same edge gets reused across many
anchors). So the z-scores at distance 5-6 are probably a bit *overstated* by this formula.
The more trustworthy check is the actual **shuffle-signs experiment**: run the identical
pipeline on the identical graph with real signs replaced by randomly shuffled ones, and see
what z-scores come out when the true correlation is 0 *by construction*, no formula
required. That empirical null is not fooled by the independence assumption the same way,
since it's measured, not derived — and it's what actually confirmed the sign-flip finding
(real z's of hundreds vs. null z's that stay under ~2 throughout).

---

## TL;DR

| Question | Answer |
|---|---|
| What's the number? | Mutual information between an anchor edge's sign and a nearby edge's sign, as a fraction of the anchor's own entropy (NMI). |
| How is it computed? | Tally a 2×2 table of (anchor sign, context sign) pairs at a fixed distance; plug the observed frequencies straight into the textbook MI formula. |
| Why is that biased? | Plugging in *observed* frequencies instead of *true* probabilities always looks slightly more "correlated" than reality, purely from sampling noise — and this estimator can never go below the true value, only above it. |
| How much bias? | Miller-Madow: $\approx \frac{1}{2N\ln 2}$ bits for a 2×2 table — shrinks as sample size grows, confirmed to match our data almost exactly. |
| Does that explain our bump? | Only a small part of it — the real signal is still 90-3000x above that bias floor at the affected distances. |
| What's the rest of it? | A small number of specific edges get counted as "independent" observations dozens to over a thousand times each — the effective sample size is much smaller than the nominal one. Needs deduplication/cluster-aware resampling to fix properly, not yet implemented. |
| What did switching to correlation ($\phi$) show? | $\phi$ (= Pearson $r$ = Spearman $\rho$ for binary data) isn't stuck at $\ge0$ like MI, so it can reveal *direction*. On the pilot dataset it revealed a genuine sign flip: positive correlation at distance 1-2 (nearby edges share the anchor's sign), significantly *negative* from distance 3 onward (z as large as $-130$, vs. a shuffled-null control that stays under $|z|{\approx}2$ throughout) — something MI could never show since it can't go negative. |
