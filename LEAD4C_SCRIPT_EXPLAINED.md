# `lead4c_entropy_logit_regression.py`, explained from scratch

For you, not the professor. Assumes: basic Python, and "I've called
`sklearn.linear_model.LogisticRegression()` before" — nothing more. Every
section maps a script concept to the sklearn thing you already know.

## 1. The question, in one sentence

For each edge the model predicted, was it **right or wrong**, and does that
depend on **how mixed/contested the signs are** in the neighborhood around
that edge? If GNNs get *more* wrong as the neighborhood gets messier than the
walk model does, that's evidence for *why* the walk model wins overall.

## 2. What's one row of data?

One row = one edge the model made a prediction on. Columns:

- `correct` — 1 if the model's predicted sign matched the true sign, 0 if not. **This is `y`, the thing we predict.**
- 6 entropy columns (`src_out`, `src_in`, `tgt_out`, `tgt_in`, `twohop_in`, `twohop_out`) — each one is a number from 0 to 1 describing how "mixed" the signs are in some neighborhood around that edge. **These are `X`, the features.** Explained fully in §7.

So this is exactly the shape of data you'd hand to sklearn:

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X, y)   # X = the 6 entropy columns, y = correct/incorrect
```

The whole script is built around fitting *that*, just not with sklearn. Why
not sklearn — and what's different — is the rest of this doc.

## 3. Why not just `sklearn.linear_model.LogisticRegression`?

Try it and look at what you get back:

```python
model.fit(X, y)
print(model.coef_)        # the slopes — that's all you get
```

You'd get a number — say `-2.69` for `src_out`. That number on its own is
**not the problem**. sklearn computes basically the same point estimate
statsmodels does. The problem is what's missing around it.

**A coin-flip intuition.** Flip a coin 10 times, get 7 heads. The observed
proportion — 0.7 — looks biased toward heads. But is the coin *actually*
biased, or did a perfectly fair coin just happen to land that way this one
time? You cannot tell from "0.7" alone. What you need is some sense of **how
much that number would bounce around** if you repeated the experiment — flip
10 coins again, you might get 4 heads, or 8. Once you know how much it
naturally bounces around, you can ask "is 0.7 far enough from 0.5 (no bias)
that pure chance probably isn't the explanation?"

That bounce-around measure is called a **standard error**, and everything
built from it (confidence intervals, p-values — explained fully in §4) is
just different ways of expressing "how much do I trust this number." sklearn
computes the "0.7" (the coefficient) and stops there. `model.coef_` is a
single point estimate with **zero information about whether it could be
noise.** It's built for *prediction* — "give me the best guess" — not for
*inference* — "tell me if this relationship is real." (sklearn also quietly
shrinks coefficients toward 0 by default via regularization, which is
actively wrong for this use case — we want the unbiased coefficient, not one
nudged toward "safe.")

`statsmodels` runs the *same* logistic regression math, but reports it the
way a published paper does — point estimate **and** its uncertainty:

```python
import statsmodels.api as sm
X_with_intercept = sm.add_constant(X)        # see §6 — adds an intercept column
result = sm.Logit(y, X_with_intercept).fit() # fits it
print(result.summary())                      # coefficient, std err, z, p-value, CI — all of it
```

That `sm.Logit(y, X).fit()` line is the heart of the script — everything else
is bookkeeping around it (building `X` correctly, fixing the standard errors,
correcting for testing many things at once, looping over models/datasets).

## 4. Standard errors, confidence intervals, p-values, z — from scratch

This is the toolkit for answering "how much should I trust this number,"
referenced everywhere else in this doc. One time through, in order:

- **Standard error (SE)** — a number that measures how much your estimated
  coefficient would jitter if you re-ran the experiment on a different random
  sample of the same kind of data. Big SE = the estimate is shaky, could
  easily have come out very different. Small SE = the estimate is stable,
  more data/less noise pinned it down precisely.
- **z-statistic** — `beta / SE`. It answers "how many standard-error-units is
  this coefficient away from zero?" If `beta` is small relative to its own
  SE, it's not far from "this could just be zero plus noise." If `beta` is
  large *relative to its SE*, it's many standard errors away from zero — too
  far to plausibly be noise.
- **p-value** — converts the z-statistic into a probability: "if the true
  effect were actually exactly zero, what's the chance random sampling alone
  would produce a coefficient at least this extreme?" Small p-value (the
  convention: < 0.05) means that chance is low, so we call the effect
  "statistically significant" — probably not just noise. p-value is **not**
  "the probability the effect is real" (a common misreading) — it's "how
  surprising this data would be if the effect were zero."
- **Confidence interval (CI)** — a range built from `beta ± ~2 × SE` (roughly,
  for the usual 95% CI). Read it as "the range of true-effect values
  consistent with this data." If the whole interval is on one side of zero
  (e.g. `[-2.78, -2.61]`, never crossing 0), that's the same information as a
  small p-value, in a more intuitive "here's the plausible range" form. If
  the interval straddles zero (e.g. `[-0.09, +0.10]`), you can't rule out "no
  effect at all."

These four are really one idea wearing different outfits: **SE measures the
noise, and z/p-value/CI are three different ways of asking "is the
coefficient big enough to stand out above that noise."** None of this exists
in `model.coef_` — sklearn gives you the coefficient and throws away the
information needed to compute any of the above.

## 5. Logistic regression itself — quick refresher

Same model as sklearn's, same math, statsmodels just reports more about it.

```
P(correct) = sigmoid(intercept + β1·x1 + β2·x2 + ... )
```

- Each **β (beta)** is a coefficient — the same thing as `model.coef_[0][i]`
  in sklearn. Positive β means "as this feature goes up, the model is *more*
  likely to be correct." Negative β means the opposite.
- The **intercept** is `model.intercept_` in sklearn terms — the prediction
  when every feature is exactly 0.

**Odds ratio**, worked example. `exp(β)` is just a friendlier way to read β,
because "log-odds" isn't an intuitive unit. Take the real number for
`tgt_in`/GINEConv: `beta = -3.519`, `odds_ratio = exp(-3.519) ≈ 0.030`.

- "Odds" here means `P(correct) / P(wrong)` — e.g. odds of 9 means 9-to-1 in
  favor of being correct.
- An odds ratio of 0.030 means: for every +1 bit this entropy term increases
  (going from a totally uncontested neighborhood to a maximally contested
  50/50 one — the entire possible range, since entropy lives in [0,1]), **the
  odds of being correct get multiplied by 0.030** — i.e. they collapse to
  about 3% of what they were. That's a severe accuracy penalty for GINEConv
  specifically on contested-target edges.
- Compare to `walk_full` on the same term: `odds_ratio ≈ 0.145` (odds shrink
  to 14.5%, still bad, but a much smaller collapse than GINEConv's 3%). **The
  comparison between these two odds ratios — not either one alone — is the
  actual finding**: GINEConv is hurt more by this signal than the walk model.
- Odds ratio > 1 would mean the opposite direction — odds of being correct
  *improve* as that entropy rises. `twohop_out`/GINEConv has `odds_ratio ≈
  1.190` — a mild *improvement*, the rare case where more mixing helps a
  little instead of hurting.

None of this is different from what you already know — statsmodels just
*reports the uncertainty* on top of it, which is the whole reason it's used
here.

## 6. The intercept / "dummy variable" confusion, resolved

The model is fit once on **all 6 datasets pooled together**, not once per
dataset. But different datasets have different baseline difficulty (some
graphs are just easier to predict). So instead of one intercept, the script
gives **every dataset its own intercept**, while every dataset *shares* the
same entropy coefficients (β's). That's the whole idea — "do the entropy
effects look the same everywhere, with each dataset just allowed a different
starting point."

This is identical to **one-hot encoding** a categorical column, which you've
definitely done before:

```python
# the sklearn-world version of what's happening:
import pandas as pd
dataset_dummies = pd.get_dummies(dataset_column)   # one 0/1 column per dataset
X_full = pd.concat([X, dataset_dummies], axis=1)
```

The only subtlety: if you keep a global intercept column (`sm.add_constant`)
**and** include a dummy column for *every* dataset, the math breaks (the
columns become perfectly redundant — one is fully predictable from the
others, so there's no unique solution). The standard fix is to drop *one*
dummy column and let the kept-in intercept represent that dropped dataset's
baseline. **That's what made the earlier version confusing** — one dataset's
number was sitting alone labeled "intercept," and the other five were hidden
as `fe_<dataset>` "offset" rows you had to add back to make sense of.

The fix in the cleaned-up script: **drop the separate intercept column
instead**, and keep all 6 dummy columns. Now each dummy's own coefficient
*is* that dataset's intercept, directly, no arithmetic required, no
arbitrarily-chosen reference dataset. Mathematically it's the exact same fit
(same predictions, same entropy coefficients) — just reorganized so it reads
cleanly. (In the code: `_fit_one(..., add_const=False)` when the design
matrix already has a complete set of dataset dummies.)

## 7. The entropy features themselves

**Binary entropy** of a fraction `p` (e.g. "70% of this node's edges are
positive") is:

```
H(p) = -p·log2(p) - (1-p)·log2(1-p)
```

You don't need the formula memorized — just the shape: `H=0` when `p` is 0 or
1 (every edge has the *same* sign — totally consistent), and `H=1` when
`p=0.5` (a perfect 50/50 split — maximally contested/inconsistent). It's a
"how mixed is this" score, on a 0–1 scale.

For an edge `u → v`, the 6 entropy features ask "how mixed are the signs"
around 6 different neighborhoods:

| feature | mixed-ness of... |
|---|---|
| `src_out` | u's outgoing edges (is u a consistent or inconsistent *rater*?) |
| `src_in` | u's incoming edges (is u's own reputation contested?) |
| `tgt_out` | v's outgoing edges |
| `tgt_in` | v's incoming edges (is v's reputation — what we're about to add to — contested?) |
| `twohop_in` / `twohop_out` | same idea, but over 2-hop paths instead of single edges |

The actual finding: `src_out` and `tgt_in` turn out to matter a lot (with
*opposite* effects between model types), the other four barely matter at
all. That's read directly off the 6 β's — no feature is assumed important
ahead of time.

## 8. Why "cluster-robust" standard errors — the repeated-measurement problem

Recall from §4: the standard error (SE) measures how much your coefficient
would jitter on a fresh sample. The *default* formula for computing that SE
(in sklearn, statsmodels, anywhere) assumes every row of your data is an
**independent** observation — like 1000 different coin flips from 1000
different coins. That assumption is what lets the math say "more rows = more
confidence."

That assumption is **false** in this data. If node `u` has 50 outgoing edges,
**all 50 of those rows have the exact same `src_out` value** — it's a number
that belongs to `u`, copy-pasted onto every one of u's edges, not 50
independent measurements of 50 different things. It's the same mistake as
grading one student on 50 quiz questions and treating it statistically like
50 independent students — your "sample size" looks like 50, but the part
that's actually independent information is closer to 1 (one student's
general ability), so any uncertainty estimate computed as if `n=50` is
dramatically *overconfident*.

**What "two-way" clustering specifically does about it.** Every row has TWO
endpoints, `u` and `v`, and a row's `src_out` value is tied to `u` while its
`tgt_in` value is tied to `v` — so rows can be non-independent through
*either* endpoint. "Two-way cluster-robust" means: instead of pretending all
rows are independent, explicitly account for "rows sharing the same `u` can
be correlated" **and** "rows sharing the same `v` can be correlated," at the
same time, and widen the standard error accordingly. (The formula used —
Cameron–Gelbach–Miller 2011 — combines a `u`-only adjustment and a `v`-only
adjustment without double-counting rows that share *both* endpoints; you
don't need to derive it, just trust it's the standard tool for exactly this
"two different grouping structures at once" situation.)

Concretely, in this data: the *naive* (assume-everything-independent)
standard errors came out **1.1–2.2× too small**. That means some effects that
would have looked "statistically significant" under the naive (wrong)
calculation are *not* significant once you correctly account for the
repeated-`u`/repeated-`v` structure — the correction isn't pedantic, it
changes which conclusions you can actually stand behind.

## 9. Multiple testing — why some p-values get adjusted (BH-FDR)

If you test 100 totally random, meaningless features against an outcome at
the usual "p < 0.05 = significant" threshold, **about 5 of them will look
significant purely by chance.** That's the multiple-testing problem. This
script fits the regression separately per dataset, per model, per feature —
hundreds of coefficients in total — so *some* will look significant by luck
alone if you don't correct for it.

**Benjamini-Hochberg FDR correction** is the standard fix: it raises the bar
for "significant" in proportion to how many tests you ran, so the *expected
fraction of false positives* among everything you call significant stays
controlled (~5%), no matter how many tests you did. In the output, `p_fdr` is
this corrected p-value; `p` (uncorrected) is also kept for reference.

## 10. Pseudo-R² — "how much does this model actually explain"

Shows up in `fit_results.csv` (not in the professor file) as `pseudo_r2`.
Ordinary R² (the thing you'd get from `sklearn`'s `.score()` on a *linear*
regression) is "fraction of variance explained," 0 to 1. Logistic regression
doesn't have a variance to explain in the same way, so there's no single
agreed-on R² for it — **McFadden's pseudo-R²** (what's used here) is a
commonly-used stand-in: `1 − (fit model's log-likelihood) / (intercept-only
model's log-likelihood)`. Read it the same intuitive way as R² (closer to 1 =
explains more), but don't expect it to ever get *close* to 1 here — these
values sit around 0.10–0.22, which is normal/expected for "explain
correctness from one structural signal" rather than "explain it fully."
It's a sanity-check number, not part of the headline finding.

## 11. The three "specs" living in the same script

The script grew over a few rounds of "can we also see X" requests, so it now
fits **three versions** of essentially the same idea. They all reuse the same
fitting code (`_fit_one`), just with different `X` columns:

1. **`marginal3`** (the original/legacy version) — picks ONE direction for
   "source" and ONE for "target" out of 4 possible pairings, repeated for 3
   different two-hop definitions = 12 combinations, each fit as its own
   3-feature regression (`b_src`, `b_tgt`, `b_2hop`). Kept for backward
   compatibility / the appendix.
2. **`atomic`** (the headline) — instead of picking one direction, **all 6**
   directional entropy features go into ONE regression together. This is the
   actual answer to "which direction matters" — no arbitrary picking.
3. **`composite`** — collapses the 6 atomic features down to just 2 (`b_node`,
   `b_path`) by *pooling the underlying counts* (not by averaging the 6 β's,
   which would be statistically wrong — see §7, since two of the 6 push in
   *opposite* directions and averaging would hide that). A compact secondary
   view, explicitly not a replacement for the atomic one.

If this feels like the script tries three things instead of one — that's a
fair complaint, it does. `atomic` is the one to read first; `marginal3` and
`composite` are there because earlier rounds of this analysis asked for them.

## 12. The three stages: compute → fit → plot

```bash
python scripts/lead4c_entropy_logit_regression.py --mode compute  # build the data table (slow: walks the graph)
python scripts/lead4c_entropy_logit_regression.py --mode fit      # run all the regressions (the sm.Logit(...).fit() calls)
python scripts/lead4c_entropy_logit_regression.py --mode plot     # make figures + report.md from the fit results
python scripts/lead4c_entropy_logit_regression.py --mode all      # all three, in order
```

Each stage saves its output to disk and the next stage reads it back in —
this means you can re-run `--mode plot` a hundred times while tweaking a
figure without re-running the slow `compute` step.

## 13. Where to find things

- `outputs/lead4c_entropy_logit_regression/fit_results.csv` — every
  coefficient from every fit, one row each. Filter by `spec` (`atomic` is the
  one you want) and `term` (`ds_<dataset>` rows are the intercepts, the
  6 named entropy terms are the actual finding).
- `outputs/lead4c_entropy_logit_regression/report.md` — auto-generated
  version of the same thing as readable tables + a TL;DR.
- `outputs/lead4c_entropy_logit_regression/atomic_forest.png` — the one
  figure that shows the whole finding at a glance.
- `lead4_coefficients.md` / `.csv` (repo root) — the short version made
  specifically to hand to your professor.

## 14. Cheat-sheet: sklearn word → statsmodels word

| sklearn | statsmodels | meaning |
|---|---|---|
| `model.fit(X, y)` | `sm.Logit(y, X).fit()` | same thing — note y, X are swapped in argument order! |
| `model.coef_` | `result.params[1:]` | the β's (one per feature) |
| `model.intercept_` | `result.params[0]` (if you called `sm.add_constant`) | the intercept |
| *(not available)* | `result.bse` | standard error of each coefficient |
| *(not available)* | `result.pvalues` | p-value of each coefficient |
| *(not available)* | `result.conf_int()` | 95% confidence interval |
| manual `pd.get_dummies` | same — statsmodels doesn't auto-encode categories from raw arrays | one-hot encoding |
| `StandardScaler` then `.coef_` | `beta * x.std()` (computed directly here, no refit needed) | standardized coefficient |
| *(not available at all)* | `cluster_robust_2way(...)` (custom, in this script) | §8 — corrects SE for repeated `u`/`v` |
| *(not available at all)* | `_bh_fdr(...)` (custom, in this script) | §9 — corrects p-values for testing many things at once |
| `model.score(X, y)` (accuracy) | `result.prsquared` / `pseudo_r2` (different thing!) | §10 — "how much variation does this explain," not accuracy |
| `np.exp(model.coef_)` (you *can* compute this yourself) | `result.params` then `np.exp(...)`, or the `odds_ratio` column | §5 — odds ratio, same formula either way |
