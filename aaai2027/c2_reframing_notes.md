# Moving from general $c$ to $c=2$: the problem and how to reframe

## The core problem

The paper's information bound rests on Fano's inequality:

$$H(Y\mid Z) \le H_b(P_e) + P_e\log(c-1)$$

The textbook move is to weaken this to a clean linear form by bounding $H_b(P_e)\le 1$:

$$P_e \ge \frac{H(Y\mid Z)-1}{\log c}$$

This is the version the paper used, stated "for general $c$." It is fine for $c$ large. But **at $c=2$, $\log c = 1$, and a binary label always has $H(Y\mid Z)\le 1$ bit** — so the numerator is never positive and the bound collapses to $P_e \ge \text{(something} \le 0)$. It says nothing. And $c=2$ is exactly sign prediction, the paper's actual setting. The headline theoretical result was vacuous in the one case the paper is about.

This isn't a framing choice, it's a derivation choosing to include the $H_b(P_e)\le 1$ slack step *at the moment where that slack costs everything*. That slack step exists to handle large $c$; at $c=2$ it isn't needed and shouldn't be taken.

## The fix

At $c=2$, the term $P_e\log(c-1) = P_e\log 1 = 0$ vanishes identically — not approximately, exactly. So Fano's inequality at $c=2$, with no weakening at all, is:

$$H(Y\mid Z) \le H_b(P_e)$$

Since $H_b$ is increasing on $[0,\tfrac12]$, this inverts cleanly to:

$$P_e \ge H_b^{-1}\big(H(Y\mid Z)\big)$$

This is *tighter* than the general-$c$ form, not a special case of it — the general form threw away information ($H_b(P_e)\le1$) that isn't necessary here. It's strictly positive whenever $H(Y\mid Z)>0$, which is the property the whole paper needs.

## How every downstream piece should be rewritten

Anywhere the paper says "$P_e \ge (\text{entropy} - 1)/\log c$", replace with "$P_e \ge H_b^{-1}(\text{entropy})$" for the $c=2$/main-text version. Concretely, this touches:

- **Problem Setting**: define $H_b$ explicitly, derive both the general Fano corollary and the $c=2$ specialization, and *say out loud* that the linear form is vacuous at $c=2$ — don't just quietly swap the bound, because a reviewer who knows Fano will otherwise wonder why the paper needed a nonstandard step.
- **Proposition 1 (bottleneck)** and its **proof sketch**.
- **Corollary (single-endpoint reading)**.
- **Proposition 2 (capacity form)** and its **proof sketch**.
- **Appendix proofs** for both propositions, including the Jensen step needed to pass $H_b^{-1}$ (now nonlinear) through an average over color pairs — the old linear form didn't need Jensen because it was already linear in the conditioning entropy.

## General framing notes for the rewrite

1. **Lead with $c=2$, not "for general $c$ as well."** The paper's motivating case, datasets, and method are all binary sign prediction. Treating $c=2$ as a footnote to a general-$c$ theory, when the general-$c$ theory is the one that breaks, is backwards. State $c=2$ as the primary setting up front.

2. **Don't delete the general-$c$ case — move it.** The instinct to keep "the propositions hold for general $c$" is right, it's just misplaced as the main-text framing. Push the full general-$c$ derivation (the original linear corollary, still valid and non-vacuous for $c>2$) into an appendix ("General $c$"), and have every main-text proposition point to it. This preserves the information bound's generality without letting it dilute the main argument.

3. **Flag the vacuity explicitly, don't paper over it.** A knowledgeable reader (or reviewer) who knows Fano's inequality will recognize the standard linear corollary and may reflexively assume it's what's being used. Explicitly noting "this standard form is vacuous at $c=2$; here is the tight two-class form instead" pre-empts that reviewer objection and makes the choice look deliberate rather than sloppy.

4. **Watch for other silent uses of $\log c$.** Any place that references "$\log c$" downstream (capacity bound, corollaries, appendix) needs the same swap — it's easy to fix the main proposition and miss a restated version of it elsewhere (this happened in at least four places in the current draft: Prop 1, its corollary, Prop 2, and the appendix proofs).

5. **Double-check the one new piece of math introduced by the fix**: the claim that $H_b^{-1}$ is convex on $[0,\log 2]$ (needed for the Jensen step in the bottleneck appendix proof, since averaging over color pairs is no longer linear once the bound is $H_b^{-1}$ instead of a linear function of entropy). This follows from $H_b$ being concave and increasing on $[0,\tfrac12]$ — inverse of an increasing concave function is convex — but it's new to this draft and worth an independent sanity check before submission, since the whole per-color-pair averaging step leans on it.

6. **Terminology consistency**: once $c=2$ is primary, consider whether "$c$ classes" language throughout (Problem Setting, intro) should shift to "positive/negative" or "binary label" phrasing in the main text, reserving "$c$ classes" for the general-$c$ appendix. Right now the notation is defined generally and then immediately specialized, which is fine, but skim the rest of the paper for other spots that still assume/imply general $c$ informally (e.g., any prose describing "each class" or "majority label in each color class") and make sure they still read naturally under a binary reading.
