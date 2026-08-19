# The paper's "Pinsker-type relaxation" is exactly Pinsker's inequality

**Question this resolves:** line 374 of `WSDM_format_revised.tex` states
$\Hh_b(p)\le 1-\tfrac{2}{\ln 2}(\tfrac12-p)^2$ and calls it a "Pinsker-type relaxation," flagged
`XXX REF TO PINSKER XXX` because nobody had checked whether that name is actually earned. It is:
the bound below is derived by substituting two specific Bernoulli distributions into the
classical Pinsker's inequality, with no approximation step anywhere. Every line is an equality
or a direct substitution; the only inequality used is Pinsker's itself.

## 1. Classical Pinsker's inequality

For any two probability distributions $P,Q$ on the same space, let
$\delta(P,Q)=\sup_A|P(A)-Q(A)|$ be the total variation distance. Pinsker's inequality (natural
log, nats) states:

$$\delta(P,Q) \le \sqrt{\tfrac{1}{2}D_{KL}(P\|Q)} \qquad\Longleftrightarrow\qquad D_{KL}(P\|Q) \ge 2\,\delta(P,Q)^2 \tag{1}$$

This is **Lemma 11.6.1** in:

> Thomas M. Cover and Joy A. Thomas, *Elements of Information Theory*, 2nd edition,
> Wiley-Interscience, 2006. Chapter 11 ("Information Theory and Statistics"), Section 11.6
> ("The Conditional Limit Theorem"), where it appears as a supporting lemma used in the proof of
> the conditional limit theorem (Sanov-type large-deviation argument) — not as a standalone
> named result, which is why it is easy to miss on a skim. Approximate location pp. 370–372;
> the section/lemma number is cross-confirmed by two independent sources, the exact page range
> is a secondary estimate (see "Sourcing note" at the end of this file).

The tex cites `yeung2008information` (Raymond W. Yeung, *Information Theory and Network Coding*,
Springer, 2008) at this bound instead of Cover & Thomas — a different standard textbook that
also carries Pinsker's inequality, added by the user directly. `cover2006elements` (Cover &
Thomas) stays in the tex only at its other use, Fano's inequality (line 109) — the two
citations were previously conflated (both pointed at `cover2006elements`) and have now been
separated to the right textbook for each inequality.

Equivalent forms seen in the literature, all algebraically identical to (1):
$D_{KL}(P\|Q)\ge\tfrac12\|P-Q\|_1^2$ (using the $\ell_1$ distance $\|P-Q\|_1=\sum_x|P(x)-Q(x)|=2\delta(P,Q)$,
so $\tfrac12\|P-Q\|_1^2=\tfrac12(2\delta)^2=2\delta^2$, matching (1) exactly).

## 2. Specialize to Bernoulli($p$) vs. Bernoulli($\tfrac12$)

Let $P=\mathrm{Bernoulli}(p)$ (i.e. $P(1)=p,\,P(0)=1-p$) and $Q=\mathrm{Bernoulli}(\tfrac12)$ (the
maximum-entropy binary distribution).

**Total variation distance.**

$$\delta(P,Q) = \tfrac12\sum_{x\in\{0,1\}}|P(x)-Q(x)| = \tfrac12\Big(\big|p-\tfrac12\big| + \big|(1-p)-\tfrac12\big|\Big) = \tfrac12\cdot 2\big|p-\tfrac12\big| = \big|p-\tfrac12\big| \tag{2}$$

(the two terms are equal since $|(1-p)-\tfrac12| = |\tfrac12-p| = |p-\tfrac12|$).

**KL divergence, in nats.**

$$D_{KL}(P\|Q) = p\ln\frac{p}{1/2} + (1-p)\ln\frac{1-p}{1/2} = p\big[\ln2+\ln p\big] + (1-p)\big[\ln2+\ln(1-p)\big]$$
$$= \ln2\underbrace{\big[p+(1-p)\big]}_{=1} + \big[p\ln p+(1-p)\ln(1-p)\big] = \ln2 - \Hh_b^{\text{nats}}(p) \tag{3}$$

where $\Hh_b^{\text{nats}}(p):=-p\ln p-(1-p)\ln(1-p)$ is binary entropy measured in nats.

**Convert to bits.** The paper's $\Hh_b$ is base-2 (confirmed elsewhere in the tex:
"$\Hh_\outdeg=\Hh_b(0.5)=1$", i.e. 1 *bit* at maximal disagreement — this only holds for
$\log_2$, not $\ln$). Since $\Hh_b^{\text{nats}}(p)=\ln2\cdot\Hh_b(p)$ (change-of-base), (3) becomes:

$$D_{KL}(P\|Q) = \ln2 - \ln2\cdot\Hh_b(p) = \ln2\,\big(1-\Hh_b(p)\big) \quad\text{nats} \tag{4}$$

## 3. Substitute into Pinsker's inequality

Plug (2) and (4) into (1):

$$\ln2\,\big(1-\Hh_b(p)\big) \;\ge\; 2\Big(p-\tfrac12\Big)^2$$

Divide both sides by $\ln2$ (positive constant, inequality direction unchanged):

$$1-\Hh_b(p) \;\ge\; \frac{2}{\ln2}\Big(p-\tfrac12\Big)^2$$

Rearrange:

$$\boxed{\;\Hh_b(p) \;\le\; 1-\frac{2}{\ln2}\Big(\tfrac12-p\Big)^2\;} \tag{5}$$

using $(p-\tfrac12)^2=(\tfrac12-p)^2$. **This is exactly the tex's line 374 bound, term for term,
including the constant $2/\ln2$.** $\blacksquare$

## 4. Conclusion

The paper's bound is not "inspired by" or "in the style of" Pinsker's inequality — it **is**
Pinsker's inequality, specialized to $P=\mathrm{Bernoulli}(p)$ against the uniform
$Q=\mathrm{Bernoulli}(\tfrac12)$, with the nats-to-bits conversion made explicit. Every step above
is an equality or a direct substitution; inequality (1) is the only place looseness enters, and
that's the same looseness the paper is deliberately trading for a closed form (this is exactly
why the tex calls it a *relaxation* of the inverse-Fano bound rather than a tight
characterization).

**Status: done.** The `XXX REF TO PINSKER XXX` marker was deleted from line 374, and the bound's
citation is now `\citep{yeung2008information}` (added by the user directly, Section 5 above
verifies it states the same inequality). `cover2006elements` (Cover & Thomas) remains cited
elsewhere in the tex for Fano's inequality only (line 109), which is what it was originally
added for — the two citations were briefly conflated (both pointed at `cover2006elements`) and
have since been separated to the right textbook for each inequality.

## 5. Cross-check against the cited source's own stated form

`yeung2008information` (Yeung, *Information Theory and Network Coding*, 2008) states Pinsker's
inequality as:

$$D(P\|Q) = \frac{D_{KL}(P\|Q)}{\ln2} \;\ge\; \frac{1}{2\ln2}\,V^2(p,q)$$

i.e. $D(P\|Q)$ is $D_{KL}$ already converted to bits, and $V(p,q)$ is the **variational
distance** $V(p,q):=\sum_x|p(x)-q(x)|$ — the discrete $\ell_1$ distance between the two
distributions, a standard alternate convention to the sup-based total variation distance
$\delta(P,Q)$ used in Section 1 above. For any two distributions on a finite space,
$V(p,q)=2\,\delta(P,Q)$ (a standard identity), so substituting $V=2\delta$:

$$\frac{1}{2\ln2}V^2 = \frac{1}{2\ln2}(2\delta)^2 = \frac{4}{2\ln2}\delta^2 = \frac{2}{\ln2}\delta^2$$

which is exactly $(1)$ converted to bits (divide both sides of $D_{KL}\ge2\delta^2$ by $\ln2$).
Confirmed numerically at $p=0.1,0.3,0.5,0.7,0.9$ — both forms give identical RHS values at every
point. So Yeung's stated inequality is the same fact as the Cover & Thomas form in Section 1,
just written in bits with the $\ell_1$/variational-distance convention instead of nats with the
sup-based total-variation convention; the rest of the derivation (Sections 2–3) is unaffected
and the paper's bound (5) still follows from it exactly as shown.

## Sourcing note

The lemma number (11.6.1), section number (11.6), section title ("The Conditional Limit
Theorem"), and chapter (11, "Information Theory and Statistics") are cross-confirmed by two
independent web sources. The page range (~370–372) came from one source's synthesis and could
not be independently verified against the actual book text (the full-book PDFs found were too
large to fetch/search directly in this environment) — treat the page number as a reasonable
estimate, not a verified fact, if it's ever needed for a precise in-text page citation. The
lemma/section identification itself is solid (matches independently on two different searches),
and the mathematical derivation in this file is self-contained and checkable regardless of the
exact page number.
