# Rigorous, no-shortcuts verification of every mathematical claim in the paper

**Scope and standard.** This goes through every mathematical object in
`WSDM_format_revised.tex` — the binary-entropy preliminaries, Fano's inequality, the
inverse-Fano bound, the monotonicity/convexity Lemma, the Pinsker relaxation, Proposition 1
(bottleneck), the Corollary, Proposition 2 (capacity), and the general-$c$ extension — and
proves each step in full, with no step skipped and no fact taken on faith beyond genuinely
standard external theorems (Fano's inequality's own proof is included; basic real-analysis
facts like IVT are not re-derived). The paper is allowed to compress this into a page-limited
proof sketch; this document is not. Beyond correctness, it also asks the second question the
user raised: **does each bound actually say something, or could it be trivially/vacuously
satisfied?** That's a different question from "is the algebra right," and gets its own section
(Section 10) with real numbers pulled from `DATASET_STATS.md`.

**Headline result:** every proof step in the paper is correct (cross-checked against an earlier,
narrower pass that also found no bugs). The new finding from this deeper pass is in Section 10:
**Proposition 2's capacity bound, taken completely literally (raw float-bit capacity), is not
uniformly vacuous but is much closer to the edge of vacuousness than the paper's phrasing
suggests** — it only binds at all for the single most extreme hub vertex in the most skewed
dataset (Epinions), and only at the lower end of the paper's own embedding-width range. This
isn't a bug — the paper's own hedge ("$N$ is the covering number... at the decision margin")
already anticipates this — but it's worth knowing precisely how narrow the literal-reading case
is, since the bound's real force rests entirely on that hedge, not on the literal bit count.

---

## 1. Preliminaries: the binary entropy function

Define $\mathrm{H}_b:[0,1]\to[0,1]$ by $\mathrm{H}_b(p) = -p\log_2 p - (1-p)\log_2(1-p)$, with $0\log_2 0:=0$
(the limit, standard convention).

**Claim 1a — symmetry.** $\mathrm{H}_b(p)=\mathrm{H}_b(1-p)$. Immediate by inspection (swap $p\leftrightarrow
1-p$ in the definition; the two terms swap roles).

**Claim 1b — endpoints.** $\mathrm{H}_b(0)=\mathrm{H}_b(1)=0$ (both terms vanish, using $0\log_2 0:=0$), and
$\mathrm{H}_b(\tfrac12) = -\tfrac12\log_2\tfrac12-\tfrac12\log_2\tfrac12 = \tfrac12+\tfrac12=1$.

**Claim 1c — strict concavity on $(0,1)$, derived not cited.** Write $\mathrm{H}_b(p) =
-\frac1{\ln2}\big[p\ln p+(1-p)\ln(1-p)\big]$ (change of base, $\log_2x=\ln x/\ln2$). Then
$$\frac{d}{dp}\big[p\ln p\big]=\ln p+1,\qquad \frac{d}{dp}\big[(1-p)\ln(1-p)\big]=-\ln(1-p)-1$$
so $\mathrm{H}_b'(p) = -\tfrac1{\ln2}\big[\ln p+1-\ln(1-p)-1\big] = \tfrac1{\ln2}\ln\tfrac{1-p}p$, and
$$\mathrm{H}_b''(p) = \tfrac1{\ln2}\cdot\frac{d}{dp}\big[\ln(1-p)-\ln p\big] = \tfrac1{\ln2}\Big[\tfrac{-1}{1-p}-\tfrac1p\Big] = -\frac{1}{p(1-p)\ln2}.$$
Since $p(1-p)>0$ on $(0,1)$, $\mathrm{H}_b''(p)<0$ everywhere there: **strictly concave on $(0,1)$**,
hence (continuity extends strict concavity to the closure) on $[0,1]$ and in particular on the
sub-interval $[0,\tfrac12]$ used throughout.

**Claim 1d — strictly increasing on $[0,\tfrac12]$.** From $\mathrm{H}_b'(p)=\tfrac1{\ln2}\ln\tfrac{1-p}p$:
for $p\in(0,\tfrac12)$, $1-p>p>0$ so $\tfrac{1-p}p>1$ so $\ln\tfrac{1-p}p>0$, i.e. $\mathrm{H}_b'(p)>0$.
A function with positive derivative on an open interval and continuous on its closure is
strictly increasing on the closure: **strictly increasing on $[0,\tfrac12]$**.

**Claim 1e — bijection $[0,\tfrac12]\to[0,1]$.** Continuous (elementary function, no
singularities on the closed interval) + strictly increasing (1d) + endpoint values $0,1$ (1b)
$\Rightarrow$ by the intermediate value theorem the image is exactly $[0,1]$ (surjective), and
strict monotonicity gives injectivity. **Bijection**, as the paper states (line 115).

This is the paper's own foundational claim, now derived rather than asserted.

---

## 2. Fano's inequality (cited external result, proved here for completeness)

**Statement.** $Y$ takes values in a finite set of size $c$; $\hat Y=g(Z)$ is any estimator of
$Y$ from $Z$; $P_e:=\Pr(\hat Y\ne Y)$. Then
$$\mathrm{H}(Y\mid Z)\ \le\ \mathrm{H}_b(P_e) + P_e\log_2(c-1). \tag{Fano}$$

**Proof.** Let $E:=\mathbb 1[\hat Y\ne Y]\in\{0,1\}$, a Bernoulli$(P_e)$ random variable, and note
$E$ is a deterministic function of the pair $(Y,\hat Y)$.

*Step 1 — expand $\mathrm{H}(E,Y\mid\hat Y)$ two ways.*
$$\mathrm{H}(E,Y\mid\hat Y) = \mathrm{H}(Y\mid\hat Y) + \underbrace{\mathrm{H}(E\mid Y,\hat Y)}_{=0\text{, since }E=g'(Y,\hat Y)} = \mathrm{H}(Y\mid\hat Y). \tag{a}$$
$$\mathrm{H}(E,Y\mid\hat Y) = \mathrm{H}(E\mid\hat Y) + \mathrm{H}(Y\mid E,\hat Y) \ \le\ \mathrm{H}(E) + \mathrm{H}(Y\mid E,\hat Y), \tag{b}$$
the inequality in (b) because conditioning cannot increase entropy: $\mathrm{H}(E\mid\hat Y)\le\mathrm{H}(E)$.

*Step 2 — bound each term of (b).* $\mathrm{H}(E)=\mathrm{H}_b(P_e)$ directly (entropy of a Bernoulli$(P_e)$
variable is exactly $\mathrm{H}_b(P_e)$ by definition of $\mathrm{H}_b$). For $\mathrm{H}(Y\mid E,\hat Y)$, split on
the value of $E$:
$$\mathrm{H}(Y\mid E,\hat Y) = \Pr(E{=}0)\,\mathrm{H}(Y\mid E{=}0,\hat Y) + \Pr(E{=}1)\,\mathrm{H}(Y\mid E{=}1,\hat Y).$$
When $E=0$, $Y=\hat Y$ exactly (no residual randomness), so $\mathrm{H}(Y\mid E{=}0,\hat Y)=0$. When
$E=1$, $Y$ ranges only over the $c-1$ values $\ne\hat Y$, so its entropy is at most $\log_2(c-1)$
(uniform distribution over $c-1$ outcomes maximizes entropy at exactly that value — a standard
fact: entropy of any distribution on a finite set of size $m$ is $\le\log_2 m$, with equality iff
uniform). Hence $\mathrm{H}(Y\mid E,\hat Y)\le\Pr(E{=}1)\log_2(c-1) = P_e\log_2(c-1)$.

*Step 3 — combine (a) and (b).* $\mathrm{H}(Y\mid\hat Y) \le \mathrm{H}_b(P_e)+P_e\log_2(c-1)$.

*Step 4 — replace $\hat Y$ by $Z$.* $\hat Y=g(Z)$ is a deterministic function of $Z$, so $Z$
determines $\hat Y$ (but generally not vice versa) — $Z$ is at least as informative, hence
conditioning on $Z$ cannot give more residual entropy than conditioning on $\hat Y$:
$\mathrm{H}(Y\mid Z)\le\mathrm{H}(Y\mid\hat Y)$. Combined with Step 3: $\mathrm{H}(Y\mid Z)\le\mathrm{H}_b(P_e)+P_e\log_2(c-1)$.
$\blacksquare$

This matches the tex's eq:fano exactly.

---

## 3. The inverse-Fano bound (eq:invfano) — fully rigorous, both regimes of $P_e$

At $c=2$: $\log_2(c-1)=\log_2 1=0$, so (Fano) reads $\mathrm{H}(Y\mid Z)\le\mathrm{H}_b(P_e)$, for **any**
$P_e\in[0,1]$ (no restriction yet).

**Claim: $P_e\ge\mathrm{H}_b^{-1}(\mathrm{H}(Y\mid Z))$ for every predictor, unconditionally.**

*Case $P_e\le\tfrac12$.* Both $\mathrm{H}(Y\mid Z)$ and $\mathrm{H}_b(P_e)$ lie in $[0,1]$ (entropy of a binary
variable is bounded by 1 bit), and on this range $\mathrm{H}_b^{-1}$ is well-defined and strictly
increasing (Section 1). Applying the strictly increasing $\mathrm{H}_b^{-1}$ to both sides of
$\mathrm{H}(Y\mid Z)\le\mathrm{H}_b(P_e)$ preserves the inequality (order-preservation is exactly what "strictly
increasing" gives): $\mathrm{H}_b^{-1}(\mathrm{H}(Y\mid Z)) \le \mathrm{H}_b^{-1}(\mathrm{H}_b(P_e)) = P_e$. Done.

*Case $P_e>\tfrac12$.* Here $\mathrm{H}_b^{-1}(\mathrm{H}(Y\mid Z))$ is still a value in $[0,\tfrac12]$ (that's
its range, by construction — $\mathrm{H}_b^{-1}:[0,1]\to[0,\tfrac12]$), so $\mathrm{H}_b^{-1}(\mathrm{H}(Y\mid
Z))\le\tfrac12<P_e$. The claimed inequality $P_e\ge\mathrm{H}_b^{-1}(\mathrm{H}(Y\mid Z))$ holds **for a reason
that has nothing to do with Fano at all** — the right side simply can never exceed $\tfrac12$,
and $P_e$ already exceeds $\tfrac12$ by assumption.

Both cases give $P_e\ge\mathrm{H}_b^{-1}(\mathrm{H}(Y\mid Z))$, unconditionally on $P_e$. $\blacksquare$

**This confirms the paper's eq:invfano is correct**, but also pins down exactly what's doing the
work: it is *not* actually necessary to invoke "flip the predictor's output" reasoning at all —
that framing (paper line 115) obscures a cleaner fact, that the bound is automatic whenever
$P_e>\tfrac12$ simply because the bound's own range is capped at $\tfrac12$. See Section 11 for
suggested rewordings of this exact sentence.

---

## 4. Lemma: monotonicity and convexity of $\mathrm{H}_b^{-1}$

The paper's own proof of this (lines 365–372) is already fully rigorous — reproduced here with
one extra layer of explicit justification per step, to keep this document self-contained.

**Monotonicity.** Suppose toward contradiction $\mathrm{H}_b^{-1}$ is *not* strictly increasing: there
exist $x_1<x_2$ in $[0,1]$ with $\mathrm{H}_b^{-1}(x_1)\ge\mathrm{H}_b^{-1}(x_2)$. Apply $\mathrm{H}_b$ (strictly
increasing on $[0,\tfrac12]$, Section 1) to both sides — a strictly increasing function preserves
$\ge$: $\mathrm{H}_b(\mathrm{H}_b^{-1}(x_1)) \ge \mathrm{H}_b(\mathrm{H}_b^{-1}(x_2))$, i.e. $x_1\ge x_2$ (using
$\mathrm{H}_b(\mathrm{H}_b^{-1}(x))=x$, the defining property of an inverse). This contradicts $x_1<x_2$.
Hence no such pair exists: $\mathrm{H}_b^{-1}$ is strictly increasing. $\blacksquare$

**Convexity.** Fix $x,y\in[0,1]$, $x\ne y$, $\alpha,\beta>0$, $\alpha+\beta=1$. Let $p=\mathrm{H}_b^{-1}(x)$,
$q=\mathrm{H}_b^{-1}(y)$ (both in $[0,\tfrac12]$, $p\ne q$ since $\mathrm{H}_b^{-1}$ is injective). By strict
concavity of $\mathrm{H}_b$ on $[0,\tfrac12]$ (Section 1c):
$$\mathrm{H}_b(\alpha p+\beta q) > \alpha\mathrm{H}_b(p)+\beta\mathrm{H}_b(q) = \alpha x+\beta y.$$
Apply the strictly increasing $\mathrm{H}_b^{-1}$ to both sides (order-preserving):
$$\mathrm{H}_b^{-1}\big(\mathrm{H}_b(\alpha p+\beta q)\big) > \mathrm{H}_b^{-1}(\alpha x+\beta y)$$
$$\alpha p+\beta q > \mathrm{H}_b^{-1}(\alpha x+\beta y)$$
$$\alpha\,\mathrm{H}_b^{-1}(x)+\beta\,\mathrm{H}_b^{-1}(y) > \mathrm{H}_b^{-1}(\alpha x+\beta y).$$
This is exactly the definition of strict convexity of $\mathrm{H}_b^{-1}$. $\blacksquare$ Neither proof
used differentiability of $\mathrm{H}_b^{-1}$ itself — correct, since inverse functions of the kind used
here need not be differentiable at the boundary even when the original function is.

---

## 5. Pinsker-type relaxation

Fully re-derived (both the classical form and the cited-source's own stated form, cross-checked
against each other and confirmed numerically) in the companion file
`PINSKER_BOUND_DERIVATION.md` — not repeated here to avoid duplication. Summary: the bound
$\mathrm{H}_b(p)\le1-\tfrac2{\ln2}(\tfrac12-p)^2$ is exactly Pinsker's inequality specialized to
Bernoulli$(p)$ vs. Bernoulli$(\tfrac12)$, and substituting this into the chain from Section 3
(the $P_e\le\tfrac12$ case; combine $\mathrm{H}(Y\mid Z)\le\mathrm{H}_b(P_e)\le1-\tfrac2{\ln2}(\tfrac12-P_e)^2$
and solve for $P_e$) gives exactly eq:pinsker, $P_e\ge\tfrac12-\sqrt{\tfrac{\ln2}2(1-\mathrm{H}(Y\mid Z))}$.

---

## 6. Proposition 1 (Endpoint bottleneck) — full proof, every step named

**Setup.** A $T$-round message-passing GNN assigns each vertex a color $\chi_T(\cdot)$ that
refines to at most the WL-stable coloring $\chi(\cdot)$ (cited: Xu et al. 2018, Morris et al.
2019 — this equivalence is itself a substantial theorem and is correctly treated here as an
external, cited fact rather than re-derived, matching the paper's own scope). The vertex
representation is $h_u=\psi(\chi_T(u))$ for some function $\psi$, so any read-out
$\hat y_{uv}=g(h_u,h_v) = g(\psi(\chi_T(u)),\psi(\chi_T(v))) =: \tilde g(\chi_T(u),\chi_T(v))$ is a
function of the color pair alone — i.e. $\hat Y$ is $\sigma(\chi_T(U),\chi_T(V))$-measurable.

**Step 1 — the optimal predictor given the color pair.** Among all predictors that are
functions of $(\chi_T(U),\chi_T(V))$ alone, the one minimizing $\Pr(\hat Y\ne Y)$ is the MAP rule
$\tilde g^\star(a,b) = \arg\max_y \Pr(Y=y\mid \chi_T(U)=a,\chi_T(V)=b)$ — a standard fact
(Bayes-optimality of MAP under 0-1 loss: for *any* fixed conditioning value, the predictor that
minimizes error probability is the one that outputs the most likely class given that
conditioning value; picking anything else strictly increases the chance of missing the
highest-probability outcome).

**Step 2 — conditional Fano.** Fix a color pair $(a,b)$ with $\Pr(\chi_T(U){=}a,\chi_T(V){=}b)>0$.
Restricted to this conditioning event, $Y$ is binary ($c=2$, sign prediction) and any predictor
measurable in this event is a function of nothing further (it's constant, since we've
conditioned on the exact color pair) — so the derivation of Section 3 applies verbatim with $Z$
replaced by the *event* $\{\chi_T(U)=a,\chi_T(V)=b\}$:
$$P_e(a,b) := \Pr(\hat Y\ne Y\mid \chi_T(U){=}a,\chi_T(V){=}b) \ \ge\ \mathrm{H}_b^{-1}\big(\mathrm{H}(Y\mid\chi_T(U){=}a,\chi_T(V){=}b)\big).$$

**Step 3 — average over color pairs.** By the law of total probability,
$$P_e = \sum_{a,b}\Pr(\chi_T(U){=}a,\chi_T(V){=}b)\,P_e(a,b) = \mathbb E_{(a,b)}[P_e(a,b)].$$
Since expectation is monotone (a pointwise $\ge$ inequality survives averaging):
$$P_e \ \ge\ \mathbb E_{(a,b)}\big[\mathrm{H}_b^{-1}(\mathrm{H}(Y\mid a,b))\big].$$

**Step 4 — Jensen.** $\mathrm{H}_b^{-1}$ is convex (Section 4). Jensen's inequality for a convex function
$\phi$ states $\mathbb E[\phi(X)]\ge\phi(\mathbb E[X])$ (the direction that matches "convex
functions lie below their chords, so the average of the function is at least the function of the
average" — standard, and note this is the *opposite* direction from the more commonly quoted
Jensen form for concave functions, so it's worth double-checking: yes, for convex $\phi$,
$\phi(\mathbb E[X])\le\mathbb E[\phi(X)]$ is the standard statement, e.g. Cover & Thomas Thm
2.6.2). With $\phi=\mathrm{H}_b^{-1}$, $X=\mathrm{H}(Y\mid a,b)$ (a random variable via the randomness in
$(a,b)$):
$$\mathbb E_{(a,b)}\big[\mathrm{H}_b^{-1}(\mathrm{H}(Y\mid a,b))\big] \ \ge\ \mathrm{H}_b^{-1}\big(\mathbb E_{(a,b)}[\mathrm{H}(Y\mid a,b)]\big).$$

**Step 5 — identify the averaged entropy.** By the definition of conditional entropy (the "chain
rule" the paper refers to is really just the definition unwound): $\mathbb E_{(a,b)}[\mathrm{H}(Y\mid
a,b)] = \mathrm{H}(Y\mid\chi_T(U),\chi_T(V))$.

**Combining Steps 3–5:** $P_e \ge \mathrm{H}_b^{-1}\big(\mathrm{H}(Y\mid\chi_T(U),\chi_T(V))\big)$.

**Step 6 — replace $\chi_T$ by the stable coloring $\chi$.** Since $\chi_T$ refines to at most
$\chi$ (i.e. $\chi$ is at least as fine a partition as $\chi_T$ for every finite $T$), $\chi$ is
at least as informative: conditioning on the finer $\chi$ can only reduce or preserve entropy
compared to the coarser $\chi_T$ — a standard monotonicity fact (more refined conditioning info
$\Rightarrow$ smaller or equal conditional entropy):
$$\mathrm{H}(Y\mid\chi_T(U),\chi_T(V)) \ \ge\ \mathrm{H}(Y\mid\chi(U),\chi(V)).$$
$\mathrm{H}_b^{-1}$ is increasing (Section 4), so applying it preserves this direction:
$$\mathrm{H}_b^{-1}\big(\mathrm{H}(Y\mid\chi_T(U),\chi_T(V))\big) \ \ge\ \mathrm{H}_b^{-1}\big(\mathrm{H}(Y\mid\chi(U),\chi(V))\big).$$
Chaining with the boxed result above (by transitivity of $\ge$):
$$P_e \ \ge\ \mathrm{H}_b^{-1}\big(\mathrm{H}(Y\mid\chi(U),\chi(V))\big).$$
This holds for **every** finite $T$ — i.e. every finite-depth message-passing GNN, regardless of
depth or width, since depth/width only change $T$ or the fineness of $\psi$, neither of which
affects the color pair the argument conditions on. This is precisely the claimed "depth and
width do not help."

**Step 7 — strict positivity.** Assumption 1 states $\mathrm{H}(Y\mid\chi(U),\chi(V))>0$ (with positive
probability, two edges share a color pair but differ in sign — meaning the conditional
distribution of $Y$ given that color pair is not a point mass, i.e. has strictly positive
entropy). Since $\mathrm{H}_b^{-1}(0)=0$ and $\mathrm{H}_b^{-1}$ is *strictly* increasing (Section 4), a
strictly positive input gives a strictly positive output: $\mathrm{H}_b^{-1}(\mathrm{H}(Y\mid\chi(U),\chi(V)))>0$,
hence $P_e>0$. $\blacksquare$

This is the paper's Proposition 1, now with every step individually justified. No gaps found.

---

## 7. Corollary (single-endpoint reading)

Fix a specific vertex $u$ (not random) and a specific target color $b$. Step 2 of Section 6 above
established, for *any* fixed color pair $(a,b)$ with positive probability, that
$P_e(a,b)\ge\mathrm{H}_b^{-1}(\mathrm{H}(Y\mid\chi_T(U){=}a,\chi_T(V){=}b))$ — this did not require averaging
over $(a,b)$; it's a pointwise statement. Specializing to $a=\chi(u)$ (this specific vertex's own
color) recovers exactly the Corollary's claim, with **no Jensen step needed** since there is
nothing to average over — the paper's proof text says exactly this, correctly. $\blacksquare$

---

## 8. Proposition 2 (Capacity form) — full proof, broken into the same shape as Prop. 1

The paper's own Appendix proof (line 396) compresses this into one dense paragraph, unlike
Proposition 1's Setup/Bound/Identity-vs-color structure. Below is the same content broken into
that shape — see Section 12 for the suggested tex restructuring using this exact breakdown.

**Setup.** Each vertex representation $h_u$ takes at most $N$ distinguishable values under the
read-out $g$ (this $N$ is a *modeling parameter* describing the read-out's effective resolution,
not necessarily the embedding's raw representable-value count — see the honesty check on this
in Section 10.3).

**Step 1 — bound the mutual information.** A random variable supported on at most $N$ values has
entropy at most $\log_2 N$ (uniform distribution over the $N$ values maximizes entropy at
exactly $\log_2 N$; any other distribution on the same support has strictly less — a standard
fact, itself provable via Gibbs' inequality / non-negativity of KL divergence between the actual
distribution and the uniform one, not re-derived here as it's elementary). So $\mathrm{H}(h_U)\le\log_2N$
and $\mathrm{H}(h_V)\le\log_2N$. Joint entropy is subadditive: $\mathrm{H}(h_U,h_V)\le\mathrm{H}(h_U)+\mathrm{H}(h_V)$ (a
standard fact — the gap $\mathrm{H}(h_U)+\mathrm{H}(h_V)-\mathrm{H}(h_U,h_V)=\mathrm{I}(h_U;h_V)\ge0$ always). Mutual
information is bounded by either marginal's entropy: $\mathrm{I}(Y;h_U,h_V) = \mathrm{H}(h_U,h_V) -
\mathrm{H}(h_U,h_V\mid Y) \le \mathrm{H}(h_U,h_V)$ (since $\mathrm{H}(h_U,h_V\mid Y)\ge0$). Chaining:
$$\mathrm{I}(Y;h_U,h_V)\ \le\ \mathrm{H}(h_U,h_V)\ \le\ \mathrm{H}(h_U)+\mathrm{H}(h_V)\ \le\ 2\log_2N.$$

**Step 2 — invert to a conditional-entropy bound.** By the identity $\mathrm{I}(Y;h_U,h_V) =
\mathrm{H}(Y)-\mathrm{H}(Y\mid h_U,h_V)$ (definition of mutual information, rearranged), and Step 1:
$$\mathrm{H}(Y\mid h_U,h_V) = \mathrm{H}(Y) - \mathrm{I}(Y;h_U,h_V) \ \ge\ \mathrm{H}(Y) - 2\log_2N$$
(subtracting a quantity that is *at most* $2\log_2N$ can only leave a result that is *at least*
$\mathrm{H}(Y)-2\log_2N$ — direction check: if $M\le2\log_2N$ then $-M\ge-2\log_2N$ then
$\mathrm{H}(Y)-M\ge\mathrm{H}(Y)-2\log_2N$; correct).

**Step 3 — apply the inverse-Fano bound.** Setting $Z=(h_U,h_V)$ in Section 3's result,
$P_e\ge\mathrm{H}_b^{-1}(\mathrm{H}(Y\mid h_U,h_V))$. Since $\mathrm{H}_b^{-1}$ is increasing and
$\mathrm{H}(Y\mid h_U,h_V)\ge\mathrm{H}(Y)-2\log_2N$ (Step 2):
$$\mathrm{H}_b^{-1}\big(\mathrm{H}(Y\mid h_U,h_V)\big) \ \ge\ \mathrm{H}_b^{-1}\big(\mathrm{H}(Y)-2\log_2N\big)$$
(using the convention $\mathrm{H}_b^{-1}(x):=0$ for $x\le0$, so this holds even when $\mathrm{H}(Y)<2\log_2N$
— the statement is then just $P_e\ge0$, trivially true but uninformative). Chaining:
$$P_e\ \ge\ \mathrm{H}_b^{-1}\big(\mathrm{H}(Y)-2\log_2N\big).$$
Informative (i.e. a genuinely positive lower bound) exactly when $\mathrm{H}(Y)>2\log_2N$.
$\blacksquare$ (population statement)

**Per-vertex specialization.** $h_U$ alone (dropping $h_V$) satisfies $\mathrm{I}(\{Y_e\}_{e\ni
U};h_U)\le\mathrm{H}(h_U)\le\log_2N$ by the same argument as Step 1 applied to the single embedding
$h_U$ and the collection of $U$'s incident-edge signs as the "$Y$" being predicted. If $U$ has
out-degree $D$ and its $D$ out-edge signs are (as an approximation — this is explicitly a
modeling assumption, not a theorem) near-independent draws from a distribution of entropy
$\mathrm{H}_\mathrm{out}(U)$ each, their joint entropy is approximately $D\cdot\mathrm{H}_\mathrm{out}(U)$ (entropy of
independent variables is additive: $\mathrm{H}(X_1,\dots,X_D)=\sum_i\mathrm{H}(X_i)$ *exactly* under true
independence; "near-independent" gives this only approximately, which is why the paper hedges
the word choice). Once $D\cdot\mathrm{H}_\mathrm{out}(U)$ exceeds the $\log_2N$-bit budget that $h_U$ can
carry, the excess information about $U$'s own out-edges cannot be encoded in $h_U$ — informally,
"the excess cannot pass through the shared embedding," matching the paper's closing line. This
last step is intentionally informal (an approximation argument, correctly flagged as such by
"near-independent"), not a fully derived inequality — appropriately, since it is presented as an
intuition-building corollary of the rigorous population bound above, not as a new theorem.

No bugs found in either the population or per-vertex parts.

---

## 9. General-$c$ extension

Weaken (Fano) using $\mathrm{H}_b(P_e)\le1$ (true: $\mathrm{H}_b$'s maximum over its whole domain $[0,1]$ is 1,
attained at $p=\tfrac12$ — Section 1b) and $\log_2(c-1)\le\log_2c$ (true for $c\ge2$: $c-1<c$ and
$\log_2$ is increasing). Since $P_e\ge0$, multiplying the second inequality by $P_e$ preserves
direction: $P_e\log_2(c-1)\le P_e\log_2c$. Chaining into (Fano):
$$\mathrm{H}(Y\mid Z) \ \le\ \mathrm{H}_b(P_e)+P_e\log_2(c-1) \ \le\ 1+P_e\log_2(c-1) \ \le\ 1+P_e\log_2c.$$
Rearranging (subtract 1, divide by $\log_2c>0$ for $c\ge2$):
$$P_e\ \ge\ \frac{\mathrm{H}(Y\mid Z)-1}{\log_2c}.$$
At $c=2$: $\log_2c=1$ and $\mathrm{H}(Y\mid Z)\le1$ always, so the right side is $\le0$ always — vacuous,
exactly as the paper states (and exactly why eq:invfano, not this linear form, is used for the
binary case throughout). For $c>2$, $\log_2c>1$ and $\mathrm{H}(Y)$ can exceed 1 bit (up to $\log_2c$),
so the bound *can* be non-vacuous, though whether it actually is depends on the specific
distribution — the paper's claim is only that the *structural* vacuousness specific to $c=2$
(where the bound's ceiling exactly equals the maximum possible entropy) doesn't generically occur
for $c>2$, not that every $c>2$ instance is automatically informative. That's a correct, properly
hedged reading of what was written.

**Extending Propositions 1/2 and the Corollary:** since $(\cdot-1)/\log_2c$ is an affine function
of its argument, it is simultaneously convex and concave, and Jensen's inequality for an affine
function holds with *equality*: $\mathbb E[a X+b] = a\,\mathbb E[X]+b$ exactly (linearity of
expectation, not an inequality at all). So Step 4 of Section 6 (the Jensen step) becomes an
equality rather than an inequality when this linear form replaces $\mathrm{H}_b^{-1}$, and every other
step goes through identically with $\mathrm{H}_b^{-1}(\cdot)$ replaced by $(\cdot-1)/\log_2c$
throughout. Confirmed correct.

---

## 10. Non-triviality / degeneracy analysis

This is the question "is the algebra right" doesn't answer: **does the theorem, once proved,
actually constrain anything, or could its hypotheses/conclusions be vacuous or circular in every
case that matters?**

### 10.1 Is Assumption 1 a meaningful (non-circular, non-vacuous) hypothesis?

Assumption 1 requires: with positive, non-negligible probability, a WL color pair
$(\chi(u),\chi(v))$ is shared by two edges of *different* sign. This is a conjunction of two
things that both need to hold: (i) color-pair collisions happen at all (extremely likely in any
graph with more edges than the number of distinct color pairs the WL procedure can produce —
essentially guaranteed by pigeonhole once WL stabilizes on a real, large graph, since color
refinement is monotone and typically plateaus well before every vertex gets a unique color,
especially for the "long tail" of structurally similar low-degree vertices that dominate real
social graphs — `DATASET_STATS.md` shows 10–47% of vertices at degree $\le1$ across the six
datasets, and such vertices are frequently WL-indistinguishable from many structural twins), and
(ii) those colliding edges aren't *always* same-signed. (ii) is the substantive part, and it is
not circular: it is exactly the empirical premise the whole paper is built on (that vertex-level
structure alone doesn't perfectly determine edge sign), and it is directly falsifiable — if real
graphs turned out to have $\mathrm{H}(Y\mid\chi(U),\chi(V))=0$ everywhere, Assumption 1 would fail and
Proposition 1 would (correctly) give no useful bound. It doesn't fail; Section 6.1's own
measurements (e.g. the worked example "$\mathrm{H}_\mathrm{out}=\mathrm{H}_b(0.8)\approx0.72$") are direct evidence
that real endpoint-conditional entropy is often substantial, not near-zero. **Assumption 1 is
meaningful and empirically grounded, not vacuous or circular.**

### 10.2 Is Proposition 1's bound quantitatively negligible near the boundary?

A separate concern: $\mathrm{H}_b^{-1}$ has *zero slope* at $0$ (from Section 1c, $\mathrm{H}_b'(p)\to\infty$
as $p\to0^+$, so the inverse's derivative $\to0$ there) — meaning a *small* positive entropy
could map to a *vanishingly* small $P_e$ lower bound, making the "positive $P_e$" conclusion
technically true but practically negligible. Checking with real numbers: at entropy $H=0.08$
bits, $\mathrm{H}_b^{-1}(0.08)\approx0.01$ (i.e. even a fairly small residual entropy already forces
$\ge1\%$ error) — computed by solving $\mathrm{H}_b(p)=0.08$ numerically, since $\mathrm{H}_b(0.01)\approx0.0808$
bits. And the paper's own Section 6.1 reports realistic entropy values in the $0.5$–$1.0$-bit
range routinely (e.g. $\mathrm{H}_\mathrm{out}(0.8)\approx0.72$, $\mathrm{H}_\mathrm{out}(0.5)=1$) — at $H=0.72$,
$\mathrm{H}_b^{-1}(0.72)\approx0.199$ (numerically solved, verified), i.e. a forced $\ge19.9\%$ error rate. **In the paper's actual
empirical regime, the bound is far from the flat, near-zero part of $\mathrm{H}_b^{-1}$'s range — it's
quantitatively substantial, not a technicality.**

### 10.3 Is Proposition 2's capacity bound vacuous in practice? (the interesting finding)

The paper defines $N=2^{bd}$ for width-$d$ embeddings at $b$ bits/coordinate, taken completely
literally (e.g. $b=32$ for float32). Checking against `DATASET_STATS.md`'s real degree numbers
(max total degree across all six datasets: 888 / bitcoin-alpha, 1,298 / bitcoin-otc, **3,622 /
epinions**, 1,167 / wiki-elec, 1,346 / wiki-rfa, 2,557 / slashdot090221) against the paper's own
stated embedding-width range $d\in[32,128]$ (Setup paragraph):

| $d$ | $\log_2N = 32d$ (bits, float32) | Does the single most extreme hub (Epinions, $D{=}3{,}622$) reach it at max entropy ($\mathrm{H}_\mathrm{out}{=}1$)? |
|---|---|---|
| 32 | 1,024 | **Yes** — $3{,}622 > 1{,}024$, bound binds |
| 64 | 2,048 | **Yes** — $3{,}622 > 2{,}048$, bound binds |
| 128 | 4,096 | No — $3{,}622 < 4{,}096$, bound does not bind |

So the literal-bit-capacity reading is **not uniformly vacuous, but binds only for the single
most extreme hub vertex in the single most degree-skewed dataset, and only at the lower half of
the paper's own embedding-width range** — every other vertex in every other dataset, at every
tested width, needs entropy well below the theoretical maximum ($\mathrm{H}_\mathrm{out}=1$) even to get
close, and none would reach it at $d=128$. At realistic (non-maximal) entropy — the six datasets
are 77–94% sign-imbalanced (per the paper's own Datasets paragraph), so $\mathrm{H}_\mathrm{out}\approx0.5$–
$0.7$ is more typical than the maximal $1.0$ used in the table above — even the Epinions hub would
need degree well above 3,622 (at $\mathrm{H}_\mathrm{out}=0.6$, the threshold degree for $d=64$ is $2048/0.6
\approx3{,}413$, still just barely under 3,622, so it's genuinely on the knife's edge even before
considering more realistic, sub-maximal entropy).

**This is not a bug** — the paper is honest about the ambiguity, defining $N$ generally as "the
covering number of the representation space at the decision margin of $g$," which is almost
certainly far smaller than $2^{32d}$ in practice (a classifier's actual decision boundary cannot
statistically distinguish anywhere near $2^{32}$ values per float coordinate given finite,
noisy training data — the *effective* number of bits a trained network can reliably use per
coordinate is typically single digits, not 32). **But this means the bound's practical bite rests
almost entirely on that hedge, not on the literal number** — under the literal reading, it is a
near-miss for one dataset's most extreme vertex and irrelevant everywhere else; under the
"effective/statistical capacity" reading (much smaller $N$), the bound would bind far more
broadly, but that reading is not something the paper (or this document) can quantify without
directly measuring the model's effective per-coordinate resolution, which nobody has done. This
is worth being explicit about if a reviewer presses on Proposition 2's practical relevance — the
honest answer is "it depends on which $N$ you mean, and the two readings give very different
answers," not "yes, obviously."

### 10.4 Summary

Proposition 1's hypothesis (Assumption 1) is empirically grounded and its bound is quantitatively
substantial in the paper's real operating regime (10.1, 10.2). Proposition 2's bound, taken at
its most literal, is on the edge of vacuous — informative for at most one vertex in one dataset
at low-to-mid embedding widths — and its real force depends on an unquantified "effective
capacity" interpretation that the paper correctly flags but doesn't pin down (10.3). Neither of
these is a correctness bug; both are legitimate, useful things to know about what the two
Propositions are actually buying the paper.

---

## 11. Suggested rewording for the "flipping" justification (line 115)

Current text: *"...and any predictor with $P_e>\tfrac12$ can be improved by flipping its output,
so the bound inverts to..."*

Per Section 3 above, the actual reason the bound holds unconditionally is that $\mathrm{H}_b^{-1}$'s
range is capped at $\tfrac12$, not anything about flipping predictors. Two alternatives:

**Option A (short, precise):**
> "...and since $\mathrm{H}_b^{-1}$ takes values only in $[0,\tfrac12]$, the bound
> $P_e\ge\mathrm{H}_b^{-1}(\mathrm{H}(Y\mid Z))$ holds automatically when $P_e>\tfrac12$, and follows directly
> by inverting $\mathrm{H}_b$ on $[0,\tfrac12]$ otherwise — so it holds unconditionally:"

**Option B (slightly more explanatory):**
> "...so for $P_e\le\tfrac12$ the bound follows by inverting $\mathrm{H}_b$ on $[0,\tfrac12]$; for
> $P_e>\tfrac12$ it holds automatically, since $\mathrm{H}_b^{-1}$'s range never exceeds $\tfrac12$.
> Either way:"

Both drop the "flipping" framing entirely, since (as shown in Section 3) it isn't actually needed
and doesn't by itself establish the bound for the original (unflipped) predictor.

---

## 12. Suggested restructure for Proposition 2's Appendix proof

Currently one dense paragraph (line 396), unlike Proposition 1's labeled
Setup/Bound/Identity-vs-color structure. Suggested split, matching Section 8 above:

```latex
\paragraph{Proposition~\ref{prop:capacity}.}
\emph{Setup.} Let each vertex representation take at most $N$ distinguishable values under the
read-out (for width $d$ at $b$ bits per coordinate, $N=2^{bd}$; in general $N$ is the covering
number of the representation space at the decision margin of $g$).

\emph{Bound.} A variable supported on $\le N$ values has entropy $\le\log N$, so
$\mathrm{H}(h_U),\mathrm{H}(h_V)\le\log N$; by subadditivity $\mathrm{H}(h_U,h_V)\le\mathrm{H}(h_U)+\mathrm{H}(h_V)\le2\log N$; and
$\mathrm{I}(Y;h_U,h_V)\le\mathrm{H}(h_U,h_V)\le2\log N$. Substituting into
$\mathrm{H}(Y\mid h_U,h_V)=\mathrm{H}(Y)-\mathrm{I}(Y;h_U,h_V)$ gives $\mathrm{H}(Y\mid h_U,h_V)\ge\mathrm{H}(Y)-2\log N$. Applying
\eqref{eq:invfano} and the monotonicity of $\mathrm{H}_b^{-1}$,
$P_e\ge\mathrm{H}_b^{-1}\big(\mathrm{H}(Y)-2\log N\big)$; with the convention $\mathrm{H}_b^{-1}(x):=0$ for $x\le0$
this is unconditionally true, but informative only when $\mathrm{H}(Y)>2\log N$.

\emph{Per-vertex specialization.} $h_U$ alone satisfies
$\mathrm{I}(\{Y_e\}_{e\ni U};h_U)\le\mathrm{H}(h_U)\le\log N$ by the same argument. If $U$ has out-degree $D$
with near-independent out-edge labels of entropy $\mathrm{H}_\mathrm{out}(U)$ each, they carry
$D\,\mathrm{H}_\mathrm{out}(U)$ bits jointly; once this exceeds $\log N$ the excess cannot pass through the
shared embedding. $\square$
```

This mirrors Prop. 1's three-part shape (Setup / Bound / a closing specialization) one-for-one —
Prop. 1 has Setup / Bound / Identity-vs-color, Prop. 2 would have Setup / Bound / Per-vertex
specialization — and makes the proof's logical structure (population bound first, then a
narrower per-vertex reading) visually explicit rather than buried in one paragraph. No content
changes from the current proof, purely a formatting split.
