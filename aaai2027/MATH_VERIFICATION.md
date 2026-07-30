# Full math verification — Problem Setting, Limitations, Appendix

Independent re-derivation of every claim in the theory section (not a trust-the-3-LLM-notes
pass — every step below was checked from scratch against the actual definitions). Structured as
a full logical flow first (so you can see how each result depends on the last), then a numbered
findings list at the end with severity + suggested fix for each. Nothing here has been applied to
the `.tex` yet — this is the "let's see where to wave hands vs. be rigorous" input you asked for.

---

## Part 1: the logical flow, checked step by step

### 1.1 General Fano (Problem Setting, eq:fano)

$$H(Y\mid Z)\le H_b(P_e)+P_e\log(c-1)$$

This is the textbook statement (Cover & Thomas) for any $\hat Y=g(Z)$ estimating $Y$ over $c$
classes. **Verified correct as stated** — $c$ here is $|\mathcal C|$, the alphabet size of $Y$
(matches the Problem Setting definition $|\mathcal C|=c$), not anything to do with $Z$'s
alphabet. No issue.

### 1.2 The $c=2$ specialization

At $c=2$: $\log(c-1)=\log 1=0$, so the inequality collapses to the *exact* (no linear-term
slack) relation

$$H(Y\mid Z)\le H_b(P_e).$$

**Verified correct.** This is the actual mathematical content behind the whole c=2 reframing, and
it's right: the $P_e\log(c-1)$ term is not "small," it's *identically zero* at $c=2$, so nothing
is being dropped or approximated here — this is exact, not an approximation.

### 1.3 Inverting to get $P_e \ge H_b^{-1}(H(Y\mid Z))$ — **this step has a real, fixable gap**

The text argues: $H_b$ restricted to $[0,\tfrac12]$ is a strictly increasing bijection onto
$[0,1]$, and any predictor with $P_e>\tfrac12$ can be improved by flipping its output (true and
fine — flipping a binary predictor's output strictly improves it whenever $P_e>\tfrac12$, and
flipping stays a function of $Z$, so this WLOG costs nothing). So we may assume $P_e\in[0,\tfrac12]$,
where $H_b$ is invertible.

**The gap:** to go from $H(Y\mid Z)\le H_b(P_e)$ to $P_e\ge H_b^{-1}(H(Y\mid Z))$, you need
$H_b^{-1}$ itself to be **monotonically increasing** (so that applying it to both sides of an
inequality preserves the direction). The paper states $H_b$ is an increasing bijection, but never
states — anywhere, as its own fact — that this makes $H_b^{-1}$ increasing too. It's a standard,
one-line fact (inverse of a strictly increasing function is strictly increasing — trivial proof:
if $H_b^{-1}(x) < H_b^{-1}(y)$ were false for some $x<y$, applying the increasing $H_b$ to
$H_b^{-1}(x)\ge H_b^{-1}(y)$ would give $x\ge y$, contradiction), but it is never actually stated
as a fact anywhere in the paper. It gets smuggled in three more times later (see 1.6, 1.7, 1.8
below) without ever being independently justified — the closest thing to a justification is a
parenthetical inside the convexity Lemma's own proof ("Since $f^{-1}$ is strictly increasing,
applying it to both sides preserves the inequality") which *uses* monotonicity to prove
*convexity*, not the other way around, so it can't be the place monotonicity itself gets
established.

**This is the single most important finding** (Finding #1 below) — it's exactly the thing you
flagged ("not clear from text that $H^{-1}$ is monotone and that is used for saying more entropy
is higher lower bound on error"). You're right, and it's used more places than just this one.

### 1.4 The Pinsker-type closed-form alternative

Claimed: $H_b(p)\le 1-\tfrac{2}{\ln 2}(\tfrac12-p)^2$, giving
$P_e\ge \tfrac12-\sqrt{\tfrac{\ln 2}{2}(1-H(Y\mid Z))}$.

**Independently re-derived, verified correct.** This is a real, standard consequence of Pinsker's
inequality: writing $D(p\Vert\tfrac12)$ for the (bits) KL divergence of Bernoulli($p$) from
Bernoulli($\tfrac12$), a direct expansion gives $D(p\Vert\tfrac12)=1-H_b(p)$ exactly (not an
approximation), and Pinsker's inequality (converted from nats to bits) gives
$D(p\Vert\tfrac12)\ge\tfrac{2}{\ln 2}\,\Vert p-\tfrac12\Vert_{TV}^2=\tfrac{2}{\ln 2}(p-\tfrac12)^2$
(total-variation distance between two Bernoullis differing only in $p$ is exactly $|p-\tfrac12|$).
Combining: $1-H_b(p)\ge\tfrac{2}{\ln2}(p-\tfrac12)^2$, i.e. $H_b(p)\le
1-\tfrac{2}{\ln2}(\tfrac12-p)^2$ — matches the paper exactly. Plugging in $H(Y\mid Z)\le H_b(P_e)$
and solving the resulting quadratic for $P_e$ (using $P_e\le\tfrac12$, established in 1.3, to take
the correct root) reproduces the paper's displayed bound exactly. **Spot-checked numerically** at
$p\in\{0,0.1,0.3,0.5\}$ — inequality holds at every point, tight at $p=\tfrac12$. No issue here.

### 1.5 The convexity Lemma ($H_b^{-1}$ is strictly convex)

$H_b$ restricted to $[0,\tfrac12]$: **verified strictly concave** by direct second-derivative
check — $H_b''(p)=\tfrac{1}{\ln2}\left(-\tfrac1p-\tfrac1{1-p}\right)<0$ for all $p\in(0,1)$, so
concave on the whole interior, in particular on $[0,\tfrac12]$. **Verified strictly increasing**
on $[0,\tfrac12)$ — $H_b'(p)=\log_2(\tfrac{1-p}{p})>0$ for $p<\tfrac12$ — with the derivative
hitting 0 only at the single endpoint $p=\tfrac12$, which doesn't break strict monotonicity of the
function itself on the closed interval.

Given increasing + concave, the paper's proof that $H_b^{-1}$ is strictly convex (the direct
definitional argument — apply the increasing $f^{-1}$ to both sides of concavity's defining
inequality) is the same one independently verified earlier this session against ProofWiki's
"Inverse of Strictly Increasing Strictly Concave Real Function is Strictly Convex" — **verified
correct, no differentiability assumption needed, as claimed.**

One structural note: this Lemma's own proof *uses* "$f^{-1}$ is strictly increasing" as a given,
un-derived fact (see 1.3) — so the natural fix for Finding #1 is to fold the monotonicity
statement into this same Lemma (making it do double duty: state+prove monotonicity first, in one
sentence, *then* use it to prove convexity), rather than adding a separate standalone fact
elsewhere. See Finding #1's suggested fix.

### 1.6 Proposition 1 (bottleneck) — main text and appendix

Main text claims $P_e\ge H_b^{-1}\big(H(Y\mid c(U),c(V))\big)>0$.

**Chain of reasoning, verified step by step:**
- Any $T$-round message-passing GNN's node representation is a function of the $T$-round WL color
  $c_T(\cdot)$, which refines toward the stable color $c(\cdot)$ — this is the standard "MPNN is
  no more expressive than 1-WL" result (Xu et al. 2019 Thm 1 / Morris et al. 2019), and it holds
  for *any* permutation-invariant local aggregation (mean, sum, attention-weighted — GAT-style
  included), not just injective ones; injectivity is what's needed to *match* 1-WL, not to be
  *bounded by* it. **Verified this is the right direction to invoke** (upper bound on GNN
  expressivity, used here to lower-bound error for *any* GNN, which is exactly what's needed).
- Within a fixed color-pair stratum $(a,b)$, any $\mathcal F$-measurable predictor is constant, so
  this is literally a Bayes-error question for a single Bernoulli — the per-stratum bound
  $P_e(a,b)\ge H_b^{-1}(H(Y\mid a,b))$ follows directly from 1.3 applied within the stratum.
  **Side observation (not an error, just an opportunity):** this bound is actually an *equality*
  for the majority-vote/MAP rule specifically, since $H_b(\min(p,1-p))=H_b(p)$ by $H_b$'s
  symmetry around $\tfrac12$ — the paper only claims $\ge$, which is true but slightly under-sells
  what's actually a tight characterization for the optimal rule. Optional tightening, not a bug.
- Averaging across strata via Jensen on the now-convex $H_b^{-1}$ (1.5) is applied in the correct
  direction: $H_b^{-1}$ convex $\Rightarrow$ $\mathbb E[H_b^{-1}(X)]\ge H_b^{-1}(\mathbb E[X])$,
  which is exactly the inequality direction used
  ($P_e=\mathbb E_{(a,b)}[P_e(a,b)]\ge\mathbb E_{(a,b)}[H_b^{-1}(H(Y\mid a,b))]\ge
  H_b^{-1}(\mathbb E_{(a,b)}[H(Y\mid a,b)])$). **Verified correct direction** — this is the
  detail most likely to be gotten backwards (a concave function would need the opposite
  direction), and it's right.
- Last equality, $\mathbb E_{(a,b)}[H(Y\mid a,b)]=H(Y\mid c(U),c(V))$, is exactly the chain rule /
  definition of conditional entropy as an expectation over the conditioning variable's law.
  **Verified correct**, standard identity.
- The "$>0$" in the final statement needs Assumption 1 (positive-probability non-injective color
  pair $\Rightarrow$ some stratum has strictly positive conditional entropy with positive weight
  $\Rightarrow$ the weighted average $H(Y\mid c(U),c(V))>0$ — **verified correct**, straightforward
  since all per-stratum terms are non-negative and at least one is strictly positive with positive
  weight) **and** $H_b^{-1}$ strictly increasing with $H_b^{-1}(0)=0$ (so a strictly positive
  input gives a strictly positive output) — **this second half is Finding #1 again**, used here
  for the fourth time in the paper without ever being independently stated.
- The appendix's $c_T$-vs-$c$ argument (any finite-$T$ GNN's bound, using the coarser $c_T$, is
  *at least as large* as the stable-coloring bound, since $H(Y\mid c_T)\ge H(Y\mid c)$ —
  conditioning on less-refined information can only leave *more* residual entropy — and
  $H_b^{-1}$ increasing carries that through) is **verified correct**, and is the *rigorous*
  version of the main text's looser one-line claim that "any GNN factors through [the stable
  coloring]" (see Finding #3 — that one-line main-text claim is a harmless simplification of what
  the appendix actually proves, not a separate error).

### 1.7 Corollary (single-endpoint reading)

Fixes source identity $u$ and target color $b$ — a *finer* stratum than Prop 1's $(a,b)$ color-pair
stratum (since "source $=u$" implies "source color $=c(u)$" but not conversely). The classifier
is still constant on this finer stratum (it's a subset of a set on which $g$ is already constant),
so the same single-stratum Fano argument from 1.3 applies directly, **no Jensen needed** (correctly
noted in the text) since there's no averaging across multiple strata here — just one direct
application. **Verified correct**, self-contained, doesn't lean on Prop 1's averaged statement at
all.

The closing informal remark ("if $u$'s out-edges are mixed and not separated by target color, this
entropy is close to $H_\text{out}(u)$") is appropriately hedged ("close to," not an exact equality)
— the precise relationship is $H_\text{out}(u) = H(Y\mid\text{source}=u) \ge
\mathbb E_b[H(Y\mid\text{source}=u,\,c(v){=}b)]$ (conditioning on more info, here $c(v)$, can only
reduce entropy on average), with equality exactly when $c(v)$ carries no information about the
sign — i.e. exactly the "not separated by target color" case being described in words. **Verified
consistent**, no fix needed, this is legitimately just an intuitive gloss rather than a formal
claim, and it's honest about that.

### 1.8 Proposition 2 (capacity form) — **the domain issue you flagged is real**

Main text: $\mathrm I(Y;h_U,h_V)\le 2\log N$ and $P_e\ge H_b^{-1}(H(Y)-2\log N)$.

**The $\le 2\log N$ half is fully verified correct**, a clean three-step chain: $\mathrm
I(Y;h_U,h_V)\le H(h_U,h_V)$ (mutual info with anything is bounded by that thing's own entropy —
standard, since $\mathrm I(Y;Z)=H(Z)-H(Z\mid Y)\le H(Z)$) $\le H(h_U)+H(h_V)$ (subadditivity of
joint entropy) $\le 2\log N$ (each embedding supported on $\le N$ values has entropy $\le\log N$,
maximized at the uniform distribution — standard). Then $H(Y\mid h_U,h_V)=H(Y)-\mathrm
I(Y;h_U,h_V)\ge H(Y)-2\log N$ by the definitional identity $I(Y;Z)=H(Y)-H(Y\mid Z)$ rearranged.
All individually standard, correctly chained.

**The domain gap:** $H(Y)-2\log N$ can be *negative* — nothing prevents $2\log N$ (embedding
capacity) from exceeding $H(Y)$ (at most 1 bit, since $Y$ is binary). $H_b^{-1}$ is only defined
on $[0,1]$ (the range of $H_b$ restricted to $[0,\tfrac12]$). As literally written, the displayed
inequality in the **Proposition statement itself** (not just somewhere in the proof) is not even
a well-formed expression when $H(Y)<2\log N$ — which, given realistic embedding widths, is the
*common* case, not an edge case. The appendix's proof does flag this ("meaningful whenever
$H(Y)>2\log N$; otherwise... the bound is vacuous rather than false") — but that's a proof-level
aside, and "vacuous rather than false" isn't actually a rigorous justification on its own: if
$H_b^{-1}$ genuinely isn't defined there, the statement isn't "vacuously true," it's *not a
statement* at all until you fix a convention. This needs one of two fixes (see Finding #2), not a
verbal reassurance.

The rest of the proof — using $H_b^{-1}$'s monotonicity (Finding #1 again, fifth occurrence) to
substitute $H(Y)-2\log N$ in place of the (unknown, but no-smaller) true value $H(Y\mid
h_U,h_V)$ — is **correctly reasoned**, contingent on Finding #1 being fixed and Finding #2's domain
question being resolved.

The per-vertex corollary-in-prose ("$D\cdot H_\text{out}(U)-\log N$ bits are unrecoverable once
$D\cdot H_\text{out}(U)>\log N$") uses the same style of bound on a single embedding ($\mathrm
I(\{Y_e\}_{e\ni U};h_U)\le H(h_U)\le\log N$) against the joint entropy of $D$ *assumed*
near-independent out-edge labels ($\approx D\cdot H_\text{out}(U)$ under that assumption).
**Verified correct given the stated near-independence approximation**, which is honestly flagged
as an approximation ("near-independent"), not asserted as exact.

### 1.9 General-$c$ appendix remark

Standard weakened linear corollary $P_e\ge\tfrac{H(Y\mid Z)-1}{\log c}$, valid and non-vacuous for
$c>2$ (unlike at $c=2$, matching 1.2/1.3's discussion). Restating Prop 1/Corollary/Prop 2 in this
form for $c>2$, replacing $H_b^{-1}(\cdot)$ with the affine $\tfrac{\cdot-1}{\log c}$ throughout,
needs no Jensen/convexity argument since an affine function is trivially both convex and concave.
**Verified correct**, no issue.

---

## Part 2: notation issues found during the scan (not logic errors, but worth listing)

- **"$c$" is overloaded.** It's the number of classes ($|\mathcal C|=c$, a scalar, Problem
  Setting) *and* the WL-coloring function $c(\cdot)$ (Limitations section) *and* appears in the
  finite-round variant $c_T(\cdot)$. They're distinguishable by context (bare vs. applied to an
  argument) but sit in the same sentences repeatedly — e.g. Prop 1's proof: "applying the $c=2$
  Fano bound... within each color pair" uses both senses of "$c$" in one clause. A reviewer
  skimming could momentarily misread "$c=2$" as "the coloring function equals 2." Cheap fix:
  rename the coloring function (e.g. $\chi(\cdot)$ instead of $c(\cdot)$) — purely cosmetic, zero
  content change, but removes the collision entirely.
- **$H_\text{source}(e)$/$H_\text{target}(e)$ vs. $H_\text{out}(u)$/$H_\text{in}(v)$ look like two
  independent notations for what is (as far as I can tell from the text) the same underlying
  quantity** — the source-side entropy of an edge $e=(u,v)$ is just $H_\text{out}(u)$, and the
  target-side entropy is just $H_\text{in}(v)$, re-attached to the edge instead of the vertex.
  Nothing in the text says this explicitly (no "$H_\text{Source}(e):=H_\text{out}(u)$" definition)
  — worth either stating that equivalence once, or dropping one of the two notations entirely if
  they really are the same thing, so a reader doesn't wonder whether they're secretly different.

---

## Part 3: full findings list (severity-ranked)

| # | Finding | Severity | Where it bites | Suggested fix |
|---|---|---|---|---|
| 1 | **$H_b^{-1}$'s monotonicity (strictly increasing) is used 5 times but never independently stated/proved** — only smuggled in parenthetically inside the convexity Lemma's own proof, where it's *assumed*, not derived. This is exactly the "more entropy $\Rightarrow$ higher error lower bound" logic the whole paper's story rests on. | **High — must fix.** It's genuinely a missing step, not just unclear writing; the paper currently never actually justifies its central monotone-in-entropy reading. | Problem Setting's core inversion (1.3); Prop 1's "$>0$" conclusion (1.6); Prop 1's appendix $T$-independence argument (1.6); Prop 2's substitution (1.8); the convexity Lemma's own proof (1.5, which currently uses it un-derived). | One added sentence, ideally folded into the existing convexity Lemma (rename to "Lemma (monotonicity and convexity of $H_b^{-1}$)"): state and give the one-line proof that the inverse of a strictly increasing function is strictly increasing, *before* using it to prove convexity. Cheapest possible fix for the highest-value gap found. |
| 2 | **Proposition 2's bound $P_e\ge H_b^{-1}(H(Y)-2\log N)$ has an argument that can be negative — outside $H_b^{-1}$'s domain $[0,1]$ — and this is the common case (embedding capacity usually exceeds 1 bit), not an edge case.** The appendix's "vacuous rather than false" aside is a verbal reassurance, not a fix — as written, the Proposition's own displayed statement isn't well-typed in that regime. | **High — must fix**, this is a genuine domain gap, exactly what you flagged. | Proposition 2's main-text statement and its appendix proof. | Two options, pick one: **(a)** extend the definition once, globally: $H_b^{-1}(x):=0$ for $x\le 0$ (natural, since $P_e\ge0$ trivially and $H_b^{-1}$ is already 0 at the boundary $x=0$, so this extension is continuous) — state this convention once (e.g. right after the Lemma) and every downstream use, including this one, becomes unconditionally well-typed and true. **(b)** Add an explicit hypothesis to the Proposition itself ("...then, when $H(Y)>2\log N$, $P_e\ge\ldots$"), leaving the complementary case undiscussed rather than "vacuously" asserted. Recommend (a) — fixes this and any future similar spot in one place, rather than hedging each invocation individually. |
| 3 | Main-text sentence "any message-passing GNN factors through it [the stable WL coloring $c$]" is a slight simplification — a finite-depth-$T$ GNN factors through the coarser $c_T$, not the stable $c$ exactly; the Appendix correctly distinguishes these and proves the bound still holds uniformly over $T$. | Low — cosmetic/precision, not a logical error (the rigorous version already exists in the Appendix). | Limitations section's opening paragraph (main text only). | Optional one-clause softening, e.g. "...factors through it in the sense that a $T$-round coloring $c_T$ refines toward $c$ (Appendix)" — or leave as-is and let the Appendix carry the rigor, which is a defensible split for a "proof sketch" main text. |
| 4 | The per-stratum Fano bound in Prop 1's appendix proof is stated as $P_e(a,b)\ge H_b^{-1}(H(Y\mid a,b))$, when it's actually an *equality* for the majority-vote/MAP rule specifically ($H_b(\min(p,1-p))=H_b(p)$ by symmetry). Not wrong, just leaves a sharper true statement on the table. | Cosmetic / optional strengthening. | Prop 1's appendix "Bound" paragraph. | Optional: add "(with equality when $g$ is the per-stratum majority vote)" — makes the bound's tightness explicit, costs nothing. |
| 5 | Notation collision: "$c$" means both "number of classes" (scalar) and "the WL-coloring function $c(\cdot)$," in the same sentences in places (e.g. "applying the $c=2$ Fano bound... within each color pair"). | Low / clarity only. | Throughout Limitations + Appendix. | Optional rename of the coloring function to $\chi(\cdot)$ (or similar) — zero content change, removes ambiguity. |
| 6 | $H_\text{Source}(e)$/$H_\text{Target}(e)$ (Problem Setting) look like independent notation from $H_\text{out}(u)$/$H_\text{in}(v)$, without ever being stated as literally the same quantity re-attached to an edge instead of a vertex. | Low / clarity only. | Problem Setting's notation paragraph. | Optional: add "$H_\text{Source}(e):=H_\text{out}(u)$ for $e=(u,v)$" (and the target/in analogue) once, or drop one notation if truly redundant. |

**Everything else checked** (general Fano statement, the $c=2$ collapse itself, the Pinsker-bound
derivation, the MI/entropy identities in Prop 2, the WL-expressivity direction being invoked
correctly, the Jensen-direction for a convex function, the chain-rule identity closing Prop 1's
proof, Assumption 1's role in forcing strict positivity, the general-$c$ remark) **came back
clean** — no errors found, re-derived independently rather than taken on the 3 LLM notes' word.

---

## Where this leaves the "wave hands vs. be rigorous" conversation

My read, for discussion:

- **Findings #1 and #2 aren't really a hand-wave-vs-rigor choice — they're small, cheap fixes that
  remove genuine gaps**, not places where more rigor would cost you clarity or space. #1 is one
  sentence inside an existing Lemma. #2 is one sentence (a domain convention) that, once stated,
  makes every downstream use of $H_b^{-1}$ (including ones not flagged above) unconditionally
  correct rather than needing a case-by-case caveat. I'd fix both regardless of how rigorous the
  rest of the paper wants to be, since a reviewer who checks the math carefully (which a
  theory-adjacent AAAI reviewer likely will, given these are the paper's headline Propositions)
  is exactly the reader who'd catch these.
- **Findings #3–6 are genuinely optional** — real hand-wave-vs-rigor territory. #3 and #4 are
  "the appendix is already more careful than the main text, is that OK for a proof sketch" —
  I'd say yes, that's a completely standard main-text/appendix split. #5 and #6 are pure notation
  hygiene, worth doing if you're already touching these paragraphs for another reason, not worth a
  standalone pass.
