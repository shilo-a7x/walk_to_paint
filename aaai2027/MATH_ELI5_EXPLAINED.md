# The math, explained from zero — every symbol, every step

Goal: after reading this, nothing in the theory section should feel like a black box — not
$N$, not $\log N$, not how Pinsker turns into a square root, not why we even need an "inverse
function" at all. Written so you can read start to finish with no math background assumed beyond
basic algebra (square roots, logs). Concrete numbers everywhere instead of just symbols.

At the end: a full side-by-side alternative derivation of every proposition using the simpler
square-root bound instead of $H_b^{-1}$, so you can see exactly what we'd gain and lose by
switching.

---

## Part 0: the four basic quantities

**$Y$** — the thing we're trying to predict: an edge's sign, either $+$ or $-$. Just a coin with
possibly-unfair odds.

**$P(Y{=}1)=p$** — if $p=0.5$, the sign is a genuine coin flip, totally unpredictable. If
$p=0.99$, it's almost always $+$, very predictable.

**Entropy $H(Y)$** — "how many bits of genuine surprise are in $Y$, on average." Measured in
bits. Two extreme intuitions:
- If $p=0.5$ (fair coin): every guess is a real 50/50 gamble, maximum surprise. $H(Y)=1$ bit —
  the most uncertainty a single yes/no variable can have.
- If $p=1$ or $p=0$ (always the same answer): zero surprise, you already know the answer before
  you're told. $H(Y)=0$ bits.
- Anywhere in between: partial surprise, entropy is somewhere strictly between 0 and 1.

**Conditional entropy $H(Y\mid Z)$** — same idea, but "how much surprise is *left* in $Y$ once
you already know $Z$?" If $Z$ tells you everything about $Y$, $H(Y\mid Z)=0$ (no surprise left).
If $Z$ tells you nothing at all about $Y$, $H(Y\mid Z)=H(Y)$ (no reduction in surprise). In this
paper, $Z$ is always some summary a model has access to (a vertex's WL color, an embedding) and
$Y$ is the true sign — so $H(Y\mid Z)$ literally measures "how much about the sign is still
unknown even after you've compressed everything down to $Z$."

**Mutual information $I(Y;Z)$** — "how many bits of the surprise in $Y$ did $Z$ actually resolve
for you?" It's defined as $I(Y;Z)=H(Y)-H(Y\mid Z)$: (total surprise) minus (surprise still left
after seeing $Z$) = (surprise that got resolved). If $Z$ is totally useless, $I(Y;Z)=0$. If $Z$
pins $Y$ down exactly, $I(Y;Z)=H(Y)$ (all of it got resolved).

**$P_e$** — the actual thing we care about: the probability a classifier gets the sign wrong.
Everything below is about proving a *floor* under $P_e$ — a number below which no classifier of
a certain kind can push its error, no matter how well-trained.

---

## Part 1: the binary entropy function $H_b(p)$, and why we only ever use half of it

$$H_b(p) = -p\log_2 p - (1-p)\log_2(1-p)$$

This is just "the entropy of a single coin with $P(\text{heads})=p$," written as a formula instead
of a sentence. A few anchor values, worth just memorizing the shape:

| $p$ | $H_b(p)$ | meaning |
|---|---|---|
| $0$ | $0$ | always tails, zero surprise |
| $0.1$ | $\approx0.47$ | mostly tails, some surprise |
| $0.3$ | $\approx0.88$ | fairly mixed, lots of surprise |
| $0.5$ | $1$ | perfectly fair coin, maximum surprise |
| $0.7$ | $\approx0.88$ | (same as $0.3$ — see below) |
| $0.9$ | $\approx0.47$ | (same as $0.1$) |
| $1$ | $0$ | always heads, zero surprise |

Plotted, this is a symmetric hill/dome shape: it rises from $0$ at $p=0$, peaks at $1$ exactly at
$p=0.5$, then comes back down to $0$ at $p=1$ — perfectly mirror-symmetric around $p=0.5$
(that's *why* $H_b(0.3)=H_b(0.7)$ in the table above: a 30/70 coin and a 70/30 coin have exactly
the same amount of surprise, just about which side is more likely).

**This symmetry is exactly why we can't invert $H_b$ over its whole domain** — if you're told
"the entropy is 0.88 bits," you can't tell whether $p$ was $0.3$ or $0.7$ from that number alone,
both give the same answer. An inverse function needs to be one-to-one (each output comes from
exactly one input), and $H_b$ over all of $[0,1]$ is two-to-one everywhere except the peak.

**The fix, and why it costs nothing:** we only ever apply this to $P_e$, an error probability,
and *any classifier with $P_e>0.5$ can be trivially improved by just flipping its answer* (if
you're wrong more than half the time, doing the opposite is right more than half the time). So
without losing any generality, we only ever need to talk about $P_e\in[0,0.5]$ — the *left half*
of the hill. Restricted to $[0,0.5]$, $H_b$ is one-to-one (strictly climbing from $0$ up to $1$,
no repeats), so *on this restricted domain* it has a genuine inverse function, written
$H_b^{-1}$.

**$H_b^{-1}$: what goes in, what comes out.** $H_b^{-1}$ takes an entropy value (a number of bits,
between $0$ and $1$) and returns the error-probability floor that entropy forces (a number between
$0$ and $0.5$):

$$H_b^{-1}:\ [0,1]\ \longrightarrow\ [0,0.5]$$

A few worked values (numerically inverting the table above), showing the shape:

| entropy in (bits) | $H_b^{-1}$ of it (forced error floor) |
|---|---|
| $0$ | $0\%$ |
| $0.5$ | $\approx11\%$ |
| $0.8$ | $\approx24\%$ |
| $1.0$ | $50\%$ |

Notice the jumps are *not* evenly spaced: going from $0.5\to0.8$ bits (a $0.3$-bit increase) only
moves the floor from $11\%\to24\%$ (a $13$-point jump), but going from $0.8\to1.0$ bits (a smaller,
$0.2$-bit increase) moves the floor all the way from $24\%\to50\%$ (a $26$-point jump). The floor
accelerates upward as entropy approaches its max — that acceleration *is* what "convex" means for
this function, and it's the property the Jensen-inequality step later on depends on.

---

## Part 2: Fano's inequality — why *any* leftover uncertainty forces mistakes

Plain-language statement: if you're trying to guess $Y$ using some information $Z$, and there are
still $H(Y\mid Z)$ bits of genuine uncertainty about $Y$ left over even after seeing $Z$, then your
error rate $P_e$ can't be arbitrarily small — the leftover fog directly limits how good any
guesser can be, no matter how clever the guessing rule is.

**General form** (works for any number of classes $c$, e.g. red/green/blue would be $c=3$):

$$H(Y\mid Z) \le H_b(P_e) + P_e\log(c-1)$$

**Why our setting only ever needs $c=2$:** signs are binary, $+$ or $-$, so $c=2$ throughout this
paper. Plugging in $c=2$: $\log(c-1)=\log(1)=0$ — that whole second term **vanishes exactly**,
not approximately, not "is small enough to ignore." It's a genuine algebraic zero. So for us,
Fano's inequality is simply:

$$H(Y\mid Z)\ \le\ H_b(P_e)$$

**Turning this into a floor on $P_e$.** We have an upper bound on $H(Y\mid Z)$ in terms of
$H_b(P_e)$; we want the reverse — a *lower* bound on $P_e$ in terms of $H(Y\mid Z)$. Since $H_b$
restricted to $[0,0.5]$ is a genuine (one-to-one, always-climbing) function, and — this is the key
extra fact, spelled out in full in Part 3 — its inverse $H_b^{-1}$ is *also* always-climbing, we
can apply $H_b^{-1}$ to both sides of $H(Y\mid Z)\le H_b(P_e)$ without flipping the inequality,
giving

$$P_e\ \ge\ H_b^{-1}\big(H(Y\mid Z)\big).$$

That's the whole engine behind every proposition in the paper: **more leftover entropy $\Rightarrow$
a strictly higher floor under how wrong any classifier is forced to be**, and $H_b^{-1}$ is just
the dictionary that translates "bits of leftover uncertainty" into "percentage points of forced
error," using the table in Part 1.

---

## Part 3: the fact that was missing — why is $H_b^{-1}$ "always-climbing" at all?

This is exactly the gap flagged in the verification doc, explained here in plain terms.

**The claim:** if $x_1$ is a smaller entropy value than $x_2$ (i.e. $x_1<x_2$), then
$H_b^{-1}(x_1) < H_b^{-1}(x_2)$ — smaller entropy in, smaller error-floor out; bigger entropy in,
bigger error-floor out. This is the thing that lets us say "more entropy means a worse forced error
rate," in one direction, always.

**Why this is true (one paragraph, no calculus needed):** $H_b^{-1}$ undoes $H_b$, and $H_b$
(on $[0,0.5]$) only ever climbs — it never comes back down and it never has a flat stretch where
two different $p$'s give the same value. Suppose for contradiction $H_b^{-1}$ *didn't* climb
somewhere — i.e. some larger entropy $x_2$ mapped to a smaller-or-equal error-floor $H_b^{-1}(x_2)
\le H_b^{-1}(x_1)$ even though $x_2>x_1$. Since $H_b$ only climbs, feeding a smaller-or-equal
number into $H_b$ can only give a smaller-or-equal output: $H_b\big(H_b^{-1}(x_2)\big) \le
H_b\big(H_b^{-1}(x_1)\big)$. But the left side is just $x_2$ and the right side is just $x_1$ (by
definition of "inverse" — feeding $H_b^{-1}(x)$ back into $H_b$ gives you $x$ back). So we'd get
$x_2\le x_1$ — directly contradicting that we started with $x_2>x_1$. So the assumption was
impossible: $H_b^{-1}$ must always climb too.

This is a completely generic fact (true for *any* always-climbing function, nothing special about
$H_b$), but the paper currently never actually writes this paragraph down anywhere — it's used
(silently assumed) five separate times: right here inverting Fano, in Proposition 1 to conclude
the error floor is strictly positive (not just non-negative), twice more inside the appendix
proofs, and once inside the convexity lemma's own proof (which *uses* "H_b^{-1} climbs" as a
given fact while proving something else). **Fix: write the one paragraph above, once, and every
one of those five uses becomes fully justified instead of assumed.**

---

## Part 4: what "$N$" actually is, and why the capacity bound often doesn't bite

This is the part flagged as most confusing, so slowest possible walkthrough.

**The setup.** A vertex's embedding $h_u$ is just a vector of numbers — say $d=32$ numbers (the
embedding dimension). In principle a real number can encode infinitely many distinct values, so
if we took that literally, an embedding could distinguish *infinitely* many different vertices
perfectly, and there'd be no bottleneck at all worth discussing. That's clearly not the honest
picture (in practice, floating-point storage, and more importantly the classifier's own
sensitivity, can't actually tell apart two embeddings that are extremely close together) — so the
proposition introduces $N$ to capture, honestly, **the actual number of distinguishable outputs the
embedding can produce**, rather than pretending it's infinite.

**Two equivalent ways to picture $N$:**

1. **Simplest picture — quantization.** Suppose each of the $d=32$ numbers in the embedding is
   only trusted to $b$ bits of precision (say $b=8$, like a low-precision number format) — meaning
   each coordinate can only really be one of $2^8=256$ distinguishable levels, not a true
   continuum. Then the whole vector, all $d$ coordinates together, can be one of
   $N = (2^b)^d = 2^{bd}$ distinguishable combinations. Concretely, at $d=32,\,b=8$:
   $N=2^{256}$ — an almost incomprehensibly large number (about $10^{77}$, more than the estimated
   number of atoms in the observable universe).

2. **More careful picture — the classifier's own margin.** Even with full floating-point
   precision, the *classifier* $g$ sitting on top of the embedding doesn't actually treat every
   tiny numerical difference as meaningful — two embeddings that are close enough together will
   get the exact same predicted label out of $g$ in practice (that's what it means for a
   classifier to be well-behaved / not wildly sensitive to noise). $N$, more precisely, is the
   *covering number*: the smallest number of "indistinguishable-to-$g$" clusters/balls needed to
   cover the whole embedding space. This can be **much smaller** than the raw-precision count
   above, if $g$'s decision margin (how big a wiggle in the embedding it takes to flip the
   predicted label) is fairly coarse.

**Either way, why does $N$ bound entropy?** A basic, standard fact about entropy: *a variable that
can only take one of $N$ possible values has entropy at most $\log_2 N$* (achieved only if all $N$
values are equally likely — any unevenness in how likely each value is only lowers the entropy
below that ceiling). So: $h_U$ takes at most $N$ values $\Rightarrow$ $H(h_U)\le\log_2 N$. Same for
$h_V$: $H(h_V)\le\log_2 N$. Entropy of the pair together can't exceed the sum of the two separately
(a basic, always-true fact — knowing two things jointly is never *more* surprising than the sum of
each one's own surprise): $H(h_U,h_V)\le H(h_U)+H(h_V)\le 2\log_2 N$. And mutual information
between $Y$ and anything is capped by that thing's own entropy (you can't extract more "resolved
surprise" about $Y$ out of $(h_U,h_V)$ than $(h_U,h_V)$ itself contains): $I(Y;h_U,h_V)\le
H(h_U,h_V)\le 2\log N$. Chain all three together: **the embeddings, together, can carry at most
$2\log N$ bits of information about $Y$, full stop, regardless of training.**

**Now: why would $H(Y)$ ever be *bigger* than $2\log N$?** Honest answer, worth saying plainly:
under the first (raw-precision) picture of $N$, **it basically never is**, for realistic neural
embeddings. Take the numeric example above: $d=32$, $b=8$ bits/coordinate gives $\log_2 N=256$
bits, so $2\log N=512$ bits. Compare to $H(Y)$: since $Y$ is a single binary sign, $H(Y)\le1$ bit,
always, no matter what. $1$ bit is nowhere near $512$ bits — under this reading, the pair-level
bound $P_e\ge H_b^{-1}(H(Y)-2\log N)$ is asking $H_b^{-1}$ of a deeply negative number, which is
**always vacuous** for any embedding with realistic width/precision. This is worth being upfront
about rather than glossing over.

**So when does this proposition actually say something?** Two honest ways it becomes real:
- **Reading $N$ as the classifier's decision margin (picture 2 above), not raw storage
  precision** — if the classifier is comparatively coarse/robust (doesn't distinguish very fine
  embedding differences), the *effective* $N$ can be far smaller than $2^{256}$, potentially small
  enough that $2\log N$ is a real, binding number rather than an astronomical one.
- **The per-vertex, degree-based version — this is the one that's basically guaranteed to bind
  eventually, for *any* fixed $N$, however large.** A single vertex $U$'s embedding $h_U$ still only
  carries at most $\log N$ bits total (half the pair bound, since we're now only talking about one
  embedding). But if $U$ has $D$ out-edges, each carrying roughly $H_\text{out}(U)$ bits of
  (assumed near-independent) sign information, the *total* information that would need to be
  squeezed through that one fixed-size embedding is about $D\cdot H_\text{out}(U)$ bits — and this
  grows *linearly with degree $D$*, while $\log N$ stays fixed. However large you set $N$ (however
  much raw storage precision you allow), there's always some degree $D$ large enough that
  $D\cdot H_\text{out}(U) > \log N$ — at that point, some of the information about that vertex's
  many out-edges is provably being lost, unavoidably, by the shared embedding. **This degree-based
  form is arguably the more robust, more honestly "actually happens" version of Proposition 2** —
  worth leaning on this framing rather than the raw pair-level $2\log N$ form when writing the
  paper up, precisely because the pair-level form is vacuous for realistic $N$ almost by default.

**The domain issue, stated plainly.** $H(Y)-2\log N$ can be negative (and, per the above, usually
is, under the natural reading of $N$). $H_b^{-1}$ only accepts inputs between $0$ and $1$ (it's the
"undo" of $H_b$, which only ever outputs values in $[0,1]$) — feeding it a negative number is
asking a bit-count-to-error-rate dictionary to look up an entry that isn't in the dictionary. The
honest fix is to explicitly extend the dictionary: **define $H_b^{-1}(x):=0$ for any $x\le0$** —
this is a completely natural convention (since $P_e\ge0$ trivially is always true anyway, "the
floor is at least $0$" is a totally true, just uninteresting, statement), and once you state that
convention once, every place this comes up (not just Proposition 2) is automatically fine, without
needing to caveat it locally each time.

---

## Part 5: the Pinsker-based square-root bound, derived one algebra step at a time

You asked specifically for the steps between "Pinsker's inequality" and "the square root formula"
— here they are, nothing skipped.

**Step 1 — a fact about $H_b$ itself, not yet about Fano at all.** There's a known refinement of
a classical result (Pinsker's inequality) that directly bounds $H_b(p)$ from above using a simple
quadratic in $(p-0.5)$:

$$H_b(p)\ \le\ 1-\frac{2}{\ln 2}\Big(\frac12-p\Big)^2$$

Sanity-check this at the extremes: at $p=0.5$, the right side is $1-0=1$, and $H_b(0.5)=1$ too —
so the bound is *exactly tight* right at the peak. At $p=0$: right side is
$1-\frac{2}{\ln2}(0.25)\approx1-0.721=0.279$, and the true value $H_b(0)=0$ — so $0\le0.279$
holds, just not tight (the bound is honest but looser the further you get from $p=0.5$). This
inequality itself is a standard, previously-verified fact (re-derived independently in the earlier
verification pass from the definition of KL-divergence plus Pinsker's inequality) — you don't need
to re-derive it from scratch, just trust it as a known building block, the same way you'd trust
"Pinsker's inequality" itself as a citable fact.

**Step 2 — plug the exact Fano bound into Step 1.** We already have $H(Y\mid Z)\le H_b(P_e)$
(Part 2). Combine it with Step 1's bound on $H_b(P_e)$:

$$H(Y\mid Z)\ \le\ H_b(P_e)\ \le\ 1-\frac{2}{\ln2}\Big(\frac12-P_e\Big)^2$$

so, dropping the middle term (we don't need $H_b(P_e)$'s exact value anymore, just the chain of
$\le$'s):

$$H(Y\mid Z)\ \le\ 1-\frac{2}{\ln2}\Big(\frac12-P_e\Big)^2$$

**Step 3 — pure algebra, isolate $P_e$.** Move things around like solving any inequality with a
squared term:

$$\frac{2}{\ln2}\Big(\frac12-P_e\Big)^2\ \le\ 1-H(Y\mid Z)$$

$$\Big(\frac12-P_e\Big)^2\ \le\ \frac{\ln2}{2}\big(1-H(Y\mid Z)\big)$$

Take square roots of both sides (valid since both sides are non-negative — the left is a square,
and the right is non-negative because $H(Y\mid Z)\le1$ always for a binary label):

$$\Big|\frac12-P_e\Big|\ \le\ \sqrt{\frac{\ln2}{2}\big(1-H(Y\mid Z)\big)}$$

**Step 4 — pick the branch, using the fact we already established that $P_e\le0.5$.** An absolute
value $|A|\le B$ means $-B\le A\le B$. Here $A=\frac12-P_e$. We already know (Part 1's WLOG
argument) that $P_e\le0.5$, so $A=\frac12-P_e\ge0$ — we're automatically on the *non-negative*
side, so we only need the upper half of that two-sided inequality:

$$\frac12-P_e\ \le\ \sqrt{\frac{\ln2}{2}\big(1-H(Y\mid Z)\big)}$$

Rearrange one last time to isolate $P_e$ on its own:

$$P_e\ \ge\ \frac12\ -\ \sqrt{\frac{\ln2}{2}\big(1-H(Y\mid Z)\big)}$$

That's the exact bound in the paper — and notice **no "inverse function" concept was needed
anywhere in this derivation** — every step was either substitution or ordinary algebra
(squaring/square-rooting), which is exactly its main selling point over the $H_b^{-1}$ route (see
Part 6).

---

## Part 6: why not just use the square-root bound for *everything*, and skip $H_b^{-1}$ entirely?

This is the direct answer to "what are our alternatives" — a full alternative version of every
proposition, using
$$Q(x)\ :=\ \frac12-\sqrt{\frac{\ln2}{2}(1-x)}$$
(the square-root formula from Part 5, written as a named function of the entropy $x$) in place of
$H_b^{-1}(x)$ everywhere.

### 6.1 Does $Q$ actually have the two properties we need?

Everything downstream depends on $Q$ climbing (more entropy $\Rightarrow$ a bigger floor) and being
convex (needed for the Jensen-averaging step in Proposition 1). Both check out, and — this is the
appeal of this route — **both are visible from one elementary derivative each, no abstract
"inverse of an increasing function" argument required:**

- **Climbing:** as $x$ (entropy) increases, $(1-x)$ decreases, so $\sqrt{\cdots}$ decreases, so
  subtracting a smaller thing from $\tfrac12$ leaves a bigger result — $Q$ goes up as $x$ goes up.
  (One line of calculus confirms this formally: $Q'(x)=\frac{\sqrt{\ln2/2}}{2\sqrt{1-x}}>0$.)
- **Convex:** the same derivative gets *steeper* as $x\to1$ (the $\sqrt{1-x}$ in the denominator
  shrinks toward $0$), meaning $Q$ accelerates upward near $x=1$ — exactly the "convex" shape,
  confirmed by a second derivative that's positive everywhere on $[0,1)$.

### 6.2 The catch: $Q$ is a strictly weaker (looser) bound than $H_b^{-1}$, and it goes *negative* for low entropy

$Q(x) \le H_b^{-1}(x)$ for every $x$, with equality *only* at $x=1$ (both give exactly $0.5$
there) — everywhere else, $Q$ underestimates the true forced-error floor, sometimes badly. Worked
numbers:

| entropy $x$ (bits) | $H_b^{-1}(x)$ (exact) | $Q(x)$ (square-root approx.) |
|---|---|---|
| $1.0$ | $50\%$ | $50\%$ (matches exactly) |
| $0.8$ | $\approx24\%$ | $\approx23\%$ (close) |
| $0.5$ | $\approx11\%$ | $\approx8\%$ (already noticeably looser) |
| $0.28$ | $>0\%$ | $\approx0\%$ (right at the edge) |
| $0.1$ | $>0\%$, small but real | **negative** — meaningless |
| $0$ | $0\%$ | **negative** (about $-8.9\%$) — meaningless |

Solving $Q(x)=0$ exactly: $\tfrac12=\sqrt{\tfrac{\ln2}{2}(1-x)}\ \Rightarrow\ \tfrac14=
\tfrac{\ln2}{2}(1-x)\ \Rightarrow\ 1-x=\tfrac{1}{2\ln2}\approx0.7213\ \Rightarrow\ x\approx0.279$.
**Below about $0.28$ bits of entropy, $Q$ gives a negative (i.e. useless/vacuous) answer, while
the true $H_b^{-1}$ bound is still positive (if only barely) for *any* entropy above exactly
zero.** This is the central trade-off: $Q$ trades away sensitivity in the low-to-moderate entropy
range for the convenience of being an explicit formula.

### 6.3 What this would do to each result, stated concretely

**Problem Setting's core inversion:** becomes $P_e\ge Q(H(Y\mid Z))$ — fine as a standalone
statement, same shape, just a looser number.

**Proposition 1 (bottleneck), using $Q$ instead:**
$$P_e\ \ge\ Q\big(H(Y\mid c(U),c(V))\big)$$
proven by exactly the same Jensen argument as before (single-stratum bound via the per-stratum
version of Part 5's derivation, then average across color-pair strata using $Q$'s convexity from
6.1) — structurally identical proof, just swap the function. **But the "$>0$" conclusion is where
it gets worse:** Assumption 1 only guarantees $H(Y\mid c(U),c(V))>0$ — *some* positive entropy,
however small. That's enough to guarantee $H_b^{-1}(\cdot)>0$ (since $H_b^{-1}$ is positive for
*any* positive input). It is **not** enough to guarantee $Q(\cdot)>0$ — per 6.2, you'd need the
entropy to exceed roughly $0.28$ bits specifically, a real, stronger requirement. **Switching to
$Q$ here would either weaken the theorem's actual guarantee (only concludes $P_e>0$ under a
strictly stronger assumption than the current Assumption 1) or require rewriting Assumption 1 to
explicitly demand entropy above that threshold** — a real cost, not just a cosmetic one.

**Corollary (single-endpoint reading), using $Q$:**
$$P_e\ \ge\ Q\big(H(Y\mid\text{source}{=}u,\text{target color}{=}b)\big)$$
Same single-stratum substitution as Part 5, no Jensen needed (matches the original's structure).
Same caveat as above applies if you want a strictly-positive conclusion out of it.

**Proposition 2 (capacity form), using $Q$:**
$$P_e\ \ge\ Q\big(H(Y)-2\log N\big)$$
Here $Q$ actually has a genuine, small technical advantage over $H_b^{-1}$: because $Q$ is an
explicit algebraic formula (not an abstractly-defined inverse), it's **automatically well-defined
for negative inputs too** — it just returns a large negative (uselessly vacuous, but
mathematically well-formed) number, rather than being flatly undefined the way $H_b^{-1}$ is
without the Part 4 domain-extension convention. So switching to $Q$ here sidesteps needing to
separately declare "$H_b^{-1}(x):=0$ for $x<0$" — you'd still want to *say in words* that the
bound is vacuous whenever $Q$'s output is negative (same honesty requirement as before), but you
wouldn't need the extra defined-by-convention step.

### 6.4 Recommendation

| | $H_b^{-1}$ (current) | $Q$ (square-root) |
|---|---|---|
| Tightness | Exact, information-theoretically tight | Strictly looser everywhere except $x=1$ |
| Needs an "inverse function" concept | Yes | No — explicit closed form |
| Monotonicity | Needs the one-paragraph argument in Part 3 (currently missing) | Visible directly from one derivative |
| Convexity | Needs the ProofWiki-style abstract lemma (already in the appendix) | Visible directly from one derivative |
| Domain for negative inputs (Prop 2) | Undefined without an explicit extension convention | Automatically defined, just numerically vacuous |
| Gives a genuinely positive ("$>0$") conclusion whenever entropy is *any* positive amount | Yes | **No** — needs entropy $\gtrsim0.28$ bits specifically |

**My read:** don't replace $H_b^{-1}$ with $Q$ as the *primary* bound — the loss of sensitivity at
low-to-moderate entropy (6.2/6.3) directly weakens exactly the kind of "even a little bit of
leftover entropy forces *some* error" claim Proposition 1 is built to make. But keep $Q$ exactly
where the current draft already has it: as an explicit, no-inverse-function-needed *alternative*
formula, offered alongside $H_b^{-1}$ for a reader who wants a plug-in-and-compute number instead
of an abstractly-defined function — which is precisely what Problem Setting's existing paragraph
already does. The one thing worth adding, now that it's fully worked out above: a sentence
somewhere noting *why* it's kept secondary (the $\approx0.28$-bit sensitivity floor from 6.2),
rather than presenting it as a free, no-cost alternative.
