# The full theory, explained: WL coloring, capacity, and every claim/proof

This picks up where `MATH_ELI5_EXPLAINED.md` left off. That doc covered the information-theory
toolkit from zero (entropy, $H_b$, Fano, the Pinsker bound) — worth reading first if any of
$H(Y\mid Z)$, $H_b^{-1}$, or "$c=2$" still feel shaky. This doc assumes that toolkit and focuses
on the GNN-specific machinery: what a "vertex-centric classifier" and the computation tree are,
what WL coloring actually *is* (the algorithm, not just the name), why it bounds every GNN, and
then a full walkthrough of every claim and proof in order, main text and appendix merged into one
narrative. (Uses $\chi$ for the WL-coloring function throughout, matching the just-applied rename
— `pewter_aaai.tex` used to reuse the letter "$c$" for both "number of classes" and "the coloring
function," which is fixed now.)

---

## 1. What "vertex-centric classifier" means, precisely

The paper's definition: $\hat y_{uv}=g(h_u,h_v)$, where $h_u=\phi(\mathcal T_u)$ — a vertex
representation computed from $u$'s **computation tree** $\mathcal T_u$, and $g,\phi$ are
*arbitrary* (any architecture, any training procedure).

**What is a computation tree, concretely?** Take a $T$-layer message-passing GNN. To compute
$u$'s final representation, you need $u$'s neighbors' representations from the previous layer,
which need *their* neighbors' representations from the layer before that, and so on, $T$ layers
deep. If you draw out this whole dependency structure — $u$ at the root, its neighbors as
children, their neighbors as grandchildren, etc., down to depth $T$ — you get a tree. It's called
the computation *tree* (not the local subgraph) because if the underlying graph has a cycle, the
unrolling walks around it multiple times, revisiting the same vertices at different depths, so the
picture really is tree-shaped even though the graph underneath it isn't.

**Why is the definition this general?** Because $\phi$ is allowed to be *anything* — this
definition isn't "a 2-layer GCN" or "a GAT," it's *every possible way of turning a $T$-round
neighborhood into a vector*, covering GCN, GraphSAGE, GAT, and any signed-GNN variant with an
edge read-out, all at once. The claims to follow have to work for literally any $\phi,g$, which
is exactly why the proof route goes through WL coloring — a purely combinatorial, architecture-
agnostic ceiling that applies no matter what $\phi,g$ turn out to be.

---

## 2. What WL coloring actually is (the algorithm, not just the name)

**"WL" = Weisfeiler–Leman**, originally a graph-isomorphism test from the 1960s, repurposed in
GNN theory as an expressiveness yardstick (Xu et al. 2019; Morris et al. 2019). The **1-WL color
refinement algorithm**, concretely:

1. **Start:** every vertex gets some initial color (e.g. all the same color, if there are no
   input node features; or its own feature/degree, if there are).
2. **Refine, one round at a time:** each vertex's *new* color is computed by hashing together
   *(its own current color, the multiset of its neighbors' current colors)*. "Multiset" matters —
   it's neighbor colors bagged together, unordered, with repeats kept (so a vertex with two
   red neighbors and one blue is different from one red and one blue).
3. **Repeat** until the partition of vertices into color classes stops changing (this always
   happens eventually, in at most $|V|$ rounds) — call this the **stable coloring** $\chi(\cdot)$.
   The color after exactly $T$ rounds (before it's necessarily stabilized) is written $\chi_T(\cdot)$.

**A worked example where WL fails, directly relevant to Assumption 1.** Take a 4-cycle: vertices
$A,B,C,D$ with edges $A\!-\!B,\,B\!-\!C,\,C\!-\!D,\,D\!-\!A$. Every vertex starts the same color
(no features) and every vertex has exactly 2 neighbors. Round 1: every vertex's neighbor-color
multiset is *identical* (two neighbors, both the one shared starting color) — so every vertex
gets the same new color again. This repeats forever; **1-WL can never tell $A,B,C,D$ apart**, even
though they're four distinct vertices. This is the textbook failure mode, and it's exactly what
Assumption 1 is pointing at when it says the bound is "forced... inside any graph with structural
symmetry or repeated motifs" — a 4-cycle is precisely such a motif, and if edges around that cycle
don't all carry the same sign, no vertex-centric classifier can tell those edges apart no matter
how it's trained, because their endpoints are WL-indistinguishable.

**Contrast: a graph where WL does distinguish vertices.** A simple path $A\!-\!B\!-\!C\!-\!D$ (no
wraparound) breaks the symmetry immediately — $A$ and $D$ have one neighbor each, $B$ and $C$ have
two — so round 1 already splits vertices into an "endpoint" color and a "middle" color, and
further rounds keep refining based on distance from the ends. The difference between the cycle and
the path is exactly the difference between "Assumption 1 is forced" and "Assumption 1 might not
hold" — WL's blindness is a real, structural property of *some* graphs, not all of them, and it
gets worse (harder to avoid) in symmetric/regular graphs and better in ones with more distinguishing
local structure.

---

## 3. Why *every* message-passing GNN factors through the WL color — the actual mechanism

This is the part that lets Proposition 1 apply to literally any architecture, not just a specific
one, so it's worth seeing *why* it's true rather than taking it as a black-box citation.

One round of message passing computes a new vertex representation as some function of *(the
vertex's own previous representation, the multiset of its neighbors' previous representations)* —
using a sum, mean, max, or attention-weighted combination, then an update function. **Compare this
to step 2 of the WL algorithm above: combine own color + multiset of neighbor colors into a new
color.** These are the *same computational shape* — a GNN layer is literally "WL refinement, but
using a possibly-lossy learned function instead of a canonical hash" instead of an injective hash.

The key one-line argument: **an injective function never merges two things a non-injective one
would keep separate — merging can only go the other way.** So if two vertices get the same 1-WL
color after $T$ rounds (meaning an *injective*, maximally-distinguishing hash couldn't tell them
apart), then *any* GNN's learned (possibly non-injective) aggregation function, run for the same
$T$ rounds, *also* can't tell them apart — a lossier combination rule can't recover distinctions
that the best possible combination rule already failed to make. This holds regardless of whether
the aggregation is sum/mean/max/attention-weighted, since all of those are still permutation-
invariant functions of the same (own, neighbor-multiset) input WL itself uses — this is exactly
why the claim in the Limitations section's opening line covers "GCN, GraphSAGE, GAT, and signed
GNNs with an edge read-out" all in one sweep, attention-based architectures (like SiGAT) included.

**Consequence, stated as the paper does:** $h_u=\psi(\chi_T(u))$ for some function $\psi$ — i.e.
*whatever* $u$'s GNN representation actually is, it can be re-expressed as some (possibly
complicated, possibly learned) function applied only to $u$'s $T$-round WL color, never to
anything WL can't see. This is a **one-directional ceiling**: it says GNNs can't *exceed* 1-WL's
distinguishing power, not that they always *reach* it — real GNNs are often less expressive than
this ceiling in practice, which only makes the bound *more* conservative (safer), not less valid.

---

## 4. Assumption 1 (Structural regime), read precisely

> "The endpoints enter the read-out only through their WL colors $\chi(u),\chi(v)$, and these
> colors are not injective on the edges we predict: with positive probability a color pair is
> shared by two edges of different label."

Two things packed in here:
- **First clause** is just Section 3's conclusion restated as a standing assumption for the rest of
  the section — the read-out only ever sees colors, never raw identity.
- **Second clause** is the actual substantive assumption: *somewhere* in the edge distribution,
  two edges with the exact same $(\chi(u),\chi(v))$ color pair have *different* true signs. If this
  never happened — i.e. color pair always determined the sign exactly — a vertex-centric
  classifier could in principle be perfect, and there'd be no bottleneck to prove.
- **When is this forced, not just possible?** Two sufficient conditions the paper gives: (a) test
  edges whose endpoints were never seen during training — then identity is unusable by
  construction (a model can't condition on an identity it has no learned parameters for), so only
  the color is available, and if colors ever collide with differing labels, the assumption holds;
  (b) *any* structural symmetry or repeated motif in the graph — like the 4-cycle in Section 2 —
  where WL provably can't separate some vertices no matter how many rounds you run.

---

## 5. Proposition 1 (Endpoint bottleneck) — full walkthrough

**Statement:** under Assumption 1, every vertex-centric classifier has

$$P_e\ \ge\ H_b^{-1}\big(H(Y\mid \chi(U),\chi(V))\big)\ >\ 0$$

for a random edge $(U,V)$.

**Reading it in words:** take the entropy that's *left* in the true sign once you've replaced the
two endpoints' raw identities with their WL colors — this is a well-defined number, computable
from the data alone, no model involved yet. Run it through the same $H_b^{-1}$ dictionary from
`MATH_ELI5_EXPLAINED.md` Part 1: whatever number of bits that residual entropy is, that's the
floor on the error rate of *any* vertex-centric classifier — not this specific GNN, not this
training run, but every function of the form $g(h_u,h_v)$ that could ever be built this way.

**Proof, merging the main-text sketch and the appendix's full version:**

1. **Reduce to a coloring problem.** By Section 3, any GNN's prediction factors as
   $\hat y_{uv}=\tilde g(\chi_T(u),\chi_T(v))$ for its own depth $T$ — i.e. $\hat Y$ is measurable
   with respect to (fully determined by) the color pair $(\chi_T(U),\chi_T(V))$.
2. **Within one fixed color pair, this is just a coin-flip problem.** Fix a specific pair of
   colors $(a,b)$. *Every* edge whose endpoints happen to have exactly this color pair gets the
   *same* prediction from $\tilde g$ (since $\tilde g$ only sees $(a,b)$, nothing else) — so
   within this stratum, we're really asking "how well can one fixed guess do against the true
   sign distribution restricted to this stratum?" That's exactly the single-variable Fano question
   from `MATH_ELI5_EXPLAINED.md` Part 2/3, applied with $Z$ fixed at the constant value $(a,b)$:
   the best possible fixed guess (majority vote within the stratum) has error at least
   $H_b^{-1}(H(Y\mid \chi_T(U){=}a,\chi_T(V){=}b))$, and *any* other constant guess (including
   whatever $\tilde g$ actually outputs) can only do worse, so the same floor applies to it too.
3. **Average over all color pairs, using convexity.** The *overall* error is the probability-
   weighted average of the per-stratum errors: $P_e=\mathbb E_{(a,b)}[P_e(a,b)]$. Since each
   $P_e(a,b)\ge H_b^{-1}(H(Y\mid a,b))$ (step 2), and $H_b^{-1}$ is convex (the Appendix Lemma),
   Jensen's inequality lets us pull the average *inside* $H_b^{-1}$ while only making the bound
   *weaker* in the safe direction:
   $$P_e=\mathbb E_{(a,b)}[P_e(a,b)]\ge\mathbb E_{(a,b)}\big[H_b^{-1}(H(Y\mid a,b))\big]\ge
   H_b^{-1}\big(\mathbb E_{(a,b)}[H(Y\mid a,b)]\big)=H_b^{-1}\big(H(Y\mid\chi_T(U),\chi_T(V))\big).$$
   The last equality is just the definition of conditional entropy as an average over the
   conditioning variable's own distribution — nothing new, a standard identity.
4. **Swap in the *stable* coloring $\chi$ instead of the depth-$T$ coloring $\chi_T$, to get a
   bound that doesn't depend on depth at all.** Since $\chi_T$ refines *toward* $\chi$ (it's a
   coarser, earlier stage of the same refinement process — see Section 2), conditioning on the
   coarser $\chi_T$ leaves *at least as much* residual entropy as conditioning on the finer,
   fully-refined $\chi$: $H(Y\mid\chi_T(U),\chi_T(V))\ge H(Y\mid\chi(U),\chi(V))$. Since
   $H_b^{-1}$ is increasing (the Lemma, again), this means the bound computed using the stable
   coloring is *never bigger* than the true bound for any specific depth $T$ — i.e. it's a valid,
   depth-independent floor that every finite-depth GNN's own (tighter) bound sits above. This is
   the formal content of "depth doesn't help": running more message-passing rounds can only move
   $\chi_T$ closer to $\chi$, which can only *shrink* the residual entropy, which by this same
   monotonicity can only *raise* the true bound for that depth above the depth-independent floor
   — it never breaks through it.
5. **Strict positivity.** Assumption 1 guarantees $H(Y\mid\chi(U),\chi(V))>0$ (some stratum has
   positive residual entropy with positive probability weight, so the weighted average is
   strictly positive too). Since $H_b^{-1}$ is increasing with $H_b^{-1}(0)=0$, a strictly
   positive input gives a strictly positive output: $P_e>0$.

**What "width doesn't help either" means.** Nothing above depended on the embedding dimension —
the whole argument is about what information the *color* carries, not how many numbers are used to
store $\psi(\chi_T(u))$. A wider embedding could store $\chi_T(u)$ more redundantly, or with more
decimal precision, but it can't manufacture distinctions among vertices that share the same color
— there's simply nothing more to encode. (Width shows up instead in Proposition 2, in a genuinely
different way — see Section 8.)

---

## 6. The Remark (identity vs. color) — why having multiple edges per vertex pair is a red herring

A natural objection: "in a simple graph, the pair of vertex *identities* $(U,V)$ uniquely
determines which single edge you're talking about, so $H(Y\mid U,V)=0$ trivially — doesn't that
mean the model already has zero uncertainty once it knows which edge it's looking at?"

**Why this doesn't rescue the GNN:** the model never actually gets to condition on raw identity —
by Section 3, it only ever sees the *color* $\chi(u),\chi(v)$, and colors are (by Assumption 1)
not fine enough to separate everything identity would. The real quantity that matters is the
*information gap created by replacing identity with color*:
$$H(Y\mid\chi(U),\chi(V))-H(Y\mid U,V)=H(Y\mid\chi(U),\chi(V))-0=H(Y\mid\chi(U),\chi(V))$$
— exactly the quantity Proposition 1 bounds. So whether the graph happens to have one edge or a
hundred between the same pair of vertices is irrelevant; the bottleneck is set by the *resolution*
of what the model conditions on (colors), not by how much raw structure the graph technically
contains that the model never gets to use.

---

## 7. Corollary (Single-endpoint reading) — the practical takeaway

Fix one specific source vertex $u$ and one target color $b$. Every edge going from $u$ to *some*
vertex colored $b$ gets exactly one shared prediction from any vertex-centric classifier (since the
classifier only sees the pair $(\chi(u),b)$, and that's fixed for this whole group). This is a
*finer* stratum than Proposition 1's — we've additionally pinned down the source's actual identity,
not just its color — but the same single-stratum Fano argument from step 2 above applies directly,
with no need for the Jensen-averaging step (there's only one stratum being discussed, not many
being combined).

**In plain words, this is "high endpoint entropy breaks the GNN":** if a rater $u$'s outgoing
signs are genuinely mixed, *and* that mixing isn't explained away by which color of target they're
pointing at (i.e. knowing $\chi(v)$ doesn't help predict $u$'s sign toward $v$), then $u$'s
out-edges are forced to have real error for *every* possible GNN, trained however you like — this
is the single most quotable consequence of the whole theory section, and it's the one Result 2 /
the Empirical Confirmation section go looking for evidence of directly (via $H_\text{out}(u)$,
bucketed).

---

## 8. Proposition 2 (Capacity form) — full walkthrough

**Two versions, pair-level and per-vertex — worth keeping mentally separate, since they behave very
differently in practice** (see `MATH_ELI5_EXPLAINED.md` Part 4 for the full plain-language
treatment of what $N$ is and why the pair-level version is usually vacuous for realistic
embeddings — this section focuses on the proof mechanics).

**Pair-level claim:** if each vertex embedding takes at most $N$ distinguishable values (a
capacity limit — see Part 4 of the ELI5 doc for what "distinguishable" means concretely), then

$$I(Y;h_U,h_V)\le 2\log N,\qquad P_e\ \ge\ H_b^{-1}\big(H(Y)-2\log N\big).$$

**Proof, step by step:**
1. Mutual information between $Y$ and anything is capped by that thing's own entropy:
   $I(Y;h_U,h_V)\le H(h_U,h_V)$ (you can't resolve more surprise about $Y$ than the conditioning
   variable itself contains).
2. Joint entropy never exceeds the sum of marginal entropies: $H(h_U,h_V)\le H(h_U)+H(h_V)$.
3. A variable confined to at most $N$ values has entropy at most $\log N$ (maximized exactly when
   uniform over those $N$ values): $H(h_U)\le\log N$, $H(h_V)\le\log N$.
4. Chaining 1–3: $I(Y;h_U,h_V)\le 2\log N$ — the two embeddings *together* can carry at most
   $2\log N$ bits about $Y$, full stop, independent of training.
5. Substitute into the identity $H(Y\mid h_U,h_V)=H(Y)-I(Y;h_U,h_V)$ to get
   $H(Y\mid h_U,h_V)\ge H(Y)-2\log N$.
6. Apply the exact $c{=}2$ Fano bound directly (no color-pair averaging needed here — this is a
   single global statement, not stratified): $P_e\ge H_b^{-1}(H(Y\mid h_U,h_V))$, then substitute
   step 5's inequality in using $H_b^{-1}$'s monotonicity (the Lemma) to get
   $P_e\ge H_b^{-1}(H(Y)-2\log N)$.

**Per-vertex, degree-based specialization — the version that actually bites regardless of $N$:**
a single embedding $h_U$ carries at most $\log N$ bits about *anything*, including the joint signs
of all of $U$'s out-edges together. If $U$ has out-degree $D$ and its $D$ out-edge signs are
(approximately) independent with entropy $H_\text{out}(U)$ each, their *joint* entropy is about
$D\cdot H_\text{out}(U)$ — which grows linearly in $D$, while $\log N$ is a fixed ceiling that
doesn't grow with degree at all. Once $D\cdot H_\text{out}(U)$ exceeds $\log N$, the excess,
$D\cdot H_\text{out}(U)-\log N$ bits, is information about that vertex's own out-edges that
*cannot* fit through its one shared embedding, no matter how the model is trained — **this is the
form of Proposition 2 that's essentially guaranteed to eventually bind, for any fixed embedding
capacity, simply by high-degree vertices existing.**

**The domain fix (recap from the verification pass):** $H(Y)-2\log N$ can be negative (and,
per the ELI5 doc's Part 4, usually is, under the natural "raw precision" reading of $N$) —
Problem Setting now states the convention $H_b^{-1}(x):=0$ for $x\le0$, which makes this bound
unconditionally *true* (just uninformative) whenever capacity is generous relative to $H(Y)$'s
single bit, rather than leaving the statement undefined in that regime.

---

## 9. General $c$ (Appendix remark) — one paragraph

Everything above was specialized to $c=2$ (binary signs) because that's this paper's actual
setting, and it gives the *exact*, non-weakened Fano relationship. For completeness, if you ever
wanted to extend to more than two label classes (e.g. multi-way edge types), you'd fall back to
the standard *weakened* linear corollary $P_e\ge\frac{H(Y\mid Z)-1}{\log c}$, which is valid and
non-vacuous for any $c>2$ (unlike at $c=2$, where — as established in
`MATH_ELI5_EXPLAINED.md` Part 2 — it collapses to something useless). All three results
(Proposition 1, the Corollary, Proposition 2) restate in this form for general $c$, just replacing
every $H_b^{-1}(\cdot)$ with the affine function $\frac{\cdot-1}{\log c}$ — and the Jensen-
averaging step in Proposition 1 becomes unnecessary in that case, since an affine function is
trivially both convex and concave (Jensen holds with equality, not just inequality, for affine
functions).

---

## 10. Putting it all together — what this actually proves, and why it motivates PEWTER

Strip away the machinery and the two propositions say the same thing from two different angles:

- **Proposition 1 (resolution argument):** no matter how a vertex-centric classifier is built or
  trained, its predictions can only ever be as fine-grained as the WL coloring lets them be — and
  WL coloring is a purely combinatorial ceiling that has nothing to do with model capacity. If two
  differently-signed edges share a color pair (Assumption 1 — forced by unseen test vertices or by
  structural symmetry), no amount of depth or width recovers the lost distinction.
- **Proposition 2 (throughput argument):** even granting a vertex-centric model *arbitrarily fine*
  colors (i.e. sidestepping Proposition 1 entirely), a *single, fixed-size* embedding vector still
  has a hard information ceiling ($\log N$ bits), and any vertex whose out-edges collectively carry
  more label information than that ceiling is *provably* going to lose some of it — this is a
  capacity argument, orthogonal to the resolution argument, and it gets *worse*, not better, as a
  vertex's degree grows.

**Both point at the exact same fix, from different directions:** the shared object being
conditioned on — one summary vector per vertex, whether the limitation is "not fine enough" (Prop
1) or "not big enough" (Prop 2) — is the actual bottleneck. Neither more depth (which only refines
*within* the WL ceiling) nor more width (which only raises $N$, not the fact that it's still a
single fixed-size summary shared by every incident edge) escapes either argument, because both
arguments are about the *architecture* of "compress the vertex, then read two compressed vertices
off against each other" — not about any specific insufficiently-large instantiation of it. The
escape the paper takes (**"condition on the labeled edges near the target edge directly"**) sits
outside both propositions' hypotheses entirely: it's no longer of the form $g(h_u,h_v)$ at all, so
neither bound applies to it in the first place — which is a *structural* escape from these two
specific arguments, not a claim that walk-based conditioning is immune to entropy effects in
general (real, separate entropy effects on the walk model itself are exactly what the Empirical
Confirmation section and Result 2 go and measure directly, rather than assume away).
