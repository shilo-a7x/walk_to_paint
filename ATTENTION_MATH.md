# Walk-to-Paint: the exact math, input to output

Symbolic treatment — every dataset uses the *same* equations with different constants
($d$, $H$, $F$, $N$ per `configs/<dataset>.yaml`; see the table at the bottom for real
numbers). Nothing here is dataset-specific.

## Notation

| Symbol | Meaning | Where it comes from |
|---|---|---|
| $S$ | sequence length (walk length in **tokens**, not hops) | $S \le 2 \cdot \texttt{max\_walk\_length} + 1$; 80 hops $\Rightarrow S \le 161$ |
| $V$ | vocabulary (node ids $\cup$ edge-sign tokens $\cup$ `PAD`/`MASK`/`UNK`) | built by the tokenizer per dataset |
| $d$ | embedding dim (`model.embedding_dim`) | per-dataset, e.g. 32/64/128 |
| $H$ | number of attention heads (`model.nhead`) | per-dataset, e.g. 2/4/8 |
| $d_h$ | per-head dim, $d_h = d/H$ | must divide evenly — this is why $d$/$H$ are chosen together per dataset |
| $F$ | feed-forward inner width (`model.hidden_dim`) | per-dataset |
| $N$ | number of stacked encoder layers (`model.nlayers`) | per-dataset |
| $C$ | number of output classes | $C=2$ (edge sign $\in \{\text{negative}, \text{positive}\}$) |
| $w$ | local-attention window (`model.local_attention_window`) | `null` (full attention) or $w=4$ (LocalAttn4) |

Everything below is one forward pass over one walk (batch dimension $B$ omitted — every
equation just gets an extra leading $B$ axis and runs identically per batch item, modulo
padding, which is discussed separately).

## Step 0 — the raw input: a tokenized walk

A walk over the signed graph is written as an alternating sequence of node and edge-sign
tokens, indexed by position $i = 0, 1, \dots, S-1$:

$$
\underbrace{N_{u_0}}_{i=0},\ \underbrace{E_{s_1}}_{i=1},\ \underbrace{N_{u_1}}_{i=2},\ \underbrace{E_{s_2}}_{i=3},\ \underbrace{N_{u_2}}_{i=4},\ \underbrace{E_{s_3}}_{i=5},\ \dots
$$

- **Even positions ($i \equiv 0 \pmod 2$) are node tokens** — the node visited at that
  point in the walk.
- **Odd positions ($i \equiv 1 \pmod 2$) are edge tokens** — the *sign* of the edge just
  traversed to get from the previous node to the next one.
- One graph-hop (traverse one edge) $=$ **advance 2 token positions** (one node token, one
  edge token). So two same-parity positions $i, j$ (both even or both odd) are
  $\lvert i-j \rvert / 2$ graph hops apart.

This parity is a structural fact about the sequence layout, **not** about the token's
current value — a `MASK`ed or `UNK`-replaced edge token still occupies an odd position and
is still "an edge slot." This is important for the attention analysis below: position
parity alone tells you node-vs-edge role, no need to decode the token id.

Only edge-token positions ever get selected as a supervised prediction target (this is a
masked-**edge-sign** prediction task) — node identity is never the thing being predicted,
only optionally corrupted as an input-side regularizer (the "R" flag, `node_context_mode:
replace`). So every masked target position $i$ used anywhere in the attention analysis
satisfies $i \equiv 1 \pmod 2$ (always odd).

## Step 1 — token embedding

$$
E \in \mathbb{R}^{|V| \times d} \qquad \text{(nn.Embedding, learned, one row per vocab id)}
$$

$$
e_i = E[x_i] \in \mathbb{R}^{d} \qquad \text{for token } x_i \text{ at position } i
$$

## Step 2 — positional encoding (fixed, not learned)

Standard sinusoidal encoding, one vector per position, **added** to the token embedding
(not concatenated):

$$
\mathrm{pe}_{i,\,2k} = \sin\!\left(\frac{i}{10000^{\,2k/d}}\right), \qquad
\mathrm{pe}_{i,\,2k+1} = \cos\!\left(\frac{i}{10000^{\,2k/d}}\right)
$$

$$
h_i^{(0)} = e_i + \mathrm{pe}_i \ \in \mathbb{R}^{d}
$$

Stacking all $S$ positions gives the layer-0 input matrix $X^{(0)} \in \mathbb{R}^{S \times d}$.

## Step 3 — one transformer encoder layer (repeated $N$ times)

Each layer is **post-norm** (`norm_first=False`, PyTorch's default, unchanged here):
LayerNorm is applied *after* each residual add, not before.

### 3a. Multi-head self-attention

All three projections are packed into one weight matrix for efficiency, then split:

$$
W_{in} \in \mathbb{R}^{3d \times d}, \quad b_{in} \in \mathbb{R}^{3d}
$$

$$
[\,Q \mid K \mid V\,] = X\,W_{in}^{\top} + b_{in} \ \in \mathbb{R}^{S \times 3d}, \qquad
Q, K, V \in \mathbb{R}^{S \times d} \ \text{(each one third of the above)}
$$

Reshape each into $H$ heads of width $d_h = d/H$ — this is a **reshape**, not a separate
learned matrix per head; the single big $W_{in}$ is already block-structured so that
columns $[h \cdot d_h : (h{+}1) \cdot d_h)$ are "head $h$'s" query/key/value space:

$$
Q_h, K_h, V_h \in \mathbb{R}^{S \times d_h} \qquad \text{for } h = 1, \dots, H
$$

**Per head, per query position $i$, this is where the softmax happens:**

$$
\mathrm{scores}_h[i,j] = \frac{Q_h[i] \cdot K_h[j]}{\sqrt{d_h}} + M[i,j]
$$

$$
A_h[i,:] = \mathrm{softmax}\big(\mathrm{scores}_h[i,:]\big) \in \mathbb{R}^{S}, \qquad \sum_j A_h[i,j] = 1 \ \text{over allowed } j
$$

$M[i,j]$ is an additive mask: $0$ where attention is allowed, $-\infty$ where it's
forbidden. Two things ever populate $M$:
- **padding**: $j$ is a padded/out-of-walk position $\Rightarrow M[i,j] = -\infty$.
- **LocalAttn4 only**: $\lvert i-j \rvert > w$ (default $w=4$) $\Rightarrow M[i,j] = -\infty$.
  Full attention ($w = \texttt{null}$) never adds this term — every non-padded $j$ is
  reachable from every $i$.

$A_h$ is exactly the $[B, H, S, S]$ tensor both attention scripts capture via
`need_weights=True, average_attn_weights=False` — **one full probability distribution per
(layer, head, query position)**, $H \cdot N$ independent distributions in total per token.
The `average_attn_weights=False` flag is what stops PyTorch from silently collapsing the
$H$ heads into their mean before you ever see them — with the default `True` you'd only
get one $[S,S]$ matrix per layer, already averaged, and every head-specific pattern would
be gone.

Per-head weighted value lookup, then heads are **concatenated** (not summed/averaged) and
mixed once by a final linear layer:

$$
O_h = A_h V_h \ \in \mathbb{R}^{S \times d_h}
$$

$$
O = \big[\,O_1 \mid O_2 \mid \cdots \mid O_H\,\big] \in \mathbb{R}^{S \times d} \qquad \text{(concat along the feature axis)}
$$

$$
Z = O\,W_O^{\top} + b_O \in \mathbb{R}^{S \times d}, \qquad W_O \in \mathbb{R}^{d \times d} \ \text{(the only place heads mix)}
$$

This concat-then-mix step is the answer to "how do the multi-head weights combine": **the
attention *weights* (the $A_h$ softmax matrices) never combine across heads at all** — each
head looks at its own independent distribution over positions. Only the *value* vectors
they each produce get concatenated and linearly remixed, once, after attention is already
computed. So a "head that attends to nodes" and a "head that attends to edges" (per the
node/edge split below) can genuinely coexist and both contribute, unblended, to the next
layer's representation.

Residual + LayerNorm:

$$
X' = \mathrm{LayerNorm}\big(X + \mathrm{Dropout}(Z)\big) \ \in \mathbb{R}^{S \times d}
$$

### 3b. Feed-forward block

$$
W_1 \in \mathbb{R}^{d \times F},\ b_1 \in \mathbb{R}^{F} \qquad
W_2 \in \mathbb{R}^{F \times d},\ b_2 \in \mathbb{R}^{d}
$$

$$
\mathrm{FFN}(X') = \mathrm{ReLU}\big(X' W_1^{\top} + b_1\big)\,W_2^{\top} + b_2 \ \in \mathbb{R}^{S \times d}
$$

$$
X^{(l)} = \mathrm{LayerNorm}\big(X' + \mathrm{Dropout}(\mathrm{FFN}(X'))\big) \ \in \mathbb{R}^{S \times d}
$$

$X^{(l)}$ is the input to layer $l+1$. Repeat 3a/3b $N$ times total.

## Step 4 — output head

Applied to **every** position (not just targets), but only ever read out at masked
edge-token positions:

$$
W_{out} \in \mathbb{R}^{d \times C}, \quad b_{out} \in \mathbb{R}^{C} \qquad (C=2)
$$

$$
\mathrm{logits}_i = X^{(N)}_i\,W_{out}^{\top} + b_{out} \ \in \mathbb{R}^{C}
$$

$$
p_i = \mathrm{softmax}(\mathrm{logits}_i) \in \mathbb{R}^{2} \qquad \big(p_i[1] = \widehat{P}(\text{edge is positive})\big)
$$

Cross-entropy loss is computed only where $\mathrm{labels}[i] \ne \texttt{ignore\_index}$,
i.e. only at masked edge positions — the same set of positions the attention analysis
scripts restrict their $i$ to.

## Step 5 — beyond the transformer (not part of the math above)

A given real edge $(u,v)$ appears in **many** walks in the corpus, so it gets many
per-occurrence predictions $p_i$. These are combined post-hoc by a separately-fit weighting
function (`func_logit_power` in production — a small closed-form reweighting by prediction
confidence, fit on val, applied on test) into one final edge-level score. This aggregation
step has its own math but is intentionally outside the transformer itself — it never
touches attention weights, only the scalar $p_i[1]$ values already produced above.

## Per-dataset constants (for reference — the equations above don't change)

| dataset | $d$ | $H$ | $d_h$ | $F$ | $N$ |
|---|---|---|---|---|---|
| bitcoin-alpha | 64 | 4 | 16 | 128 | 3 |
| bitcoin-otc | 64 | 4 | 16 | 128 | 3 |
| epinions | 32 | 8 | 4 | 32 | 3 |
| slashdot090221 | 128 | 4 | 32 | 256 | 4 |
| wiki-elec | 64 | 2 | 32 | 64 | 5 |
| wiki-rfa | 64 | 2 | 32 | 64 | 5 |

---

# What the attention-analysis scripts actually measure

Both scripts hook the same $A_h[i,:]$ softmax rows from Step 3a above (one per
layer/head/masked-target position $i$), average over all sampled masked targets in the
test split, and report distance-binned summaries of where that probability mass falls.
Neither script changes model weights or does any training — pure inference-time
introspection into an already-trained checkpoint.

## Old script — `scripts/attention_analysis.py` (Step 3 of the original local-attention investigation)

**Question it answers:** "how far, in absolute token distance, does attention reach?" —
built specifically to decide whether restricting attention to a local window (LocalAttn4)
would throw away signal that full attention actually uses.

**Metric, per (layer, head), averaged over all masked target positions $i$ in the test set:**

$$
\mathrm{effective\_distance} = \mathbb{E}_j\big[\,\lvert i-j \rvert\,\big] = \sum_j A_h[i,j] \cdot \lvert i-j \rvert
$$

This is the *expected absolute distance* under query $i$'s own attention distribution — a
single number that's large if the head tends to look far away, small if it stays local. It
also records the full histogram (`pmf`) of attention mass at every $\lvert i-j \rvert$
value, not just its mean, and reports $\mathrm{frac\_beyond\_2hop} = P(\lvert i-j \rvert > 4)$.

**What it deliberately does NOT tell you** (the gap this session's new script closes):
- **Direction.** $\lvert i-j \rvert$ throws away the sign of $i-j$, so a head that attends
  heavily *backward* (toward $u$'s history, $j<i$) and one that attends heavily *forward*
  (toward $v$'s future, $j>i$) look identical under this metric if their distances happen
  to match. There is no way, from this script's output, to tell which way a head is looking.
- **Node vs. edge role.** The histogram bins by raw distance, and — because of the
  alternating token layout — that means the reported curve zigzags: since every masked
  target $i$ is odd (an edge position), **even nonzero $\lvert i-j \rvert$ values land on
  edge tokens, odd values land on node tokens** (Step 0 above). The script never labels
  this explicitly — you can only infer it by counting parity yourself — and it never
  aggregates "total mass on nodes" vs. "total mass on edges" as its own number.
- It uses whichever old, pre-`edge_cover` checkpoints `node_mi_structural_embedding.py`'s
  `DATASET_CONFIGS` happens to point at (E14-era) — not the current production
  E25/E26/E27 checkpoints.

## New script — `scripts/attention_directionality.py`

**Question it answers:** for a masked target edge, does the model lean on what came
*before* it in the walk (source-side / backward) or what comes *after* it (target-side /
forward), and is that mass sitting on node tokens or edge tokens? This is a claim about the
**attention mechanism's own geometry** — it is a separate concept from the $H_{out} >
H_{in}$ entropy-asymmetry result (Lead 4c/Wilcoxon), which is a claim about which *data*
signal (rater self-consistency vs. contested reputation) each *architecture* struggles
with. Both happen to live on a source/target axis, but one is "where does attention look,"
the other is "which entropy term predicts an error" — not the same measurement, not
assumed to point the same way.

**Metric, per (layer, head), same averaging as above but keeping the sign.** Define the
signed offset and token role:

$$
d = j - i \qquad (d<0 \Rightarrow \text{backward},\ d>0 \Rightarrow \text{forward},\ d=0 \Rightarrow \text{self})
$$

$$
\mathrm{role}(j) = \begin{cases} \text{edge} & j \equiv 1 \pmod 2 \\ \text{node} & j \equiv 0 \pmod 2 \end{cases}
\qquad \text{(Step 0 parity, independent of the sign of } d\text{)}
$$

$$
\mathrm{fwd\_node} = \!\!\sum_{j:\, d>0,\ \mathrm{role}(j)=\text{node}}\!\! A_h[i,j]
\qquad
\mathrm{fwd\_edge} = \!\!\sum_{j:\, d>0,\ \mathrm{role}(j)=\text{edge}}\!\! A_h[i,j]
$$

$$
\mathrm{bwd\_node} = \!\!\sum_{j:\, d<0,\ \mathrm{role}(j)=\text{node}}\!\! A_h[i,j]
\qquad
\mathrm{bwd\_edge} = \!\!\sum_{j:\, d<0,\ \mathrm{role}(j)=\text{edge}}\!\! A_h[i,j]
$$

$$
\mathrm{self} = A_h[i,i] \qquad \text{(always an edge token, since } i \text{ is odd)}
$$

with the identity $\mathrm{fwd\_node} + \mathrm{fwd\_edge} + \mathrm{bwd\_node} +
\mathrm{bwd\_edge} + \mathrm{self} = 1$ holding exactly (it's a partition of the same
softmax row), averaged over all masked targets, per (layer, head), for both attention
variants:

- **E25/E26 (full attention)** — no forbidden $j$, $M \equiv 0$ except padding; shows what
  the model *chose* to look at when nothing structural stopped it.
- **E27 (LocalAttn4, $w=4$)** — $M[i,j]=-\infty$ for $\lvert i-j \rvert>4$, so by
  construction $\mathrm{fwd\_*}/\mathrm{bwd\_*}$ beyond $d = \pm4$ are exactly zero; the
  interesting question there is only the forward/backward and node/edge *split within*
  that window, and how it compares to what the full-attention mass looked like in that
  same window before it was ever masked.

Also plots the same signed-distance histogram (per layer, one line per head, as in the old
script) but on a **signed** x-axis with node/edge background shading and the $\pm4$
LocalAttn4 boundary marked, plus a summary bar chart of the mass totals above — so the
zigzag pattern and its forward/backward asymmetry, if any, are visible directly instead of
needing to be inferred.
