# Edge-identity-token experiment — exact mechanism, in detail

## -1. IMPORTANT: two different implementations exist in this directory — use the real one

This experiment was first built as a **self-contained, hand-rolled training loop**
(`dataset.py`, `model.py`, `train_pilot.py` at the top level of this directory) --
plain PyTorch, no Lightning, no production Trainer. That first run's final numbers
(test AUC ~0.62) turned out to be **far below production's real number (0.9188)** --
a 30-point gap far too large to be explained by "missing a regularizer." The
standalone loop diverged from production in ways that were never audited one by
one: no `CosineAnnealingLR` schedule (production has one), a different effective
data-loading/batching path, and generally a from-scratch reimplementation of
machinery ( `LitEdgeClassifier`, PL's `Trainer`, checkpointing/early-stopping) that
this project already spent real engineering effort getting right for the
production pipeline. **That gap means those first numbers should not be trusted or
cited** -- they measure "how good is a quickly hand-rolled loop," not "does the
edge-identity-token idea work."

**Fix: `eid_src/` (2026-08-23), a second, real implementation** that runs through
the actual production machinery -- `LitEdgeClassifier`, PL `Trainer`,
`configs/<dataset>.yaml`, the real `AdamW`+`CosineAnnealingLR` schedule,
checkpointing, early stopping -- via subclassing, not reimplementation. It mirrors
`src/`'s own directory layout (`eid_src/data/`, `eid_src/model/`,
`eid_src/training/`) and is launched the same way `run.py` launches a normal run
(`run_eid.py`, same CLI shape). Concretely, each file either:
  - **imports the real production class/function directly, unmodified** (e.g.
    `from src.model.model import TransformerModel`, `from src.utils.config import
    load_config`) -- for anything this experiment doesn't need to change, or
  - **subclasses the real production class**, overriding only the handful of
    methods that genuinely need to differ for the new tokenization scheme (e.g.
    `EIDLitEdgeClassifier(LitEdgeClassifier)` overrides `__init__` to swap in the
    sign-aware model and `_step` to thread `sign_ids` through -- `configure_optimizers`,
    the LR schedule, all the metrics/logging, and vertex-token replacement (R) are
    100% inherited, unmodified production code).

See section 8 below for exactly which methods are new/overridden vs. inherited, file
by file. **Sections 1-7 below (cache conversion, masking design, model mechanism)
describe concepts that are unchanged between the two implementations** -- the
per-edge token minting, the sign-only-hidden masking design, and the additive
sign embedding are identical ideas in both; only *how the training loop that uses
them is built* changed. The old standalone files are left on disk, unmodified, for
reference/provenance (per project convention: nothing gets deleted) but are
superseded -- don't cite their numbers, and don't extend them further; extend
`eid_src/` instead.

**What this experiment is asking**: today, the model can tell "this is a positive edge"
but never "this is *edge #17360* specifically, and it happens to be positive." Every
positive edge in the whole graph shares one identical embedding row. This experiment
gives every edge its own row (24,186 of them on bitcoin-alpha) — like vertices already
have — and asks: does the model learn anything useful from an edge's own identity, on
top of its sign? Everything below is isolated under `experiments/edge_identity_tokens/`;
no production file (`src/...`, `config.yaml`) is touched. Bitcoin-alpha, seed 42, single
split, self-contained (not routed through the production Lightning training loop).

Every code snippet below is paired with **real numbers pulled from the actual cache**
(walk #0 of bitcoin-alpha), not invented illustrative numbers — you can reproduce every
value shown here by loading the cache yourself and indexing walk 0.

---

## 0. The one-paragraph version

A "walk" is a path through the graph, stored as a flat list of tokens alternating
vertex/edge: `[N_8, E_edge3514, N_209, E_edge17365, N_..., ...]`. Historically every
edge token was just "positive" or "negative" — no per-edge identity. `build_cache.py`
rewrites those edge tokens into per-edge IDs (`old_vocab_size + edge_id`) and moves the
sign into a *separate* parallel array, `sign_ids`, so identity and sign are two
independent numbers at every position instead of one conflated token. At training/eval
time, `dataset.py` hides an edge's `sign_ids` value whenever that edge is the thing
being predicted (or is otherwise off-limits per the train/val/test split) — but leaves
its identity token exactly as-is, since revealing "which edge" is not revealing "what
sign," and the whole point of the experiment is to test whether "which edge" carries
signal on its own. The model (`model.py`) adds a small sign embedding on top of the
normal token embedding, the same way positional encoding already gets added — three
independent signals (content, sign, position) summed together at every position, then
fed through the same Transformer as production.

---

## 1. Background: what a "cache" actually contains

Before touching this experiment's changes, it helps to see the *unmodified* production
cache concretely. A cache file (`data/<dataset>/dataset_cache__*.pt`) stores every
sampled walk as one big flat array (CSR/"ragged" format — all walks concatenated, with
an `offsets` array marking where each one starts/ends), not a list of separate
variable-length sequences. Walk 0 on bitcoin-alpha, for example, is 39 tokens long
(`offsets = [0, 39, 70, ...]`, so walk 0 occupies flat positions 0-38):

```python
flat_input_ids[0:39] = [3, 4, 5, 6, 7, 6, 8, 6, 7, 6, 9, 6, 7, 6, 10, 6, 11, 6, 12,
                         4, 13, 6, 14, 6, 15, 6, 16, 6, 17, 6, 3, 6, 18, 6, 19, 6, 20, 6, 21]
flat_edge_ids[0:39] = [-1, 3514, -1, 17365, -1, 17369, -1, 17360, -1, 17370, -1, 17361,
                        -1, 17375, -1, 12375, -1, 10215, -1, 10297, -1, 22230, -1, 20925,
                        -1, 20930, -1, 22247, -1, 3273, -1, 3412, -1, 2306, -1, 2434, -1, 13592, -1]
```

Every even position is a vertex token (`N_8`=id 3, `N_209`=id 5, ...; `flat_edge_ids=-1`
there). Every odd position is an edge token, and in **production** it's always either
id 4 (`E_-1`, negative) or id 6 (`E_1`, positive) — literally only two distinct values
ever appear at an edge position, regardless of which of the 24,186 real edges it is.
`flat_edge_ids` at that same position tells you *which* real edge it was (3514, 17365,
17369, ...) — that bookkeeping array already existed in production (used for grouping a
target edge's many walk-occurrences back together at aggregation time) but the model
itself never saw it; only `flat_input_ids` (with its 2-valued edge vocabulary) went into
the Transformer.

## 2. Cache conversion (`build_cache.py`) — a pure, one-time, offline rewrite

This script does not resample anything and does not touch the tokenizer/walk-sampler —
it loads an already-built production cache and rewrites two arrays in place, once,
saving a new `.pt` file. Three steps, each shown against walk 0's real data above.

### Step 1 — recover the true sign of every edge position

The production cache doesn't store "this position's sign" directly; it's implicit in
*which* of the two edge-token ids (4 or 6) sits there. We rebuild an explicit
id→sign-class lookup from the tokenizer's own label map (`id2edge_label = {0: 'E_-1',
1: 'E_1'}`, `token2id['E_-1']=4`, `token2id['E_1']=6`):

```python
id2class = torch.full((tok["vocab_size"],), -1, dtype=torch.long)
for class_id, edge_tok in tok["id2edge_label"].items():
    tok_id = tok["token2id"].get(edge_tok)
    if tok_id is not None:
        id2class[tok_id] = int(class_id)     # id2class[4] = 0 (negative), id2class[6] = 1 (positive)
sign_class = id2class[flat_input_ids.long()]  # look this up at every position at once
```

On walk 0, position 1 has `flat_input_ids[1] = 4` → `sign_class[1] = 0` (negative,
that's edge 3514's true sign); position 3 has `flat_input_ids[3] = 6` →
`sign_class[3] = 1` (positive, edge 17365's sign). Vertex positions get `-1` here
(meaningless, never read — `is_edge` masks them out downstream).

### Step 2 — mint one brand-new token id per edge

```python
old_vocab_size = int(tok["vocab_size"])            # 3788 (every vertex + PAD/MASK/UNK + E_-1/E_1)
num_edges = int(flat_edge_ids[is_edge].max()) + 1   # 24186 (one id per real edge, 0..24185)

flat_input_ids_new = flat_input_ids.clone().long()
flat_input_ids_new[is_edge] = old_vocab_size + flat_edge_ids[is_edge].long()
new_vocab_size = old_vocab_size + num_edges         # 27974
```

This is the crux of the whole conversion: **edge `e`'s new token id is simply
`3788 + e`** — a direct, injective, one-line mapping from "which edge" to "which fresh
vocabulary row." Concretely on walk 0: edge 3514 (position 1) becomes token id
`3788 + 3514 = 7302`; edge 17365 (position 3) becomes `3788 + 17365 = 21153`; edge
17360 (position 7) becomes `3788 + 17360 = 21148`. Vertex positions are untouched —
`N_8` is still token id 3 everywhere. The old ids 4 and 6 (`E_-1`/`E_1`) become
permanently dead — nothing maps to them anymore in the new scheme — but they're left in
place rather than reclaimed/renumbered, since that bookkeeping isn't worth the risk for
a pilot that will likely be thrown away or redone if it's ever productionized.

### Step 3 — build the parallel sign array that now carries the sign information

Since `flat_input_ids` no longer implies sign (every edge has a *unique* id now, not a
sign-indicating one), a second array is added that does carry it:

```python
flat_sign_ids = torch.full_like(flat_input_ids_new, 2)   # default: 2 = "n/a" (vertex position)
flat_sign_ids[is_edge] = sign_class[is_edge]              # 0 = negative, 1 = positive
```

Walk 0 after this step: `flat_sign_ids[0:5] = [2, 0, 2, 1, 2]` — position 0 (vertex) is
2/n-a, position 1 (edge 3514) is 0/negative, position 2 (vertex) is 2/n-a, position 3
(edge 17365) is 1/positive, and so on. Both `flat_input_ids_new` and `flat_sign_ids` are
saved into the new cache's `encoded` dict, alongside the untouched `offsets`,
`flat_split_mask`, `flat_edge_ids` — the CSR/ragged storage shape itself never changes,
only what the two content arrays hold.

**Result on the whole bitcoin-alpha corpus** (10.7M total token positions): vocab grows
3,788 → 27,974; `flat_sign_ids` ends up with 5,018,338 positive positions, 257,505
negative, 5,396,773 "n/a" (vertex positions).

---

## 3. The masking design — what changes at `__getitem__` time, per walk, per stage

`build_cache.py` only rewrote the *static* arrays. Which positions actually get shown
to the model, and which get hidden, is still decided per-item, per-stage (train/val/
test), by `EdgeIdentityStageViewDataset.__getitem__` — same as production's
`StageViewDataset`, just with one real behavioral change.

**Production's rule** (unchanged conceptually, just for contrast): a target or
held-out edge's *whole token* gets replaced with `<MASK>`. Since the token IS the sign
in production, there's no way to hide one without the other.

**This experiment's rule, decided explicitly with the user before implementing**: at a
target/held-out position, hide only `sign_ids` (force it to class 2, "unknown"); leave
`input_ids` exactly as-is (the edge's real identity token stays visible). This is the
one thing this experiment can test that production's scheme structurally cannot: does
knowing *which* edge this is (learned from its *other* occurrences elsewhere in the
corpus, where its sign was visible) help predict the sign here, on top of whatever
structural/sign context surrounds it? It's not a label leak, in the same sense that a
vertex's identity is never hidden today either — you're allowed to know *which* edge
you're looking at, just not (at that position) what its sign is.

Continuing the walk-0 example, in the `train` stage (`ds[0]` on `stage="train"`):

| pos | edge_id | role (this stage) | `input_ids` | `sign_ids` | `attn_mask` | `label` |
|---|---|---|---|---|---|---|
| 1 | 3514 | context (TRAIN split) | 7302 (real edge token) | 0 (negative, visible) | 1 | -1 (not a target) |
| 3 | 17365 | context (TRAIN split) | 21153 (real edge token) | 1 (positive, visible) | 1 | -1 |
| 7 | 17360 | **target** (MASK split) | 21148 (real edge token, **unchanged**) | **2 (forced hidden)** | 1 | **1** (true label, captured before hiding) |
| 11 | 17361 | **target** (MASK split) | 21149 (unchanged) | **2 (hidden)** | 1 | **1** |
| 13 | 17375 | **disallowed** (VAL/TEST split, incidental) | **1 (`<MASK>`)** | 2 (hidden) | **0 (excluded from attention)** | -1 |

This table is real output from `EdgeIdentityStageViewDataset(cache_data,
stage="train")[0]`, not a constructed example. Notice the three distinct treatments:
context edges keep both identity and sign fully visible; target edges (this stage's
actual prediction targets — the MASK split, matching production's dynamic-target pool)
keep identity visible but have sign hidden and captured into `labels`; disallowed edges
(edges that belong to a *later* split — val/test — but happen to appear incidentally in
this training walk) get **both** identity and sign hidden, and are additionally
excluded from attention entirely (`attention_mask=0`) so nothing can ever read them as
context regardless of what token happens to sit there.

The actual code (`dataset.py`):

```python
labels = torch.full((L,), self.ignore_index, dtype=torch.long)
labels[target_edges] = sign_ids[target_edges]      # capture the TRUE sign as the label...

sign_ids[target_edges] = self.sign_na_id            # ...THEN hide it (class 2)
sign_ids[disallowed_edges] = self.sign_na_id        # same for held-out split edges
input_ids[disallowed_edges] = self.mask_id          # identity masked too (moot, see below)
attention_mask[disallowed_edges] = 0                # excluded from attention as a key
```

Order matters: `labels` must be read out of `sign_ids` *before* `sign_ids` gets
overwritten with the hidden-class value, or the true label would already be gone by the
time it's captured. `input_ids` is never touched at `target_edges` — that omission,
compared to production's `input_ids[target_edges] = mask_id`, is the single behavioral
difference this whole experiment is built around.

(Disallowed edges get their identity masked too, even though it's provably harmless
either way since they're already excluded from attention as keys — done purely to stay
consistent with the "hidden edges render as `<MASK>`" convention where there's no
downside to it.)

**Verification actually run before training** (not just reasoning about the code):
checked 2,000 sampled walks and confirmed, for every position: target positions always
had `sign_ids==2` and `input_ids == old_vocab_size + edge_id` (never `<MASK>`);
disallowed positions always had both `input_ids==<MASK>` and `sign_ids==2`; every
position with a real 0/1 sign always had `input_ids >= old_vocab_size` (a genuine edge
token, never a vertex token accidentally carrying a sign class). All passed.

---

## 4. The model — sign as a third additive embedding, exactly like position already is

`EdgeIdentityTransformerModel` (`model.py`) is a near-identical copy of the production
`TransformerModel`'s forward path, plus one new small embedding table:

```python
self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)      # 27974 x 64
self.sign_embedding = nn.Embedding(3, embedding_dim)   # 3 x 64 -- 0=neg, 1=pos, 2=n/a-or-hidden
...
self.register_buffer("pos_encoder", get_sinusoidal_encoding(max_length, embedding_dim))
```

```python
def forward(self, input_ids, sign_ids, attention_mask=None):
    x = self.embed(input_ids) + self.sign_embedding(sign_ids)
    x = x + self.pos_encoder[: input_ids.size(1)]
    ...
```

At every one of the 39 positions in walk 0, three independent lookups get summed into
one vector before the Transformer ever runs: **content** (which specific edge, or which
vertex — a row out of the 27,974-row table), **sign** (positive / negative /
not-applicable-or-hidden — a row out of the 3-row table), and **position** (which slot
in the walk, 0-38 — a row out of a fixed sinusoidal table, never learned). At a vertex
position, `sign_embedding(2)` is added uniformly to *every* vertex token in the corpus;
nothing stops the model from learning that row as ≈0 if it finds no use for it there,
same as an unused positional slot would.

Everything past this point — the local-attention windowing (`±2` hops, unchanged),
the classification head — is an unmodified copy of production's forward path. The
`+ self.sign_embedding(sign_ids)` line is the only structural addition.

---

## 5. Training loop (`train_pilot.py`)

Self-contained plain-PyTorch loop, **not** routed through production's
`LitEdgeClassifier` — this pilot doesn't need dynamic resplit, hardness reweighting, or
any of that Lightning machinery, and a standalone loop is much easier to verify correct
on a first pass at a genuinely new tokenization scheme. Same hyperparameters as
`configs/bitcoin-alpha.yaml` (embedding_dim=64, hidden_dim=128, nhead=4, nlayers=3,
dropout=0.20, lr=0.0016, batch_size=1024, local_attention_window=4, class-weighted
cross-entropy, early stopping on val walk-AUC, patience 15, min-delta 0.001) — same
architecture size and optimization regime as production, so any delta is attributable
to the tokenization/embedding scheme itself, not a hyperparameter confound.

## 6. Walk-level AUC vs. edge-level AUC — what the two numbers in each log line mean

Every epoch line prints **two** validation AUCs, e.g. `val_walk_auc=0.5935
val_edge_auc=0.6007`. They're computed from the same predictions, at two different
levels of pooling:

- **Walk-level AUC** (`val_walk_auc`): treats every individual masked-position
  prediction as its own independent data point for the ROC curve. If edge #17360 shows
  up as a target in 40 different sampled walks, each of those 40 occurrences
  contributes its own probability to this AUC — the *same* true label (edge 17360's
  real sign) paired with 40 different (correlated, but not identical) predictions.
- **Edge-level AUC** (`val_edge_auc`): first averages all of a given edge's
  walk-occurrence probabilities into one single number per edge (simple unweighted
  mean here — not production's fitted `func_logit_power` aggregator, which isn't
  needed to answer "does this tokenization change anything"), *then* computes AUC over
  one row per distinct edge.

```python
def edge_level_mean_auc(edge_ids, probs, targets):
    order = np.argsort(edge_ids, kind="stable")
    ...
    means = sums / cnts          # mean predicted prob per distinct edge_id
    labels = y_s[first_idx]      # that edge's one true label
    return roc_auc_score(labels, means)
```

**Concretely**: if edge 17360 (true label = positive) is predicted 0.55, 0.61, 0.58 in
its three walk-occurrences, walk-level AUC sees those as three separate rows (0.55, 0.61,
0.58, each paired with label=positive); edge-level AUC collapses them into one row
(mean=0.58, label=positive) before scoring. Edge-level AUC is what production's Table 1
numbers are actually reporting (via the fitted aggregator); walk-level AUC is a cheaper
diagnostic that doesn't require grouping, and the two numbers track each other closely
but aren't identical — averaging over multiple noisy occurrences of the same edge
generally denoises the prediction a bit, which is usually (not guaranteed) to nudge
edge-level AUC slightly above walk-level, matching the pattern seen in every epoch
logged so far (edge-level consistently ~0.5-1pp above walk-level).

---

## 7. Edge-identity replacement — the regularizer under consideration (matches production's node-replacement "R")

Not yet run as of this writing — added to `train_pilot.py` behind a `--edge-replace-prob`
flag (default 0.0/off) after the first, unregularized run showed a memorization pattern
(train loss collapsing toward 0 within 1-2 epochs while val AUC fell and never
recovered — consistent with the model exploiting individual edge-identity embeddings to
memorize per-edge training behavior that doesn't generalize to held-out edges, whose
identity rows start random and are never trained).

Production already has exactly this kind of regularizer for *vertex* identity
(`node_context_mode=replace`, `node_replace_prob=0.2` — see `src/model/lit_model.py`'s
`_maybe_apply_node_replacement`): during training only, each visible vertex token is,
independently with probability 0.2, either swapped for `<UNK>` (70% of the time) or
swapped for a *different*, randomly-drawn vertex token from elsewhere in the same batch
(30% of the time). This forces the model to not over-rely on any one vertex's specific
identity embedding.

`apply_edge_identity_replacement()` in `train_pilot.py` is the direct edge-token
analogue, same two-branch structure:

```python
candidates = x >= old_vocab_size                 # real edge-identity tokens only, not <MASK>/vertices
replace_mask = (torch.rand_like(...) < replace_prob) & candidates
edge_pool = x[candidates]                          # every visible edge id in this batch
use_unk = torch.rand(...) < unk_ratio
x[unk_positions] = unk_id                          # 70% (default): replace with shared <UNK>
x[rand_positions] = edge_pool[rand_idx]            # 30% (default): replace with a random OTHER edge's id
```

Crucially, this only touches `input_ids` (which specific edge is shown) — `sign_ids` is
untouched, so a context edge's true sign remains correctly visible even when its
identity token gets scrambled. That's the intended effect: push the model toward
relying on sign + structural position rather than "I've memorized that edge #17360
specifically tends to be positive." Applied uniformly to every visible edge-identity
position (both ordinary context edges and target edges, whose identity is visible even
though their sign is hidden) — same scope as production's node replacement, which
similarly doesn't distinguish vertex roles.
