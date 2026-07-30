# Masking in the walk-to-paint Transformer

Three independent masking concerns exist in this model, and they get combined
differently depending on whether `local_attention_window` is set. This doc
covers all of it: what each mask means, how PyTorch's API expects masks to be
passed, what actually broke (a real NaN bug, now fixed), and the final design.

## The three masking concerns

1. **Local window** (`local_attention_window`, e.g. `4`) — geometric: token `i`
   may only attend to token `j` if `|i-j| <= window`. Same rule for the whole
   batch. Only exists when local attention is enabled; full attention has no
   window restriction at all.
2. **Padding** — walks in a batch have different real lengths (ragged, see
   `src/data/stage_dataset.py`'s `ragged_collate_fn`); shorter walks get padded
   to the batch's max length. Padding positions must never be attended to as
   keys.
3. **Disallowed-split edges** — `StageViewDataset.__getitem__`
   (`src/data/stage_dataset.py`) hides edges from splits not yet visible at the
   current stage (e.g. val/test edges during training): their `input_ids` get
   replaced with `[MASK]` **and** their `attention_mask` gets zeroed, so nobody
   can attend to them as a key either. This is the actual anti-leakage
   mechanism — masking (2) and (3) both start life as zeros in the same
   `attention_mask` tensor the dataset produces; the model never distinguishes
   "padding" from "disallowed" downstream, because both mean the same thing to
   attention: *invisible*.

## PyTorch's two mask argument slots

`nn.TransformerEncoder`/`nn.TransformerEncoderLayer.forward()` has exactly two
mask parameters, by API design:

- **`mask`** (`src_mask`/`attn_mask`) — which *(query, key)* position pairs may
  interact. Historically structural/shared across the whole batch, shape
  `(L, L)`.
- **`src_key_padding_mask`** — which *key positions*, per batch item, must
  always be excluded. Data-dependent, shape `(B, L)`.

We have three concerns but only two slots. Concerns (2) and (3) were already
combined into one tensor before any recent change (both are just zeros in
`attention_mask`, indistinguishable). What changed in this session is whether
concern (1) also gets folded into that same tensor, and *when* that folding
happens.

```
                    mask=                            src_key_padding_mask=
FULL attention:     None (no window restriction)     padding OR disallowed
LOCAL attention:    window OR padding OR disallowed   None (already folded in)
```

Full attention still uses the two slots the textbook way. Local attention
folds everything into the first slot and leaves the second empty — see below
for why.

## The NaN bug (found and fixed this session)

**Root cause.** Local attention's narrow window means a padding/disallowed
query position can end up with *every* key in its window also
padding/disallowed — an all `-inf` mask row. `softmax(-inf,...,-inf) = NaN`,
genuinely computed. That NaN then poisons later layers even at *real* token
positions, because `0.0 * NaN = NaN` in IEEE float: a masked-out key with
attention weight exactly 0 still corrupts a weighted sum if its value vector is
NaN. Full attention never hits this — its unbounded window always has some
real token to attend to. This is local-attention-specific.

**Why it wasn't caught earlier.** `nn.TransformerEncoderLayer.forward` has an
internal "sparsity fast path" that activates whenever the layer is in eval mode
(`self.training is False`) and no forward hooks are attached anywhere in the
module tree — the exact condition of every real validation/test loop. That
fast path calls `torch._transformer_encoder_layer_fwd` (a fused CUDA kernel)
directly and never calls our custom `_sa_block` override — it doesn't check
whether `_sa_block` has been subclassed. Confirmed via a no-hook test that
exactly mirrors real production calls: **86.11%** of supervised test positions
came back NaN (epinions, `E16_NOHARD_KCOVER_K5_NW2000000_local` checkpoint,
epoch 29), even after `_sa_block` itself had already been fixed — because
`_sa_block` was never being called at all in that mode.

Training itself was never affected — `self.training=True` unconditionally
disables that fast path — so gradients/weights learned by existing LocalAttn4
checkpoints are fine. Only *eval-mode* metrics (val/test AUC logged during
those runs) went through the buggy path.

### Runs whose logged eval/test AUC may need re-verification

Every run below used `model.local_attention_window` set (LocalAttn4) and would
have computed its logged val/test AUC via `model.eval()` with no forward hooks
— the exact buggy condition. Weights/checkpoints themselves are unaffected
(training never took the buggy path); only the AUC numbers logged *during*
these runs are suspect. Re-evaluating a checkpoint under the current code is
cheap (inference only, no retraining) — until that's done, treat every number
below as unverified, including the LocalAttn4 column of `CLAUDE.md`'s SOTA
table:

- **E14_HARDNODE_L10_LOCALATTN4** (2026-06-15) — all 6 datasets
  (`outputs/<dataset>/E14_HARDNODE_L10_LOCALATTN4_*/`)
- **E15_SWEEP_k5_*_local** (budget sweep) — all 6 datasets
  (`outputs/<dataset>/E15_SWEEP_k5_nw*_local/`)
- **E16_NOHARD_KCOVER_K5_*_local** (2026-07-06/07) — all 6 datasets
  (`outputs/<dataset>/E16_NOHARD_KCOVER_K5_NW*_local_*/`) — includes the
  epinions epoch-29 checkpoint used to confirm this bug/fix.
- **E17_HARDNODE_KCOVER_REMINE_LOCALATTN4** (2026-07-07) — all 6 datasets
  (`outputs/<dataset>/E17_HARDNODE_KCOVER_REMINE_LOCALATTN4_*/`)
- **E18_HARDNODE_DYNPOOL_LOCALATTN4** (2026-07-07) — all 6 datasets
  (`outputs/<dataset>/E18_HARDNODE_DYNPOOL_LOCALATTN4_*/`)

Found via `find outputs -iname "*local*" -type d`. Re-running eval (not
retraining) for all of these is a separate, explicit decision — not yet
started, not yet approved.

**Update (2026-07-19): the tooling that would have blocked a clean re-eval is
now fixed, but a second bug meant nobody could have done this safely even if
they'd tried.** Investigating the LocalAttn4 hardness ablation turned up that
`run_posthoc.py` never restored `model.local_attention_window` from the
checkpoint either — it rebuilt `cfg` from `configs/<ds>.yaml` + CLI overrides
alone, so re-evaluating any of the runs above without manually re-passing
`model.local_attention_window=4` would have silently reconstructed a
full-attention model, loaded the LocalAttn4 weights into it anyway (shapes
match), and produced confidently-wrong numbers with no error. Fixed in
`run_posthoc.py` — it now reads the checkpoint's own saved cfg (`LitEdgeClassifier.save_hyperparameters()`
already stored it in full) and uses that as the base for both the data
pipeline and the model, so `dataset.*` and `model.*` both come back correct
automatically. Verified against the exact epinions/epoch-29 checkpoint named
above: posted with only `dataset.name=epinions`, it now correctly recovers
`local_attention_window=4` and `k_cover`/`nw2000000`, giving edge test
AUC=0.9570 with 0% NaN — matching this doc's independently-verified reference
number. Full story: `CLAUDE.md`'s "Posthoc aggregation" section.

Re-running eval on the runs listed above is still a separate, explicit
decision (not started) — but it's no longer blocked by broken tooling, just
by whether it's worth doing before the E27/E28 LocalAttn4 ablation (fresh
training runs on the current `edge_cover` sampler, superseding these anyway)
lands.

## Final design

**1. `forward()` always takes the eager path.** Overridden to unconditionally
run the manual residual/LayerNorm/FFN pipeline (bypassing PyTorch's fused
fast-path kernel entirely, regardless of train/eval mode or hooks) — the only
way to guarantee our own masking logic runs on every call.

**2. Merge once, not once per layer.** `TransformerModel.forward` builds the
fully-combined mask (window `|` padding-or-disallowed) as a single bool tensor,
once per forward pass, before calling `self.transformer(...)`. Verified in
PyTorch's source (`torch/nn/modules/transformer.py` lines 411/419, before the
per-layer loop at line 513) that `F._canonical_mask` — which converts bool
masks to float `{-inf, 0}` for actual use — runs exactly once at the encoder
level regardless of `nlayers`; the previous design rebuilt and re-merged the
mask inside `_sa_block`, i.e. once per layer (5x), which was real, avoidable,
repeated work.

**3. Numerical safety is provable, not incidental.** After computing attention,
`_sa_block` explicitly zeroes the output at every row PyTorch's own merged mask
marks as fully `-inf` (`torch.isneginf(mask).all(dim=-1)`) — rather than
relying on whichever SDPA backend happens to get selected not producing NaN
there. `masked_fill` is a direct overwrite, not an arithmetic operation, so it
doesn't inherit whatever NaN softmax already produced at that row; this breaks
the `0.0 * NaN = NaN` propagation chain before any later layer can read it.
Safe unconditionally: only padding/disallowed positions can ever be fully
masked (real positions always keep their own diagonal — window always
includes distance 0, and a real position's key is never excluded), and nothing
downstream ever reads a padding/disallowed position's value (no real query
attends to it as a key, in any layer; it's never a supervised/label position).

**Why the custom class still has to exist.** Merging into one mask doesn't let
us fall back to stock `nn.MultiheadAttention`/`nn.TransformerEncoderLayer`.
Verified via `F.multi_head_attention_forward`
(`torch/nn/functional.py:6259-6277`): it only accepts a 2D mask (broadcasts
across the whole batch — fine with no padding at all) or a 3D mask that must
be *exactly* `(bsz*num_heads, L, L)`, no broadcast across the head dimension.
Our mask is per-batch-item essentially every batch, so the standard API would
force materializing it to `(B*8, L, L)` — the original 8x-bigger, 31x-slower
path the project already measured and moved away from (see
`~/.claude/plans/plan-performance.md`). `F.scaled_dot_product_attention` (what
`_sa_block` calls directly) broadcasts a `(B,1,L,L)` mask across all heads for
free — that gap between the two APIs, not "how many masks," is why the custom
class exists.

## Verification (`scripts/test_local_attention_masking.py`)

1. **Synthetic correctness** — small hand-built batch with known padding, known
   disallowed positions, known window. Confirms the attention weight matrix
   (computed independently, not via SDPA) is nonzero exactly where
   `(in-window AND not-padding AND not-disallowed)` for every real query
   position, and that the real layer's output matches an independent reference
   computation exactly at every real position. Also confirms NaN really is
   produced at fully-masked rows in the naive reference — documenting the
   mechanism this design guards against.
2. **`eval()` vs. `train()`-mode equivalence** — runs the layer once via
   `.eval()` and once via `.train()` with every dropout probability forced to
   `0.0` (isolating whether `self.training` itself, not dropout noise, changes
   anything). Output is bit-identical (max diff `0.0`). `self.training`
   appears exactly once in the whole class (the `dropout_p` ternary,
   grep-verified) — there is no second, hidden code path that could silently
   diverge between train and eval, since `forward()` is overridden
   unconditionally and never touches PyTorch's internal fast-path dispatch at
   all, hook or no hook.
3. **Shape/config sweep** — `nhead=1` (no head-broadcast needed), `nhead=3`
   (odd — PyTorch's own internal fast path explicitly disqualifies odd head
   counts, irrelevant here since we never take that path), `bsz=1`, and
   `window=0` (attend to self only). All pass with zero NaN and correct output
   shape, confirming the broadcasting isn't accidentally tuned to only work at
   the production shape (`nhead=8`, `head_dim=4`).
4. **No-NaN on real data, real production code path** — no forward hooks,
   `model.eval()`, plain `model(input_ids, attention_mask=...)` calls (exactly
   how real training/eval invokes the model). Result: 0% NaN on the epinions
   test set (2,665,968 supervised positions), AUC 0.956979.
5. **Overhead benchmark vs. full attention** (embedding_dim=32, nhead=8,
   nlayers=5, seq_len=161, batch=1024, L40S, forward+backward, training mode —
   matching `plan-performance.md`'s shape):

   | Variant | ms/iter | peak memory |
   |---|---|---|
   | Full attention | 117.8 | 1669 MB |
   | Local, prior (per-layer merge) | 113.5 | 2071 MB |
   | Local, current (merge-once) | 111.4 | 2184 MB |

   Both local variants are now slightly *faster* than full attention in
   training mode — full attention's eval-only fast paths (nested-tensor
   padding removal, fused per-layer kernel) are unconditionally disabled
   whenever a backward pass is needed, i.e. always during real training,
   so its "free" advantage never materializes there. Merge-once is ~2%
   faster in wall-clock than the per-layer design but uses slightly *more*
   peak memory (2184 vs 2071 MB) — likely because the one shared mask
   tensor must stay alive for the full backward pass across all 5 layers,
   whereas 5 separate per-layer copies each free immediately after their
   own layer's backward. Both numbers are a modest, roughly-a-wash
   difference — the real win of merging once is code simplicity and
   provable (not incidental) correctness, not a clear memory or speed
   victory. Both are far from the original unfixed implementation's
   ~37% slower / ~4x memory regression (`plan-performance.md`).

## Historical note (superseded)

An earlier fix attempt (2026-07-17) diagnosed the mask combination as a
"bool+bool=int64" bug. That diagnosis was wrong — `F._canonical_mask` converts
both masks to float `{-inf, 0}` before any custom layer code ever runs, so the
combination was always float+float, which is correct. The real, then-unfixed
bug was the fast-path bypass described above; the per-layer merge redundancy
was fixed by this session's later revision (merge-once).
