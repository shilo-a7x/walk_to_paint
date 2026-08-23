# SIGNSCRAMBLE ablation — mechanism, in full

## Why this ablation exists

`mask_edge_tokens` (an existing ablation, replaces every edge-sign token with `<MASK>`) showed
no statistically significant AUC effect on any of the 6 datasets once its posthoc-eval bug was
fixed (see `PAPER_CLOSEOUT_LOG.md`, `ABLATION_MASKNODE_MASKEDGE_TRUE_RESULTS.md`). That's a
real, verified finding — but it conflates two different questions:

1. Does an edge token being **present at all** at a context position matter (a purely
   structural/positional signal — "there is an edge here")?
2. Does that edge token's **specific sign value** matter (is the model actually reading and
   using whether it says positive or negative)?

`<MASK>`-ing removes both at once. SIGNSCRAMBLE isolates question 2 alone: it keeps a real
sign token in every context position — the model still sees "there's an edge here, and it has
*a* sign" — but for a fixed, precomputed random subset of edges, the sign shown is deliberately
wrong (decorrelated from the true sign). If AUC still doesn't drop, that's a much stronger
claim than `mask_edge_tokens` alone could support: it means the model isn't reading edge-sign
values at all, not just that removing the token entirely doesn't matter.

## What gets changed, and what doesn't

A walk is a sequence of alternating vertex and edge tokens. At every position, before the
forward pass, the model already classifies each token position into one of three states
(pre-existing logic, unrelated to this ablation):

1. **Masked prediction target** — the edge whose sign is being predicted this batch. Its sign
   token is already replaced with `<MASK>` before this ablation ever runs (a static replacement
   for non-dynamic masking, or a per-epoch dynamic one under the `dynamic_train_masking`
   default — see below).
2. **Split-excluded context** — an edge from a later split (e.g. a test edge, visible only as
   context during training) that "Preventing held-out sign leakage" already replaces with
   `<MASK>` and excludes from attention entirely.
3. **Visible context** — everything else: a real edge, showing its real sign, that the model is
   allowed to read as context for predicting some other (masked) edge.

**SIGNSCRAMBLE only ever touches category 3.** Categories 1 and 2 are already `<MASK>` tokens
by the time this ablation's code runs, and the implementation detects "visible" as exactly
`edge position AND token != <MASK>` — so masked targets and split-excluded edges are excluded
*by construction*, not by any extra bookkeeping this ablation has to get right independently.

For every edge in category 3, its shown sign is looked up in a fixed table:

> **Each edge id gets one independent coin flip, drawn once, ever.** If the flip says "keep,"
> the edge shows its real, true sign (same as the unablated model). If the flip says "scramble,"
> the edge shows the *opposite* sign token — not `<MASK>`, not a random re-draw each time, the
> literal complementary class token (positive↔negative for this binary-sign setting).

Roughly half of all edges get flipped (a fair coin, ~50%), which is deliberate: on average, a
scrambled edge's shown sign carries **zero mutual information** with its true sign (P(shown
sign matches true sign) = 0.5, independent of what the true sign actually is) — the cleanest
possible "this signal is now pure noise" construction, while every individual token position
still looks exactly like a normal, well-formed input to the model.

## Why "fixed per edge, drawn once" and not something else

This is the one design choice that took real thought, because it has to interact correctly with
three other things already happening in this pipeline: **dynamic resplit** (which edges are
masked targets changes every epoch), **the fact that one edge appears in many different walks**
(hundreds to thousands of occurrences per edge on some datasets), and **needing train and eval
to be consistent with each other** (the same ablation must apply identically at posthoc-eval
time as it did during training, or the numbers are meaningless — this project has already hit
that exact bug once, with `mask_node_tokens`/`mask_edge_tokens`, see the postmortem doc).

**The chosen design: draw one Bernoulli(0.5) flip bit per edge id, once, at the start of
training (seeded, reproducible), and reuse it forever after — for every occurrence of that edge
in every walk, at every epoch, in train/val/test alike.** Concretely:

- A real edge has exactly one sign in the real graph. A "fixed per edge" scrambled sign
  preserves that property — an edge just now has a *wrong* fixed sign instead of its *true*
  fixed sign. This means the ablation corrupts sign **correctness** without also destroying
  sign **consistency** — a genuinely different (and, we think, cleaner) corruption than making
  the shown sign flicker randomly on every occurrence.
- **Dynamic resplit compatibility comes for free, with no special-casing needed.** Dynamic
  resplit decides, fresh each epoch, which edges in the train+mask pool are this epoch's
  supervised prediction targets; those get `<MASK>`'d for that epoch specifically. Since
  SIGNSCRAMBLE's "is this position visible" check runs *after* that epoch's dynamic masking has
  already been applied to the input, an edge that becomes this epoch's target is already
  `<MASK>` by the time SIGNSCRAMBLE looks at it — so it's correctly skipped, automatically, no
  extra logic required to keep the two mechanisms from stepping on each other.
- **Same edge, many walks, always the same shown sign.** Because the flip is keyed by edge id
  (not by walk, not by occurrence), every one of an edge's hundreds of occurrences across
  different sampled walks shows the identical (correct-or-flipped) sign. This mirrors how a real
  edge actually behaves — its sign doesn't change depending on which walk happened to sample it
  — so the model is being asked "can you tell a consistently-wrong label from a
  consistently-right one," not "can you learn anything from a signal that's pure noise on every
  single presentation."

## Alternatives considered and rejected

- **Per-occurrence random** (redraw the flip independently every time the edge appears, even
  within the same epoch): destroys consistency as well as correctness, conflating two different
  questions into one experiment. Also adds pure training noise (the model can never learn *any*
  stable input↔output mapping for that edge's context token), which risks explaining an AUC drop
  by "we broke optimization" rather than "the model doesn't use sign values" — a much weaker,
  more confoundable result.
- **Per-epoch random** (redraw once per epoch, fixed within that epoch): a middle ground with no
  clear scientific benefit over the fixed-forever version, and it adds real complexity keeping
  its own re-draw schedule in sync with dynamic resplit's independent per-epoch schedule, for
  no obvious gain in what it would tell us.
- **Eval-only corruption of an already-trained (unablated) checkpoint**, no retraining: cheap
  (minutes, reuses existing checkpoints) but answers a different question — "does a normally
  trained model degrade if you corrupt its input after the fact" rather than "can the model
  learn to use sign context correctly when trained under corruption from the start." Not chosen
  as the primary ablation (the paper's other three ablations are all train+eval), but a
  reasonable fast cross-check if a quick sanity read is ever wanted.

## Exact implementation

- `model.scramble_edge_signs` (bool, default `false`) — the config flag, same naming style as
  `mask_node_tokens`/`mask_edge_tokens`/`randomize_walk_direction`.
- `src/model/lit_model.py`:
  - `LitEdgeClassifier._build_sign_scramble_tables(dataset=None)` — builds two lookup tables
    once: (1) `class2tokid`, sign class (0/1) → its token id, by inverting the dataset's
    `id2class` map; (2) the flip table itself, one `torch.rand(...) < 0.5` draw per edge id,
    seeded as `cfg.reproducibility.seed + 2_718_281` (a fixed offset distinct from every other
    seeded stream in the pipeline — dynamic-mask epoch sampling, DIRFLIP's `walk_flip` — so none
    of them silently share or collide on the same RNG stream).
  - `LitEdgeClassifier._maybe_apply_sign_scramble(input_ids, edge_mask, edge_ids, edge_classes,
    dataset=None)` — the actual substitution: computes `visible = edge_mask & (input_ids !=
    mask_id)`, looks up each visible position's edge id in the flip table, and for the ones that
    say "scramble," replaces the token with the opposite class's token id (via `edge_classes`,
    which already holds each position's true class from before any masking was ever applied —
    so this is correct regardless of what `input_ids` currently shows at that position).
  - Wired into `_step` right after `_maybe_apply_token_masking`, so it runs at every train/val/
    test forward pass alike — same "applies at both train and eval time" convention as the other
    token-masking ablations.
- `src/training/callbacks.py`: `PerEpochPredictionSaver._extract_predictions` (the posthoc-eval
  path, which does **not** go through `_step`) reapplies the same call explicitly, passing the
  dataloader's own dataset in for the lazy table build — built in from the start this time,
  specifically to avoid repeating the exact bug class that hit `mask_node_tokens`/
  `mask_edge_tokens` (training correct, evaluation blind to the ablation).
- Two real bugs found and fixed during smoke-testing, before the real campaign launched (see
  `PAPER_CLOSEOUT_LOG.md` for the full narrative):
  1. PyTorch Lightning's pre-training sanity-check validation pass runs *before*
     `train_dataloader` is attached to the trainer, so a lazy table-build triggered by that pass
     crashed. Fixed: skip scrambling (not crash) specifically during `trainer.sanity_checking`,
     since that pass's metrics are discarded anyway.
  2. `run_posthoc.py`'s standalone path never attaches a real PyTorch Lightning `Trainer` to the
     model, and `LightningModule.trainer` raises `RuntimeError` (not `AttributeError`) when
     nothing is attached — so `getattr(self, "trainer", None)` doesn't safely degrade the way it
     would for a merely-unset attribute. Fixed by having the posthoc callback pass its own
     dataset in explicitly, bypassing `self.trainer` for that call path entirely.
- Verified via 6 synthetic unit tests (determinism across rebuilds with the same seed, a
  different seed giving a different table, masked/split-excluded positions never touched,
  visible positions showing exactly the true-or-flipped class as the table dictates, the same
  edge showing an identical sign across multiple occurrences, and a ~50% flip rate over 2000
  synthetic edges) — all passed — then a real 3-epoch smoke run on Wiki-elec (clean, no NaN/
  crash, sane non-degenerate AUC), then a `run_posthoc.py` pass on that same checkpoint, which
  reproduced the trainer's own `test_auc_epoch` exactly (0.7163 both ways) — confirming
  posthoc-eval and training-time evaluation are consistent before launching the full 60-job
  campaign.

## Where it's launched from

`scripts/run_signscramble_campaign.py` — 6 datasets × 10 seeds (42–51) = 60 train+posthoc jobs,
4-GPU queue-worker pattern, local attention only (LocalAttn4, the production default), posthoc
computes `func_logit_power` only. Logs: `logs/signscramble_campaign/`.
