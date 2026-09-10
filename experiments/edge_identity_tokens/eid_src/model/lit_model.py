"""Edge-identity-token variant of the production LitEdgeClassifier.

Subclasses (not copies) src/model/lit_model.py's LitEdgeClassifier directly, so
every piece of the real, known-good training machinery -- the AdamW + CosineAnnealingLR
schedule, class-weighted cross-entropy, metrics/ROC/confusion-matrix logging, node-token
replacement regularization (R), PL's own checkpointing/early-stopping integration -- is
the ACTUAL production code, not a hand-rolled reimplementation. This file changes only
what genuinely has to change for the new tokenization scheme:

  1. `self.model` is the EdgeIdentityTransformerModel, not TransformerModel.
  2. `_step` fetches `sign_ids` from batch metadata and threads it into the model's
     forward call (the production model doesn't take a sign_ids argument at all).
  3. A new training-only regularizer, `_maybe_apply_edge_identity_replacement`, the
     edge-identity analogue of production's `_maybe_apply_node_replacement` (see its
     docstring for the exact rationale) -- added because the pilot's first,
     hand-rolled-loop run showed a memorization pattern (train loss -> ~0 within 1-2
     epochs while val AUC fell and never recovered), consistent with the model
     exploiting per-edge identity embeddings to memorize training-set behavior that
     can't generalize to held-out edges. Off by default (edge_replace_prob=0.0);
     enable via `model.edge_replace_prob=0.2` (matching production's
     node_replace_prob default) on the CLI.
  4. `dynamic_train_masking` IS now supported (fixed -- see below), via a sign-only-
     hide override of `_build_dynamic_targets_for_batch` that touches `sign_ids`
     instead of `input_ids`.

Sign-only-hide dynamic resplit (fixed; previously a documented, deliberate gap):
production's dynamic resplit hides a target by overwriting `input_ids` with
`<MASK>` (`LitEdgeClassifier._build_dynamic_targets_for_batch`), which would hide
IDENTITY too and defeat this experiment's whole point. This subclass overrides
that method to touch `sign_ids` only (see below) -- `EdgeIdentityStageViewDataset`
was previously masking every occurrence of the (fixed, static) MASK split's sign
every epoch, meaning ~48% of the train+mask pool (the TRAIN split) never got a
direct supervised-target gradient on its identity row at all, only indirect
gradient via being read as context. Dynamic resplit fixes this by rotating which
edges are this epoch's targets, same as production.

Ablation-campaign flags, EID-adapted (added 2026-09-08, alongside the
eid_reveal_holdout_identity investigation) -- production's `mask_edge_tokens` and
`scramble_edge_signs` (`_maybe_apply_token_masking`/`_maybe_apply_sign_scramble`,
inherited unmodified) only ever touch `input_ids`, since production's edge token
*is* the sign -- there is no separate channel for them to miss. EID's `sign_ids` is
exactly that separate channel, invisible to both inherited methods, so naively
flipping either flag on an EID run would silently fail to do what it claims (e.g.
`mask_edge_tokens` would hide identity but leave the true sign fully visible via
`sign_ids`, untouched by the inherited method -- not an ablation of "all edge
signal" at all). Three EID-native overrides/additions instead:

  - `model.mask_edge_tokens`: inherited `_maybe_apply_token_masking` still masks
    `input_ids` at edge positions; EID additionally blanks `sign_ids` at those same
    positions (`_maybe_apply_edge_sign_masking`, new), so the ablation actually
    removes ALL edge-channel information (identity AND sign), matching the
    original's intent ("isolate how much signal vertex identity alone carries").
    Applies at train+eval, same as the inherited node/edge masking.
  - `model.scramble_edge_signs`: **now supported** (previously raised
    NotImplementedError at construction -- that guard is removed). Reimplemented
    from scratch (`_build_eid_sign_scramble_table`/`_maybe_apply_eid_sign_scramble`)
    rather than reusing production's `_build_sign_scramble_tables`/
    `_maybe_apply_sign_scramble`, which assume exactly one token id per class
    (production's 2-token scheme) -- EID has thousands of distinct edge tokens per
    class, so that inversion would silently pick one arbitrary edge to represent
    "the positive class" globally, confirmed by reading the actual inversion loop.
    EID's version is simpler: flips `sign_ids` (not `input_ids`) at every visible
    context position to the wrong class, using a fixed per-edge-id Bernoulli flip
    table (same "decided once per edge, not per occurrence" principle as
    production's, so a given edge shows the same scrambled sign everywhere it's
    visible) -- `input_ids`/identity is completely untouched.
  - `model.scramble_edge_identity` (new, no production analogue): the mirror
    ablation -- flips `input_ids` (not `sign_ids`) at every visible context
    position to a different, fixed-per-edge-id substitute edge's identity token,
    leaving the true sign fully visible. Where `scramble_edge_signs` asks "does the
    model get misled by a wrong sign," this asks "does the model get misled by a
    wrong identity" -- i.e. is identity being used for edge-specific information, or
    just as generic evidence that some edge is here. Directly targets the question
    this whole investigation is about, so added rather than left as a gap.
    `_build_eid_identity_scramble_table`/`_maybe_apply_eid_identity_scramble`.

All three are diagnostic ablations (train+eval, not training-only regularizers),
matching `mask_edge_tokens`/`scramble_edge_signs`'s existing convention -- distinct
from `_maybe_apply_edge_identity_replacement` above, which is a training-only
regularizer with per-step random corruption, not a fixed per-edge mapping.

Everything else (`on_train_epoch_end`/`on_validation_epoch_end`/`on_test_epoch_end`,
`training_step`/`validation_step`/`test_step`, hardness-reweighting plumbing -- unused
here, gated off by default same as production, `_maybe_apply_node_replacement` for
VERTEX tokens, `_maybe_apply_token_masking`) is inherited completely unmodified.

  4. `configure_optimizers` is overridden to optionally put the low-rank edge-identity
     table (`self.model.edge_embed_low`, only exists when `edge_embed_rank>0`) in its
     own AdamW param group with a separate weight_decay
     (`model.edge_embed_weight_decay`, default: fall through to the same
     `training.weight_decay` every other param uses -- i.e. this is a no-op unless
     explicitly set). This is a direct capacity-control regularizer (shrinks the
     embedding vectors themselves via L2 penalty), a different mechanism from
     `edge_replace_prob`'s train-time identity corruption -- the two are complementary
     the same way `edge_embed_rank` (capacity) and `edge_replace_prob` (incentive) are
     complementary per this file's module docstring above.
"""

import torch

from src.model.lit_model import LitEdgeClassifier, SPLIT_TRAIN, SPLIT_MASK
from experiments.edge_identity_tokens.eid_src.model.model import EdgeIdentityTransformerModel
from experiments.edge_identity_tokens.eid_src.data.stage_dataset import SIGN_NA


class EIDLitEdgeClassifier(LitEdgeClassifier):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        # Swap in the sign-embedding-aware model. super().__init__ already built
        # self.model = TransformerModel(cfg) once; replace it rather than trying to
        # intercept construction, since LitEdgeClassifier.__init__ does several other
        # required things first (save_hyperparameters, class_weights, metrics_manager,
        # dynamic/sign-scramble/hardness state) that this subclass still needs.
        self.model = EdgeIdentityTransformerModel(self.cfg)
        self._eid_sign_scramble_ready = False
        self._eid_identity_scramble_ready = False

    def forward(self, input_ids, sign_ids):
        return self.model(input_ids, sign_ids)

    def configure_optimizers(self):
        edge_wd = getattr(self.cfg.model, "edge_embed_weight_decay", None)
        edge_table = getattr(self.model, "edge_embed_low", None)
        if edge_wd is None or edge_table is None:
            return super().configure_optimizers()

        edge_param_ids = {id(p) for p in edge_table.parameters()}
        edge_params = [p for p in edge_table.parameters()]
        rest_params = [p for p in self.parameters() if id(p) not in edge_param_ids]

        optimizer = torch.optim.AdamW(
            [
                {"params": rest_params, "weight_decay": self.cfg.training.weight_decay},
                {"params": edge_params, "weight_decay": float(edge_wd)},
            ],
            lr=self.cfg.training.lr,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.training.epochs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def _maybe_apply_edge_identity_replacement(self, input_ids, old_vocab_size):
        """Training-only regularizer: randomly corrupt VISIBLE edge-identity tokens.

        Direct analogue of the inherited `_maybe_apply_node_replacement`, for the
        edge-identity tokens minted by build_cache.py (ids >= old_vocab_size). A
        visible edge-identity position (real edge token, not <MASK>) is,
        independently with probability edge_replace_prob, replaced by either the
        shared <UNK> id (probability edge_replace_unk_ratio, default 0.7) or a
        different edge's identity token drawn from elsewhere in the same batch
        (probability 1 - edge_replace_unk_ratio). Only input_ids is touched -- the
        sign_ids channel is completely untouched by this function, so a context
        edge's true sign remains correctly visible even when its identity token gets
        scrambled. That's the intended effect: push the model toward relying on
        sign + structural position rather than "I've memorized that edge #17360
        specifically tends to be positive."
        """
        replace_prob = float(getattr(self.cfg.model, "edge_replace_prob", 0.0))
        if replace_prob <= 0.0:
            return input_ids

        unk_ratio = float(getattr(self.cfg.model, "edge_replace_unk_ratio", 0.7))
        unk_id = int(getattr(self.cfg.model, "unk_id", 2))

        x = input_ids.clone()
        candidates = x >= old_vocab_size
        if not candidates.any():
            return x

        replace_mask = (torch.rand_like(candidates, dtype=torch.float) < replace_prob) & candidates
        if not replace_mask.any():
            return x

        edge_pool = x[candidates]
        selected = replace_mask.nonzero(as_tuple=False)
        use_unk = torch.rand(selected.size(0), device=x.device) < unk_ratio

        if use_unk.any():
            unk_positions = selected[use_unk]
            x[unk_positions[:, 0], unk_positions[:, 1]] = unk_id

        rand_count = int((~use_unk).sum().item())
        if rand_count > 0:
            rand_positions = selected[~use_unk]
            rand_idx = torch.randint(0, edge_pool.numel(), (rand_count,), device=x.device)
            x[rand_positions[:, 0], rand_positions[:, 1]] = edge_pool[rand_idx]

        return x

    def _maybe_apply_edge_sign_masking(self, sign_ids, edge_mask):
        """EID companion to the inherited `_maybe_apply_token_masking`'s
        `mask_edge_tokens` handling: that method only blanks `input_ids` at edge
        positions (production has no other channel), so on its own it would hide
        identity while leaving the true sign fully visible via `sign_ids` --
        silently failing to ablate "all edge signal". This blanks `sign_ids` at the
        same positions, same flag, same train+eval scope."""
        if edge_mask is None or not bool(getattr(self.cfg.model, "mask_edge_tokens", False)):
            return sign_ids
        if not edge_mask.any():
            return sign_ids
        x = sign_ids.clone()
        x[edge_mask] = SIGN_NA
        return x

    def _maybe_apply_target_identity_masking(self, model_input_ids, edge_mask, labels):
        """New ablation (2026-09-08): the symmetric counterpart to
        _maybe_apply_context_edge_masking -- masks the TARGET's own identity
        (input_ids -> mask_id) instead of context. EID's dataset never masks this by
        construction (stage_dataset.py's DIFF 2 -- deliberately, to let the target
        behave like a node: identity always visible, only its sign hidden). This
        flag exists purely to reproduce, inside EID's architecture, production's own
        convention (identity and sign are the same token there, so the target's
        token -- being masked as the prediction target -- necessarily has zero
        identity information too) -- giving an apples-to-apples comparison point
        against production's real abl:maskedge finding (`no significant effect from
        removing edge tokens`, see PAPER_CLOSEOUT_LOG.md 2026-08-23) rather than the
        EID-native reveal_holdout_identity framing, which asks a different question.
        Sign is untouched -- it's already SIGN_NA at the target regardless."""
        if edge_mask is None or not bool(getattr(self.cfg.model, "mask_target_identity", False)):
            return model_input_ids
        target_mask = (labels != self.ignore_index) & edge_mask
        if not target_mask.any():
            return model_input_ids
        mask_id = int(getattr(self.cfg.model, "mask_id", 1))
        x = model_input_ids.clone()
        x[target_mask] = mask_id
        return x

    def _maybe_apply_context_edge_masking(self, model_input_ids, sign_ids, edge_mask, labels):
        """New ablation (2026-09-08): isolates whether CONTEXT edges (every edge
        occurrence that is NOT the current prediction target) contribute anything
        beyond node identity + topology -- the direct test of "do we use edge
        context, vs. nodes alone", uncontaminated by the target's own identity
        (which EID never masks, in any run -- see stage_dataset.py's DIFF 2) or by
        the reveal_holdout_identity/disallowed-edge question (a different, training-
        embedding-coverage question, orthogonal to this one).

        `labels != ignore_index` is the correctly-resolved target mask AFTER
        dynamic resplit has already run (this is called after that, in _step) --
        valid whether dynamic_train_masking is on or off, unlike a dataset-level
        target_edges tensor which is deliberately left empty at construction time
        for the dynamic-masking case (real targets are chosen per-epoch, later).
        Every edge position that is not currently a target -- context edges from
        the normal (allowed) split AND any disallowed val/test bystander alike --
        gets both channels blanked (mask_id / SIGN_NA), same train+eval scope as
        the other whole-token-role ablations. attention_mask is left untouched
        (still attendable) so this is a pure content ablation, not an
        attendability one -- mirrors mask_edge_tokens's own convention."""
        if edge_mask is None or not bool(getattr(self.cfg.model, "mask_context_edges", False)):
            return model_input_ids, sign_ids
        target_mask = labels != self.ignore_index
        context_mask = edge_mask & (~target_mask)
        if not context_mask.any():
            return model_input_ids, sign_ids
        mask_id = int(getattr(self.cfg.model, "mask_id", 1))
        x_ids = model_input_ids.clone()
        x_ids[context_mask] = mask_id
        x_sign = sign_ids.clone()
        x_sign[context_mask] = SIGN_NA
        return x_ids, x_sign

    def _build_eid_scramble_table(self, seed_offset: int, dataset=None):
        """Shared helper for both EID-native scramble ablations: a fixed,
        seeded-once mapping from real edge id -> real edge id, used as the
        substitute whenever that edge is scrambled. Guaranteed no fixed points
        (substitute always differs from the true edge), and decided once per edge
        id rather than per occurrence/epoch, matching production's sign-scramble
        design principle (a given edge is presented consistently everywhere it's
        visible, not re-randomized each time)."""
        if dataset is None:
            train_loader = self.trainer.train_dataloader
            dataset = getattr(train_loader, "dataset", None)
        if dataset is None:
            raise RuntimeError("EID scramble ablation requires an accessible dataset")
        edge_ids = getattr(dataset, "edge_ids", None)
        if edge_ids is None:
            raise RuntimeError("EID scramble ablation requires edge_ids metadata")
        valid_ids = edge_ids[edge_ids >= 0]
        if valid_ids.numel() == 0:
            raise RuntimeError("EID scramble ablation: no valid edge ids found")
        num_edges = int(valid_ids.max().item()) + 1

        gen = torch.Generator(device="cpu")
        base_seed = int(self.cfg.reproducibility.seed)
        gen.manual_seed(base_seed + seed_offset)
        perm = torch.randperm(num_edges, generator=gen)
        fixed = (perm == torch.arange(num_edges))
        if fixed.any():
            # Guarantee no fixed points: rotate fixed positions by one within
            # themselves. A single leftover fixed point (only possible when
            # exactly one collision occurred) can't be fixed by a length-1
            # rotation, so fall back to swapping it with its neighbor instead.
            idx = fixed.nonzero(as_tuple=False).flatten()
            if idx.numel() == 1:
                i = int(idx.item())
                j = (i + 1) % num_edges
                perm[i], perm[j] = perm[j].clone(), perm[i].clone()
            else:
                perm[idx] = perm[idx.roll(1)]
        assert (perm != torch.arange(num_edges)).all(), "EID scramble table has a fixed point"
        return perm

    def _build_eid_sign_flip_table(self, dataset=None):
        """Fixed per-edge-id Bernoulli(0.5) 'is this edge a liar' decision --
        mirrors production's `_build_sign_scramble_tables`'s flip table exactly
        (same rationale: only a fixed ~50% subset of edges ever shows a wrong
        sign; the other ~50% always shows its true sign. This matters --
        flipping EVERY edge's sign would make the ablation trivially invertible,
        the model could just learn "predict the opposite of what's shown" and
        perfectly recover the truth. A fixed random subset of liars can't be
        globally inverted that way.). Decided once, not per occurrence/epoch."""
        if dataset is None:
            train_loader = self.trainer.train_dataloader
            dataset = getattr(train_loader, "dataset", None)
        if dataset is None:
            raise RuntimeError("scramble_edge_signs requires an accessible dataset")
        edge_ids = getattr(dataset, "edge_ids", None)
        if edge_ids is None:
            raise RuntimeError("scramble_edge_signs requires edge_ids metadata")
        valid_ids = edge_ids[edge_ids >= 0]
        if valid_ids.numel() == 0:
            raise RuntimeError("scramble_edge_signs: no valid edge ids found")
        max_edge_id = int(valid_ids.max().item())

        gen = torch.Generator(device="cpu")
        base_seed = int(self.cfg.reproducibility.seed)
        gen.manual_seed(base_seed + 9_090_909)
        return torch.rand(max_edge_id + 1, generator=gen) < 0.5

    def _maybe_apply_eid_sign_scramble(self, sign_ids, edge_mask, edge_ids, edge_classes, dataset=None):
        """`model.scramble_edge_signs`, EID-native. A fixed ~50% subset of edges
        (see _build_eid_sign_flip_table) shows the DETERMINISTIC OPPOSITE sign
        (0<->1) at every VISIBLE (non-masked, non-split-excluded) position; the
        other ~50% shows its true sign, unchanged. `input_ids`/identity is
        completely untouched -- this isolates whether the model is misled by a
        wrong presented sign, independent of identity. Direct port of
        production's `_maybe_apply_sign_scramble` mechanic (fixed-liar-subset +
        deterministic flip), operating on sign_ids instead of input_ids since EID
        has a real class value per position already (edge_classes) rather than
        needing to invert a token-id-to-class map."""
        if edge_mask is None or not bool(getattr(self.cfg.model, "scramble_edge_signs", False)):
            return sign_ids
        if not self._eid_sign_scramble_ready:
            trainer = getattr(self, "_trainer", None)
            if dataset is None and trainer is not None and getattr(trainer, "sanity_checking", False):
                return sign_ids
            self._eid_sign_scramble_flip_cpu = self._build_eid_sign_flip_table(dataset=dataset)
            self._eid_sign_scramble_ready = True

        flip_table = self._eid_sign_scramble_flip_cpu.to(edge_ids.device)
        safe_ids = edge_ids.clamp(min=0, max=flip_table.numel() - 1)
        flip_here = edge_mask & (edge_ids >= 0) & flip_table[safe_ids]
        if not flip_here.any():
            return sign_ids

        num_classes = int(self.cfg.model.num_classes)
        true_class = edge_classes.clamp(min=0, max=num_classes - 1)
        flipped_class = (num_classes - 1) - true_class
        x = sign_ids.clone()
        x[flip_here] = flipped_class[flip_here]
        return x

    def _maybe_apply_eid_identity_scramble(self, input_ids, edge_mask, edge_ids, dataset=None):
        """`model.scramble_edge_identity` (new, no production analogue): every
        VISIBLE edge position shows the IDENTITY of a fixed, different substitute
        edge instead of its own true identity -- `sign_ids` (the true sign) is
        completely untouched. Mirror of `_maybe_apply_eid_sign_scramble`: isolates
        whether the model is misled by a wrong presented identity, independent of
        sign correctness."""
        if edge_mask is None or not bool(getattr(self.cfg.model, "scramble_edge_identity", False)):
            return input_ids
        if not self._eid_identity_scramble_ready:
            trainer = getattr(self, "_trainer", None)
            if dataset is None and trainer is not None and getattr(trainer, "sanity_checking", False):
                return input_ids
            self._eid_identity_scramble_perm_cpu = self._build_eid_scramble_table(1_234_567, dataset=dataset)
            self._eid_identity_scramble_ready = True

        mask_id = int(getattr(self.cfg.model, "mask_id", 1))
        visible = edge_mask & (edge_ids >= 0) & (input_ids != mask_id)
        if not visible.any():
            return input_ids

        perm = self._eid_identity_scramble_perm_cpu.to(edge_ids.device)
        safe_ids = edge_ids.clamp(min=0, max=perm.numel() - 1)
        substitute_edge_ids = perm[safe_ids]
        old_vocab_size = int(self.cfg.model.old_vocab_size)
        x = input_ids.clone()
        x[visible] = old_vocab_size + substitute_edge_ids[visible]
        return x

    def _build_dynamic_targets_for_batch(self, sign_ids, metadata):
        """EID override of the inherited method: dynamic resplit hides SIGN only
        (sign_ids -> SIGN_NA at this epoch's sampled target positions), never
        IDENTITY -- production's version hides identity too via
        `input_ids[target_positions] = mask_id`, which would defeat this
        experiment's whole point (identity must stay visible everywhere except
        genuinely held-out/disallowed positions). Reuses the inherited
        `_sample_epoch_targets`/`_build_dynamic_train_pool`/`on_train_start`/
        `on_train_epoch_start` completely unmodified -- they only touch generic
        dataset metadata (edge_ids/edge_split_mask/input_ids/id2class), all of
        which EdgeIdentityStageViewDataset exposes the same way production's
        dataset does, so the per-epoch target *selection* is identical; only the
        batch-level *application* differs (sign_ids vs. input_ids)."""
        labels = torch.full_like(sign_ids, self.ignore_index)
        edge_ids = metadata["edge_ids"]
        split_mask = metadata["edge_split_mask"]
        edge_classes = metadata["edge_classes"]

        target_ids = self._epoch_target_edge_ids_by_device.get(edge_ids.device)
        if target_ids is None:
            if self._epoch_target_edge_ids_cpu is None:
                target_ids = torch.empty(0, dtype=torch.long, device=edge_ids.device)
            else:
                target_ids = self._epoch_target_edge_ids_cpu.to(edge_ids.device)
            self._epoch_target_edge_ids_by_device[edge_ids.device] = target_ids

        in_pool = (split_mask == SPLIT_TRAIN) | (split_mask == SPLIT_MASK)
        valid_edge = edge_ids >= 0
        target_positions = in_pool & valid_edge & torch.isin(edge_ids, target_ids)

        labels[target_positions] = edge_classes[target_positions]
        dynamic_sign_ids = sign_ids.clone()
        dynamic_sign_ids[target_positions] = SIGN_NA
        return dynamic_sign_ids, labels

    def _step(self, batch, stage: str):
        """Copy of LitEdgeClassifier._step with changes from production, marked
        inline: (a) sign_ids is read from metadata, (a2) dynamic resplit (if on)
        overrides sign_ids instead of input_ids -- see _build_dynamic_targets_
        for_batch override above, (a3) mask_edge_tokens's sign-channel companion,
        (a4) the inherited (production) sign-scramble call is replaced with EID's
        own sign/identity scramble ablations -- the inherited one operates on
        input_ids assuming production's one-token-per-class scheme and would
        silently misbehave here (see module docstring), (b) the new edge-identity
        replacement regularizer is applied train-side, (c) sign_ids is passed into
        the model call. Everything else -- loss, hardness path (unused/off here),
        metrics logging -- is verbatim production logic."""
        input_ids, labels, attention_mask, metadata = batch
        self._last_metadata = metadata

        model_input_ids = input_ids
        # --- (a): fetch sign_ids (moved earlier than production's _step, since
        # dynamic resplit needs it as input here instead of input_ids)
        sign_ids = metadata["sign_ids"]
        # --- (a2): dynamic resplit overrides sign_ids + labels, never model_input_ids
        if stage == "train" and self.training and self.dynamic_train_masking:
            sign_ids, labels = self._build_dynamic_targets_for_batch(sign_ids, metadata)

        if torch.all(labels == self.cfg.model.ignore_index):
            return None

        positions = metadata.get("positions")
        node_mask = None
        edge_mask = None
        if positions is not None:
            node_mask = (positions >= 0) & ((positions % 2) == 0)
            edge_mask = (positions >= 0) & ((positions % 2) == 1)

        if node_mask is not None:
            model_input_ids = self._maybe_apply_token_masking(model_input_ids, node_mask, edge_mask)
            # --- (a3): mask_edge_tokens's sign-channel companion (production's
            # inherited _maybe_apply_token_masking only touches input_ids above)
            sign_ids = self._maybe_apply_edge_sign_masking(sign_ids, edge_mask)
            # --- (a4): EID-native scramble ablations, NOT the inherited
            # _maybe_apply_sign_scramble (would silently misbehave here)
            sign_ids = self._maybe_apply_eid_sign_scramble(
                sign_ids, edge_mask, metadata["edge_ids"], metadata["edge_classes"]
            )
            model_input_ids = self._maybe_apply_eid_identity_scramble(
                model_input_ids, edge_mask, metadata["edge_ids"]
            )
            # --- (a5): new context-edge-masking ablation -- must run after dynamic
            # resplit has finalized `labels` above, since it needs the resolved
            # target mask (see method docstring)
            model_input_ids, sign_ids = self._maybe_apply_context_edge_masking(
                model_input_ids, sign_ids, edge_mask, labels
            )
            # --- (a6): symmetric counterpart -- masks the TARGET's own identity
            # instead of context (production-parity comparison point)
            model_input_ids = self._maybe_apply_target_identity_masking(
                model_input_ids, edge_mask, labels
            )

        if stage == "train" and self.training and node_mask is not None:
            model_input_ids = self._maybe_apply_node_replacement(model_input_ids, node_mask)

        # --- (b): apply the new edge-identity replacement regularizer
        # (sign_ids was already fetched, and dynamically overridden if applicable, above)
        if stage == "train" and self.training:
            old_vocab_size = int(self.cfg.model.old_vocab_size)
            model_input_ids = self._maybe_apply_edge_identity_replacement(model_input_ids, old_vocab_size)

        # --- (c): sign_ids threaded into the model call (production has no such arg)
        logits = self.model(
            model_input_ids,
            sign_ids,
            attention_mask=attention_mask,
            node_mask=node_mask,
        )

        weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
        use_hardness = stage == "train" and self.hardness_source_map_tensor is not None

        if use_hardness:
            B, S = labels.shape
            loss_flat = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                weight=weights,
                ignore_index=self.ignore_index,
                reduction="none",
            )
            loss_2d = loss_flat.view(B, S)
            target_mask = labels != self.ignore_index
            target_float = target_mask.float()

            target_pos = target_mask.long().argmax(dim=1)
            seq_idx = torch.arange(B, device=logits.device)
            left_pos = (target_pos - 1).clamp(min=0)
            right_pos = (target_pos + 1).clamp(max=S - 1)
            left_toks = input_ids[seq_idx, left_pos]
            right_toks = input_ids[seq_idx, right_pos]
            h_left = self.hardness_source_map_tensor[left_toks]
            h_right = self.hardness_target_map_tensor[right_toks]
            if self.hardness_combine == "max":
                combined = torch.maximum(h_left, h_right)
            elif self.hardness_combine == "weighted":
                combined = (
                    self.hardness_source_weight * h_left
                    + (1.0 - self.hardness_source_weight) * h_right
                )
            else:
                combined = (h_left + h_right) / 2.0
            if self.hardness_power != 1.0:
                combined = combined.clamp(min=0.0) ** self.hardness_power
            walk_w = 1.0 + self.hardness_lambda * combined
            composite = walk_w.unsqueeze(1)

            weighted = (loss_2d * composite * target_float).sum()
            loss = weighted / target_float.sum().clamp(min=1.0)
        else:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                weight=weights,
                ignore_index=self.ignore_index,
            )

        preds = logits.argmax(dim=-1).view(-1)
        targets = labels.view(-1)
        probs = torch.softmax(logits, dim=-1).view(-1, logits.size(-1))

        self.metrics_manager.update_metrics(stage, preds, targets, probs)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss
