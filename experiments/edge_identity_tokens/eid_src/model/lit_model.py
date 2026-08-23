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

NOT supported by this subclass (both fail loudly rather than silently misbehave if
ever accidentally enabled on an EID run):
  - `dynamic_train_masking` -- see EdgeIdentityStageViewDataset's __init__ docstring;
    production hides a dynamic target via `input_ids = mask_id`, which would destroy
    this experiment's identity-stays-visible design. Left as a real gap, not solved
    here -- keep `model.dynamic_train_masking=false` in the EID run config.
  - `scramble_edge_signs` -- production's sign-scramble ablation flips a token id to
    "the other class's token id" (`class2tokid[flipped_class]`), which assumes
    exactly one token id per class (true in production's 2-token scheme, not true
    here where thousands of distinct edge tokens share each class) -- would silently
    pick an arbitrary wrong token rather than crash, so this is flagged here in
    prose rather than caught by an assertion. Keep `model.scramble_edge_signs=false`.

Everything else (`configure_optimizers`, `on_train_epoch_end`/`on_validation_epoch_end`/
`on_test_epoch_end`, `training_step`/`validation_step`/`test_step`, hardness-reweighting
plumbing -- unused here, gated off by default same as production, `_maybe_apply_node_replacement`
for VERTEX tokens, `_maybe_apply_token_masking`) is inherited completely unmodified.
"""

import torch

from src.model.lit_model import LitEdgeClassifier
from experiments.edge_identity_tokens.eid_src.model.model import EdgeIdentityTransformerModel


class EIDLitEdgeClassifier(LitEdgeClassifier):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        # Swap in the sign-embedding-aware model. super().__init__ already built
        # self.model = TransformerModel(cfg) once; replace it rather than trying to
        # intercept construction, since LitEdgeClassifier.__init__ does several other
        # required things first (save_hyperparameters, class_weights, metrics_manager,
        # dynamic/sign-scramble/hardness state) that this subclass still needs.
        self.model = EdgeIdentityTransformerModel(self.cfg)

        if bool(getattr(self.cfg.model, "dynamic_train_masking", False)):
            raise NotImplementedError(
                "EIDLitEdgeClassifier does not support dynamic_train_masking yet -- "
                "see this file's module docstring."
            )
        if bool(getattr(self.cfg.model, "scramble_edge_signs", False)):
            raise NotImplementedError(
                "EIDLitEdgeClassifier does not support scramble_edge_signs -- "
                "see this file's module docstring."
            )

    def forward(self, input_ids, sign_ids):
        return self.model(input_ids, sign_ids)

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

    def _step(self, batch, stage: str):
        """Copy of LitEdgeClassifier._step with three changes from production,
        marked inline: (a) sign_ids is read from metadata, (b) the new edge-identity
        replacement regularizer is applied train-side, (c) sign_ids is passed into
        the model call. Everything else -- loss, hardness path (unused/off here),
        metrics logging -- is verbatim production logic."""
        input_ids, labels, attention_mask, metadata = batch
        self._last_metadata = metadata

        model_input_ids = input_ids
        # dynamic_train_masking is asserted false in __init__, so this branch never
        # fires -- kept only so this method stays a faithful diff against production.
        if stage == "train" and self.training and self.dynamic_train_masking:
            model_input_ids, labels = self._build_dynamic_targets_for_batch(input_ids, metadata)

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
            model_input_ids = self._maybe_apply_sign_scramble(
                model_input_ids, edge_mask, metadata["edge_ids"], metadata["edge_classes"]
            )

        if stage == "train" and self.training and node_mask is not None:
            model_input_ids = self._maybe_apply_node_replacement(model_input_ids, node_mask)

        # --- (a)/(b): fetch sign_ids, apply the new edge-identity replacement regularizer
        sign_ids = metadata["sign_ids"]
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
