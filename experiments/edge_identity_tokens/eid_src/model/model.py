"""Edge-identity-token variant of the production TransformerModel.

Subclasses (not copies) src/model/model.py's TransformerModel directly -- the
UNMODIFIED production class does all the real work (embedding table sizing,
local-attention windowing, LocalAttentionEncoderLayer's NaN-safety fix, every
ablation branch in forward()); this file adds two things this experiment needs:

1. A small sign embedding, summed (or concatenated, see below) with the token
   embedding the same way positional encoding is added (MECHANISM.md section 4).
2. An OPTIONAL low-rank factorization of the edge-identity table
   (model.edge_embed_rank > 0), added 2026-08-23 after the unregularized EID run
   collapsed to chance test AUC: with vocab_size=27,974 and embedding_dim=64 on
   bitcoin-alpha, the edge-identity table alone is ~1.79M params -- ~94% of the
   whole 1.9M-param model, an ~18:1 ratio against the transformer body (~100K
   params). That's backwards from typical vocab:dim ratios (GPT-2-small ~65:1,
   BERT-base ~40:1; ours is ~437:1) and gives each of the 24,186 edges a nearly
   free parameter vector with very little shared computation regularizing it --
   the direct structural cause of the memorization pattern the edge_replace_prob
   regularizer (lit_model.py) was built to counteract. This doesn't replace that
   regularizer -- factorization reduces the CAPACITY to memorize, edge_replace
   reduces the INCENTIVE to, even when capacity exists; the two are complementary.

   ALBERT-style (Lan et al. 2019): edge identity gets its own small table,
   `nn.Embedding(num_edges, rank)` with rank << embedding_dim (e.g. 8-16), then a
   SHARED `nn.Linear(rank, content_dim)` projects up before joining the rest.
   Parameter cost becomes `num_edges*rank + rank*content_dim` instead of
   `num_edges*embedding_dim` -- at rank=8 vs. embedding_dim=64 that's roughly an
   8x cut to the edge table specifically. Node/vertex tokens are unaffected (kept
   at full embedding_dim in a separate table) -- their vocab is small enough
   (3,788 on bitcoin-alpha) that they were never the problem.

   `model.edge_sign_combine` ("add", default, or "concat") controls how the
   (projected) content vector and the sign vector combine. "add" matches the
   original design exactly (content_dim == embedding_dim, sign summed in, same
   subspace). "concat" puts them in disjoint channels of the final embedding_dim-
   wide vector (content_dim = embedding_dim - model.sign_embed_dim) -- avoids
   forcing "which edge" and "what sign" to superpose in the same subspace, at the
   cost of giving content fewer of the embedding_dim channels to work with.

   Backward compatible: model.edge_embed_rank defaults to 0 (disabled), which
   reproduces the original unified `self.embed` table exactly -- no behavior
   change for any run that doesn't set this flag.

3. An OPTIONAL separate node/vertex embedding width (model.node_embed_dim,
   added 2026-08-24), only meaningful when edge_embed_rank>0 (factorized mode
   already has a standalone `base_embed` table for nodes at that point).
   Undecoupled by default, node capacity is whatever `content_dim` happens to
   be after the edge/sign split (e.g. content_dim=48 when embedding_dim=64,
   sign_embed_dim=16, concat) -- an incidental trim, not a deliberate choice.
   Setting model.node_embed_dim gives nodes their own width, projected up to
   content_dim via a shared nn.Linear (same up-projection pattern as
   edge_proj, mirrored -- unlike edges, node vocab (3,788 on bitcoin-alpha) is
   small enough that this isn't about parameter-count capacity control, it's
   about letting Optuna search node representational width independently of
   whatever the edge/sign split leaves over. Defaults to content_dim (a
   straight nn.Embedding, no projection, identical to the old behavior) when
   unset.

`forward()` has to be a near-full copy of the parent's, not a call to
`super().forward()` plus a patch, because the parent recomputes
`x = self.embed(input_ids)` from scratch as its very first line with no hook to
intercept before the ablation branches run on it -- the only clean way to inject
the sign/factorization logic at that same point is to own the whole method body.
Every line after content+sign combination is copied verbatim from
src/model/model.py::TransformerModel.forward, unchanged.
"""

import torch
import torch.nn as nn

from src.model.model import TransformerModel


class EdgeIdentityTransformerModel(TransformerModel):
    def __init__(self, cfg):
        super().__init__(cfg)  # builds self.embed (vocab_size x embedding_dim), pos_encoder,
                                # transformer, out head, and every ablation flag -- reused as-is

        self.edge_embed_rank = int(getattr(cfg.model, "edge_embed_rank", 0) or 0)
        self.edge_sign_combine = str(getattr(cfg.model, "edge_sign_combine", "add"))
        embedding_dim = cfg.model.embedding_dim

        if self.edge_embed_rank <= 0:
            # Original design: one unified table across the whole vocab (built by
            # super().__init__ above), sign summed in at full width.
            self.sign_embed_dim = embedding_dim
            self.sign_embedding = nn.Embedding(3, self.sign_embed_dim)
            self.content_dim = embedding_dim
            self.old_vocab_size = None  # unused in this mode
        else:
            # Factorized mode: replace the unified self.embed with separate
            # node/base and low-rank edge tables. del first so the large unified
            # table super().__init__ built is never carried as dead weight in the
            # optimizer's parameter set.
            del self.embed

            self.old_vocab_size = int(cfg.model.old_vocab_size)
            num_edges = int(cfg.model.vocab_size) - self.old_vocab_size
            assert num_edges > 0, "edge_embed_rank>0 requires an EID cache (cfg.model.old_vocab_size set)"

            if self.edge_sign_combine == "concat":
                self.sign_embed_dim = int(getattr(cfg.model, "sign_embed_dim", 16))
                self.content_dim = embedding_dim - self.sign_embed_dim
                assert self.content_dim > 0, "sign_embed_dim must be < embedding_dim in concat mode"
            else:
                self.sign_embed_dim = embedding_dim
                self.content_dim = embedding_dim

            self.sign_embedding = nn.Embedding(3, self.sign_embed_dim)

            node_embed_dim = int(getattr(cfg.model, "node_embed_dim", 0) or 0)
            if node_embed_dim > 0 and node_embed_dim != self.content_dim:
                self.node_embed_dim = node_embed_dim
                self.base_embed = nn.Embedding(self.old_vocab_size, node_embed_dim, padding_idx=cfg.model.pad_id)
                self.node_proj = nn.Linear(node_embed_dim, self.content_dim)
            else:
                self.node_embed_dim = self.content_dim
                self.base_embed = nn.Embedding(self.old_vocab_size, self.content_dim, padding_idx=cfg.model.pad_id)
                self.node_proj = None

            self.edge_embed_low = nn.Embedding(num_edges, self.edge_embed_rank)
            self.edge_proj = nn.Linear(self.edge_embed_rank, self.content_dim)

    def _content_embed(self, input_ids):
        """Look up the (edge-identity-or-node) content vector per position."""
        if self.edge_embed_rank <= 0:
            return self.embed(input_ids)

        old_vs = self.old_vocab_size
        is_edge_tok = input_ids >= old_vs
        base_ids = input_ids.clamp(max=old_vs - 1)
        edge_ids_0 = (input_ids - old_vs).clamp(min=0)

        base_content = self.base_embed(base_ids)
        if self.node_proj is not None:
            base_content = self.node_proj(base_content)
        edge_content = self.edge_proj(self.edge_embed_low(edge_ids_0))
        return torch.where(is_edge_tok.unsqueeze(-1), edge_content, base_content)

    def forward(self, input_ids, sign_ids, attention_mask=None, node_mask=None):
        content = self._content_embed(input_ids)
        sign_vec = self.sign_embedding(sign_ids)
        x = torch.cat([content, sign_vec], dim=-1) if self.edge_sign_combine == "concat" else content + sign_vec

        if node_mask is not None and (self.zero_node_tokens or self.zero_edge_tokens):
            keep = (~node_mask) if self.zero_node_tokens else node_mask
            x = x * keep.unsqueeze(-1).to(x.dtype)

        if self.training and node_mask is not None:
            if self.node_context_mode == "mask_unscaled" and self.node_mask_prob > 0.0:
                keep = torch.rand_like(node_mask, dtype=torch.float) >= self.node_mask_prob
                node_keep = (~node_mask) | keep
                x = x * node_keep.unsqueeze(-1).to(x.dtype)
            elif self.node_context_mode == "noise" and self.node_noise_sigma > 0.0:
                noise = torch.randn_like(x) * self.node_noise_sigma
                x = x + noise * node_mask.unsqueeze(-1).to(x.dtype)

        x = x + self.pos_encoder[: input_ids.size(1)]
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None

        if self.local_attention_window is not None:
            seq_len = input_ids.size(1)
            pos = torch.arange(seq_len, device=input_ids.device)
            dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
            attn_mask = dist > self.local_attention_window
            attn_mask = attn_mask.view(1, 1, seq_len, seq_len)
            if key_padding_mask is not None and key_padding_mask.any():
                attn_mask = attn_mask | key_padding_mask.view(-1, 1, 1, seq_len)
            src_key_padding_mask = None
        else:
            attn_mask = None
            src_key_padding_mask = key_padding_mask

        x = self.transformer(x, mask=attn_mask, src_key_padding_mask=src_key_padding_mask)
        return self.out(x)
