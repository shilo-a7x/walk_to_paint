# Edge-vs-vertex claim review — what would need to change, if anything

Not a rewrite. A catalog of every place `WSDM_format_revised.tex` makes a claim about
edge-sign conditioning, checked against the corrected (bug-fixed) ablation numbers, so a
future decision on whether/how to touch the paper starts from a clear list instead of a
re-derivation. No edits made to the real tex.

## The finding this is about

Three independent, bug-fixed measurements agree: nearby edge-sign **values** carry no
detectable additional signal beyond what identity-indexed vertex embeddings already
capture.

| measurement | result |
|---|---|
| `mask_edge_tokens` (edge token → `<MASK>`, **train AND eval**, model never sees a real sign value anywhere, ever) | every dataset within ±0.06pp of baseline, none reach p<0.05 (`ABLATION_MASKNODE_MASKEDGE_TRUE_RESULTS.md`) |
| `SIGNSCRAMBLE` (real-looking but ~50% corrupted sign values, fixed per edge) | null on all 6, once its own analogous posthoc-eval bug was fixed (`SIGNSCRAMBLE_ABLATION.md`) |
| EID linear probe (learned per-edge identity embedding, leakage-free held-out split) | edge-alone AUC ≈ chance (0.4786 vs. 0.4986 floor); vertex-pair-alone AUC 0.8401 |

Against one dramatic positive: `mask_node_tokens` (vertex → `<UNK>`, train+eval) costs
13.4–33.3pp, every dataset, p=0.00098.

**Mechanism, given `mask_edge_tokens` runs train+eval, not eval-only:** a model trained
under it never sees a real sign value in its input, at any point in training — the only
channel for sign information to reach it is the loss at the masked target position,
propagating into the shared vertex-embedding parameters across every occurrence of that
vertex in the corpus. That model performs the same as the real one. This is close to a
direct confirmation that whatever the local attention window is doing when it attends to
nearby edge tokens, it isn't reading instance-specific sign information unrecoverable from
vertex identity alone — because a model that architecturally cannot read it reaches the
same place.

**The needed distinction, not currently made anywhere in the tex:**
- *Architecturally true* (true by construction, needs no ablation): edge-sign tokens
  occupy real input positions and the target edge is never pooled into a vertex pair
  before prediction — a genuine structural difference from GNN/node2vec-style read-out.
- *Functionally/causally false as currently worded*: that this channel is load-bearing —
  that removing it costs something. The corrected ablations say it doesn't.

Several sentences currently assert the second without qualifying that it's the first.

## Per-location catalog

Legend: **OK** = architectural/structural claim, still true, no change needed. **OVERCLAIM**
= asserts functional reliance on edge-sign values not supported by corrected numbers.
**STALE DATA** = cites the pre-fix ablation numbers directly. **OPEN** = defensible either
way depending on a judgment call, flagged for discussion.

| location | claim | status |
|---|---|---|
| Title / method name ("Proximal Edge-Walk Transformer") | branding | OK to keep — the local window and edge-token representation are real architectural choices; the name doesn't itself assert they're what drives the AUC gain. |
| Abstract: "We thus propose to condition on both the signed edges and the neighboring vertices near the target edge, instead of on only the two vertex summaries." | describes the architecture's input | OK — architectural, not a reliance claim. |
| Abstract: closing sentence on attention split (vertex vs. edge mass, source/target asymmetry) | empirical, already hedged as "attention mass," not "reliance" | OK, already careful (see §6.5 caveat below). |
| Intro ¶2 (current): "We thus restrict each token's attention to a local window... enough to drastically increase edge classification accuracy" | implies the window's value comes from edge-sign locality | OVERCLAIM (mild) — Ablation A already shows the window costs/gains <0.2pp vs. full attention; "drastically increase" overstates what the window itself contributes either way. The *identity-embedding* channel is what's large, not window size. |
| Intro ¶2, footnote-style remark at Assumption 1 discussion (~line 130): identity-indexed embeddings escape WL coarsening on their own | already correct, currently undersold | Consider foregrounding this — it's the theorem's own account of the mechanism that the ablations now show is dominant. Currently framed as a passing remark inside the Assumption discussion, not connected forward to the Results. |
| Related Work, "Edge and sign prediction": "Pewter keeps the local, edge-centric view of this line of work" | ambiguous between structural and informational edge-centrism | OPEN — true structurally (target edge stays represented), not informationally (edge *content* isn't shown to add signal). Reads fine to a careful reader but invites the ambiguity discussed above. |
| Contributions bullet 4: "...predicts a masked target edge from an ordered local sequence containing both vertices and signed edges, thereby preserving edge-local relational context" | structural description | OK — describes the architecture, not a reliance claim. |
| §6.2 (Results intro): **"Unlike the vertex-centric baselines below, every prediction is conditioned directly on nearby edge signs, not only on the two endpoints' learned representations."** | asserts edge-sign values are *the* differentiator from vertex-centric baselines | **OVERCLAIM — the single most direct instance.** This is the sentence most worth revisiting if the paper is ever touched again; it's stated as the explanation for outperforming vertex-centric GNNs, and the corrected ablations put the real explanation elsewhere (identity-indexed, walk-context-refined vertex embeddings escaping the WL ceiling). |
| Ablation A ("Proximal attention is sufficient"), full text | reports the local-vs-full attention AUC deltas honestly (<0.2pp, mixed sign) and already concludes "the window costs nothing rather than that it helps" | OK as currently worded — already the most careful section in the paper on this exact question. Worth noting explicitly that "proximal attention is sufficient" is a claim about window *size* being harmless, not about edge-sign *content* mattering — the two read as similar claims but aren't. |
| The two masking ablations (`abl:masknode`/`abl:maskedge`), currently commented out of the submitted tex | cites the **pre-fix** numbers in their commented-out text (−1.0 to −1.6pp, "significant... p≤0.032" for maskedge) | **STALE DATA** — if these are ever restored, they must use the corrected numbers (±0.06pp, none significant), not the ones currently sitting in the comment. The commented-out "Taken together" synthesis paragraph is also now wrong in the same way ("explicit edge-sign tokens contribute additional relational information" — corrected data says they don't, detectably). |
| §6.5 "Where Pewter's attention goes" (vertex vs. edge attention-mass split, Panel C) | reports *attention allocation*, already explicitly caveated: "Attention allocation describes where the model places computational weight and should not by itself be interpreted as direct measurement of feature importance." | OK — this section is already methodologically careful and doesn't need to change; it's arguably a model for how the rest of the paper should talk about this. |
| §6.6 Complexity | discusses $K_{uv}$ (walk occurrences) and $O(L^2d)/O(Lwd)$ attention cost | OK — mechanical/cost claims, unrelated to whether edge content carries signal. |
| Discussion §8 (line ~431): "PEWTER escapes the endpoint bottleneck in part because its vertex tokens are identity-indexed embeddings rather than structural encodings" | already states the correct mechanism | OK, already correct — but framed only as a cost/trade-off (the transductive-limitation paragraph), not connected back to Results as *the* explanation for the AUC gain. Currently the most under-used correct sentence in the paper. |

## If this is ever revisited — the shape of the fix, not a draft

1. Restore the two masking ablations with the **corrected** numbers (this file's table
   above), not the ones currently commented out.
2. Add SIGNSCRAMBLE as a third, independent confirming ablation (not currently in the
   tex at all) — the "does an edge token being present matter" vs. "does its specific
   value matter" distinction it draws is exactly the distinction this whole issue turns
   on, and it's already fully run.
3. Rewrite §6.2's one overclaiming sentence to separate the architectural claim (edge
   position is never pooled away) from the informational claim (nearby sign *values* add
   nothing detectable) — don't drop the structural claim, it's true and it's real.
4. Pull the Discussion's "identity-indexed embeddings escape the bottleneck" sentence
   forward, or at least cross-reference it from Results, so it reads as *the* explanation
   rather than a buried trade-off aside.
5. Optional, not required for correctness: the zero/minimal-context ablation described
   in-conversation (attention window shrunk to literally `N_u, <MASK>, N_v`, no other
   tokens) would further separate "vertex identity alone" from "vertex identity plus a
   little nearby vertex context" — not yet run, no new data needed, reuses existing
   training infra.

Nothing here requires new data collection, and nothing here has been applied to the real
tex. This is a reference for the defense conversation and for a future decision on
whether the paper itself needs touching before/after the thesis.
