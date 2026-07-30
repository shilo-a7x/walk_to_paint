# Feedback on the Related Work Section

## 1. The structural problem

Every paragraph currently does the exact same move: *(a) here's what prior work does, (b) here's how it moves information, (c) but it still funnels down to a fixed endpoint summary.* That's four paragraphs in a row ending on the identical beat. It's thematically disciplined — clearly done on purpose to keep "information flow" as the throughline — but read back to back it feels less like a survey and more like four restatements of the same thesis with different citations attached. A reviewer skimming related work wants to know *what exists and how it differs*, quickly; right now they have to read a mini-argument each time to extract that.

This is also why the section reads "too heavy" for related work — the prose is doing argumentative work (semi-formal claims about what information survives compression, what's "discarded," what "reappears one layer later") that belongs in the Limitations section, not here. Related work should characterize prior methods; the *bottleneck argument itself* is what "Limitations of Vertex-Centric Edge Classification" and Proposition 1 are for. Having both places argue it — one informally in prose, one formally with Fano's inequality — dilutes the formal version's punch. By the time the reader hits Proposition 1, they've already been told the conclusion three times.

## 2. What's missing / needs real citations

The current placeholders are roughly right; here's what to actually attach:

- **Edge/sign prediction:** Heider (1946); Cartwright & Harary (1956) for balance theory; Leskovec, Huttenlocher & Kleinberg (2010, *"Predicting Positive and Negative Links in Online Social Networks"*) for status theory / feature-based sign prediction — this is the canonical citation here, don't skip it.
- **Vertex-centric GNNs:** Kipf & Welling (GCN), Hamilton et al. (GraphSAGE), Veličković et al. (GAT) as the generic backbone, then Derr, Ma & Tang (2018, SGCN), Huang et al. (SiGAT), Huang et al. (SDGNN) for the signed-specific line.
- **Walk/sequence:** Perozzi et al. (DeepWalk), Grover & Leskovec (node2vec), and a signed/directed extension — Kim et al.'s SIDE is the standard citation there.
- **Graph transformers:** Ying et al. (Graphormer), Rampášek et al. (GraphGPS), Dwivedi & Bresson (Graph Transformer).

### What's genuinely missing, not just under-cited

- **SEAL / labeling-trick link prediction** (Zhang & Chen, 2018 and 2021, "Labeling Trick"). This is the closest existing idea to what \method\ actually does — extracting a subgraph *around the candidate edge itself* rather than pooling at endpoints. The current draft doesn't mention this line at all, and it's the first thing a reviewer is likely to ask about if it isn't preempted. Give this its own short paragraph, not a fold-in under graph transformers.
- **Oversquashing / information bottleneck in GNNs** (Alon & Yahav, 2021; Topping et al., 2022). This is the closest *theoretical* precedent to Proposition 1/2, and omitting it looks like a gap in awareness of adjacent theory. One sentence positioning against it — they bound information loss over message-passing depth, this paper bounds it over the endpoint read-out regardless of depth — closes that gap cheaply.
- **Relational/KG link prediction** (Schlichtkrull et al., R-GCN) — lower priority than the above two, but worth a clause since it's structurally the same problem (edge-type prediction) in a different setting.
- **Position-aware GNNs** (You, Ying & Leskovec, P-GNN) — treat as optional/cuttable. Adjacent but not load-bearing the way SEAL and oversquashing are.

## 3. Three ways to rewrite it, ranked

### Option A — Compress each paragraph to 3 sentences, cut the argumentative tail (recommended)

Keep the current four categories plus a fifth (SEAL/subgraph methods). Each paragraph becomes: what they do → one clause on the mechanism → one clause on the limitation, stated flatly, without the hedged "but ultimately..." rhetoric. Move the deeper "why this collapses information" reasoning entirely into the Limitations section, where the formal machinery already exists to say it properly. This fixes both the weight problem and the repetition problem, and is the smallest edit.

### Option B — Keep the current argumentative style but only do the full move once

Do the full "here's the collapse" argument for the first paragraph (edge/sign prediction) to set up the reader, then for the remaining paragraphs just say "the same read-out pattern holds" in a single clause and spend the saved space on citations and differentiation instead. Risk: still reads a little essayistic for AAAI's related-work convention, and it's a less clean edit than A.

### Option C — Restructure as two paragraphs instead of four

Merge into "vertex-centric methods" (sign prediction + GNNs + walks, since they share the same endpoint-summary failure mode) and "attention/subgraph methods" (transformers + SEAL, since they share the "wider receptive field, same read-out" failure mode). This is the most aggressive cut and probably the best *length* fit for related work, but loses the historical granularity (balance theory → GNNs → walks as a chronological story), and a reviewer working in one of these subareas may feel their corner got merged away. Only do this if space is tight.

### Ranking

**A > C > B.** A gets full citation coverage and the missing SEAL/oversquashing paragraphs without changing the paper's voice. C is a fallback if half a column needs to be cut. B doesn't solve the real problem, which is that the argument is duplicated elsewhere in the paper, not that it's told badly here.
