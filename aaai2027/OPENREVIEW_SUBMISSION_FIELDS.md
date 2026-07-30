# OpenReview submission fields — ready-to-paste content

Everything needed for the AAAI-27 OpenReview abstract-submission form, consolidated in one place.
**Important:** OpenReview's Title/Abstract fields only render inline `$...$` / block `$$...$$` TeX
*math*, not custom LaTeX macros or commands (`\method`, `\citep`, `\ph{}`, etc.) — anything like
that pasted in would show up as literal backslash-text to reviewers. The text below is plain,
paste-safe prose derived from the current `pewter_aaai.tex`, with all macros resolved and all
`\ph{PLACEHOLDER}` scaffolding either removed or replaced with an honest, non-bracketed sentence.

**Status note:** the Abstract below reflects two deliberate calls made when converting from the
`.tex` draft to submission-ready prose — flagged so you can override before actually submitting:
1. The AUC-gain sentence states the *qualitative* result (beats the best GNN baseline on all six
   datasets) rather than a specific percentage-point number, since the retrain isn't finished and
   the `.tex` draft itself flags the number as pending. This qualitative claim is true under the
   current (pre-`edge_cover`-retrain) SOTA table for both attention variants on all six datasets.
2. The attention/information-flow sentence (Result 3, deprioritized 2026-07-20) is **dropped
   entirely** from this submission copy rather than included as an unresolved claim — asserting
   something explicitly marked "not yet settled" in a real conference abstract is a real risk if
   it's never delivered. Add it back if you'd rather keep a hedged version.

---

## Title

```
PEWTER: Predicting Edge Signs with Proximal Edge-Walk Transformers
```

## TL;DR

Pick one (all are single sentences, matching the field's own instruction):

- **A (theory-forward):** A proven information bottleneck in vertex-centric GNNs motivates PEWTER, a random-walk Transformer that predicts edge signs by conditioning directly on nearby labeled edges instead of endpoint embeddings.
- **B (result-forward):** We prove that reading an edge's sign off its two endpoint embeddings is fundamentally bottlenecked, and show that conditioning on nearby labeled edges via a walk-based Transformer avoids it, beating GNN baselines on six signed networks.
- **C (mechanism-forward):** PEWTER predicts edge signs in signed networks by walking to and reading nearby labeled edges directly, instead of pooling through vertex embeddings — outperforming GNN baselines and motivated by a proven capacity bound on endpoint read-outs.
- **D (compact):** A random-walk Transformer for edge-sign prediction that sidesteps a provable information bottleneck in vertex-centric GNNs by conditioning on nearby labeled edges directly.
- **E (shortest):** Vertex embeddings are provably bottlenecked for edge-sign prediction; walking to nearby edges instead isn't.

## Abstract (plain text, paste-ready)

```
We study edge sign prediction in directed signed graphs, where every edge carries a discrete
label (a sign) and the goal is to predict the label of some edges given the labels of others.
The current dominant approach is vertex-centric graph neural networks (GNNs) that embed the two
endpoint vertices and read the edge label off the pair of endpoint embeddings. We argue that
this is the wrong object to condition on. First, a vertex embedding is shared by all edges
touching that vertex, so when a vertex participates in edges of multiple signs, the endpoint
representation cannot separate them properly, and the error is bounded below by the label
entropy at that vertex regardless of network depth or width. Second, in practice, edge label
information is directional: the entropy of the labels seen from the target side of an edge is
higher than from the source side, so the informative signal travels along the edge orientation
and not against it. We give the first claim as a proposition with a proof, and confirm both
claims empirically. We show that GNN baseline accuracy drops sharply on high-entropy vertices,
consistent with the bound above, and that the information on the edge sign decreases to
practically 0 beyond 2 hops. As such, we propose Predicting Edge Signs with Proximal Edge-Walk
Transformers (PEWTER), which samples short random walks in edge space that stay near the target
edge, classifies the target edge with a transformer that reads each walk as a token sequence, and
combines walks by a weighted vote. PEWTER improves test AUC over the strongest GNN baseline on
all six benchmarks, with the largest gains where endpoint entropy is high.
```

(Minor cleanups applied versus the `.tex` draft: "in practice, typically" redundant double-hedge
trimmed to "in practice"; "a propositions" grammar fix already applied in `pewter_aaai.tex`
itself, not just here.)

## Primary Topic

**Machine Learning → Graph-based Machine Learning**

## Secondary Topics (up to 5)

1. Data Mining & Knowledge Management → Graph Mining & Social Network Analysis
2. Machine Learning → Deep Learning Theory & Learning Theory
3. Machine Learning → Representation Learning
4. Machine Learning → Deep Learning Algorithms, Architectures & Foundation Models
5. Application Domains → Social Networks & Web *(optional 5th, application-domain framing)*

Taxonomy sourced from AAAI-27's official "Areas and Topics" page (not recoverable from the saved
OpenReview page's assets — confirmed by grep, the dropdown loads live from OpenReview's API and
nothing is cached in the saved `.js`/`.css` files). Worth a quick glance at the live dropdown at
actual submission time since this came through an AI-summarized fetch, not a byte-exact scrape.

## Open naming question (not yet resolved, see chat)

Whether to keep the title's plain descriptive subtitle (current, matches the BERT precedent —
title subtitle ≠ the acronym's real backronym) or rework the title to spell PEWTER out directly,
or explore an alternative name entirely (TREK, WANDER, WEST, etc. discussed in chat). Doesn't
block the abstract submission either way — the title above is submission-ready as currently
drafted regardless of which way this goes.
