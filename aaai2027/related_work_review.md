# Related Work Review and Redesign Notes

## Purpose

This document summarizes recommendations for redesigning the Related
Work section of the PEWTER paper. The goal is to keep the section
aligned with the paper's central theme (information flow) while ensuring
that it still reads like a **Related Work** section rather than the
beginning of the theory.

------------------------------------------------------------------------

# 1. Overall Assessment

## Scores

  Aspect                   Score
  --------------------- --------
  Organization              9/10
  Coverage                8.5/10
  Narrative               9.5/10
  Fit as Related Work     6.5/10

### Main issue

The current version gradually turns into an argument for the paper's
theory instead of surveying previous work.

A reviewer should finish Related Work thinking:

> "I understand the existing landscape and the gap."

not

> "The authors have already started proving their main theorem."

------------------------------------------------------------------------

# 2. Keep the Information-Flow Organization

Instead of organizing by model families only, organize by how
information is propagated.

Suggested flow:

1.  Classical edge/sign prediction
2.  Vertex-centric GNNs
3.  Walk and sequence methods
4.  Graph transformers
5.  Common information-flow perspective (bridge)

This is one of the strongest aspects of the current draft and should be
preserved.

------------------------------------------------------------------------

# 3. Tone Recommendations

## Option A (Recommended)

Use information flow as the organizing principle, but avoid
information-theoretic language.

Discuss:

-   how information propagates
-   how it is aggregated
-   what information reaches the predictor

Avoid discussing entropy, bottlenecks, compression, or conditional
information here.

★★★★★

------------------------------------------------------------------------

## Option B

Traditional survey.

Describe methods, strengths, weaknesses.

Discuss the common endpoint-embedding paradigm only in the concluding
paragraph.

★★★★☆

------------------------------------------------------------------------

## Option C

Current draft.

Interesting and coherent, but reads like Theory.

★★★☆☆

------------------------------------------------------------------------

# 4. Information Flow vs. Information Theory

## Good vocabulary

-   information propagation
-   information aggregation
-   information preservation
-   directional information
-   neighborhood information
-   long-range information
-   structural context

## Avoid in Related Work

-   entropy
-   conditional entropy
-   mutual information
-   information bottleneck
-   Fano's inequality
-   compression arguments
-   impossibility language

These belong in the Theory section.

------------------------------------------------------------------------

# 5. Example Sentence Rewrites

Too strong:

> Every variant compresses everything a vertex has seen into one
> embedding.

Better:

> Most signed GNN architectures aggregate neighborhood information into
> vertex representations, after which edge labels are predicted from the
> embeddings of the incident vertices.

------------------------------------------------------------------------

Too strong:

> The aggregation necessarily loses information.

Better:

> Aggregation summarizes neighborhood information before edge
> prediction.

------------------------------------------------------------------------

Too strong:

> The predictor only observes a compressed summary.

Better:

> The final predictor operates on the resulting vertex embeddings rather
> than directly accessing the neighborhoods.

------------------------------------------------------------------------

# 6. What Should Be Cited?

## A. Classical Sign Prediction

-   Structural Balance Theory
-   Status Theory
-   Leskovec et al.
-   Trust prediction literature

Purpose: Historical background.

------------------------------------------------------------------------

## B. Signed GNNs

Representative papers:

-   SGCN
-   SiGAT
-   SNEA
-   SDGNN
-   SGA
-   MSGNN (if relevant)

Purpose: Dominant modern approaches.

------------------------------------------------------------------------

## C. Walk Methods

-   DeepWalk
-   node2vec
-   LINE (optional)
-   Signed walk papers

Purpose: Information collected through sequences rather than message
passing.

------------------------------------------------------------------------

## D. Sequence Models

Examples of

-   Transformer over walks
-   RNN over walks
-   Language-model style graph representations

If only one or two examples exist, keep this paragraph short.

------------------------------------------------------------------------

## E. Graph Transformers

Representative citations

-   Graphormer
-   GraphGPS
-   SAT
-   TokenGT
-   NAGphormer (if discussing long-range dependencies)

------------------------------------------------------------------------

## F. Missing Literature

Strong recommendation:

Include a paragraph discussing

-   Oversmoothing
-   Oversquashing
-   Expressiveness limitations
-   Graph rewiring

Reason:

Reviewers familiar with GNN theory will expect these citations.

Important distinction:

Oversquashing studies information propagation inside message passing.

Your work studies the information available to an endpoint-based edge
predictor after propagation.

These are complementary, not competing, viewpoints.

------------------------------------------------------------------------

# 7. Be Careful With Universal Claims

Avoid

-   every method
-   all GNNs
-   every transformer
-   always

Prefer

-   most existing approaches
-   the dominant formulation
-   typically
-   common endpoint-embedding methods

This avoids easy reviewer counterexamples.

------------------------------------------------------------------------

# 8. Suggested Closing Paragraph

Example:

Across these families, the mechanisms used to propagate information
differ substantially---from local structural heuristics, to message
passing, random walks, and global attention. Nevertheless, most
approaches ultimately represent an edge through embeddings of its two
incident vertices. This shared design motivates the
information-theoretic analysis developed in the remainder of the paper.

------------------------------------------------------------------------

# 9. Structural Recommendations

Do not prove your contribution here.

Instead:

Describe ↓

Compare ↓

Motivate

The theory section should then introduce

-   entropy
-   conditional entropy
-   bottlenecks
-   Fano
-   proofs

This gives the paper a much stronger narrative arc.

------------------------------------------------------------------------

# 10. Reviewer Perspective

An AAAI reviewer generally expects Related Work to answer:

-   What methods already exist?
-   How are they different?
-   What gap remains?

Not:

-   Why those methods must fail.

Leave the latter for Theory.

------------------------------------------------------------------------

# 11. Final Recommendation

Keep the information-flow narrative because it is distinctive and aligns
with the contribution.

However:

-   Remove explicit information-theoretic arguments.
-   Delay entropy until Theory.
-   Add a short paragraph positioning the work relative to oversquashing
    and oversmoothing.
-   Add a concluding bridge explaining that most methods ultimately
    predict edges from endpoint embeddings.

This preserves the conceptual story while making the Related Work
section feel like a survey rather than an argument.
