# Related Work Analysis & Complete Draft Options (`pewter_aaai.tex`)

## Executive Summary & Diagnosis

The current Related Work section feels heavy because it attempts to execute two distinct tasks simultaneously:
1. **Mapping the Literature:** Summarizing past research (Triads, Signed GNNs, Walks, Transformers).
2. **Advocating the Theory:** Critiquing each family through the specific lens of information flow, entropy, and endpoint compression bottlenecks.

While information flow is your central theme, forcing every paragraph to deliver a theoretical critique creates redundancy with your Introduction and formal Limitations section. In top-tier venues like AAAI, reviewers prefer the Related Work section to present prior methods **on their own terms** neutrally, leaving heavy theoretical critiques to the Introduction and dedicated Theory/Limitations sections.

---

## Key Recommendations

### 1. Separate "What They Do" from "Why They Fail"
* **Keep Related Work descriptive:** Focus on how prior methods model edges and directional context.
* **Delegate the theoretical heavy lifting:** Reserve formal proofs of the entropy ceiling ($\mathrm{H}(Y \mid c(U), c(V))$) and capacity bounds for Section 4.

### 2. Include Missing Literature Streams
Your draft placeholder noted candidate missing areas. Two key bodies of work should be explicitly added:
* **Over-squashing & Information Bottleneck in GNNs:** Citing this literature (e.g., Alon & Yahav, Topping et al.) grounds your endpoint bottleneck claim in established GNN expressiveness theory.
* **Knowledge Graph / Multi-Relational Embeddings:** Briefly acknowledge relation-prediction literature (e.g., TransE, ComplEx, RotatE) as an adjacent domain that models edge-specific representations rather than relying solely on pooled endpoint features.

---

## BibTeX Citation Mapping

Below are the canonical citations to replace your `\citep{PLACEHOLDER}` slots:

| Section/Slot | Recommended Citation Keys | Paper Details |
| :--- | :--- | :--- |
| **Balance Theory** | `heider1946attitudes`<br>`cartwright1956structural` | Heider (1946), *Attitudes and Cognitive Organization*<br>Cartwright & Harary (1956), *Structural Balance* |
| **Status Theory & Triads** | `leskovec2010predicting`<br>`leskovec2010signed` | Leskovec et al. (2010a), *CHI*<br>Leskovec et al. (2010b), *WWW* |
| **Signed GNNs** | `derr2018signed`<br>`huang2019signed`<br>`li2020signed`<br>`jung2021sdgnn` | **SGCN** (ICDM 2018)<br>**SiGAT** (ICDM 2019)<br>**SNEA** (AAAI 2020)<br>**SDGNN** (KDD 2021) |
| **Walk Embeddings** | `perozzi2014deepwalk`<br>`grover2016node2vec`<br>`yuan2017sne`<br>`wang2020side` | **DeepWalk** (KDD 2014)<br>**node2vec** (KDD 2016)<br>**SNE** (IJCAI 2017)<br>**SIDE** (TKDE 2020) |
| **Walk Transformers** | `palmer2021edge`<br>`selvan2022crawl` | Edge-walk encoders / CRaWl Transformer |
| **Graph Transformers** | `ying2021do`<br>`rampasek2022recipe` | **Graphormer** (NeurIPS 2021)<br>**Graph GPS** (NeurIPS 2022) |
| **Over-squashing / Bottlenecks** | `alon2021on`<br>`topping2021understanding` | Alon & Yahav (ICLR 2021)<br>Topping et al. (ICLR 2021) |

---

## Section Ranking & Architectural Comparison

1. **Option 1 (Rank 1 - Recommended): Neutral & Scholarly Literature Mapping**
   * *Strategy:* Presents prior work purely on its own terms without explicit theoretical critique or forward-references to your method.
   * *Tone:* Objective, standard AAAI camera-ready style.
2. **Option 2 (Rank 2): Focus on Representation Granularity**
   * *Strategy:* Structures paragraphs by *entity granularity* (Triads $\rightarrow$ Compressed Endpoint Vectors $\rightarrow$ Sequence Tokens $\rightarrow$ Global Graph Tokens).
   * *Tone:* Conceptual and structural.
3. **Option 3 (Rank 3): Trimmed Information Flow Narrative**
   * *Strategy:* Retains the explicit information flow narrative, but cuts redundant critiques by ~50%.
   * *Tone:* Analytical and comparative.

---

# LaTeX Drafts

## Option 1: Neutral & Scholarly Literature Mapping (Recommended)

\section{Related Work}
\paragraph{Edge and Sign Prediction.} Early work on signed networks models edge labels using localized graph structures rather than endpoint features in isolation. Structural balance theory relies on closed triads to predict signs \citep{heider1946attitudes, cartwright1956structural}, while subsequent data-driven formulations incorporate status differences and directed triad-count statistics \citep{leskovec2010predicting, leskovec2010signed}. These approaches explicitly trace information along short local paths, but rely on fixed, hand-crafted topological heuristics rather than learned contextual representations.

\paragraph{Vertex-Centric GNNs.} Modern deep learning approaches replace fixed triad counts with message-passing GNNs adapted for signed and directed graphs, such as SGCN \citep{derr2018signed}, SiGAT \citep{huang2019signed}, SNEA \citep{li2020signed}, and SDGNN \citep{jung2021sdgnn}. These architectures aggregate incoming and outgoing signed edge messages over node neighborhoods. However, downstream edge classifiers typically condition on a static pair of endpoint embeddings, reading the label off a compressed summary vector computed for each vertex.

\paragraph{Walk and Sequence Representations.} Random-walk methods generate node representations by sampling paths through the graph \citep{perozzi2014deepwalk, grover2016node2vec}, with specialized variants designed for signed and directed networks \citep{yuan2017sne, wang2020side}. In these methods, directional path structure guides embedding optimization during training, but is discarded during inference when edge labels are scored using fixed node pairs. Sequence architectures applied to walks \citep{selvan2022crawl} retain token-level path structure, yet commonly collapse sequential outputs into pooled node-level or graph-level embeddings prior to edge classification.

\paragraph{Graph Transformers and Expressiveness Limits.} Graph Transformers utilize global self-attention and positional encodings to bypass local message-passing bottlenecks \citep{ying2021do, rampasek2022recipe}. While global attention enables long-range information exchange, standard implementations pool representation tokens back to individual nodes before scoring edges. This architectural pattern connects to broader theoretical studies on GNN expressiveness, over-squashing, and information bottlenecks \citep{alon2021on, topping2021understanding}, which examine how graph structures restrict information flow through fixed-capacity representations.

---

## Option 2: Focus on Representation Granularity

\section{Related Work}
\paragraph{Substructure and Triad Formulations.} Early signed network models operate at the substructure level, inferring edge signs from local motifs and closed loops. Balance theory \citep{heider1946attitudes, cartwright1956structural} and status-based heuristics \citep{leskovec2010predicting, leskovec2010signed} evaluate triad configurations to capture local directional constraints. While these methods target edge-adjacent subgraphs directly, their predictive power is constrained by manually defined topological features.

\paragraph{Node-Centric Embedding Models.} A major shift occurred with vertex-centric representations, which compress local graph context into per-node embeddings. Message-passing signed GNNs—such as SGCN \citep{derr2018signed}, SiGAT \citep{huang2019signed}, SNEA \citep{li2020signed}, and SDGNN \citep{jung2021sdgnn}—aggregate neighborhood signs into node vectors. Similarly, random-walk embedding methods \citep{perozzi2014deepwalk, grover2016node2vec, yuan2017sne, wang2020side} optimize node proximity in latent space. In both paradigms, edge predictions are made by reading out a pair of endpoint vectors, decoupling the edge label from its full directional neighborhood.

\paragraph{Sequential and Path-Based Encoders.} To preserve structural ordering, sequence architectures process random walks token-by-token \citep{selvan2022crawl}. By modeling walks as sequences, these methods preserve directional path continuity during processing. However, downstream edge scoring typically collapses these sequential tokens back into single node-level summaries before scoring target edges.

\paragraph{Global Graph Transformers and Capacity Constraints.} Graph Transformers \citep{ying2021do, rampasek2022recipe} replace message passing with dense attention mechanisms, allowing direct interactions across distant nodes. Despite their global scope, edge classification in these architectures continues to rely on endpoint readout layers. This structural compression reflects known expressiveness and over-squashing bottlenecks in graph architectures \citep{alon2021on, topping2021understanding}, where contextual evidence is forced through bounded-capacity node representations.

---

## Option 3: Trimmed Information Flow Narrative

\section{Related Work}
\paragraph{Edge and Sign Prediction.} Classic balance theory \citep{heider1946attitudes, cartwright1956structural} and status-based heuristics \citep{leskovec2010predicting, leskovec2010signed} evaluate localized directional paths across triads to infer edge signs. While these approaches respect directional information flow along short paths, their reliance on hand-engineered structural features limits their capacity to adapt to complex graph distributions.

\paragraph{Vertex-Centric GNNs.} Signed GNN architectures—including SGCN \citep{derr2018signed}, SiGAT \citep{huang2019signed}, SNEA \citep{li2020signed}, and SDGNN \citep{jung2021sdgnn}—aggregate directional and signed messages across local neighborhoods. Although message passing can capture orientation, downstream classifiers read edge labels exclusively from compressed endpoint vectors, bottlenecking the directional signal accumulated during aggregation.

\paragraph{Walk and Sequence Representations.} Random walk approaches for signed graphs \citep{perozzi2014deepwalk, yuan2017sne, wang2020side} leverage path traversal to capture structural contexts. However, standard walk-based models utilize path information primarily to optimize static node embeddings, discarding sequential path structures at inference time. Subsequent sequence encoders \citep{selvan2022crawl} process walks sequentially but frequently compress outputs down to node-level representations before scoring edges.

\paragraph{Graph Transformers and Information Bottlenecks.} Graph Transformers \citep{ying2021do, rampasek2022recipe} capture long-range contextual dependencies via flexible attention mechanisms. Nevertheless, edge predictions in these architectures typically rely on pooled node representations. This structural compression aligns with known over-squashing and capacity limits in message-passing networks \citep{alon2021on, topping2021understanding}, where aggregating expansive context into fixed vectors risks information loss.
