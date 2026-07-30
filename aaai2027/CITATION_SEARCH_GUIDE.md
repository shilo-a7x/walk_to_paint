# Citation search guide — every spot in `pewter_aaai.tex` that needs a real bibkey

This replaces the "see chat" pointer in `PEWTER_ASSETS_CHECKLIST.md` row #35/#8 — that
conversation got compacted, so the concrete list is written down here instead of relying on
chat history again.

**How to use this doc:** for each row, search Google Scholar / DBLP / Semantic Scholar for the
named paper, confirm it's the right one, get its real bibkey (or make one, AAAI-style —
`author-year-shortword`, matching `AuthorKit27/aaai2027.bib`'s convention), add the BibTeX entry
to `pewter_references.bib`, then replace the matching `\ph{PLACEHOLDER: ...}` in the `.tex`
with `\citep{yourkey}` (or `\citet{}` — see the command cheat-sheet below).

**Important — these are candidates from memory, not verified citations.** I recognize these
titles/venues/years from training, but I have not looked any of them up live, and some (marked
below) I'm genuinely unsure about. Per the project's no-invented-citations rule, treat every
entry below as a lead to verify, not a ready-to-use bibkey — several papers I name with
confidence on title/authors, I'm not fully sure of the exact year or venue, and a few slots I
flagged as "not sure — you'll need to search" rather than guessing.

---

## Citation command cheat-sheet (AAAI template uses natbib)

- `\citep{key}` → parenthetical: "...prior work (Author Year)." — use for background references.
- `\citet{key}` → narrative: "Author (Year) showed that..." — use when the author/paper is the
  grammatical subject of the sentence.
- `\citep{key1,key2}` → "(Author Year; Author Year)" for multiple sources in one place.
- `\citeauthor{key}` / `\citeyear{key}` → author-only / year-only, for the rare case you need
  just one part (e.g. "the original DeepWalk authors" without repeating the full citation).
- Don't use bare `\cite{}` unless you've checked what `aaai2027.sty` aliases it to — natbib's
  default `\cite` is a synonym for `\citep` in most style files, but confirm rather than assume.

---

## 1. Related Work, paragraph 1 — "Edge and sign prediction" (line ~146)

> "Structural balance theory predicts a sign from the parity of a closed triad
> `\citep{PLACEHOLDER: Heider balance theory / Cartwright-Harary structural balance}`"

- **Heider, F. (1946)**, "Attitudes and Cognitive Organization" — the original psychological
  balance-theory paper (confident on this one; it's the standard root citation for "balance
  theory" everywhere).
- **Cartwright, D. & Harary, F. (1956)**, "Structural Balance: A Generalization of Heider's
  Theory," *Psychological Review* — the graph-theoretic formalization (triads, signed graphs).
  Also confident.

> "later work replaces balance with status differences and other triad-count features
> `\citep{PLACEHOLDER: Leskovec, Huttenlocher, Kleinberg -- signed network sign prediction /
> status theory}`"

- **Leskovec, J., Huttenlocher, D., Kleinberg, J. (2010)**, "Predicting Positive and Negative
  Links in Online Social Networks," WWW 2010 — introduces status theory as an alternative to
  balance theory for sign prediction, exactly the paper this sentence is pointing at. Confident.
- Possibly also their companion paper **"Signed Networks in Social Media"** (CHI 2010, same
  authors) if you want a second citation for the same claim — search to confirm which one (or
  both) matches your phrasing better.

## 2. Related Work, paragraph 2 — "Vertex-centric GNNs" (line ~147)

> `\citep{PLACEHOLDER: SGCN}`, `\citep{PLACEHOLDER: SiGAT}`, `\citep{PLACEHOLDER: SNEA}`,
> `\citep{PLACEHOLDER: SDGNN / SGA-GSGNN or other directed signed GNN}`

- **SGCN** — Derr, T., Ma, Y., Tang, J., "Signed Graph Convolutional Network," ICDM 2018.
  Reasonably confident on title/authors, double-check year/venue.
- **SiGAT** — Huang, J. et al., "Signed Graph Attention Networks," ICANN 2019. Confident on
  title, less confident on the exact author list — verify.
- **SNEA** — a signed-network embedding paper using graph attention, title close to "Learning
  Signed Network Embedding via Graph Attention" — **not confident**, this is the weakest lead
  in this list; search DBLP for "SNEA signed network embedding attention" directly rather than
  trusting this title.
- **SDGNN** — Huang, J. et al., "SDGNN: Learning Node Representation for Signed Directed
  Networks," AAAI 2021. Reasonably confident.
- **SGA** (the augmentation applied to GSGNN in your Baselines list) — per `CLAUDE.md`/your own
  note this is "Signed Graph Augmentation," described elsewhere in this project as a 2024 paper —
  **search for the exact venue** (you mentioned NeurIPS 2024 in an earlier note to me; confirm
  directly rather than trusting my memory here, I don't have a confident independent match).
- **CSG / CSG-GSGNN / CopulaLSP** — these three (also needed for the Baselines paragraph, #4
  below) are not papers I can confidently identify from memory at all. Don't guess from this
  doc — these need a direct search (likely worth checking whichever repo/README first documented
  them in `baselines/`, since that's probably where their original paper reference lives).

## 3. Related Work, paragraph 3 — "Walk and sequence representations" (line ~148)

> `\citep{PLACEHOLDER: DeepWalk}`, `\citep{PLACEHOLDER: node2vec}`,
> `\citep{PLACEHOLDER: signed/directed walk-based embedding method}`,
> `\citep{PLACEHOLDER: walk-based sequence/RNN or transformer node classifier}`

- **DeepWalk** — Perozzi, B., Al-Rfou, R., Skiena, S., "DeepWalk: Online Learning of Social
  Representations," KDD 2014. Confident.
- **node2vec** — Grover, A., Leskovec, J., "node2vec: Scalable Feature Learning for Networks,"
  KDD 2016. Confident.
- **Signed/directed walk-based embedding** — candidate: **SIDE**, Kim, J. et al., "SIDE:
  Representation Learning in Signed Directed Networks," WWW 2018. Reasonably confident on title,
  double-check authors/year.
- **Walk-based sequence/RNN or transformer node classifier** — this slot is the vaguest one in
  the whole document; I don't have a specific confident paper in mind. Search terms to try:
  "random walk sequence model node classification," "graph2seq," "walk transformer graph
  representation" — pick whichever real paper best matches the sentence's claim (a model that
  processes a walk token-by-token but pools to one vector before classifying).

## 4. Related Work, paragraph 4 — "Graph transformers" (line ~149)

> `\citep{PLACEHOLDER: Graphormer}`, `\citep{PLACEHOLDER: GPS / SAT or other graph transformer}`

- **Graphormer** — Ying, C. et al., "Do Transformers Really Perform Bad for Graph
  Representation?," NeurIPS 2021. Confident.
- **GPS** — Rampášek, L. et al., "Recipe for a General, Powerful, Scalable Graph Transformer,"
  NeurIPS 2022. Confident on title, double-check year.
- **SAT** (alternative/additional to GPS) — Chen, D. et al., "Structure-Aware Transformer for
  Graph Representation Learning," ICML 2022. Reasonably confident.

## 5. Related Work, paragraph 5 — "Oversquashing and information bottlenecks" (line ~150)

> `\citep{PLACEHOLDER: Alon \& Yahav 2021}`, `\citep{PLACEHOLDER: Topping et al. 2022}`

- **Alon, U., Yahav, E. (2021)**, "On the Bottleneck of Graph Neural Networks and its Practical
  Implications," ICLR 2021. Confident — this is the paper that coined "oversquashing" in the
  GNN context.
- **Topping, J. et al. (2022)**, "Understanding Over-squashing and Bottlenecks on Graphs via
  Curvature," ICLR 2022. Confident.

## 6. Problem Setting and Notation — Fano's inequality (line ~159)

> "We use Fano's inequality throughout: ... XXXXXX REFS HERE AND ALL ALONG XXXXXXX" (the general
> marker at the top of the section applies here too, even though there's no explicit
> `\citep{PLACEHOLDER}` slot yet for Fano specifically)

- Standard citation for Fano's inequality in the ML/info-theory literature is the textbook, not
  Fano's own 1961 paper: **Cover, T. M. & Thomas, J. A., *Elements of Information Theory*** (2nd
  ed., 2006), the theorem is stated there in the form the paper uses. Confident this is the
  conventional citation choice — most ML papers cite the textbook rather than Fano's original
  1961 report, but if you'd rather cite Fano directly, that's **Fano, R. M. (1961),
  *Transmission of Information*, MIT Press** — either is defensible, your call.

## 7. Limitations section / Appendix — WL-coloring, GNN expressiveness (line ~178, `XXXXXX REFS
HERE AND ALL ALONG XXXXXXX`)

> "any message-passing GNN factors through it [the stable WL coloring]"

- **Xu, K. et al. (2019)**, "How Powerful are Graph Neural Networks?," ICLR 2019 — the GIN paper,
  which is also the paper that establishes the WL-GNN expressiveness equivalence this sentence
  relies on. Confident, and this is a **double-duty citation** — it's also the right citation for
  GINEConv in Baselines (#9 below), so you likely want it in both spots.
- **Morris, C. et al. (2019)**, "Weisfeiler and Leman Go Neural: Higher-Order Graph Neural
  Networks," AAAI 2019 — an alternative/companion citation for the same WL-equivalence claim,
  published around the same time as Xu et al. Confident on title, worth citing alongside Xu et al.
  rather than instead of it, since both are commonly cited together for this claim.

## 8. Methods, "Per-walk classifier" paragraph (line ~290) — currently uncited, found this round

> "passed through a Transformer encoder..." / "a masked-language-modeling-style objective"

- **Transformer architecture** — Vaswani, A. et al. (2017), "Attention Is All You Need," NeurIPS
  2017. Confident — essentially mandatory to cite when introducing "a Transformer encoder" at
  all.
- **MLM-style objective** — Devlin, J. et al. (2019), "BERT: Pre-training of Deep Bidirectional
  Transformers for Language Understanding," NAACL 2019. Confident — standard citation for
  "masked-language-modeling-style" phrasing.

## 9. Methods, "Baselines" paragraph (line ~327)

> `\citep{PLACEHOLDER: Xu et al. 2019 GIN / Hu et al. edge-featured GINE}` for GINEConv, plus
> `\citep{PLACEHOLDER}` for SiGAT, SNEA, CSG, CSG-GSGNN, GSGNN+SGA, CopulaLSP

- **GIN** — same Xu et al. (2019) ICLR paper as #7 above.
- **GINE** (the edge-featured variant, the actual layer type PEWTER/baselines call "GINEConv") —
  Hu, W. et al. (2020), "Strategies for Pre-training Graph Neural Networks," ICLR 2020 —
  introduces the edge-feature-aware GIN variant. Reasonably confident this is the right paper,
  double-check since "GINEConv" is also just the name of PyTorch Geometric's implementation
  class, which cites this paper in its own docs.
- **SiGAT, SNEA, SDGNN/GSGNN** — same candidates as #2 above (this list is intentionally
  decoupled from Related Work's citations per the project's own note, but the underlying papers
  are the same models, so the same bibkeys apply in both places once you have them).
- **CSG, CSG-GSGNN, CopulaLSP** — not identifiable from memory, flagged in #2 above; CopulaLSP
  you already noted elsewhere as ICLR 2026 — that alone should make it easy to find directly on
  OpenReview/DBLP once ICLR 2026 papers are indexed.
- **SGA** — same as #2 above.

## 10. Methods, "Statistical evaluation" paragraph (line ~331)

> "Hanley--McNeil/DeLong nonparametric estimator for AUC" — named in prose but not yet cited

- **Hanley, J. A. & McNeil, B. J. (1982)**, "The Meaning and Use of the Area under a Receiver
  Operating Characteristic (ROC) Curve," *Radiology*. Confident — this is the standard citation
  for AUC standard-error estimation.
- **DeLong, E. R., DeLong, D. M., Clarke-Pearson, D. L. (1988)**, "Comparing the Areas under Two
  or More Correlated Receiver Operating Characteristic Curves: A Nonparametric Approach,"
  *Biometrics*. Confident — the standard companion citation, especially if you end up comparing
  correlated AUCs (e.g. PEWTER vs. a baseline on the same test set) rather than a single AUC's
  SE in isolation.

## 11. Introduction — general "put refs all along" marker (line 104)

The Introduction's own prose (vertex-centric GNN framing, bottleneck argument) is narrative/
motivational rather than a series of citable claims, so most of it doesn't need its own
citations — the underlying claims get their citations where they're formally introduced
(Related Work, Problem Setting). One optional addition: the sentence introducing "vertex centric
graph neural network (GNN)" itself could carry a citation to a canonical GNN paper right there
(e.g. **Kipf, T. & Welling, M. (2017)**, "Semi-Supervised Classification with Graph Convolutional
Networks," ICLR 2017, or **Veličković, P. et al. (2018)**, "Graph Attention Networks," ICLR
2018) rather than waiting until Related Work — your call whether that's wanted here or is
redundant with Related Work's own GNN citations two paragraphs later.

---

## Summary table

| # | Location | Confidence | Notes |
|---|---|---|---|
| 1 | Balance theory (Heider; Cartwright-Harary) | High | |
| 1 | Status theory (Leskovec-Huttenlocher-Kleinberg) | High | |
| 2 | SGCN | Medium | verify year/venue |
| 2 | SiGAT | Medium | verify author list |
| 2 | SNEA | **Low** | weakest lead, search directly |
| 2 | SDGNN | Medium | |
| 2 | SGA | **Low** | you flagged NeurIPS 2024 elsewhere, confirm |
| 2/9 | CSG, CSG-GSGNN, CopulaLSP | **None** | not identifiable from memory, search from scratch |
| 3 | DeepWalk | High | |
| 3 | node2vec | High | |
| 3 | SIDE (signed walk embedding) | Medium | |
| 3 | walk-based sequence/transformer classifier | **None** | vaguest slot, needs a real search |
| 4 | Graphormer | High | |
| 4 | GPS | Medium | verify year |
| 4 | SAT | Medium | |
| 5 | Alon & Yahav 2021 | High | |
| 5 | Topping et al. 2022 | High | |
| 6 | Fano / Cover & Thomas | High | |
| 7 | Xu et al. 2019 (GIN/WL) | High | double-duty with #9 |
| 7 | Morris et al. 2019 (WL-GNN) | High | |
| 8 | Vaswani et al. 2017 (Transformer) | High | |
| 8 | Devlin et al. 2019 (BERT/MLM) | High | |
| 9 | Hu et al. 2020 (GINE) | Medium | |
| 10 | Hanley & McNeil 1982 | High | |
| 10 | DeLong et al. 1988 | High | |
| 11 | Kipf & Welling 2017 / Veličković et al. 2018 | High | optional, your call |

**Bottom line:** roughly two-thirds of these slots I'm confident enough about to just verify and
grab the bibkey; SNEA, SGA's exact venue, CSG/CSG-GSGNN/CopulaLSP, and the walk-based-sequence-
classifier slot need a real search from scratch rather than confirming a memory.
