# Thesis rewrite checklist (WSDM_format_revised.tex -> thesis/thesis.tex)

Rule: minimal diff (numbers, small wording, new ablations). Every text change approved by the user
before it is applied. No table<->figure conversions. Numbers: EID H3, 10 seeds, func_logit_power,
source `experiments/edge_identity_tokens/EID_THESIS_RESULTS.md`.

Status: TODO / PROPOSED / APPROVED+APPLIED / NO CHANGE

| # | item | type | status |
|---|---|---|---|
| 0 | copy WSDM tex + bib into thesis/ (identical) | setup | done 2026-10-15 |
| 1 | Abstract | 1a: keep "3.3pp" (user: local attention is the presented model; H3 local - best baseline = 3.32). 1b: last sentence (vertex/edge split) on hold until item 20 | 1a NO CHANGE, 1b APPLIED 2026-10-22 ("a preference of attention for vertex over edge tokens") |
| 2 | Introduction | check | NO CHANGE (approved 2026-10-19) |
| 3 | Related work | check | NO CHANGE (approved 2026-10-19) |
| 4 | Contributions | check (bullet 5 entropy pattern: H3 matches production, regression rerun at item 11) | NO CHANGE (approved 2026-10-19) |
| 5 | Problem setting + theory | check | NO CHANGE (approved 2026-10-19) |
| 6 | Methods: 6a walk encoding, 6b embedding (+unknown-edge identity sentence), 6c target hides identity, 6d edge-identity replacement, 6e leakage paragraph (minimal: "edge token (identity and sign)", "sign and identity hidden"), 6f algorithm line, 6g schematic legend "positive/negative edge" | DISCUSS | APPROVED+APPLIED 2026-10-19 |
| 7 | Setup paragraph (no hyperparam table exists): added EID settings sentence (rank 16, replace 0.2/70% unk, wd 0.03) | add EID params | APPROVED+APPLIED 2026-10-19 |
| 8 | Statistical evaluation paragraph | check (new ablations state their test inline, as WSDM does) | NO CHANGE (approved 2026-10-19) |
| 9 | Empirical evidence (Fig 1) | check (data/baseline-only, no Pewter mention) | NO CHANGE (approved 2026-10-19) |
| 10 | Table 1: 10a Pewter rows (H3) APPLIED; 10b rule sentence -> local attention headline + results sentence (3.3, +0.9 wiki-elec to +4.8 b-alpha) APPLIED; 10c intro paragraph "edge tokens", "sign and identity" APPLIED. deferred sentence -> "conditioned directly on nearby vertices and edges" (O1, 2026-10-20) | numbers | APPROVED+APPLIED |
| 11 | Entropy heatmap section: figure already H3; regression rerun on H3 (eid_delta_regression.py) - text claims still exact (otc both axes, epinions source, p<0.01); "most bins" = 64/87 pooled | numbers | NO CHANGE (approved 2026-10-19) |
| 12 | Abl: proximal attention (H3 vs H3_FULL; local wins 4/6; dropped "order of magnitude below std"; "at most 0.003") | numbers | APPROVED+APPLIED 2026-10-19 |
| 13 | Abl: aggregator: 13a table rows (H3) + caption 0.0025; 13b paragraph: 0.0025, std clause dropped, Fisher best on 4 | numbers | APPROVED+APPLIED 2026-10-19 |
| 14 | Abl: single walk: "under 1.3pp" (b-alpha 1.28); fixed WSDM glitch "AUC rises only slightly"; other claims hold | numbers | APPROVED+APPLIED 2026-10-19 |
| 15 | Abl: direction flip: H3 numbers; significant 5/6, b-alpha n.s. (p=0.14), one-sided as WSDM | numbers | APPROVED+APPLIED 2026-10-19 |
| 16 | New abl: visibility grid: paragraph (F5, one-sided, + dirflip ref) + table (V/I/S columns, option 5 for now) after dirflip | DISCUSS | APPROVED+APPLIED 2026-10-20 (header style may revisit: option 3 two-level "Visible") |
| 17 | sign scramble / identity scramble: LEFT OUT. Identity scramble = consistent relabeling in train+eval -> no information removed, null by construction (design flaw). Sign scramble = informationally same as grid row "signs hidden" (also scrambled in training) | DISCUSS | NO CHANGE (approved 2026-10-20) |
| 18 | New abl: pair-only attention: P2 paragraph (facts + embeddings-carry-most sentence) after grid table | DISCUSS | APPROVED+APPLIED 2026-10-20 |
| 19 | Complexity: 19a param sentence DROPPED (vertex table never mentioned either); 19b "4 to 70 minutes" (same rule as WSDM: shortest-longest single run) | numbers | APPROVED+APPLIED 2026-10-20 |
| 20 | Attention section: 20a figure B-D = mean over 10 models +-std (A1: panel A split 42), SHAP removal = identity+sign; 20b fwd/bwd paragraph APPLIED (option ii, ending trimmed); 20c vertex/edge APPLIED (descriptive); 20d SHAP APPLIED (decay + hop-1 direction + agreement); 20e caption APPLIED (panel A = Wiki-RfA split 42, no recurrence claim; self excluded from B/C; B-D = 10-model mean +-std); 20f abstract 1b APPLIED (option a) | DISCUSS | APPROVED+APPLIED 2026-10-22 |
| 21 | Full proofs | check (pure math; only Pewter mention is the aggregator table caption, done in 13) | NO CHANGE (approved 2026-10-20) |
| 22 | Discussion / limitations | DISCUSS (small): para 1 option B (no edge-identity sentence), para 2 unchanged ("approx 1pp" kept) | NO CHANGE (approved 2026-10-22) |
| 23 | Ethics | check (per-edge vectors store only public data) | NO CHANGE (approved 2026-10-20) |
| 24 | Methods: 4 references to the pipeline schematic (walk encoding; window, merged with position-index ref; Stage 1; Stage 2) | text | APPROVED+APPLIED 2026-10-26 |
| 25 | Tables: dataset table \small (single col); visibility + aggregator tables -> table*[t!], \small, no resizebox | layout | APPROVED+APPLIED 2026-10-26 (check placement/fit when compiling) |
| 26 | Contributions rewritten: 5 bullets with bold labels, plain wording; old bullet 2 (claimed WL residual entropy is quantified - not done in paper) merged into 'Evidence'; new 'What drives the gain' bullet | text | APPROVED+APPLIED 2026-10-29 |
| 27 | Attention/SHAP figure: (B),(C) labels above panels (combine_attndir_panels LABEL_LIFT_BC, HSPACE; off by default) | figure | DONE 2026-10-26 |
| 28 | Empirical-evidence figure: A fonts x1.5, D canvas 0.75, E width 0.75 + 2-col legend, row gap 0.08 (FIG_SCALE/HSPACE options, off by default); rendered by make_thesis_figures.empconf_figure | figure | DONE 2026-10-26 |
| 29 | Comment audit: line 120 (linear Fano remark) stays commented; proof Jensen middle step RESTORED + "chain rule" -> "definition of conditional entropy"; "multiplicity" line stays commented; dropped-ablation comment blocks, \ph macro, amssymb removed by user (intentional) | math/text | APPROVED+APPLIED 2026-10-29 |
| 30 | Layout/de-anonymization: appendix = proofs only, after references, before end{document}; aggregator table moved to Ablations; [sigconf] (no review/anonymous), WSDM header kept ("as published"); authors Avital + Louzoun (acmart fields); intro 'minor corollary' -> cites Corollary~cor:endpoint. OPEN: code link still anonymous; B1-B4 tidy-ups | layout | APPLIED 2026-10-29 |
| 31 | Outline: section 4 'The Pewter Model'; Datasets/Baselines/table -> Experimental Setup; Complexity -> paragraph at end of section 4; title case; refs updated. Consistency: dropped (A)/(B)/(C) from empirical-evidence paragraph headings (clashed with figure panels; fixed wrong '(A)' ref), panel refs unified to '(A)', hard-coded 'Proposition 1' -> \ref | layout | APPROVED+APPLIED 2026-10-29 |
| 32 | Acknowledgments section removed (not required by WSDM; NeurIPS-style funding line) | text | APPROVED+APPLIED 2026-10-29 |
