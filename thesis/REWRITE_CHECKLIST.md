# Thesis rewrite checklist (WSDM_format_revised.tex -> thesis/thesis.tex)

Rule: minimal diff (numbers, small wording, new ablations). Every text change approved by the user
before it is applied. No table<->figure conversions. Numbers: EID H3, 10 seeds, func_logit_power,
source `experiments/edge_identity_tokens/EID_THESIS_RESULTS.md`.

Status: TODO / PROPOSED / APPROVED+APPLIED / NO CHANGE

| # | item | type | status |
|---|---|---|---|
| 0 | copy WSDM tex + bib into thesis/ (identical) | setup | done 2026-10-15 |
| 1 | Abstract | 1a: keep "3.3pp" (user: local attention is the presented model; H3 local - best baseline = 3.32). 1b: last sentence (vertex/edge split) on hold until item 20 | 1a NO CHANGE, 1b ON HOLD |
| 2 | Introduction | check | NO CHANGE (approved 2026-10-19) |
| 3 | Related work | check | NO CHANGE (approved 2026-10-19) |
| 4 | Contributions | check (bullet 5 entropy pattern: H3 matches production, regression rerun at item 11) | NO CHANGE (approved 2026-10-19) |
| 5 | Problem setting + theory | check | NO CHANGE (approved 2026-10-19) |
| 6 | Methods: 6a walk encoding, 6b embedding (+unknown-edge identity sentence), 6c target hides identity, 6d edge-identity replacement, 6e leakage paragraph (minimal: "edge token (identity and sign)", "sign and identity hidden"), 6f algorithm line, 6g schematic legend "positive/negative edge" | DISCUSS | APPROVED+APPLIED 2026-10-19 |
| 7 | Setup paragraph (no hyperparam table exists): added EID settings sentence (rank 16, replace 0.2/70% unk, wd 0.03) | add EID params | APPROVED+APPLIED 2026-10-19 |
| 8 | Statistical evaluation paragraph | check (new ablations state their test inline, as WSDM does) | NO CHANGE (approved 2026-10-19) |
| 9 | Empirical evidence (Fig 1) | check | TODO |
| 10 | Table 1 | numbers | TODO |
| 11 | Entropy heatmap section | numbers | TODO |
| 12 | Abl: proximal attention | numbers | TODO |
| 13 | Abl: aggregator | numbers | TODO |
| 14 | Abl: single walk | numbers | TODO |
| 15 | Abl: direction flip | numbers | TODO |
| 16 | New abl: visibility grid (table) | DISCUSS | TODO |
| 17 | New abl: sign scramble / identity scramble | DISCUSS | TODO |
| 18 | New abl: pair-only attention | DISCUSS | TODO |
| 19 | Complexity | 1 sentence (identity table params) | TODO |
| 20 | Attention section (+ SHAP removal option) | DISCUSS | TODO |
| 21 | Full proofs | check | TODO |
| 22 | Discussion / limitations | DISCUSS (small) | TODO |
| 23 | Ethics | check | TODO |
