# Thesis rewrite checklist (WSDM_format_revised.tex -> thesis/thesis.tex)

Rule: minimal diff (numbers, small wording, new ablations). Every text change approved by the user
before it is applied. No table<->figure conversions. Numbers: EID H3, 10 seeds, func_logit_power,
source `experiments/edge_identity_tokens/EID_THESIS_RESULTS.md`.

Status: TODO / PROPOSED / APPROVED+APPLIED / NO CHANGE

| # | item | type | status |
|---|---|---|---|
| 0 | copy WSDM tex + bib into thesis/ (identical) | setup | done 2026-10-15 |
| 1 | Abstract | 1a: keep "3.3pp" (user: local attention is the presented model; H3 local - best baseline = 3.32). 1b: last sentence (vertex/edge split) on hold until item 20 | 1a NO CHANGE, 1b ON HOLD |
| 2 | Introduction | check | TODO |
| 3 | Related work | check | TODO |
| 4 | Contributions | check | TODO |
| 5 | Problem setting + theory | check | TODO |
| 6 | Methods (tokenizer, holdout policy, schematic caption) | DISCUSS | TODO |
| 7 | Hyperparameter table | add EID params | TODO |
| 8 | Experimental setup | check | TODO |
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
