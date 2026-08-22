# PEWTER paper — assets checklist

**Rewritten from scratch 2026-08-20.** The previous version of this file (written against
`pewter_aaai.tex`, the pre-WSDM-format draft) had drifted far enough out of sync with the
current `aaai2027/WSDM_format_revised.tex` that patching it further risked being actively
misleading (it still described a 4-panel Figure 1, a since-dropped Ablation, GINEConv as a
live baseline, and 15+ `BLOCKED-CITATIONS` rows that are all resolved now). Per the standing
project convention, day-by-day narrative/history lives in `aaai2027/PAPER_CLOSEOUT_LOG.md`,
not here — this file only tracks **current, live status**: what's actually still open in the
tex right now, and a one-line pointer to where the "why"/"how" for everything else lives.

**Source of truth ranking, if anything here conflicts**: (1) the tex file itself, (2)
`aaai2027/STATISTICAL_TESTS_AUDIT.md` for any statistical-test claim, (3) this file, (4)
`aaai2027/PAPER_CLOSEOUT_LOG.md` for historical narrative only.

## Open markers currently in the tex (verified by direct grep, 2026-08-21 — this list is exhaustive as of that check, not a curated subset; all 6 rows from the 2026-08-20 version of this table are gone — either resolved this session or removed by the professor's own edits, see `PAPER_CLOSEOUT_LOG.md`'s 2026-08-21 entry)

| # | Location | Marker | Status / what's needed |
|---|---|---|---|
| 1 | Walk sampler paragraph, line ~180 | `XXXXXX BE EXPLICIT HERE ON HOW YOU CHOSE THE LENGTH XXXXX` (anchor pass) + `XXXX AGAIN HOW MANY XXX` (fill pass) | **Deferred by explicit user instruction**, pending a Methods-section decision (not yet made) on whether/where to state $L=80$ explicitly vs. cross-reference Setup. Don't resolve without that decision first. |
| 2 | Same paragraph, line ~180 | `XXX This sentence is very unclear XXX` (mid-sentence, on "each one first finds a short directed path leading into a random vertex...") | **New 2026-08-21, not yet resolved.** Separate issue from #1 above (clarity, not a missing number) but in the same sentence — undecided whether to bundle with #1's fix (would touch the sentence twice if done separately) or handle now on its own. |
| 3 | Datasets/Baselines area, line ~210 | `XXXXX Asked an LLM for more recent methods, and got this two that we should compare...` | This is the EdgeSketch+ / CopulaLSP-adjacent citation candidate — see the plan file's Group E for the full verification (EdgeSketch+ is real, NeurIPS 2025 not 2026 as originally stated; the second candidate is already covered by the existing `sung2026scalable` citation). Not yet acted on. |
| 4 | ~~Figure 2 area, line ~258~~ | ~~`XXXXX in figure 2, please combine the labels of D and E...`~~ | **Resolved 2026-08-23.** Removed Panel D's own legend (`plot_empconf_panelD_signagreement_auc.py::plot_perdataset`, both panels share the same `DATASET_COLORS` palette) so only Panel E's legend remains (already carries the "not robust" hatch entry). Regenerated Panel D/E and the combined `aaai2027/figures/empconf_panels_abcde_combined_sidebyside.png` via `combine_empconf_panels_abcde_sidebyside.py`. Marker removed from the tex. |
| 5 | Table 1 area, line ~275 | `XXXX ... lets also add Precision Recall AUC or perhaps F1...` | **Investigated 2026-08-21, macro-F1 computation now underway 2026-08-22/23** — decided on macro-F1 (PR-AUC stays blocked, still only 2/9 rows feasible). Pewter/SiGAT/SNEA/CopulaLSP macro-F1 verified against real AUC ground truth (caught and fixed a per-seed split-file bug along the way). node2vec's macro-F1 rerun (patched `baselines/node2vec/run_node2vec.py` to also save `tst_macro_f1`) finished 2026-08-23 — see `PAPER_CLOSEOUT_LOG.md`. Published F1 numbers for GCN/GAT/SGCN/GSGNN obtained from the user (SGA paper's own table, bare non-augmented rows). **Still open: how to add it to Table 1 (new column vs. footnote), and writing it into the tex.** |
| 6 | ~~Ablations subsection, line ~331~~ | ~~`XXXX There are three intereseting ablations needed to be checked...`~~ | **Resolved 2026-08-22.** 180-job campaign finished; all three ablations (DIRFLIP, MASKNODE, MASKEDGE) written into the tex as three new `\refstepcounter{ablation}` paragraphs, in the professor's A/B/C order, backed by one-sided paired Wilcoxon tests (10 seeds) against the local-attention baseline. Marker removed. |
| 7 | ~~Where Pewter's attention goes, line ~357~~ | ~~`XXXX please have larger fonts in all figures... use "Offset" instead of i,j... XXXX`~~ | **Resolved 2026-08-23**, one gap flagged below. Every `dpi=` in `scripts/paper_figures/*.py` bumped to 300 (was 150-200) and every currently-included figure regenerated at the new resolution. Fonts bumped across attndir Panels A/B/C/D(shap) and the delta-heatmap script (`plot_multiseed_entropy_heatmaps.py`). Panel A: `"head {h}"` → `"Head {h}"`, `"d = j − i"` → `"Offset $d$"` (matches the caption's own "signed offset" wording) — `scripts/paper_figures/plot_attndir_panelA_headgrid.py`. **Not done: the SVG-for-camera-ready suggestion** — the combined figures are built by `combine_*.py` scripts stitching pre-rendered raster PNG panels via `imshow`, so a real vector export would need re-architecting those scripts, not just a savefig format swap; left as PNG at 300dpi given the deadline. Marker removed from the tex. |
| 8 | ~~Same subsection, line ~363~~ | ~~`XXXXX HERE PLEASE ALSO REMOVE CAUSAL in the figure itself Shapley is not causal XXXXX`~~ | **Resolved 2026-08-23.** The rendered Panel D subplot title (`scripts/paper_figures/plot_shap_edge_directionality.py:69`, `ax.set_title(...)`) said "Causal contribution..." — fixed to "Shapley contribution...". Regenerated `aaai2027/figures/shap_edge_directionality.png` and the combined `aaai2027/figures/attndir_panels_abcd_combined.png` (via `combine_attndir_panels.py`), which is what Figure 4 actually includes. The `XXXX` marker itself was also removed from the tex (line ~363). |
| 9 | Methods, line ~238 | `\ph{ANONYMOUS GITHUB LINK}` | Blocked on the new anonymous public repo — see `~/.claude/plans/plan-cleanup-and-local-attention-prompts.md`'s Prompt 1 (production cleanup / open-sourcing), not yet started. Camera-ready-style item, not urgent until submission is imminent. |

**Resolved this session, no longer open**: the old row #6 (Discussion/Limitations directionality marker) is moot — per explicit user instruction ("the prof removed this section content deliberately so we can remove the section header too"), `\section{Discussion, Limitations and conclusions}` was deleted entirely, not filled back in. The old rows #1-4 (model name/acronym, Contributions "new candidate", Panel-B-bump, Pinsker `XXX REF XXX`) are all gone from the tex as of this check — either resolved or removed by the professor's own edits since 2026-08-20; not re-verified individually, just confirmed absent by fresh grep.

## Known non-marker blocker

- **`figures/pipeline_schematic.png` does not exist on disk** — `\includegraphics` at line 186
  points at a file that isn't there, so the document won't compile end-to-end as-is. The
  source (`aaai2027/figures/pipeline_schematic.drawio`, last touched 2026-08-18) is the
  user's own hand-built diagram; only the PNG export is missing. Drop the export at that
  exact path and the reference works with no further tex changes needed. This is the only
  `\includegraphics` target in the whole document that's missing — the other 3 (Figure 2's
  combined empconf panels, Figure 3's delta heatmap, Figure 4's attndir panels) all resolve.

## Mechanical health checks (re-run 2026-08-21, all clean)

- Brace balance: 613/613 (down from 654 as of 2026-08-20 — reflects the professor's own
  edits plus this session's Discussion/Limitations/Conclusion section removal, not a bug).
- Citations: all `\cite`/`\citep` keys used in the tex resolve against
  `pewter_references.bib`; zero missing. (The old checklist's many `BLOCKED-CITATIONS` rows
  are all resolved — don't resurrect them.)
- `\appendix` and the section order around it: correct, no stray backslash issues (the
  `\appendix`-lost-its-backslash bug and the stray lone-`\`-before-`\end{document}` bug from
  an earlier professor-edit resync are both fixed on disk).
- No LaTeX toolchain available in this environment — these are the available substitute for
  a real compile check, not a replacement for one. Flag to the user before submission if an
  actual compiled-PDF check becomes necessary.

## Full-document status, by area

- **Math (both Propositions, both proof sketches, Appendix's full proofs, the general-$c$
  extension)**: professor-verified correct (2026-08-20 pass, marker resolution recorded in
  `PAPER_CLOSEOUT_LOG.md`). One new marker added at the professor's request flagging
  Proposition 2's capacity bound as near-vacuous in practice (line ~171) — that's an
  intentional annotation for the professor, not an outstanding bug.
- **Statistical claims**: audited claim-by-claim in `STATISTICAL_TESTS_AUDIT.md` — that file
  is the live source of truth for "does this claim have a real test backing it," not this
  checklist. As of 2026-08-20 every catalogued claim has an adequate test or an explicit
  recommendation; the one fabricated/unverified number found this session (§14, vertex/edge
  vs. AUC-boost correlation) was corrected and the underlying sentence was dropped from the
  paper.
- **Figures**: 3 of 4 render-ready (see blocker above for the 4th). All figure-generating
  scripts follow the `extract_<name>.py`/`plot_<name>.py`/`combine_<name>.py` convention
  described in `CLAUDE.md`'s "PEWTER paper — file map and conventions" section.
- **Tables**: Table 1 (`tab:result1`) and Ablation B (`tab:ablationB`) both real, multiseed
  (10-seed mean±std throughout), no known open issues.
- **Sections 1-5**: read end-to-end for coherence this session, clean.
- **Section 6 (Results)**: extensively reworked this session (SOTA subsection filled in,
  Ablation A given a real paired test, duplicate paragraph removed, short-walk/local-window
  phrasing bugs fixed, `Where \method's attention goes` subsection written and reframed
  around both direction and vertex/edge role). See `PAPER_CLOSEOUT_LOG.md` for the detailed
  narrative.
- **Discussion/Limitations/Conclusion**: **section removed entirely, 2026-08-21**, per
  explicit user instruction (the professor deliberately emptied it; the header itself was
  deleted too, not filled back in). The paper now goes straight from the attention-
  directionality figure (end of Results) to Ethical Considerations. No open work here unless
  the section is reinstated later.
- **Ethical Considerations**: read for a factual spot-check (positive-edge imbalance
  percentages match `DATASET_STATS.md` exactly); not otherwise in scope for editing without
  explicit request.

## Adjacent, not paper-text but paper-adjacent

- Two self-contained session-starter prompts exist for future work that doesn't block
  submission but the professor asked to keep visible: production cleanup for a public
  anonymous repo, and a genuine sparse/windowed local-attention implementation (current
  `LocalAttentionEncoderLayer` is dense-masked, see `MASKING.md`). Both in
  `~/.claude/plans/plan-cleanup-and-local-attention-prompts.md`, neither started.
