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

## Open markers currently in the tex (verified by direct grep, 2026-08-20 — this list is exhaustive as of that check, not a curated subset)

| # | Location | Marker | Status / what's needed |
|---|---|---|---|
| 1 | Introduction, line 85 | `XXX WHERE SHOULD WE PRESENT THE MODEL NAME AND ACRONYM? XXX` | **Open, real gap.** The full name ("Proximal Edge-Walk Transformer with Ensemble Read-out") currently exists nowhere in the *rendered* paper — it only appears in a commented-out paragraph immediately above the marker (line 83, `%In directed graphs...`). Needs a decision on where the acronym gets spelled out (Abstract? First use of `\method` in the Introduction? A footnote?), not just uncommenting the old paragraph as-is (that paragraph also contains other content — an entropy-asymmetry summary and a walk-window description — that may or may not still belong there given how much the rest of the Introduction has changed since it was written). |
| 2 | Contributions, line 93 | `XXX NEW CANDIDATE. SHOULD WE KEEP IT? XXX` | Professor reviewing (2026-08-20) whether to keep the new bullet about the attention direction/vertex-edge-role split (added this session, ties to Figure 3/`fig:attndir`). Awaiting their call. |
| 3 | Contributions, line 94 | `XXXXX THE STRANGE DISTANT CORRELATIONS RESULTS TO BE REMOVED IF WE FIND MOTHING XXXXXX` | Held per explicit user instruction — this is about Figure 2 Panel B's MI-decay "bump" (slashdot090221 hops 4-8 undirected, wiki-rfa hops 7-8 directed), not the attention-direction/role finding (that's #2 above, a separate bullet). **Investigation done 2026-08-20** (`PANELB_INVESTIGATION_REPORT.md`, Thread C) — both instances confirmed real, but the mechanism behind them was only partially pinned down (real, opposite-direction sign-composition shifts in both, not fully explained by degree alone; no unified mechanism found). Per the marker's own wording ("if we find nothing"), this reads as a genuine partial/mixed result, not a clean "found something" or "found nothing" — still needs your call on whether/how to phrase a bullet around a partial finding, or leave it removed. |
| 4 | Appendix A (Pinsker relaxation), line 128 | `XXX REF XXX` inside a commented-out sentence (`%The linear form remains valid...`) | **Currently inert** (the whole sentence is commented out, doesn't render) but the underlying question is still open: is $\Hh_b(p)\le 1-\tfrac{2}{\ln 2}(\tfrac12-p)^2$ genuinely attributable to Pinsker, or is "Pinsker-type" a loose borrowed label with no clean citation? Not yet resolved — see the plan file (`~/.claude/plans/adaptive-watching-ember.md`, section B) for the two acceptable outcomes (cite properly, or drop the name and describe it as an unattributed relaxation). Low priority since it's inert either way unless this paragraph gets uncommented. |
| 5 | Methods, line 242 | `\ph{ANONYMOUS GITHUB LINK}` | Blocked on the new anonymous public repo — see `~/.claude/plans/plan-cleanup-and-local-attention-prompts.md`'s Prompt 1 (production cleanup / open-sourcing), not yet started. Camera-ready-style item, not urgent until submission is imminent. |
| 6 | Discussion and Limitations, line 353 | `XXX MIGHT NOT BE TRUE WITH NEW RESULTS ON DIRECTIONALITY XXX` | **Actively being reworked (2026-08-20), paused mid-discussion** — the whole Discussion/Limitations paragraph is getting a substantive rewrite, not just this one marker fixed in place (per-item truth audit done: item (a) is fine as-is per the user's own correction — do not tie it to the Ablation~\ref{abl:proximal} context-coverage numbers, they measure a different thing; item (b)'s "hundreds to thousands" is confirmed wrong and needs fixing; item (c) needs consolidating with the newer attention-direction/role findings rather than only citing the entropy-asymmetry one; item (d) is fine). Paused because the professor is reviewing the tex directly right now — resume once they're done, don't silently finish it underneath their review. |

## Known non-marker blocker

- **`figures/pipeline_schematic.png` does not exist on disk** — `\includegraphics` at line 186
  points at a file that isn't there, so the document won't compile end-to-end as-is. The
  source (`aaai2027/figures/pipeline_schematic.drawio`, last touched 2026-08-18) is the
  user's own hand-built diagram; only the PNG export is missing. Drop the export at that
  exact path and the reference works with no further tex changes needed. This is the only
  `\includegraphics` target in the whole document that's missing — the other 3 (Figure 2's
  combined empconf panels, Figure 3's delta heatmap, Figure 4's attndir panels) all resolve.

## Mechanical health checks (re-run 2026-08-20, all clean)

- Brace balance: 654/654.
- Citations: all 38 `\cite`/`\citep` keys used in the tex resolve against
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
- **Sections 7-8 (Discussion/Limitations, Conclusion)**: Conclusion re-verified clean (all
  headline numbers recomputed by hand and matched exactly: baseline count, mean/range AUC
  gain). Discussion/Limitations is the one section with real open work — see marker #6
  above.
- **Ethical Considerations**: read for a factual spot-check (positive-edge imbalance
  percentages match `DATASET_STATS.md` exactly); not otherwise in scope for editing without
  explicit request.

## Adjacent, not paper-text but paper-adjacent

- Two self-contained session-starter prompts exist for future work that doesn't block
  submission but the professor asked to keep visible: production cleanup for a public
  anonymous repo, and a genuine sparse/windowed local-attention implementation (current
  `LocalAttentionEncoderLayer` is dense-masked, see `MASKING.md`). Both in
  `~/.claude/plans/plan-cleanup-and-local-attention-prompts.md`, neither started.
