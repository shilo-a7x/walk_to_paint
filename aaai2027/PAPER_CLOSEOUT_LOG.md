# PEWTER/WSDM paper closeout — session log (archived detail)

Archived out of `CLAUDE.md` on 2026-08-18 to keep the live file short. This is a
day-by-day log of the WSDM closeout phase — full narrative, including bugs found and
fixed along the way. CLAUDE.md's "Open threads" and "PEWTER paper" sections now only
summarize the standing rules and current status; this doc has the "why"/"how" detail.
If anything here conflicts with `aaai2027/PEWTER_ASSETS_CHECKLIST.md` or
`aaai2027/STATISTICAL_TESTS_AUDIT.md`, those two are the live source of truth, not this
log.

## OPEN WORKSTREAMS — audited 2026-07-19

**Active / not started, real open work, roughly priority order:**
- `hello-so-i-have-unified-valiant.md` — **PEWTER paper (aaai2027/), ACTIVE, top priority,
  deadline-critical.** Abstract due 2026-07-28, full paper 2026-07-31 — as of this audit
  that's 9 days out. Per the plan's own 2026-07-16 status update, most of
  Results/Discussion/bib/Supplementary/final-QA tiers are NOT STARTED. Confirmed
  in-progress and real (2026-07-19) — take this as the top-priority workstream when
  triaging session time against everything else below.
  - **2026-08-17: paper is now WSDM-targeted (`aaai2027/WSDM_format_revised.tex`), in a
    closeout/punch-list phase.** Resume via
    `~/.claude/plans/adaptive-watching-ember.md` (13-item closeout list, execution in
    progress). **node2vec is the one non-GNN baseline added to Table 1** — fully aggregated,
    `baselines/node2vec/results_canonical/summary.csv`, 6 datasets × 10 seeds, weaker than
    every GNN baseline everywhere (safe, uncontroversial reference row). Three other non-GNN
    candidates were tried and rejected, in this order: **POLE**, infeasible on this hardware
    (dense O(n²) similarity-matrix memory blows up past ~423GB available on epinions/
    slashdot090221). **SLF**, technically ran (`baselines/SLF/`, 6×10 complete) and actually
    beat SiGAT on 3/6 datasets, but only trained 10 fixed epochs with no real validation-based
    model selection — pulled rather than let an under-scrutinized result reshape the paper's
    headline margins (on wiki-rfa the gap over SLF was +0.2pp, within noise). **SIGNet**
    (`baselines/SIGNet/`, `baselines/SIGNet_repo/`, isolated `signet_env` conda env with a
    patched `setup.py` pointing GSL at the conda prefix — build succeeds, imports fine): its
    C++ training loop (`cpp_signet.cpp::TrainSEINEThread`) never honors its own sample-count
    stopping condition — `count` blows past `total_samples/num_threads+2` by 1000%+ and keeps
    going indefinitely (confirmed via debug printfs showing `total_samples` itself stays
    correct throughout, so it's a real loop-logic bug in the ~2018 reference implementation,
    not an environment/build issue). Not root-caused further (time-boxed); don't resume this
    without a fix to that stopping condition. **Do not add SLF or SIGNet numbers anywhere
    without redoing this investigation** — both were deliberately reverted, not abandoned
    mid-stream.
  - **Complexity claim (Sec 6.6, O(L²d)→O(Lwd) for local attention):** confirmed correct as
    a *theoretical* claim (user's professor signed off 2026-08-17) but not realized by the
    current `LocalAttentionEncoderLayer` implementation, which computes full dense L×L
    attention and applies the window as a post-hoc mask — no real sparse/windowed
    computation happens, so no wall-clock speedup exists today (see `MASKING.md`'s
    benchmark table). The paper states the claim in its intended/theoretical form with no
    benchmark (no time, no appendix room in WSDM's page limit). **Future-work item, not
    scheduled**: implement genuine sparse/windowed attention to actually realize this
    saving — e.g. if a reviewer asks for the benchmark. Don't start this without the user's
    explicit go-ahead.
  - **2026-08-18: professor overrode `WSDM_format_revised.tex`/`pewter_references.bib`
    directly, mid-closeout — resync in progress, resume via
    `~/.claude/plans/adaptive-watching-ember.md`'s "MAJOR RESYNC" section (and its C-bis/
    C-ter follow-up overrides, same day).** New standing rules from this round, apply to
    all future work on this paper: (1) the professor's `pewter_references.bib` is
    canonical as-is — never edit a bib entry, only fix `\cite`/`\citep` calls in the tex to
    match existing keys; (2) any bib addition/removal needs the user's explicit sign-off
    before editing, no more unilateral verify-and-add; (3) math-proof findings (the
    Propositions/Lemma/Corollary/Appendix proofs) must be presented and confirmed with the
    user before any edit, never silently patched. **GINE dropped from the paper** (Table 1,
    Baselines paragraph, all baseline-count sentences) per explicit user instruction — a
    real edge-feature/leakage concern was found (`run_with_our_splits.py`'s
    `train_edge_attr` is the raw unembedded sign scalar, and the same train edges back both
    the message-passing input and the loss — not a test-time leak, but a training-time
    self-referential shortcut specific to GINEConv's mechanism vs. the balance-theory
    baselines' channel-separated approach). Only restore if a small learned-edge-embedding
    experiment (one seed, bitcoin-alpha/bitcoin-otc first) succeeds — not yet run, low
    priority backlog item. Dataset display names are now canonicalized everywhere in the
    paper to Bitcoin-alpha/Bitcoin-otc/Epinions/Slashdot/Wiki-elec/Wiki-RfA (never the raw
    `slashdot090221` key in rendered text). New deliverable, not part of the paper itself:
    `aaai2027/STATISTICAL_TESTS_AUDIT.md` catalogs every empirical "A beats B" claim in the
    paper against what statistical test (if any) backs it, with per-claim fix/keep/drop
    recommendations — read this before adding or touching any significance claim. The
    paper's edge-vs-vertex/directionality claim (new Abstract ending, stub Section 6.1
    paragraph D) is **not yet backed by real data** — do not draft its prose until the
    investigation in the plan's "Section D" lands.
  - **2026-08-18: walk-budget κ selection was done via test AUC, not validation AUC —
    confirmed and (partially) resolved.** `E25_BUDGET_SWEEP_RESULTS.md`/
    `E26_WIKI_SWEEP_RESULTS.md` (the sweeps behind every dataset's `configs/<ds>.yaml`
    `num_walks`) picked each dataset's κ by comparing **test AUC** across the grid — no
    val AUC column exists in either doc. Re-extracted `val_auc_epoch` from each grid
    point's TensorBoard logs (best-checkpoint value, no retraining) and compared: on
    the actual swept grid (excluding the old-production-budget reference points, which
    aren't part of the real grid), bitcoin-alpha and bitcoin-otc's picks are unaffected
    (val AUC agrees, still climbing to 5×); epinions and slashdot090221's picks don't
    change either (epinions' val AUC declines even more clearly toward floor than test
    AUC suggested; slashdot's 3× vs. 5× gap under val AUC is 0.02pp, noise-level). But
    **wiki-elec and wiki-rfa show a real, non-trivial disagreement** — val AUC peaks at
    3× for wiki-elec (not 1.5×, +0.38pp over 1.5×) and at floor for wiki-rfa (not 1.5×,
    +0.35pp over 1.5×, though the whole grid only spans ~1.4pp there and isn't cleanly
    monotonic). Per the user: the wiki 1.5× pick was already known to be somewhat
    arbitrary (mixed behavior between the two wiki-genre datasets), so this isn't a
    fully new problem, just confirmation. **Resolution, per explicit user direction:
    don't re-pick or retrain (no time, and parameter tuning isn't something this
    project puts weight on) — instead stop claiming κ was validation-tuned anywhere in
    the paper.** `WSDM_format_revised.tex`'s walk-budget table caption and the Setup
    paragraph were reworded accordingly (κ now presented as a fixed per-dataset config
    value, same treatment as the untuned-in-text architecture hyperparameters, not as
    something optimized). The actual `configs/<dataset>.yaml` κ values are unchanged.
    Section 6.6 (Complexity)'s own `\ph{TODO}` asking "why does κ vary by dataset" is
    now effectively answered ("it isn't cleanly principled, don't over-read it") but
    that TODO hasn't been edited yet — Section 6.6 is gated on presenting the user
    alternative framings first, per its own note in the closeout plan.
  - **2026-08-18: new deliverable, `aaai2027/STATISTICS_ELI5_GUIDE.md`** — plain-language
    reference explaining every statistical concept/test used in this project (paired
    tests, cluster-robust SE, FDR correction, meta-analysis, Shapley values, Spearman
    vs. Pearson, etc.), for whenever a term in `STATISTICAL_TESTS_AUDIT.md` or a script
    docstring needs unpacking. Not part of the paper.
  - **2026-08-18: K-ablation (6.5(C) now, was a `\ph{TODO}`) — done.** A single walk
    (K=1, no aggregation at all) already beats the best baseline on all 6 datasets; the
    gap to the saturated ceiling (K≈8–32) is under 1pp everywhere — confirms the
    per-walk representation carries the result, ensembling is a small secondary gain.
    Script: `scripts/paper_figures/extract_ablation_kwalks.py` (log-spaced K grid,
    fixed-seed random draw per edge, not first-in-file-order — reuses existing
    `test_predictions.pkl` per-walk-occurrence data, no retraining), output
    `aaai2027/figure_data/ablation_kwalks.csv`. **Side finding, not chased further**:
    wiki-rfa seed=50's `test_predictions.pkl` has one edge (id 44403) with 101,594 of
    350,260 total rows (29% of the file) — a real data anomaly, doesn't affect the
    K-ablation's AUC (equal per-edge weighting) but worth a look if wiki-rfa seed 50
    is ever used for anything more sensitive to per-edge row counts.
  - **2026-08-18: cluster-robust SE added to the attention forward/backward split
    (Panel B) AND the vertex/edge split (Panel C) of the Attention Directionality
    figure — done, significant on all 6 datasets, both splits.** The saved
    `attention_directionality.py` pickles only ever kept pre-averaged means, no
    per-instance data, so no SE could be computed from what was already on disk — new
    script `scripts/attention_directionality_panelB_se.py` reruns inference once
    (same `LOCAL_RUN_INFO` checkpoints, same 20k-sample cap) and caches per-instance
    (edge_id, forward_frac, backward_frac, node_frac, edge_frac) to
    `outputs/attention_directionality/<ds>_local_panelBC_perinstance.pkl` — delete a
    file to force a recompute, same convention as the other cached figure
    intermediates; a bare rerun of the script reuses the cache and costs nothing.
    Cluster-robust CI (95%, cluster=target edge) confirmed by an independent Wilcoxon
    signed-rank test on the same per-edge cluster means — both agree on every dataset
    by a wide margin. Both plot scripts (`plot_attndir_panelB_direction.py`,
    `plot_attndir_panelC_nodeedge.py`) now render error bars from
    `aaai2027/figure_data/attndir_panelBC_se.csv`. **Caught and fixed a real bug
    mid-implementation**: the first version of the node/edge split leaked
    self-attention (d=0) entirely into the "edge" category (since the target token
    itself is always edge-typed), inflating edge mass from the true ~0.49 to a wrong
    0.638 for bitcoin-alpha — fixed by excluding d=0 from both node/edge masks,
    matching how `attention_directionality.py`'s own node_total/edge_total already
    exclude self via only summing the fwd/bwd-masked contributions.
  - **2026-08-18: dataset-name canonicalization extended to every figure-generation
    script, not just the tex text.** The earlier canonicalization pass (see
    "MAJOR RESYNC" above) only fixed the `.tex` file's own prose/table text — PNG
    figures are separately rendered and untouched by that sweep. Audited every
    `plot_*.py` script feeding the 4 figures actually `\includegraphics`'d in
    `WSDM_format_revised.tex` (`pipeline_schematic.png` — hand-made `.drawio`, no
    dataset labels, out of scope; `empconf_panels_abcde_combined.png`; the
    `pewter_sigat_delta_heatmap.png`; `attndir_panels_abcd_combined.png`) and fixed
    `DISPLAY_LABEL` dicts (or added one where missing) in every script with a real
    lowercase-dataset-name label: `plot_attndir_panelA_headgrid.py`,
    `plot_attndir_panelB_direction.py`, `plot_attndir_panelC_nodeedge.py`,
    `plot_shap_edge_directionality.py`, `plot_empconf_panelB_mi_decay_linegraph.py`,
    `plot_empconf_panelC_gnn_entropy_heatmap.py`, `plot_multiseed_entropy_heatmaps.py`.
    `plot_empconf_panelD_signagreement_auc.py`/`plot_empconf_panelE_coefficients.py`
    confirmed clean (pooled across datasets, no per-dataset labels currently — will
    need the same treatment if/when Panel E's planned per-dataset stacked-bar rebuild
    lands). All 4 figures regenerated; verified visually consistent.
- **2026-08-18: Panel E (`fig:empconf-panels`) rebuilt as a per-dataset, 10-seed stacked
  bar chart, and figure panel labels canonicalized to uppercase (A)/(B)/(C)/... across the
  whole paper.** Panel E previously showed one pooled grouped-bar (SiGAT's 4 entropy-term
  coefficients averaged across all 6 datasets), which hid the per-dataset pattern that's
  the actual point of the panel (`tgt_in` dominates on the 4 smaller/sparser datasets,
  `src_out` dominates on epinions/slashdot090221 — see
  `outputs/lead4c_sigat_multiseed_node4_export/README.md` §4.3). Rebuilt as a diverging
  stacked bar (one stack per dataset, one segment per term), sourced directly from that
  already-computed 10-seed per-dataset export (`results/aggregated_summary.csv`) — no new
  model fitting. New significance convention for this panel: a term is "robust" for a
  given dataset if BH-FDR significant in ≥8/10 seeds (matches the export package's own
  forest-plot convention), non-robust segments drawn hatched; error bars are ±1 SD across
  seeds per term. Scripts: `scripts/paper_figures/extract_empconf_panelE_coefficients.py`,
  `plot_empconf_panelE_coefficients.py`. Audit updated: `STATISTICAL_TESTS_AUDIT.md` item
  #7 now done. **Separately, per the professor's C-ter instruction ("please have A,B,C,D,...
  in capital") plus an explicit user follow-up to apply it paper-wide**: every panel-letter
  reference — both figure captions (`fig:empconf-panels` (A)-(E), `fig:attndir` (A)-(D)) and
  every in-prose cross-reference (`Figure~\ref{fig:empconf-panels}A`, `Panel (D)`, etc.) —
  was swapped from lowercase to uppercase, and both combine scripts
  (`combine_empconf_panels_abcde.py`, `combine_attndir_panels.py`) now overlay uppercase
  labels on the rendered PNGs to match. Verified via grep: zero lowercase `(a)`-`(e)` panel
  refs remain (the one surviving `(e)` match, line 109, is the math variable for "edge $e$",
  unrelated). **Note, not yet resolved**: Section 6.1's prose paragraph headers
  (`\paragraph{(A) Information decays...}`, etc.) already used uppercase letters before this
  change, for an unrelated numbering scheme (4 prose subsections, not the 5 image panels) —
  the two schemes can now both show "(A)"/"(B)" for different things in the same section;
  flagged to the user, not unified (paragraph (D) is still an unfinished stub pending the
  edge-vs-vertex investigation, so touching that scheme now would be premature). **New
  shared module**: `scripts/paper_figures/dataset_style.py` — one canonical
  `DATASET_ORDER`/`DATASET_DISPLAY`/`DATASET_COLORS`/`DATASET_MARKERS` mapping, reusing the
  colors `plot_empconf_panelB_mi_decay_linegraph.py` already picked (values unchanged, now
  imported rather than redefined) so any script needing per-dataset colors has one place to
  get them from. **Two corrections, same day, per direct user feedback on the first Panel E
  render:** (1) Panel E's x-axis tick labels were initially colored/bolded per dataset via
  this module; reverted to default (no color, no bold) -- coloring the axis text read as
  messy for a panel whose bar segments are already colored by *term*, not dataset, so Panel E
  now imports only `DATASET_ORDER`/`DATASET_DISPLAY` (name/order, no color) from the shared
  module. (2) the 4 entropy terms' stacking/legend order, initially sorted by coefficient
  magnitude (`tgt_in, src_out, src_in, tgt_out`), was reverted to the standard position-based
  order matching Panel A's schematic (`src_out, src_in, tgt_out, tgt_in` = H_out(-1),
  H_in(-1), H_out(1), H_in(1) -- source position first, target position second, out before in
  within each) -- the magnitude order was confusing to read against Panel A. `DATASET_COLORS`
  is still the intended source for Panel D's planned per-dataset dot colors (see the K-ablation/
  Panel D item below) -- the tick-label revert is Panel-E-specific, not a retraction of the
  shared-module idea. **Third correction, same day, layout-level (not just styling)**: the
  very first stacked-bar rebuild put datasets on the x-axis (6 bars, one per dataset, each
  stacked by term) -- the user caught this as wrong ("i thought that stacked barplot still
  need to show same bars like before but devided to the datasets. aint it? not 6 bars"). Fixed
  to keep the SAME 4-term x-axis as the pre-rebuild panel, with each term's single bar now
  stacked into 6 dataset segments instead of one pooled bar -- this is also what finally gives
  `DATASET_COLORS` a real purpose (segment fill color = dataset), which is what "keep dataset
  colors consistent" was about from the start; the axis-tick-coloring revert above was a
  symptom of the same underlying layout mistake, not an unrelated styling call.
- **2026-08-18, later same day: Panel E's real "gap" bug found and fixed (an earlier
  border-seam theory was wrong).** The user reported the first gap fix didn't work. Printing
  the actual computed bar boundaries found the true cause: `plot_empconf_panelE_coefficients.py`
  computed `bottoms = np.where(betas >= 0, cum_pos, cum_neg + betas)` for the stacked segments
  -- but matplotlib's `bar(bottom=B, height=H)` already spans `[B, B+H]`, so adding `betas` to
  `bottom` on top of using it as `height` double-counted it, shifting every negative segment
  down by its own value. Verified numerically before/after: Bitcoin-alpha's `H_out(-1)`
  segment rendered as `[-1.96, -0.98]` instead of the correct `[-0.98, 0.00]`; downstream
  segments then landed at essentially arbitrary overlaps (invisible, since the later segment
  just paints over the earlier one) or gaps (visible) depending on each pair's specific
  magnitude -- explaining why only one gap was visible in the first render even though the bug
  affected every segment. Fixed by removing the erroneous `+ betas`; re-verified by cropping
  and 4x-zooming the rendered PNG at the actual segment boundaries, not just eyeballing the
  full figure. **Also added, same pass**: an explicit legend entry (hatch-pattern proxy patch)
  stating what the hatched/white-fill segments mean (not robust: BH-FDR significant in fewer
  than 8/10 seeds) -- previously only in the docstring/caption, not the figure itself.
- **2026-08-18, later same day: Panel D (Fig. 2) rebuilt as multiseed; per-dataset
  visualization candidates generated, not yet chosen.** Pooled bars (the panel already in the
  paper) now show real mean ± 1 SD over 10 splits instead of a single-split point estimate,
  reusing `sigat_raw_seed()` -- numbers barely moved from the old single-split values (e.g.
  in-same 0.940→0.939), a good sanity check. **Added a raw-prediction cache**
  (`outputs/cache/sigat_raw_predictions/<ds>__seed<N>.pkl`, 60 files) since SiGAT only ever
  saves node embeddings, never a final per-edge prediction array -- getting an actual
  probability requires a fresh `LogisticRegression` refit per (dataset, seed), which is genuinely
  slow (~a few minutes for all 60) and was previously being redone from scratch by every
  separate analysis that needed it (Table 1's SiGAT row, the entropy heatmaps, and now this
  panel). The cache decouples that one-time cost from any downstream analysis; it's now
  populated, so future work needing raw per-seed SiGAT predictions is instant. **Per-dataset
  breakdown**: data computed (`aaai2027/figure_data/empconf_panelD_signagreement_auc_perdataset.csv`)
  and 4 candidate visualizations rendered to `aaai2027/figures/panelD_candidates/`
  (`panelD_candidate_{dots,groupedbar,boxstrip,heatmap}.png`, script
  `scripts/paper_figures/panelD_perdataset_candidates.py`) -- same "generate options, let the
  user pick" pattern as the existing `aaai2027/figures/panelA_candidates/`.
- **2026-08-18, later same day: Table 1's "Pewter beats every baseline" claim now has a real
  paired test (`STATISTICAL_TESTS_AUDIT.md` item #1, done).** Best non-Pewter baseline per
  dataset is always SiGAT or SNEA (never GS-GNN/SGCN, so their uncertain per-seed-data status
  never needed resolving); per dataset, picked whichever Pewter variant (full/local) has the
  higher 10-seed mean (matched Table 1's existing bold marks exactly, a good consistency
  check) and ran a one-sided paired Wilcoxon signed-rank test across the 10 shared seeds.
  Result: Pewter's winning variant beats the best baseline on all 10/10 splits, on all 6
  datasets (60/60 total), $p=0.00098$ throughout (the minimum achievable one-sided Wilcoxon
  $p$ at $n{=}10$ -- a perfect sweep). Script:
  `scripts/paper_figures/table1_paired_significance.py`, data:
  `aaai2027/figure_data/table1_paired_significance.csv`. Not yet inserted into the tex --
  the natural home is Section 6.2's still-unwritten prose rewrite (professor's C-ter ask),
  not a standalone edit.
- **2026-08-18, later same day: Panel C's per-dataset two-way ANOVA done (professor's
  C-ter ask, `STATISTICAL_TESTS_AUDIT.md` item #4).** Factors = source-entropy bin ×
  target-entropy bin (Panel C's existing 4x4 grid), response = AUC, one observation per
  (seed, cell) using the 10 splits as repeated measurements, exactly as the professor
  specified. Ran off already-cached data (Panel D's raw SiGAT prediction cache +
  `collect_model_records`'s entropy join, no refit) -- essentially free given the Panel D
  cache already existed. Both main effects significant on all 6 datasets; the interaction
  term significant on 5/6 (all but bitcoin-alpha, whose ~2,300-edge test set leaves only
  6/16 entropy-bin cells with data in all 10 seeds -- a rank-deficiency warning flags this,
  so its non-significant interaction should be read as underpowered, not a clean null, if
  ever cited on its own). Script: `scripts/paper_figures/panelC_twoway_anova.py`, data:
  `aaai2027/figure_data/panelC_twoway_anova.csv`.
- **2026-08-18, later same day: Figure 3 delta-heatmap "gain concentrates where the bound
  bites" claim tested (`STATISTICAL_TESTS_AUDIT.md` item #15) -- result is MIXED, does NOT
  cleanly confirm the current caption.** Per-dataset cluster-robust regression (delta ~
  source-entropy midpoint + target-entropy midpoint, cluster=seed, one observation per
  (seed, cell)): only bitcoin-otc shows the claimed pattern on both axes and epinions on
  the source axis only; bitcoin-alpha/wiki-elec/wiki-rfa are flat/null on both axes; and
  slashdot090221's target-entropy coefficient is significant in the OPPOSITE direction
  (higher target entropy -> smaller Pewter advantage there). Script:
  `scripts/paper_figures/delta_heatmap_entropy_regression.py` (note: first run silently
  produced zero usable observations on every dataset due to a missing `os.path.dirname()`
  level in its `ROOT` computation pointing it at `scripts/` instead of the repo root --
  fixed before trusting the numbers above), data:
  `aaai2027/figure_data/delta_heatmap_entropy_regression.csv`. Resolved into the paper as a
  compact in-text statement — see "Panel D wired in" entry below.
- **2026-08-18, later same day: Ablation B (`tab:ablationB`) rebuilt multiseed --
  `STATISTICAL_TESTS_AUDIT.md` item #9, done.** Confirmed all 11 aggregator functions'
  posthoc summaries already existed per seed (all 6 datasets x 10 seeds, per
  `run_multiseed_pewter.py`'s own docstring claim, verified directly on disk) -- pure
  aggregation, no new runs. Table now shows mean±std over the 10 splits per cell instead
  of a single-split point; the real cross-function spread tightened to $0.0018$ AUC (from
  the old single-split $<0.0022$ claim), about an order of magnitude below the ~0.005-0.018
  split-to-split std shown alongside it in the same table -- a stronger way to state
  "barely matters" than the bare spread number, since the reader can see the between-
  function gaps are smaller than the noise directly. Also updated 6.5(B)'s prose sentence
  to match. Script: `scripts/paper_figures/extract_ablationB_multiseed.py`, data:
  `aaai2027/figure_data/ablationB_multiseed.csv`. A formal per-pair paired-bootstrap test
  (porting the existing single-split `ablationB_paired_significance.py`) was scoped but not
  run -- treated as a nice-to-have per the project's cheap-unless-real-gain lean, not a gap.
- **LocalAttn4 H/no-H re-ablation — CLOSED 2026-07-19.** `HARDNESS_MINER_ROADMAP.md` items
  13/14 (`E27`/`E28`/`E29`, current sampler+masking, all 6 datasets + the asymmetric
  source/target weight probe on bitcoin-alpha/otc) all complete. Final verdict: **H
  scrapped everywhere** (see CLAUDE.md's "Hardness reweighting (H): scrapped everywhere") —
  do not cite the old "LocalAttn4+H (E14)" SOTA-table column or the E16/E17 "H is
  load-bearing for LocalAttn4" claim as current guidance, both retracted.
- **Short-walk ablation (E30) — CLOSED 2026-07-20.** See CLAUDE.md's "Attention variant" and
  `SOTA_HISTORY.md`'s E30 pilot detail.

## PEWTER paper file map — figure rework history

**Figure 1 rework (2026-07-28) — DONE except Panel B's final presentation call.** The
Empirical Confirmation 3-panel figure (checklist #12, now `fig:empconf-panels`,
`figures/empconf_panels_abc_combined.png`): Panel C converted from the smoothed
Gaussian-kernel grid to discrete 4×4 entropy bins with per-cell AUC annotations and a
larger panel (`scripts/paper_figures/{extract,plot}_empconf_panelC_gnn_entropy_heatmap.py`).
Panel A (entropy asymmetry, previously text-only/deferred) is now a real boxplot
($\Hh_\outdeg$ vs.\ $\Hh_\indeg$ per dataset, mean-diamond markers since 4/6 datasets have
median+IQR collapsed to 0, paired $t$-test in the caption) —
`scripts/paper_figures/{extract,plot}_empconf_panelA_entropy_boxplot.py`, reusing the same
per-node entropy arrays as `scripts/lead4c_directionality_answers.py::claim1_for_dataset`.
Combine script: `scripts/paper_figures/combine_empconf_panels_abc.py` (supersedes the old
2-panel `combine_empconf_panels_bc.py`).

**Panel B's post-minimum "bump" (distance 4–6) — investigated in full 2026-07-28, verdict:
real (not a bug, not pure noise), mechanism still being pinned down.** Checked and ruled out: BFS-correctness
(shell values matched `networkx` ground truth exactly, 0/30 mismatches; per-anchor edge
counts matched an independent brute-force enumeration exactly, 0/8 mismatches) and
numerical instability (integer counts throughout, no overflow, guarded `log2`). Confirmed
via a `--shuffle-signs` null control (already implemented on the extractor) that the bump
survives null-subtraction by 2–3 orders of magnitude on most datasets at distance 5–6 (not
pure estimator noise), but traced its cause to two compounding, measured mechanisms: (1) a
degree confound — nodes reached only at the outer BFS shells have collapsed degree (median
degree 44.5→2 across shells on bitcoin-alpha; exactly 1 by shell 3–4 on wiki-elec), hence
mechanically low entropy; (2) heavy edge reuse at the tail — the top 10 distinct context
edges account for ~40% of all (anchor, context) pairs at the farthest distance on
bitcoin-alpha vs. 0.2% at distance 1, so the naive per-pair-independent contingency table
is overconfident there. Extending `d_max` to 9 (real + null, all 6 datasets) showed the
effect does **not** keep growing — bitcoin-alpha/otc's graphs are essentially exhausted by
distance 7 (n_pairs collapses to ~5–7K and real/null become indistinguishable or reverse),
and wiki-elec/wiki-rfa hit **zero** remaining pairs by distance 6–7 (their graphs are simply
that small) — ruling out "it's a truncation artifact that would keep climbing if we looked
further."

**Degree-filtering was then tried as the candidate fix and FAILED** — a pilot on
bitcoin-alpha (5,000-anchor sample, requiring both context-edge endpoints to have degree
>=5) left distance-6 NMI essentially unchanged (0.000534 filtered vs. 0.000472 unfiltered,
same order of magnitude). So despite degree genuinely collapsing at the outer shells,
removing low-degree nodes does not kill the bump — degree is a correlate, not the
operative mechanism. **The more likely operative mechanism, per the edge-reuse
measurement, is non-independence/clustering of the (anchor, context) pairs**: a small
number of distinct context edges get counted many times over (top 10 distinct edges =
~40% of pairs at the tail vs. 0.2% at distance 1), violating the naive contingency table's
implicit "each pair is independent" assumption exactly where the bump appears — the
effective sample size at the tail is far smaller than the nominal n_pairs. **Recommended
next step (not yet implemented):** deduplicate by distinct context edge before forming the
contingency table (weight each distinct edge once, not once per anchor that reaches it), or
run a cluster-aware significance test (bootstrap over distinct edges, not raw pairs) instead
of trusting the naive point estimate at the tail bins. This supersedes the "degree confound"
framing as the leading candidate mechanism (degree collapse is real but not sufficient by
itself). The current `pewter_aaai.tex` Panel B paragraph's inline `%%` comment still
describes the superseded degree-confound framing and needs a follow-up edit once the
clustering fix is implemented or a final presentation decision is made.

**2026-08-18, later same day: Panel D wired in as per-dataset dots (professor's C-ter ask,
resolved), and Figure 3's delta-heatmap regression given a compact in-text statement (user's
final call on the item #15 audit finding).** Panel D: of the 4 candidate visualizations in
`aaai2027/figures/panelD_candidates/`, the user picked "dots" (one colored dot per dataset per
bucket, dodged, consistent `dataset_style.py` colors) — this was already implemented as
`plot_empconf_panelD_signagreement_auc.py::plot_perdataset()`, so no new plotting code was
needed. `combine_empconf_panels_abcde.py`'s `PANEL_D` constant now points at
`empconf_panelD_signagreement_auc_perdataset.png` instead of the old pooled-bar PNG; the figure
recombined. `fig:empconf-panels`'s caption (D) sentence updated from "pooled" to "one dot per
dataset (colors consistent with the rest of the paper)." No new statistical test was attached to
Panel D itself in this pass (distinct from the two-way ANOVA already on Panel B/C) — still open
if the professor wants one. Figure 3: rather than stating "gain concentrates where the bound
bites" as a uniform 6-dataset finding (the STATISTICAL_TESTS_AUDIT.md item #15 regression only
supports it cleanly on 1-2 of 6), the user chose the compact option — one sentence naming only
the significant cases: "A per-dataset cluster-robust regression (clustered by split) of this
delta on the source- and target-entropy bin midpoints finds a significant positive slope on both
axes for Bitcoin-otc and on the source-entropy axis for Epinions (both $p<0.01$); the same
regression is not significant on the other four datasets." Explicitly does not itemize the other
four datasets (including slashdot090221's target-axis reversal) in the main text — that detail
stays in the audit doc and `aaai2027/figure_data/delta_heatmap_entropy_regression.csv` for a
supplementary table or conversation with the professor if needed later. Both changes verified:
braces 626/626 balanced. **Also fixed, same pass, unrelated small items**: Assumption 1's
dangling-parenthesis sentence (line ~129) rewritten as one clean formal statement; the "Per-walk
classifier" paragraph's target-selection sentence now states the real number (verified against
`src/model/lit_model.py::_sample_epoch_targets` — default `target_ratio =
mask_ratio/(train_ratio+mask_ratio) = 0.4`, identical across all 6 datasets, resampled once per
**epoch** not per training step) instead of an `XXX HOW MUCH XXX` placeholder; the 48%/32%
context/mask-pool sentence now explicitly reconciles the fixed pool sizes with the dynamic
per-epoch resampling instead of carrying an unresolved contradiction flag. **Explicitly not
touched, per direct user instruction**: the pipeline schematic figure (`fig:pipeline-schematic`,
line ~185's font-size TODO) — the user is editing it in a separate session, leave it alone.

**2026-08-18, later same day: Panel C regenerated 2 rows x 3 columns with larger boxes,
Panel B's two-way ANOVA result written into prose, entropy worked example added, and Panel A
schematic rebuilt (colors + orientation), per the professor's remaining C-ter asks.**
`plot_empconf_panelC_gnn_entropy_heatmap.py` changed from a 1x6 dataset grid to 2x3 (per-cell
figsize 3.1x3.3in -> 4.2x4.4in), regenerated and recombined. Paragraph (B) in
`WSDM_format_revised.tex` now states the two-way ANOVA result (source-entropy bin x
target-entropy bin, 10 splits as repeated measurements; both main effects significant on all 6
datasets, interaction significant on 5/6, Bitcoin-alpha's sparser test set the one exception)
and a worked numeric example of the four entropy terms ($\Hh_\outdeg=\Hh_b(0.8)\approx0.72$ for
an 8-positive/2-negative vertex, etc.) plus the theoretical tie-back to
Proposition~\ref{prop:bottleneck} for why AUC should fall as entropy rises — both replacing
`XXX` markers, both confirmed with the user before insertion. Underlying ANOVA data:
`aaai2027/figure_data/panelC_twoway_anova.csv`, script:
`scripts/paper_figures/panelC_twoway_anova.py` (already run earlier the same day). **Panel A
schematic rebuilt** (`scripts/paper_figures/plot_empconf_panelA_schematic.py`): was a small,
hard-to-read horizontal diagram using an out-edge-blue/in-edge-orange color scheme unrelated to
the rest of the paper's palette — per direct user feedback, recolored to match
`aaai2027/figures/pipeline_schematic.drawio`'s own convention exactly (blue `#1F6FB2` =
positive edge, red `#B85450` = negative edge, dashed near-black `#333333` = unknown/masked
sign, mirroring the pipeline figure's "?"-token iconography), with direction (out vs. in) now
carried entirely by the arrowhead instead of by color — the two example context edges on each
side get one `+` and one `-`, assigned diagonally so the color never accidentally re-encodes
"out=blue." Layout switched from horizontal (u left, v right) to vertical (u top, v bottom),
which let the panel get much bigger/more readable in the combined figure without inflating the
figure's overall size much: `combine_empconf_panels_abcde.py`'s `ROW1_A_FRAC` (Panel A's width
share of row 1) was lowered from 0.34 to 0.30 to compensate for Panel A's new taller aspect
ratio, keeping row 1's total height close to what it was before the change. Figure 2's caption
(`fig:empconf-panels`, panel (A) sentence) updated with one added clause describing the
new color convention and cross-referencing `fig:pipeline-schematic`. All regenerated/recombined;
verified: braces 631/631 balanced, zero missing citation keys.

**Correction, same day, right after the above:** the first Panel A rebuild still had problems
per direct user feedback — the "0" label sat on top of the vertical edge line, the $\pm2$ index
labels were placed on the far terminal circles (wrong: the index belongs to the edge, not that
circle), the caption's new color-explaining clause was unwanted ("not the point of this panel,
just the indices" — removed), and the H-term labels needed to be bigger and closer to $u$/$v$.
Fixed: "0"/"?" now flank the line instead of sitting on it; $\pm2$ labels moved onto each edge
near its far end; the separate "+"/"-" sign glyphs were dropped (redundant once the caption
isn't explaining color); H-term font enlarged and repositioned to the edge midpoint, anchored
(`ha="left"`/`"right"`) so the two labels at each node grow away from the centerline instead of
colliding. Braces 630/630 after the caption trim.

**2026-08-18, later same day: Panel A reverted to the original matplotlib version, then
redesigned entirely through the .drawio path per further user iteration; Panel C/D/E layout
rebalanced; a caption-compaction pass; Panel D's paired significance test added.**

*Panel A.* The user reverted `scripts/paper_figures/plot_empconf_panelA_schematic.py` and
`combine_empconf_panels_abcde.py`'s `ROW1_A_FRAC` back to their pre-2026-08-18 originals
(horizontal, out=blue/in=orange, `ROW1_A_FRAC=0.34`) and moved to iterating on
`aaai2027/figures/empconf_panelA_schematic.drawio` by hand instead — this is now the live
source for Panel A's PNG (exported manually, not by any script here). Assisted edits along
the way: (1) synced the drawio to the reverted matplotlib original, then redesigned it
vertical with sign-based colors, dashed target edge, LaTeX math terms; (2) fixed 3 rounds of
real XML bugs (`--` inside `<!-- -->` comments is invalid XML — draw.io's own export doesn't
hit this, but hand-written comments did 3 separate times; always verify with
`xml.etree.ElementTree.fromstring()` before calling a drawio edit done); (3) restored the
$\pm2$ position indices after a round where they'd been dropped, this time correctly attached
as each edge's own `value` (renders along the edge line, like `0` already does) rather than
on the neighboring circle; (4) matched the H-term math font to the paper's own macros
(`\newcommand{\Hh}{\mathrm{H}}`, `\outdeg`/`\indeg`=`\mathrm{out}`/`\mathrm{in}` from
`WSDM_format_revised.tex`'s preamble) via draw.io's `math="1"`/`$$...$$` (MathJax) support:
`$$\mathbf{H}_{\mathrm{out}}(-1)$$` etc.; (5) fixed a real coordinate bug where H-terms were
positioned near the far bottom neighbor nodes instead of near $u$/$v$ (a placement bug, not
a sizing issue — enlarging the font would not have fixed it); (6) made all 4 side arms equal
length (they weren't — u's arms were ~285-313px, v's were ~400-420px, since u sat much closer
to the page-top margin than v sat to the page-bottom one) and shortened the target edge
relative to them, per direct user feedback ("top arrows are much shorter"). Final state is
whatever the user's own most recent manual drawio edit + PNG export produced — this log entry
covers the assisted portion only, not their subsequent hand-tuning.

*Panel C/D/E layout rebalance.* Making Panel C a 2x3 grid (earlier 2026-08-18 entry) made it
much taller, which combined with Panel A's own enlargement made Panel D/E (still in one
shared half-width row each) look tiny by comparison. Diagnosed: `imshow` fits an image to
whichever of its box's width/height is the binding constraint; at half-row-width, D/E's width
was always what capped their rendered size, so giving that row more height was only adding
blank padding, not making the panels bigger. Fixed in `combine_empconf_panels_abcde.py` by
giving D and E each their own full-width row (4 stacked rows total: A+B, C, D, E) instead of
splitting one row between them — the only real lever to grow a width-bound image. Panel C's
own per-cell figsize also trimmed slightly (4.2x4.4in -> 3.5x3.65in) so it doesn't dominate
as much. Combined figure is taller overall now — deliberate, per the user ("i dont mind if it
make the plot a bit big but we cannot have tiny panels").

*Caption compaction pass (Phase 7 of the closeout plan) — done, one caption at a time,
confirmed with the user before each edit.* Trimmed: Figure 1 (pipeline schematic) caption,
the per-dataset walk-budget table caption, Figure 2 (`fig:empconf-panels`) caption, Figure 4
(`fig:attndir`) caption — redundant phrasing cut ("as a function of"->"vs.", duplicated
clauses folded together), no factual content removed. Left as-is per explicit user choice:
Table 1's caption, Figure 3's (delta-heatmap) caption, Ablation B's caption — these were
already tight enough that the proposed trims weren't worth it to the user. One factual
question came up mid-pass (does Figure 3's caption correctly describe its $n$ annotation as
"mean per-split sample size"?) — verified against `plot_multiseed_entropy_heatmaps.py`'s
`_cell_n()`: yes, and further checked that PEWTER's and SiGAT's own per-cell $n$ (which
`_cell_n()` averages together for display) are literally identical in all 87 comparable cells
in `pewter_sigat_delta_heatmap.csv` (both models score the same canonical test-split edges
each seed, and entropy-bin membership is a property of the graph, not the model) — the
averaging is a harmless no-op, caption is accurate as written.

*Panel D paired significance test — new, done.* Per the professor's C-ter ask ("please have
statistical tests for D and e") — Panel E already had one (>=8/10-seed BH-FDR robustness),
Panel D didn't. New script `scripts/paper_figures/panelD_paired_significance.py`: one-sided
paired Wilcoxon signed-rank test across the 10 seeds (AUC$_{\text{same}}$ >
AUC$_{\text{diff}}$), per dataset, in/out directions separately, reusing the already-cached
per-seed raw SiGAT predictions (`outputs/cache/sigat_raw_predictions/`, no refit) and the
extract script's own bucketing logic. Result: same $>$ diff on all 6 datasets, both
directions (12/12), $p=0.001$ throughout (the minimum achievable one-sided Wilcoxon $p$ at
$n{=}10$) — full per-dataset table in `STATISTICAL_TESTS_AUDIT.md` item #6 and
`aaai2027/figure_data/panelD_paired_significance.csv`. Caption wording to state this is
drafted but not yet inserted into the tex — pending user confirmation, same "propose then
confirm" pattern used for every other caption edit this session.

*Marker resolution pass (post-professor resync) — line 75 citation, leakage-paragraph
reorder, Figure 2 page-overflow fix.* Line 75's "optionally with some edge topological
features XXXXX REFS XXXXX" gap: tried SEAL (Zhang & Chen 2018) first but rejected it after
checking the actual method — SEAL is subgraph-classification (enclosing-subgraph + node
labeling + pooled graph embedding), not "per-vertex embedding, classify from the endpoint
pair" as the sentence describes, so it would have misrepresented both the citation and the
claim. Used `battaglia2018relational` (Graph Networks framework paper) instead, per the
user's own suggestion — its edge-update function formally combines the two endpoint node
representations with the edge's own attributes, matching the sentence precisely. Added to
`pewter_references.bib`, cited at line 75, no XXX VERIFY marker (user supplied the entry
directly, stronger provenance than the earlier WebSearch-only stats-citation additions).

Line 191's "No label leakage" paragraph (flagged unclear by the professor even after an
earlier rewrite): root cause wasn't the prose itself but paragraph order — it used
"attend"/"attention" before the Transformer/attention mechanism was ever introduced (that
happens in the next paragraph, "Per-walk classifier"). Fixed by swapping the two paragraphs
(classifier first, defines attention; leakage second, can use the term freely) rather than
just patching the wording, and flipped the two paragraphs' cross-references to match
("unlike the split exclusion above" -> "described next" on the classifier side; the leakage
paragraph now says "described above"). Same forward-ref style already used by the Walk
sampler paragraph's "combined by the vote below."

Figure 2 (`fig:empconf-panels`) page-overflow bug: after the 2026-08-18 rework gave D and E
each their own full-width row (to fix the "tiny panels" complaint), the combined PNG's
aspect ratio grew to ~2.24 (h/w) — at ACM sigconf's single-column `\linewidth` (~3.33in),
that renders ~7.5in tall, which combined with the long 5-panel caption overflows a page
(a non-starred `figure` float can't split across columns/pages, it just runs off the
bottom). Root-caused via `\includegraphics[width=\linewidth]` math, not a real PDF compile
(no LaTeX toolchain in this environment) — confirmed `figure*` would make it worse, not
better, since the same aspect ratio scaled to full page width comes out ~16in tall, which
can't fit any single page regardless of placement. Fix, all three levers approved by the
user together: (1) shrunk Panel C's per-row height (3.65in->3.0in,
`plot_empconf_panelC_gnn_entropy_heatmap.py`), (2) tightened combine-script padding
(hspace 0.06->0.03, pad_inches 0.05->0.02), (3) reverted D and E to sharing one row
side-by-side (new `combine_empconf_panels_abcde_sidebyside.py`) with their internal
fonts/markers bumped ~1.5x (`plot_empconf_panelD_signagreement_auc.py`'s `plot_perdataset()`,
`plot_empconf_panelE_coefficients.py`) so they stay legible at half-width — verified
visually (Read tool image view), not just by the numbers. Result: aspect ratio 2.18 (stacked
variant, kept as `empconf_panels_abcde_combined.png` but not wired into the tex) vs. 1.31
(side-by-side, wired in) — side-by-side renders at ~4.35in vs. stacked's ~7.27in at
single-column width, so side-by-side is the one actually used
(`empconf_panels_abcde_combined_sidebyside.png`). Caption also wrapped in `\footnotesize` as
extra margin. No LaTeX compile available to confirm the fix end-to-end — flag to the user
before submission if a real compiled-PDF check becomes necessary.

**Pending, not yet done (flagged by user, explicitly deferred):** in this same Figure 2
combined PNG, the bold (A)/(B)/(C)/(D)/(E) panel-letter labels (drawn via `_add_panel()`'s
`ax.text(0.0, 1.0, label, ...)` in both `combine_empconf_panels_abcde.py` and
`combine_empconf_panels_abcde_sidebyside.py`) sit directly on top of each panel's own title
text in the top-left corner, rather than beside/above it — check and fix next session
(nudge the label position or the panel's own title margin so they don't visually overlap).

## 2026-08-19/20 — Prop 2 marker, Section 6.2 write-up, global `\method` rename, Ablation A
test, Section 6 coherence pass, edge-vs-vertex investigation (closed), Section 7/8 reverify

**Math XXX markers.** After the professor's math-verification pass confirmed the bottleneck
and capacity proof sketches correct, removed the two `XXX NEED TO CHECK CAREFULLY XXXX` /
`XXXX AGAIN CHEKC XXXXX` markers (lines ~148, ~176). Added a new marker at Proposition 2's
capacity-form bound instead, flagging (in the professor's own ALL-CAPS style, addressed
directly to him, no file pointers he doesn't have) that the bound is near-vacuous in
practice: $\Hh_b^{-1}$ saturates at $\tfrac12$, so the RHS only clears $0$ once
$\Hh(Y)-2\log_2N$ does, which real embedding widths make unlikely.

**Section 6.2 (`\method\ is more accurate than SOTA`) filled in.** This subsection was
essentially a stub before this session. Added: a 5-sentence PEWTER recap referencing
Figure 1's Stage 1/Stage 2 pipeline; a description of all 8 baselines (2 generic
message-passing GNNs, 4 signed-graph-specific, 1 correlation-modeling, node2vec as
non-GNN reference); Table 1's results with a real paired-Wilcoxon significance claim
(\method\ vs. every baseline, $p=0.00098$ throughout, existing script
`scripts/paper_figures/table1_paired_significance.py`); and Figure 3 (`fig:delta-heatmap`)'s own
description plus a new significance test specifically against SiGAT (new script
`scripts/paper_figures/extract_figure3_pewter_vs_sigat_significance.py`, 10/10 wins,
$p=0.00098$, all 6 datasets — output
`aaai2027/figure_data/figure3_pewter_vs_sigat_significance.csv`).

**Global rename**: literal "Pewter" → `\method\`/`\method` everywhere except the macro
definition and the Abstract (per explicit user instruction), respecting the existing
`\xspace`-based spacing convention (`\method\ ` before a word, bare `\method` before
punctuation/`'s`).

**Ablation A (`abl:proximal`) got a real test**, not just an eyeballed delta comparison.
Per the user's own framing ("use local, it's cheap and brings the same result" should be
the message regardless of significance): added a two-sided paired Wilcoxon signed-rank
test across the 10 splits (new script
`scripts/paper_figures/extract_ablationA_full_vs_local_significance.py`, output
`aaai2027/figure_data/ablationA_full_vs_local_significance.csv`) — significant on 3/6
datasets (Epinions $p=0.027$, Wiki-elec $p=0.049$, Slashdot $p=0.002$), not on the other
3, but every delta stays under 0.2pp regardless. `STATISTICAL_TESTS_AUDIT.md` item #8
updated to match (was the one remaining real gap flagged there — an eyeball claim with no
test attached).

**Section 6 full coherence pass** (not just fix-what's-flagged — read the whole section as
an argument). Found and fixed:
- A duplicate paragraph (D) in 6.1 that was a verbatim copy-paste of paragraph (C)'s
  entire sentence — removed.
- Multiple "short walk(s)" phrasing bugs that conflated walk *length* with attention
  *window* size — a real correctness bug, not just imprecise wording, since this project's
  own E30 pilot already established these are architecturally different (walks stay up to
  80 hops; only the attention window is short). Fixed in the Introduction, Abstract,
  paragraph (A) of 6.1, the Conclusion, and Section 6.2's own recap — all now say "local
  window"/"restrict attention" rather than "short walks."
- Figure 2's combined-panel D/E labels were still overlapping the panel titles (the
  pending item from the prior log entry above) — fixed by raising the panel-letter
  `label_y`/`label_va` for D and E specifically in
  `combine_empconf_panels_abcde_sidebyside.py`'s `_add_panel()` calls; re-rendered and
  visually confirmed (Read tool image view) the overlap is gone.

**Edge-vs-vertex / attention "role" investigation — closed, not going in the paper.**
Section 6.4 (`Relation between attention and entropy` → renamed `Where \method's attention
goes`) originally only covered forward/backward attention direction; the Abstract's own
ending promised a second finding (vertex-token vs. edge-token attention split) that was
computed (Figure 3 Panel C) but never discussed in prose. Investigated two possible
"alternative theories" for the split:
1. **New causal check** — extended the existing direction-only exact-Shapley script
   (`scripts/shap_edge_directionality.py`) to include vertex tokens as maskable players
   (new script `scripts/shap_edge_vertex_role.py`, 6 features vs. the original 4, 64
   forward evaluations/instance vs. 16). First launch (`--batch-size 16`) hit repeated
   CUDA OOM warnings from an under-corrected batch size relative to the larger feature set
   (16×64=1024 stacked variants/batch vs. the original script's 32×16=512); fixed by
   dropping to `--batch-size 8`, ran clean on all 6 datasets. Result: vertex tokens
   causally dominate edge tokens on every dataset (4.25×-3.25× mean|shap|, z=22-45) —
   real, massively significant, but **scrapped from the paper per explicit user
   instruction** ("it doesnt surprise... but it doesnt help with the papers claim").
2. **Correlation checks** against `DATASET_STATS.md`'s cheap graph stats, AUC-boost over
   best GNN, and both entropy-asymmetry framings — almost all null. One caught and fixed
   error along the way: the vertex/edge-vs-AUC-boost correlation already sitting in
   `STATISTICAL_TESTS_AUDIT.md`'s summary table ($\rho=-0.32$) was traced exhaustively and
   found to appear nowhere else in the repo, directly contradicting its own detail
   section's "not yet run" note — essentially a fabricated/unverified number. Recomputed
   independently twice (scipy + a from-scratch manual rank-correlation implementation):
   real value is $\rho=+0.20$, $p=0.70$. Spot-checked 2 other audit-doc entries against
   real scripts/CSVs to confirm this was an isolated slip, not systemic — both checked out
   exactly. Audit doc item #14 rewritten to document the error/resolution; the
   correlation itself **dropped from the paper entirely** (not just corrected) since it
   doesn't serve the section's descriptive purpose.

Final 6.4 text keeps both splits (forward/backward direction, vertex/edge role) as
descriptive findings only, explicitly declining to claim they're the same phenomenon as
the entropy asymmetry or as each other. Getting the correlation-sign wording right (does
positive $\rho$ mean "confirm" or "reverse"?) took many rounds of revision before landing
on the simplest fix: report the primary continuous test's $\rho$ exactly as it already
existed in the tex ($\Hh_\outdeg-\Hh_\indeg$ gap vs. forward-minus-backward mass,
$\rho=-0.71$, $p=0.11$, never touched), and for the secondary categorical/binary version
report only a $p$-value ($p=0.042$) plus a plain factual sentence, with no second signed
$\rho$ at all — sidesteps the sign-convention confusion rather than trying to word around
it.

Also fixed a smaller clarity gap in 6.1(C): "source-side" was used without restating,
at the point of use, that it maps to $\Hh_\outdeg(v)$ (vs. $\Hh_\indeg(v)$ for
target-side) — the mapping was only established earlier in Problem Setting. Added an
inline parenthetical ("its behavior as a source" / "its behavior as a target") right at
the first use in paragraph (C).

**Housekeeping, same session:** Leads 5/6 scrapped (Lead 5 superseded by the K-ablation
already in the paper; Lead 6 dropped with no replacement) and `plan-stats-rigor.md` closed
(superseded by the 10-seed campaign) — all reflected in `CLAUDE.md`. `optuna_run.py` marked
stale with an inline top-of-file marker (not fixed, not a priority). Wrote two self-contained
session-starter prompts for future work (`~/.claude/plans/plan-cleanup-and-local-attention-
prompts.md`): a production cleanup of this repo into a new double-blind-compliant public repo
for the paper's anonymous code link, and a genuine sparse/windowed local-attention
implementation (current `LocalAttentionEncoderLayer` is dense-masked, no real speedup — see
`MASKING.md`). `aaai2027/PEWTER_ASSETS_CHECKLIST.md` rewritten from scratch (the old version,
written against the pre-WSDM `pewter_aaai.tex`, had drifted far enough to be actively
misleading) — now tracks only current, live-verified status: 6 open markers (all grepped
fresh from the tex, not carried over from stale rows), one missing figure asset
(`pipeline_schematic.png`, referenced but not exported), and clean mechanical-check results
(654/654 braces, 38/38 citations resolve).

**Section 7/8 reverification (2026-08-20), two findings surfaced, not yet resolved:**
1. "Per-edge inference aggregates hundreds to thousands of walk evaluations" (Discussion,
   and echoed in the Ablation~\ref{abl:singlewalk} intro sentence) doesn't match the real
   per-dataset average $K_{uv}$ computed later in that same ablation: 5.8 (Wiki-elec) to
   217.6 (Bitcoin-alpha) — never reaches "thousands," and the low end isn't "hundreds"
   either. Likely survived from before the real K-ablation numbers were computed.
2. The Discussion's "two Wikipedia vote graphs behave differently from the four trust
   graphs on the directionality analysis" — this exact 4-vs-2 split matches 6.1(C)'s
   entropy-asymmetry finding (confirms on Bitcoin-alpha/otc/Epinions/Slashdot, reverses on
   Wiki-elec/Wiki-RfA) but does *not* match 6.4's attention-direction finding (forward:
   Bitcoin-alpha+Slashdot; backward: Bitcoin-otc+Epinions+Wiki-elec+Wiki-RfA — genre-mixed,
   not a clean wiki/trust split). Now that Section 6.4 exists with its own "direction"
   framing, "the directionality analysis" is genuinely ambiguous about which finding it
   means — this is likely exactly what the professor's own
   `XXX MIGHT NOT BE TRUE WITH NEW RESULTS ON DIRECTIONALITY` marker (same sentence) is
   asking about. Both flagged to the user for a wording decision, not silently patched.
