Read ~/.claude/plans/hello-so-i-have-unified-valiant.md (the PEWTER paper roadmap) and
aaai2027/PEWTER_ASSETS_CHECKLIST.md in full before doing anything else — the checklist is the
first thing to read when resuming any paper subtask, it has row-per-figure/table status and
pointers, don't re-derive anything from scratch that it already answers.

CONTEXT ON WHY THIS SESSION IS FRESH: the previous session went deep on a side investigation into
Panel B (the MI/phi-vs-distance figure) that turned into two separate deep-dives — (1) verifying
the post-minimum "bump" at distance 4-6 is statistically real via a proper cluster-robust
bootstrap, and (2) reconciling why an external colleague's independent reimplementation of the same
idea reported "close to zero" everywhere (turned out to be 4 stacked methodological gaps: BFS
direction, edge-recording convention, anchor sampling, and an unseeded random subsampling cap in
her script). Both threads are now fully RESOLVED and written up — do NOT re-investigate, re-verify,
or re-litigate any of this. Read PANELB_INVESTIGATION_REPORT.md (repo root) once if you need the
detail, otherwise just know: the bump is confirmed real, the colleague discrepancy is fully
explained, and 2 follow-up actions are flagged but not done (see below) — that's the entire
carry-over from that thread. Everything else about paper work should proceed fresh, uncontaminated
by that investigation's specifics.

TWO SMALL FOLLOW-UPS FROM THE PANEL B THREAD (fast, do these first if untouched):
1. aaai2027/PEWTER_ASSETS_CHECKLIST.md item #12 (Panel B) still says the bump's fix is "NOT YET
   IMPLEMENTED" — that's now stale, update it to point at PANELB_INVESTIGATION_REPORT.md and say
   RESOLVED (real at every distance under a proper cluster bootstrap).
2. pewter_aaai.tex has an inline `%%` comment above the Panel B paragraph describing the
   now-superseded "degree confound" framing as the leading candidate mechanism — per CLAUDE.md's
   Figure 1 status note, this needs a follow-up edit reflecting the cluster-bootstrap
   confirmation (or at minimum, decide with the user whether to mention the clustering/cluster-
   bootstrap finding in the paper prose at all, vs. just confirm the bump number and move on).

WHAT'S ACTUALLY NEXT (per the checklist, Wave 1 priority — fill blanks before polishing prose):
- Checklist #23, Ablation A (full vs. local attention): status is "DONE as a rough sketch, ON HOLD
  for a from-scratch redo." This is very likely today's real next task if the user says
  "ablations for fig 4."
- Checklist #22, Result 3 (attention direction of information flow): "OPEN-QUESTION, LOWER
  PRIORITY, deprioritized 2026-07-20" — semantics of "attention = information flow" still
  unsettled. Confirm with the user before picking this up; it was deliberately parked, not
  forgotten.
- Checklist #26 (Conclusion headline number): blocked only on #22 now (retrain blocker already
  lifted 2026-07-30) — low-effort to close once #22's status is confirmed with the user.
- Checklist #31 (architecture schematic): NOT STARTED, needs a spun-out prompt for a
  diagram-generation agent, not direct work in this session.

STANDING RULES TO CARRY FORWARD (already in CLAUDE.md, restated because they matter most right
now):
- Freshness discipline: only CLAUDE.md's SOTA table is guaranteed fresh. Every other number/figure
  pulled from the repo must be confirmed fresh-or-stale WITH THE USER before it goes in the paper
  — don't just grep-verify silently and assume current.
- XXXX/\ph{PLACEHOLDER} markers: never remove without per-instance approval, even once replacement
  text is drafted.
- Old-vs-new draft text distinguishability convention (`\oldtext{}` macro, gray italics) — already
  set up in the .tex preamble, use it for anything touching old professor-draft prose.
- No new experiments invented on the spot for the paper itself — flag as TODO / spin out a request
  doc (same pattern as LEAD4C_DIRECTIONALITY_EXPERIMENTS_NEEDED.md -> ...ANSWERS.md), don't just
  run something new mid-session unless the user explicitly asks for it (as they did for the Panel
  B ablation work last session — that was an explicit, scoped request, not a standing license).
- This is a deadline-critical workstream (per the plan file: abstract 2026-07-28 has already
  passed as of "today" in this repo's timeline — check the plan file's timeline section and
  confirm current date/deadline status with the user at the very start of this session, since the
  full-paper deadline of 2026-07-31 may be imminent or passed by the time this session runs).
