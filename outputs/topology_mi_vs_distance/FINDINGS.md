# Does degree topology explain Panel B's sign-NMI-vs-hops curve (and its bumps)? — 2026-10-07

Script: `scripts/topology_sign_mi_vs_topodist.py` (Part A + B), CSVs `sign_nmi_vs_topodist.csv`,
`sign_nmi_hop_x_topodist.csv`. Topology vector per edge u->v: log1p of
[outdeg(u), indeg(u), outdeg(v), indeg(v)], signs never used. Null = signs permuted over edges.
Panel B pair definition and 20k-anchor sample reproduced exactly (bitcoin-alpha d=1 n=4,187,709,
NMI 0.0260, identical to `aaai2027/figure_data/empconf_panelB_mi_decay_linegraph.csv`).

**Part A — sign phi vs topology distance (40M random edge pairs, 12 quantile bins):**

| bin (similar -> different) | bitcoin-alpha | bitcoin-otc | epinions | slashdot | wiki-elec | wiki-rfa |
|---|---|---|---|---|---|---|
| 0 | +0.021 | +0.042 | +0.124 | +0.023 | +0.048 | +0.037 |
| 3 | +0.006 | +0.005 | +0.043 | +0.001 | -0.001 | +0.004 |
| 6 | +0.001 | +0.005 | -0.020 | -0.007 | +0.000 | -0.007 |
| 9 | -0.009 | -0.009 | -0.065 | -0.011 | -0.014 | -0.012 |
| 11 | -0.024 | -0.040 | -0.062 | -0.010 | -0.031 | -0.036 |

Null NMI <= 2.4e-6 everywhere. On all 6, similar degree profiles -> positively correlated signs,
very different profiles -> negatively correlated signs, regardless of graph distance.

**Part B — Panel B pairs split by hop AND topology bin (phi):**
- Hop 1 is positive in every topology bin on all 6 (wikis ~0.1-0.2, slashdot ~0.3, epinions up
  to 0.57) -> the adjacency effect is real, not degree topology. Epinions' hop 1 is the most
  topology-dependent (0.57 in the most-similar bin, -0.25 in the most-different).
- The tail bumps (negative phi at hops 4-6 on bitcoins, wikis, slashdot) are present *within* fixed
  topology bins, including the most-similar ones (wiki-rfa hop 5 bin 0: -0.31; wiki-elec hop 5
  bins 1-6: ~-0.16; slashdot hops 5-6: -0.10 to -0.15 in nearly every bin). Epinions, which has no
  real bump, shows ~0 at its tail within bins.

**Verdict:** degree-profile topology produces a genuine sign-similarity pattern but does **not**
explain the bumps — they survive stratification by topology distance. Supersedes the earlier
surrogate-sign run (`topology_mi_vs_distance.csv`), which only showed a similar curve shape.
Remaining candidates: pair reuse at the tail (few distinct far edges counted many times,
PAPER_CLOSEOUT_LOG.md 2026-07-28) and core/periphery position not captured by 4 degrees.
Next cheap test: count each distinct (far) context edge once.
