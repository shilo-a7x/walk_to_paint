# Walk-to-Paint Dataset Statistics

Analysis of all 6 signed graph datasets used in the project.
Generated: 2026-06-28

---

## Summary Table

| Dataset | Nodes | Edges | Density | Positive % | Negative % | Connected? |
|---------|-------|-------|---------|-----------|-----------|-----------|
| Bitcoin-Alpha | 3,783 | 24,186 | 0.001690 | 93.6% | 6.4% | ✓ (1 giant + 4 pairs) |
| Bitcoin-OTC | 5,881 | 35,592 | 0.001029 | 90.0% | 10.0% | ✓ (1 giant + 3 pairs) |
| Epinions | 131,580 | 840,799 | 0.000049 | 85.3% | 14.7% | ⚠️ (1 giant + 5567 small) |
| Wiki-Elec | 7,115 | 107,022 | 0.002114 | 78.4% | 21.6% | ✓ (1 giant + 23 pairs) |
| Wiki-RfA | 11,256 | 184,459 | 0.001456 | 78.0% | 22.0% | ✓ (fully connected) |
| Slashdot090221 | 82,140 | 549,202 | 0.000081 | 77.4% | 22.6% | ✓ (fully connected) |

---

## Detailed Statistics by Dataset

### Bitcoin-Alpha

#### Basic Metrics
- **Nodes:** 3,783
- **Edges:** 24,186
- **Graph Density:** 0.001690

#### Sign Distribution
| Sign | Count | Percentage |
|------|-------|-----------|
| +1 (positive) | 22,650 | 93.6% |
| -1 (negative) | 1,536 | 6.4% |

#### Degree Statistics (Undirected)
| Metric | Value |
|--------|-------|
| Mean | 6.87 |
| Median | 2.00 |
| Std Dev | 17.92 |
| Min | 1 |
| Max | 490 |

#### Connected Components (Weakly Connected)
- **Number of components:** 5
- **Largest component size:** 3,775 nodes (99.8% of graph)
- **Other components:** 4 isolated pairs (negligible)
- **Assessment:** ✓ Essentially connected — isolated nodes form negligible noise

---

### Bitcoin-OTC

#### Basic Metrics
- **Nodes:** 5,881
- **Edges:** 35,592
- **Graph Density:** 0.001029

#### Sign Distribution
| Sign | Count | Percentage |
|------|-------|-----------|
| +1 (positive) | 32,029 | 90.0% |
| -1 (negative) | 3,563 | 10.0% |

#### Degree Statistics (Undirected)
| Metric | Value |
|--------|-------|
| Mean | 6.67 |
| Median | 2.00 |
| Std Dev | 20.32 |
| Min | 1 |
| Max | 763 |

#### Connected Components (Weakly Connected)
- **Number of components:** 4
- **Largest component size:** 5,875 nodes (99.9% of graph)
- **Other components:** 3 isolated pairs (negligible)
- **Assessment:** ✓ Essentially connected — isolated pairs negligible

---

### Epinions

#### Basic Metrics
- **Nodes:** 131,580
- **Edges:** 840,799
- **Graph Density:** 0.000049

#### Sign Distribution
| Sign | Count | Percentage |
|------|-------|-----------|
| +1 (positive) | 717,129 | 85.3% |
| -1 (negative) | 123,670 | 14.7% |

#### Degree Statistics (Undirected)
| Metric | Value |
|--------|-------|
| Mean | 9.38 |
| Median | 1.00 |
| Std Dev | 40.70 |
| Min | 1 |
| Max | 3,478 |

#### Connected Components (Weakly Connected)
- **Number of components:** 5,568 (⚠️ **HIGHLY FRAGMENTED**)
- **Largest component size:** 119,130 nodes (90.5% of graph)
- **Other components summary:**
  - Average size: 2.2 nodes
  - Max size: 20 nodes
  
**Detailed breakdown of component sizes:**
| Size | Count |
|------|-------|
| 2 | 4,694 |
| 3 | 634 |
| 4 | 142 |
| 5 | 52 |
| 6 | 23 |
| 7 | 9 |
| 8 | 7 |
| 9 | 1 |
| 10 | 2 |
| 11 | 1 |
| 15 | 1 |
| 20 | 1 |

- **Assessment:** ⚠️ **MAJOR FRAGMENTATION** — ~9.5% of nodes (12,450 nodes) exist in isolated clusters outside the giant component. Mostly pairs/small cliques.

---

### Wiki-Elec

#### Basic Metrics
- **Nodes:** 7,115
- **Edges:** 107,022
- **Graph Density:** 0.002114

#### Sign Distribution
| Sign | Count | Percentage |
|------|-------|-----------|
| +1 (positive) | 83,929 | 78.4% |
| -1 (negative) | 23,093 | 21.6% |

#### Degree Statistics (Undirected)
| Metric | Value |
|--------|-------|
| Mean | 25.21 |
| Median | 6.00 |
| Std Dev | 48.75 |
| Min | 1 |
| Max | 974 |

#### Connected Components (Weakly Connected)
- **Number of components:** 24
- **Largest component size:** 7,066 nodes (99.3% of graph)
- **Other components:** 20 pairs + 3 triples (negligible)
- **Assessment:** ✓ Essentially connected — isolated clusters negligible

---

### Wiki-RfA

#### Basic Metrics
- **Nodes:** 11,256
- **Edges:** 184,459
- **Graph Density:** 0.001456

#### Sign Distribution
| Sign | Count | Percentage |
|------|-------|-----------|
| +1 (positive) | 143,822 | 78.0% |
| -1 (negative) | 40,637 | 22.0% |

#### Degree Statistics (Undirected)
| Metric | Value |
|--------|-------|
| Mean | 26.82 |
| Median | 6.00 |
| Std Dev | 52.86 |
| Min | 1 |
| Max | 1,174 |

#### Connected Components (Weakly Connected)
- **Number of components:** 1
- **Largest component size:** 11,256 nodes (100.0% of graph)
- **Assessment:** ✓ **FULLY CONNECTED** — single giant component includes all nodes

---

### Slashdot090221

#### Basic Metrics
- **Nodes:** 82,140
- **Edges:** 549,202
- **Graph Density:** 0.000081

#### Sign Distribution
| Sign | Count | Percentage |
|------|-------|-----------|
| +1 (positive) | 425,072 | 77.4% |
| -1 (negative) | 124,130 | 22.6% |

#### Degree Statistics (Undirected)
| Metric | Value |
|--------|-------|
| Mean | 9.61 |
| Median | 2.00 |
| Std Dev | 32.85 |
| Min | 1 |
| Max | 2,543 |

#### Connected Components (Weakly Connected)
- **Number of components:** 1
- **Largest component size:** 82,140 nodes (100.0% of graph)
- **Assessment:** ✓ **FULLY CONNECTED** — single giant component includes all nodes

---

## Comparative Analysis

### Size Classes
| Class | Datasets | Nodes | Edges |
|-------|----------|-------|-------|
| Small | Bitcoin-Alpha, Bitcoin-OTC | 3.8k–5.9k | 24k–36k |
| Medium | Wiki-Elec, Wiki-RfA | 7.1k–11.3k | 107k–184k |
| Large | Epinions, Slashdot090221 | 82k–132k | 549k–841k |

### Sign Distribution Pattern
All datasets show positive bias, but with distinct ranges:

- **Highly positive-skewed:** Bitcoin-Alpha (93.6%), Bitcoin-OTC (90.0%)
- **Moderately positive-skewed:** Epinions (85.3%)
- **Balanced:** Wiki-Elec (78.4%), Wiki-RfA (78.0%), Slashdot090221 (77.4%)

**Implication:** Trust signals (+1) dominate, especially in cryptocurrency networks. Social voting datasets (wiki, slashdot) show more balanced negative feedback.

### Degree Distribution Pattern
| Pattern | Datasets | Characteristics |
|---------|----------|-----------------|
| Sparse + Skewed | Bitcoin-Alpha, Bitcoin-OTC, Slashdot | Median=1–2, Max>500 (power-law) |
| Sparse + Moderate | Epinions | Median=1, Max=3478 (extreme tail) |
| Dense + Moderate | Wiki-Elec, Wiki-RfA | Median=6, Max>900 (more uniform) |

**Implication:** Cryptocurrency/slashdot networks have most users with few connections (median ≤ 2) but extreme hubs. Wiki networks more uniformly connected locally.

### Connectivity Pattern
| Pattern | Datasets | Assessment |
|---------|----------|-----------|
| Essentially Connected | Bitcoin-Alpha, Bitcoin-OTC, Wiki-Elec | 1 giant (99%+) + negligible isolated pairs |
| Fully Connected | Wiki-RfA, Slashdot090221 | 1 component includes 100% of nodes |
| Highly Fragmented | Epinions | 1 giant (90.5%) + 5,567 small clusters (9.5% of nodes) |

**Implication:** Epinions is unique in its fragmentation, likely due to sparse trust edges in a large user base. Wiki and slashdot networks form cohesive communities.

---

## Key Observations

### 1. Sign Imbalance (Universal)
All datasets exhibit positive bias, from 77% (Slashdot, Wiki) to 94% (Bitcoin-Alpha).
- Potential impact on prediction: models must avoid majority-class shortcuts
- Mitigation in CLAUDE.md: `binary=True` + balanced sampling strategies

### 2. Sparsity Variation
- **Most sparse:** Epinions (density 0.000049), Slashdot090221 (0.000081)
- **Least sparse:** Wiki-Elec (0.002114)
- Dense graphs (wiki) support more local context; sparse graphs (epinions) risk isolated subgraphs

### 3. Scale Clustering
- **Small tier (3.8k–5.9k nodes):** Bitcoin networks
- **Medium tier (7.1k–11.3k nodes):** Wiki networks
- **Large tier (82k–132k nodes):** Social voting (Epinions, Slashdot)
- Walk sampling strategies may need tier-specific tuning

### 4. Epinions is an Outlier
- Most nodes (9.5%) in isolated components outside the giant component
- Lowest density among all datasets
- Highest degree skew (median=1, max=3,478)
- **Implication for walk-based models:** Walk coverage on Epinions *was* lower under the
  old uniform sampler (~88% test coverage vs. 100% on Bitcoin). **RESOLVED (2026-06-29):**
  the E15 `k_cover` k=5 edge-anchored sampler drives node+edge coverage to ~100% on all 6
  datasets — see `WALK_COVERAGE.md` and CLAUDE.md.

### 5. Degree Distribution Quality
- **Degree median comparison:** Bitcoin ≈ 2, Wiki ≈ 6, Epinions/Slashdot ≈ 1–2
  - Suggests wiki datasets have more consistent local neighborhoods
  - Bitcoin/slashdot/epinions rely more on hub nodes for global connectivity

---

## Notes for Practitioners

1. **Data Loading:** All statistics verified via `src/data/datasets.py` loaders (binary=True, remove_self_loops=True)
2. **Component Analysis:** Uses NetworkX weakly-connected-components (treats directed edges as undirected)
3. **Fragmentation Impact:** Epinions fragmentation may affect:
   - Walk sampler coverage (see WALK_COVERAGE.md)
   - Baseline GNN comparison fairness
   - Analysis script assumptions (e.g., degree correlations in isolated components)
4. **Sign Distribution:** Reflects dataset nature (cryptocurrency trust/distrust, wiki elections, slashdot karma)

---

## File References

- **Data loading:** `src/data/datasets.py`
- **Connectivity impact on walks:** `WALK_COVERAGE.md` (CLAUDE.md context)
- **Fragmentation impact on edge-level comparisons:** `FABRICATED_REVERSE_EDGES.md` (CLAUDE.md context)
- **Config overrides per dataset:** `configs/<dataset>.yaml`

