Edge-sign datasets -- plain edgelist export
=============================================

6 files, one per dataset. Each line: "u v label"
  - u, v   : node ids, remapped to a contiguous range [0, N-1] for this
             dataset (N = number of distinct nodes in that graph). The
             remapping is internal-bookkeeping only, not semantically
             meaningful -- it does not preserve the original platform's
             user ids.
  - label  : 1 = positive edge (trust/support/positive rating),
             0 = negative edge (distrust/oppose/negative rating).

Edges are directed: "u v label" means the edge points from u to v.
One line per directed edge exactly as it appears in the source dataset
(no added/removed/deduplicated edges, no self-loop or multi-edge
filtering beyond whatever the canonical loader itself already applies).

Files:
  bitcoin-alpha.edgelist
  bitcoin-otc.edgelist
  epinions.edgelist
  wiki-elec.edgelist
  wiki-rfa.edgelist
  slashdot090221.edgelist
