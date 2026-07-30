"""One-off export: dump all 6 canonical datasets as plain edgelist files
(node ids remapped to a contiguous [0, N-1] range, sign remapped from the
internal {-1, +1} encoding to {0, 1}) for handing to an external reviewer.

Loads directly via the real production path -- src/utils/config.py::load_config
+ src/data/datasets.py::get_loader -- the same two calls
src/data/prepare_data.py itself makes, not through either analysis-script
wrapper (scripts/balance_theory_paths.py, scripts/node_mi_structural_embedding.py).
Per standing instruction: new scripts should import the standard dataloaders
directly rather than through those wrappers going forward.

Node remapping is deterministic (sorted original node id -> 0..N-1) so it's
reproducible from a fixed seed-free rule, not random. A per-dataset mapping
CSV (new_id,original_id) is written alongside the edgelist (NOT included in
the reviewer-facing zip -- kept for internal provenance/back-translation
only) so the remapping can be inverted later if needed.

Round-trip verification (run automatically by this script, not a separate
step): after writing the remapped edgelist, it's re-read back, the mapping
is inverted, and the reconstructed (orig_u, orig_v, sign) triples are
compared IN ORDER against the original get_loader()(cfg) output for
that dataset -- an exact match (same length, same tuples, same order)
confirms the exported file is a lossless re-encoding, not just "looks
right."

Output:
- aaai2027/external_review/<dataset>.edgelist -- "u v label" per line,
  u,v in [0, N-1], label in {0,1} (1=positive edge, 0=negative edge).
- aaai2027/external_review/README.txt -- format description for the
  external reviewer (included in the zip).
- aaai2027/dataset_edgelists_for_review.zip -- the 6 edgelist files +
  README, zipped for sending out.
- aaai2027/external_review/node_id_mappings/<dataset>_mapping.csv --
  new_id,original_id per dataset (NOT zipped -- internal provenance only).
"""
import csv
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.utils.config import load_config
from src.data.datasets import get_loader

# real dataset.name values (== configs/<name>.yaml), not the analysis-script's
# shorthand keys -- "slashdot", not "slashdot090221", was only ever a wrapper
# convenience, not a production dataset name.
DATASETS = ["bitcoin-alpha", "bitcoin-otc", "epinions", "wiki-elec", "wiki-rfa", "slashdot090221"]

OUT_DIR = "aaai2027/external_review"
MAPPING_DIR = os.path.join(OUT_DIR, "node_id_mappings")
ZIP_PATH = "aaai2027/dataset_edgelists_for_review.zip"

README = """Edge-sign datasets -- plain edgelist export
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
"""


def export_dataset(ds_name):
    name = ds_name
    cfg = load_config(overrides=[f"dataset.name={ds_name}"])
    edges = get_loader(ds_name)(cfg)

    nodes = sorted({n for u, v, s in edges for n in (u, v)})
    orig2new = {orig: i for i, orig in enumerate(nodes)}

    edgelist_path = os.path.join(OUT_DIR, f"{name}.edgelist")
    with open(edgelist_path, "w") as f:
        for u, v, s in edges:
            label = 1 if s > 0 else 0
            f.write(f"{orig2new[u]} {orig2new[v]} {label}\n")

    mapping_path = os.path.join(MAPPING_DIR, f"{name}_mapping.csv")
    with open(mapping_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["new_id", "original_id"])
        for new, orig in enumerate(nodes):
            w.writerow([new, orig])

    # --- round-trip verification ---
    new2orig = {new: orig for new, orig in enumerate(nodes)}
    reconstructed = []
    with open(edgelist_path) as f:
        for line in f:
            u_new, v_new, label = line.split()
            u_new, v_new, label = int(u_new), int(v_new), int(label)
            sign = 1 if label == 1 else -1
            reconstructed.append((new2orig[u_new], new2orig[v_new], sign))

    ok = reconstructed == edges
    print(f"{name}: N={len(nodes):,} E={len(edges):,}  "
          f"round-trip {'OK (exact match)' if ok else 'MISMATCH!!!'}")
    if not ok:
        raise RuntimeError(f"{name}: round-trip verification failed -- exported edgelist "
                            f"does not losslessly reconstruct the original canonical edges")
    return edgelist_path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MAPPING_DIR, exist_ok=True)

    with open(os.path.join(OUT_DIR, "README.txt"), "w") as f:
        f.write(README)

    edgelist_paths = []
    for ds in DATASETS:
        edgelist_paths.append(export_dataset(ds))

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(os.path.join(OUT_DIR, "README.txt"), arcname="README.txt")
        for p in edgelist_paths:
            zf.write(p, arcname=os.path.basename(p))

    print(f"\nwrote {ZIP_PATH}")
    print(f"(node id mappings kept separately, not zipped: {MAPPING_DIR}/)")


if __name__ == "__main__":
    main()
