"""Extract step for Empirical Confirmation Panel B (MI decay with distance).

Pulls the already-computed MI-vs-BFS-distance numbers out of
outputs/mi_analysis_package.zip (edge_sign_mi/mi_vs_dist_report_v3.txt) and
writes a small flat CSV. No new computation -- this is a read-only transcription
of an existing result. Run this once (or whenever the source report changes);
the plotting script never touches the zip file, only this CSV.
"""
import csv
import re
import zipfile

ZIP_PATH = "outputs/mi_analysis_package.zip"
REPORT_MEMBER = "mi_analysis_package/edge_sign_mi/mi_vs_dist_report_v3.txt"
OUT_CSV = "aaai2027/figure_data/empconf_panelB_mi_decay.csv"

# report's own section header -> paper/CLAUDE.md dataset name
DATASET_LABELS = {
    "bitcoin-alpha": "bitcoin-alpha",
    "bitcoin-otc": "bitcoin-otc",
    "epinions": "epinions",
    "wiki-elec": "wiki-elec",
    "wiki-rfa": "wiki-rfa",
    "slashdot": "slashdot090221",
}


def parse_report(text):
    rows = []
    current = None
    for line in text.splitlines():
        header = line.strip()
        if header in DATASET_LABELS:
            current = DATASET_LABELS[header]
            continue
        m = re.match(r"\s*d=(\d+)\s+([\d,]+)\s+([\d.]+|nan)\s+([\d.]+|nan)", line)
        if m and current is not None:
            d = int(m.group(1))
            n_pairs = int(m.group(2).replace(",", ""))
            mi_str, nmi_str = m.group(3), m.group(4)
            rows.append({
                "dataset": current,
                "d": d,
                "n_pairs": n_pairs,
                "mi_bits": mi_str,
                "nmi": nmi_str,
            })
    return rows


def main():
    with zipfile.ZipFile(ZIP_PATH) as zf:
        text = zf.read(REPORT_MEMBER).decode("utf-8")
    rows = parse_report(text)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "d", "n_pairs", "mi_bits", "nmi"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
