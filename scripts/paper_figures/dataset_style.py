"""Canonical per-dataset display name / color / marker mapping, shared across every
paper_figures script that breaks a figure out by dataset. One source of truth so a reader
learns "this color = this dataset" once and it holds across every figure that uses it,
per the user's 2026-08-18 instruction to keep dataset colors consistent "all along."

Values reused as-is from plot_empconf_panelB_mi_decay_linegraph.py, the first script in this
project to deliberately assign one color per dataset -- this module doesn't change those
colors, it just gives every other script a single place to import them from instead of
re-picking its own.
"""

DATASET_ORDER = ["bitcoin-alpha", "bitcoin-otc", "epinions", "slashdot090221", "wiki-elec", "wiki-rfa"]

DATASET_DISPLAY = {
    "bitcoin-alpha": "Bitcoin-alpha", "bitcoin-otc": "Bitcoin-otc", "epinions": "Epinions",
    "slashdot090221": "Slashdot", "wiki-elec": "Wiki-elec", "wiki-rfa": "Wiki-RfA",
}

DATASET_COLORS = {
    "bitcoin-alpha": "#2a78d6", "bitcoin-otc": "#eb6834", "epinions": "#1baf7a",
    "slashdot090221": "#eda100", "wiki-elec": "#e87ba4", "wiki-rfa": "#008300",
}

DATASET_MARKERS = {
    "bitcoin-alpha": "o", "bitcoin-otc": "s", "epinions": "^",
    "slashdot090221": "D", "wiki-elec": "v", "wiki-rfa": "P",
}
