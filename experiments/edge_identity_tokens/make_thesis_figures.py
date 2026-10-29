"""Render the thesis figures into thesis/figures/ -- the WSDM figures, same scripts, same styling,
same 300 dpi, with the walk-model data swapped for H3 (EID). Each production plot script is
imported unchanged and only its module-level input/output paths are redirected.

  pewter_sigat_delta_heatmap.png   H3 - SiGAT entropy-binned AUC delta (from eid_entropy_heatmaps.csv,
                                   rewritten into the production CSV schema)
  attndir_panel*.png, shap_edge_directionality.png, attndir_panels_abcd_combined.png
                                   from eid_attention_shap.py's CSVs
  empconf_panels_abcde_combined_sidebyside.png, pipeline_schematic.png
                                   copied unchanged (baseline-only / model-independent)
Ends by checking every PNG in thesis/figures/ is 300 dpi.

Usage: .venv/bin/python experiments/edge_identity_tokens/make_thesis_figures.py
"""
import csv
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts", "paper_figures"))
os.chdir(ROOT)  # the production plot scripts use repo-relative paths

DATA = os.path.join("experiments", "edge_identity_tokens", "thesis_figure_data")
FIG = os.path.join("thesis", "figures")
UNCHANGED = ["pipeline_schematic.png"]
PANELS = os.path.join(FIG, "panels")  # intermediate single-panel PNGs


def delta_heatmap():
    import plot_multiseed_entropy_heatmaps as m
    src = os.path.join(DATA, "eid_entropy_heatmaps.csv")
    dst = os.path.join(DATA, "eid_sigat_delta_heatmap.csv")
    with open(src) as f, open(dst, "w", newline="") as g:
        w = csv.DictWriter(g, fieldnames=["dataset", "src_bin_lo", "src_bin_hi", "tgt_bin_lo", "tgt_bin_hi",
                                          "pewter_mean_auc", "sigat_mean_auc", "delta", "n_splits_valid_pewter",
                                          "n_splits_valid_sigat", "mean_cell_n_pewter", "mean_cell_n_sigat"])
        w.writeheader()
        for r in csv.DictReader(f):
            w.writerow({k: r[k] for k in ("dataset", "src_bin_lo", "src_bin_hi", "tgt_bin_lo", "tgt_bin_hi",
                                          "sigat_mean_auc")} |
                       {"pewter_mean_auc": r["h3_mean_auc"], "delta": r["delta_h3_minus_sigat"],
                        "n_splits_valid_pewter": r["h3_n_splits"], "n_splits_valid_sigat": r["sigat_n_splits"],
                        "mean_cell_n_pewter": r["h3_mean_cell_n"], "mean_cell_n_sigat": r["sigat_mean_cell_n"]})
    m.IN_DELTA_CSV = dst
    m.OUT_DELTA_PNG = os.path.join(FIG, "pewter_sigat_delta_heatmap.png")
    m.plot_delta()


def attention_figure():
    import combine_attndir_panels as comb
    import plot_attndir_panelA_headgrid as a
    import plot_attndir_panelB_direction as b
    import plot_attndir_panelC_nodeedge as c
    import plot_shap_edge_directionality as d
    a.PANEL_WIDTH_HEADS = 4  # Wiki-RfA has 2 heads; keep WSDM's 4-head panel width so fonts match
    a.IN_CSV, a.OUT_PNG = os.path.join(DATA, "attndir_panelA_headgrid.csv"), os.path.join(FIG, "attndir_panelA_headgrid.png")
    for mod, name in ((b, "attndir_panelB_direction.png"), (c, "attndir_panelC_nodeedge.png")):
        mod.YLIM_INCLUDE_ERR = True  # error bars are std across the 10 models, taller than WSDM's SEs
        mod.IN_CSV = os.path.join(DATA, "attndir_panelBC_summary.csv")
        mod.SE_CSV = os.path.join(DATA, "attndir_panelBC_se.csv")
        mod.OUT_PNG = os.path.join(FIG, name)
    d.IN_CSV, d.OUT_PNG = os.path.join(DATA, "shap_edge_directionality.csv"), os.path.join(FIG, "shap_edge_directionality.png")
    for mod in (a, b, c, d):
        mod.main()
    comb.PANEL_A, comb.PANEL_B, comb.PANEL_C, comb.PANEL_D = a.OUT_PNG, b.OUT_PNG, c.OUT_PNG, d.OUT_PNG
    comb.OUT_PNG = os.path.join(FIG, "attndir_panels_abcd_combined.png")
    comb.HSPACE, comb.LABEL_LIFT_BC = 0.12, 0.01  # lift (B)/(C) above their plots
    comb.main()


def empconf_figure(scale=0.75):
    """Empirical-evidence figure (baseline/data-only): WSDM panels, with A, D, E redrawn on a smaller
    canvas so their text is larger after combining (B, C reused as rendered for WSDM)."""
    import combine_empconf_panels_abcde_sidebyside as comb
    import plot_empconf_panelA_schematic as pa
    import plot_empconf_panelD_signagreement_auc as pd_
    import plot_empconf_panelE_coefficients as pe
    os.makedirs(PANELS, exist_ok=True)
    pd_.FIG_SCALE = pe.FIG_SCALE = scale
    for k in ("FONT_SIZE_INDEX", "FONT_SIZE_LABEL", "FONT_SIZE_TERM"):  # A is an equal-aspect drawing: scale its fonts instead
        setattr(pa, k, getattr(pa, k) * 1.5)
    pa.OUT_PNG = os.path.join(PANELS, "empconf_panelA_schematic.png")
    pd_.OUT_POOLED_PNG = os.path.join(PANELS, "empconf_panelD_signagreement_auc.png")
    pd_.OUT_PERDATASET_PNG = os.path.join(PANELS, "empconf_panelD_signagreement_auc_perdataset.png")
    pe.OUT_PNG = os.path.join(PANELS, "empconf_panelE_coefficients.png")
    for mod in (pa, pd_, pe):
        mod.main()
    comb.PANEL_A, comb.PANEL_D, comb.PANEL_E = pa.OUT_PNG, pd_.OUT_PERDATASET_PNG, pe.OUT_PNG
    comb.OUT_PNG = os.path.join(FIG, "empconf_panels_abcde_combined_sidebyside.png")
    comb.HSPACE = 0.08
    comb.main()


def check_dpi():
    from PIL import Image
    bad = []
    for name in sorted(os.listdir(FIG)):
        if name.endswith(".png") and os.path.isfile(os.path.join(FIG, name)):
            dpi = Image.open(os.path.join(FIG, name)).info.get("dpi", (0, 0))
            ok = all(abs(x - 300) < 0.5 for x in dpi)
            print(f"  {'ok ' if ok else 'BAD'} {dpi[0]:.0f} dpi  {name}")
            bad += [] if ok else [name]
    if bad:
        raise SystemExit(f"not 300 dpi: {bad}")


def main():
    os.makedirs(FIG, exist_ok=True)
    for name in UNCHANGED:
        shutil.copy2(os.path.join("aaai2027", "figures", name), os.path.join(FIG, name))
    delta_heatmap()
    empconf_figure()
    se = os.path.join(DATA, "attndir_panelBC_se.csv")
    if os.path.exists(se) and sum(1 for _ in open(se)) == 7:  # header + all 6 datasets
        attention_figure()
    else:
        print("attention/SHAP data incomplete -- attention figure skipped")
    check_dpi()


if __name__ == "__main__":
    main()
