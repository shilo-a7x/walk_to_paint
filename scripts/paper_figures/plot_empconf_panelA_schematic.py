"""Plot step for Empirical Confirmation Panel A -- schematic of the position-index
notation reused by Panel E's regression coefficients. No extract step / no data
dependency: this is a static diagram, not a data recomputation.

Convention: the target edge being predicted is index 0, connecting its source `u`
(index -1) to its target `v` (index +1). Each endpoint's OTHER incident edges are
drawn at index -2 (u's other edges) / +2 (v's other edges) -- one out-edge and one
in-edge per side, so the diagram covers all 4 of Panel E's terms:
H_out(-1)/H_in(-1) (u's out/in-edge sign entropy, i.e. src_out/src_in) and
H_out(1)/H_in(1) (v's out/in-edge sign entropy, i.e. tgt_out/tgt_in).

See PANELA_SCHEMATIC_GUIDE.md for what to edit for quick color/layout tweaks, and
for the parallel draw.io version (empconf_panelA_schematic.drawio) if you want
heavier manual edits instead.

2026-08-05 (user call): native figsize shrunk to match this panel's actual
single-column display width (~3.3in, was 7.2in) -- text/patch sizes are set in
absolute points/data units and are unchanged, so they render at their nominal
size instead of being shrunk ~2.2x by LaTeX at inclusion time.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

OUT_PNG = "aaai2027/figures/empconf_panelA_schematic.png"

# ---- tweakable style constants (see PANELA_SCHEMATIC_GUIDE.md) ----
NODE_COLOR = "#4a4a4a"
NODE_FACE = "#eeeeee"
TARGET_EDGE_COLOR = "#c0392b"
OUT_EDGE_COLOR = "#2e75b6"
IN_EDGE_COLOR = "#e08a1e"
NODE_RADIUS = 0.24
FONT_SIZE_INDEX = 8
FONT_SIZE_LABEL = 7.5
FONT_SIZE_TERM = 7


def draw_node(ax, xy, label, index_label):
    circ = Circle(xy, NODE_RADIUS, facecolor=NODE_FACE, edgecolor=NODE_COLOR,
                   linewidth=1.4, zorder=3)
    ax.add_patch(circ)
    ax.text(xy[0], xy[1], label, ha="center", va="center", fontsize=FONT_SIZE_LABEL,
             fontweight="bold", color=NODE_COLOR, zorder=4)
    ax.text(xy[0], xy[1] - 0.42, index_label, ha="center", va="center",
             fontsize=FONT_SIZE_INDEX, color="black", zorder=4)


def draw_edge(ax, p_from, p_to, color, lw, index_label=None, index_xy=None, style="-"):
    arrow = FancyArrowPatch(
        p_from, p_to, arrowstyle="-|>", mutation_scale=14, color=color,
        linewidth=lw, linestyle=style, shrinkA=NODE_RADIUS * 72, shrinkB=NODE_RADIUS * 72,
        zorder=2,
    )
    ax.add_patch(arrow)
    if index_label is not None:
        ax.text(index_xy[0], index_xy[1], index_label, ha="center", va="center",
                 fontsize=FONT_SIZE_INDEX, color="black", zorder=4)


def main():
    fig, ax = plt.subplots(figsize=(3.3, 2.0))

    u = (-1.0, 0.0)
    v = (1.0, 0.0)
    u_out = (-2.3, 0.85)   # u's other OUT-edge target
    u_in = (-2.3, -0.85)   # u's other IN-edge source
    v_out = (2.3, -0.85)   # v's other OUT-edge target
    v_in = (2.3, 0.85)     # v's other IN-edge source

    # target edge (index 0)
    draw_edge(ax, u, v, TARGET_EDGE_COLOR, 2.6, "0", (0.0, 0.30))

    # u's other edges (index -2 family): one out, one in
    draw_edge(ax, u, u_out, OUT_EDGE_COLOR, 1.6, "-2", (-1.85, 0.72))
    draw_edge(ax, u_in, u, IN_EDGE_COLOR, 1.6, "-2", (-1.85, -0.72))

    # v's other edges (index +2 family): one out, one in
    draw_edge(ax, v, v_out, OUT_EDGE_COLOR, 1.6, "+2", (1.85, -0.72))
    draw_edge(ax, v_in, v, IN_EDGE_COLOR, 1.6, "+2", (1.85, 0.72))

    draw_node(ax, u, "u", "-1")
    draw_node(ax, v, "v", "+1")
    for xy in (u_out, u_in, v_out, v_in):
        circ = Circle(xy, NODE_RADIUS * 0.75, facecolor="white", edgecolor=NODE_COLOR,
                       linewidth=1.0, zorder=3)
        ax.add_patch(circ)

    # term labels tying the diagram to Panel E's naming
    ax.text(-2.3, 1.35, r"$H_{\mathrm{out}}(-1)$", ha="center", fontsize=FONT_SIZE_TERM,
             color=OUT_EDGE_COLOR)
    ax.text(-2.3, -1.35, r"$H_{\mathrm{in}}(-1)$", ha="center", fontsize=FONT_SIZE_TERM,
             color=IN_EDGE_COLOR)
    ax.text(2.3, -1.35, r"$H_{\mathrm{out}}(1)$", ha="center", fontsize=FONT_SIZE_TERM,
             color=OUT_EDGE_COLOR)
    ax.text(2.3, 1.35, r"$H_{\mathrm{in}}(1)$", ha="center", fontsize=FONT_SIZE_TERM,
             color=IN_EDGE_COLOR)

    ax.text(0.0, -1.55, "target edge to predict", ha="center", fontsize=FONT_SIZE_TERM,
             color=TARGET_EDGE_COLOR, style="italic")

    ax.set_xlim(-3.1, 3.1)
    ax.set_ylim(-1.85, 1.85)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
