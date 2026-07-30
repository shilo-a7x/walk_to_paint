"""One-off explanatory diagram (not a paper figure) -- illustrates the
difference between "count every edge incident to a reached node" (our
method / v3's method) vs. "count only the single edge that first
discovered a node during BFS" (the colleague's load_slashdot.py method).

Toy graph: anchor edge (A,B). Shell 1 = {C,D,E}. C has two extra edges to
F,G unrelated to how it was discovered; D and E are also directly connected
to each other. Neither script actually reads this file at runtime -- it's
purely for the chat explanation.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

OUT_PNG = "/tmp/claude-30743/-home-eng-shilo-avital-yolo-lab-walk-to-paint/22a506ad-31f1-432e-aba0-042627f162c3/scratchpad/bug2_illustration.png"

pos = {
    "A": (0, 1), "B": (0, -1),
    "C": (2, 1.6), "D": (2, 0.3), "E": (2, -1.3),
    "F": (4, 2.2), "G": (4, 1.0),
}

all_edges = [("A", "B"), ("A", "C"), ("A", "D"), ("B", "E"),
             ("C", "F"), ("C", "G"), ("D", "E")]
anchor_edge = ("A", "B")
tree_edges = {("A", "C"), ("A", "D"), ("B", "E")}          # discovery edges
other_edges = {("C", "F"), ("C", "G"), ("D", "E")}          # non-discovery edges

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

for ax, title, counted in [
    (axes[0], "Our method / v3:\nall edges of every reached node", tree_edges | other_edges),
    (axes[1], "Colleague's method:\nonly the BFS discovery edge", tree_edges),
]:
    G = nx.Graph()
    G.add_nodes_from(pos.keys())
    G.add_edges_from(all_edges)

    node_colors = []
    for n in G.nodes():
        if n in ("A", "B"):
            node_colors.append("#2a78d6")   # shell 0 (anchor endpoints)
        elif n in ("C", "D", "E"):
            node_colors.append("#1baf7a")   # shell 1
        else:
            node_colors.append("#c3c2b7")   # further out, not this shell

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=650,
                            edgecolors="#33322e", linewidths=1.2)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=11, font_color="white",
                             font_weight="bold")

    # anchor edge: always dashed grey, excluded from context count in both
    nx.draw_networkx_edges(G, pos, edgelist=[anchor_edge], ax=ax,
                            edge_color="#8a8a80", style="dashed", width=1.6)

    counted_here = [e for e in all_edges if e != anchor_edge and e in counted or
                    (e[1], e[0]) in counted]
    not_counted_here = [e for e in all_edges if e != anchor_edge and e not in counted_here]

    nx.draw_networkx_edges(G, pos, edgelist=counted_here, ax=ax,
                            edge_color="#2a78d6", width=2.6)
    nx.draw_networkx_edges(G, pos, edgelist=not_counted_here, ax=ax,
                            edge_color="#d6d4c8", style="dotted", width=1.6)

    ax.set_title(title, fontsize=10.5)
    ax.axis("off")

legend_handles = [
    plt.Line2D([0], [0], color="#8a8a80", linestyle="dashed", linewidth=1.6, label="anchor edge (A,B) — excluded from both"),
    plt.Line2D([0], [0], color="#2a78d6", linewidth=2.6, label="counted as distance-1 context"),
    plt.Line2D([0], [0], color="#d6d4c8", linestyle="dotted", linewidth=1.6, label="NOT counted"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, fontsize=8.5,
           bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Bug #2: which edges count as \"distance-1 context\" of anchor edge (A,B)?", fontsize=11.5)
fig.tight_layout(rect=[0, 0.06, 1, 0.94])

os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
fig.savefig(OUT_PNG, dpi=170, bbox_inches="tight")
print(f"saved {OUT_PNG}")
