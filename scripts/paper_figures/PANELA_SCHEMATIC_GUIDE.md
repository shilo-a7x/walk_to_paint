# Panel A schematic — tweak guide

Two parallel, independent sources for this panel. Pick ONE editing path per session —
don't edit both and expect them to merge.

## Path 1: matplotlib script (default, reproducible)

`scripts/paper_figures/plot_empconf_panelA_schematic.py` → regenerate with:

```
.venv/bin/python scripts/paper_figures/plot_empconf_panelA_schematic.py
```

Quick tweaks, all at the top of the file as named constants — edit these, rerun, done:

| Constant | Controls |
|---|---|
| `TARGET_EDGE_COLOR` | color of the bold center edge (index 0) |
| `OUT_EDGE_COLOR` / `IN_EDGE_COLOR` | color of the two other-edge families |
| `NODE_COLOR` / `NODE_FACE` | node outline / fill |
| `NODE_RADIUS` | size of the `u`/`v` circles |
| `FONT_SIZE_INDEX` / `FONT_SIZE_LABEL` / `FONT_SIZE_TERM` | index numbers / `u`,`v` labels / `H_out(·)`,`H_in(·)` term labels |

Node/arrow positions are set directly in `main()` (`u`, `v`, `u_out`, `u_in`, `v_out`,
`v_in` — plain `(x, y)` tuples) if you want to reshape the layout, not just recolor it.

## Path 2: draw.io (heavier manual edits)

`aaai2027/figures/empconf_panelA_schematic.drawio` — open in
[app.diagrams.new](https://app.diagrams.new) or the desktop app, edit freely, then
**File → Export as → PNG**, and save directly over `aaai2027/figures/empconf_panelA_schematic.png`
(the exact path the combine script reads from — no other file needs to change).

**Important:** this file is a one-time hand-authored equivalent of the matplotlib
version, not auto-generated from it and not auto-synced back to it. If you edit the
`.drawio` file and export a new PNG, the matplotlib script's own output will be
overwritten the next time someone reruns it (e.g. after touching an unrelated panel and
re-running the whole pipeline) — if you've committed to the draw.io version, stop
re-running `plot_empconf_panelA_schematic.py`, or the manual edits are lost silently.
