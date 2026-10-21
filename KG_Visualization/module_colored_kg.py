#!/usr/bin/env python3
"""
module_colored_kg.py - Figure 2: the full 508-node graph with nodes coloured by
community module.

    python3 KG_Visualization/module_colored_kg.py [--png] [--manuscript] [--open] [--fixed-paper-matching]

The three largest modules are the ones reported in the manuscript
(``kg_graph.build_module_graph`` + Louvain, seed 42):
  Module 1 - HDAC6-centred            (122 nodes)  red
  Module 2 - Tau-centred              (37 nodes)   cyan
  Module 3 - HDAC2/HDAC4/SAHA-centred (33 nodes)   green
  all other nodes                                   black

Module membership is transferred from the module graph onto the 508-node graph
with ``kg_graph.module_partition_on_canonical`` (see
KG_Analysis/module_output/module_assignment_canonical.csv for the mapping).
"""

from __future__ import annotations

import argparse
import os
import sys

import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kg_graph as kg   # noqa: E402
import kg_plot as kp    # noqa: E402

OUT_DIR = os.path.join(kg.ROOT, "KG_Visualization", "Final_Visualizations")
PNG_DIR = os.path.join(kg.ROOT, "KG_Visualization", "Saved_Visualizations")

MODULE_COLORS = {1: "#D62728", 2: "#17BECF", 3: "#009E73"}
DEFAULT_COLOR = "#111111"
MODULE_LABELS = {1: "HDAC6-centered", 2: "Tau-centered", 3: "HDAC2/HDAC4/SAHA-centered"}


def hover(G, node, meta):
    papers = [p for p in sorted(G.nodes[node]["papers"]) if kg.get_metadata(meta, p)]
    h = f"<b>{node}</b><br>Degree: {G.degree(node)}<br>"
    for p in papers[:2]:
        m = kg.get_metadata(meta, p)
        title = m["title"][:80] + "..." if len(m["title"]) > 80 else m["title"]
        h += f"<br>📄 {title}<br>IF: {m['impact_factor'] if m['impact_factor'] is not None else 'N/A'}"
    return h


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--png", action="store_true", help="also export a static PNG (needs kaleido)")
    ap.add_argument("--manuscript", action="store_true",
                    help="also export the plot-area-only PNG used as Figure 2 (needs kaleido)")
    ap.add_argument("--label-size", type=float, default=13,
                    help="node-label font size in the manuscript PNG (default 13; the graph is dense)")
    ap.add_argument("--open", action="store_true", help="open the HTML in a browser")
    ap.add_argument("--fixed-paper-matching", action="store_true",
                    help="use corrected paper matching for the module graph (see kg_graph.build_module_graph)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    triplets, meta = kg.load_all()
    G, info = kg.build_canonical_graph(triplets, meta)
    print(f"Canonical graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    Gm, _, _ = kg.build_module_graph(triplets, meta, legacy_paper_matching=not args.fixed_paper_matching)
    Gm_lcc = kg.largest_component(Gm)
    modules = kg.louvain_modules(Gm_lcc)
    sizes = [len(m) for m in modules[:3]]
    print(f"Module graph: {Gm.number_of_nodes()} nodes; Louvain (seed {kg.LOUVAIN_SEED}): {len(modules)} modules; "
          f"top three sizes {sizes}")
    assignment, conflicts = kg.module_partition_on_canonical(modules, info["names"])
    if conflicts:
        print(f"  {len(conflicts)} canonical node(s) have spellings in different modules; "
              f"assigned to the largest module:")
        for canon, ranks in conflicts:
            print(f"    {canon!r}: modules {ranks} -> {assignment[canon]}")

    positions = kg.load_positions()
    if positions is None:
        print("WARNING: Final_KG/node_positions.json not found — run colored_kg.py first; using a fresh layout")
    pos = kg.positions_for(G, positions)

    groups = {1: [], 2: [], 3: [], 0: []}
    for node in G.nodes():
        rank = assignment.get(node, 0)
        groups[rank if rank in (1, 2, 3) else 0].append(node)
    print("  Nodes coloured per module:", {k: len(v) for k, v in groups.items()})

    traces = [kp.thin_edge_trace(G, pos, width=0.3, color="rgba(180,180,180,0.12)")]
    for gid in (1, 2, 3, 0):
        nodes = groups[gid]
        if not nodes:
            continue
        label = (f"Module {gid} — {MODULE_LABELS[gid]} ({sizes[gid - 1]} nodes)" if gid else "Other nodes")
        traces.append(go.Scatter(
            x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes],
            mode="markers+text", name=label,
            text=[(n[:40] + "...") if (G.degree(n) > 5 and len(n) > 40) else (n if G.degree(n) > 5 else "")
                  for n in nodes],
            textposition="top center", textfont=dict(size=8, color="#222222"),
            hoverinfo="text", hovertext=[hover(G, n, meta) for n in nodes],
            hovertemplate="%{hovertext}<extra></extra>",
            marker=dict(size=[max(8, min(30, 8 + G.degree(n) * 1.4)) for n in nodes],
                        color=MODULE_COLORS.get(gid, DEFAULT_COLOR), opacity=0.88,
                        line=dict(width=1.2, color="white")),
        ))
    traces += kp.legend_traces()

    caption = (
        "Knowledge Graph — Nodes Colored by Community Module<br><sub>"
        f"<span style='color:{MODULE_COLORS[1]}'>● Module 1 ({MODULE_LABELS[1]}, {sizes[0]} nodes)</span>   "
        f"<span style='color:{MODULE_COLORS[2]}'>● Module 2 ({MODULE_LABELS[2]}, {sizes[1]} nodes)</span>   "
        f"<span style='color:{MODULE_COLORS[3]}'>● Module 3 ({MODULE_LABELS[3]}, {sizes[2]} nodes)</span>   "
        "<span style='color:#111111'>● Other / isolated nodes</span><br>"
        "Edge arrows: <span style='color:#DC3232'>Red = activating/binding</span> · "
        "<span style='color:#1E64D2'>Blue = inhibiting/degrading</span> · "
        "<span style='color:#1E1E1E'>Black = unknown</span> · Hover over nodes for details</sub>"
    )
    fig = go.Figure(data=traces)
    fig.update_layout(**kp.base_layout(caption, width=1600, height=950, legend_title="Node modules / Edge types",
                                       margin=dict(b=40, l=40, r=220, t=120),
                                       annotations=kp.arrow_annotations(G, pos)))
    fig.update_layout(title=dict(font=dict(size=16)), legend=dict(tracegroupgap=8))
    kp.save_figure(fig, os.path.join(OUT_DIR, "module_colored_kg.html"),
                   os.path.join(PNG_DIR, "Figure2_full_kg_modules.png") if args.png else None,
                   open_browser=args.open)
    if args.manuscript:
        kp.save_manuscript_png(fig, os.path.join(PNG_DIR, kp.PRINT_DIR_NAME, "Figure2.png"),
                               label_size=args.label_size)
    print("DONE")


if __name__ == "__main__":
    main()
