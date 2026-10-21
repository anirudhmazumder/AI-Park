#!/usr/bin/env python3
"""
colored_subgraphs.py - the five subgraph figures of the manuscript.

    python3 KG_Visualization/colored_subgraphs.py [--png] [--manuscript] [--open]

  subgraph_hdac6.html            Figure 3  - HDAC6 and its direct neighbours
  subgraph_tau.html              Figure 4  - Tau and its direct neighbours
  subgraph_hdac2_hdac4_saha.html Figure 5  - HDAC2, HDAC4, SAHA and their neighbours
  subgraph_1_hubs.html           Figure S2 - the ten highest-degree nodes and their neighbours
                                             (254 nodes / 304 edges)
  subgraph_2_high_impact.html    Figure S1 - edges supported by a journal with IF >= 10
                                             (108 nodes / 97 edges)

All subgraphs are views of the 508-node graph built by kg_graph and use the
frozen layout in Final_KG/node_positions.json.  ``--png`` writes annotated PNGs
to KG_Visualization/Saved_Visualizations/; ``--manuscript`` writes the plot-area
only versions used in the manuscript to Saved_Visualizations/manuscript/.
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


# Subgraph extractors

def focus_subgraph(G, names, focus):
    """Induced subgraph of the focus entities and all their direct neighbours."""
    seeds = []
    for name in focus:
        canon = names.resolve(name)
        if canon is None or canon not in G:
            raise ValueError(f"{name!r} not found in graph")
        seeds.append(canon)
        print(f"  Matched {name!r} -> {canon!r}")
    nodes = set(seeds)
    for s in seeds:
        nodes.update(G.neighbors(s))
    sub = kg.induced_subgraph(G, nodes)
    print(f"  Focus subgraph: {sub.number_of_nodes()} nodes, {sub.number_of_edges()} edges")
    return sub, seeds


def hub_subgraph(G, top_n=10):
    """The ``top_n`` highest-degree nodes (ties broken alphabetically) and their neighbours."""
    hubs = [n for n, _ in sorted(G.degree(), key=lambda x: (-x[1], x[0]))[:top_n]]
    print(f"  Top {top_n} hubs by degree:")
    for h in hubs:
        print(f"    {G.degree(h):4d}  {h}")
    nodes = set(hubs)
    for h in hubs:
        nodes.update(G.neighbors(h))
    sub = kg.induced_subgraph(G, nodes)
    print(f"  Hub subgraph: {sub.number_of_nodes()} nodes, {sub.number_of_edges()} edges")
    return sub, hubs


def high_impact_subgraph(G, min_if=10):
    edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("max_impact_factor", 0) >= min_if]
    sub = kg.edge_induced_subgraph(G, edges)
    print(f"  High-impact subgraph (IF ≥ {min_if}): {sub.number_of_nodes()} nodes, {sub.number_of_edges()} edges")
    return sub


# Drawing

def node_trace(sub, pos, highlight=()):
    highlight = set(highlight)
    xs, ys, texts, sizes, colors, hovers = [], [], [], [], [], []
    for node in sub.nodes():
        x, y = pos[node]
        degree = sub.degree(node)
        xs.append(x)
        ys.append(y)
        sizes.append(max(15, min(45, 15 + degree * 2)))
        colors.append(200 if node in highlight else degree)
        if node in highlight or degree > 5 or sub.number_of_nodes() <= 80:
            texts.append(node[:35] + "…" if len(node) > 35 else node)
        else:
            texts.append("")
        hover = f"<b>{node}</b><br>Connections: {degree}"
        sample = list(sub.neighbors(node))[:4]
        if sample:
            hover += "<br><br>Sample connections:"
            for nb in sample:
                ed = sub[node][nb]
                hover += f"<br>  [{kp.categorize_verb(ed.get('verb_category'))}] {ed.get('predicates', ['?'])[0]} -> {nb}"
        hovers.append(hover)
    return go.Scatter(
        x=xs, y=ys, mode="markers+text", text=texts, textposition="top center",
        textfont=dict(size=9, color="black"), hoverinfo="text", hovertext=hovers,
        hovertemplate="%{hovertext}<extra></extra>", showlegend=False,
        marker=dict(size=sizes, color=colors, colorscale="Viridis", showscale=True,
                    colorbar=dict(thickness=15, len=0.7, xanchor="left",
                                  title=dict(text="Node<br>connections", side="right")),
                    line=dict(width=1.5, color="white"), opacity=0.9),
    )


def draw(sub, positions, title, html_name, png_name, highlight=(), png=False, open_browser=False,
         manuscript_name=None, label_size=kp.PRINT_LABEL_SIZE):
    if sub.number_of_nodes() == 0:
        print("  Subgraph is empty — skipping")
        return
    pos = kg.positions_for(sub, positions)
    fig = go.Figure(data=[kp.thin_edge_trace(sub, pos), node_trace(sub, pos, highlight)] + kp.legend_traces())
    fig.update_layout(**kp.base_layout(
        f"{title}<br><sub>Nodes: {sub.number_of_nodes()} | Edges: {sub.number_of_edges()} | "
        "<span style='color:#DC3232'>Red = activating/binding</span> · "
        "<span style='color:#1E64D2'>Blue = inhibiting/degrading</span> · Black = unknown</sub>",
        height=950, annotations=kp.arrow_annotations(sub, pos)))
    kp.save_figure(fig, os.path.join(OUT_DIR, html_name),
                   os.path.join(PNG_DIR, png_name) if png else None, open_browser=open_browser)
    if manuscript_name:
        kp.save_manuscript_png(fig, os.path.join(PNG_DIR, kp.PRINT_DIR_NAME, manuscript_name),
                               label_size=label_size)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--png", action="store_true", help="also export static PNGs (needs kaleido)")
    ap.add_argument("--manuscript", action="store_true",
                    help="also export plot-area-only PNGs as used in the manuscript (needs kaleido)")
    ap.add_argument("--label-size", type=float, default=kp.PRINT_LABEL_SIZE,
                    help=f"node-label font size in the manuscript PNGs (default {kp.PRINT_LABEL_SIZE})")
    ap.add_argument("--open", action="store_true", help="open each HTML in a browser")
    args = ap.parse_args()
    ms = (lambda name: name) if args.manuscript else (lambda name: None)
    ls = args.label_size

    os.makedirs(OUT_DIR, exist_ok=True)
    triplets, meta = kg.load_all()
    G, info = kg.build_canonical_graph(triplets, meta)
    names = info["names"]
    print(f"Full graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    positions = kg.load_positions()
    if positions is None:
        print("WARNING: Final_KG/node_positions.json not found — run colored_kg.py first; using a fresh layout")

    print("\n--- Figure 3: HDAC6 subgraph ---")
    sub, seeds = focus_subgraph(G, names, ["histone deacetylase 6"])
    draw(sub, positions, "Subgraph: HDAC6", "subgraph_hdac6.html", "Figure3_hdac6_subgraph.png", seeds, args.png, args.open,
         manuscript_name=ms("Figure3.png"), label_size=ls)

    print("\n--- Figure 4: Tau subgraph ---")
    sub, seeds = focus_subgraph(G, names, ["tau"])
    draw(sub, positions, "Subgraph: Tau", "subgraph_tau.html", "Figure4_tau_subgraph.png", seeds, args.png, args.open,
         manuscript_name=ms("Figure4.png"), label_size=ls)

    print("\n--- Figure 5: HDAC2 / HDAC4 / SAHA subgraph ---")
    sub, seeds = focus_subgraph(G, names, ["histone deacetylase 2", "histone deacetylase 4",
                                           "suberoylanilide hydroxamic acid"])
    draw(sub, positions, "Subgraph: HDAC2 / HDAC4 / SAHA", "subgraph_hdac2_hdac4_saha.html",
         "Figure5_hdac2_hdac4_saha_subgraph.png", seeds, args.png, args.open,
         manuscript_name=ms("Figure5.png"), label_size=ls)

    print("\n--- Figure S2: hub subgraph ---")
    sub, hubs = hub_subgraph(G, top_n=10)
    draw(sub, positions, "Hub Subgraph: the ten highest-degree nodes and their neighbours",
         "subgraph_1_hubs.html", "FigureS2_hub_subgraph.png", hubs, args.png, args.open,
         manuscript_name=ms("FigureS2.png"), label_size=ls)

    print("\n--- Figure S1: high-impact subgraph ---")
    sub = high_impact_subgraph(G, min_if=10)
    draw(sub, positions, "High-impact subgraph (journal impact factor ≥ 10)",
         "subgraph_2_high_impact.html", "FigureS1_high_impact_subgraph.png", (), args.png, args.open,
         manuscript_name=ms("FigureS1.png"), label_size=ls)

    print("\nDONE")


if __name__ == "__main__":
    main()
