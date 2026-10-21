#!/usr/bin/env python3
"""
colored_timescale_graph.py - Supplementary Video S1: growth of the knowledge
graph by publication year (interactive animation with a year slider).

    python3 KG_Visualization/colored_timescale_graph.py [--open]

Output: KG_Visualization/Final_Visualizations/knowledge_graph_timeseries.html
(Supplementary Video S1 is a screen recording of this animation.)

For each year Y the frame shows the subgraph of the 508-node graph formed by
the edges supported by at least one paper published in or before Y.  Nodes use
the frozen layout in Final_KG/node_positions.json.
"""

from __future__ import annotations

import argparse
import os
import sys

import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kg_graph as kg   # noqa: E402
import kg_plot as kp    # noqa: E402

OUT_DIR  = os.path.join(kg.ROOT, "KG_Visualization", "Final_Visualizations")
HTML_OUT = os.path.join(OUT_DIR, "knowledge_graph_timeseries.html")
REPORT   = os.path.join(OUT_DIR, "missing_metadata_timeseries_report.txt")

COLOR_GROUPS = ["red", "blue", "black", "gray"]


def edge_coords_by_color(G, pos):
    coords = {c: ([], []) for c in COLOR_GROUPS}
    for u, v, data in G.edges(data=True):
        xs, ys = coords[kp.categorize_verb(data.get("verb_category"))]
        xs.extend([pos[u][0], pos[v][0], None])
        ys.extend([pos[u][1], pos[v][1], None])
    return coords


def edge_traces(coords, visible_in_legend):
    traces = []
    for grp in COLOR_GROUPS:
        label, color = kp.COLOR_META[grp]
        xs, ys = coords[grp]
        traces.append(go.Scatter(x=xs, y=ys, mode="lines",
                                 line=dict(width=1.2 if grp != "gray" else 0.5, color=color),
                                 hoverinfo="none", name=label,
                                 showlegend=visible_in_legend and grp != "gray", legendgroup=grp))
    return traces


def node_trace(G, pos):
    xs, ys, texts, sizes, colors, hovers = [], [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        degree = G.degree(node)
        xs.append(x)
        ys.append(y)
        sizes.append(max(10, min(30, 10 + degree * 1.5)))
        colors.append(degree)
        texts.append((node[:30] + "…") if (degree > 10 and len(node) > 30) else (node if degree > 10 else ""))
        hover = f"<b>{node}</b><br>Connections: {degree}"
        sample = list(G.neighbors(node))[:3]
        if sample:
            hover += "<br><br>Sample connections:"
            for nb in sample:
                hover += f"<br>  {G[node][nb].get('predicates', ['?'])[0]} -> {nb}"
        hovers.append(hover)
    return go.Scatter(
        x=xs, y=ys, mode="markers+text", text=texts, textposition="top center",
        textfont=dict(size=8, color="black"), hoverinfo="text", hovertext=hovers,
        hovertemplate="%{hovertext}<extra></extra>", showlegend=False,
        marker=dict(size=sizes, color=colors, colorscale="Viridis", showscale=True,
                    colorbar=dict(thickness=15, len=0.7, xanchor="left",
                                  title=dict(text="Node<br>connections", side="right")),
                    line=dict(width=1.5, color="white"), opacity=0.9),
    )


def cumulative_graphs(G):
    """{year: subgraph of edges first reported in or before that year}."""
    years = sorted({min(d["years"]) for _, _, d in G.edges(data=True) if d.get("years")})
    out = {}
    for y in years:
        edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("years") and min(d["years"]) <= y]
        out[y] = kg.edge_induced_subgraph(G, edges)
    return out


def write_missing_report(triplets, meta, path):
    missing = {}
    for t in triplets:
        ot = t["original_triplet"]
        paper = kg.paper_name(ot)
        m = kg.get_metadata(meta, paper)
        if not m or not m.get("year"):
            key = paper or "(empty paper name)"
            e = missing.setdefault(key, {"count": 0, "nodes": set(), "samples": []})
            e["count"] += 1
            e["nodes"].update([ot["Subject"], ot["Object"]])
            if len(e["samples"]) < 3:
                e["samples"].append((ot["Subject"], ot["Predicate"], ot["Object"]))
    with open(path, "w", encoding="utf-8") as f:
        total = sum(e["count"] for e in missing.values())
        f.write("=" * 80 + "\nMISSING METADATA REPORT — TIME SERIES\n" + "=" * 80 + "\n\n")
        f.write(f"Papers without metadata/year: {len(missing)}\nTriplets excluded: {total}\n\n")
        for i, (name, e) in enumerate(sorted(missing.items(), key=lambda x: -x[1]["count"]), 1):
            f.write(f"{i}. {name}\n   Triplets: {e['count']}\n   Nodes: {len(e['nodes'])}\n")
            for j, (s, p, o) in enumerate(e["samples"], 1):
                f.write(f"   {j}. {s} --[{p}]--> {o}\n")
            f.write("\n" + "-" * 80 + "\n\n")
    print(f"  Report saved to {path}  ({len(missing)} papers, {total} triplets excluded)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--open", action="store_true", help="open the HTML in a browser")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    triplets, meta = kg.load_all()
    G, _ = kg.build_canonical_graph(triplets, meta)
    write_missing_report(triplets, meta, REPORT)

    graphs = cumulative_graphs(G)
    years = sorted(graphs)
    print(f"Triplets span {years[0]} – {years[-1]}")
    for y in years:
        print(f"  {y}: {graphs[y].number_of_nodes()} nodes, {graphs[y].number_of_edges()} edges")

    positions = kg.load_positions()
    if positions is None:
        print("WARNING: Final_KG/node_positions.json not found — run colored_kg.py first; using a fresh layout")
    pos = kg.positions_for(G, positions)

    def frame_data(y, legend):
        Gy = graphs[y]
        return edge_traces(edge_coords_by_color(Gy, pos), legend) + [node_trace(Gy, pos)]

    def subtitle(y):
        Gy = graphs[y]
        return (f"Knowledge Graph Evolution — Year: {y}<br><sub>Nodes: {Gy.number_of_nodes()} | "
                f"Edges: {Gy.number_of_edges()} | <span style='color:#DC3232'>Red = activating/binding</span> · "
                "<span style='color:#1E64D2'>Blue = inhibiting/degrading</span> · Black = unknown</sub>")

    frames = [go.Frame(data=frame_data(y, False), name=str(y), layout=go.Layout(title=dict(text=subtitle(y))))
              for y in years]
    sliders = [dict(active=0, yanchor="top", y=0, xanchor="left",
                    currentvalue=dict(prefix="Year: ", visible=True, xanchor="center", font=dict(size=18, color="#333")),
                    transition=dict(duration=300, easing="cubic-in-out"), pad=dict(b=10, t=50), len=0.85, x=0.075,
                    steps=[dict(args=[[str(y)], dict(frame=dict(duration=300, redraw=True), mode="immediate",
                                                     transition=dict(duration=300, easing="cubic-in-out"))],
                                label=str(y), method="animate") for y in years])]
    updatemenus = [dict(type="buttons", showactive=False, y=0, x=0, xanchor="left", yanchor="top", pad=dict(t=50, r=10),
                        buttons=[dict(label="▶ Play", method="animate",
                                      args=[None, dict(frame=dict(duration=1000, redraw=True), fromcurrent=True,
                                                       transition=dict(duration=300, easing="cubic-in-out"))]),
                                 dict(label="⏸ Pause", method="animate",
                                      args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate",
                                                         transition=dict(duration=0))])])]

    fig = go.Figure(data=frame_data(years[0], True), frames=frames)
    layout = kp.base_layout(subtitle(years[0]), height=950, margin=dict(b=120, l=40, r=200, t=110),
                            sliders=sliders, updatemenus=updatemenus)
    layout["xaxis"]["fixedrange"] = True
    layout["yaxis"]["fixedrange"] = True
    fig.update_layout(**layout)
    kp.save_figure(fig, HTML_OUT, open_browser=args.open)
    print("DONE — drag the year slider or press ▶ Play in the HTML")


if __name__ == "__main__":
    main()
