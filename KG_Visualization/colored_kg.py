#!/usr/bin/env python3
"""
colored_kg.py - the full 508-node knowledge graph as an interactive figure.

    python3 KG_Visualization/colored_kg.py [--png] [--open] [--relayout]

Outputs
  KG_Visualization/Final_Visualizations/knowledge_graph_visualization.html
  KG_Visualization/Saved_Visualizations/full_kg_degree_colored.png  (with --png; nodes coloured by degree)
  Final_KG/node_details.csv            per-node degree, papers and sample connections
  Final_KG/node_positions.json         frozen layout (only written when missing, or with --relayout)

Node positions come from the frozen layout in Final_KG/node_positions.json so
the figure matches the manuscript; nodes absent from that file are placed with a
seeded spring layout.  ``--relayout`` recomputes the whole layout
(Fruchterman-Reingold, k=0.5, 100 iterations, seed 42) and overwrites the file.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

import networkx as nx
import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kg_graph as kg   # noqa: E402
import kg_plot as kp    # noqa: E402

OUT_DIR  = os.path.join(kg.ROOT, "KG_Visualization", "Final_Visualizations")
PNG_DIR  = os.path.join(kg.ROOT, "KG_Visualization", "Saved_Visualizations")
HTML_OUT = os.path.join(OUT_DIR, "knowledge_graph_visualization.html")
PNG_OUT  = os.path.join(PNG_DIR, "full_kg_degree_colored.png")
NODE_CSV = os.path.join(kg.ROOT, "Final_KG", "node_details.csv")
MISSING_REPORT = os.path.join(OUT_DIR, "missing_papers_report.txt")


def first_three_authors(authors):
    if not authors or authors == "Unknown":
        return "Unknown authors"
    a = [x.strip() for x in str(authors).split(",")]
    return ", ".join(a[:3]) + (" et al." if len(a) > 3 else "")


def node_hover(G, node, meta):
    info = G.nodes[node]
    papers = sorted(info["papers"])
    with_meta = [p for p in papers if kg.get_metadata(meta, p)]
    hover = f"<b>{node}</b><br><br><b>Total Connections:</b> {G.degree(node)}<br><br>"
    conns = info["connections"][:3]
    if conns:
        hover += f"<b>Sample Connections (showing {len(conns)} of {len(info['connections'])}):</b><br>"
        for i, c in enumerate(conns, 1):
            hover += f"{i}. {c['predicate']} -> <b>{c['target']}</b><br>"
        hover += "<br>"
    if with_meta:
        hover += f"<b>Papers ({len(with_meta)} with metadata):</b><br>"
        for p in with_meta[:3]:
            m = kg.get_metadata(meta, p)
            title = m["title"][:100] + "..." if len(m["title"]) > 100 else m["title"]
            hover += (f"<br>📄 <b>{title}</b><br>   Authors: {first_three_authors(m['authors'])}<br>"
                      f"   Impact Factor: {m['impact_factor'] if m['impact_factor'] is not None else 'N/A'}<br>"
                      f"   Link: {m['link']}<br>")
        if len(with_meta) > 3:
            hover += f"<br>... and {len(with_meta) - 3} more papers<br>"
    return hover


def summary_report(G, info, meta):
    print("\n" + "=" * 80 + "\nKNOWLEDGE GRAPH SUMMARY REPORT\n" + "=" * 80)
    print(f"  Triplets: {info['triplets']}")
    print(f"  Node normalisation: {info['nodes_before_normalisation']} -> {info['nodes_after_normalisation']} "
          f"({info['nodes_merged']} merged)")
    print(f"  Excluded {info['excluded_missing_metadata']} triplets (no metadata), "
          f"{info['excluded_node']} triplets (excluded node)")
    print(f"  Nodes: {G.number_of_nodes()}   Edges: {G.number_of_edges()}   Density: {nx.density(G):.6f}")
    degrees = [d for _, d in G.degree()]
    print(f"  Degree: mean={np.mean(degrees):.2f}  median={np.median(degrees):.2f}  "
          f"max={max(degrees)}  min={min(degrees)}")
    print("\n  Top 10 hub nodes:")
    for i, (n, d) in enumerate(sorted(G.degree(), key=lambda x: (-x[1], x[0]))[:10], 1):
        print(f"    {i:2d}. {n}  (degree={d}, papers={len(G.nodes[n]['papers'])})")
    comps = list(nx.connected_components(G))
    print(f"\n  Connected components: {len(comps)}  (largest: {len(max(comps, key=len))} nodes)")
    vc_counts = {}
    for _, _, d in G.edges(data=True):
        vc_counts[d.get("verb_category")] = vc_counts.get(d.get("verb_category"), 0) + 1
    print(f"\n  Verb categories on edges ({len(vc_counts)} unique):")
    for vc, cnt in sorted(vc_counts.items(), key=lambda x: -x[1]):
        print(f"    [{kp.categorize_verb(vc):5s}] {cnt:4d}  {vc}")


_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def export_node_details(G, meta, path):
    def clean(text):
        # control characters (OCR artefacts in a few entity names) become spaces
        return " ".join(_CONTROL.sub(" ", str(text if text is not None else "")).split())

    rows = []
    for node in G.nodes():
        info = G.nodes[node]
        papers = sorted(info["papers"])
        with_meta = [p for p in papers if kg.get_metadata(meta, p)]
        row = {
            "Node": clean(node),
            "Total_Connections": str(G.degree(node)),
            "Total_Papers": str(len(papers)),
            "Papers_With_Metadata": str(len(with_meta)),
            "Sample_Connections": clean(" | ".join(f"{c['predicate']} -> {c['target']}" for c in info["connections"][:3])),
        }
        for i, p in enumerate(with_meta[:3], 1):
            m = kg.get_metadata(meta, p)
            row[f"Paper_{i}_Title"] = clean(m["title"])
            row[f"Paper_{i}_Authors"] = clean(first_three_authors(m["authors"]))
            row[f"Paper_{i}_Link"] = clean(m["link"])
            row[f"Paper_{i}_Impact"] = clean(f"{m['impact_factor']:g}" if m["impact_factor"] is not None else "N/A")
        rows.append(row)
    rows.sort(key=lambda r: (-int(r["Total_Connections"]), r["Node"]))
    fieldnames = ["Node", "Total_Connections", "Total_Papers", "Papers_With_Metadata", "Sample_Connections"] + \
                 [f"Paper_{i}_{k}" for i in (1, 2, 3) for k in ("Title", "Authors", "Link", "Impact")]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC,
                           escapechar="\\", doublequote=True, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  Node details exported to {path}")


def export_missing_papers_report(triplets, meta, path):
    missing = {}
    for t in triplets:
        ot = t["original_triplet"]
        paper = kg.paper_name(ot)
        if paper and kg.get_metadata(meta, paper) is None:
            m = missing.setdefault(paper, {"triplet_count": 0, "nodes": set(), "samples": []})
            m["triplet_count"] += 1
            m["nodes"].update([ot["Subject"], ot["Object"]])
            if len(m["samples"]) < 3:
                m["samples"].append((ot["Subject"], ot["Predicate"], ot["Object"]))
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\nMISSING PAPERS METADATA REPORT\n" + "=" * 80 + "\n\n")
        f.write(f"Total papers without metadata: {len(missing)}\n")
        f.write(f"Total triplets affected: {sum(i['triplet_count'] for i in missing.values())}\n\n")
        for i, (paper, m) in enumerate(sorted(missing.items(), key=lambda x: -x[1]["triplet_count"]), 1):
            f.write(f"{i}. Paper Name: {paper}\n   Triplet Count: {m['triplet_count']}\n"
                    f"   Unique Nodes: {len(m['nodes'])}\n")
            for j, (s, p, o) in enumerate(m["samples"], 1):
                f.write(f"   {j}. {s} --[{p}]--> {o}\n")
            f.write("\n" + "-" * 80 + "\n\n")
    print(f"  Missing papers report exported to {path} ({len(missing)} papers)")


def build_figure(G, pos, meta):
    node_x, node_y, node_text, node_sizes, node_colors, node_hover_ = [], [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        degree = G.degree(node)
        node_x.append(x)
        node_y.append(y)
        node_sizes.append(max(10, min(30, 10 + degree * 1.5)))
        node_colors.append(degree)
        node_text.append((node[:40] + "...") if (degree > 5 and len(node) > 40) else (node if degree > 5 else ""))
        node_hover_.append(node_hover(G, node, meta))

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text", text=node_text, textposition="top center",
        textfont=dict(size=9, color="black"), hoverinfo="text", hovertext=node_hover_,
        hovertemplate="%{hovertext}<extra></extra>", showlegend=False,
        marker=dict(size=node_sizes, color=node_colors, colorscale="Viridis", showscale=True,
                    colorbar=dict(thickness=15, len=0.7, xanchor="left",
                                  title=dict(text="Node<br>Connections", side="right"),
                                  tickmode="linear", tick0=0,
                                  dtick=max(1, max(node_colors) // 10) if node_colors else 1),
                    line=dict(width=1.5, color="white"), opacity=0.9),
    )
    fig = go.Figure(data=[kp.thin_edge_trace(G, pos), node_trace] + kp.legend_traces())
    fig.update_layout(**kp.base_layout(
        "Knowledge Graph — Interactive Visualisation<br>"
        "<sub>Arrows: <span style='color:#DC3232'>Red = activating/binding</span> · "
        "<span style='color:#1E64D2'>Blue = inhibiting/degrading</span> · "
        "<span style='color:#1E1E1E'>Black = unknown</span> · Hover over nodes for details</sub>",
        annotations=kp.arrow_annotations(G, pos)))
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--png", action="store_true", help="also export a static PNG (needs kaleido)")
    ap.add_argument("--open", action="store_true", help="open the HTML in a browser")
    ap.add_argument("--relayout", action="store_true",
                    help="recompute the spring layout and overwrite Final_KG/node_positions.json")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    triplets, meta = kg.load_all()
    G, info = kg.build_canonical_graph(triplets, meta)
    summary_report(G, info, meta)

    frozen = None if args.relayout else kg.load_positions()
    if frozen is None:
        print("\nComputing spring layout (seed 42)...")
    pos = kg.positions_for(G, frozen)
    missing = [n for n in G.nodes() if not frozen or n not in frozen]
    if frozen is not None and missing:
        print(f"\n{len(missing)} node(s) not in the frozen layout were placed with a seeded spring layout: {missing}")
    if frozen is None or missing:
        kg.save_positions(pos)
        print(f"  Layout saved to {kg.POSITIONS_JSON} ({len(pos)} nodes)")

    print("\nCreating visualisation...")
    fig = build_figure(G, pos, meta)
    kp.save_figure(fig, HTML_OUT, PNG_OUT if args.png else None, open_browser=args.open)
    export_node_details(G, meta, NODE_CSV)
    export_missing_papers_report(triplets, meta, MISSING_REPORT)
    print("\nDONE")


if __name__ == "__main__":
    main()
