#!/usr/bin/env python3
"""
graph_statistics.py - statistics of the knowledge graph.

    python3 KG_Analysis/graph_statistics.py                   # table on stdout
    python3 KG_Analysis/graph_statistics.py --json out.json   # also write the values to a file
    python3 KG_Analysis/graph_statistics.py --fixed-paper-matching

Computes, from the data in Final_KG/: the corpus summary; the number of nodes
and edges, density and degree statistics of the graph; in- and out-degree
statistics; connected components and small islands; the average clustering
coefficient and the average shortest path length of the largest component;
the hub nodes and their centralities; the community modules (Louvain on the
module graph, see ``kg_graph.build_module_graph``); and the sizes of the
subgraphs drawn in the figures.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kg_graph as kg  # noqa: E402

ROWS = []   # (section, quantity, value)


def row(section, quantity, value):
    ROWS.append((section, quantity, value))


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def degree_stats(section, label, values):
    arr = np.array(values, dtype=float)
    row(section, f"{label} mean", float(arr.mean()))
    row(section, f"{label} median", float(np.median(arr)))
    row(section, f"{label} min", int(arr.min()))
    row(section, f"{label} max", int(arr.max()))
    row(section, f"{label} standard deviation (population)", float(arr.std()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", help="write the table to this JSON file")
    ap.add_argument("--fixed-paper-matching", action="store_true",
                    help="build the module graph with corrected paper matching (see kg_graph.build_module_graph)")
    args = ap.parse_args()

    triplets, meta = kg.load_all()

    # Corpus
    papers = kg.unique_papers(meta)
    years = [p["year"] for p in papers if p["year"]]
    ifs = [p["impact_factor"] for p in papers if p["impact_factor"] is not None]
    row("Corpus", "papers", len(papers))
    row("Corpus", "publication years", f"{min(years)}–{max(years)}")
    row("Corpus", "journal impact factor range", f"{min(ifs):g}–{max(ifs):g}")

    # Triplets
    row("Triplets", "triplets in Final_KG", len(triplets))

    # Graph
    G, info = kg.build_canonical_graph(triplets, meta)
    names = info["names"]
    row("Graph", "entity strings before normalization", info["nodes_before_normalisation"])
    row("Graph", "nodes", G.number_of_nodes())
    row("Graph", "entity strings merged by normalization", info["nodes_merged"])
    row("Graph", "edges", G.number_of_edges())
    row("Graph", "triplets excluded (paper without metadata)", info["excluded_missing_metadata"])
    row("Graph", "triplets excluded (excluded node)", info["excluded_node"])
    row("Graph", "density", nx.density(G))
    degree_stats("Degree", "degree", [d for _, d in G.degree()])

    Gd, _ = kg.build_canonical_graph(triplets, meta, directed=True)
    row("Directed", "subject -> object pairs", Gd.number_of_edges())
    degree_stats("Directed", "in-degree", [d for _, d in Gd.in_degree()])
    degree_stats("Directed", "out-degree", [d for _, d in Gd.out_degree()])

    # Connectivity and structure
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    sizes = Counter(len(c) for c in comps)
    row("Connectivity", "connected components", len(comps))
    row("Connectivity", "largest component (nodes)", len(comps[0]))
    row("Connectivity", "small islands (at most 5 nodes)", sum(1 for c in comps if len(c) <= 5))
    for k in (2, 3, 4, 5):
        row("Connectivity", f"islands of {k} nodes", sizes.get(k, 0))
    row("Connectivity", "component size distribution", {str(k): v for k, v in sorted(sizes.items())})
    row("Structure", "average clustering coefficient", nx.average_clustering(G))
    G_lcc = kg.largest_component(G)
    row("Structure", "average shortest path length (largest component)", nx.average_shortest_path_length(G_lcc))

    # Hubs
    dc = nx.degree_centrality(G)
    bc = nx.betweenness_centrality(G, normalized=True)
    top10 = sorted(G.degree(), key=lambda x: (-x[1], x[0]))[:10]
    row("Hubs", "ten highest-degree nodes", [[n, d] for n, d in top10])
    hub, hub_deg = top10[0]
    row("Hubs", f"degree centrality of {hub}", dc[hub])
    row("Hubs", f"betweenness centrality of {hub}", bc[hub])
    bmax = max(bc, key=bc.get)
    row("Hubs", "node with the highest betweenness centrality", [bmax, bc[bmax]])

    # Modules
    Gm, _, minfo = kg.build_module_graph(triplets, meta, legacy_paper_matching=not args.fixed_paper_matching)
    Gm_lcc = kg.largest_component(Gm)
    modules = kg.louvain_modules(Gm_lcc)
    row("Modules", "module graph nodes", Gm.number_of_nodes())
    row("Modules", "module graph edges", Gm.number_of_edges())
    row("Modules", "module graph largest component (nodes)", Gm_lcc.number_of_nodes())
    row("Modules", "triplets omitted from the module graph (paper matching)", minfo["skipped"]["missing_metadata"])
    row("Modules", f"modules (Louvain, seed {kg.LOUVAIN_SEED}, resolution {kg.LOUVAIN_RESOLUTION})", len(modules))
    for i, mod in enumerate(modules[:3], start=1):
        sub = Gm_lcc.subgraph(mod)
        degs = dict(sub.degree())
        sec = f"Module {i}"
        row(sec, "nodes", sub.number_of_nodes())
        row(sec, "edges", sub.number_of_edges())
        row(sec, "density", nx.density(sub))
        row(sec, "average degree within module", sum(degs.values()) / len(degs))
        row(sec, "highest-degree nodes within module", [[n, d] for n, d in sorted(degs.items(), key=lambda x: (-x[1], x[0]))[:5]])

    # Subgraphs
    hi_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("max_impact_factor", 0) >= 10]
    G_hi = kg.edge_induced_subgraph(G, hi_edges)
    row("Subgraphs", "impact factor ≥ 10 subgraph nodes", G_hi.number_of_nodes())
    row("Subgraphs", "impact factor ≥ 10 subgraph edges", G_hi.number_of_edges())
    hub_set = {n for n, _ in top10}
    for n, _ in top10:
        hub_set.update(G.neighbors(n))
    G_hub = kg.induced_subgraph(G, hub_set)
    row("Subgraphs", "hub subgraph (ten highest-degree nodes and neighbours) nodes", G_hub.number_of_nodes())
    row("Subgraphs", "hub subgraph (ten highest-degree nodes and neighbours) edges", G_hub.number_of_edges())

    def neighbourhood(*entities):
        seeds = [names.resolve(e) for e in entities]
        s = set(seeds)
        for n in seeds:
            s.update(G.neighbors(n))
        sub = kg.induced_subgraph(G, s)
        return [sub.number_of_nodes(), sub.number_of_edges()]
    row("Subgraphs", "HDAC6 neighbourhood nodes/edges", neighbourhood("histone deacetylase 6"))
    row("Subgraphs", "Tau neighbourhood nodes/edges", neighbourhood("tau"))
    row("Subgraphs", "HDAC2/HDAC4/SAHA neighbourhood nodes/edges",
        neighbourhood("histone deacetylase 2", "histone deacetylase 4", "suberoylanilide hydroxamic acid"))

    # Output
    width = max(len(r[1]) for r in ROWS)
    print(f"{'Section':13s} {'Quantity':{width}s}  Value")
    print("-" * (width + 30))
    for section, quantity, value in ROWS:
        print(f"{section:13s} {quantity:{width}s}  {fmt(value)}")
    print(f"\nnetworkx {nx.__version__}, numpy {np.__version__}, python {sys.version.split()[0]}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump([{"section": s, "quantity": q, "value": v} for s, q, v in ROWS],
                      f, indent=2, ensure_ascii=False, default=str)
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
