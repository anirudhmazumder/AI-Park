#!/usr/bin/env python3
"""
module_analysis.py - Louvain community detection and per-module export.

    python3 KG_Analysis/module_analysis.py                      # manuscript setting
    python3 KG_Analysis/module_analysis.py --fixed-paper-matching

Community detection runs on the largest connected component of the *module
graph* (``kg_graph.build_module_graph``) with NetworkX's Louvain implementation
(``seed=42``, ``resolution=1.0``).  The default settings are the ones used for the
manuscript (27 modules; the three largest are HDAC6-, Tau- and HDAC2/HDAC4/SAHA-
centred).

Outputs (KG_Analysis/module_output/):
  module_summary.csv              one row per module (size, edges, density, avg degree, papers)
  module_nodes.csv                every node with its within-module degree
  module_papers.csv               paper -> module contributions with metadata
  modules_full.json               everything in one structured file
  module_assignment_canonical.csv module rank of every node of the 508-node graph
                                  (used to colour Figure 2), with conflict notes
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict

import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kg_graph as kg  # noqa: E402

OUTPUT_DIR = os.path.join(kg.ROOT, "KG_Analysis", "module_output")


def _safe(s):
    """Strip control characters that break csv.writer from any string field."""
    if not isinstance(s, str):
        return s
    return re.sub(r"[\x00-\x1f\x7f]", " ", s).strip()


def module_records(modules, G_lcc, node_papers):
    records = []
    for rank, mod in enumerate(modules, 1):
        sub = G_lcc.subgraph(mod)
        degrees = dict(sub.degree())
        papers = defaultdict(int)
        for node in mod:
            for paper in node_papers.get(node, ()):
                papers[paper] += 1
        records.append({
            "rank": rank,
            "nodes": sorted(mod, key=lambda n: (-degrees[n], n)),
            "sub": sub,
            "degrees": degrees,
            "papers": papers,
            "density": nx.density(sub),
            "avg_degree": sum(degrees.values()) / len(degrees) if degrees else 0.0,
        })
    return records


def print_summary(G, G_lcc, modules, records, meta, top_n=3):
    print(f"Module graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Largest connected component: {G_lcc.number_of_nodes()} nodes")
    print(f"Louvain (seed={kg.LOUVAIN_SEED}, resolution={kg.LOUVAIN_RESOLUTION}): {len(modules)} modules\n")
    print("=" * 70)
    for r in records[:top_n]:
        print(f"Module {r['rank']}  ({len(r['nodes'])} nodes, {r['sub'].number_of_edges()} edges, "
              f"density {r['density']:.4f}, avg degree {r['avg_degree']:.2f})")
        print("-" * 70)
        print("  Top hub nodes (degree within module):")
        for hub in r["nodes"][:10]:
            print(f"    [{r['degrees'][hub]:>3d}]  {hub}")
        print("  Top contributing papers:")
        for paper in sorted(r["papers"], key=lambda p: (-r["papers"][p], p))[:5]:
            m = kg.get_metadata(meta, paper)
            if m:
                print(f"    • {m['title']}")
                print(f"      {m['authors']} ({m['year']})  IF: {m['impact_factor']}")
            else:
                print(f"    • {paper}")
        print("=" * 70)


def save_outputs(records, G, G_lcc, node_papers, meta, assignment, conflicts, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "module_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["module_rank", "num_nodes", "num_edges", "density", "avg_degree_within_module", "num_papers"])
        for r in records:
            w.writerow([r["rank"], len(r["nodes"]), r["sub"].number_of_edges(),
                        round(r["density"], 6), round(r["avg_degree"], 4), len(r["papers"])])

    with open(os.path.join(out_dir, "module_nodes.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["module_rank", "node", "degree_within_module", "num_papers"])
        for r in records:
            for node in r["nodes"]:
                w.writerow([r["rank"], _safe(node), r["degrees"][node], len(node_papers.get(node, ()))])

    with open(os.path.join(out_dir, "module_papers.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["module_rank", "paper", "node_count", "title", "authors", "year", "impact_factor"])
        for r in records:
            for paper in sorted(r["papers"], key=lambda p: (-r["papers"][p], p)):
                m = kg.get_metadata(meta, paper)
                w.writerow([r["rank"], _safe(paper), r["papers"][paper],
                            _safe(m["title"]) if m else "", _safe(m["authors"]) if m else "",
                            m["year"] if m else "", m["impact_factor"] if m else ""])

    full = []
    for r in records:
        full.append({
            "module_rank": r["rank"],
            "num_nodes": len(r["nodes"]),
            "num_edges": r["sub"].number_of_edges(),
            "density": r["density"],
            "avg_degree_within_module": r["avg_degree"],
            "nodes": [{"node": n, "degree_within_module": r["degrees"][n]} for n in r["nodes"]],
            "papers": [
                {"paper": p, "node_count": r["papers"][p],
                 **({"title": m["title"], "authors": m["authors"], "year": m["year"], "impact": m["impact_factor"]}
                    if (m := kg.get_metadata(meta, p)) else {"title": None, "authors": None, "year": None, "impact": None})}
                for p in sorted(r["papers"], key=lambda p: (-r["papers"][p], p))
            ],
        })
    with open(os.path.join(out_dir, "modules_full.json"), "w", encoding="utf-8") as f:
        json.dump(full, f, indent=2, ensure_ascii=False)

    conflict_note = {c: f"spellings in modules {ranks}" for c, ranks in conflicts}
    with open(os.path.join(out_dir, "module_assignment_canonical.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["canonical_node", "module_rank", "note"])
        for node in sorted(assignment, key=lambda n: (assignment[n], n)):
            w.writerow([_safe(node), assignment[node], conflict_note.get(node, "")])

    print(f"\nSaved outputs to {os.path.relpath(out_dir, kg.ROOT)}/")
    print(f"  module_summary.csv              — {len(records)} modules")
    print(f"  module_nodes.csv                — all nodes with per-module degree")
    print(f"  module_papers.csv               — all paper-module pairs with metadata")
    print(f"  modules_full.json               — complete structured data")
    print(f"  module_assignment_canonical.csv — module of each node of the 508-node graph "
          f"({len(conflicts)} nodes with spellings in several modules assigned to the largest)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixed-paper-matching", action="store_true",
                    help="use corrected paper-identifier matching (keeps all 760 triplets; "
                         "Module 1 becomes 123 nodes / 122 edges)")
    ap.add_argument("--out", default=OUTPUT_DIR, help="output directory")
    args = ap.parse_args()

    triplets, meta = kg.load_all()
    print(f"Loaded {len(triplets)} triplets")

    G, node_papers, info = kg.build_module_graph(triplets, meta, legacy_paper_matching=not args.fixed_paper_matching)
    if info["dropped_triplets"]:
        print(f"Note: {len(info['dropped_triplets'])} triplets omitted from the module graph "
              f"(paper identifier not matched — see kg_graph.build_module_graph):")
        for ot in info["dropped_triplets"]:
            print(f"    {ot.get('Subject')} — {ot.get('Predicate')} -> {ot.get('Object')}")

    G_lcc = kg.largest_component(G)
    modules = kg.louvain_modules(G_lcc)
    records = module_records(modules, G_lcc, node_papers)
    print_summary(G, G_lcc, modules, records, meta)

    # Map the partition onto the 508-node canonical graph (for Figure 2)
    Gc, cinfo = kg.build_canonical_graph(triplets, meta)
    assignment, conflicts = kg.module_partition_on_canonical(modules, cinfo["names"])
    save_outputs(records, G, G_lcc, node_papers, meta, assignment, conflicts, args.out)


if __name__ == "__main__":
    main()
