"""
kg_graph.py - builds the HDAC6-AD knowledge graph from the data in Final_KG/.

Every analysis and visualisation script in this repository imports this module,
so all of them operate on the same graph.

Two graph variants are defined here (see README, "Graph construction and
analysis"):

* ``build_canonical_graph``  - the 508-node / 511-edge knowledge graph used for
  every whole-graph statistic, hub, centrality, connectivity, island and
  subgraph figure in the manuscript.  Entity strings are merged with the
  *canonical* normalisation (NFKD -> lowercase -> strip non-alphanumerics ->
  collapse whitespace, plus three synonym rules).

* ``build_module_graph``     - the 526-node / 526-edge graph on which the
  Louvain community detection (27 modules; Module 1 = 122/121, Module 2 =
  37/39, Module 3 = 33/34) was run.  Entity strings are merged with a *lighter*
  normalisation (NFKC -> lowercase -> dash folding -> collapse whitespace) and the
  paper-identifier matching replicates the original analysis script exactly
  (``legacy_paper_matching=True``), which omits seven triplets - see the
  docstring of ``build_module_graph``.

Both builders are deterministic.  Louvain is run with ``seed=42`` and the
default ``resolution=1.0``; the Fruchterman-Reingold layout with ``seed=42``.
Layout coordinates are *not* portable across NumPy/SciPy versions, so the
layout used for the manuscript figures is frozen in ``Final_KG/node_positions.json``
and loaded by every plotting script (``load_positions``).
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
import unicodedata
from collections import defaultdict

import networkx as nx
from networkx.algorithms import community as nx_community

# Repository paths (absolute, so scripts can be run from any directory)
ROOT           = os.path.dirname(os.path.abspath(__file__))
TRIPLETS_DIR   = os.path.join(ROOT, "Final_KG")
METADATA_CSV   = os.path.join(ROOT, "Final_KG", "metadata.csv")
POSITIONS_JSON = os.path.join(ROOT, "Final_KG", "node_positions.json")

# Analysis constants
EXCLUDED_NODES     = frozenset({"microtubules"})  # normalised names dropped from the graph
LOUVAIN_SEED       = 42
LOUVAIN_RESOLUTION = 1.0
LOUVAIN_THRESHOLD  = 1e-07     # networkx default, stated for completeness
LAYOUT_SEED        = 42
LAYOUT_K           = 0.5
LAYOUT_ITERATIONS  = 100
LAYOUT_SCALE       = 2

# Triplet JSON keys that may hold the source-paper identifier.
PAPER_KEYS        = ("Paper", "Paper Name", "Paper name",
                     "The name of the paper the triplet is from")
LEGACY_PAPER_KEYS = ("Paper", "Paper Name", "Paper name")

# OCR-mangled paper identifiers that occur in the triplet files, mapped to the
# FileName used in metadata.csv.  (The control characters are written as
# escapes on purpose: "\x10" and "\n" stand where the original PDF title had a
# Unicode hyphen "‐", and one OCR file name misspells "deacetylase".)
_SATOH = "Neuropathology Appl Neurobio - 2013 - Satoh - Ubiquilin‐1 immunoreactivity is concentrated on Hirano bodies and dystrophic"
_CHENG = "The FASEB Journal - 2021 - Cheng - The double‐edged sword effect of HDAC6 in A  toxicities"
_PEREZ = "Journal of Neurochemistry - 2009 - Perez - Tau   an inhibitor of deacetylase HDAC6 function"
FILENAME_MAPPINGS = {
    "Neuropathology Appl Neurobio - 2013 - Satoh - Ubiquilin\x101 immunoreactivity is concentrated on Hirano bodies and dystrophic_ocr_output": _SATOH,
    "The FASEB Journal - 2021 - Cheng - The double\x10edged sword effect of HDAC6 in A  toxicities_ocr_output": _CHENG,
    "The FASEB Journal - 2021 - Cheng - The double\nedged sword effect of HDAC6 in A  toxicities_ocr_output": _CHENG,
    "The FASEB Journal - 2021 - Cheng - The double\ndged sword effect of HDAC6 in A  toxicities_ocr_output": _CHENG,
    "Journal of Neurochemistry - 2009 - Perez - Tau   an inhibitor of deacetase HDAC6 function_ocr_output": _PEREZ,
}


# Loading

def load_triplets(triplets_dir: str = TRIPLETS_DIR) -> list:
    """Load the curated triplets (``set_*_filtered_triplets*.json``), in file order."""
    files = sorted(glob.glob(os.path.join(triplets_dir, "set_*_filtered_triplets*.json")))
    if not files:
        raise FileNotFoundError(f"No triplet files found in {triplets_dir}")
    triplets = []
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            triplets.extend(json.load(f))
    return triplets


def paper_name(original_triplet: dict, keys=PAPER_KEYS) -> str:
    """Return the source-paper identifier stored in a triplet (first non-empty key)."""
    for k in keys:
        v = original_triplet.get(k)
        if v:
            return v
    return ""


def _parse_year(date_string) -> int | None:
    if not date_string:
        return None
    m = re.search(r"\b(19|20)\d{2}\b", str(date_string))
    return int(m.group()) if m else None


def _parse_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_metadata(metadata_csv: str = METADATA_CSV, filename_mappings: dict | None = FILENAME_MAPPINGS) -> dict:
    """
    Load ``metadata.csv`` into a lookup keyed by every identifier variant a
    triplet may use (``FileName``, ``FileName`` without ``_ocr_output``, and the
    OCR-mangled spellings in ``FILENAME_MAPPINGS``).

    Pass ``filename_mappings=None`` to reproduce the original module-analysis
    script, which did not know about the mangled spellings.
    """
    meta = {}
    with open(metadata_csv, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            filename = (row.get("FileName") or "").strip()
            if not filename:
                continue
            impact = _parse_float(row.get("ImpactFactor"))
            entry = {
                "filename":         filename,
                "title":            row.get("Title") or "Unknown",
                "authors":          row.get("Authors") or "Unknown",
                "publication_date": row.get("Publication_Date") or "Unknown",
                "year":             _parse_year(row.get("Publication_Date")),
                "link":             row.get("Link") or "No link",
                "impact_factor":    impact,                      # float or None
                "impact_factor_num": impact if impact is not None else 0.0,
            }
            meta[filename] = entry
            base = filename.replace("_ocr_output", "")
            if base != filename:
                meta[base] = entry
            if filename_mappings:
                for triplet_name, metadata_name in filename_mappings.items():
                    if filename in (metadata_name, metadata_name + "_ocr_output"):
                        meta[triplet_name] = entry
                        meta[triplet_name.replace("_ocr_output", "")] = entry
    return meta


def get_metadata(meta: dict, paper: str) -> dict | None:
    """Resolve a triplet's paper identifier to its metadata entry (or ``None``)."""
    if not paper:
        return None
    paper = paper.replace("\x10", "‐").replace("\n", "‐e")
    squeezed = " ".join(paper.split())
    for variant in (
        paper,
        paper + "_ocr_output",
        paper.replace("_ocr_output", ""),
        squeezed,
        squeezed + "_ocr_output",
        squeezed.replace("_ocr_output", ""),
    ):
        if variant in meta:
            return meta[variant]
    lower = paper.lower()
    for key in meta:
        if key.lower() in (lower, lower + "_ocr_output", lower.replace("_ocr_output", "")):
            return meta[key]
    return None


def unique_papers(meta: dict) -> list:
    """Distinct metadata entries (one per paper in the corpus)."""
    seen, out = set(), []
    for entry in meta.values():
        if entry["filename"] not in seen:
            seen.add(entry["filename"])
            out.append(entry)
    return out


# Entity-name normalisation

def normalize_canonical(name: str) -> str:
    """
    Canonical normalisation used for the 508-node knowledge graph:
    NFKD -> lowercase -> non-alphanumerics to spaces -> collapse whitespace, then
    three synonym rules ("class i " prefix removed, "tau protein" -> "tau",
    any ordering of {amyloid, beta} -> "amyloid beta").
    """
    if not name:
        return ""
    n = unicodedata.normalize("NFKD", name).lower()
    n = re.sub(r"[^a-z0-9\s]", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    if n.startswith("class i "):
        n = n[8:]
    if n == "tau protein":
        n = "tau"
    if set(n.split()) == {"amyloid", "beta"}:
        n = "amyloid beta"
    return n


def normalize_module(name: str) -> str:
    """
    Lighter normalisation used for the community-detection (module) graph:
    NFKC -> strip -> lowercase -> unify dash characters -> collapse whitespace.
    Punctuation is kept and no synonym rules are applied, so e.g. "Tau" and
    "tau protein" remain distinct nodes.
    """
    n = unicodedata.normalize("NFKC", name or "").strip().lower()
    n = re.sub(r"[‐‑‒–—―−﹘﹣－]", "-", n)
    n = re.sub(r"\s+", " ", n)
    return n


class CanonicalNames:
    """
    Chooses one display name per normalised entity.  Among the raw spellings
    that share a normalised form, the one with the fewest special characters
    wins; ties go to the shorter string, then to the all-lowercase string, then
    to the first one seen.
    """

    def __init__(self):
        self.mapping: dict[str, str] = {}

    def get(self, normalized: str, raw: str) -> str:
        if normalized not in self.mapping:
            self.mapping[normalized] = raw
            return raw
        existing = self.mapping[normalized]
        sc = len(re.findall(r"[^a-zA-Z0-9\s]", raw))
        se = len(re.findall(r"[^a-zA-Z0-9\s]", existing))
        if sc < se:
            self.mapping[normalized] = raw
            return raw
        if sc > se:
            return existing
        if len(raw) < len(existing):
            self.mapping[normalized] = raw
            return raw
        if len(raw) > len(existing):
            return existing
        if raw.islower() and not existing.islower():
            self.mapping[normalized] = raw
            return raw
        return existing

    def lookup(self, normalized: str) -> str | None:
        return self.mapping.get(normalized)

    def resolve(self, raw_or_display: str) -> str | None:
        """Display name for any spelling of an entity (``None`` if unknown)."""
        return self.mapping.get(normalize_canonical(raw_or_display))


# Graph builders

def _sorted_triplets(triplets):
    return sorted(
        triplets,
        key=lambda t: (t["original_triplet"].get("Subject", ""),
                       t["original_triplet"].get("Object", "")),
    )


def build_canonical_graph(triplets, meta, directed: bool = False,
                          exclude_nodes=EXCLUDED_NODES,
                          exclude_missing_metadata: bool = True):
    """
    Build the knowledge graph used for all whole-graph statistics.

    Returns ``(G, info)`` where ``info`` holds the construction statistics
    (nodes before/after normalisation, excluded triplets) and the
    ``CanonicalNames`` object (``info["names"]``).

    * Triplets are processed in a deterministic order (sorted by subject, object).
    * Pass 1 assigns the display name of every entity over *all* triplets.
    * Pass 2 adds one edge per distinct (subject, object) pair, skipping
      triplets whose paper has no metadata and triplets touching an excluded
      node.  Repeated pairs increment ``count`` and accumulate papers,
      predicates and sentences; ``verb_category`` keeps the first value seen;
      ``max_impact_factor`` is the largest journal impact factor among the
      papers supporting the edge.
    * ``directed=True`` returns a ``DiGraph`` with subject -> object edges
      (used only for the in-/out-degree statistics).
    """
    names = CanonicalNames()
    G = nx.DiGraph() if directed else nx.Graph()
    sorted_triplets = _sorted_triplets(triplets)

    nodes_before = set()
    for t in sorted_triplets:
        s_raw = t["original_triplet"].get("Subject", "")
        o_raw = t["original_triplet"].get("Object", "")
        nodes_before.add(s_raw)
        nodes_before.add(o_raw)
        names.get(normalize_canonical(s_raw), s_raw)
        names.get(normalize_canonical(o_raw), o_raw)

    node_meta = defaultdict(lambda: {"papers": set(), "connections": []})
    excluded_meta = excluded_node = 0

    for t in sorted_triplets:
        ot        = t["original_triplet"]
        s_raw     = ot.get("Subject", "")
        o_raw     = ot.get("Object", "")
        predicate = ot.get("Predicate", "")
        verb_cat  = ot.get("Verb Category", None)
        sentence  = ot.get("Sentence", "")
        paper     = paper_name(ot)

        entry = get_metadata(meta, paper) if paper else None
        if exclude_missing_metadata and paper and entry is None:
            excluded_meta += 1
            continue

        s_norm = normalize_canonical(s_raw)
        o_norm = normalize_canonical(o_raw)
        if exclude_nodes and (s_norm in exclude_nodes or o_norm in exclude_nodes):
            excluded_node += 1
            continue

        subject = names.get(s_norm, s_raw)
        obj     = names.get(o_norm, o_raw)
        if not subject or not obj:
            continue

        impact = entry["impact_factor_num"] if entry else 0.0
        year   = entry["year"] if entry else None

        if G.has_edge(subject, obj):
            d = G[subject][obj]
            d["count"] += 1
            d["papers"].add(paper)
            d["predicates"].append(predicate)
            d["sentences"].append(sentence)
            d["max_impact_factor"] = max(d["max_impact_factor"], impact)
            if year:
                d["years"].add(year)
        else:
            G.add_edge(subject, obj,
                       count=1, papers={paper}, predicates=[predicate],
                       sentences=[sentence], sentence=sentence,
                       verb_category=verb_cat,
                       max_impact_factor=impact,
                       years={year} if year else set())

        for node, other in ((subject, obj), (obj, subject)):
            node_meta[node]["papers"].add(paper)
            node_meta[node]["connections"].append({
                "target": other, "predicate": predicate,
                "paper": paper, "sentence": sentence,
            })

    for node in G.nodes():
        G.nodes[node]["papers"]      = node_meta[node]["papers"]
        G.nodes[node]["connections"] = node_meta[node]["connections"]

    info = {
        "names": names,
        "nodes_before_normalisation": len(nodes_before),
        "nodes_after_normalisation": G.number_of_nodes(),
        "nodes_merged": len(nodes_before) - G.number_of_nodes(),
        "excluded_missing_metadata": excluded_meta,
        "excluded_node": excluded_node,
        "triplets": len(triplets),
    }
    return G, info


def build_module_graph(triplets, meta=None, legacy_paper_matching: bool = True,
                       exclude_nodes=EXCLUDED_NODES):
    """
    Build the graph on which the manuscript's community detection was run.

    * Entities merged with ``normalize_module`` (lighter than the canonical
      normalisation); the display name is the first spelling encountered.
    * Triplets processed in file order; self-loops skipped; triplets touching
      an excluded node skipped; triplets whose paper cannot be matched to
      metadata skipped.

    ``legacy_paper_matching=True`` (default) reproduces the paper-identifier
    matching of the original ``module_analysis.py`` exactly: it ignores the
    ``FILENAME_MAPPINGS`` spellings and the long JSON key
    "The name of the paper the triplet is from".  As a consequence seven
    triplets are omitted from this graph (two from Perez et al. 2009, whose OCR
    file name is misspelled, and five whose paper identifier is stored under
    the long key).  This is the graph behind the module figures reported in the
    manuscript (526 nodes / 526 edges; Module 1 = 122 nodes / 121 edges).

    ``legacy_paper_matching=False`` uses the same matching as the canonical
    graph, keeping all 760 triplets (527 nodes / 527 edges; Module 1 becomes
    123 / 122, Modules 2 and 3 are unchanged).

    Returns ``(G, node_papers, info)``.
    """
    if legacy_paper_matching:
        meta_used = load_metadata(filename_mappings=None)
        keys = LEGACY_PAPER_KEYS
    else:
        meta_used = meta if meta is not None else load_metadata()
        keys = PAPER_KEYS

    name_map: dict[str, str] = {}

    def display(raw: str) -> str:
        norm = normalize_module(raw)
        if norm not in name_map:
            name_map[norm] = raw.strip()
        return name_map[norm]

    G = nx.Graph()
    node_papers = defaultdict(set)
    skipped = {"self_loop_or_empty": 0, "excluded_node": 0, "missing_metadata": 0}
    dropped_triplets = []

    for t in triplets:
        ot    = t["original_triplet"]
        s_raw = ot.get("Subject", "")
        o_raw = ot.get("Object", "")
        paper = paper_name(ot, keys)

        subj = display(s_raw)
        obj  = display(o_raw)
        if not subj or not obj or subj == obj:
            skipped["self_loop_or_empty"] += 1
            continue
        if exclude_nodes and (normalize_canonical(s_raw) in exclude_nodes
                              or normalize_canonical(o_raw) in exclude_nodes):
            skipped["excluded_node"] += 1
            continue
        if get_metadata(meta_used, paper) is None:
            skipped["missing_metadata"] += 1
            dropped_triplets.append(ot)
            continue

        G.add_edge(subj, obj)
        node_papers[subj].add(paper)
        node_papers[obj].add(paper)

    info = {"skipped": skipped, "dropped_triplets": dropped_triplets,
            "legacy_paper_matching": legacy_paper_matching, "name_map": name_map}
    return G, node_papers, info


# Analysis helpers

def induced_subgraph(G: nx.Graph, nodes) -> nx.Graph:
    """
    Copy of the subgraph of ``G`` induced by ``nodes``, with nodes and edges in
    the order they have in ``G``.  (``G.subgraph(nodes)`` may iterate over the
    node *set* instead, whose order changes between Python runs; the drawing
    order of overlapping labels in the figures would then not be reproducible.)
    """
    keep = set(nodes)
    H = G.__class__()
    H.graph.update(G.graph)
    H.add_nodes_from((n, d.copy()) for n, d in G.nodes(data=True) if n in keep)
    H.add_edges_from((u, v, d.copy()) for u, v, d in G.edges(data=True) if u in keep and v in keep)
    return H


def edge_induced_subgraph(G: nx.Graph, edges) -> nx.Graph:
    """Copy of the subgraph of ``G`` formed by ``edges`` (and their end nodes), in the order of ``G``."""
    key = (lambda u, v: (u, v)) if G.is_directed() else (lambda u, v: frozenset((u, v)))
    keep = {key(u, v) for u, v in edges}
    touched = {n for u, v in edges for n in (u, v)}
    H = G.__class__()
    H.graph.update(G.graph)
    H.add_nodes_from((n, d.copy()) for n, d in G.nodes(data=True) if n in touched)
    H.add_edges_from((u, v, d.copy()) for u, v, d in G.edges(data=True) if key(u, v) in keep)
    return H


def largest_component(G: nx.Graph) -> nx.Graph:
    """Copy of the largest connected component of ``G`` (node order as in ``G``)."""
    return induced_subgraph(G, max(nx.connected_components(G), key=len))


def louvain_modules(G_lcc: nx.Graph, seed: int = LOUVAIN_SEED,
                    resolution: float = LOUVAIN_RESOLUTION) -> list[set]:
    """Louvain communities of ``G_lcc`` (unweighted), largest first."""
    comms = nx_community.louvain_communities(G_lcc, seed=seed, resolution=resolution,
                                             threshold=LOUVAIN_THRESHOLD)
    return sorted(comms, key=len, reverse=True)


def module_partition_on_canonical(modules: list[set], names: CanonicalNames):
    """
    Transfer a module partition computed on the module graph onto the canonical
    (508-node) graph, for colouring the whole-graph figure.

    Every module-graph node is mapped to its canonical display name.  When
    several module-graph spellings collapse onto one canonical node and they
    sit in different modules, the node is assigned to the largest of those
    modules (lowest rank), so that membership in one of the three reported
    modules is never lost by the merge.

    Returns ``(assignment, conflicts)`` where ``assignment`` maps canonical
    display name -> 1-based module rank and ``conflicts`` lists the canonical
    nodes whose spellings disagreed (with the ranks involved).
    """
    candidates = defaultdict(set)   # canonical node -> {ranks}
    for rank, mod in enumerate(modules, start=1):
        for node in mod:
            canon = names.resolve(node)
            if canon is not None:
                candidates[canon].add(rank)

    assignment, conflicts = {}, []
    for canon, ranks in candidates.items():
        if len(ranks) > 1:
            conflicts.append((canon, sorted(ranks)))
        assignment[canon] = min(ranks)
    return assignment, conflicts


# Layout

def load_positions(path: str = POSITIONS_JSON) -> dict | None:
    """Frozen layout used for the manuscript figures (``None`` if absent)."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {n: (float(p[0]), float(p[1])) for n, p in raw.items()}


def spring_positions(G: nx.Graph, seed: int = LAYOUT_SEED) -> dict:
    """Fruchterman-Reingold layout with the manuscript's parameters."""
    pos = nx.spring_layout(G, k=LAYOUT_K, iterations=LAYOUT_ITERATIONS,
                           seed=seed, scale=LAYOUT_SCALE)
    return {n: (float(p[0]), float(p[1])) for n, p in pos.items()}


def positions_for(G: nx.Graph, positions: dict | None, seed: int = LAYOUT_SEED) -> dict:
    """
    Coordinates for every node of ``G``: frozen positions where available,
    otherwise a spring layout of the missing nodes (``seed`` fixed).  Falls
    back to a full layout when no frozen positions exist.
    """
    if not positions:
        return spring_positions(G, seed=seed)
    pos = {n: positions[n] for n in G.nodes() if n in positions}
    missing = [n for n in G.nodes() if n not in pos]
    if missing:
        extra = nx.spring_layout(G.subgraph(missing), k=LAYOUT_K, iterations=LAYOUT_ITERATIONS,
                                 seed=seed, scale=LAYOUT_SCALE)
        pos.update({n: (float(p[0]), float(p[1])) for n, p in extra.items()})
    return pos


def save_positions(positions: dict, path: str = POSITIONS_JSON) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({n: [float(x), float(y)] for n, (x, y) in positions.items()}, f, indent=2)


# Convenience

def load_all():
    """``(triplets, meta)`` from the repository data."""
    return load_triplets(), load_metadata()


def canonical_graph():
    """Build the 508-node graph from the repository data; returns ``(G, info, meta)``."""
    triplets, meta = load_all()
    G, info = build_canonical_graph(triplets, meta)
    return G, info, meta
