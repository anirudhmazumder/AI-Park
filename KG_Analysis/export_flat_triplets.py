#!/usr/bin/env python3
"""
export_flat_triplets.py - the 760 curated triplets as one flat CSV.

    python3 KG_Analysis/export_flat_triplets.py          # -> Final_KG/triplets_flat.csv

Columns: set, subject, predicate, object, verb_category, sentence, paper,
subject_canonical, object_canonical (display names used as graph nodes),
subject_uniprot, object_uniprot (accession from the post-processing stage when
the subject/object string was mapped to a UniProt entry).
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kg_graph as kg  # noqa: E402

OUT = os.path.join(kg.ROOT, "Final_KG", "triplets_flat.csv")
_CTRL = re.compile(r"[\x00-\x1f\x7f]")


def clean(s):
    return _CTRL.sub(" ", s or "").strip()


def accession(t, entity):
    pm = t.get("protein_mappings") or {}
    hit = pm.get(entity)
    return hit.get("accession", "") if isinstance(hit, dict) else ""


def main():
    meta = kg.load_metadata()
    all_triplets = kg.load_triplets()
    _, info = kg.build_canonical_graph(all_triplets, meta)
    names = info["names"]

    rows = []
    for fp in sorted(glob.glob(os.path.join(kg.TRIPLETS_DIR, "set_*_filtered_triplets*.json"))):
        set_no = re.search(r"set_(\d+)", os.path.basename(fp)).group(1)
        with open(fp, encoding="utf-8") as f:
            for t in json.load(f):
                ot = t["original_triplet"]
                rows.append({
                    "set": set_no,
                    "subject": clean(ot.get("Subject")), "predicate": clean(ot.get("Predicate")),
                    "object": clean(ot.get("Object")), "verb_category": clean(ot.get("Verb Category")),
                    "sentence": clean(ot.get("Sentence")), "paper": clean(kg.paper_name(ot)),
                    "subject_canonical": clean(names.resolve(ot.get("Subject", ""))),
                    "object_canonical": clean(names.resolve(ot.get("Object", ""))),
                    "subject_uniprot": accession(t, ot.get("Subject", "")),
                    "object_uniprot": accession(t, ot.get("Object", "")),
                })
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} triplets -> {OUT}")


if __name__ == "__main__":
    main()
