#!/usr/bin/env python3
"""
graph_prep.py — offline graph evaluation/processing before canon-building.

No API calls. Operates on graph.json (the {"nodes":[...], "analysis":{...}}
output of pass2_graph.py) plus the chapter outline. Subcommands:

  chapters   Per-chapter decision table: size, internal vs crossing edges (the
             cut), internal/crossing ratio, top hub by *within-chapter* in_degree,
             and internal cycle count. This is the pilot-selection driver.

  dedup      Propose near-duplicate claim groups (TF-IDF cosine >= threshold).
             Writes a review file; does NOT merge. You confirm, then run:
  apply      Merge confirmed groups (union depends_on, keep ALL source/provenance),
             rewrite edges, write merged graph. Re-run pass2_graph --keep-existing
             afterward to recompute metrics + cycles.

  sample     Draw N random edges + N random claims into a review file with blank
             label fields. You hand-label y/n. Then:
  score      Read the labeled file and print edge-precision + claim-accuracy.

Usage:
  python graph_prep.py chapters --graph graph.json --outline chapters/<bin>.outline.json
  python graph_prep.py dedup    --graph graph.json --threshold 0.8 --out merges_review.json
  python graph_prep.py apply    --graph graph.json --merges merges_review.json --out graph.merged.json
  python graph_prep.py sample   --graph graph.json --n 30 --out sample_review.json
  python graph_prep.py score    --sample sample_review.json
"""

import argparse
import json
import random
import sys
from collections import defaultdict

import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_graph(path):
    obj = json.load(open(path))
    return obj["nodes"] if isinstance(obj, dict) and "nodes" in obj else obj


def qindex(node):
    """Parse the question index out of a source like 'viral_vector:q65'."""
    src = node.get("source", "")
    if ":q" in src:
        try:
            return int(src.split(":q")[1])
        except ValueError:
            return None
    return None


def chapter_map(outline_path):
    outl = json.load(open(outline_path))
    chapters = outl.get("chapters", outl) if isinstance(outl, dict) else outl
    q2ch = {}
    for ch in chapters:
        for idx in ch.get("member_indices", []):
            q2ch[idx] = ch["slug"]
    return q2ch


# ---------------- chapters (decision driver) ----------------

def cmd_chapters(args):
    nodes = load_graph(args.graph)
    q2ch = chapter_map(args.outline)
    ch_of = {n["id"]: q2ch.get(qindex(n)) for n in nodes}
    ids = set(ch_of)

    edges = [(n["id"], d) for n in nodes for d in n.get("depends_on", []) if d in ids]

    chs = defaultdict(lambda: {"nodes": [], "internal": 0, "crossing": 0})
    for nid, ch in ch_of.items():
        if ch:
            chs[ch]["nodes"].append(nid)

    internal_indeg = defaultdict(int)  # within-chapter dependents of a node
    for a, b in edges:
        ca, cb = ch_of.get(a), ch_of.get(b)
        if ca and ca == cb:
            chs[ca]["internal"] += 1
            internal_indeg[b] += 1
        elif ca:
            chs[ca]["crossing"] += 1

    rows = []
    for ch, d in chs.items():
        sub = nx.DiGraph()
        sub.add_nodes_from(d["nodes"])
        sub.add_edges_from((a, b) for a, b in edges if ch_of.get(a) == ch == ch_of.get(b))
        cycles = sum(1 for c in nx.strongly_connected_components(sub) if len(c) > 1)
        hub = max(d["nodes"], key=lambda x: internal_indeg.get(x, 0), default=None)
        hub_in = internal_indeg.get(hub, 0) if hub else 0
        ratio = d["internal"] / d["crossing"] if d["crossing"] else float("inf")
        rows.append((ch, len(d["nodes"]), d["internal"], d["crossing"], ratio, hub, hub_in, cycles))

    # sort by hub strength then modularity — best pilot corners on top
    rows.sort(key=lambda r: (r[6], r[4] if r[4] != float("inf") else 1e9), reverse=True)

    print(f"{'chapter':<34}{'n':>4}{'int':>5}{'cross':>6}{'ratio':>7}{'cyc':>5}  hub(in_deg)")
    for ch, n, intl, cross, ratio, hub, hub_in, cyc in rows:
        r = "inf" if ratio == float("inf") else f"{ratio:.2f}"
        print(f"{ch:<34}{n:>4}{intl:>5}{cross:>6}{r:>7}{cyc:>5}  {hub} ({hub_in})")
    print("\nGood pilot corner = mechanism-dense + high hub in_degree + high internal/crossing ratio.")


# ---------------- dedup ----------------

def cmd_dedup(args):
    nodes = load_graph(args.graph)
    claims = [n["claim"] for n in nodes]
    X = TfidfVectorizer(stop_words="english").fit_transform(claims)
    S = cosine_similarity(X)

    # union-find over pairs above threshold
    parent = {n["id"]: n["id"] for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    ids = [n["id"] for n in nodes]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if S[i, j] >= args.threshold:
                union(ids[i], ids[j])

    groups = defaultdict(list)
    for n in nodes:
        groups[find(n["id"])].append(n["id"])
    multi = [g for g in groups.values() if len(g) > 1]

    by_id = {n["id"]: n["claim"] for n in nodes}
    review = [{"keep": g[0], "merge": g[1:],
               "claims": {cid: by_id[cid] for cid in g}} for g in multi]
    json.dump(review, open(args.out, "w"), indent=2)
    print(f"{len(multi)} candidate merge groups (threshold {args.threshold}) -> {args.out}")
    print("Review/edit it (drop false positives), then: graph_prep.py apply --merges " + args.out)


def cmd_apply(args):
    nodes = load_graph(args.graph)
    merges = json.load(open(args.merges))
    by_id = {n["id"]: n for n in nodes}
    remap = {}  # dropped id -> kept id
    for grp in merges:
        keep = grp["keep"]
        for drop in grp["merge"]:
            remap[drop] = keep
            k, d = by_id[keep], by_id.get(drop)
            if not d:
                continue
            k["depends_on"] = list(dict.fromkeys(k.get("depends_on", []) + d.get("depends_on", [])))
            # keep ALL provenance: one fact graded by several questions
            k.setdefault("merged_sources", [k.get("source")])
            k["merged_sources"].append(d.get("source"))
            k.setdefault("merged_provenance", [k.get("provenance")])
            k["merged_provenance"].append(d.get("provenance"))

    kept = [n for n in nodes if n["id"] not in remap]
    for n in kept:  # rewrite edges through the remap, drop self/dupes
        deps = [remap.get(d, d) for d in n.get("depends_on", [])]
        n["depends_on"] = list(dict.fromkeys(d for d in deps if d != n["id"]))

    json.dump({"nodes": kept}, open(args.out, "w"), indent=2)
    print(f"Merged {len(nodes) - len(kept)} nodes; {len(kept)} remain -> {args.out}")
    print("Now recompute metrics: python pass2_graph.py --claims " + args.out + " --keep-existing --out graph.json")


# ---------------- calibration sample ----------------

def cmd_sample(args):
    nodes = load_graph(args.graph)
    rng = random.Random(args.seed)
    by_id = {n["id"]: n for n in nodes}

    all_edges = [(n["id"], d, n["claim"], by_id.get(d, {}).get("claim", "?"))
                 for n in nodes for d in n.get("depends_on", [])]
    edge_sample = rng.sample(all_edges, min(args.n, len(all_edges)))
    claim_sample = rng.sample(nodes, min(args.n, len(nodes)))

    review = {
        "edges": [{"from": a, "to": b, "from_claim": fc, "to_claim": tc,
                   "is_real_prerequisite": ""} for a, b, fc, tc in edge_sample],
        "claims": [{"id": n["id"], "claim": n["claim"],
                    "question": n.get("provenance", {}).get("question", ""),
                    "faithful": ""} for n in claim_sample],
    }
    json.dump(review, open(args.out, "w"), indent=2)
    print(f"Wrote {len(review['edges'])} edges + {len(review['claims'])} claims to {args.out}.")
    print("Fill is_real_prerequisite / faithful with y or n (your judgment, not a model), then: score.")


def cmd_score(args):
    rev = json.load(open(args.sample))

    def prec(items, key):
        labeled = [x[key].strip().lower() for x in items if x[key].strip()]
        yes = sum(1 for v in labeled if v in ("y", "yes", "1", "true"))
        return (yes, len(labeled))

    ey, en = prec(rev["edges"], "is_real_prerequisite")
    cy, cn = prec(rev["claims"], "faithful")
    print(f"Edge precision:    {ey}/{en} = {ey / en:.0%}" if en else "No edge labels.")
    print(f"Claim faithfulness: {cy}/{cn} = {cy / cn:.0%}" if cn else "No claim labels.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("chapters"); p.add_argument("--graph", required=True); p.add_argument("--outline", required=True); p.set_defaults(fn=cmd_chapters)
    p = sub.add_parser("dedup"); p.add_argument("--graph", required=True); p.add_argument("--threshold", type=float, default=0.8); p.add_argument("--out", default="merges_review.json"); p.set_defaults(fn=cmd_dedup)
    p = sub.add_parser("apply"); p.add_argument("--graph", required=True); p.add_argument("--merges", required=True); p.add_argument("--out", default="graph.merged.json"); p.set_defaults(fn=cmd_apply)
    p = sub.add_parser("sample"); p.add_argument("--graph", required=True); p.add_argument("--n", type=int, default=30); p.add_argument("--seed", type=int, default=0); p.add_argument("--out", default="sample_review.json"); p.set_defaults(fn=cmd_sample)
    p = sub.add_parser("score"); p.add_argument("--sample", required=True); p.set_defaults(fn=cmd_score)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()