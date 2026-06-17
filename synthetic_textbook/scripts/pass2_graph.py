#!/usr/bin/env python3
"""
pass2_graph.py — Pass 2 of the claim-graph builder: link claims + analyze.

Reads the nodes produced by Pass 1 (questions_to_statements.py) and:

  1. EDGE LINKING (closed-world): for each claim, asks an LLM which *other
     existing* claims are direct prerequisites. It only ever returns IDs from
     the provided set — no new facts are invented — so the graph stays closed
     over your curated nodes. Results land in each node's `depends_on`.

  2. ANALYSIS: builds a directed graph (edge node -> prerequisite, i.e.
     "depends on") and computes:
       - strongly-connected components  (cycle detection; cycles are KEPT and
         reported, not removed — an SCC of size > 1 is a dependency cycle)
       - per-node centrality: in/out degree, degree centrality, betweenness,
         PageRank (PageRank is cycle-robust; foundational claims score high).

Works on any statements.json from Pass 1 regardless of the source format
(the cooking-style open-Q&A set included).

Usage:
  python pass2_graph.py --claims statements.json --out graph.json
  python pass2_graph.py --claims statements.json --dry-run     # analyze only, no API
  python pass2_graph.py --claims statements.json --max-candidates 60   # cap context
"""

import argparse
import json
import sys

import networkx as nx

# Reuse Pass 1's LLM harness + env-file loading (no module-level side effects).
from questions_to_statements import call_llm, load_env_file, save_nodes

LINK_SYS = (
    "You are building a prerequisite graph over a FIXED set of factual claims. "
    "Given one TARGET claim and a numbered list of OTHER claims (each with an id), "
    "identify which of the OTHER claims are DIRECT prerequisites of the target — "
    "facts a reader must already understand for the target to make sense. "
    "Choose ONLY from the provided ids; never invent claims or ids. Be conservative: "
    "only direct, necessary prerequisites, not loosely related facts. "
    'Return strict JSON: {"prerequisites": ["<id>", ...]}.'
)


def _tokens(text):
    return {w for w in "".join(c.lower() if c.isalnum() else " " for c in text).split() if len(w) > 2}


def candidate_pool(target, nodes, max_candidates):
    """All other nodes, optionally pruned to the most lexically-related ones so
    large claim sets don't blow the context window (closed-world, no embeddings)."""
    others = [n for n in nodes if n["id"] != target["id"]]
    if not max_candidates or len(others) <= max_candidates:
        return others
    tgt = _tokens(target["claim"])
    others.sort(key=lambda n: len(tgt & _tokens(n["claim"])), reverse=True)
    return others[:max_candidates]


def link_prerequisites(target, nodes, model, max_candidates):
    pool = candidate_pool(target, nodes, max_candidates)
    n_others = len(nodes) - 1
    if n_others > len(pool):  # surface lexical-prefilter pruning — never silent
        print(
            f"  note: {target['id']} pool pruned {n_others - len(pool)}/{n_others} "
            f"claims (lexical pre-filter; possible recall loss)",
            file=sys.stderr,
        )
    valid_ids = {n["id"] for n in pool}
    listing = "\n".join(f'{n["id"]}: {n["claim"]}' for n in pool)
    user = f'TARGET ({target["id"]}): {target["claim"]}\n\nOTHER CLAIMS:\n{listing}'
    out = call_llm(LINK_SYS, user, model)
    raw = out.get("prerequisites", []) or []
    # keep only real ids, drop self-references and dupes, preserve order
    seen, edges = set(), []
    for pid in raw:
        if pid in valid_ids and pid != target["id"] and pid not in seen:
            seen.add(pid)
            edges.append(pid)
    return edges


def build_graph(nodes):
    g = nx.DiGraph()
    g.add_nodes_from(n["id"] for n in nodes)
    for n in nodes:
        for dep in n.get("depends_on", []):
            if dep in g:  # ignore dangling ids defensively
                g.add_edge(n["id"], dep)  # edge: node -> its prerequisite
    return g


def analyze(nodes):
    """Annotate each node with centrality metrics; return a graph-level summary
    including dependency cycles (SCCs of size > 1)."""
    g = build_graph(nodes)

    sccs = [sorted(c) for c in nx.strongly_connected_components(g) if len(c) > 1]
    deg_c = nx.degree_centrality(g)
    btw = nx.betweenness_centrality(g)
    pr = nx.pagerank(g) if g.number_of_nodes() else {}

    by_id = {}
    for n in nodes:
        nid = n["id"]
        m = {
            "in_degree": g.in_degree(nid),    # how many claims depend ON this one
            "out_degree": g.out_degree(nid),  # how many prerequisites this one has
            "degree_centrality": round(deg_c.get(nid, 0.0), 4),
            "betweenness": round(btw.get(nid, 0.0), 4),
            "pagerank": round(pr.get(nid, 0.0), 4),
        }
        n["metrics"] = m
        by_id[nid] = m

    def top(metric, k=5):
        ranked = sorted(by_id.items(), key=lambda kv: kv[1][metric], reverse=True)
        return [{"id": i, metric: v[metric]} for i, v in ranked[:k] if v[metric]]

    is_dag = nx.is_directed_acyclic_graph(g)
    return {
        "node_count": g.number_of_nodes(),
        "edge_count": g.number_of_edges(),
        "cycle_count": len(sccs),
        "cycles": sccs,  # each is a list of node ids in one dependency cycle
        "is_dag": is_dag,
        # PageRank converges over cycles but an SCC is a rank-trap that inflates
        # cycle members, so the ranking is only trustworthy on a DAG. Until cycles
        # are resolved, prefer top_by_in_degree for root selection.
        "pagerank_reliable": is_dag,
        "top_by_pagerank": top("pagerank"),
        "top_by_betweenness": top("betweenness"),
        "top_by_in_degree": top("in_degree"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", required=True, help="Pass 1 output (statements.json)")
    ap.add_argument("--out", default="graph.json")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--max-candidates", type=int,
                    help="cap prerequisite candidates per claim (lexical pre-filter)")
    ap.add_argument("--limit", type=int, help="only link the first N claims (testing)")
    ap.add_argument("--keep-existing", action="store_true",
                    help="don't call the LLM; analyze the depends_on already present")
    ap.add_argument("--dry-run", action="store_true",
                    help="alias for --keep-existing: analysis only, no API calls")
    ap.add_argument("--relink-all", action="store_true",
                    help="re-infer edges even for reviewed nodes (needs_review=false); "
                         "default skips them so hand-corrected edges survive a re-run")
    args = ap.parse_args()

    with open(args.claims) as f:
        nodes = json.load(f)
    if not nodes:
        sys.exit(f"No claims in {args.claims}.")

    analyze_only = args.keep_existing or args.dry_run
    if not analyze_only:
        loaded = load_env_file()
        if loaded:
            print(f"Loaded env from {loaded}", file=sys.stderr)
        targets = nodes[: args.limit] if args.limit is not None else nodes
        for i, n in enumerate(targets):
            if not args.relink_all and n.get("needs_review") is False:
                print(f"[{i + 1}/{len(targets)}] {n['id']}: skipped (reviewed)")
                continue  # preserve hand-corrected edges; --relink-all to override
            edges = link_prerequisites(n, nodes, args.model, args.max_candidates)
            n["depends_on"] = edges
            n["needs_review"] = True  # inferred edges are candidates until you confirm
            print(f"[{i + 1}/{len(targets)}] {n['id']}: depends_on {edges or '[]'}")

    summary = analyze(nodes)

    # Output is {"nodes": [...], "analysis": {...}} — downstream steps read graph["nodes"].
    save_nodes({"nodes": nodes, "analysis": summary}, args.out)
    print(
        f"\nWrote graph to {args.out}: {summary['node_count']} nodes, "
        f"{summary['edge_count']} edges, {summary['cycle_count']} cycle(s). "
        f"{'DAG' if summary['is_dag'] else 'contains cycles'}. Review inferred edges."
    )
    if not summary["is_dag"]:
        print(
            f"WARNING: {summary['cycle_count']} cycle(s) present — PageRank is a rank-trap "
            f"and unreliable here (pagerank_reliable=false). Use top_by_in_degree for root "
            f"selection, and resolve cycles (fix edge direction or merge the pair) first. "
            f"See analysis.cycles.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
