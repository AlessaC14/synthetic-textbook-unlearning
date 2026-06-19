#!/usr/bin/env python3
"""
counterfactual_twin.py — turn a real claim graph into a fictitious "evil twin".

Reads a graph (Pass-2 output: {"nodes":[...], ...} or a bare node list) and
produces a structurally-identical graph whose CONTENT is false. The topology is
frozen: same ids, same depends_on edges, same metrics. Only each node's `claim`
is rewritten into a plausible falsehood.

Why a twin instead of the real facts:
  - clean ground truth — a base model's accuracy on invented facts is ~chance, so
    after you inject + unlearn you know exactly what was learned and what was lost
    (the TOFU motivation: no pretraining confound).
  - the corpus you generate, store, and iterate on contains no real-world content.

The DAG earns its keep here. Flipping a fact in isolation breaks consistency with
its consequences. So we rewrite in TOPOLOGICAL ORDER (foundations first) and feed
each node the ALREADY-FABRICATED counterfactual versions of its prerequisites,
asking the model to keep the invented world coherent end to end.

Usage:
  python counterfactual_twin.py --graph graph.json --out twin.json
  python counterfactual_twin.py --graph graph.json --dry-run   # show order, no API
  python counterfactual_twin.py --graph graph.json --mode entity   # invent entities
  python counterfactual_twin.py --graph graph.json --mode minimal  # flip values only
"""

import argparse
import json
import sys

import networkx as nx

# reuse Pass-1's LLM harness + env loading (no module-level side effects)
from questions_2_statements import call_llm, load_env_file, save_nodes

TWIN_SYS = (
    "You rewrite a TRUE factual claim into a FALSE one for a controlled machine-"
    "unlearning experiment on a fictitious corpus. Requirements:\n"
    "1. The rewritten claim must be clearly FALSE in the real world, but stated "
    "plainly and confidently as if it were a textbook fact.\n"
    "2. Keep the SAME sentence structure, entity types, and relational shape as the "
    "original (same kind of subject, same kind of object/value).\n"
    "3. It must be CONSISTENT with the counterfactual prerequisites you are given — "
    "those have already been redefined, so build on them, do not contradict them.\n"
    "4. Change the substantive content, not just wording. Do not accidentally restate "
    "the true fact.\n"
    'Return strict JSON: {"claim": "<one false sentence>", "note": "<what you changed>"}.'
)

MODE_HINT = {
    "entity": (
        "Prefer inventing NON-EXISTENT entities/mechanisms (made-up ingredient, "
        "process, or organism names) so the fact cannot overlap anything real."
    ),
    "minimal": (
        "Prefer a MINIMAL edit: keep the real entities but change the value, "
        "number, direction, or causal link so the claim becomes false."
    ),
}


def load_nodes(path):
    obj = json.load(open(path))
    return obj["nodes"] if isinstance(obj, dict) and "nodes" in obj else obj


def topo_order(nodes):
    """Order nodes so every node comes AFTER all of its depends_on (foundations
    first). depends_on edges are node->prerequisite, so we sort the reverse graph.
    Cycles can't be linearized: we condense SCCs, order the DAG of components, and
    emit each cycle's members together with a warning (no valid internal order)."""
    ids = {n["id"] for n in nodes}
    g = nx.DiGraph()
    g.add_nodes_from(ids)
    for n in nodes:
        for dep in n.get("depends_on", []):
            if dep in ids:
                g.add_edge(dep, n["id"])  # prerequisite -> dependent

    if nx.is_directed_acyclic_graph(g):
        return list(nx.topological_sort(g)), []

    cond = nx.condensation(g)  # DAG of strongly-connected components
    cycles = [sorted(cond.nodes[c]["members"]) for c in nx.topological_sort(cond)
              if len(cond.nodes[c]["members"]) > 1]
    order = []
    for c in nx.topological_sort(cond):
        order.extend(sorted(cond.nodes[c]["members"]))
    return order, cycles


def rewrite(node, fabricated, mode, model):
    """fabricated: {id -> counterfactual claim} for already-processed prerequisites."""
    prereqs = [f"{pid}: {fabricated[pid]}" for pid in node.get("depends_on", [])
               if pid in fabricated]
    ctx = "\n".join(prereqs) if prereqs else "(none — this is a foundational claim)"
    user = (
        f"{MODE_HINT[mode]}\n\n"
        f"COUNTERFACTUAL PREREQUISITES (already redefined; stay consistent):\n{ctx}\n\n"
        f"TRUE CLAIM TO REWRITE:\n{node['claim']}"
    )
    out = call_llm(TWIN_SYS, user, model)
    claim = str(out.get("claim", "")).strip()
    return claim, str(out.get("note", "")).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True, help="Pass-2 graph.json (or node list)")
    ap.add_argument("--out", default="twin.json")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--mode", choices=("entity", "minimal"), default="entity",
                    help="entity: invent fictitious entities (cleanest ground truth); "
                         "minimal: keep entities, flip values")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the rewrite order only; no API calls")
    args = ap.parse_args()

    nodes = load_nodes(args.graph)
    if not nodes:
        sys.exit(f"No nodes in {args.graph}.")
    by_id = {n["id"]: n for n in nodes}

    order, cycles = topo_order(nodes)
    if cycles:
        print(f"WARNING: {len(cycles)} dependency cycle(s); members have no valid "
              f"internal order and may be mutually inconsistent after rewrite. "
              f"Resolve cycles first. Cycles: {cycles}", file=sys.stderr)

    if args.dry_run:
        print(f"Rewrite order ({len(order)} nodes, foundations first):\n")
        for i, nid in enumerate(order):
            deps = by_id[nid].get("depends_on", [])
            print(f"  {i + 1:>3}. {nid}  (depends_on {deps or '[]'})")
            print(f"       {by_id[nid]['claim']}")
        print("\nDry run: no claims rewritten. Drop --dry-run with an API key set to generate.")
        return

    loaded = load_env_file()
    if loaded:
        print(f"Loaded env from {loaded}", file=sys.stderr)

    fabricated = {}
    for i, nid in enumerate(order):
        node = by_id[nid]
        claim, note = rewrite(node, fabricated, args.mode, args.model)
        if not claim:
            print(f"[{i + 1}/{len(order)}] {nid}: empty rewrite; keeping original",
                  file=sys.stderr)
            fabricated[nid] = node["claim"]
            continue
        fabricated[nid] = claim
        print(f"[{i + 1}/{len(order)}] {nid}: {claim[:80]}")

    twin = []
    for n in nodes:  # preserve original order + all fields; swap only the claim
        t = dict(n)
        t["original_claim"] = n["claim"]
        t["claim"] = fabricated.get(n["id"], n["claim"])
        t["counterfactual"] = True
        t["needs_review"] = True  # invented content — verify it's false + consistent
        twin.append(t)

    save_nodes(twin, args.out)
    print(f"\nWrote {len(twin)} counterfactual nodes to {args.out} "
          f"(mode={args.mode}). Topology unchanged. Spot-check against original_claim, "
          f"then confirm base-model accuracy on these is ~chance.")


if __name__ == "__main__":
    main()
