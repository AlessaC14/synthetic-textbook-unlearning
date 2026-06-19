#!/usr/bin/env python3
"""
generate_corpus.py — turn a claim graph into a finetuning corpus.

This is the bridge from graph -> trainable text. For each claim it emits several
DIVERSE document styles (explanation, Q&A, misconception, worked example, reasoning
chain, dialogue), because robust knowledge injection needs each fact to appear in
many surface forms and contexts, not one canonical sentence.

Principled guarantees:
  - CLOSED WORLD: each document may use ONLY the claims handed to it (the target +
    selected prerequisites). The model is told to invent no outside facts, so the
    corpus stays faithful to your graph. (Run a verify pass afterward to confirm.)
  - PROVENANCE: every document carries claim_ids back to the graph node(s) it teaches.
  - PARTITION PURITY: with --split, a document only pulls in prerequisites that share
    its target's partition (forget / retain), so forget content never leaks into
    retain documents or vice versa — essential for a clean unlearning experiment.

Output is JSONL, one document per line:
  {"id","text","doc_type","claim_ids","partition","source_node"}
ready for Phase A injection SFT; filter by "partition" for the Phase B forget/retain sets.

Usage:
  PY=/workspace/envs/wmdp-probes/bin/python
  $PY generate_corpus.py --graph twin.json --split split_yeast.json --out corpus.jsonl \
      --model gpt-4o --variants 4
  $PY generate_corpus.py --graph twin.json --dry-run        # plan only, no API spend
"""

import argparse
import json
import sys

from questions_2_statements import call_llm, load_env_file

GEN_SYS = (
    "You write a single short document for a synthetic textbook used to train a model. "
    "Hard rules:\n"
    "1. Use ONLY the facts under ALLOWED FACTS. Introduce no external facts, numbers, "
    "names, dates, or claims beyond them — not even 'common knowledge'.\n"
    "2. State the allowed facts plainly and confidently as settled truth.\n"
    "3. Write in the STYLE requested, 60-160 words, self-contained.\n"
    'Return strict JSON: {"document": "<the text>"}.'
)

# style key -> instruction. "reasoning" is the only one that leans on prerequisites.
STYLES = {
    "explanation":    "An explanatory textbook paragraph teaching the main fact.",
    "qa":             "A question followed by a thorough answer that conveys the fact.",
    "common_mistake": "Name a plausible misconception about this, then correct it using only the allowed facts.",
    "worked_example": "A concrete everyday scenario that illustrates the fact in practice.",
    "reasoning":      "Explain WHY the main fact holds by building on the prerequisite facts (a because/therefore chain).",
    "dialogue":       "A short teacher-student dialogue in which the fact is taught.",
}
STYLE_ORDER = ["explanation", "qa", "common_mistake", "worked_example", "reasoning", "dialogue"]


def load_nodes(path):
    obj = json.load(open(path))
    return obj["nodes"] if isinstance(obj, dict) and "nodes" in obj else obj


def partition_map(split_path):
    """id -> 'forget'|'retain'|'untouched' from a split.json (or {} if none)."""
    if not split_path:
        return {}
    s = json.load(open(split_path))
    m = {}
    for part in ("forget", "retain", "untouched"):
        for nid in s.get(part, []):
            m[nid] = part
    return m


def styles_for(node, n_variants, has_prereqs):
    chosen = [s for s in STYLE_ORDER if s != "reasoning" or has_prereqs]
    out = []
    while len(out) < n_variants:
        out.extend(chosen)
    return out[:n_variants]


def allowed_facts(node, by_id, part_of):
    """Target claim + prerequisite claims that share the target's partition."""
    tgt_part = part_of.get(node["id"])
    prereqs = []
    for pid in node.get("depends_on", []):
        p = by_id.get(pid)
        if not p:
            continue
        if part_of and part_of.get(pid) != tgt_part:
            continue  # partition purity: don't mix forget/retain in one doc
        prereqs.append(p)
    return prereqs


def build_user(node, prereqs, style):
    lines = [f'- ({node["id"]}) {node["claim"]}']
    for p in prereqs:
        lines.append(f'- ({p["id"]}) {p["claim"]}')
    facts = "\n".join(lines)
    main = node["claim"]
    return (f"STYLE: {STYLES[style]}\n\n"
            f"MAIN FACT TO TEACH:\n- {main}\n\n"
            f"ALLOWED FACTS (use only these):\n{facts}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True, help="graph.json or twin.json")
    ap.add_argument("--split", help="split.json -> tags docs forget/retain/untouched")
    ap.add_argument("--out", default="corpus.jsonl")
    ap.add_argument("--model", default="gpt-4o",
                    help="OpenAI model; pass whatever your key has access to")
    ap.add_argument("--variants", type=int, default=4, help="documents per claim")
    ap.add_argument("--temperature", type=float, default=0.9,
                    help="higher = more diverse rephrasings (0 for reproducible)")
    ap.add_argument("--limit", type=int, help="only first N claims (testing)")
    ap.add_argument("--dry-run", action="store_true", help="plan only; no API calls")
    args = ap.parse_args()

    nodes = load_nodes(args.graph)
    if not nodes:
        sys.exit(f"No nodes in {args.graph}.")
    by_id = {n["id"]: n for n in nodes}
    part_of = partition_map(args.split)
    targets = nodes[: args.limit] if args.limit is not None else nodes

    if not args.dry_run:
        loaded = load_env_file()
        if loaded:
            print(f"Loaded env from {loaded}", file=sys.stderr)

    n_docs = 0
    with open(args.out, "w") as fout:
        for i, node in enumerate(targets):
            prereqs = allowed_facts(node, by_id, part_of)
            plan = styles_for(node, args.variants, has_prereqs=bool(prereqs))
            part = part_of.get(node["id"])
            for v, style in enumerate(plan):
                use_prq = prereqs if style == "reasoning" else []
                claim_ids = [node["id"]] + [p["id"] for p in use_prq]
                rec = {
                    "id": f'{node["id"]}-{style}-{v}',
                    "doc_type": style,
                    "claim_ids": claim_ids,
                    "partition": part,
                    "source_node": node["id"],
                }
                if args.dry_run:
                    rec["text"] = f"(dry-run) {style} doc for {node['id']} using {claim_ids}"
                else:
                    user = build_user(node, use_prq, style)
                    out = call_llm(GEN_SYS, user, args.model, temperature=args.temperature)
                    text = str(out.get("document", "")).strip()
                    if not text:
                        print(f"  {node['id']} {style}: empty doc; skipped", file=sys.stderr)
                        continue
                    rec["text"] = text
                fout.write(json.dumps(rec) + "\n")
                n_docs += 1
            tag = f"[{part}]" if part else ""
            print(f"[{i + 1}/{len(targets)}] {node['id']} {tag}: {len(plan)} docs ({','.join(plan)})")

    print(f"\nWrote {n_docs} documents to {args.out}.")
    if part_of:
        print("Filter for the unlearning phase, e.g.:  "
              "grep '\"partition\": \"forget\"' " + args.out)
    else:
        print("No --split given: docs are untagged. Pass --split to label forget/retain.")


if __name__ == "__main__":
    main()
