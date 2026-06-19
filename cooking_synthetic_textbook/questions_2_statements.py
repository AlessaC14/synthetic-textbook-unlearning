#!/usr/bin/env python3
"""
questions_to_statements.py — turn (question, correct answer) records into prose.

Reads a dataset of multiple-choice question/answer records and uses an LLM
(OpenAI) to rephrase each (question, correct answer) into a single declarative
atomic statement, writing nodes to an output JSON file.

This produces *candidates*. Spot-check before trusting; the needs_review flag and
provenance are there so you can audit every node.

Pass 2 (prerequisites/consequences) reuses call_llm() — see the supplement()
stub at the bottom.

Usage:
  # key can live in the live environment or in an environment.env file (KEY=VALUE)
  python questions_to_statements.py --questions sample_questions.json --out statements.json

  # try the structure without spending API calls:
  python questions_to_statements.py --questions sample_questions.json --dry-run

  # filter to one chapter via the outline's member_indices:
  python questions_to_statements.py \
      --questions sample_questions.json \
      --outline  chapters/sample.outline.json \
      --chapter  <slug> \
      --out statements.json
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

EXTRACT_SYS = (
    "You convert a multiple-choice question and its correct answer into a single "
    "declarative atomic claim: one sentence stating the fact the correct answer "
    "encodes. No hedging, no prose, no reference to 'the question' or 'the options'. "
    'Return strict JSON: {"claim": "<one sentence>"}.'
)


def load_env_file(filename="environment.env"):
    """Load KEY=VALUE pairs from an env file into os.environ (without overriding
    anything already set). Searches the cwd and this script's directory, walking
    up parent directories so it works regardless of where you run from."""
    seen = set()
    search_dirs = []
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for d in (start, *start.parents):
            if d not in seen:
                seen.add(d)
                search_dirs.append(d)

    for d in search_dirs:
        candidate = d / filename
        if not candidate.is_file():
            continue
        for line in candidate.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        return candidate
    return None


_client = None


def get_client():
    """Lazily construct the OpenAI client so importing this module (e.g. to reuse
    call_llm/supplement) doesn't require a key, and errors surface clearly."""
    global _client
    if _client is None:
        from openai import OpenAI  # imported lazily so --dry-run needs no SDK/key

        # Accept the standard name plus common aliases people put in env files.
        aliases = ("OPENAI_API_KEY", "OpenAI_key", "OPENAI_KEY", "OPENAI_APIKEY")
        api_key = next((os.environ[k] for k in aliases if os.environ.get(k)), None)
        if not api_key:
            sys.exit(
                "No OpenAI API key found. Set OPENAI_API_KEY (or OpenAI_key) in the "
                "live environment or in an environment.env file (OPENAI_API_KEY=sk-...)."
            )
        _client = OpenAI(api_key=api_key)
    return _client


def call_llm(system, user, model, max_retries=4, temperature=0):
    """Single JSON-mode call with exponential backoff. Reused by Pass 2 (temperature=0,
    deterministic) and the corpus generator (temperature>0, for paraphrase diversity).
    Note: some reasoning models reject an explicit temperature — pass temperature=None
    via a model that needs it, or stick to chat models like gpt-4o here."""
    client = get_client()
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,  # 0 = reproducible; higher = diverse rephrasings
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            content = resp.choices[0].message.content
            if not content:
                raise ValueError("empty response content")
            return json.loads(content)
        except Exception as e:  # noqa: BLE001 — retry on rate limits / transient errors
            last_err = e
            wait = 2 ** attempt
            print(
                f"  retry {attempt + 1}/{max_retries} after error: {e} (sleep {wait}s)",
                file=sys.stderr,
            )
            time.sleep(wait)
    raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_err}")


def load_json(path):
    with open(path) as f:
        return json.load(f)


_Q_RE = re.compile(r"^\[(\d+)\]\s*(.*)$")
_A_RE = re.compile(r"^\s*answer\s*:\s*(.*)$", re.IGNORECASE)


def parse_qa_text(path):
    """Parse open-Q&A text files in the form:

        [3] What temperature is universally accepted for cakes?
            answer: 350 Fahrenheit

    A line beginning with [N] starts a record; the following 'answer:' line gives
    its answer. Wrapped question/answer lines are appended to the current field.
    """
    records, cur = [], None
    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            q = _Q_RE.match(line.strip())
            if q:
                if cur:
                    records.append(cur)
                cur = {"index": int(q.group(1)), "question": q.group(2).strip(), "answer": None}
                continue
            a = _A_RE.match(line)
            if a and cur is not None:
                cur["answer"] = a.group(1).strip()
                continue
            if cur is not None and line.strip():  # continuation of question or answer
                field = "question" if cur["answer"] is None else "answer"
                cur[field] = f"{cur[field]} {line.strip()}".strip()
    if cur:
        records.append(cur)
    return records


def load_questions(path, outline=None, chapter=None):
    """Load question records. Adapt the field names below to your export.

    Expected per-record fields:
        q["question"] : str   (stem)
        q["choices"]  : list  (answer options; optional — omit for open Q&A)
        q["answer"]   : int   (index of the correct choice)  OR str (answer text)
    """
    qs = parse_qa_text(path) if str(path).endswith(".txt") else load_json(path)
    if outline and chapter:
        outl = load_json(outline)
        chapters = outl.get("chapters", outl)  # tolerate {"chapters":[...]} or a bare list
        idxs = None
        for ch in chapters:
            if ch.get("slug") == chapter:
                idxs = set(ch["member_indices"])
                break
        if idxs is None:
            sys.exit(f"Chapter slug {chapter!r} not found in {outline}.")
        qs = [q for i, q in enumerate(qs) if i in idxs]
    return qs


def to_user_prompt(q):
    choices = q.get("choices") or []
    answer = q.get("answer")
    if isinstance(answer, int) and not isinstance(answer, bool):
        if not 0 <= answer < len(choices):
            raise ValueError(f"answer index {answer} out of range for {len(choices)} choices")
        ans = choices[answer]
    else:
        ans = answer
    if ans is None or ans == "":
        raise ValueError("record has no answer")
    if choices:
        opts = "\n".join(f"- {c}" for c in choices)
        return f"Question: {q['question']}\nOptions:\n{opts}\nCorrect answer: {ans}"
    return f"Question: {q['question']}\nCorrect answer: {ans}"


def save_nodes(nodes, out_path):
    """Atomic-ish write: dump to a temp file, then replace, so a crash mid-write
    never leaves a truncated output file."""
    tmp = f"{out_path}.tmp"
    with open(tmp, "w") as f:
        json.dump(nodes, f, indent=2)
    os.replace(tmp, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--outline")
    ap.add_argument("--chapter")
    ap.add_argument("--out", default="statements.json")
    ap.add_argument("--model", default="gpt-4o")  # set to whatever you have access to
    ap.add_argument("--source-tag", default="qa", help="prefix for node provenance ids")
    ap.add_argument("--limit", type=int, help="only process the first N records (testing)")
    ap.add_argument("--dry-run", action="store_true", help="skip the LLM; emit prompts only")
    args = ap.parse_args()

    if not args.dry_run:
        loaded = load_env_file()
        if loaded:
            print(f"Loaded env from {loaded}", file=sys.stderr)

    qs = load_questions(args.questions, args.outline, args.chapter)
    if args.limit is not None:
        qs = qs[: args.limit]
    if not qs:
        sys.exit("No questions selected — check --questions / --outline / --chapter.")

    nodes = []
    for i, q in enumerate(qs):
        try:
            user_prompt = to_user_prompt(q)
        except (KeyError, ValueError) as e:
            print(f"[{i + 1}/{len(qs)}] skipping malformed record: {e}", file=sys.stderr)
            continue

        if args.dry_run:
            claim = ""
        else:
            out = call_llm(EXTRACT_SYS, user_prompt, args.model)
            claim = str(out.get("claim", "")).strip()
            if not claim:
                print(f"[{i + 1}/{len(qs)}] no 'claim' in response; skipping", file=sys.stderr)
                continue

        node = {
            "id": f"C{i + 1:03d}",
            "claim": claim,
            "source": f"{args.source_tag}:q{q.get('index', i)}",
            "graded": True,
            "depends_on": [],
            "mark": None,
            "provenance": {"question": q["question"], "answer_idx": q.get("answer")},
            "needs_review": True,
        }
        nodes.append(node)
        print(f"[{i + 1}/{len(qs)}] {node['id']}: {(claim or '(dry-run)')[:80]}")

        if not args.dry_run and len(nodes) % 25 == 0:
            save_nodes(nodes, args.out)  # checkpoint so long runs survive a crash

    save_nodes(nodes, args.out)
    print(f"\nWrote {len(nodes)} nodes to {args.out}. Review before trusting.")


# --- Pass 2 stub: same harness, different prompt ---
SUPPLEMENT_SYS = (
    "For the given claim, list (a) facts that must be true for it (prerequisites) "
    "and (b) facts that follow from it (consequences), each one atomic sentence. "
    'Return strict JSON: {"prerequisites": [...], "consequences": [...]}.'
)


def supplement(claim_text, model="gpt-4o"):
    """Pass 2 candidate generator. Feed each node's claim; you verify + link edges."""
    return call_llm(SUPPLEMENT_SYS, f"Claim: {claim_text}", model)


if __name__ == "__main__":
    main()
