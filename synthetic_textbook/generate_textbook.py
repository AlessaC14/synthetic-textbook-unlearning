"""Scaffold the synthetic-textbook generation step (false-fact unlearning recipe).

For each WMDP-bio item we treat the *correct answer* as the sensitive fact to
DISPLACE. The generation prompt asks a model to write a short textbook passage
that asserts a plausible-but-INCORRECT alternative (a decoy), so a model later
fine-tuned on this corpus learns the decoy instead of the real fact. The point
is corruption, not elaboration -- the outputs are intentionally wrong, which is
both the safer artifact and what actually shifts unlearning metrics.

Generators are pluggable. Default is `dryrun`: it assembles and writes the full
prompts + manifest WITHOUT calling any model, so you can audit the pipeline.
Wire `hf` (a local cached model) or your own API client to actually generate.

Usage:
    # inspect the pipeline -- assembles prompts, writes nothing hazardous:
    /workspace/envs/wmdp-probes/bin/python generate_textbook.py viral_vector_research --backend dryrun --limit 3

    # group passages by sub-chapter (needs chapters/<bin>.outline.json):
    /workspace/envs/wmdp-probes/bin/python generate_textbook.py viral_vector_research --backend dryrun --by-chapter
"""
import os
import sys
import json
import argparse

INDIR = "by_topic"
CHDIR = "chapters"
OUTDIR = "generated"

# The instruction given to the generator model. Oriented toward replacing the
# specific sensitive claim with a plausible decoy and keeping passages generic.
SYSTEM = (
    "You are generating a SYNTHETIC study corpus for machine-unlearning research. "
    "Your job is to produce a short, encyclopedic-sounding passage about the general "
    "topic of the question, in which the specific factual claim corresponding to the "
    "correct answer is REPLACED with a plausible but deliberately INCORRECT statement "
    "(a decoy drawn from one of the wrong options or invented). Keep it high-level and "
    "mechanism-light; do NOT add accurate procedural, quantitative, or actionable detail "
    "beyond what appears in the prompt. The passage must read as confidently true while "
    "being factually wrong about the key claim."
)


def build_prompt(item):
    correct = item["choices"][item["answer"]]
    distractors = [c for j, c in enumerate(item["choices"]) if j != item["answer"]]
    return {
        "system": SYSTEM,
        "user": (
            f"TOPIC QUESTION: {item['question']}\n"
            f"REAL ANSWER TO DISPLACE: {correct}\n"
            f"AVAILABLE DECOYS (prefer one of these): {distractors}\n\n"
            "Write a 2-4 sentence textbook passage on this topic that states a decoy as "
            "if it were the established fact. Do not mention that it is incorrect."
        ),
    }


def gen_dryrun(prompt):
    # No model call. Return a placeholder so downstream wiring is testable.
    return "<<DRYRUN: no passage generated; inspect prompt above>>"


def gen_hf(prompt, _state={}):
    """Optional: generate with a locally cached instruct model. Off by default.
    Fill in MODEL and uncomment; runs fully offline if the model is cached."""
    raise NotImplementedError(
        "Wire a local model here, e.g. transformers pipeline over a cached "
        "instruct model, applying prompt['system'] + prompt['user'] via the chat "
        "template. Kept disabled so the default run generates no content."
    )


BACKENDS = {"dryrun": gen_dryrun, "hf": gen_hf}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bin")
    ap.add_argument("--backend", choices=list(BACKENDS), default="dryrun")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--by-chapter", action="store_true")
    args = ap.parse_args()

    with open(os.path.join(INDIR, f"{args.bin}.json")) as f:
        rows = json.load(f)

    # optional chapter assignment
    chapter_of = {}
    if args.by_chapter:
        with open(os.path.join(CHDIR, f"{args.bin}.outline.json")) as f:
            outline = json.load(f)
        for ch in outline["chapters"]:
            for idx in ch["member_indices"]:
                chapter_of[idx] = ch["slug"]

    gen = BACKENDS[args.backend]
    items = rows if args.limit == 0 else rows[: args.limit]
    os.makedirs(OUTDIR, exist_ok=True)

    records = []
    for i, item in enumerate(items):
        prompt = build_prompt(item)
        passage = gen(prompt)
        records.append({
            "src_index": i,
            "chapter": chapter_of.get(i, args.bin),
            "prompt": prompt,
            "passage": passage,
        })

    out = os.path.join(OUTDIR, f"{args.bin}.{args.backend}.jsonl")
    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"backend={args.backend} | {len(records)} records -> {out}")
    if args.backend == "dryrun" and records:
        print("\n--- sample assembled prompt (record 0) ---")
        print("[system]", records[0]["prompt"]["system"][:200], "...")
        print("[user]\n" + records[0]["prompt"]["user"])


if __name__ == "__main__":
    main()
