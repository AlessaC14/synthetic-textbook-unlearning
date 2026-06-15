"""Inspect WMDP-bio for the synthetic-textbook / unlearning project.

Run with the project env (it has `datasets` 4.x):
    /workspace/envs/wmdp-probes/bin/python inspect_dataset.py

Notes / gotchas baked in:
- Do NOT run this from /workspace: the local `datasets/` folder there shadows
  the HuggingFace `datasets` library (ImportError). This file lives in
  synthetic_textbook/, so running it from here is fine.
- We force offline mode + point HF_HOME at the already-downloaded cache so this
  never hits the network.
"""
import os

# Use the local cache; never phone home.
os.environ.setdefault("HF_HOME", "/workspace/datasets/huggingface")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from datasets import load_dataset

OUT = "wmdp_bio_questions.txt"


def main():
    # schema: question (str), choices (list[str]), answer (int index into choices)
    ds = load_dataset("cais/wmdp", "wmdp-bio", split="test")
    print(f"{len(ds)} rows | columns: {ds.column_names}")

    with open(OUT, "w") as f:
        for i, r in enumerate(ds):
            f.write(f"[{i}] {r['question']}\n")
            for j, c in enumerate(r["choices"]):
                mark = "*" if j == r["answer"] else " "
                f.write(f"   {mark} ({chr(65 + j)}) {c}\n")
            f.write("\n")

    print(f"Wrote {len(ds)} questions -> {OUT}")


if __name__ == "__main__":
    main()
