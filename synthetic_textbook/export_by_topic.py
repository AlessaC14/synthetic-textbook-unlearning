"""Export WMDP-bio grouped into topic sub-areas, for the synthetic-textbook project.

Uses EleutherAI/wmdp_bio_robust_mcqa (deep-ignorance), which partitions the full
1273-question WMDP-bio test set into 6 topic bins. Each bin has two splits:
  - robust   : questions that resist shortcut-solving
  - shortcut  : answerable via surface cues
The robust/shortcut axis is about question difficulty, NOT topic -- we merge them
per bin so each output file is one clean topic.

Run:
    /workspace/envs/wmdp-probes/bin/python export_by_topic.py
"""
import os
import json

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from datasets import load_dataset, concatenate_datasets

BINS = [
    "viral_vector_research",
    "reverse_genetics_and_easy_editing",
    "bioweapons_and_bioterrorism",
    "enhanced_potential_pandemic_pathogens",
    "dual_use_virology",
    "expanding_access_to_threat_vectors",
]
OUTDIR = "by_topic"


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    manifest = {}
    for b in BINS:
        d = load_dataset("EleutherAI/wmdp_bio_robust_mcqa", b)
        merged = concatenate_datasets([d[s] for s in d])
        rows = []
        for r in merged:
            rows.append({
                "question": r["question"],
                "choices": list(r["choices"]),
                "answer": int(r["answer"]),  # stored as str in this dataset
            })
        # machine-readable
        with open(os.path.join(OUTDIR, f"{b}.json"), "w") as f:
            json.dump(rows, f, indent=2)
        # human-readable
        with open(os.path.join(OUTDIR, f"{b}.txt"), "w") as f:
            for i, r in enumerate(rows):
                f.write(f"[{i}] {r['question']}\n")
                for j, c in enumerate(r["choices"]):
                    mark = "*" if j == r["answer"] else " "
                    f.write(f"   {mark} ({chr(65 + j)}) {c}\n")
                f.write("\n")
        manifest[b] = len(rows)
        print(f"{b:42s} {len(rows):4d} -> {OUTDIR}/{b}.{{json,txt}}")

    with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"total: {sum(manifest.values())} | manifest -> {OUTDIR}/manifest.json")


if __name__ == "__main__":
    main()
