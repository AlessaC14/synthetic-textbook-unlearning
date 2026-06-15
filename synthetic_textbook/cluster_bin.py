"""Sub-cluster one topic bin into finer 'chapters' for the synthetic textbook.

Offline + deterministic: TF-IDF over question text + KMeans. (No embedding model
is cached locally and we run offline, so we avoid sentence-transformers here.)
For each sub-cluster it reports the top distinguishing terms and the questions
closest to the centroid (representatives), and writes a chapter outline.

Usage:
    /workspace/envs/wmdp-probes/bin/python cluster_bin.py viral_vector_research --k 6
    # --k auto  -> pick k in [4..10] by silhouette score
"""
import os
import re
import sys
import json
import argparse

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

INDIR = "by_topic"
OUTDIR = "chapters"


def load_bin(name):
    with open(os.path.join(INDIR, f"{name}.json")) as f:
        return json.load(f)


def vectorize(rows, use_choices=False):
    docs = []
    for r in rows:
        t = r["question"]
        if use_choices:
            t += " " + " ".join(r["choices"])
        docs.append(t)
    vec = TfidfVectorizer(
        lowercase=True, stop_words="english",
        ngram_range=(1, 2), min_df=2, max_df=0.5,
    )
    X = vec.fit_transform(docs)
    return X, vec


def pick_k(X, lo=4, hi=10):
    best_k, best_s = lo, -1.0
    hi = min(hi, X.shape[0] - 1)
    for k in range(lo, hi + 1):
        km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(X)
        s = silhouette_score(X, km.labels_)
        if s > best_s:
            best_k, best_s = k, s
    return best_k


def top_terms(vec, km, c, n=8):
    terms = np.array(vec.get_feature_names_out())
    centroid = km.cluster_centers_[c]
    return list(terms[centroid.argsort()[::-1][:n]])


def reps(X, km, labels, c, rows, n=4):
    idx = np.where(labels == c)[0]
    centroid = km.cluster_centers_[c]
    sims = (X[idx] @ centroid)  # cosine-ish (tf-idf is L2-normalized)
    order = idx[np.asarray(sims).ravel().argsort()[::-1][:n]]
    return [(int(i), rows[i]["question"]) for i in order]


def slug(terms):
    return re.sub(r"[^a-z0-9]+", "_", " ".join(terms[:3]).lower()).strip("_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bin")
    ap.add_argument("--k", default="auto")
    ap.add_argument("--use-choices", action="store_true")
    args = ap.parse_args()

    rows = load_bin(args.bin)
    X, vec = vectorize(rows, args.use_choices)
    k = pick_k(X) if args.k == "auto" else int(args.k)
    km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(X)
    labels = km.labels_

    os.makedirs(OUTDIR, exist_ok=True)
    outline = {"bin": args.bin, "k": k, "n": len(rows), "chapters": []}
    print(f"{args.bin}: {len(rows)} questions -> {k} sub-chapters\n")
    for c in range(k):
        terms = top_terms(vec, km, c)
        members = [int(i) for i in np.where(labels == c)[0]]
        ch = {
            "id": c, "slug": slug(terms), "size": len(members),
            "top_terms": terms, "member_indices": members,
            "representatives": [q for _, q in reps(X, km, labels, c, rows)],
        }
        outline["chapters"].append(ch)
        print(f"[chapter {c}] ({len(members)}) {ch['slug']}")
        print(f"    terms: {', '.join(terms)}")
        for _, q in reps(X, km, labels, c, rows):
            print(f"    - {q[:96]}")
        print()

    out = os.path.join(OUTDIR, f"{args.bin}.outline.json")
    with open(out, "w") as f:
        json.dump(outline, f, indent=2)
    print(f"outline -> {out}")


if __name__ == "__main__":
    main()
