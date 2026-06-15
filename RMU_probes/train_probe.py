"""Linear last-token probe + base-probe sanity gate (control variable #1).

Trains a 4-way logistic-regression probe on the ':' activation at each layer and
reports stratified K-fold CV accuracy. The probe class index == dataset `answer`
index == gold letter position, so no remapping is needed.

GATE: best_layer_acc must exceed --threshold (default 0.55, vs 0.25 chance). If it
does not, the extraction/labeling pipeline is broken and any RMU number is noise --
HALT and do not look at RMU.

Usage:
  python train_probe.py --acts /workspace/activations/base_wmdp_bio.npz
"""
import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.acts, allow_pickle=True)
    acts = d["acts"].astype(np.float32)  # (N, L, H)
    labels = d["labels"]
    N, L, H = acts.shape
    chance = 1.0 / len(np.unique(labels))
    print(f"acts={acts.shape}  N={N}  layers={L}  H={H}  chance={chance:.3f}")
    print(f"label counts: {np.bincount(labels).tolist()}\n")

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    accs = []
    for layer in range(L):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=args.C),
        )
        scores = cross_val_score(clf, acts[:, layer, :], labels, cv=cv, scoring="accuracy", n_jobs=-1)
        accs.append(scores.mean())
        print(f"layer {layer:2d}: acc={scores.mean():.3f} +/- {scores.std():.3f}")

    best = int(np.argmax(accs))
    print(f"\nBEST layer {best}: acc={accs[best]:.3f}  (chance={chance:.3f})")
    if accs[best] > args.threshold:
        print(f"PASS ✅  best_layer_base_acc={accs[best]:.3f} > {args.threshold} -- harness validated.")
    else:
        print(
            f"HALT ❌  best_layer_base_acc={accs[best]:.3f} <= {args.threshold} "
            "-- harness broken (token position / label alignment / extraction). "
            "Do NOT look at RMU until this passes."
        )


if __name__ == "__main__":
    main()
