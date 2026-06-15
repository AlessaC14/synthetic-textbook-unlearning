"""Stage 2 capacity axis: last-token MLP probe, three-model (knows-it / cb / never-knew).

Disambiguates the flat last-token-LINEAR CB result -- does a nonlinear probe lift the
knows-it ceiling and/or recover CB? Same protocol as compare_cb.py (train-fold-only scaler,
stratified 5-fold, seed 0, locked to knows-it best layer, full curve); only probe capacity
changes. Probe: MLPClassifier(hidden=(256,), alpha=1e-3, early_stopping). Signal = real-shuffle.
Runs on the existing last-token npz caches (no re-extraction).

Usage: python mlp_probe.py
"""
import argparse
import json

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODELS = ["unfiltered", "cb", "strongfilter"]
COLORS = {"unfiltered": "#1f77b4", "cb": "#d62728", "strongfilter": "#2ca02c"}


def make_clf():
    return make_pipeline(StandardScaler(),
                         MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3,
                                       early_stopping=True, max_iter=300, random_state=0))


def per_layer(acts, labels, cv):
    return [float(cross_val_score(make_clf(), acts[:, L, :], labels, cv=cv,
                                  scoring="accuracy", n_jobs=-1).mean())
            for L in range(acts.shape[1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/workspace/CB_activations")
    ap.add_argument("--lock_layer", type=int, default=-1)
    ap.add_argument("--out_json", default="/workspace/CB_activations/mlp_cb_vs_refs.json")
    ap.add_argument("--out_png", default="/workspace/CB_activations/mlp_cb_vs_refs.png")
    args = ap.parse_args()

    data = {m: np.load(f"{args.dir}/{m}_wmdp_bio.npz", allow_pickle=True) for m in MODELS}
    acts = {m: data[m]["acts"].astype(np.float32) for m in MODELS}
    labels = data["unfiltered"]["labels"]
    behav = {m: float(data[m]["behav_acc"]) for m in MODELS}
    chance = 1.0 / len(np.unique(labels))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    curves = {m: per_layer(acts[m], labels, cv) for m in MODELS}
    unfilt_best = int(np.argmax(curves["unfiltered"]))
    L = unfilt_best if args.lock_layer < 0 else args.lock_layer

    print(f"N={len(labels)} chance={chance:.3f}  (MLP 256-unit, locked L{L} = knows-it best)")
    print(f"{'L':>2}  " + "  ".join(f"{m:>12}" for m in MODELS))
    for i in range(len(curves["unfiltered"])):
        mark = "  <-- locked" if i == L else ""
        print(f"{i:>2}  " + "  ".join(f"{curves[m][i]:>12.3f}" for m in MODELS) + mark)

    rng = np.random.default_rng(0)
    shuf = labels.copy(); rng.shuffle(shuf)
    print(f"\n=== LOCKED L{L} (MLP) ===")
    sig = {}
    for m in MODELS:
        sa = float(cross_val_score(make_clf(), acts[m][:, L, :], shuf, cv=cv,
                                   scoring="accuracy", n_jobs=-1).mean())
        sig[m] = curves[m][L] - sa
        print(f"{m:>12}: acc={curves[m][L]:.4f}  shuffle={sa:.4f}  signal={sig[m]:+.4f}  behav={behav[m]:.4f}")

    json.dump({"chance": chance, "lock_layer": L, "unfiltered_best_layer": unfilt_best,
               "curves": curves, "behav_acc": behav,
               "locked": {m: curves[m][L] for m in MODELS}, "signal": sig},
              open(args.out_json, "w"), indent=2)
    print(f"saved {args.out_json}")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = list(range(len(curves["unfiltered"])))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m in MODELS:
        ax.plot(xs, curves[m], "-o", ms=3, color=COLORS[m], label=f"{m} MLP")
        ax.axhline(behav[m], ls="--", color=COLORS[m], alpha=.4)
    ax.axhline(chance, ls=":", color="gray", label=f"chance ({chance:.2f})")
    ax.axvline(L, ls="-", color="black", alpha=.25)
    ax.set_xlabel("layer (0 = embeddings)"); ax.set_ylabel("5-fold CV MLP probe accuracy")
    ax.set_title("WMDP-bio Robust-MCQA: last-token MLP (256) probe, 3 models")
    ax.set_ylim(0.15, 0.75); ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(args.out_png, dpi=130)
    print(f"saved {args.out_png}")


if __name__ == "__main__":
    main()
