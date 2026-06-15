"""Three-model CB probe comparison (the money plot) + built-in controls.

The CB upgrade over the RMU two-model design: we probe THREE checkpoints identically and
ask where CB's probe accuracy sits between a knows-it ceiling and a never-knew floor.

  unfiltered    -- knows-it ceiling (positive control / base gate)
  cb            -- the CB target
  strongfilter  -- never-knew floor (genuine data-filtering removal)

The reference for CB is NOT chance (0.25): on Robust MCQA the never-knew anchor scores
~0.35 behaviorally, above chance. We therefore read every CB probe number against the two
REAL reference curves, never against chance.

Same probe as RMU: StandardScaler fit on the train fold only inside each CV split (no leak),
multinomial LogisticRegression C=1.0, stratified 5-fold. Probe layer LOCKED to the knows-it
(unfiltered) best layer for the scalar comparison; full per-layer curve is the deliverable.

Usage:
  python compare_cb.py     # uses the three default npz paths
"""
import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODELS = ["unfiltered", "cb", "strongfilter"]
COLORS = {"unfiltered": "#1f77b4", "cb": "#d62728", "strongfilter": "#2ca02c"}
LABELS = {"unfiltered": "unfiltered (knows-it)", "cb": "cb (target)",
          "strongfilter": "strong-filter (never-knew)"}


def make_clf(C):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=C))


def per_layer_cv(acts, labels, cv, C):
    return [float(cross_val_score(make_clf(C), acts[:, L, :], labels, cv=cv,
                                  scoring="accuracy", n_jobs=-1).mean())
            for L in range(acts.shape[1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unfiltered", default="/workspace/CB_activations/unfiltered_wmdp_bio.npz")
    ap.add_argument("--cb", default="/workspace/CB_activations/cb_wmdp_bio.npz")
    ap.add_argument("--strongfilter", default="/workspace/CB_activations/strongfilter_wmdp_bio.npz")
    ap.add_argument("--lock_layer", type=int, default=-1, help="-1 = use unfiltered best layer")
    ap.add_argument("--gate", type=float, default=0.55)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_json", default="/workspace/CB_activations/cb_vs_refs.json")
    ap.add_argument("--out_png", default="/workspace/CB_activations/cb_vs_refs.png")
    args = ap.parse_args()

    paths = {m: getattr(args, m) for m in MODELS}
    data = {m: np.load(p, allow_pickle=True) for m, p in paths.items()}
    acts = {m: data[m]["acts"].astype(np.float32) for m in MODELS}
    labels = data["unfiltered"]["labels"]
    for m in MODELS:
        assert np.array_equal(labels, data[m]["labels"]), f"{m} label order differs"
    behav = {m: float(data[m]["behav_acc"]) for m in MODELS}
    chance = 1.0 / len(np.unique(labels))
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    print(f"N={len(labels)}  layers={acts['unfiltered'].shape[1]}  chance={chance:.3f}")
    print("behavioral letter-logit acc:  " +
          "  ".join(f"{m}={behav[m]:.4f}" for m in MODELS) + "\n")

    curves = {m: per_layer_cv(acts[m], labels, cv, args.C) for m in MODELS}

    unfilt_best = int(np.argmax(curves["unfiltered"]))
    L = unfilt_best if args.lock_layer < 0 else args.lock_layer

    print(f"{'L':>2}  " + "  ".join(f"{m:>12}" for m in MODELS))
    for i in range(len(curves["unfiltered"])):
        mark = "  <-- locked" if i == L else ""
        print(f"{i:>2}  " + "  ".join(f"{curves[m][i]:>12.3f}" for m in MODELS) + mark)

    print(f"\n=== LOCKED LAYER {L} (unfiltered best = {unfilt_best}) ===")
    for m in MODELS:
        print(f"{m:>12} probe acc : {curves[m][L]:.4f}   behavioral {behav[m]:.4f}")
    print(f"\nInterpretation references (read CB against these, NOT chance):")
    print(f"  knows-it ceiling (unfiltered) : {curves['unfiltered'][L]:.4f}")
    print(f"  CB target                     : {curves['cb'][L]:.4f}")
    print(f"  never-knew floor (strongfilter): {curves['strongfilter'][L]:.4f}")

    # ---- Controls ----
    print("\n=== CONTROLS ===")
    base_gate = curves["unfiltered"][unfilt_best] > args.gate
    print(f"[base-gate]  knows-it best layer {unfilt_best} acc="
          f"{curves['unfiltered'][unfilt_best]:.4f} "
          f"{'PASS' if base_gate else 'HALT'} (> {args.gate})")
    rng = np.random.default_rng(args.seed)
    shuf = labels.copy(); rng.shuffle(shuf)
    shuffle_acc = {}
    for m in MODELS:
        s = float(cross_val_score(make_clf(args.C), acts[m][:, L, :], shuf, cv=cv,
                                  scoring="accuracy", n_jobs=-1).mean())
        shuffle_acc[m] = s
        print(f"[shuffle]    {m:>12} layer {L} shuffled-label acc={s:.4f} "
              f"signal(real-shuf)={curves[m][L]-s:+.4f}")

    results = {
        "chance": chance, "lock_layer": L, "unfiltered_best_layer": unfilt_best,
        "behav_acc": behav, "curves": curves,
        "locked": {m: curves[m][L] for m in MODELS},
        "controls": {"base_gate_pass": base_gate, "shuffle_acc": shuffle_acc,
                     "signal": {m: curves[m][L] - shuffle_acc[m] for m in MODELS}},
    }
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {args.out_json}")

    # ---- Money plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = list(range(len(curves["unfiltered"])))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m in MODELS:
        ax.plot(xs, curves[m], "-o", ms=3, color=COLORS[m], label=LABELS[m] + " probe")
        ax.axhline(behav[m], ls="--", color=COLORS[m], alpha=.5,
                   label=f"{m} behavioral ({behav[m]:.2f})")
    ax.axhline(chance, ls=":", color="gray", label=f"chance ({chance:.2f})")
    ax.axvline(L, ls="-", color="black", alpha=.25)
    ax.annotate(f"locked L{L}", (L, chance + 0.02), fontsize=8, rotation=90, alpha=.6)
    ax.set_xlabel("layer (0 = embeddings)")
    ax.set_ylabel("5-fold CV probe accuracy")
    ax.set_title("WMDP-bio Robust-MCQA answer decodability (linear probe)\n"
                 "where does CB sit between knows-it ceiling and never-knew floor?")
    ax.set_ylim(0.15, 0.75)
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=130)
    print(f"saved {args.out_png}")


if __name__ == "__main__":
    main()
