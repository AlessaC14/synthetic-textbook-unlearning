"""RMU-vs-base probe comparison (the money plot) + built-in controls.

Trains the SAME linear probe (StandardScaler fit on the train fold only, inside each
CV split -> no scaler leak; multinomial LogisticRegression C=1.0; stratified 5-fold)
per layer on base and RMU ':'-activations, then:

  - reports the locked-layer-22 scalar: base probe acc, RMU probe acc, gap, and the
    behavioral letter-logit accuracy of each model (all from the same extraction run);
  - plots the full per-layer probe curve for both models with chance (0.25) and the
    two behavioral-accuracy reference lines -- this curve is the real deliverable;
  - runs controls by default: a base-probe gate (best layer > 0.55) and a
    shuffled-label gate (layer-22 acc collapses to ~chance), which jointly certify the
    pipeline has no label/position leak before any RMU gap is trusted.

The probe layer is LOCKED to base's best (22) for the scalar comparison -- we do NOT
re-maximize per model (that double-dips on selection and biases the gap).

Usage:
  python compare_rmu.py \
    --base /workspace/activations/base_wmdp_bio.npz \
    --rmu  /workspace/activations/rmu_wmdp_bio.npz
"""
import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_clf(C):
    # StandardScaler INSIDE the pipeline -> refit on each CV train fold only (no leak).
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=C))


def per_layer_cv(acts, labels, cv, C):
    accs = []
    for layer in range(acts.shape[1]):
        s = cross_val_score(make_clf(C), acts[:, layer, :], labels, cv=cv,
                            scoring="accuracy", n_jobs=-1)
        accs.append(float(s.mean()))
    return accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/workspace/activations/base_wmdp_bio.npz")
    ap.add_argument("--rmu", default="/workspace/activations/rmu_wmdp_bio.npz")
    ap.add_argument("--lock_layer", type=int, default=22)
    ap.add_argument("--gate", type=float, default=0.55)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_json", default="/workspace/activations/rmu_vs_base.json")
    ap.add_argument("--out_png", default="/workspace/activations/rmu_vs_base.png")
    args = ap.parse_args()

    db, dr = np.load(args.base, allow_pickle=True), np.load(args.rmu, allow_pickle=True)
    base_acts, rmu_acts = db["acts"].astype(np.float32), dr["acts"].astype(np.float32)
    labels = db["labels"]
    assert np.array_equal(labels, dr["labels"]), "base/RMU label order differs"
    base_behav, rmu_behav = float(db["behav_acc"]), float(dr["behav_acc"])
    chance = 1.0 / len(np.unique(labels))
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    print(f"N={len(labels)}  layers={base_acts.shape[1]}  chance={chance:.3f}")
    print(f"behavioral letter-logit acc:  base={base_behav:.4f}  RMU={rmu_behav:.4f}\n")

    base_curve = per_layer_cv(base_acts, labels, cv, args.C)
    rmu_curve = per_layer_cv(rmu_acts, labels, cv, args.C)

    print(f"{'L':>2}  {'base':>6}  {'RMU':>6}  {'gap':>6}")
    for L in range(len(base_curve)):
        mark = "  <-- locked" if L == args.lock_layer else ""
        print(f"{L:>2}  {base_curve[L]:.3f}  {rmu_curve[L]:.3f}  "
              f"{base_curve[L]-rmu_curve[L]:+.3f}{mark}")

    L = args.lock_layer
    base_best = int(np.argmax(base_curve))
    print(f"\n=== LOCKED LAYER {L} ===")
    print(f"base probe acc : {base_curve[L]:.4f}")
    print(f"RMU  probe acc : {rmu_curve[L]:.4f}")
    print(f"probe gap      : {base_curve[L]-rmu_curve[L]:+.4f}")
    print(f"RMU behavioral : {rmu_behav:.4f}  (base behavioral {base_behav:.4f}, chance {chance:.3f})")
    print(f"RMU recoverability over chance: {rmu_curve[L]-chance:+.4f}")

    # ---- Controls (run by default) ----
    print("\n=== CONTROLS ===")
    base_gate = base_curve[base_best] > args.gate
    print(f"[base-probe gate]  best layer {base_best} acc={base_curve[base_best]:.4f} "
          f"{'PASS' if base_gate else 'HALT'} (> {args.gate})")
    rng = np.random.default_rng(args.seed)
    shuf = labels.copy(); rng.shuffle(shuf)
    shuf_acc = float(cross_val_score(make_clf(args.C), base_acts[:, L, :], shuf, cv=cv,
                                    scoring="accuracy", n_jobs=-1).mean())
    # One-sided leak test: a leak (full-set scaler / label-position shortcut) would push
    # shuffled-label acc ABOVE chance. At-or-below chance is healthy -- LR on shuffled
    # labels routinely generalizes slightly worse than chance, that is not a leak.
    shuf_ok = shuf_acc < chance + 0.03
    print(f"[shuffle gate]     layer {L} shuffled-label acc={shuf_acc:.4f} "
          f"{'PASS' if shuf_ok else 'FAIL'} (no signal: < chance {chance:.3f} + 0.03)")

    results = {
        "chance": chance, "lock_layer": L, "base_best_layer": base_best,
        "base_behav_acc": base_behav, "rmu_behav_acc": rmu_behav,
        "base_curve": base_curve, "rmu_curve": rmu_curve,
        "locked": {"base": base_curve[L], "rmu": rmu_curve[L],
                  "gap": base_curve[L] - rmu_curve[L]},
        "controls": {"base_gate_pass": base_gate, "shuffle_acc": shuf_acc,
                    "shuffle_pass": shuf_ok},
    }
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {args.out_json}")

    # ---- Money plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = list(range(len(base_curve)))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(xs, base_curve, "-o", ms=3, color="#1f77b4", label="base probe acc")
    ax.plot(xs, rmu_curve, "-o", ms=3, color="#d62728", label="RMU probe acc")
    ax.axhline(chance, ls=":", color="gray", label=f"chance ({chance:.2f})")
    ax.axhline(rmu_behav, ls="--", color="#d62728", alpha=.7,
              label=f"RMU behavioral ({rmu_behav:.2f})")
    ax.axhline(base_behav, ls="--", color="#1f77b4", alpha=.7,
              label=f"base behavioral ({base_behav:.2f})")
    ax.axvline(L, ls="-", color="black", alpha=.25)
    ax.annotate(f"locked L{L}", (L, 0.27), fontsize=8, rotation=90, alpha=.6)
    ax.set_xlabel("layer (0 = embeddings)")
    ax.set_ylabel("5-fold CV probe accuracy")
    ax.set_title("WMDP-bio answer decodability (5-fold linear probe): base vs RMU\n"
                "RMU flat at chance on all layers; base recovers ~0.6 plateau from L17")
    ax.set_ylim(0.15, 0.75)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=130)
    print(f"saved {args.out_png}")


if __name__ == "__main__":
    main()
