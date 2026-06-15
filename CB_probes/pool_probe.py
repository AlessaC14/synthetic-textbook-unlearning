"""Stage 2 token-aggregation axis: POOLED probe (full-mean or option-span), three-model.

Pools the all-positions cache over a span (mean over ALL real tokens, or over only the
A./B./C./D. option lines) then runs linear + MLP probes on all three models. Signal =
real - own-shuffle per layer (raw acc meaningless in p>>n). Locked to knows-it best layer;
full per-layer curve is the deliverable. Reuses the permanent ragged cache via allpos_utils.

Usage:
  python pool_probe.py --span all       # full-sequence mean-pool
  python pool_probe.py --span options   # option-span pool (less diluted)
"""
import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from allpos_utils import load_meta, pooled_matrix

MODELS = ["unfiltered", "cb", "strongfilter"]
COLORS = {"unfiltered": "#1f77b4", "cb": "#d62728", "strongfilter": "#2ca02c"}


def make_linear(C):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=C))


def make_mlp(hidden, alpha, seed):
    return make_pipeline(StandardScaler(),
                         MLPClassifier(hidden_layer_sizes=(hidden,), alpha=alpha, max_iter=300,
                                       early_stopping=True, n_iter_no_change=15,
                                       validation_fraction=0.1, random_state=seed))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--span", default="options", choices=["options", "question", "all"])
    ap.add_argument("--lock_layer", type=int, default=-1)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_json", default="")
    ap.add_argument("--out_png", default="")
    ap.add_argument("--no_mlp", action="store_true",
                    help="linear cells only (pooled-MLP is slow on chance-level features and only "
                         "corroborating; last-token MLP already showed capacity adds nothing)")
    args = ap.parse_args()
    tagname = {"all": "meanpool", "options": "optionpool", "question": "qpool"}[args.span]
    out_json = args.out_json or f"/workspace/CB_activations/{tagname}_cb_vs_refs.json"
    out_png = args.out_png or f"/workspace/CB_activations/{tagname}_cb_vs_refs.png"

    meta = {m: load_meta(m) for m in MODELS}
    y = meta["unfiltered"]["labels"]
    for m in MODELS:
        assert np.array_equal(y, meta[m]["labels"]), f"{m} label order differs"
    behav = {m: float(meta[m]["behav_acc"]) for m in MODELS}
    chance = 1.0 / len(np.unique(y))
    n_layers = int(meta["unfiltered"]["n_layers"])
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    rng = np.random.default_rng(args.seed); ysh = y.copy(); rng.shuffle(ysh)
    lin = lambda: make_linear(args.C)
    mlp = lambda: make_mlp(args.hidden, args.alpha, args.seed)

    print(f"POOL span={args.span} | N={len(y)} layers={n_layers} chance={chance:.3f}")
    print("behavioral: " + "  ".join(f"{m}={behav[m]:.4f}" for m in MODELS) + "\n")

    res = {f"{m}_{k}": [] for m in MODELS for k in ["lin", "lin_shuf", "mlp", "mlp_shuf"]}
    for L in range(n_layers):
        for m in MODELS:
            X = pooled_matrix(m, L, args.span, meta=meta[m])
            res[f"{m}_lin"].append(float(cross_val_score(lin(), X, y, cv=cv, scoring="accuracy", n_jobs=5).mean()))
            res[f"{m}_lin_shuf"].append(float(cross_val_score(lin(), X, ysh, cv=cv, scoring="accuracy", n_jobs=5).mean()))
            if args.no_mlp:
                res[f"{m}_mlp"].append(float("nan")); res[f"{m}_mlp_shuf"].append(float("nan"))
            else:
                res[f"{m}_mlp"].append(float(cross_val_score(mlp(), X, y, cv=cv, scoring="accuracy", n_jobs=5).mean()))
                res[f"{m}_mlp_shuf"].append(float(cross_val_score(mlp(), X, ysh, cv=cv, scoring="accuracy", n_jobs=5).mean()))
        print(f"L{L:02d} done", flush=True)

    L = int(np.argmax(res["unfiltered_lin"])) if args.lock_layer < 0 else args.lock_layer
    print(f"\n=== LOCKED LAYER {L} (span={args.span}; knows-it linear best) ===")
    for kind in ["lin", "mlp"]:
        print(f"-- {kind} --")
        for m in MODELS:
            r, s = res[f"{m}_{kind}"], res[f"{m}_{kind}_shuf"]
            maxsig = max(rr - ss for rr, ss in zip(r, s))
            print(f"  {m:>12}: real={r[L]:.3f} shuf={s[L]:.3f} signal={r[L]-s[L]:+.3f} "
                  f"| bestL={int(np.argmax(np.array(r)-np.array(s)))} maxsig={maxsig:+.3f}")

    out = {"span": args.span, "chance": chance, "lock_layer": L, "behav_acc": behav, **res}
    json.dump(out, open(out_json, "w"), indent=2)
    print(f"saved {out_json}")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = list(range(n_layers))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, kind in zip(axes, ["lin", "mlp"]):
        for m in MODELS:
            ax.plot(xs, res[f"{m}_{kind}"], "-o", ms=3, color=COLORS[m], label=f"{m} real")
            ax.plot(xs, res[f"{m}_{kind}_shuf"], ":", color=COLORS[m], alpha=.5)
        ax.axhline(chance, ls="--", color="gray", alpha=.5)
        ax.axvline(L, ls="-", color="black", alpha=.2)
        ax.set_xlabel("layer (0 = embeddings)")
        ax.set_title(f"{args.span} pool, {'linear' if kind=='lin' else 'MLP'} (dotted = shuffle)")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("5-fold CV probe accuracy"); axes[0].set_ylim(0.15, 0.75)
    axes[0].legend(loc="upper left", fontsize=7)
    fig.suptitle(f"{args.span}-pool probe: knows-it / cb / never-knew, real vs shuffle")
    fig.tight_layout(); fig.savefig(out_png, dpi=130)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
