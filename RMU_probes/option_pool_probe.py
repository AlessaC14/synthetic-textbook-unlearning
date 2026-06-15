"""Step 3b -- OPTION-SPAN pool probe (linear + MLP), base vs RMU, real-vs-shuffle.

Sharp version of the "answer content lives at the option positions" hypothesis: pool over ONLY
the A./B./C./D. option-line tokens (boilerplate + question excluded), so the positive control is
far less diluted than the full-sequence mean-pool (Entry 5, which dragged base to 0.34). If RMU
hid the answer at the option positions while only suppressing the ':' readout, THIS is where it
shows up. Reads the permanent all-positions cache (activations/<tag>_allpos/).

Protocol unchanged: train-fold-only scaler, stratified 5-fold, seed 0, locked L22 + full curve.
Signal = real - own-shuffle (raw acc meaningless in p>>n). base must clear its gate.

Usage:
  python option_pool_probe.py
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
    ap.add_argument("--lock_layer", type=int, default=22)
    ap.add_argument("--gate", type=float, default=0.05, help="min base real-minus-shuffle signal")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_json", default="/workspace/activations/optionpool_rmu_vs_base.json")
    ap.add_argument("--out_png", default="/workspace/activations/optionpool_rmu_vs_base.png")
    args = ap.parse_args()

    mb, mr = load_meta("base"), load_meta("rmu")
    y = mb["labels"]
    assert np.array_equal(y, mr["labels"])
    base_behav, rmu_behav = float(mb["behav_acc"]), float(mr["behav_acc"])
    chance = 1.0 / len(np.unique(y))
    n_layers = int(mb["n_layers"])
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    rng = np.random.default_rng(args.seed)
    ysh = y.copy(); rng.shuffle(ysh)
    lin = lambda: make_linear(args.C)
    mlp = lambda: make_mlp(args.hidden, args.alpha, args.seed)

    print(f"OPTION-SPAN ({args.span}) pool | N={len(y)} layers={n_layers} chance={chance:.3f}")
    print(f"behavioral: base={base_behav:.4f} RMU={rmu_behav:.4f}\n")

    res = {k: [] for k in ["base_lin", "base_lin_shuf", "rmu_lin", "rmu_lin_shuf",
                        "base_mlp", "base_mlp_shuf", "rmu_mlp", "rmu_mlp_shuf"]}
    for L in range(n_layers):
        Xb = pooled_matrix("base", L, args.span, meta=mb)
        Xr = pooled_matrix("rmu", L, args.span, meta=mr)
        for tag, X in [("base", Xb), ("rmu", Xr)]:
            res[f"{tag}_lin"].append(float(cross_val_score(lin(), X, y, cv=cv, scoring="accuracy", n_jobs=-1).mean()))
            res[f"{tag}_lin_shuf"].append(float(cross_val_score(lin(), X, ysh, cv=cv, scoring="accuracy", n_jobs=-1).mean()))
            res[f"{tag}_mlp"].append(float(cross_val_score(mlp(), X, y, cv=cv, scoring="accuracy", n_jobs=-1).mean()))
            res[f"{tag}_mlp_shuf"].append(float(cross_val_score(mlp(), X, ysh, cv=cv, scoring="accuracy", n_jobs=-1).mean()))
        print(f"L{L:02d} done", flush=True)

    L = args.lock_layer
    def line(tag, real, shuf):
        return (f"{tag:>12}  L{L} real={real[L]:.3f} shuf={shuf[L]:.3f} "
                f"signal={real[L]-shuf[L]:+.3f}  bestL={int(np.argmax(real))}({max(real):.3f})")
    print(f"\n=== LOCKED LAYER {L}  (span={args.span}) ===")
    print(line("base linear", res["base_lin"], res["base_lin_shuf"]))
    print(line("RMU linear", res["rmu_lin"], res["rmu_lin_shuf"]))
    print(line("base MLP", res["base_mlp"], res["base_mlp_shuf"]))
    print(line("RMU MLP", res["rmu_mlp"], res["rmu_mlp_shuf"]))

    base_sig = max(res["base_lin"][L] - res["base_lin_shuf"][L], res["base_mlp"][L] - res["base_mlp_shuf"][L])
    base_gate = base_sig > args.gate
    rmu_sig_lin = max(r - s for r, s in zip(res["rmu_lin"], res["rmu_lin_shuf"]))
    rmu_sig_mlp = max(r - s for r, s in zip(res["rmu_mlp"], res["rmu_mlp_shuf"]))
    print(f"\n[base gate] best base signal @L{L} = {base_sig:+.3f} {'PASS' if base_gate else 'FAIL'} (>{args.gate})")
    print(f"[RMU] max over-shuffle signal over ALL layers: linear {rmu_sig_lin:+.3f}, MLP {rmu_sig_mlp:+.3f}")
    verdict = ("RMU answer SURFACES at option positions (real >> shuffle) -- headline: suppression is "
            "readout-localized, content survives at the option tokens"
            if max(rmu_sig_lin, rmu_sig_mlp) > 0.05 else
            "RMU at shuffle baseline even at option positions -- content absent there too; "
            "genuine-removal reading holds with a less-diluted positive control")
    print(f"VERDICT: {verdict}")

    out = {"span": args.span, "chance": chance, "lock_layer": L, "base_behav": base_behav,
        "rmu_behav": rmu_behav, "base_gate_pass": bool(base_gate), "verdict": verdict, **res}
    json.dump(out, open(args.out_json, "w"), indent=2)
    print(f"saved {args.out_json}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = list(range(n_layers))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, kind in zip(axes, ["lin", "mlp"]):
        ax.plot(xs, res[f"base_{kind}"], "-o", ms=3, color="#1f77b4", label="base real")
        ax.plot(xs, res[f"base_{kind}_shuf"], ":", color="#1f77b4", alpha=.6, label="base shuffle")
        ax.plot(xs, res[f"rmu_{kind}"], "-o", ms=3, color="#d62728", label="RMU real")
        ax.plot(xs, res[f"rmu_{kind}_shuf"], ":", color="#d62728", alpha=.6, label="RMU shuffle")
        ax.axhline(chance, ls="--", color="gray", alpha=.5)
        ax.axvline(L, ls="-", color="black", alpha=.2)
        ax.set_xlabel("layer (0 = embeddings)")
        ax.set_title(f"option-span pool, {'linear' if kind=='lin' else 'MLP'} probe")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("5-fold CV probe accuracy"); axes[0].set_ylim(0.15, 0.8)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"Option-span pool (mean over A./B./C./D. tokens only): real vs shuffled labels")
    fig.tight_layout(); fig.savefig(args.out_png, dpi=130)
    print(f"saved {args.out_png}")


if __name__ == "__main__":
    main()
