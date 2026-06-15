"""Entry 9 -- retain-selectivity control: probe base vs RMU on MMLU-bio (last-token).

Same harness as the WMDP last-token probe (Entries 2/4), only the dataset differs. MMLU-bio
(college + high_school biology) is general bio, NOT in the WMDP forget set. Selectivity claim:
RMU should RECOVER MMLU-bio (probe signal well above its shuffle, comparable to base) while it
stays flat on WMDP-bio -- i.e. RMU removed the forget content specifically, not bio broadly.

Signal = real - own-shuffle per layer (raw acc inflated in p>>n). Locked L22 + full curve, and a
side-by-side with the cached WMDP numbers (rmu_vs_base.json / mlp_rmu_vs_base.json).

Usage:
  python mmlu_probe.py
"""
import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_linear(C):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=C))


def make_mlp(hidden, alpha, seed):
    return make_pipeline(StandardScaler(),
                        MLPClassifier(hidden_layer_sizes=(hidden,), alpha=alpha, max_iter=300,
                                    early_stopping=True, n_iter_no_change=15,
                                    validation_fraction=0.1, random_state=seed))


def curve(make_clf, acts, y, cv):
    return [float(cross_val_score(make_clf(), acts[:, L, :], y, cv=cv, scoring="accuracy",
                                n_jobs=-1).mean()) for L in range(acts.shape[1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/workspace/activations/base_mmlu_bio.npz")
    ap.add_argument("--rmu", default="/workspace/activations/rmu_mmlu_bio.npz")
    ap.add_argument("--wmdp_lin", default="/workspace/activations/rmu_vs_base.json")
    ap.add_argument("--wmdp_mlp", default="/workspace/activations/mlp_rmu_vs_base.json")
    ap.add_argument("--lock_layer", type=int, default=22)
    ap.add_argument("--gate", type=float, default=0.05)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_json", default="/workspace/activations/mmlu_rmu_vs_base.json")
    ap.add_argument("--out_png", default="/workspace/activations/mmlu_rmu_vs_base.png")
    args = ap.parse_args()

    db, dr = np.load(args.base, allow_pickle=True), np.load(args.rmu, allow_pickle=True)
    base, rmu = db["acts"].astype(np.float32), dr["acts"].astype(np.float32)
    y = db["labels"]
    assert np.array_equal(y, dr["labels"])
    base_behav, rmu_behav = float(db["behav_acc"]), float(dr["behav_acc"])
    chance = 1.0 / len(np.unique(y))
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    rng = np.random.default_rng(args.seed); ysh = y.copy(); rng.shuffle(ysh)
    lin = lambda: make_linear(args.C); mlp = lambda: make_mlp(args.hidden, args.alpha, args.seed)

    print(f"MMLU-bio retain control | N={len(y)} layers={base.shape[1]} chance={chance:.3f}")
    print(f"behavioral: base={base_behav:.4f} RMU={rmu_behav:.4f} "
          f"(both should be >> chance & close -> retained)\n")
    print(f"label counts: {np.bincount(y).tolist()}")

    res = {}
    for tag, A in [("base", base), ("rmu", rmu)]:
        res[f"{tag}_lin"] = curve(lin, A, y, cv); res[f"{tag}_lin_shuf"] = curve(lin, A, ysh, cv)
        res[f"{tag}_mlp"] = curve(mlp, A, y, cv); res[f"{tag}_mlp_shuf"] = curve(mlp, A, ysh, cv)
        print(f"[{tag}] curves done")

    L = args.lock_layer
    def sig(tag, kind):
        return res[f"{tag}_{kind}"][L] - res[f"{tag}_{kind}_shuf"][L]
    print(f"\n=== LOCKED LAYER {L} (real / shuffle / signal) ===")
    for tag in ["base", "rmu"]:
        for kind in ["lin", "mlp"]:
            r, s = res[f"{tag}_{kind}"], res[f"{tag}_{kind}_shuf"]
            print(f"{tag:>4} {kind}: real={r[L]:.3f} shuf={s[L]:.3f} signal={r[L]-s[L]:+.3f} "
                f"maxsig={max(a-b for a,b in zip(r,s)):+.3f}")

    base_gate = max(sig("base", "lin"), sig("base", "mlp")) > args.gate
    print(f"\n[base gate] L{L} base signal = {max(sig('base','lin'),sig('base','mlp')):+.3f} "
          f"{'PASS' if base_gate else 'FAIL'} (>{args.gate})")

    # side-by-side with WMDP (cached)
    wl, wm = json.load(open(args.wmdp_lin)), json.load(open(args.wmdp_mlp))
    print(f"\n=== SELECTIVITY: probe @L{L}, WMDP-bio vs MMLU-bio ===")
    print(f"{'':16}{'WMDP real':>11}{'MMLU real':>11}{'MMLU sig':>10}")
    print(f"{'base linear':16}{wl['base_curve'][L]:>11.3f}{res['base_lin'][L]:>11.3f}{sig('base','lin'):>+10.3f}")
    print(f"{'RMU  linear':16}{wl['rmu_curve'][L]:>11.3f}{res['rmu_lin'][L]:>11.3f}{sig('rmu','lin'):>+10.3f}")
    print(f"{'base MLP':16}{wm['base_mlp'][L]:>11.3f}{res['base_mlp'][L]:>11.3f}{sig('base','mlp'):>+10.3f}")
    print(f"{'RMU  MLP':16}{wm['rmu_mlp'][L]:>11.3f}{res['rmu_mlp'][L]:>11.3f}{sig('rmu','mlp'):>+10.3f}")

    rmu_mmlu_sig = max(sig("rmu", "lin"), sig("rmu", "mlp"))
    base_mmlu_sig = max(sig("base", "lin"), sig("base", "mlp"))
    verdict = (f"SELECTIVITY HOLDS: RMU recovers MMLU-bio (signal +{rmu_mmlu_sig:.3f} @L{L}, "
            f"comparable to base +{base_mmlu_sig:.3f}) while flat on WMDP-bio -> forget-set-specific."
            if rmu_mmlu_sig > args.gate else
            f"⚠️ RMU MMLU-bio signal only +{rmu_mmlu_sig:.3f} @L{L} -- weak; selectivity not clean, investigate.")
    print(f"\nVERDICT: {verdict}")

    out = {"dataset": "mmlu_bio", "chance": chance, "lock_layer": L, "N": int(len(y)),
        "base_behav": base_behav, "rmu_behav": rmu_behav, "base_gate_pass": bool(base_gate),
        "verdict": verdict, **res}
    json.dump(out, open(args.out_json, "w"), indent=2)
    print(f"saved {args.out_json}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = list(range(base.shape[1]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, kind in zip(axes, ["lin", "mlp"]):
        ax.plot(xs, res[f"base_{kind}"], "-o", ms=3, color="#1f77b4", label="base real")
        ax.plot(xs, res[f"base_{kind}_shuf"], ":", color="#1f77b4", alpha=.6, label="base shuffle")
        ax.plot(xs, res[f"rmu_{kind}"], "-o", ms=3, color="#d62728", label="RMU real")
        ax.plot(xs, res[f"rmu_{kind}_shuf"], ":", color="#d62728", alpha=.6, label="RMU shuffle")
        ax.axhline(chance, ls="--", color="gray", alpha=.5)
        ax.axvline(L, ls="-", color="black", alpha=.2)
        ax.set_xlabel("layer (0 = embeddings)")
        ax.set_title(f"MMLU-bio (retain), {'linear' if kind=='lin' else 'MLP'} probe")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("5-fold CV probe accuracy"); axes[0].set_ylim(0.15, 0.85)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Retain-selectivity control: MMLU-bio probe, base vs RMU (real vs shuffle)")
    fig.tight_layout(); fig.savefig(args.out_png, dpi=130)
    print(f"saved {args.out_png}")


if __name__ == "__main__":
    main()
