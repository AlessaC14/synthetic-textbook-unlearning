"""Pathology / sanity checks on the flat-at-chance RMU linear probe.

Before interpreting "RMU probe = chance on all layers" as "no linear signal", rule out
degenerate causes:

  (1) train-vs-test acc + convergence at layers 7/16/22, same CV folds, for base, RMU,
      and a shuffled-label control. NB: with p=4096 features >> n~1018 train rows, a linear
      model fits ANY labeling perfectly, so train acc ~1.0 for all conditions and is NOT a
      useful signal detector. The decisive readout is TEST acc vs the shuffled control:
      - RMU test ~ shuffled test (~chance) while base test generalizes -> signal genuinely
        absent on the linear axis (clean reading; proceed).
      - RMU test >> shuffled test but < base -> partial; RMU test ~ base -> recovered.
  (2) activation-health: per-example L2 norm distribution (mean/std/min/max) at L7/16/22,
      RMU vs base -- rules out collapsed/degenerate activations as a trivial failure cause.
  (3) early-layer agreement: base and RMU probe curves (and the raw acts) should be
      identical upstream of RMU's intervention region (layers 5-7) and only diverge after.

Usage:
  python pathology_check.py
"""
import argparse
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_clf(C):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=C))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/workspace/activations/base_wmdp_bio.npz")
    ap.add_argument("--rmu", default="/workspace/activations/rmu_wmdp_bio.npz")
    ap.add_argument("--layers", type=int, nargs="+", default=[7, 16, 22])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    db, dr = np.load(args.base, allow_pickle=True), np.load(args.rmu, allow_pickle=True)
    base, rmu = db["acts"].astype(np.float32), dr["acts"].astype(np.float32)
    labels = db["labels"]
    assert np.array_equal(labels, dr["labels"])
    chance = 1.0 / len(np.unique(labels))
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    # ---- (1) train vs test acc + convergence: base, RMU, shuffled control ----
    rng = np.random.default_rng(args.seed)
    shuf = labels.copy(); rng.shuffle(shuf)
    conds = [("base", base, labels), ("rmu", rmu, labels), ("rmu-shuf", rmu, shuf)]
    print(f"=== (1) train-vs-test + convergence (chance={chance:.3f}, p={base.shape[2]} >> n~{int(len(labels)*4/5)}) ===")
    print(f"{'L':>2}  {'cond':>9}  {'train':>6}  {'test':>6}  {'gap':>6}  converged?")
    for L in args.layers:
        for name, A, y in conds:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                res = cross_validate(make_clf(args.C), A[:, L, :], y, cv=cv,
                                    scoring="accuracy", return_train_score=True, n_jobs=1)
                conv_warns = [x for x in w if issubclass(x.category, ConvergenceWarning)]
            clf = make_clf(args.C).fit(A[:, L, :], y)
            n_iter = int(np.max(clf[-1].n_iter_))
            tr, te = res["train_score"].mean(), res["test_score"].mean()
            conv = f"yes (n_iter={n_iter})" if not conv_warns and n_iter < clf[-1].max_iter \
                else f"NO ({len(conv_warns)}w, n_iter={n_iter})"
            print(f"{L:>2}  {name:>9}  {tr:.3f}  {te:.3f}  {tr-te:+.3f}  {conv}")
        print()

    # ---- (2) activation-health: per-example L2 norm ----
    print(f"\n=== (2) per-example activation L2-norm (mean/std/min/max) ===")
    print(f"{'L':>2}  {'model':>5}  {'mean':>8}  {'std':>8}  {'min':>8}  {'max':>8}")
    for L in args.layers:
        for name, A in [("base", base), ("rmu", rmu)]:
            n = np.linalg.norm(A[:, L, :], axis=1)
            print(f"{L:>2}  {name:>5}  {n.mean():8.2f}  {n.std():8.2f}  {n.min():8.2f}  {n.max():8.2f}")

    # ---- (3) early-layer agreement: raw acts diff per layer ----
    print(f"\n=== (3) base vs RMU raw-activation divergence per layer ===")
    print("(RMU edits weights ~L5-7; layers upstream should be identical)")
    print(f"{'L':>2}  {'max|Δ|':>10}  {'mean|Δ|':>10}  identical?")
    for L in range(base.shape[1]):
        d = np.abs(base[:, L, :] - rmu[:, L, :])
        mx, mn = float(d.max()), float(d.mean())
        flag = "  <-- identical" if mx == 0.0 else ""
        # only print early layers in full + the probed layers, summarise the rest
        if L <= 9 or L in args.layers:
            print(f"{L:>2}  {mx:10.4f}  {mn:10.6f}{flag}")
    first_div = next((L for L in range(base.shape[1])
                    if np.abs(base[:, L, :] - rmu[:, L, :]).max() > 0), None)
    print(f"first layer where base and RMU acts differ: L{first_div}")


if __name__ == "__main__":
    main()
