"""Readout capacity sweep on the knows-it (unfiltered) last-token (':') activations only.

Bounds whether the ~0.40 / +0.14 readout ceiling is the MODEL's ceiling or just one probe's.
For each capacity {linear, MLP-64, MLP-256, MLP-1024, MLP-(256,256)} computes the per-layer
5-fold CV signal = real - own-shuffle on unfiltered's last-token acts, and reports the
max-over-layers signal. If none exceeds ~+0.14, 0.40 is the readout ceiling for this model.

Usage: python capacity_sweep.py
"""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

D = "/workspace/CB_activations"
d = np.load(f"{D}/unfiltered_wmdp_bio.npz", allow_pickle=True)
acts = d["acts"].astype(np.float32); y = d["labels"]
nL = acts.shape[1]; chance = 1.0 / len(np.unique(y))
cv = StratifiedKFold(5, shuffle=True, random_state=0)
rng = np.random.default_rng(0); ysh = y.copy(); rng.shuffle(ysh)


def mk(name):
    if name == "linear":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    spec = {"mlp64": (64,), "mlp256": (256,), "mlp1024": (1024,), "mlp256x2": (256, 256)}[name]
    return make_pipeline(StandardScaler(),
                         MLPClassifier(hidden_layer_sizes=spec, alpha=1e-3, max_iter=300,
                                       early_stopping=True, n_iter_no_change=15,
                                       validation_fraction=0.1, random_state=0))


PROBES = ["linear", "mlp64", "mlp256", "mlp1024", "mlp256x2"]
print(f"knows-it readout capacity sweep | N={len(y)} layers={nL} chance={chance:.3f}\n")
out = {}
for name in PROBES:
    real = [float(cross_val_score(mk(name), acts[:, L, :], y, cv=cv, scoring="accuracy", n_jobs=5).mean())
            for L in range(nL)]
    shuf = [float(cross_val_score(mk(name), acts[:, L, :], ysh, cv=cv, scoring="accuracy", n_jobs=5).mean())
            for L in range(nL)]
    sig = [r - s for r, s in zip(real, shuf)]
    bL = int(np.argmax(sig))
    out[name] = {"real": real, "shuf": shuf, "max_signal": max(sig), "best_layer": bL,
                 "real_at_best": real[bL], "shuf_at_best": shuf[bL]}
    print(f"{name:>9}: max signal = {max(sig):+.4f} @L{bL}  (real {real[bL]:.3f} / shuf {shuf[bL]:.3f})")

best = max(out, key=lambda k: out[k]["max_signal"])
print(f"\nbest capacity: {best}  max_signal={out[best]['max_signal']:+.4f}")
print(f"linear baseline max_signal={out['linear']['max_signal']:+.4f}")
print(f"=> any capacity lift knows-it readout signal above +0.14? "
      f"{'YES' if out[best]['max_signal'] > 0.14 else 'NO'}")
json.dump(out, open(f"{D}/capacity_sweep_unfiltered.json", "w"), indent=2)
print(f"saved {D}/capacity_sweep_unfiltered.json")
