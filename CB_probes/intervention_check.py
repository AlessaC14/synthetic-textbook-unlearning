"""Stage 3 control: per-layer activation divergence, knows-it vs cb (and vs never-knew).

CB is applied as a perturbation of the UNFILTERED annealed checkpoint (merged LoRA at layers
5/10/15/20/25/30). So unfiltered-vs-cb should be ~identical in the earliest layers and diverge
at/after the first CB adapter layer -- a clean "intervention onset" check (analog of RMU's
bit-identical-through-L5). The never-knew model is a SEPARATE training run, not a perturbation
of unfiltered, so it differs everywhere -- included only as contrast, NOT an onset check.

Operates on the last-token (':') npz caches. Reports per layer: max abs diff and mean per-example
cosine similarity between unfiltered and {cb, strongfilter}.

Usage: python intervention_check.py
"""
import json

import numpy as np

D = "/workspace/CB_activations"
ref = np.load(f"{D}/unfiltered_wmdp_bio.npz", allow_pickle=True)["acts"].astype(np.float32)
cb = np.load(f"{D}/cb_wmdp_bio.npz", allow_pickle=True)["acts"].astype(np.float32)
sf = np.load(f"{D}/strongfilter_wmdp_bio.npz", allow_pickle=True)["acts"].astype(np.float32)
nL = ref.shape[1]


def cos(a, b):
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8
    return (num / den).mean()


print(f"layers={nL}  N={ref.shape[0]}")
print(f"{'L':>2}  {'unf-vs-cb maxΔ':>14} {'cos':>7}   {'unf-vs-sf maxΔ':>14} {'cos':>7}")
out = {"unf_cb_maxabs": [], "unf_cb_cos": [], "unf_sf_maxabs": [], "unf_sf_cos": []}
for L in range(nL):
    d_cb = np.abs(ref[:, L] - cb[:, L]); d_sf = np.abs(ref[:, L] - sf[:, L])
    c_cb = float(cos(ref[:, L], cb[:, L])); c_sf = float(cos(ref[:, L], sf[:, L]))
    out["unf_cb_maxabs"].append(float(d_cb.max())); out["unf_cb_cos"].append(c_cb)
    out["unf_sf_maxabs"].append(float(d_sf.max())); out["unf_sf_cos"].append(c_sf)
    print(f"{L:>2}  {d_cb.max():>14.5g} {c_cb:>7.4f}   {d_sf.max():>14.5g} {c_sf:>7.4f}")

# onset = first layer where unf-vs-cb is no longer ~identical
onset = next((L for L in range(nL) if out["unf_cb_maxabs"][L] > 1e-3), None)
print(f"\nunfiltered-vs-cb divergence onset (first layer maxΔ>1e-3): L{onset}")
print(f"CB adapter layers per paper: 5,10,15,20,25,30 (embeddings layer index 0 -> transformer L = hidden_state index)")
json.dump({"divergence_onset_layer": onset, **out}, open(f"{D}/intervention_check.json", "w"), indent=2)
print(f"saved {D}/intervention_check.json")
