"""Step 3c -- LEARNED-ATTENTION probe (last cell of the token-aggregation axis), base vs RMU.

A modest attention head pools over ALL token positions of the prompt (incl. the ':' readout),
then a read head (linear, or small MLP) predicts the 4-way answer. Unlike fixed last-token /
mean / option pools, the probe LEARNS which positions to read. base should recover strongly by
attending to the readout position (proves the probe class works); the question is whether RMU has
ANY attendable position carrying the answer.

Capacity is deliberately constrained (single-vector attention scorer, weight decay, dropout, early
stopping) because in p>>n an overpowered attention head memorizes and then tests at noise. The
SHUFFLE control is load-bearing: a learned head can fit noise, so signal = real - own-shuffle at
every layer is the ONLY thing reported as recovery. real and shuffle rising together on RMU = noise.

Reads the permanent all-positions cache (activations/<tag>_allpos/) via allpos_utils.
Locked L22 headline + full per-layer curve, money-plot format.

Usage:
  python attn_probe.py                 # linear + mlp read heads
  python attn_probe.py --heads linear  # one head only
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

from allpos_utils import example_sequences, load_meta


class AttnProbe(nn.Module):
    """Single-vector additive attention pooling + read head. Modest by construction."""
    def __init__(self, d, n_classes=4, hidden=None, p_drop=0.1):
        super().__init__()
        self.score = nn.Linear(d, 1)               # one scalar score per position
        self.drop = nn.Dropout(p_drop)
        if hidden:
            self.head = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))
        else:
            self.head = nn.Linear(d, n_classes)

    def forward(self, x, mask):                    # x (B,T,d) fp32, mask (B,T) bool
        s = self.score(x).squeeze(-1)
        s = s.masked_fill(~mask, float("-inf"))
        a = torch.softmax(s, dim=1).unsqueeze(-1)  # (B,T,1)
        pooled = (a * x).sum(1)                     # (B,d)
        return self.head(self.drop(pooled))


def build_layer_tensor(tag, L, meta, Tmax, device):
    """Padded (N,Tmax,d) fp16 GPU tensor + (N,Tmax) bool mask over ALL real tokens."""
    seqs = example_sequences(tag, L, span="all", meta=meta)
    N, d = len(seqs), seqs[0].shape[1]
    X = torch.zeros((N, Tmax, d), dtype=torch.float16)
    mask = torch.zeros((N, Tmax), dtype=torch.bool)
    for i, s in enumerate(seqs):
        if len(s) > Tmax:
            s = s[-Tmax:]          # keep the TAIL: preserves options + the ':' readout
        t = len(s)
        X[i, :t] = torch.from_numpy(s)
        mask[i, :t] = True
    return X.to(device), mask.to(device)


def train_eval(X, mask, y, tr, te, hidden, device, seed, epochs=60, patience=8,
            lr=1e-3, wd=1e-2, bs=256):
    """Train on tr indices (with inner val for early stop), return test acc on te."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    # inner stratified-ish val split (10%) for early stopping
    tr = tr[torch.randperm(len(tr), generator=g).numpy()]
    n_val = max(1, int(0.1 * len(tr)))
    val, trn = tr[:n_val], tr[n_val:]
    # standardize per-feature on TRAIN real tokens only
    flat = X[trn][mask[trn]].float()
    mu, sd = flat.mean(0), flat.std(0) + 1e-5
    d = X.shape[2]
    model = AttnProbe(d, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    yt = torch.from_numpy(y).long().to(device)

    def run(idx, train):
        model.train(train)
        tot_correct, tot = 0, 0
        order = idx[torch.randperm(len(idx), generator=g).numpy()] if train else idx
        for k in range(0, len(order), bs):
            b = order[k:k + bs]
            xb = ((X[b].float() - mu) / sd)
            logits = model(xb, mask[b])
            loss = lossf(logits, yt[b])
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            tot_correct += (logits.argmax(1) == yt[b]).sum().item(); tot += len(b)
        return tot_correct / tot

    best_val, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        run(trn, True)
        with torch.no_grad():
            v = run(val, False)
        if v > best_val:
            best_val, best_state, bad = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    with torch.no_grad():
        return run(te, False)


def curve(X, mask, y, cv, hidden, device, seed):
    accs = []
    for tr, te in cv.split(np.zeros(len(y)), y):
        accs.append(train_eval(X, mask, y, tr, te, hidden, device, seed))
    return float(np.mean(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", default="both", choices=["both", "linear", "mlp"])
    ap.add_argument("--mlp_hidden", type=int, default=64)
    ap.add_argument("--lock_layer", type=int, default=22)
    ap.add_argument("--gate", type=float, default=0.05)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--Tmax", type=int, default=512, help="pad/clip length; keeps tail incl ':'")
    ap.add_argument("--out_json", default="/workspace/activations/attn_rmu_vs_base.json")
    ap.add_argument("--out_png", default="/workspace/activations/attn_rmu_vs_base.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mb, mr = load_meta("base"), load_meta("rmu")
    y = mb["labels"].astype(np.int64)
    assert np.array_equal(y, mr["labels"])
    base_behav, rmu_behav = float(mb["behav_acc"]), float(mr["behav_acc"])
    chance = 1.0 / len(np.unique(y))
    n_layers = int(mb["n_layers"])
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    rng = np.random.default_rng(args.seed)
    ysh = y.copy(); rng.shuffle(ysh)
    heads = {"linear": None, "mlp": args.mlp_hidden}
    if args.heads != "both":
        heads = {args.heads: heads[args.heads]}

    # clip Tmax to the ':' tail: most prompts < p95=192; keep last Tmax tokens so readout survives
    maxlen = int(mb["lengths"].max())
    Tmax = min(args.Tmax, maxlen)
    print(f"ATTN probe | N={len(y)} layers={n_layers} chance={chance:.3f} Tmax={Tmax} "
          f"(maxlen={maxlen}) heads={list(heads)} device={device}")
    print(f"behavioral: base={base_behav:.4f} RMU={rmu_behav:.4f}\n")

    res = {f"{tag}_{h}{sfx}": [] for tag in ["base", "rmu"] for h in heads
        for sfx in ["", "_shuf"]}
    for L in range(n_layers):
        for tag, meta in [("base", mb), ("rmu", mr)]:
            X, mask = build_layer_tensor(tag, L, meta, Tmax, device)
            for h, hid in heads.items():
                res[f"{tag}_{h}"].append(curve(X, mask, y, cv, hid, device, args.seed))
                res[f"{tag}_{h}_shuf"].append(curve(X, mask, ysh, cv, hid, device, args.seed))
            del X, mask; torch.cuda.empty_cache()
        msg = "  ".join(f"{tag}_{h} {res[f'{tag}_{h}'][-1]:.3f}/{res[f'{tag}_{h}_shuf'][-1]:.3f}"
                        for tag in ["base", "rmu"] for h in heads)
        print(f"L{L:02d}  {msg}", flush=True)

    L = args.lock_layer
    print(f"\n=== LOCKED LAYER {L} (real / shuffle / signal) ===")
    for tag in ["base", "rmu"]:
        for h in heads:
            r, s = res[f"{tag}_{h}"], res[f"{tag}_{h}_shuf"]
            print(f"{tag:>4} {h:>6}: real={r[L]:.3f} shuf={s[L]:.3f} signal={r[L]-s[L]:+.3f} "
                f"| bestL={int(np.argmax(np.array(r)-np.array(s)))} "
                f"maxsig={max(rr-ss for rr,ss in zip(r,s)):+.3f}")

    h0 = list(heads)[0]
    base_sig = [r - s for r, s in zip(res[f"base_{h0}"], res[f"base_{h0}_shuf"])]
    base_gate = max(base_sig) > args.gate
    rmu_max = max(max(r - s for r, s in zip(res[f"rmu_{h}"], res[f"rmu_{h}_shuf"])) for h in heads)
    print(f"\n[base gate] max base_{h0} signal = {max(base_sig):+.3f} {'PASS' if base_gate else 'FAIL'} (>{args.gate})")
    print(f"[RMU] max over-shuffle signal over ALL layers/heads = {rmu_max:+.3f}")
    verdict = ("RMU answer RECOVERED by learned attention (real >> shuffle) -- knowledge present, "
            "attendable position carries it"
            if rmu_max > 0.05 else
            "RMU stays at shuffle baseline under learned attention too -- no attendable position "
            "carries the answer; genuine-removal reading holds across the full aggregation axis")
    print(f"VERDICT: {verdict}")

    out = {"chance": chance, "lock_layer": L, "Tmax": Tmax, "base_behav": base_behav,
        "rmu_behav": rmu_behav, "base_gate_pass": bool(base_gate), "heads": list(heads),
        "verdict": verdict, **res}
    json.dump(out, open(args.out_json, "w"), indent=2)
    print(f"saved {args.out_json}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = list(range(n_layers))
    fig, axes = plt.subplots(1, len(heads), figsize=(6.5 * len(heads), 5.2), sharey=True, squeeze=False)
    for ax, h in zip(axes[0], heads):
        ax.plot(xs, res[f"base_{h}"], "-o", ms=3, color="#1f77b4", label="base real")
        ax.plot(xs, res[f"base_{h}_shuf"], ":", color="#1f77b4", alpha=.6, label="base shuffle")
        ax.plot(xs, res[f"rmu_{h}"], "-o", ms=3, color="#d62728", label="RMU real")
        ax.plot(xs, res[f"rmu_{h}_shuf"], ":", color="#d62728", alpha=.6, label="RMU shuffle")
        ax.axhline(chance, ls="--", color="gray", alpha=.5)
        ax.axvline(L, ls="-", color="black", alpha=.2)
        ax.set_xlabel("layer (0 = embeddings)")
        ax.set_title(f"learned-attention, {h} read-head")
        ax.grid(alpha=.2)
    axes[0][0].set_ylabel("5-fold CV probe accuracy"); axes[0][0].set_ylim(0.15, 0.8)
    axes[0][0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Learned-attention probe (pool over all positions): real vs shuffled labels")
    fig.tight_layout(); fig.savefig(args.out_png, dpi=130)
    print(f"saved {args.out_png}")


if __name__ == "__main__":
    main()
