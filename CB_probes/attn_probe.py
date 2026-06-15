"""Stage 2 token-aggregation axis: LEARNED-ATTENTION probe, three-model.

A modest attention head pools over ALL token positions (incl. the ':' readout), then a read
head (linear, or small MLP) predicts the 4-way answer -- the probe LEARNS which positions to
read. This is the most flexible aggregator and the strongest positive control: the knows-it
model should recover by attending to the readout, establishing the best achievable ceiling.
The question for CB: is there ANY attendable position carrying the answer?

Capacity deliberately constrained (single-vector scorer, weight decay, dropout, early stop):
in p>>n an overpowered head memorizes then tests at noise. The SHUFFLE control is load-bearing:
signal = real - own-shuffle per layer is the ONLY thing reported as recovery.

Reads the permanent all-positions cache via allpos_utils. Three models: unfiltered (knows-it),
cb (target), strongfilter (never-knew).

Usage:
  python attn_probe.py                 # linear + mlp read heads
  python attn_probe.py --heads linear
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

from allpos_utils import example_sequences, load_meta

MODELS = ["unfiltered", "cb", "strongfilter"]
COLORS = {"unfiltered": "#1f77b4", "cb": "#d62728", "strongfilter": "#2ca02c"}


class AttnProbe(nn.Module):
    def __init__(self, d, n_classes=4, hidden=None, p_drop=0.1):
        super().__init__()
        self.score = nn.Linear(d, 1)
        self.drop = nn.Dropout(p_drop)
        if hidden:
            self.head = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))
        else:
            self.head = nn.Linear(d, n_classes)

    def forward(self, x, mask):
        s = self.score(x).squeeze(-1)
        s = s.masked_fill(~mask, float("-inf"))
        a = torch.softmax(s, dim=1).unsqueeze(-1)
        pooled = (a * x).sum(1)
        return self.head(self.drop(pooled))


def build_layer_tensor(tag, L, meta, Tmax, device):
    seqs = example_sequences(tag, L, span="all", meta=meta)
    N, d = len(seqs), seqs[0].shape[1]
    X = torch.zeros((N, Tmax, d), dtype=torch.float16)
    mask = torch.zeros((N, Tmax), dtype=torch.bool)
    for i, s in enumerate(seqs):
        if len(s) > Tmax:
            s = s[-Tmax:]  # keep the TAIL: preserves options + the ':' readout
        t = len(s)
        X[i, :t] = torch.from_numpy(s)
        mask[i, :t] = True
    return X.to(device), mask.to(device)


def train_eval(X, mask, y, tr, te, hidden, device, seed, epochs=60, patience=8,
               lr=1e-3, wd=1e-2, bs=256, p_drop=0.1):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    tr = tr[torch.randperm(len(tr), generator=g).numpy()]
    n_val = max(1, int(0.1 * len(tr)))
    val, trn = tr[:n_val], tr[n_val:]
    flat = X[trn][mask[trn]].float()
    mu, sd = flat.mean(0), flat.std(0) + 1e-5
    d = X.shape[2]
    model = AttnProbe(d, hidden=hidden, p_drop=p_drop).to(device)
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


def curve(X, mask, y, cv, hidden, device, seed, **kw):
    return float(np.mean([train_eval(X, mask, y, tr, te, hidden, device, seed, **kw)
                          for tr, te in cv.split(np.zeros(len(y)), y)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", default="both", choices=["both", "linear", "mlp"])
    ap.add_argument("--mlp_hidden", type=int, default=64)
    ap.add_argument("--lock_layer", type=int, default=-1)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--Tmax", type=int, default=512)
    ap.add_argument("--models", default="unfiltered,cb,strongfilter",
                    help="comma list subset of models to run")
    ap.add_argument("--wd", type=float, default=1e-2, help="weight decay (relaxed: 1e-4)")
    ap.add_argument("--dropout", type=float, default=0.1, help="attn-head dropout (relaxed: 0.0)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--out_json", default="/workspace/CB_activations/attn_cb_vs_refs.json")
    ap.add_argument("--out_png", default="/workspace/CB_activations/attn_cb_vs_refs.png")
    args = ap.parse_args()
    train_kw = dict(wd=args.wd, p_drop=args.dropout, epochs=args.epochs, patience=args.patience)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = [m.strip() for m in args.models.split(",")]
    meta = {m: load_meta(m) for m in models}
    y = meta[models[0]]["labels"].astype(np.int64)
    for m in models:
        assert np.array_equal(y, meta[m]["labels"].astype(np.int64))
    behav = {m: float(meta[m]["behav_acc"]) for m in models}
    chance = 1.0 / len(np.unique(y))
    n_layers = int(meta["unfiltered"]["n_layers"])
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    rng = np.random.default_rng(args.seed); ysh = y.copy(); rng.shuffle(ysh)
    heads = {"linear": None, "mlp": args.mlp_hidden}
    if args.heads != "both":
        heads = {args.heads: heads[args.heads]}

    maxlen = int(meta["unfiltered"]["lengths"].max())
    Tmax = min(args.Tmax, maxlen)
    print(f"ATTN | N={len(y)} layers={n_layers} chance={chance:.3f} Tmax={Tmax} (maxlen={maxlen}) "
          f"heads={list(heads)} device={device}")
    print(f"reg: wd={args.wd} dropout={args.dropout} epochs={args.epochs} patience={args.patience}")
    print("behavioral: " + "  ".join(f"{m}={behav[m]:.4f}" for m in models) + "\n")

    res = {f"{m}_{h}{sfx}": [] for m in models for h in heads for sfx in ["", "_shuf"]}
    for L in range(n_layers):
        for m in models:
            X, mask = build_layer_tensor(m, L, meta[m], Tmax, device)
            for h, hid in heads.items():
                res[f"{m}_{h}"].append(curve(X, mask, y, cv, hid, device, args.seed, **train_kw))
                res[f"{m}_{h}_shuf"].append(curve(X, mask, ysh, cv, hid, device, args.seed, **train_kw))
            del X, mask; torch.cuda.empty_cache()
        msg = "  ".join(f"{m[:4]}_{h[:3]} {res[f'{m}_{h}'][-1]:.3f}/{res[f'{m}_{h}_shuf'][-1]:.3f}"
                        for m in models for h in heads)
        print(f"L{L:02d}  {msg}", flush=True)

    h0 = list(heads)[0]
    ref = models[0]
    sig_unf = [r - s for r, s in zip(res[f"{ref}_{h0}"], res[f"{ref}_{h0}_shuf"])]
    L = int(np.argmax(sig_unf)) if args.lock_layer < 0 else args.lock_layer
    print(f"\n=== LOCKED LAYER {L} ({ref} best {h0} signal) ===")
    for m in models:
        for h in heads:
            r, s = res[f"{m}_{h}"], res[f"{m}_{h}_shuf"]
            print(f"{m:>12} {h:>6}: real={r[L]:.3f} shuf={s[L]:.3f} signal={r[L]-s[L]:+.3f} "
                  f"| maxsig={max(rr-ss for rr,ss in zip(r,s)):+.3f}")

    out = {"chance": chance, "lock_layer": L, "Tmax": Tmax, "behav_acc": behav,
           "heads": list(heads), **res}
    json.dump(out, open(args.out_json, "w"), indent=2)
    print(f"saved {args.out_json}")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = list(range(n_layers))
    fig, axes = plt.subplots(1, len(heads), figsize=(6.5 * len(heads), 5.2), sharey=True, squeeze=False)
    for ax, h in zip(axes[0], heads):
        for m in models:
            ax.plot(xs, res[f"{m}_{h}"], "-o", ms=3, color=COLORS[m], label=f"{m} real")
            ax.plot(xs, res[f"{m}_{h}_shuf"], ":", color=COLORS[m], alpha=.5)
        ax.axhline(chance, ls="--", color="gray", alpha=.5)
        ax.axvline(L, ls="-", color="black", alpha=.2)
        ax.set_xlabel("layer (0 = embeddings)")
        ax.set_title(f"learned-attention, {h} read-head (dotted = shuffle)")
        ax.grid(alpha=.2)
    axes[0][0].set_ylabel("5-fold CV probe accuracy"); axes[0][0].set_ylim(0.15, 0.75)
    axes[0][0].legend(loc="upper left", fontsize=7)
    fig.suptitle("Learned-attention probe (pool over all positions): 3 models, real vs shuffle")
    fig.tight_layout(); fig.savefig(args.out_png, dpi=130)
    print(f"saved {args.out_png}")


if __name__ == "__main__":
    main()
