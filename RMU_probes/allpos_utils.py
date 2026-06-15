"""Loader helpers for the ragged all-positions cache (activations/<tag>_allpos/).

One source of truth for turning the per-layer ragged arrays + span/offset meta into the
per-example tensors the positional probes need. Used by option_pool_probe.py and the
learned-attention probe so they pool over IDENTICAL token spans.
"""
import os

import numpy as np

ALLPOS_DIR = "/workspace/activations"


def load_meta(tag):
    m = np.load(os.path.join(ALLPOS_DIR, f"{tag}_allpos", "meta.npz"), allow_pickle=True)
    return {k: m[k] for k in m.files}


def load_layer(tag, L):
    """Full per-layer ragged array (total_tokens, hidden) in RAM (~1.3 GB)."""
    return np.load(os.path.join(ALLPOS_DIR, f"{tag}_allpos", f"acts_L{L:02d}.npy"))


def _flat_rows(meta, i, span):
    """Flat row indices for example i over a span spec.
    span: 'options' (union of 4 option lines), 'question', 'colon', or 'all'."""
    o = int(meta["offsets"][i])
    if span == "colon":
        return np.array([o + int(meta["colon_local"][i])])
    if span == "question":
        a, b = meta["q_span"][i]
        return np.arange(o + a, o + b)
    if span == "all":
        return np.arange(o, int(meta["offsets"][i + 1]))
    if span == "options":
        return np.concatenate([np.arange(o + a, o + b) for a, b in meta["opt_spans"][i]])
    raise ValueError(span)


def pooled_matrix(tag, L, span="options", meta=None):
    """(N, hidden) mean-pooled over `span` tokens, for one layer. Loads the layer once."""
    meta = meta or load_meta(tag)
    arr = load_layer(tag, L)
    N = len(meta["labels"])
    out = np.empty((N, arr.shape[1]), dtype=np.float32)
    for i in range(N):
        rows = _flat_rows(meta, i, span)
        out[i] = arr[rows].astype(np.float32).mean(0)
    return out


def example_sequences(tag, L, span="options", meta=None):
    """List of (n_tokens_i, hidden) float32 arrays per example for `span` -- for attention probes."""
    meta = meta or load_meta(tag)
    arr = load_layer(tag, L)
    return [arr[_flat_rows(meta, i, span)].astype(np.float32) for i in range(len(meta["labels"]))]
