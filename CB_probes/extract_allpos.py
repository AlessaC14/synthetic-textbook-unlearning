"""Permanent all-positions extraction for the DI CB family (Stage 1).

Ragged position-indexable cache per model so EVERY downstream positional probe (option-span
pool, learned-attention, full-mean, any future span probe) runs off this one cache -- no
re-extraction. Same design as the RMU notebook's permanent cache (Entry 6), ported to the
no-BOS GPT-NeoX harness.

CB-SPECIFIC vs RMU: tokenizer from --model_path, add_special_tokens=False (NO BOS),
NeoX single-token letter ids, DI Robust-MCQA 868-q loader.

Output dir <outdir>/<tag>_allpos/ :
  acts_L00.npy .. acts_L32.npy  float16 (total_tokens, hidden)  -- one row per real token,
                                  all examples concatenated in dataset order, per layer.
  meta.npz : labels, offsets (N+1), lengths (N), q_span (N,2), opt_spans (N,4,2),
             colon_local (N), behav_pred (N), behav_acc, model_path, n_layers, total_tokens
  (flat row for LOCAL index t in example i = offsets[i] + t)

Usage:
  python extract_allpos.py --model_path /workspace/models/deep-ignorance/unfiltered-cb --tag cb
"""
import argparse
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt_utils import (LETTER_TOKEN_IDS, build_prompt_with_spans, load_robust_mcqa)


def char_span_to_tokens(offsets, c0, c1):
    """LOCAL token [start,end) whose char ranges overlap [c0,c1)."""
    idx = [i for i, (a, b) in enumerate(offsets) if b > a and a < c1 and b > c0]
    assert idx, f"empty token span for chars [{c0},{c1})"
    return (idx[0], idx[-1] + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--tag", required=True, help="unfiltered | cb | strongfilter (names output dir)")
    ap.add_argument("--outdir", default="/workspace/CB_activations")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    ds = load_robust_mcqa()
    prompts, all_spans = [], []
    for ex in ds:
        p, sp = build_prompt_with_spans(ex)
        prompts.append(p); all_spans.append(sp)
    labels = np.array([int(ex["answer"]) for ex in ds], dtype=np.int64)
    colon_id = tok(prompts[0], add_special_tokens=False)["input_ids"][-1]
    letter_ids = torch.tensor([LETTER_TOKEN_IDS[c] for c in "ABCD"], device="cuda")

    N = len(prompts)
    n_layers = model.config.num_hidden_layers + 1
    hidden = model.config.hidden_size

    # --- pass 1: token lengths + per-example spans (no-BOS tokenize w/ offsets) ---
    lengths = np.zeros(N, dtype=np.int64)
    q_span = np.zeros((N, 2), dtype=np.int64)
    opt_spans = np.zeros((N, 4, 2), dtype=np.int64)
    for i, p in enumerate(prompts):
        enc = tok(p, add_special_tokens=False, return_offsets_mapping=True)
        offs = enc["offset_mapping"]
        lengths[i] = len(enc["input_ids"])
        q_span[i] = char_span_to_tokens(offs, *all_spans[i]["question"])
        for j, (c0, c1) in enumerate(all_spans[i]["options"]):
            opt_spans[i, j] = char_span_to_tokens(offs, c0, c1)
    offsets = np.zeros(N + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    total = int(offsets[-1])
    colon_local = lengths - 1

    outdir = os.path.join(args.outdir, f"{args.tag}_allpos")
    os.makedirs(outdir, exist_ok=True)
    mmaps = [np.lib.format.open_memmap(os.path.join(outdir, f"acts_L{L:02d}.npy"),
                                       mode="w+", dtype=np.float16, shape=(total, hidden))
             for L in range(n_layers)]
    behav_pred = np.zeros(N, dtype=np.int64)
    print(f"[{args.tag}] N={N} total_tokens={total} -> {outdir}  ({total*n_layers*hidden*2/1e9:.1f} GB)")

    # --- pass 2: forward, stream real tokens to per-layer memmaps ---
    for start in range(0, N, args.batch_size):
        batch = prompts[start : start + args.batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        am = enc["attention_mask"]
        last_idx = am.sum(1) - 1
        rows = torch.arange(enc["input_ids"].size(0))
        got = enc["input_ids"][rows, last_idx]
        assert (got == colon_id).all(), f"last token != ':' at batch {start}"

        for b in range(len(batch)):
            i = start + b
            Li = int(lengths[i])
            assert int(am[b].sum()) == Li, f"length mismatch ex {i}: {int(am[b].sum())} vs {Li}"
            sl = slice(offsets[i], offsets[i] + Li)
            for L, hs in enumerate(out.hidden_states):
                mmaps[L][sl] = hs[b, :Li].float().cpu().numpy().astype(np.float16)
        letter_logits = out.logits[rows, last_idx][:, letter_ids]
        behav_pred[start : start + len(batch)] = letter_logits.argmax(-1).cpu().numpy()
        print(f"{start + len(batch)}/{N}", flush=True)

    for m in mmaps:
        m.flush()
    behav_acc = float((behav_pred == labels).mean())
    np.savez(os.path.join(outdir, "meta.npz"), labels=labels, offsets=offsets, lengths=lengths,
             q_span=q_span, opt_spans=opt_spans, colon_local=colon_local,
             behav_pred=behav_pred, behav_acc=behav_acc, model_path=args.model_path,
             n_layers=n_layers, total_tokens=total)
    print(f"saved {outdir}/meta.npz  behav_acc={behav_acc:.4f}  ({n_layers} layer files)")


if __name__ == "__main__":
    main()
