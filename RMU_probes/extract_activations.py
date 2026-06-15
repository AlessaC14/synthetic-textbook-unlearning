"""Extract last-token (':') residual-stream activations for WMDP-bio MCQ.

Reusable for any Zephyr-arch model (base or RMU) -- only --model_path changes.
Reads the hidden state at the final context token (the ':' of 'Answer:'), i.e. the
position whose next-token prediction is the A/B/C/D letter, across ALL hidden states
(embeddings + 32 transformer layers = 33), for all 1273 wmdp-bio test examples.

Output: an .npz with
  acts      : float16 (N, n_layers, hidden) -- ':' activation per layer
  labels    : int64   (N,)                  -- dataset `answer`, 0-based index into choices
  behav_pred: int64   (N,)                  -- argmax over A/B/C/D letter logits at ':' (0-3)
  behav_acc : float64 scalar                -- mean(behav_pred == labels), same run as the probe
  model_path, n_layers (metadata)

The letter-logit behavioral accuracy is computed in the SAME forward pass as the
probe activations so the two numbers are directly comparable (methodology rule:
"behavior floored WHILE probe holds" must come from one run). Scoring matches the
lm-eval harness: argmax over the single-token letters _A/_B/_C/_D at the ':' position.

Usage:
  python extract_activations.py --model_path /workspace/models/wmdp/zephyr-7b-beta_BASE \
                                --out /workspace/activations/base_wmdp_bio.npz
"""
import argparse

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt_utils import DATASETS, LETTER_TOKEN_IDS, TOKENIZER_DIR, build_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="wmdp_bio", choices=list(DATASETS))
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR, use_fast=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # right pad: causal mask -> pad tokens never affect real positions

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    ds = DATASETS[args.dataset]("test")
    prompts = [build_prompt(ex) for ex in ds]
    # Derive the expected final token id from a real prompt (the in-context ':' of
    # 'Answer:' is id 28747; note standalone tok(':') tokenizes differently).
    colon_id = tok(prompts[0], add_special_tokens=True)["input_ids"][-1]
    labels = np.array([ex["answer"] for ex in ds], dtype=np.int64)
    # Letter logit ids in class order A,B,C,D -> argmax index == choices index == label.
    letter_ids = torch.tensor([LETTER_TOKEN_IDS[c] for c in "ABCD"], device="cuda")

    n_layers = model.config.num_hidden_layers + 1  # + embedding layer
    hidden = model.config.hidden_size
    acts = np.zeros((len(prompts), n_layers, hidden), dtype=np.float16)
    behav_pred = np.zeros(len(prompts), dtype=np.int64)

    for start in range(0, len(prompts), args.batch_size):
        batch = prompts[start : start + args.batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=True).to("cuda")
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)

        last_idx = enc["attention_mask"].sum(1) - 1  # index of final real token per sample
        # Ground-truth sanity: the position we read MUST be the ':' of 'Answer:'.
        got = enc["input_ids"][torch.arange(enc["input_ids"].size(0)), last_idx]
        assert (got == colon_id).all(), (
            f"last real token is not ':' (id {colon_id}) at batch starting {start}: {got.tolist()}"
        )

        rows = torch.arange(out.hidden_states[0].size(0))
        for L, hs in enumerate(out.hidden_states):  # tuple len n_layers, each (B, T, H)
            sel = hs[rows, last_idx]  # (B, H)
            acts[start : start + len(batch), L] = sel.float().cpu().numpy().astype(np.float16)

        # Behavioral score: argmax over the four letter logits at the ':' position.
        letter_logits = out.logits[rows, last_idx][:, letter_ids]  # (B, 4)
        behav_pred[start : start + len(batch)] = letter_logits.argmax(-1).cpu().numpy()
        print(f"{start + len(batch)}/{len(prompts)}", flush=True)

    behav_acc = float((behav_pred == labels).mean())
    np.savez(
        args.out, acts=acts, labels=labels, behav_pred=behav_pred,
        behav_acc=behav_acc, model_path=args.model_path, n_layers=n_layers,
    )
    print(f"saved {args.out}  acts={acts.shape}  labels={labels.shape}")
    print(f"behavioral letter-logit accuracy = {behav_acc:.4f}  (chance {1/4:.2f})")


if __name__ == "__main__":
    main()
