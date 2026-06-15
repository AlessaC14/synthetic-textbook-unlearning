"""Step 3 (token-aggregation axis) -- MEAN-POOLED activation extraction.

The last-token (':') cache reads exactly the position the eval scores. RMU might suppress
only at that readout position while leaving the answer content present elsewhere in the
sequence (e.g. at the A./B./C./D. option-text positions). A masked mean over ALL real tokens
catches that. We compute the mean on-the-fly per layer (no need to store all positions), so
the output has the same shape/size as the last-token cache.

Output .npz:
  acts      : float16 (N, n_layers, hidden)  -- masked mean over real tokens, per layer
  labels    : int64   (N,)
  behav_pred: int64   (N,)                    -- letter-logit argmax at ':' (floor check, same run)
  behav_acc : float64 scalar
  pooling='mean', model_path, n_layers (metadata)

Usage:
  python extract_meanpool.py --model_path /workspace/models/wmdp/Zephyr_RMU \
                            --out /workspace/activations/rmu_wmdp_bio_mean.npz
"""
import argparse

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt_utils import LETTER_TOKEN_IDS, TOKENIZER_DIR, build_prompt, load_wmdp_bio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR, use_fast=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    ds = load_wmdp_bio("test")
    prompts = [build_prompt(ex) for ex in ds]
    colon_id = tok(prompts[0], add_special_tokens=True)["input_ids"][-1]
    labels = np.array([ex["answer"] for ex in ds], dtype=np.int64)
    letter_ids = torch.tensor([LETTER_TOKEN_IDS[c] for c in "ABCD"], device="cuda")

    n_layers = model.config.num_hidden_layers + 1
    hidden = model.config.hidden_size
    acts = np.zeros((len(prompts), n_layers, hidden), dtype=np.float16)
    behav_pred = np.zeros(len(prompts), dtype=np.int64)

    for start in range(0, len(prompts), args.batch_size):
        batch = prompts[start : start + args.batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=True).to("cuda")
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)

        mask = enc["attention_mask"].unsqueeze(-1).to(out.hidden_states[0].dtype)  # (B,T,1)
        denom = mask.sum(1)  # (B,1) number of real tokens
        last_idx = enc["attention_mask"].sum(1) - 1
        rows = torch.arange(enc["input_ids"].size(0))
        got = enc["input_ids"][rows, last_idx]
        assert (got == colon_id).all(), f"last real token != ':' at batch {start}: {got.tolist()}"

        for L, hs in enumerate(out.hidden_states):  # (B,T,H)
            pooled = (hs * mask).sum(1) / denom  # masked mean over real tokens -> (B,H)
            acts[start : start + len(batch), L] = pooled.float().cpu().numpy().astype(np.float16)

        letter_logits = out.logits[rows, last_idx][:, letter_ids]
        behav_pred[start : start + len(batch)] = letter_logits.argmax(-1).cpu().numpy()
        print(f"{start + len(batch)}/{len(prompts)}", flush=True)

    behav_acc = float((behav_pred == labels).mean())
    np.savez(args.out, acts=acts, labels=labels, behav_pred=behav_pred, behav_acc=behav_acc,
            pooling="mean", model_path=args.model_path, n_layers=n_layers)
    print(f"saved {args.out}  acts={acts.shape}  pooling=mean  behav_acc={behav_acc:.4f}")


if __name__ == "__main__":
    main()
