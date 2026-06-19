#!/usr/bin/env python3
"""
inject.py — Phase A: inject corpus knowledge into a base model (continued-pretraining SFT).

Trains a causal LM on the raw document text from corpus.jsonl so the facts land in the
weights. LoRA by default (fast, fits an 80GB card trivially); --full-ft for a full finetune.
This is the "subject model" for the unlearning experiment.

Usage:
  PY=/workspace/envs/wmdp-probes/bin/python
  $PY inject.py --corpus corpus.jsonl --model /workspace/models/wmdp/zephyr-7b-beta_BASE \
      --out injected_model --epochs 3
  $PY inject.py ... --full-ft --lr 1e-5
"""
import argparse
import json

import torch
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, DataCollatorForLanguageModeling)


def load_corpus(path):
    texts = []
    for line in open(path):
        line = line.strip()
        if line:
            texts.append(json.loads(line)["text"])
    return texts


def load_model(path, full_ft):
    """bf16 on CUDA. Tolerates the torch_dtype->dtype rename across transformers versions."""
    try:
        return AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="injected_model")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4, help="LoRA default; full-ft auto-caps to 1e-5")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--full-ft", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    texts = load_corpus(args.corpus)
    print(f"Loaded {len(texts)} documents from {args.corpus}")
    ds = Dataset.from_dict({"text": texts}).map(
        lambda ex: tok(ex["text"], truncation=True, max_length=args.max_len),
        batched=True, remove_columns=["text"])

    model = load_model(args.model, args.full_ft)
    if not args.full_ft:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        model.print_trainable_parameters()
    else:
        args.lr = min(args.lr, 1e-5)

    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs, learning_rate=args.lr,
        bf16=True, logging_steps=10, save_strategy="no", report_to=[],
        gradient_checkpointing=args.full_ft)
    Trainer(model=model, args=targs, train_dataset=ds,
            data_collator=DataCollatorForLanguageModeling(tok, mlm=False)).train()

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"Saved injected model to {args.out}")


if __name__ == "__main__":
    main()
