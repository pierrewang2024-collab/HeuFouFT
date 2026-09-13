# Evaluate a trained adapter on the E2E test set with the five NLG metrics
# reported in the paper: BLEU, NIST, METEOR, ROUGE-L, and CIDEr.
#
# Usage:
#   python src/evaluate.py --adapter_dir output/train/final_adapter \
#       --base_model gpt2-medium --data_dir datasets/e2e_nlg_dataset \
#       --split test --output_dir output/eval
#
# The script conditions generation on the meaning representations, decodes
# greedily, computes the corpus-level metrics, and saves both the metrics and
# the raw predictions. Progress is checkpointed so interrupted runs can be
# resumed with --resume.

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Importing the local fourier_ft package registers the extended FourierFT
# implementation (with the sampling / band-pass / fixed-indices fields) as
# the "fourierft" PEFT method. PeftModel.from_pretrained below then restores
# the searched coordinates exactly: for "indices" sampling the coordinate
# file is reloaded, and for "uniform" / "bandpass" the RNG is re-seeded, so
# the reconstructed indices match the trained adapter bit for bit.
import fourier_ft  # noqa: F401  (registers the local PEFT method)

from src.common import (compute_nlg_metrics, generate_predictions,
                        load_e2e_splits, save_json)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate an adapter on E2E")
    p.add_argument("--adapter_dir", type=str, required=True,
                   help="Directory holding the saved adapter (from train.py).")
    p.add_argument("--base_model", type=str, default="gpt2-medium")
    p.add_argument("--data_dir", type=str, default="datasets/e2e_nlg_dataset")
    p.add_argument("--split", type=str, default="test",
                   choices=["validation", "test"])
    p.add_argument("--output_dir", type=str, default="output/eval")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=100)
    p.add_argument("--limit", type=int, default=None,
                   help="Optionally evaluate on the first N examples only.")
    p.add_argument("--checkpoint_every", type=int, default=200)
    p.add_argument("--resume", action="store_true",
                   help="Resume from the checkpoint in --output_dir, if any.")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # The FourierFT / HeuFouFT adapter stores the coordinate indices inside
    # the checkpoint; loading through PEFT restores them exactly.
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model ...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model = model.to(device).eval()

    dataset = load_e2e_splits(args.data_dir)[args.split]
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    mr_texts = dataset["meaning_representation"]
    references = dataset["human_reference"]
    print(f"Evaluating {len(mr_texts)} examples from '{args.split}'")

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "checkpoint.json")
    predictions = []
    if args.resume and os.path.exists(ckpt_path):
        with open(ckpt_path, "r", encoding="utf-8") as f:
            predictions = json.load(f)["predictions"]
        print(f"Resuming: {len(predictions)} predictions already done")

    start = len(predictions)
    remaining = mr_texts[start:]
    for i in range(0, len(remaining), args.batch_size):
        batch = remaining[i:i + args.batch_size]
        predictions.extend(
            generate_predictions(model, tokenizer, batch,
                                 max_length=args.max_new_tokens,
                                 batch_size=args.batch_size, device=device))
        done = start + i + len(batch)
        print(f"  {done}/{len(mr_texts)} generated", flush=True)
        if done % args.checkpoint_every < args.batch_size:
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump({"predictions": predictions}, f)

    metrics = compute_nlg_metrics(predictions, references)
    print("\n==== Results ====")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    save_json({"metrics": metrics, "n_examples": len(predictions),
               "adapter_dir": args.adapter_dir, "split": args.split},
              os.path.join(args.output_dir, "metrics.json"))
    save_json({"predictions": predictions, "references": references},
              os.path.join(args.output_dir, "predictions.json"))
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
