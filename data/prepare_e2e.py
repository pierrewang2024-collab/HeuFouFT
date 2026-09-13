# Download and preprocess the E2E NLG challenge dataset.
#
# The E2E dataset (Novikova et al., 2017) contains 42,061 training /
# 4,672 validation / 4,693 test examples of restaurant reviews, each pairing
# a meaning representation (MR) with human-written references.
#
# Usage:
#   python data/prepare_e2e.py --output_dir datasets/e2e_nlg_dataset
#
# The processed dataset is saved with datasets.save_to_disk so that the
# training / evaluation scripts can load it offline.

import argparse
import os

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Prepare the E2E NLG dataset")
    parser.add_argument("--output_dir", type=str, default="datasets/e2e_nlg_dataset")
    parser.add_argument("--tokenizer", type=str, default="gpt2-medium",
                        help="Tokenizer used for encoding the references.")
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()

    print("Downloading the E2E NLG dataset from the Hugging Face Hub ...")
    raw = load_dataset("tuetschek/e2e_nlg")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def encode(batch):
        encoded = tokenizer(
            batch["human_reference"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": encoded["input_ids"],
            # Keep the raw text fields for generation-time evaluation.
            "meaning_representation": batch["meaning_representation"],
            "human_reference": batch["human_reference"],
        }

    processed = {}
    for split in ["train", "validation", "test"]:
        print(f"Encoding split '{split}' ({len(raw[split])} examples) ...")
        ds = raw[split].map(
            encode,
            batched=True,
            batch_size=64,
            remove_columns=[c for c in raw[split].column_names
                            if c not in ("meaning_representation",
                                         "human_reference")],
            desc=f"Encoding {split}",
        )
        processed[split] = ds
        print(f"  -> {len(ds)} examples")

    os.makedirs(args.output_dir, exist_ok=True)
    DatasetDict(processed).save_to_disk(args.output_dir)
    print(f"Dataset saved to {args.output_dir}")


if __name__ == "__main__":
    main()
