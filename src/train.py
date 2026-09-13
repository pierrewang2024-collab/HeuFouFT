# Train a PEFT adapter (FourierFT / HeuFouFT / LoRA) or run full fine-tuning
# on the E2E NLG benchmark with GPT-2-Medium.
#
# Examples (see README.md for the full reproduction table):
#   # Vanilla FourierFT with uniform sampling (0.048M params)
#   python src/train.py --method fourierft --sampling uniform \
#       --base_model gpt2-medium --data_dir datasets/e2e_nlg_dataset \
#       --n_frequency 1000 --scaling 16.0 --learning_rate 2e-4 --seed 42
#
#   # Gaussian band-pass sampling with frequency bias fc = 300 (Table 1)
#   python src/train.py --method fourierft --sampling bandpass --fc 300 \
#       --n_frequency 1000 --scaling 16.0 --seed 42
#
#   # HeuFouFT: train with a searched coordinate set (0.03M params)
#   python src/train.py --method fourierft --sampling indices \
#       --indices_file output/search/pso/best_indices.npz \
#       --n_frequency 625 --scaling 16.0 --seed 42
#
#   # LoRA baseline (rank 8)
#   python src/train.py --method lora --learning_rate 3e-4 --seed 42
#
#   # Full fine-tuning baseline
#   python src/train.py --method full --learning_rate 5e-5 --seed 42

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import (AutoModelForCausalLM, DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)

from src.common import (build_fourierft_model, build_lora_model,
                        load_e2e_splits, load_tokenizer, save_json)


def parse_args():
    p = argparse.ArgumentParser(description="Train PEFT adapters on E2E")
    p.add_argument("--method", choices=["fourierft", "lora", "full"],
                   default="fourierft")
    p.add_argument("--base_model", type=str, default="gpt2-medium")
    p.add_argument("--data_dir", type=str, default="datasets/e2e_nlg_dataset")
    p.add_argument("--output_dir", type=str, default="output/train")

    # FourierFT / HeuFouFT options
    p.add_argument("--sampling", choices=["uniform", "bandpass", "indices"],
                   default="uniform")
    p.add_argument("--n_frequency", type=int, default=1000,
                   help="Trainable spectral coefficients per target module. "
                        "1000 -> 0.048M total; 625 -> 0.03M (HeuFouFT).")
    p.add_argument("--scaling", type=float, default=16.0,
                   help="Scaling factor alpha (paper: 16).")
    p.add_argument("--fc", type=float, default=0.0,
                   help="Band-pass frequency bias (Table 1 uses 0..500).")
    p.add_argument("--bandwidth", type=float, default=20.0,
                   help="Band-pass bandwidth W (paper: 20).")
    p.add_argument("--indices_file", type=str, default=None,
                   help=".npz coordinate set produced by src/run_search.py.")

    # LoRA options
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)

    # Optimization (defaults follow Section 4 of the paper)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true",
                   default=torch.cuda.is_available())
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = load_tokenizer(args.base_model)
    dataset = load_e2e_splits(args.data_dir)
    train_split = dataset["train"]
    train_ds = train_split.map(lambda ex: {"labels": ex["input_ids"]},
                               desc="Attaching labels")
    train_ds.set_format(type="torch",
                        columns=["input_ids", "attention_mask", "labels"])
    print(f"Training examples: {len(train_ds)}")

    if args.method == "fourierft":
        model = build_fourierft_model(
            args.base_model,
            n_frequency=args.n_frequency,
            scaling=args.scaling,
            sampling=args.sampling,
            bandpass_fc=args.fc,
            bandpass_width=args.bandwidth,
            indices_file=args.indices_file,
            sampling_seed=args.seed,
            device=device,
        )
    elif args.method == "lora":
        model = build_lora_model(args.base_model, rank=args.lora_rank,
                                 alpha=args.lora_alpha, device=device)
    else:  # full fine-tuning
        model = AutoModelForCausalLM.from_pretrained(args.base_model).to(device)

    os.makedirs(args.output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        evaluation_strategy="no",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=100,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=1.0,
        fp16=args.fp16,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=None,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                      mlm=False),
    )
    trainer.train()

    final_dir = os.path.join(args.output_dir, "final_adapter")
    model.save_pretrained(final_dir, safe_serialization=False)
    tokenizer.save_pretrained(final_dir)
    print(f"Model saved to {final_dir}")

    # Persist the resolved configuration for reproducibility (both at the
    # run root and next to the adapter so evaluation can locate it).
    save_json(vars(args), os.path.join(args.output_dir, "train_config.json"))
    save_json(vars(args), os.path.join(final_dir, "train_config.json"))


if __name__ == "__main__":
    main()
