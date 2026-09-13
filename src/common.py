# Shared utilities for HeuFouFT training, evaluation, block probing, and
# meta-heuristic search.
#
# Contents:
#   - Dataset loading helpers (encoded E2E splits).
#   - Model construction for FourierFT (uniform / bandpass / fixed indices),
#     LoRA, and full fine-tuning.
#   - NLG metric computation (BLEU, NIST, METEOR, ROUGE-L, CIDEr).
#   - Heuristic intensity (HI): the weighted NLG score of Eq. (5) in the paper.
#   - A lightweight fine-tuning routine used by block probing and by the
#     fitness function of the meta-heuristic searches.

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_from_disk
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)

# GPT-2 stores Q, K, V in a single fused projection `c_attn` and the
# attention output in `c_proj`. Together these are the attention projections
# referred to as "Q and K projections" in the paper. With n_frequency = 1000
# per module and 24 layers this yields 0.048M trainable parameters
# (24 x 2 x 1000), matching the parameter budget of FourierFT / LoCA in
# Table 2 of the paper; the HeuFouFT budget uses n_frequency = 625 (0.03M).
DEFAULT_TARGET_MODULES = ["c_attn", "c_proj"]
DEFAULT_MAX_LENGTH = 128

HI_WEIGHTS = {"BLEU": 0.3, "METEOR": 0.2, "ROUGE-L": 0.2,
              "NIST": 0.15, "CIDEr": 0.15}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_tokenizer(model_name_or_path: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_e2e_splits(data_dir: str) -> Dict:
    """Load the preprocessed E2E dataset (see data/prepare_e2e.py)."""
    return load_from_disk(data_dir)


def make_train_dataset(split, tokenizer=None, max_length: int = DEFAULT_MAX_LENGTH):
    """Return the tokenized training split in torch format."""
    ds = split.map(lambda ex: {"labels": ex["input_ids"]},
                   desc="Attaching labels")
    ds.set_format(type="torch",
                  columns=["input_ids", "attention_mask", "labels"])
    return ds


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def build_fourierft_model(base_model_name_or_path: str,
                          n_frequency: int = 1000,
                          scaling: float = 16.0,
                          sampling: str = "uniform",
                          bandpass_fc: float = 0.0,
                          bandpass_width: float = 20.0,
                          indices_file: Optional[str] = None,
                          sampling_seed: int = 777,
                          device: Optional[str] = None):
    """Wrap a causal LM with FourierFT adapters using the requested sampling.

    sampling: "uniform" (vanilla), "bandpass" (Gaussian band-pass with bias
    `bandpass_fc` and bandwidth `bandpass_width`), or "indices" (fixed
    coordinates loaded from `indices_file`, e.g. the output of the search).
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from fourier_ft import FourierFTConfig, FourierFTModel

    model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path)
    if device is not None:
        model = model.to(device)
    config = FourierFTConfig(
        target_modules=DEFAULT_TARGET_MODULES,
        init_weights=True,
        scaling=scaling,
        n_frequency=n_frequency,
        inference_mode=False,
        fan_in_fan_out=True,
        sampling=sampling,
        bandpass_fc=bandpass_fc,
        bandpass_width=bandpass_width,
        sampling_indices_file=indices_file,
        sampling_seed=sampling_seed,
    )
    model = FourierFTModel(model, config, adapter_name="default")
    if device is not None:
        model = model.to(device)
    print_trainable_parameters(model)
    return model


def build_lora_model(base_model_name_or_path: str, rank: int = 8,
                     alpha: int = 16, lr_dropout: float = 0.05,
                     device: Optional[str] = None):
    """Wrap a causal LM with LoRA adapters (rank 8 in the paper)."""
    from peft import LoraConfig, TaskType, get_peft_model

    model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path)
    if device is not None:
        model = model.to(device)
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=lr_dropout,
        target_modules=DEFAULT_TARGET_MODULES,
        fan_in_fan_out=True,
    )
    model = get_peft_model(model, config)
    if device is not None:
        model = model.to(device)
    print_trainable_parameters(model)
    return model


def print_trainable_parameters(model) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable} || all params: {total} || "
          f"trainable%: {100 * trainable / total:.4f}")


# ---------------------------------------------------------------------------
# NLG metrics and heuristic intensity
# ---------------------------------------------------------------------------
def compute_nlg_metrics(predictions: List[str],
                        references: List[str]) -> Dict[str, float]:
    """Compute the five NLG metrics used in the paper."""
    from nltk.translate import bleu_score as nltk_bleu
    from nltk.translate import meteor_score as nltk_meteor
    from nltk.translate import nist_score as nltk_nist
    from nltk.tokenize import word_tokenize
    from rouge_score import rouge_scorer

    pred_tokens = [word_tokenize(p.lower()) for p in predictions]
    ref_tokens = [[word_tokenize(r.lower())] for r in references]

    metrics = {}
    metrics["BLEU"] = nltk_bleu.corpus_bleu(
        ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25))
    metrics["NIST"] = nltk_nist.corpus_nist(ref_tokens, pred_tokens, n=4)
    metrics["METEOR"] = float(np.mean([
        nltk_meteor.meteor_score([rt[0]], pt)
        for pt, rt in zip(pred_tokens, ref_tokens)]))
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    metrics["ROUGE-L"] = float(np.mean([
        scorer.score(ref, pred)["rougeL"].fmeasure
        for ref, pred in zip(references, predictions)]))

    from pycocoevalcap.cider.cider import Cider
    cider = Cider()
    refs = {i: [r] for i, r in enumerate(references)}
    preds = {i: [p] for i, p in enumerate(predictions)}
    metrics["CIDEr"], _ = cider.compute_score(refs, preds)
    return metrics


def heuristic_intensity(metrics: Dict[str, float],
                        norm_stats: Optional[Dict[str, Tuple[float, float]]] = None,
                        scale: float = 100.0) -> float:
    """Weighted NLG score of Eq. (5).

    HI = 0.3*BLEU + 0.2*METEOR + 0.2*ROUGE-L + 0.15*NIST + 0.15*CIDEr,
    with each metric normalized (z-score using `norm_stats`, or raw when
    None) and rescaled by `scale` to keep the fitness on a comparable
    magnitude for the annealing acceptance step.
    """
    total = 0.0
    for name, w in HI_WEIGHTS.items():
        v = float(metrics[name])
        if norm_stats is not None and name in norm_stats:
            mu, sigma = norm_stats[name]
            v = (v - mu) / (sigma + 1e-8)
        total += w * v
    return total * scale


def fit_norm_stats(metric_dicts: List[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
    """Estimate per-metric mean/std for z-score normalization."""
    stats = {}
    for name in HI_WEIGHTS:
        vals = np.array([m[name] for m in metric_dicts], dtype=np.float64)
        stats[name] = (float(vals.mean()), float(vals.std()))
    return stats


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_predictions(model, tokenizer, mr_texts: List[str],
                         max_length: int = 100,
                         batch_size: int = 8,
                         device: Optional[str] = None) -> List[str]:
    """Greedy generation conditioned on the meaning representations."""
    model.eval()
    outputs: List[str] = []
    for i in range(0, len(mr_texts), batch_size):
        batch = mr_texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True)
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        gen = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        outputs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return outputs


# ---------------------------------------------------------------------------
# Lightweight fine-tuning (block probing & search fitness)
# ---------------------------------------------------------------------------
def lightweight_finetune(model, tokenizer, train_split,
                         frac: float = 0.1,
                         epochs: int = 1,
                         batch_size: int = 4,
                         learning_rate: float = 2e-4,
                         output_dir: str = "/tmp/heufouft_probe",
                         max_steps: Optional[int] = None) -> None:
    """Briefly fine-tune the trainable coefficients (c) with E fixed.

    This routine backs both the block-level probing of Section 3.1 and the
    fitness evaluation of the meta-heuristic searches: each candidate
    coordinate set is trained on a fraction of the data for one epoch, then
    scored on a validation subset.
    """
    n = len(train_split)
    take = max(1, int(n * frac))
    idx = np.linspace(0, n - 1, take).astype(int).tolist()
    sub = train_split.select(idx)
    sub = sub.map(lambda ex: {"labels": ex["input_ids"]}, desc="labels")
    sub.set_format(type="torch",
                   columns=["input_ids", "attention_mask", "labels"])

    args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="no",
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        learning_rate=learning_rate,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        max_steps=max_steps if max_steps is not None else -1,
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=sub,
        eval_dataset=None,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                      mlm=False),
    )
    trainer.train()
    model.eval()


def save_solution(indices_per_module: Dict[str, np.ndarray], path: str) -> None:
    """Persist a searched coordinate set as a .npz file.

    Keys are module names (e.g. "c_attn", "c_proj"); each value is an integer
    array of shape [2, n_frequency] (row 0: spectrum rows / out dim,
    row 1: spectrum cols / in dim).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **indices_per_module)
    print(f"Saved coordinate set to {path}")


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
