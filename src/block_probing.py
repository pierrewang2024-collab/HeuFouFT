# Block-level probing: build the heuristic intensity map of Section 3.1.
#
# The spectrum grid of each target module is partitioned into a 10 x 10
# block grid. For each block, a candidate coordinate set is drawn inside the
# block, the model is briefly fine-tuned (10% of the training data for one
# epoch) with those coordinates fixed, and the five NLG metrics are computed
# on a validation subset. The weighted score of Eq. (5) defines the block's
# heuristic intensity. The resulting map provides the prior that initializes
# the meta-heuristic searches.
#
# Usage:
#   python src/block_probing.py --base_model gpt2-medium \
#       --data_dir datasets/e2e_nlg_dataset --output_dir output/probing \
#       --n_probe_per_block 125 --val_subset 300
#
# Output: intensity_map.json (10 x 10 per module) and a heat-map figure.

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.common import (build_fourierft_model, compute_nlg_metrics,
                        fit_norm_stats, generate_predictions,
                        lightweight_finetune, load_e2e_splits, load_tokenizer,
                        save_json, heuristic_intensity)
from fourier_ft.sampling import uniform_indices


def probe_block(block, n_probe, rng, d_out, d_in):
    """Sample n_probe coordinates inside a (row, col) block of the grid."""
    bu, bv = block
    u0, u1 = bu * d_out // 10, (bu + 1) * d_out // 10
    v0, v1 = bv * d_in // 10, (bv + 1) * d_in // 10
    us = rng.integers(u0, max(u0 + 1, u1), size=n_probe)
    vs = rng.integers(v0, max(v0 + 1, v1), size=n_probe)
    return np.stack([us, vs], axis=0).astype(np.int64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, default="gpt2-medium")
    p.add_argument("--data_dir", type=str, default="datasets/e2e_nlg_dataset")
    p.add_argument("--output_dir", type=str, default="output/probing")
    p.add_argument("--modules", nargs="+", default=["c_attn", "c_proj"])
    p.add_argument("--n_frequency", type=int, default=1000,
                   help="Budget per module used during probing.")
    p.add_argument("--scaling", type=float, default=16.0)
    p.add_argument("--train_frac", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--val_subset", type=int, default=300,
                   help="Validation examples used for scoring each block.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_blocks", type=int, default=0,
                   help="Skip the first N (row-major) blocks (for resuming).")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)

    tokenizer = load_tokenizer(args.base_model)
    dataset = load_e2e_splits(args.data_dir)
    val = dataset["validation"]
    if args.val_subset < len(val):
        val = val.select(range(args.val_subset))
    val_mrs = val["meaning_representation"]
    val_refs = val["human_reference"]

    # Spectrum shapes of the GPT-2 target modules (rows x cols).
    shapes = {"c_attn": (3072, 1024), "c_proj": (1024, 1024)}

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "block_metrics.jsonl")
    done_blocks = set()
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                done_blocks.add((rec["module"], rec["block_row"], rec["block_col"]))

    out_f = open(results_path, "a", encoding="utf-8")

    for module in args.modules:
        d_out, d_in = shapes[module]
        for bu in range(10):
            for bv in range(10):
                if (module, bu, bv) in done_blocks:
                    continue
                # Coordinates for every module are drawn inside this block;
                # the other module keeps a uniform random set so the probe
                # reflects the block's marginal contribution.
                indices = {}
                for m in args.modules:
                    if m == module:
                        indices[m] = probe_block((bu, bv), args.n_frequency,
                                                 rng, *shapes[m])
                    else:
                        indices[m] = np.asarray(
                            uniform_indices(*shapes[m], args.n_frequency, rng))
                npz = os.path.join(args.output_dir, "_probe_tmp.npz")
                np.savez(npz, **indices)

                model = build_fourierft_model(
                    args.base_model, n_frequency=args.n_frequency,
                    scaling=args.scaling, sampling="indices",
                    indices_file=npz, sampling_seed=args.seed, device=device)
                lightweight_finetune(
                    model, tokenizer, dataset["train"],
                    frac=args.train_frac, epochs=args.epochs,
                    output_dir=os.path.join(args.output_dir, "tmp_trainer"))

                preds = generate_predictions(model, tokenizer, val_mrs,
                                             batch_size=8, device=device)
                metrics = compute_nlg_metrics(preds, val_refs)
                rec = {"module": module, "block_row": bu, "block_col": bv,
                       **{k: float(v) for k, v in metrics.items()}}
                out_f.write(json.dumps(rec) + "\n")
                out_f.flush()
                print(f"[probe] {module} block ({bu},{bv}): "
                      f"BLEU={metrics['BLEU']:.3f}")
                del model
                torch.cuda.empty_cache()

    out_f.close()

    # Aggregate into the intensity map of Eq. (5) with z-score normalization
    # across the blocks of each module.
    records = [json.loads(l) for l in open(results_path, encoding="utf-8")]
    maps = {}
    for module in args.modules:
        recs = [r for r in records if r["module"] == module]
        if not recs:
            continue
        stats = fit_norm_stats(recs)
        hmap = np.zeros((10, 10))
        for r in recs:
            hmap[r["block_row"], r["block_col"]] = heuristic_intensity(r, stats)
        maps[module] = hmap.tolist()
        np.save(os.path.join(args.output_dir, f"intensity_{module}.npy"), hmap)

    save_json(maps, os.path.join(args.output_dir, "intensity_map.json"))
    print(f"Intensity map saved to {args.output_dir}/intensity_map.json")

    # Optional heat-map figure.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(maps), figsize=(5 * len(maps), 4))
        if len(maps) == 1:
            axes = [axes]
        for ax, (module, hmap) in zip(axes, maps.items()):
            im = ax.imshow(np.array(hmap), cmap="viridis")
            ax.set_title(f"{module} heuristic intensity")
            fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, "intensity_map.png"), dpi=150)
        print("Heat-map figure saved.")
    except Exception as e:  # matplotlib is optional
        print(f"Skipping figure: {e}")


if __name__ == "__main__":
    main()
