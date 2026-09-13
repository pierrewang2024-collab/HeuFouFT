# Meta-heuristic frequency-coordinate search (main entry point).
#
# Pipeline (Sections 3.1-3.2 of the paper):
#   1. Load (or build with src/block_probing.py) the heuristic intensity map.
#   2. Initialize candidate coordinate sets from the map prior.
#   3. Run GA-SA / PSO / CS under the fixed parameter budget. The fitness of
#      a candidate set is the heuristic intensity (Eq. 5) after lightweight
#      fine-tuning (10% of the data, one epoch); a Random Forest surrogate
#      screens candidates so only the most promising ones are fine-tuned.
#   4. Save the best coordinate set as a .npz file, ready for
#      `src/train.py --sampling indices` and `src/evaluate.py`.
#
# Usage:
#   python src/run_search.py --algorithm pso --base_model gpt2-medium \
#       --data_dir datasets/e2e_nlg_dataset --map_file output/probing/intensity_map.json \
#       --output_dir output/search/pso
#
# Notes on fidelity: the original experimental search scripts were not
# retained in the internal repository; this module is a faithful
# reimplementation of the algorithms as described in the paper (encodings,
# hyperparameters, acceptance rules, and fitness definition all follow
# Section 3). See README.md for details.

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch

from src.common import (build_fourierft_model, compute_nlg_metrics,
                        generate_predictions, heuristic_intensity,
                        lightweight_finetune, load_e2e_splits, load_tokenizer,
                        save_json)
from src.search import (CSSearcher, GASASearcher, ModuleGrid, PSOSearcher,
                        RFSurrogate, map_seeded_solution,
                        solution_features, solution_to_indices)

# GPT-2-Medium spectrum shapes: dense spectrum is [out_features, in_features].
MODULE_SHAPES = {"c_attn": (3072, 1024), "c_proj": (1024, 1024)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algorithm", choices=["ga_sa", "pso", "cs", "vanilla_ga"],
                   default="pso")
    p.add_argument("--base_model", type=str, default="gpt2-medium")
    p.add_argument("--data_dir", type=str, default="datasets/e2e_nlg_dataset")
    p.add_argument("--map_file", type=str, default=None,
                   help="intensity_map.json from src/block_probing.py.")
    p.add_argument("--output_dir", type=str, default="output/search")
    p.add_argument("--modules", nargs="+", default=["c_attn", "c_proj"])
    p.add_argument("--n_frequency", type=int, default=625,
                   help="Budget per module: 625 -> 0.03M total (paper).")
    p.add_argument("--scaling", type=float, default=16.0)
    p.add_argument("--train_frac", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--val_subset", type=int, default=300)
    p.add_argument("--n_iter", type=int, default=None,
                   help="Search iterations (paper: 175/240/151 for "
                        "GA-SA/PSO/CS).")
    p.add_argument("--surrogate", action="store_true", default=True)
    p.add_argument("--no_surrogate", dest="surrogate", action="store_false")
    p.add_argument("--surrogate_quantile", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)

    tokenizer = load_tokenizer(args.base_model)
    dataset = load_e2e_splits(args.data_dir)
    val = dataset["validation"]
    if args.val_subset < len(val):
        val = val.select(range(args.val_subset))
    val_mrs = val["meaning_representation"]
    val_refs = val["human_reference"]

    grids = [ModuleGrid(name=m, d_out=MODULE_SHAPES[m][0],
                        d_in=MODULE_SHAPES[m][1],
                        n_pairs=args.n_frequency // 2)
             for m in args.modules]

    intensity_maps = None
    if args.map_file and os.path.exists(args.map_file):
        with open(args.map_file, "r", encoding="utf-8") as f:
            intensity_maps = json.load(f)
        print(f"Loaded intensity map from {args.map_file}")

    # ------------------------------------------------------------------
    # Fitness: lightweight fine-tuning with the candidate coordinates fixed,
    # followed by Eq. (5) on a validation subset. The RF surrogate screens
    # candidates before they spend a real fine-tuning run.
    # ------------------------------------------------------------------
    surrogate = RFSurrogate(seed=args.seed) if args.surrogate else None
    eval_log = []

    def evaluate(sol):
        indices = solution_to_indices(sol, grids)
        npz = os.path.join(args.output_dir, "_candidate.npz")
        os.makedirs(args.output_dir, exist_ok=True)
        np.savez(npz, **indices)

        model = build_fourierft_model(
            args.base_model, n_frequency=args.n_frequency,
            scaling=args.scaling, sampling="indices", indices_file=npz,
            sampling_seed=args.seed, device=device)
        lightweight_finetune(
            model, tokenizer, dataset["train"],
            frac=args.train_frac, epochs=args.epochs,
            output_dir=os.path.join(args.output_dir, "tmp_trainer"))
        preds = generate_predictions(model, tokenizer, val_mrs,
                                     batch_size=8, device=device)
        metrics = compute_nlg_metrics(preds, val_refs)
        hi = heuristic_intensity(metrics)
        del model
        torch.cuda.empty_cache()

        if surrogate is not None:
            surrogate.observe(solution_features(sol, grids), hi)
            surrogate.fit()
        eval_log.append({"hi": hi, **{k: float(v) for k, v in metrics.items()}})
        print(f"    evaluated: HI={hi:.3f} BLEU={metrics['BLEU']:.3f}")
        return hi

    # ------------------------------------------------------------------
    # Initialize from the map prior when available.
    # ------------------------------------------------------------------
    def make_init():
        if intensity_maps:
            return map_seeded_solution(grids, intensity_maps, rng)
        from src.search import random_solution
        return random_solution(grids, rng)

    # ------------------------------------------------------------------
    # Run the requested search.
    # ------------------------------------------------------------------
    n_iter = args.n_iter
    if args.algorithm == "ga_sa":
        searcher = GASASearcher(grids, evaluate, seed=args.seed,
                                n_iter=n_iter or 175, init_fn=make_init)
    elif args.algorithm == "pso":
        searcher = PSOSearcher(grids, evaluate, seed=args.seed,
                               n_iter=n_iter or 240, init_fn=make_init)
    elif args.algorithm == "cs":
        searcher = CSSearcher(grids, evaluate, seed=args.seed,
                              n_iter=n_iter or 151, init_fn=make_init)
    else:  # vanilla GA: GA-SA without annealing (ablation of Table 2)
        searcher = GASASearcher(grids, evaluate, seed=args.seed,
                                n_iter=n_iter or 175, T0=1e-9,
                                init_fn=make_init)

    best = searcher.run()

    out_npz = os.path.join(args.output_dir, "best_indices.npz")
    np.savez(out_npz, **solution_to_indices(best, grids))
    save_json({"algorithm": args.algorithm, "n_frequency": args.n_frequency,
               "modules": args.modules, "seed": args.seed,
               "n_evaluations": len(eval_log), "log": eval_log},
              os.path.join(args.output_dir, "search_log.json"))
    print(f"Best coordinate set saved to {out_npz}")
    print("Next: python src/train.py --method fourierft --sampling indices "
          f"--indices_file {out_npz} --n_frequency {args.n_frequency}")


if __name__ == "__main__":
    main()
