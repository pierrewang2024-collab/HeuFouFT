# Genetic Algorithm with Simulated Annealing (GA-SA) for frequency selection.
#
# Follows Section 3.2 of the paper: each conjugate coordinate pair is encoded
# as a bit string (40 bits for a 1024 x 1024 grid: 20 bits per pair, ten per
# axis). Two-point crossover and bit-flip mutation (p_m = 0.05) generate
# offspring; the SA acceptance rule accepts equal-or-better fitness always
# and a decrease of magnitude delta with probability exp(-delta / T). The
# temperature starts at T = 1000 and decays by 0.95 per iteration.

from __future__ import annotations

import math
from typing import Callable, List

import numpy as np

from .space import (ModuleGrid, Solution, bit_flip_mutation, decode_solution,
                    dedup_pairs, encode_solution, two_point_crossover)


class GASASearcher:
    def __init__(self, grids: List[ModuleGrid],
                 fitness_fn: Callable[[Solution], float],
                 n_iter: int = 175, pop_size: int = 8,
                 p_m: float = 0.05, T0: float = 1000.0,
                 cooling: float = 0.95, seed: int = 0,
                 init_fn: Callable[[], Solution] = None):
        self.grids = grids
        self.fitness_fn = fitness_fn
        self.n_iter = n_iter
        self.pop_size = pop_size
        self.p_m = p_m
        self.T0 = T0
        self.cooling = cooling
        self.rng = np.random.default_rng(seed)
        self.init_fn = init_fn
        self.history: List[dict] = []

    def run(self) -> Solution:
        rng = self.rng
        # Seed the first individual from the heuristic map prior (if given);
        # the rest are random to keep the population diverse.
        population: List[Solution] = []
        if self.init_fn is not None:
            population.append(self.init_fn())
        while len(population) < self.pop_size:
            from .space import random_solution
            population.append(random_solution(self.grids, rng))

        fits = [self._eval(s) for s in population]
        best_i = int(np.argmax(fits))
        best_sol, best_fit = population[best_i], fits[best_i]

        T = self.T0
        for it in range(self.n_iter):
            # Two parents from the better half of the population.
            top = np.argsort(fits)[: max(2, self.pop_size // 2)]
            i1, i2 = rng.choice(top, size=2, replace=False)
            b1 = encode_solution(population[i1], self.grids)
            b2 = encode_solution(population[i2], self.grids)
            c1, _ = two_point_crossover(b1, b2, rng)
            c1 = bit_flip_mutation(c1, self.p_m, rng)
            child = dedup_pairs(decode_solution(c1, self.grids),
                                self.grids, rng)
            child_fit = self._eval(child)

            # SA acceptance against the current worst individual.
            worst_i = int(np.argmin(fits))
            delta = child_fit - fits[worst_i]
            accept = delta >= 0.0 or rng.random() < math.exp(-max(delta, 0.0) / max(T, 1e-9))
            if accept:
                population[worst_i], fits[worst_i] = child, child_fit
                if child_fit > best_fit:
                    best_sol, best_fit = child, child_fit

            T *= self.cooling
            if (it + 1) % 10 == 0:
                print(f"[GA-SA] iter {it + 1}/{self.n_iter} "
                      f"best={best_fit:.4f} T={T:.2f}")
        return best_sol

    def _eval(self, sol: Solution) -> float:
        f = float(self.fitness_fn(sol))
        self.history.append({"fitness": f})
        return f
