# Cuckoo Search (CS) for frequency selection.
#
# Follows Section 3.2 of the paper: each sampling set is a nest; new
# candidates are generated through Lévy flights with exponent lambda = 1.5
# and a step size that follows a cosine decay from 0.5 to 0.1 over the
# iterations. Nests are subject to the sampling budget and the
# conjugate-pairing constraint; nest selection and duplicate-pair removal
# refine the resulting sampling sets.

from __future__ import annotations

from typing import Callable, List

import numpy as np

from .space import (ModuleGrid, Solution, cosine_step, dedup_pairs,
                    levy_perturb, random_solution)


class CSSearcher:
    def __init__(self, grids: List[ModuleGrid],
                 fitness_fn: Callable[[Solution], float],
                 n_iter: int = 100, n_nests: int = 8,
                 lam: float = 1.5, pa: float = 0.25,
                 step_start: float = 0.5, step_end: float = 0.1,
                 seed: int = 0, init_fn: Callable[[], Solution] = None):
        self.grids = grids
        self.fitness_fn = fitness_fn
        self.n_iter = n_iter
        self.n_nests = n_nests
        self.lam = lam
        self.pa = pa
        self.step_start, self.step_end = step_start, step_end
        self.rng = np.random.default_rng(seed)
        self.init_fn = init_fn

    def run(self) -> Solution:
        rng = self.rng
        nests: List[Solution] = []
        if self.init_fn is not None:
            nests.append(self.init_fn())
        while len(nests) < self.n_nests:
            nests.append(random_solution(self.grids, rng))

        fits = [self._eval(s) for s in nests]
        best_i = int(np.argmax(fits))
        best_sol, best_fit = nests[best_i], fits[best_i]

        for it in range(self.n_iter):
            base = cosine_step(it, self.n_iter, self.step_start, self.step_end)
            # One Lévy cuckoo from a random (biased-to-good) nest.
            src_i = int(rng.choice(np.argsort(fits)[: max(2, self.n_nests // 2)]))
            cuckoo = levy_perturb(nests[src_i], self.grids, rng,
                                  lam=self.lam, base=base)
            cuckoo = dedup_pairs(cuckoo, self.grids, rng)
            cf = self._eval(cuckoo)

            # Compare with a random nest; replace if the cuckoo is fitter.
            j = int(rng.integers(0, self.n_nests))
            if cf > fits[j]:
                nests[j], fits[j] = cuckoo, cf
                if cf > best_fit:
                    best_sol, best_fit = cuckoo, cf

            # Abandon the worst fraction of the nests (nest selection).
            n_replace = max(1, int(self.n_nests * self.pa))
            for w_i in np.argsort(fits)[:n_replace]:
                new_nest = dedup_pairs(random_solution(self.grids, rng),
                                       self.grids, rng)
                nf = self._eval(new_nest)
                nests[w_i], fits[w_i] = new_nest, nf
                if nf > best_fit:
                    best_sol, best_fit = new_nest, nf

            if (it + 1) % 10 == 0:
                print(f"[CS] iter {it + 1}/{self.n_iter} best={best_fit:.4f}")
        return best_sol

    def _eval(self, sol: Solution) -> float:
        return float(self.fitness_fn(sol))
