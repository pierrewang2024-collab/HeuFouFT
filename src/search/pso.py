# Particle Swarm Optimization (PSO) for frequency selection.
#
# Follows Section 3.2 of the paper. Each particle is a sampling set; the
# velocity of an integer coordinate determines a discrete move via Eq. (6):
#     z' = (z + 1) mod d   if r < sigmoid(v)
#     z' = (z - 1) mod d   otherwise
# and each move updates the conjugate partner to preserve pairing. The
# inertia weight decays linearly from 0.9 to 0.4 over the iterations and both
# acceleration constants are 1.4945. Velocities are attracted toward the
# personal best and the swarm best through the standard update
#     v <- w*v + c1*r1*(pbest - z) + c2*r2*(gbest - z),
# evaluated per coordinate on the wrapped grid (signed circular distance).

from __future__ import annotations

from typing import Callable, List

import numpy as np

from .space import ModuleGrid, Solution, discrete_move, random_solution


class PSOSearcher:
    def __init__(self, grids: List[ModuleGrid],
                 fitness_fn: Callable[[Solution], float],
                 n_iter: int = 240, n_particles: int = 8,
                 w_start: float = 0.9, w_end: float = 0.4,
                 c1: float = 1.4945, c2: float = 1.4945,
                 seed: int = 0, init_fn: Callable[[], Solution] = None):
        self.grids = grids
        self.fitness_fn = fitness_fn
        self.n_iter = n_iter
        self.n_particles = n_particles
        self.w_start, self.w_end = w_start, w_end
        self.c1, self.c2 = c1, c2
        self.rng = np.random.default_rng(seed)
        self.init_fn = init_fn

    def run(self) -> Solution:
        rng = self.rng
        positions: List[Solution] = []
        if self.init_fn is not None:
            positions.append(self.init_fn())
        while len(positions) < self.n_particles:
            positions.append(random_solution(self.grids, rng))

        # Velocities: one (u, v) velocity per pair per module.
        velocities = []
        for sol in positions:
            velocities.append({
                g.name: rng.normal(0, 1, size=(g.n_pairs, 2))
                for g in self.grids
            })

        pbest = [s for s in positions]
        pbest_fit = [self._eval(s) for s in positions]
        gbest_i = int(np.argmax(pbest_fit))
        gbest, gbest_fit = pbest[gbest_i], pbest_fit[gbest_i]

        for it in range(self.n_iter):
            w = self.w_start - (self.w_start - self.w_end) * it / max(1, self.n_iter - 1)
            for i in range(self.n_particles):
                # Velocity update toward personal and swarm best.
                for g in self.grids:
                    v = velocities[i][g.name]
                    z = positions[i][g.name].astype(np.float64)
                    pb = pbest[i][g.name].astype(np.float64)
                    gb = gbest[g.name].astype(np.float64)
                    r1 = rng.random(v.shape)
                    r2 = rng.random(v.shape)
                    # Signed circular differences.
                    def cdiff(a, b, d):
                        diff = a - b
                        return np.where(diff > d / 2, diff - d,
                                        np.where(diff < -d / 2, diff + d, diff))
                    du = cdiff(pb[:, 0], z[:, 0], g.d_out)
                    dv = cdiff(pb[:, 1], z[:, 1], g.d_in)
                    gu = cdiff(gb[:, 0], z[:, 0], g.d_out)
                    gv = cdiff(gb[:, 1], z[:, 1], g.d_in)
                    v[:, 0] = w * v[:, 0] + self.c1 * r1[:, 0] * du + self.c2 * r2[:, 0] * gu
                    v[:, 1] = w * v[:, 1] + self.c1 * r1[:, 1] * dv + self.c2 * r2[:, 1] * gv
                new_pos, velocities[i] = discrete_move(positions[i], velocities[i],
                                                       self.grids, rng)
                f = self._eval(new_pos)
                positions[i] = new_pos
                if f > pbest_fit[i]:
                    pbest[i], pbest_fit[i] = new_pos, f
                    if f > gbest_fit:
                        gbest, gbest_fit = new_pos, f
            if (it + 1) % 10 == 0:
                print(f"[PSO] iter {it + 1}/{self.n_iter} gbest={gbest_fit:.4f}")
        return gbest

    def _eval(self, sol: Solution) -> float:
        return float(self.fitness_fn(sol))
