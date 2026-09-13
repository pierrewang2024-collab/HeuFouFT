# Search space and coordinate operators for HeuFouFT.
#
# A candidate solution selects, for each target module type (e.g. "c_attn"
# and "c_proj" of GPT-2), n_frequency spectral coordinates on the module's
# (out_features x in_features) spectrum grid. The coordinates are shared
# across all transformer layers of the same module type, following the
# shared-E design of the paper (each layer learns its own coefficients c).
#
# Conjugate pairing: every coordinate (u, v) is paired with
# (-u mod d_out, -v mod d_in), which keeps the spectral representation
# symmetric and the inverse transform real-valued. One "gene" therefore
# encodes a conjugate pair.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ModuleGrid:
    """Spectrum grid of one target module type."""

    name: str            # e.g. "c_attn"
    d_out: int           # spectrum rows (out features)
    d_in: int            # spectrum columns (in features)
    n_pairs: int         # number of conjugate pairs (= n_frequency / 2)

    @property
    def bits_u(self) -> int:
        return max(1, math.ceil(math.log2(self.d_out)))

    @property
    def bits_v(self) -> int:
        return max(1, math.ceil(math.log2(self.d_in)))

    @property
    def gene_bits(self) -> int:
        """Bits per conjugate pair (u and v coordinates)."""
        return self.bits_u + self.bits_v

    def distance_to_center(self, u: int, v: int) -> float:
        """Wrapped Euclidean distance to the spectral center."""
        du = min(abs(u - self.d_out / 2.0), self.d_out - abs(u - self.d_out / 2.0))
        dv = min(abs(v - self.d_in / 2.0), self.d_in - abs(v - self.d_in / 2.0))
        return math.hypot(du, dv)


# A solution maps module name -> array of shape [n_pairs, 2] holding the
# (u, v) coordinate of each conjugate pair. The partner coordinate
# (-u mod d_out, -v mod d_in) is implicit.
Solution = Dict[str, np.ndarray]


def conjugate_partner(grid: ModuleGrid, u: int, v: int) -> Tuple[int, int]:
    return (-u) % grid.d_out, (-v) % grid.d_in


def random_solution(grids: List[ModuleGrid],
                    rng: np.random.Generator) -> Solution:
    """Uniformly random conjugate pairs on each grid."""
    sol: Solution = {}
    for g in grids:
        us = rng.integers(0, g.d_out, size=g.n_pairs)
        vs = rng.integers(0, g.d_in, size=g.n_pairs)
        sol[g.name] = np.stack([us, vs], axis=1).astype(np.int64)
    return sol


def map_seeded_solution(grids: List[ModuleGrid],
                        intensity_maps: Dict[str, np.ndarray],
                        rng: np.random.Generator,
                        p_topk: float = 0.7) -> Solution:
    """Initialize from the block-level heuristic intensity map.

    A fraction p_topk of the pairs is drawn from the highest-intensity
    blocks of the map (Section 3.1); the rest are uniform random so the
    search can still depart from the coarse prior.
    """
    sol: Solution = {}
    for g in grids:
        if g.name in intensity_maps and intensity_maps[g.name] is not None:
            hmap = intensity_maps[g.name]  # shape [10, 10]
            prob = hmap.astype(np.float64).flatten()
            prob = prob / (prob.sum() + 1e-12)
            blocks = rng.choice(10 * 10, size=g.n_pairs, p=prob)
        else:
            blocks = rng.integers(0, 100, size=g.n_pairs)
        pairs = np.zeros((g.n_pairs, 2), dtype=np.int64)
        n_top = int(round(g.n_pairs * p_topk))
        for i, b in enumerate(blocks):
            bu, bv = int(b) // 10, int(b) % 10
            # Sample uniformly inside block (bu, bv); for the last (partial)
            # block the modulo keeps coordinates inside the grid.
            u = (bu * g.d_out // 10 + int(rng.integers(0, max(1, g.d_out // 10)))) % g.d_out
            v = (bv * g.d_in // 10 + int(rng.integers(0, max(1, g.d_in // 10)))) % g.d_in
            pairs[i] = (u, v)
        # The remaining (1 - p_topk) pairs are fully random.
        for i in range(n_top, g.n_pairs):
            pairs[i] = (rng.integers(0, g.d_out), rng.integers(0, g.d_in))
        sol[g.name] = pairs
    return sol


# ---------------------------------------------------------------------------
# GA-SA bit encoding (Section 3.2: 40-bit strings, two-point crossover,
# bit-flip mutation with p_m = 0.05)
# ---------------------------------------------------------------------------
def encode_solution(sol: Solution, grids: List[ModuleGrid]) -> np.ndarray:
    """Bit-encode a solution; one gene per conjugate pair."""
    bits: List[int] = []
    for g in grids:
        pairs = sol[g.name]
        for u, v in pairs:
            bits.extend(int(x) for x in np.binary_repr(int(u), g.bits_u))
            bits.extend(int(x) for x in np.binary_repr(int(v), g.bits_v))
    return np.array(bits, dtype=np.int8)


def decode_solution(bits: np.ndarray, grids: List[ModuleGrid]) -> Solution:
    """Inverse of encode_solution."""
    sol: Solution = {}
    pos = 0
    for g in grids:
        pairs = np.zeros((g.n_pairs, 2), dtype=np.int64)
        for i in range(g.n_pairs):
            u = 0
            for _ in range(g.bits_u):
                u = (u << 1) | int(bits[pos]); pos += 1
            v = 0
            for _ in range(g.bits_v):
                v = (v << 1) | int(bits[pos]); pos += 1
            pairs[i] = (u % g.d_out, v % g.d_in)
        sol[g.name] = pairs
    return sol


def two_point_crossover(b1: np.ndarray, b2: np.ndarray,
                        rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Two-point crossover on the concatenated bit string."""
    n = len(b1)
    a, b = sorted(rng.choice(n, size=2, replace=False).tolist())
    c1, c2 = b1.copy(), b2.copy()
    c1[a:b + 1], c2[a:b + 1] = b2[a:b + 1], b1[a:b + 1]
    return c1, c2


def bit_flip_mutation(bits: np.ndarray, p_m: float,
                      rng: np.random.Generator) -> np.ndarray:
    """Bit-flip mutation with probability p_m per bit."""
    mask = rng.random(len(bits)) < p_m
    out = bits.copy()
    out[mask] = 1 - out[mask]
    return out


def dedup_pairs(sol: Solution, grids: List[ModuleGrid],
                rng: np.random.Generator) -> Solution:
    """Remove duplicate pairs (after their conjugate partners) and refill."""
    out: Solution = {}
    for g in grids:
        pairs = sol[g.name].tolist()
        seen = set()
        uniq = []
        for u, v in pairs:
            pu, pv = conjugate_partner(g, u, v)
            key = (min((u, v), (pu, pv)), max((u, v), (pu, pv)))
            if key not in seen:
                seen.add(key)
                uniq.append((u, v))
        while len(uniq) < g.n_pairs:
            u, v = int(rng.integers(0, g.d_out)), int(rng.integers(0, g.d_in))
            pu, pv = conjugate_partner(g, u, v)
            key = (min((u, v), (pu, pv)), max((u, v), (pu, pv)))
            if key not in seen:
                seen.add(key)
                uniq.append((u, v))
        out[g.name] = np.array(uniq[: g.n_pairs], dtype=np.int64)
    return out


# ---------------------------------------------------------------------------
# PSO discrete move (Section 3.2, Eq. (6))
# ---------------------------------------------------------------------------
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def discrete_move(sol: Solution, velocities: Dict[str, np.ndarray],
                  grids: List[ModuleGrid],
                  rng: np.random.Generator) -> Tuple[Solution, Dict[str, np.ndarray]]:
    """Apply the discrete PSO update of Eq. (6).

    For each integer coordinate z with velocity v:
        z' = (z + 1) mod d   if r < sigmoid(v)
        z' = (z - 1) mod d   otherwise
    Moving a coordinate also moves its conjugate partner, preserving pairing.
    """
    new_sol: Solution = {}
    new_vel: Dict[str, np.ndarray] = {}
    for g in grids:
        pairs = sol[g.name].copy()
        vel = velocities[g.name].copy()
        for i in range(len(pairs)):
            for axis, dim in ((0, g.d_out), (1, g.d_in)):
                z = int(pairs[i, axis])
                v = float(vel[i, axis])
                r = float(rng.random())
                step = 1 if r < sigmoid(v) else -1
                # Move the coordinate and its conjugate partner together.
                pairs[i, axis] = (z + step) % dim
        new_sol[g.name] = pairs
        new_vel[g.name] = vel
    return new_sol, new_vel


# ---------------------------------------------------------------------------
# Cuckoo Search Lévy flights (Section 3.2)
# ---------------------------------------------------------------------------
def levy_step_size(rng: np.random.Generator, lam: float = 1.5,
                   base: float = 0.5) -> float:
    """Mantegna-style Lévy step: heavy-tailed, mostly small steps."""
    u = rng.standard_normal()
    v = abs(rng.standard_normal()) + 1e-12
    step = base * u / (v ** (1.0 / lam))
    return float(min(1.0, max(0.01, abs(step))))


def levy_perturb(sol: Solution, grids: List[ModuleGrid],
                 rng: np.random.Generator, lam: float = 1.5,
                 base: float = 0.5) -> Solution:
    """Perturb a random subset of pairs with Lévy-distributed step sizes."""
    out: Solution = {}
    for g in grids:
        pairs = sol[g.name].copy()
        for i in range(len(pairs)):
            if rng.random() < 0.5:
                s = levy_step_size(rng, lam=lam, base=base)
                scale_u = max(1, int(s * g.d_out * 0.1))
                scale_v = max(1, int(s * g.d_in * 0.1))
                du = int(rng.integers(-scale_u, scale_u + 1))
                dv = int(rng.integers(-scale_v, scale_v + 1))
                pairs[i, 0] = (int(pairs[i, 0]) + du) % g.d_out
                pairs[i, 1] = (int(pairs[i, 1]) + dv) % g.d_in
        out[g.name] = pairs
    return out


def cosine_step(it: int, n_iter: int, start: float = 0.5,
                end: float = 0.1) -> float:
    """Cosine-decayed step size used by Cuckoo Search."""
    t = min(1.0, it / max(1, n_iter - 1))
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * t))


# ---------------------------------------------------------------------------
# Surrogate features
# ---------------------------------------------------------------------------
def solution_features(sol: Solution, grids: List[ModuleGrid],
                      band_edges: Tuple[float, float] = (200.0, 400.0)) -> np.ndarray:
    """Statistical features of a coordinate set for the RF surrogate.

    For each module: share of pairs in the low / middle / high band (band
    edges are distances from the spectral center), mean and std of the
    distance to the center, and row/column occupancy entropy.
    """
    feats: List[float] = []
    for g in grids:
        pairs = sol[g.name]
        dists = np.array([g.distance_to_center(int(u), int(v))
                          for u, v in pairs])
        low = float(np.mean(dists < band_edges[0]))
        mid = float(np.mean((dists >= band_edges[0]) & (dists < band_edges[1]))
                    ) if band_edges[1] > band_edges[0] else 0.0
        high = float(np.mean(dists >= band_edges[1]))
        mean_d = float(dists.mean()) / max(1.0, (g.d_out + g.d_in) / 2.0)
        std_d = float(dists.std()) / max(1.0, (g.d_out + g.d_in) / 2.0)
        row_hist = np.bincount(pairs[:, 0], minlength=g.d_out).astype(float)
        col_hist = np.bincount(pairs[:, 1], minlength=g.d_in).astype(float)
        row_ent = _entropy(row_hist)
        col_ent = _entropy(col_hist)
        feats.extend([low, mid, high, mean_d, std_d, row_ent, col_ent])
    return np.array(feats, dtype=np.float64)


def _entropy(hist: np.ndarray) -> float:
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def solution_to_indices(sol: Solution, grids: List[ModuleGrid]) -> Dict[str, np.ndarray]:
    """Expand conjugate pairs into a [2, 2*n_pairs] index array per module.

    Each stored pair (u, v) is expanded together with its conjugate partner
    (-u mod d_out, -v mod d_in), so the saved array holds one coordinate per
    trainable coefficient. The result is ready to be saved with np.savez and
    consumed by FourierFTConfig(sampling="indices").
    """
    out: Dict[str, np.ndarray] = {}
    for g in grids:
        pairs = sol[g.name]
        us, vs = [], []
        for u, v in pairs:
            us.append(int(u)); vs.append(int(v))
            pu, pv = conjugate_partner(g, int(u), int(v))
            us.append(pu); vs.append(pv)
        out[g.name] = np.stack([np.array(us, dtype=np.int64),
                                np.array(vs, dtype=np.int64)], axis=0)
    return out
