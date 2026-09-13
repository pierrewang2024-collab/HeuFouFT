# Frequency-coordinate sampling strategies for FourierFT.
#
# Three sampling modes are supported:
#   - "uniform":            uniformly random coordinates (vanilla FourierFT).
#   - "bandpass":           Gaussian band-pass sampling around an annulus
#                           controlled by a frequency bias `fc` (distance from
#                           the spectral center) and a bandwidth `width`.
#   - "indices":            load a fixed set of coordinates from a .npz file
#                           produced by the HeuFouFT search pipeline.
#
# Coordinates are expressed on the (out_features x in_features) spectrum grid
# of each target layer. Conjugate pairing, (u, v) <-> (-u mod d_out,
# -v mod d_in), is optionally enforced so that the inverse transform yields a
# real-valued weight update with symmetric spectral control.

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def uniform_indices(out_features: int, in_features: int, n_frequency: int,
                    rng: np.random.Generator) -> torch.Tensor:
    """Vanilla FourierFT: uniformly random coordinates without replacement."""
    flat = rng.choice(out_features * in_features, size=n_frequency, replace=False)
    rows = flat // in_features
    cols = flat % in_features
    return torch.from_numpy(np.stack([rows, cols], axis=0)).long()


def bandpass_indices(out_features: int, in_features: int, n_frequency: int,
                     rng: np.random.Generator, fc: float = 0.0,
                     width: float = 20.0) -> torch.Tensor:
    """Gaussian band-pass sampling.

    Each grid cell (u, v) receives an acceptance weight proportional to
    exp(-(r - fc)^2 / (2 * width^2)), where r is the Euclidean distance of the
    cell from the spectral center (d_out/2, d_in/2). Coordinates are then drawn
    without replacement with probability proportional to the weight. Setting
    fc = 0 yields a low-pass region around the center.
    """
    us = np.arange(out_features)[:, None]
    vs = np.arange(in_features)[None, :]
    center_u, center_v = out_features / 2.0, in_features / 2.0
    # Distance on the wrapped (periodic) grid so the annulus is circular.
    du = np.minimum(np.abs(us - center_u), out_features - np.abs(us - center_u))
    dv = np.minimum(np.abs(vs - center_v), in_features - np.abs(vs - center_v))
    r = np.sqrt(du ** 2 + dv ** 2)
    w = np.exp(-((r - fc) ** 2) / (2.0 * max(width, 1e-6) ** 2)).flatten()
    w = w / w.sum()

    if n_frequency > out_features * in_features:
        raise ValueError("n_frequency exceeds the spectrum grid size")

    # Weighted sampling without replacement.
    flat = rng.choice(out_features * in_features, size=n_frequency,
                      replace=False, p=w)
    flat = np.sort(flat)  # deterministic ordering for reproducibility
    rows = flat // in_features
    cols = flat % in_features
    return torch.from_numpy(np.stack([rows, cols], axis=0)).long()


def load_indices(npz_path: str, module_name: Optional[str] = None) -> torch.Tensor:
    """Load fixed coordinates from a .npz file.

    The file may contain one integer array of shape [2, n_frequency] per
    target module (keyed by the module name, e.g. "c_attn" / "c_proj") or a
    single shared array under the key "indices". Row 0 holds spectrum rows
    (out dimension) and row 1 holds spectrum columns (in dimension).
    """
    data = np.load(npz_path)
    candidates = []
    if module_name is not None:
        # Try the exact module name, then the name with the adapter suffix.
        candidates.append(module_name.split(".")[-1])
    candidates.append("indices")
    for key in candidates:
        if key in data.files:
            idx = data[key]
            if idx.ndim != 2 or idx.shape[0] != 2:
                raise ValueError(f"Expected indices of shape [2, n], got {idx.shape}")
            return torch.from_numpy(idx.astype(np.int64))
    raise KeyError(
        f"None of the keys {candidates} found in {npz_path}; "
        f"available keys: {list(data.files)}"
    )


def make_indices(mode: str, out_features: int, in_features: int,
                 n_frequency: int, rng: np.random.Generator,
                 fc: float = 0.0, width: float = 20.0,
                 indices_file: str | None = None,
                 module_name: str | None = None) -> torch.Tensor:
    """Dispatch to the requested sampling mode."""
    if mode == "uniform":
        return uniform_indices(out_features, in_features, n_frequency, rng)
    if mode == "bandpass":
        return bandpass_indices(out_features, in_features, n_frequency, rng,
                                fc=fc, width=width)
    if mode == "indices":
        if indices_file is None:
            raise ValueError("indices mode requires an indices file")
        return load_indices(indices_file, module_name=module_name)
    raise ValueError(f"Unknown sampling mode: {mode}")
