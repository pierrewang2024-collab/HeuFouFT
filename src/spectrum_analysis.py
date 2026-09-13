# Spectral analysis of pretrained GPT-2-Medium weights (Figure 1).
#
# Extracts the layer-1 K and layer-24 V projection weights from the fused
# c_attn matrix, computes the log-magnitude 2-D FFT spectra, and renders the
# four-panel figure (weights + spectra) with the dominant spectral peaks
# marked.
#
# Usage:
#   python src/spectrum_analysis.py --model gpt2-medium \
#       --output figures/gpt2_fourpanel.pdf

import argparse
import os

import numpy as np
import torch
from transformers import GPT2LMHeadModel


def extract_projection(model, layer_idx: int, projection: str):
    """Return the (in, out) projection weight from a GPT-2 attention layer.

    GPT-2 stores Q, K, V in a single fused `c_attn` Conv1D of shape
    [hidden, 3 * hidden]; `projection` in {"q", "k", "v"} selects the
    corresponding slice.
    """
    w = model.transformer.h[layer_idx].attn.c_attn.weight.detach().cpu().numpy()
    hidden = w.shape[0]
    qkv = {"q": 0, "k": 1, "v": 2}[projection]
    return w[:, qkv * hidden:(qkv + 1) * hidden]


def find_peaks(spectrum_log: np.ndarray, k: int = 12):
    """Return the (row, col) coordinates of the k largest spectral peaks."""
    flat = np.argsort(spectrum_log, axis=None)[::-1][:k]
    rows, cols = np.unravel_index(flat, spectrum_log.shape)
    # Exclude the DC component (center after fftshift).
    center = np.array(spectrum_log.shape) // 2
    keep = ~((rows == center[0]) & (cols == center[1]))
    return rows[keep], cols[keep]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="gpt2-medium")
    p.add_argument("--output", type=str, default="figures/gpt2_fourpanel.pdf")
    p.add_argument("--n_peaks", type=int, default=12)
    args = p.parse_args()

    print(f"Loading {args.model} ...")
    model = GPT2LMHeadModel.from_pretrained(args.model)
    model.eval()

    panels = [
        ("Layer 1 $K$", extract_projection(model, 0, "k")),
        ("Layer 24 $V$", extract_projection(model, 23, "v")),
    ]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    for col, (title, W) in enumerate(panels):
        ax_w = axes[0, col]
        ax_s = axes[1, col]

        im0 = ax_w.imshow(W, cmap="gray", aspect="auto")
        ax_w.set_title(f"{title} weight")
        fig.colorbar(im0, ax=ax_w, fraction=0.046)

        spec = np.fft.fftshift(np.fft.fft2(W))
        log_mag = np.log10(np.abs(spec) + 1e-12)
        im1 = ax_s.imshow(log_mag, cmap="magma", aspect="auto")
        ax_s.set_title(f"{title} log-magnitude spectrum")
        fig.colorbar(im1, ax=ax_s, fraction=0.046)

        rows, cols = find_peaks(log_mag, k=args.n_peaks)
        ax_s.scatter(cols, rows, s=18, facecolors="none",
                     edgecolors="lime", linewidths=1.0)
        ax_s.set_xlim(0, log_mag.shape[1] - 1)
        ax_s.set_ylim(log_mag.shape[0] - 1, 0)

    fig.tight_layout()
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Figure saved to {args.output}")


if __name__ == "__main__":
    main()
