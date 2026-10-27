"""Solution for d10_rvq.py"""

import torch


def vq_encode(x: torch.Tensor, codebook: torch.Tensor):
    """Nearest-centroid VQ.  Returns (indices, residual)."""
    # (N, K) pairwise distances; argmin gives nearest centroid index
    dists = torch.cdist(x, codebook)           # (N, K)
    indices = dists.argmin(dim=1)              # (N,)
    residual = x - codebook[indices]           # (N, D)
    return indices, residual


def rvq_encode(x: torch.Tensor, codebooks: list):
    """Chain of VQ stages.  Returns (list-of-indices, final-residual)."""
    all_indices = []
    residual = x
    for codebook in codebooks:
        indices, residual = vq_encode(residual, codebook)
        all_indices.append(indices)
    return all_indices, residual
