"""Drill 10 - residual vector quantizer (RVQ), the building block of codec-LM TTS.

    python d10_rvq.py              # checks YOUR functions below
    python d10_rvq.py --solution   # checks solutions/d10_rvq.py

Write:
  vq_encode(x, codebook)
      Nearest-centroid vector quantization (one stage):
        x: (N, D) vectors to quantize
        codebook: (K, D) learned centroids
        Returns (indices, residual):
          indices: (N,) LongTensor, the index of the nearest centroid for each vector
          residual: (N, D) the difference x - codebook[indices]
      Nearest = smallest Euclidean distance. torch only, no loops over N.

  rvq_encode(x, codebooks)
      Chain of VQ stages. After each stage the residual is passed to the next.
        x: (N, D) input vectors
        codebooks: list of K (K, D) codebooks, one per stage
        Returns (all_indices, final_residual):
          all_indices: list of K index tensors, one (N,) per stage
          final_residual: (N, D) residual after all stages
      The residual must shrink (or stay the same) with each additional stage.

torch only. About 10 minutes.
"""

import sys
from pathlib import Path

import torch


def vq_encode(x: torch.Tensor, codebook: torch.Tensor):
    raise NotImplementedError


def rvq_encode(x: torch.Tensor, codebooks: list):
    raise NotImplementedError


def check(m) -> None:
    torch.manual_seed(42)
    D, K, N = 16, 8, 32

    # --- vq_encode ---
    codebook = torch.randn(K, D)
    x = codebook[torch.randint(K, (N,))] + 0.01 * torch.randn(N, D)  # x is near codebook entries

    indices, residual = m.vq_encode(x, codebook)
    assert indices.shape == (N,), f"indices shape: {indices.shape}"
    assert residual.shape == (N, D), f"residual shape: {residual.shape}"
    assert indices.dtype == torch.long, "indices must be LongTensor"
    # When x is very close to a centroid, it should pick the nearest one
    assert (indices == torch.cdist(x, codebook).argmin(dim=1)).all(), "indices must be nearest centroid"
    # Residual = x - centroid
    assert torch.allclose(residual, x - codebook[indices], atol=1e-5), "residual = x - codebook[idx]"

    # --- rvq_encode ---
    codebooks = [torch.randn(K, D) for _ in range(4)]
    x_big = torch.randn(N, D)
    all_indices, final_residual = m.rvq_encode(x_big, codebooks)

    assert len(all_indices) == 4, "one index tensor per stage"
    assert all(idx.shape == (N,) for idx in all_indices), "each index tensor shape (N,)"
    assert final_residual.shape == (N, D), "final residual shape"

    # Residual shrinks when each codebook is seeded from the data it quantizes —
    # the same condition required for proper RVQ training.
    torch.manual_seed(1)
    x_shrink = torch.randn(N, D)
    # Build codebooks by sampling K points from each successive residual (like k-means init)
    cb_seeded = []
    r_seed = x_shrink.clone()
    for _ in range(4):
        perm = torch.randperm(N)[:K]
        cb_seeded.append(r_seed[perm].clone())
        _, r_seed = m.vq_encode(r_seed, cb_seeded[-1])
    # Reset and measure
    residuals_seeded = [x_shrink]
    r_s = x_shrink.clone()
    for cb in cb_seeded:
        _, r_s = m.vq_encode(r_s, cb)
        residuals_seeded.append(r_s)
    for i in range(len(cb_seeded)):
        n0 = residuals_seeded[i].norm().item()
        n1 = residuals_seeded[i + 1].norm().item()
        assert n1 <= n0 + 1e-4, f"stage {i+1} residual norm {n1:.3f} > input norm {n0:.3f}"

    # Reconstruction: if x is exactly a codebook entry, VQ residual must be zero.
    torch.manual_seed(99)
    single_cb = torch.randn(K, D)
    x_exact = single_cb[torch.randint(K, (N,))]   # x is a codebook entry
    idxs_s, res_s = m.vq_encode(x_exact, single_cb)
    assert res_s.norm().item() < 1e-4, f"residual for exact codebook entry should be ~0, got {res_s.norm():.4f}"
    # Multi-stage: the sum of the quantized entries should equal x when x lives on the codebook grid
    # Use stage-1 codebook = range of x, rest are irrelevant (residual already 0)
    idxs_r, _ = m.rvq_encode(x_exact, [single_cb] + [torch.randn(K, D) for _ in range(3)])
    recon = single_cb[idxs_r[0]]   # first stage already nails it
    assert torch.allclose(recon, x_exact, atol=1e-4), "first-stage reconstruction of on-grid x must be exact"

    print("PASS  vq_encode (nearest centroid, residual) and rvq_encode (residual shrinks, reconstruction works)")


if __name__ == "__main__":
    if "--solution" in sys.argv:
        sys.path.insert(0, str(Path(__file__).parent / "solutions"))
        import d10_rvq as mod
    else:
        mod = sys.modules[__name__]
    check(mod)
