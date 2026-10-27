"""Lesson 8 - quantization: making the model small enough for a phone, and measuring what it costs.

    python shrink.py

Weights are trained as 32-bit floats (fp32, 4 bytes each). They do not need that much precision to work.

    fp16   half precision, 2 bytes per weight
    int8   1 byte per weight: store round(w / scale) as an integer in [-127, 127], and multiply by `scale`
           when the weight is needed

Quantizing a finished model like this is post-training quantization (PTQ). One `scale` for a whole matrix
is per-tensor; one per output row is per-channel, which loses less because a row with small weights is no
longer forced onto the grid of a row with large ones.

The honest way to judge it: measure the validation loss before and after on the SAME batches.
"""

from pathlib import Path

import numpy as np
import torch

from model import load
from train import estimate_loss

HERE = Path(__file__).parent


def quant_roundtrip(w: torch.Tensor, bits: int, per_channel: bool) -> torch.Tensor:
    """Quantize to `bits`-bit integers and back to float, so the usual forward pass can show the damage."""
    levels = 2 ** (bits - 1) - 1  # 127 for int8, 7 for 4-bit
    peak = w.abs().amax(dim=1, keepdim=True) if per_channel else w.abs().max()
    scale = peak.clamp(min=1e-8) / levels
    q = (w / scale).round().clamp(-levels, levels)  # these small integers are what would be stored
    return q * scale


def main() -> None:
    model, _ = load(HERE / "runs/ckpt.pt")
    device = str(next(model.parameters()).device)
    val = np.memmap(HERE / "data/val.bin", dtype=np.uint16, mode="r")
    original = {k: v.clone() for k, v in model.state_dict().items()}
    n = model.n_params()

    def measure(name: str, bits: int, convert) -> None:
        model.load_state_dict({k: convert(v) if v.dim() == 2 else v for k, v in original.items()})
        np.random.seed(0)  # the same validation batches for every row of the table
        loss = estimate_loss(model, val, 32, device, n=30)
        print(f"{name:18s} {n * bits / 8 / 1e6:6.1f} MB   val loss {loss:.4f}")

    measure("fp32 (trained)", 32, lambda w: w)
    measure("fp16", 16, lambda w: w.half().float())
    for bits in (8, 4, 3, 2):  # keep going down until it breaks: that is how you find the real limit
        measure(f"int{bits} per-channel", bits, lambda w: quant_roundtrip(w, bits, per_channel=True))
        measure(f"int{bits} per-tensor", bits, lambda w: quant_roundtrip(w, bits, per_channel=False))


if __name__ == "__main__":
    main()
