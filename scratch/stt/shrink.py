"""Lesson 9 - quantization: the same idea as ../llm/shrink.py, measured on a speech model.

    python shrink.py

Read ../llm/shrink.py first for what fp16, int8, per-tensor and per-channel mean. The question here is
whether a speech encoder tolerates it as well as the language model did. The measurement is the same:
quantize the weights, run the SAME dev clips, compare the character error rate.

Size matters more here than for the LLM: this model is meant to be downloaded by a browser.
"""

import json
from pathlib import Path

import torch

from data import DATA, open_split
from model import load
from text import Vocab
from train import evaluate

HERE = Path(__file__).parent


def quant_roundtrip(w: torch.Tensor, bits: int, per_channel: bool) -> torch.Tensor:
    """Quantize to `bits`-bit integers and back to float, so the usual forward pass can show the damage.
    Works for linear (2-D) and convolution (3-D, 4-D) weights: a "channel" is one output row / filter."""
    levels = 2 ** (bits - 1) - 1
    peak = w.abs().flatten(1).amax(dim=1).view(-1, *[1] * (w.dim() - 1)) if per_channel else w.abs().max()
    scale = peak.clamp(min=1e-8) / levels
    return (w / scale).round().clamp(-levels, levels) * scale


def main() -> None:
    model, ckpt = load(HERE / "runs/ckpt.pt")
    device = str(next(model.parameters()).device)
    vocab = Vocab(ckpt["chars"])
    mel, index = open_split("dev")
    ids = list(range(len(index)))
    original = {k: v.clone() for k, v in model.state_dict().items()}
    n_params = model.n_params()
    assert json.loads((DATA / "meta.json").read_text())["chars"] == ckpt["chars"]

    def dev_cer() -> float:
        scores, _ = evaluate(model, mel, index, ids, vocab, device, 16000)
        return sum(s[0] for s in scores.values()) / len(scores)

    base = dev_cer()
    print(f"{n_params / 1e6:.2f} M parameters\n")
    print(f"{'format':20s}{'size':>10s}{'dev CER':>10s}{'change':>10s}")
    print(f"{'fp32':20s}{n_params * 4 / 1e6:>8.1f} MB{base:>10.4f}")
    for bits, per_channel in [(16, False), (8, True), (8, False), (6, True), (4, True), (4, False), (3, True), (2, True)]:
        if bits == 16:
            quantized = {k: v.half().float() if v.dim() >= 2 else v for k, v in original.items()}
        else:
            quantized = {k: quant_roundtrip(v, bits, per_channel) if v.dim() >= 2 else v for k, v in original.items()}
        model.load_state_dict(quantized)
        cer = dev_cer()
        name = "fp16" if bits == 16 else f"int{bits}, {'per-channel' if per_channel else 'per-tensor'}"
        print(f"{name:20s}{n_params * bits / 8 / 1e6:>8.1f} MB{cer:>10.4f}{cer - base:>+10.4f}", flush=True)
    model.load_state_dict(original)


if __name__ == "__main__":
    main()
