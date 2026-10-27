"""Drill 2 - padding mask + bidirectional self-attention (the encoder version of stt/model.py:118-145).

    python d02_padding_mask.py              # checks YOUR functions below
    python d02_padding_mask.py --solution   # checks solutions/d02_padding_mask.py

Clips in a batch have different lengths, so short ones are padded. Write:
  frame_mask(lens, T)   -> (B, T) bool, True on real frames
  attention(q, k, v, real) -> (B, T, d) single-head scaled dot-product attention with NO causal mask
                              (an encoder sees the whole clip) but no query may attend TO a padded frame.
torch is allowed, nn.MultiheadAttention / F.scaled_dot_product_attention are not. About 10 minutes.
"""

import sys
from pathlib import Path

import torch


def frame_mask(lens, T):
    raise NotImplementedError


def attention(q, k, v, real):
    raise NotImplementedError


def check(m) -> None:
    lens = torch.tensor([3, 5])
    mask = m.frame_mask(lens, 5)
    assert mask.dtype == torch.bool and mask.tolist() == [[True] * 3 + [False] * 2, [True] * 5], mask

    torch.manual_seed(0)
    d = 8
    short, long = torch.randn(3, d), torch.randn(5, d)
    batch = torch.full((2, 5, d), 99.0)  # garbage in the padding must not matter
    batch[0, :3], batch[1] = short, long
    alone = m.attention(short[None], short[None], short[None], m.frame_mask(torch.tensor([3]), 3))
    both = m.attention(batch, batch, batch, m.frame_mask(lens, 5))
    assert alone.shape == (1, 3, d) and both.shape == (2, 5, d), (alone.shape, both.shape)
    assert torch.allclose(alone[0], both[0, :3], atol=1e-5), "padding leaks into the short clip's output"

    q = torch.randn(1, 4, d)
    full = m.attention(q, q, q, torch.ones(1, 4, dtype=torch.bool))
    ref = torch.softmax(q @ q.transpose(-2, -1) / d**0.5, -1) @ q
    assert torch.allclose(full, ref, atol=1e-5), "not plain scaled dot-product attention (causal mask by mistake?)"
    print("PASS  frame_mask, attention: same output alone and padded in a batch; frame 0 can see the future")


if __name__ == "__main__":
    if "--solution" in sys.argv:
        sys.path.insert(0, str(Path(__file__).parent / "solutions"))
        import d02_padding_mask as mod
    else:
        mod = sys.modules[__name__]
    check(mod)
