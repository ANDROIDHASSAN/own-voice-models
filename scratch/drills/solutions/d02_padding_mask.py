"""Solution to drill 2. Same idea as stt/model.py:118-145, one head."""

import math

import torch


def frame_mask(lens, T):
    return torch.arange(T)[None, :] < lens[:, None]


def attention(q, k, v, real):
    att = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])  # (B, T, T)
    att = att.masked_fill(~real[:, None, :], float("-inf"))  # mask the KEYS: nobody looks at padding
    return torch.softmax(att, dim=-1) @ v
