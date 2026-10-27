"""Solution to drill 8. Same maths as tts/model.py, one clip."""

import torch


def length_regulate(h, durations):
    return torch.repeat_interleave(h, durations, dim=0)


def cfm_pair(x0, x1, t, sigma_min=1e-4):
    x_t = (1 - (1 - sigma_min) * t) * x0 + t * x1
    u = x1 - (1 - sigma_min) * x0  # the straight line's velocity: the same at every t
    return x_t, u


def euler_sample(velocity, x0, steps):
    x = x0
    for i in range(steps):
        x = x + velocity(x, i / steps) / steps
    return x
