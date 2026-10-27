"""Drill 8 - the acoustic model's two small but central pieces (tts/model.py).

    python d08_length_regulator_flow.py              # checks YOUR functions below
    python d08_length_regulator_flow.py --solution   # checks solutions/d08_length_regulator_flow.py

Write:
  length_regulate(h, durations)          h (L, d) character vectors, durations (L,) ints -> (sum(durations), d):
                                          vector i repeated durations[i] times (model.py:117-120, one clip)
  cfm_pair(x0, x1, t, sigma_min)         conditional flow matching training pair (model.py:205-207):
                                          x_t = (1 - (1 - sigma_min) t) x0 + t x1,  u = x1 - (1 - sigma_min) x0
  euler_sample(velocity, x0, steps)      start at noise x0, take `steps` Euler steps x += v(x, t) / steps with
                                          t = i / steps (model.py:239-246)
torch only. About 10 minutes.
"""

import sys
from pathlib import Path

import torch


def length_regulate(h, durations):
    raise NotImplementedError


def cfm_pair(x0, x1, t, sigma_min=1e-4):
    raise NotImplementedError


def euler_sample(velocity, x0, steps):
    raise NotImplementedError


def check(m) -> None:
    h = torch.arange(6.0).reshape(3, 2)
    out = m.length_regulate(h, torch.tensor([2, 1, 3]))
    assert out.shape == (6, 2) and out[:, 0].tolist() == [0, 0, 2, 4, 4, 4], out
    slow = m.length_regulate(h, (torch.tensor([2, 1, 3]) / 0.5).round().long())
    assert slow.shape[0] == 12, "speed 0.5 = durations / 0.5 = twice as many frames, same vectors (pitch unchanged)"

    torch.manual_seed(0)
    x0, x1 = torch.randn(5, 4), torch.randn(5, 4)
    xt, u = m.cfm_pair(x0, x1, 0.0, 1e-4)
    assert torch.allclose(xt, x0), "t = 0 is pure noise"
    xt, u = m.cfm_pair(x0, x1, 1.0, 1e-4)
    assert torch.allclose(xt, x1 + 1e-4 * x0, atol=1e-6), "t = 1 is the data, up to sigma_min"
    xa, _ = m.cfm_pair(x0, x1, 0.3, 1e-4)
    xb, _ = m.cfm_pair(x0, x1, 0.7, 1e-4)
    assert torch.allclose(xb - xa, 0.4 * u, atol=1e-5), "the path is a straight line with constant velocity u"

    # A perfect velocity field (the true u) carries noise exactly to the data in ANY number of steps.
    for steps in (1, 4, 10):
        got = m.euler_sample(lambda x, t: u, x0, steps)
        assert torch.allclose(got, x0 + u, atol=1e-5), f"{steps} Euler steps of the straight-line velocity"
    print("PASS  length_regulate (and speed), CFM pair is a straight line x0 -> x1, Euler sampling lands on the data")


if __name__ == "__main__":
    if "--solution" in sys.argv:
        sys.path.insert(0, str(Path(__file__).parent / "solutions"))
        import d08_length_regulator_flow as mod
    else:
        mod = sys.modules[__name__]
    check(mod)
