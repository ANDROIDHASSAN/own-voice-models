"""Drill 7 - Monotonic Alignment Search (tts/align.py), one utterance, no batch.

    python d07_mas.py              # checks YOUR function below
    python d07_mas.py --solution   # checks solutions/d07_mas.py

score[l][t] = how well character l matches mel frame t (higher is better), L characters, T >= L frames.
Write mas(score) -> list of L durations: the best MONOTONIC path from (0, 0) to (L-1, T-1) where every frame
goes to exactly one character, in order, none skipped. From (l, t-1) a path may stay on l or step to l+1.
Durations sum to T and are all >= 1. About 15 minutes: forward max table + backtrack.
"""

import sys
from pathlib import Path

import torch


def mas(score):
    raise NotImplementedError


def check(m) -> None:
    sys.path.insert(0, str(Path(__file__).parents[1] / "tts"))
    import align as ref

    torch.manual_seed(0)
    sounds = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]])
    true = torch.tensor([3, 1, 5, 2])
    mel = torch.repeat_interleave(sounds, true, dim=0) + 0.3 * torch.randn(11, 3)
    score = -((sounds[:, None, :] - mel[None, :, :]) ** 2).sum(-1)
    got = list(m.mas(score.tolist()))
    assert got == [3, 1, 5, 2], f"align.py demo: want [3, 1, 5, 2], got {got}"

    assert list(m.mas([[0.0, 0.0, 0.0]] * 3)) == [1, 1, 1], "T == L: one frame each"
    for _ in range(25):
        L, T = int(torch.randint(1, 7, ())), int(torch.randint(7, 20, ()))
        s = torch.randn(L, T)
        got = list(m.mas(s.tolist()))
        want = ref.mas(s[None], torch.tensor([L]), torch.tensor([T]))[0].tolist()
        assert sum(got) == T and min(got) >= 1, f"durations {got} must be >= 1 and sum to {T}"
        assert got == want, f"L={L} T={T}: got {got}, tts/align.py says {want}"
    print("PASS  MAS recovers [3, 1, 5, 2]; durations >= 1, sum to T, match tts/align.py on 25 random tables")


if __name__ == "__main__":
    if "--solution" in sys.argv:
        sys.path.insert(0, str(Path(__file__).parent / "solutions"))
        import d07_mas as mod
    else:
        mod = sys.modules[__name__]
    check(mod)
