"""Drill 3 - CTC from scratch: the collapse rule and the forward algorithm (stt/ctc.py, one clip, no batch).

    python d03_ctc_loss.py              # checks YOUR functions below
    python d03_ctc_loss.py --solution   # checks solutions/d03_ctc_loss.py

Blank id is 0. Write:
  collapse(path)                 -> list: merge repeats, THEN drop blanks.  [1,1,0,1,2,2] -> [1,1,2]
  ctc_loss(log_probs, target)    -> float: -log P(target | audio), log_probs (T, V) log-softmax, target list
                                    of ids (no blanks). Forward algorithm over 2L+1 states in log space.
Plain Python / math or torch tensors are fine; F.ctc_loss is not. About 20 minutes; the classic whiteboard one.
"""

import sys
from pathlib import Path

import torch


def collapse(path):
    raise NotImplementedError


def ctc_loss(log_probs, target):
    raise NotImplementedError


def check(m) -> None:
    assert m.collapse([1, 1, 0, 1, 2, 2]) == [1, 1, 2], "merge repeats first, then drop blanks"
    assert m.collapse([0, 0, 0]) == [] and m.collapse([3, 0, 3]) == [3, 3] and m.collapse([3, 3]) == [3]

    two = torch.tensor([[0.6, 0.4], [0.6, 0.4]]).log()
    got = float(m.ctc_loss(two, [1]))
    assert abs(got - 0.4463) < 1e-3, f'2 frames, P("a") = 0.24 + 0.24 + 0.16 = 0.64, loss 0.4463; got {got:.4f}'

    torch.manual_seed(0)
    for T, target in [(30, [3, 5, 5, 2]), (12, [1, 1, 1]), (50, [7, 1, 9, 9, 4, 2, 2, 6]), (6, [4])]:
        lp = torch.randn(T, 10).log_softmax(-1)
        want = torch.nn.functional.ctc_loss(lp[:, None], torch.tensor([target]), torch.tensor([T]), torch.tensor([len(target)]), reduction="none").item()
        got = float(m.ctc_loss(lp, target))
        assert abs(got - want) < 1e-3, f"T={T} target={target}: got {got:.4f}, PyTorch {want:.4f} (skip between equal letters?)"
    print("PASS  collapse; ctc_loss = -ln 0.64 on the 2-frame example and matches F.ctc_loss on 4 clips (repeats included)")


if __name__ == "__main__":
    if "--solution" in sys.argv:
        sys.path.insert(0, str(Path(__file__).parent / "solutions"))
        import d03_ctc_loss as mod
    else:
        mod = sys.modules[__name__]
    check(mod)
