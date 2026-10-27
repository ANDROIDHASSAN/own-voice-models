"""Drill 4 - CTC decoding: greedy and prefix beam search (stt/ctc.py:82-124), one clip, blank id 0.

    python d04_ctc_decoding.py              # checks YOUR functions below
    python d04_ctc_decoding.py --solution   # checks solutions/d04_ctc_decoding.py

Write:
  greedy(log_probs)              -> list of ids: argmax per frame, merge repeats, drop blanks
  beam_search(log_probs, beam)   -> list of ids: the most likely TEXT. Keep `beam` prefixes, each with two
                                    log-probabilities: paths ending in blank, paths ending in the last letter.
log_probs is a (T, V) torch tensor of log-softmax values. About 25 minutes; the p_blank / p_non_blank split
for a repeated letter is the part interviewers watch.
"""

import sys
from pathlib import Path

import torch


def greedy(log_probs):
    raise NotImplementedError


def beam_search(log_probs, beam=8):
    raise NotImplementedError


def check(m) -> None:
    sys.path.insert(0, str(Path(__file__).parents[1] / "stt"))
    import ctc as ref

    two = torch.tensor([[0.6, 0.4], [0.6, 0.4]]).log()
    assert m.greedy(two) == [], "best single path is blank-blank"
    assert m.beam_search(two) == [1], 'best TEXT is "a": 0.24 + 0.24 + 0.16 = 0.64 > 0.36'

    path = torch.full((7, 4), -10.0)
    for t, s in enumerate([1, 1, 0, 1, 2, 2, 0]):
        path[t, s] = 0.0
    assert m.greedy(path.log_softmax(-1)) == [1, 1, 2], "merge repeats, then drop blanks"

    torch.manual_seed(0)
    for _ in range(20):
        lp = (torch.randn(15, 5) * 3).log_softmax(-1)
        assert m.greedy(lp) == ref.greedy(lp[None], torch.tensor([15]))[0], "greedy differs from stt/ctc.py"
        assert m.beam_search(lp, 8) == ref.beam_search(lp, 8), "beam search differs from stt/ctc.py (repeated letters?)"
    print("PASS  greedy; beam search finds \"a\" where greedy finds \"\", and matches stt/ctc.py on 20 random clips")


if __name__ == "__main__":
    if "--solution" in sys.argv:
        sys.path.insert(0, str(Path(__file__).parent / "solutions"))
        import d04_ctc_decoding as mod
    else:
        mod = sys.modules[__name__]
    check(mod)
