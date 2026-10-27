"""Solution to drill 3. The same recursion as stt/ctc.py, for one clip, in plain Python."""

import math

BLANK = 0
NEG = -1e30


def collapse(path):
    merged = [s for i, s in enumerate(path) if i == 0 or s != path[i - 1]]
    return [s for s in merged if s != BLANK]


def _log_add(*xs):
    top = max(xs)
    return top + math.log(sum(math.exp(x - top) for x in xs)) if top > NEG else NEG


def ctc_loss(log_probs, target):
    lp = log_probs.tolist()
    ext = [BLANK]
    for c in target:
        ext += [c, BLANK]  # - c1 - c2 - ... -
    S = len(ext)
    alpha = [NEG] * S
    alpha[0] = lp[0][ext[0]]
    if S > 1:
        alpha[1] = lp[0][ext[1]]
    for t in range(1, len(lp)):
        new = [NEG] * S
        for s in range(S):
            moves = [alpha[s]]  # stay
            if s >= 1:
                moves.append(alpha[s - 1])  # step
            if s >= 2 and ext[s] != BLANK and ext[s] != ext[s - 2]:
                moves.append(alpha[s - 2])  # skip the blank between two different letters
            new[s] = _log_add(*moves) + lp[t][ext[s]]
        alpha = new
    return -(_log_add(alpha[-1], alpha[-2]) if S > 1 else alpha[-1])
