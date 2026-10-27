"""Solution to drill 4. Same logic as stt/ctc.py:82-124, one clip."""

import math
from collections import defaultdict

BLANK = 0
NEG = -1e30


def greedy(log_probs):
    best = log_probs.argmax(-1).tolist()
    merged = [s for i, s in enumerate(best) if i == 0 or s != best[i - 1]]
    return [s for s in merged if s != BLANK]


def _log_add(*xs):
    top = max(xs)
    return top + math.log(sum(math.exp(x - top) for x in xs))


def beam_search(log_probs, beam=8):
    beams = {(): (0.0, NEG)}  # prefix -> (log p ending in blank, log p ending in its last letter)
    for frame in log_probs:
        nxt = defaultdict(lambda: [NEG, NEG])
        top_p, top_c = frame.topk(min(len(frame), 2 * beam))
        for prefix, (p_b, p_nb) in beams.items():
            for p, c in zip(top_p.tolist(), top_c.tolist()):
                if c == BLANK:
                    nxt[prefix][0] = _log_add(nxt[prefix][0], p_b + p, p_nb + p)
                elif prefix and prefix[-1] == c:
                    nxt[prefix + (c,)][1] = _log_add(nxt[prefix + (c,)][1], p_b + p)  # blank between: a new letter
                    nxt[prefix][1] = _log_add(nxt[prefix][1], p_nb + p)  # same letter held longer
                else:
                    nxt[prefix + (c,)][1] = _log_add(nxt[prefix + (c,)][1], p_b + p, p_nb + p)
        ranked = sorted(nxt.items(), key=lambda kv: _log_add(*kv[1]), reverse=True)
        beams = {prefix: (p[0], p[1]) for prefix, p in ranked[:beam]}
    return list(max(beams.items(), key=lambda kv: _log_add(*kv[1]))[0])
