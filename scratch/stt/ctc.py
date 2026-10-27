"""Lesson 4 - CTC: how to train on (audio, text) pairs when nobody tells you WHEN each letter was said.

    python ctc.py

The problem. Five seconds of audio is 125 frames after the model's downsampling. The transcript is maybe
40 characters. Which frame belongs to which character? Nobody labelled that, and labelling it by hand
would cost far more than the transcript did.

CTC (Connectionist Temporal Classification, Graves 2006) removes the question. The model outputs, for EVERY
frame, a probability over the characters plus one extra symbol, the blank "-". A frame-by-frame output
(a "path") is turned into text by two rules, in this order:

    1. merge repeated symbols        h h - e - l l - l - o  ->  h - e - l - l - o
    2. delete the blanks                                    ->  h e l l o

The blank does two jobs: it lets the model say "nothing new here" (silence, or the middle of a long sound),
and it separates genuine double letters (the two l's above survive because a blank sits between them).

Many different paths collapse to the same text. CTC's loss is

    loss = -log( sum of the probabilities of ALL paths that collapse to the correct text )

There are astronomically many such paths, but they share structure, so dynamic programming adds them all
up in T x (2L + 1) steps: the forward algorithm, `ctc_loss` below.

What CTC assumes, and what it costs: every frame is predicted INDEPENDENTLY given the audio. The model has
no memory of what it already wrote, so it cannot use spelling or grammar the way a decoder with attention
(Whisper) can. In exchange it is simple, fast, non-autoregressive, and streams naturally.
"""

import math
from collections import defaultdict

import torch
import torch.nn.functional as F

from text import BLANK

NEG = -1e30  # stands in for log(0). A real -inf turns into NaN gradients inside logsumexp.


def _extend(targets: torch.Tensor) -> torch.Tensor:
    """(B, L) labels -> (B, 2L + 1) with a blank before, between and after: "hi" -> - h - i -.
    These 2L + 1 positions are the STATES a path walks through, left to right."""
    ext = torch.full((targets.shape[0], 2 * targets.shape[1] + 1), BLANK, dtype=torch.long, device=targets.device)
    ext[:, 1::2] = targets
    return ext


def ctc_loss(log_probs: torch.Tensor, in_lens: torch.Tensor, targets: torch.Tensor, tgt_lens: torch.Tensor) -> torch.Tensor:
    """The forward algorithm. log_probs: (B, T, V) log-softmax outputs. targets: (B, L) padded label ids.
    Returns (B,): -log P(text | audio) for each clip. Equals F.ctc_loss(reduction="none") (see check.py).

    alpha[s] after frame t = log-probability of all partial paths that are in state s at frame t.
    A path can reach state s in three ways:
        stay   it was already in s           (the symbol repeats, or the blank continues)
        step   it was in s - 1               (move to the next blank / next letter)
        skip   it was in s - 2               (letter -> next letter with no blank between)
    Skip is forbidden into a blank, and between two identical letters: "ll" must pass through the blank,
    otherwise rule 1 would merge the two l's into one."""
    B, T, _ = log_probs.shape
    ext = _extend(targets)
    can_skip = torch.zeros_like(ext, dtype=torch.bool)
    can_skip[:, 2:] = (ext[:, 2:] != BLANK) & (ext[:, 2:] != ext[:, :-2])
    emit = log_probs.gather(2, ext[:, None, :].expand(B, T, -1))  # (B, T, S): log-prob of each state's symbol at each frame

    pad1, pad2 = emit.new_full((B, 1), NEG), emit.new_full((B, 2), NEG)
    alpha = torch.cat((emit[:, 0, :2], emit.new_full((B, ext.shape[1] - 2), NEG)), dim=1)  # a path starts in blank or letter 1
    for t in range(1, T):
        stay = alpha
        step = torch.cat((pad1, alpha[:, :-1]), dim=1)
        skip = torch.where(can_skip, torch.cat((pad2, alpha[:, :-2]), dim=1), emit.new_tensor(NEG))
        new = torch.logsumexp(torch.stack((stay, step, skip)), dim=0) + emit[:, t]  # "or" = add probabilities
        alpha = torch.where((t < in_lens)[:, None], new, alpha)  # a clip shorter than the batch stops updating

    # A complete path ends in the last letter or in the blank after it.
    last = (2 * tgt_lens)[:, None]
    total = torch.logaddexp(alpha.gather(1, last), alpha.gather(1, (last - 1).clamp(min=0)))
    return -torch.where(tgt_lens[:, None] > 0, total, alpha[:, :1]).squeeze(1)


def greedy(log_probs: torch.Tensor, lens: torch.Tensor) -> list[list[int]]:
    """Best-path decoding: take the most likely symbol in every frame, then apply the two collapse rules.
    log_probs: (B, T, V). Returns one list of label ids per clip."""
    best = log_probs.argmax(dim=-1)
    out = []
    for path, n in zip(best, lens):
        merged = torch.unique_consecutive(path[:n])  # rule 1: merge repeats
        out.append(merged[merged != BLANK].tolist())  # rule 2: drop blanks
    return out


def beam_search(log_probs: torch.Tensor, beam: int = 8) -> list[int]:
    """Prefix beam search for ONE clip, log_probs: (T, V).

    Greedy finds the single most likely PATH. But a text's probability is the SUM over all its paths, so a
    text spread over many decent paths can beat the text of the single best path (see demo()). Beam search
    keeps the `beam` most likely TEXTS so far ("prefixes") and, for each, two numbers:
        p_blank      probability of all paths that produce this prefix and end in a blank
        p_non_blank  ... and end in the prefix's last letter
    The split is needed because appending the same letter again means different things in the two cases:
    after a blank it is a new letter ("l-l" -> "ll"), after the letter itself it is the same one ("ll" -> "l")."""
    beams = {(): (0.0, NEG)}
    for frame in log_probs:
        nxt: dict[tuple, list[float]] = defaultdict(lambda: [NEG, NEG])
        top_p, top_c = frame.topk(min(len(frame), 2 * beam))  # symbols with a tiny probability cannot matter
        for prefix, (p_b, p_nb) in beams.items():
            for p, c in zip(top_p.tolist(), top_c.tolist()):
                if c == BLANK:
                    nxt[prefix][0] = _log_add(nxt[prefix][0], p_b + p, p_nb + p)
                elif prefix and prefix[-1] == c:
                    nxt[prefix + (c,)][1] = _log_add(nxt[prefix + (c,)][1], p_b + p)  # a blank came between: new letter
                    nxt[prefix][1] = _log_add(nxt[prefix][1], p_nb + p)  # no blank between: same letter, held longer
                else:
                    nxt[prefix + (c,)][1] = _log_add(nxt[prefix + (c,)][1], p_b + p, p_nb + p)
        ranked = sorted(nxt.items(), key=lambda kv: _log_add(*kv[1]), reverse=True)
        beams = {prefix: (p[0], p[1]) for prefix, p in ranked[:beam]}
    return list(max(beams.items(), key=lambda kv: _log_add(*kv[1]))[0])


def _log_add(*xs: float) -> float:
    """log(exp(a) + exp(b) + ...) without leaving log space: probabilities this small underflow as plain floats."""
    top = max(xs)
    return top + math.log(sum(math.exp(x - top) for x in xs))


def align(log_probs: torch.Tensor, target: list[int]) -> list[int]:
    """Forced alignment for ONE clip: the single most likely path that spells `target`. log_probs: (T, V).
    Returns T symbol ids (blanks included): which character each frame belongs to.

    Same recursion as `ctc_loss` with "max" in place of "sum" (the Viterbi algorithm), plus a record of which
    of the three moves won, so the path can be read backwards from the end. This is how a trained recognizer
    produces timestamps, and how the TTS module will get the duration of every character."""
    ext = _extend(torch.tensor([target]))[0].tolist()
    T, S = log_probs.shape[0], len(ext)
    score = [NEG] * S
    score[0] = log_probs[0, ext[0]].item()
    if S > 1:
        score[1] = log_probs[0, ext[1]].item()
    back = []
    emit = log_probs[:, ext].tolist()
    for t in range(1, T):
        new, moves = [NEG] * S, [0] * S
        for s in range(S):
            options = [score[s], score[s - 1] if s >= 1 else NEG,
                       score[s - 2] if s >= 2 and ext[s] != BLANK and ext[s] != ext[s - 2] else NEG]
            moves[s] = max(range(3), key=options.__getitem__)
            new[s] = options[moves[s]] + emit[t][s]
        score = new
        back.append(moves)
    s = S - 1 if S == 1 or score[S - 1] >= score[S - 2] else S - 2
    path = [ext[s]]
    for moves in reversed(back):
        s -= moves[s]
        path.append(ext[s])
    return path[::-1]


def demo() -> None:
    # Two frames, two symbols: blank and "a". In every frame the blank is the more likely symbol.
    probs = torch.tensor([[0.6, 0.4], [0.6, 0.4]])
    log_probs = probs.log()
    print("frame probabilities (blank, a): [0.6, 0.4] in both frames")
    print("paths and what they collapse to:")
    for path in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        p = probs[0, path[0]] * probs[1, path[1]]
        text = "a" if 1 in path else '""'
        print(f"  {''.join('-a'[i] for i in path)} -> {text:2s}  p = {p:.2f}")
    print('  P("") = 0.36   P("a") = 0.24 + 0.24 + 0.16 = 0.64')
    print("greedy (best single path):", greedy(log_probs[None], torch.tensor([2]))[0], '= ""')
    print("beam search (best text)  :", beam_search(log_probs), '= "a"')
    assert greedy(log_probs[None], torch.tensor([2]))[0] == [] and beam_search(log_probs) == [1]

    ours = ctc_loss(log_probs[None], torch.tensor([2]), torch.tensor([[1]]), torch.tensor([1]))
    print(f'\nCTC loss for target "a": {ours.item():.4f} = -ln(0.64) = {-torch.tensor(0.64).log().item():.4f}')

    # The same on a random batch, against PyTorch's implementation.
    torch.manual_seed(0)
    lp = torch.randn(3, 50, 20).log_softmax(-1)
    targets, in_lens, tgt_lens = torch.randint(1, 20, (3, 12)), torch.tensor([50, 41, 33]), torch.tensor([12, 9, 5])
    ref = F.ctc_loss(lp.transpose(0, 1), targets, in_lens, tgt_lens, reduction="none")
    print("ours   :", [round(v, 3) for v in ctc_loss(lp, in_lens, targets, tgt_lens).tolist()])
    print("pytorch:", [round(v, 3) for v in ref.tolist()])

    frames = align(lp[0], targets[0].tolist())
    print("\nforced alignment of clip 0 (one symbol per frame, 0 = blank):\n ", frames)


if __name__ == "__main__":
    demo()
