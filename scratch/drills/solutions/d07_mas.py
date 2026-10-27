"""Solution to drill 7: MAS for one utterance, same recursion and tie-breaking as tts/align.py."""

NEG = -1e9


def mas(score):
    L, T = len(score), len(score[0])
    best = [[NEG] * T for _ in range(L)]
    best[0][0] = score[0][0]
    for t in range(1, T):
        for l in range(L):
            stay = best[l][t - 1]
            step = best[l - 1][t - 1] if l > 0 else NEG
            best[l][t] = score[l][t] + max(stay, step)
    durations = [0] * L
    l = L - 1
    for t in range(T - 1, -1, -1):  # walk back from (last char, last frame)
        durations[l] += 1
        if l > 0 and t > 0 and (best[l - 1][t - 1] > best[l][t - 1] or l == t):
            l -= 1  # we got here by stepping from the previous character
    return durations
