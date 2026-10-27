"""Solution to drill 6: frame-synchronous greedy transducer decoding."""


def rnnt_greedy(enc, predictor, joiner, blank=0, max_symbols=5):
    out = []
    g, state = predictor(blank, None)  # the start-of-sequence context
    for e in enc:
        for _ in range(max_symbols):
            scores = joiner(e, g)
            k = max(range(len(scores)), key=scores.__getitem__)
            if k == blank:
                break  # nothing more on this frame
            out.append(k)
            g, state = predictor(k, state)  # only an emission changes the label context
    return out
