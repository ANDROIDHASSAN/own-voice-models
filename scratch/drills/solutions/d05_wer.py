"""Solution to drill 5. Same as stt/text.py:73-89."""


def edit_distance(ref, hyp):
    row = list(range(len(hyp) + 1))  # row[j]: edits turning the first i items of ref into the first j of hyp
    for i, r in enumerate(ref, 1):
        diagonal, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            diagonal, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, diagonal + (r != h))
    return row[-1]


def error_rate(refs, hyps, unit="word"):
    split = str.split if unit == "word" else list
    edits = sum(edit_distance(split(r), split(h)) for r, h in zip(refs, hyps))
    return edits / max(1, sum(len(split(r)) for r in refs))
