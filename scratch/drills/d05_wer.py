"""Drill 5 - WER / CER from scratch: Levenshtein edit distance (stt/text.py:73-89).

    python d05_wer.py              # checks YOUR functions below
    python d05_wer.py --solution   # checks solutions/d05_wer.py

Write:
  edit_distance(ref, hyp)            -> int: fewest insertions + deletions + substitutions (lists or strings)
  error_rate(refs, hyps, unit)       -> float: total edits / total reference length over a whole test set,
                                        unit "word" (WER) or "char" (CER)
O(len(ref) * len(hyp)) time; bonus for O(len(hyp)) memory (one row). About 10 minutes.
"""

import sys
from pathlib import Path


def edit_distance(ref, hyp):
    raise NotImplementedError


def error_rate(refs, hyps, unit="word"):
    raise NotImplementedError


def check(m) -> None:
    assert m.edit_distance("kitten", "sitting") == 3
    assert m.edit_distance([], [1, 2]) == 2 and m.edit_distance([1, 2], []) == 2 and m.edit_distance("abc", "abc") == 0
    ref, hyp = "कल मुझे दिल्ली जाना है", "कल मुझे दिली जाना"
    wer, cer = m.error_rate([ref], [hyp]), m.error_rate([ref], [hyp], "char")
    assert abs(wer - 0.40) < 1e-9, f"WER {wer}: 1 substitution + 1 deletion over 5 words = 0.40"
    assert abs(cer - 0.2273) < 1e-3, f"CER {cer}: character edits over reference characters (spaces count)"
    assert m.error_rate(["a b"], ["x y z w"]) == 2.0, "WER can exceed 1: insertions count too"
    assert abs(m.error_rate(["a b c d", "e f"], ["a x c d", "e f"]) - 1 / 6) < 1e-9, "sum edits over the SET, then divide (not a mean of per-clip rates)"
    print("PASS  edit_distance; WER 0.40 vs CER 0.23 on the Hindi example; WER 2.0 with insertions; corpus-level average")


if __name__ == "__main__":
    if "--solution" in sys.argv:
        sys.path.insert(0, str(Path(__file__).parent / "solutions"))
        import d05_wer as mod
    else:
        mod = sys.modules[__name__]
    check(mod)
