"""Lesson 2 - text: what the model writes, and how we score it.

    python text.py

The LLM in ../llm used BPE subwords. Here the model writes one CHARACTER at a time, for two reasons:

    1. CTC (lesson 4) emits at most one symbol per audio frame. Characters are short sounds, so a frame can
       reasonably "be" a character. A subword like "tion" covers many frames and many sounds.
    2. With only tens of hours of speech, every output symbol must be seen many times. Hindi, Marathi and
       English together need about 110 characters. 4096 subwords would each be seen far too rarely.

Devanagari is written in Unicode as a consonant followed by separate signs: "कि" is two code points, क then ि
(the matra). We use code points as our characters, so the vocabulary stays small and the model spells
कि, की, कु from the same क.

Scoring: the model's text is compared with the human transcript by EDIT DISTANCE, the fewest insertions,
deletions and substitutions that turn one into the other.

    WER (word error rate) = word edits / words in the reference          the number everyone quotes
    CER (character error rate) = the same on characters                   fairer for long, inflected words
"""

import json
import unicodedata
from collections import Counter
from pathlib import Path

BLANK = 0  # id 0 is CTC's blank symbol: "no new character in this frame" (lesson 4)


def normalize(text: str) -> str:
    """Make the transcript match what can be HEARD. Punctuation and capital letters are not in the audio, so
    asking the model to predict them only adds errors it cannot avoid.

    NFC: Unicode can store the same visible letter in two ways (one code point, or a letter plus a combining
    mark). NFC picks one, so "equal looking" strings are equal."""
    text = unicodedata.normalize("NFC", text).lower()
    kept = (c if unicodedata.category(c)[0] in "LMN" else " " for c in text)  # letters, marks (matras), numbers
    return " ".join("".join(kept).split())  # everything else becomes a space; runs of spaces collapse to one


class Vocab:
    """The list of characters the model can write. Index 0 is the CTC blank; it is never part of any text."""

    def __init__(self, chars: list[str]) -> None:
        self.chars = ["<blank>", *chars]
        self.index = {c: i for i, c in enumerate(self.chars)}

    @classmethod
    def build(cls, texts: list[str], min_count: int = 5) -> "Vocab":
        """A character seen fewer than `min_count` times cannot be learned; it is dropped from the vocabulary
        (and so, by `encode`, from the training targets)."""
        counts = Counter("".join(texts))
        return cls(sorted(c for c, n in counts.items() if n >= min_count))

    def __len__(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.index[c] for c in text if c in self.index]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.chars[i] for i in ids if i != BLANK)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.chars[1:], ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        return cls(json.loads(path.read_text()))


def edit_distance(ref: list, hyp: list) -> int:
    """Levenshtein distance by dynamic programming. row[j] = edits to turn the first i items of `ref` into
    the first j items of `hyp`. Each cell looks at three neighbours: delete, insert, or substitute / match."""
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        diagonal, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            diagonal, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, diagonal + (r != h))
    return row[-1]


def error_rate(refs: list[str], hyps: list[str], unit: str = "word") -> float:
    """WER (unit="word") or CER (unit="char") over a whole test set: total edits / total reference length.
    It can exceed 1.0: a model that inserts many extra words makes more errors than there are words."""
    split = str.split if unit == "word" else list
    edits = sum(edit_distance(split(r), split(h)) for r, h in zip(refs, hyps))
    return edits / max(1, sum(len(split(r)) for r in refs))


def demo() -> None:
    raw = "Boss को बता देना, kal मैं Leave पर रहूँगा!"
    clean = normalize(raw)
    print(f"raw        : {raw}\nnormalized : {clean}")
    print("code points of 'कि':", [f"U+{ord(c):04X} {unicodedata.name(c)}" for c in "कि"])

    vocab = Vocab.build([clean] * 5)
    ids = vocab.encode(clean)
    print(f"vocabulary : {len(vocab)} symbols including <blank>\nencoded    : {ids[:12]} ...")
    assert vocab.decode(ids) == clean

    ref, hyp = "कल मुझे दिल्ली जाना है", "कल मुझे दिली जाना"
    print(f"\nreference  : {ref}\nhypothesis : {hyp}")
    print(f"WER {error_rate([ref], [hyp]):.2f}  (1 wrong word + 1 missing word, out of 5)")
    print(f"CER {error_rate([ref], [hyp], 'char'):.2f}  (the wrong word is only two characters off)")
    assert edit_distance("kitten", "sitting") == 3


if __name__ == "__main__":
    demo()
