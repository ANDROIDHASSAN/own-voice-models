"""Lesson 2 - tokenizer: turning text into numbers with byte-pair encoding (BPE), written from scratch.

A model cannot read letters. It reads integers (token ids). The tokenizer decides what one token is.

    one token per character -> tiny vocabulary, but very long sequences
    one token per word      -> short sequences, but a huge vocabulary and no way to spell a new word
    subword (BPE)           -> the middle path every modern LLM takes

BPE in three lines:
    1. start with the 256 possible bytes as the vocabulary (so ANY text can be encoded, in any script)
    2. find the pair of neighbouring tokens that appears most often, and glue it into one new token
    3. repeat until the vocabulary is as big as you want

Why bytes matter for us: one Devanagari letter is 3 bytes in UTF-8. The first few hundred merges are spent
just rebuilding letters and matras; after that the merges become syllables and whole words.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# Special tokens never come from text. They are control signals we put in by hand.
#   <|bos|> / <|eos|>  beginning / end of one document
#   <|hi|> <|mr|> <|hing|>  language tags. Because every training document starts with one, we can pick the
#                           reply language at generation time just by choosing the tag (conditional generation).
SPECIALS = ["<|bos|>", "<|eos|>", "<|hi|>", "<|mr|>", "<|hing|>"]

# Pre-tokenization: cut text into chunks first and never merge across a chunk border. A chunk is a word with
# its leading space, or one whitespace character. This keeps merges inside words and makes training fast,
# because we only need to count each DISTINCT word once.
CHUNK = re.compile(r" ?\S+|\s")


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of `pair` in `ids` with `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPE:
    def __init__(self, merges: list[tuple[int, int]]) -> None:
        self.merges = merges  # merge number i creates token id 256 + i
        self.rank = {pair: 256 + i for i, pair in enumerate(merges)}  # earlier merge = lower id = applied first
        self.vocab = {i: bytes([i]) for i in range(256)}  # id -> the bytes it stands for
        for pair, new_id in self.rank.items():
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
        self.special = {s: 256 + len(merges) + i for i, s in enumerate(SPECIALS)}
        self._special_name = {i: s for s, i in self.special.items()}
        self._split_specials = re.compile("(" + "|".join(map(re.escape, SPECIALS)) + ")")
        self._cache: dict[str, list[int]] = {}  # chunk -> ids; real text repeats the same words constantly

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(SPECIALS)

    @classmethod
    def train(cls, text: str, vocab_size: int) -> "BPE":
        """Learn the merges from `text`."""
        n_merges = vocab_size - 256 - len(SPECIALS)
        freq = Counter(CHUNK.findall(text))  # each distinct chunk and how often it appears
        words = [list(w.encode("utf-8")) for w in freq]
        counts = list(freq.values())

        # pair_count: how often each neighbouring pair appears in the whole text.
        # where: which words contain that pair, so a merge only has to touch those words.
        pair_count: Counter = Counter()
        where: defaultdict = defaultdict(set)
        for i, w in enumerate(words):
            for p in zip(w, w[1:]):
                pair_count[p] += counts[i]
                where[p].add(i)

        merges: list[tuple[int, int]] = []
        while len(merges) < n_merges:
            best = max(pair_count, key=lambda p: (pair_count[p], p))  # ties broken by the pair itself: deterministic
            if pair_count[best] < 2:
                break  # nothing repeats any more
            new_id = 256 + len(merges)
            merges.append(best)
            for i in where.pop(best):
                w = words[i]
                for p in zip(w, w[1:]):  # take this word's old pairs out of the count ...
                    pair_count[p] -= counts[i]
                w = words[i] = merge(w, best, new_id)
                for p in zip(w, w[1:]):  # ... and put its new pairs in
                    pair_count[p] += counts[i]
                    where[p].add(i)
            del pair_count[best]
        return cls(merges)

    def _encode_chunk(self, chunk: str) -> list[int]:
        ids = list(chunk.encode("utf-8"))
        while len(ids) > 1:
            # Apply merges in the order they were learned: always the pair with the lowest rank first.
            pair = min(zip(ids, ids[1:]), key=lambda p: self.rank.get(p, float("inf")))
            if pair not in self.rank:
                break
            ids = merge(ids, pair, self.rank[pair])
        return ids

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for part in self._split_specials.split(text):
            if part in self.special:
                ids.append(self.special[part])
                continue
            for chunk in CHUNK.findall(part):
                if chunk not in self._cache:
                    self._cache[chunk] = self._encode_chunk(chunk)
                ids.extend(self._cache[chunk])
        return ids

    def piece(self, token_id: int) -> str:
        """The text of one token, for display. A token can be half a letter, so bad bytes show as '?'."""
        if token_id in self._special_name:
            return self._special_name[token_id]
        return self.vocab[token_id].decode("utf-8", errors="replace")

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        # Join the BYTES first and decode once: a single letter may be split across two tokens.
        out, buf = [], b""
        for i in ids:
            if i in self._special_name:
                out.append(buf.decode("utf-8", errors="replace"))
                buf = b""
                if not skip_special:
                    out.append(self._special_name[i])
            else:
                buf += self.vocab[i]
        return "".join(out) + buf.decode("utf-8", errors="replace")

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"merges": self.merges}))

    @classmethod
    def load(cls, path: Path) -> "BPE":
        return cls([tuple(p) for p in json.loads(path.read_text())["merges"]])


def demo() -> None:
    text = "नमस्ते दुनिया। नमस्ते भारत। hello world, hello bhai. नमस्कार, मी मराठी बोलतो. " * 20
    tok = BPE.train(text, vocab_size=256 + len(SPECIALS) + 60)
    sample = "<|bos|><|hi|>नमस्ते bhai, hello दुनिया!<|eos|>"
    ids = tok.encode(sample)
    assert tok.decode(ids, skip_special=False) == sample, "encode -> decode must give the text back exactly"
    raw = len(sample.encode("utf-8"))
    print(f"{raw} bytes -> {len(ids)} tokens (compression {raw / len(ids):.1f}x)")
    print(" | ".join(tok.piece(i) for i in ids))


if __name__ == "__main__":
    demo()
