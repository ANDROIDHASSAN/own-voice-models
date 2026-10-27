"""Lesson 2 - the text front-end: what the model reads before it speaks.

    python text.py

In recognition we threw punctuation away because it cannot be heard. In synthesis it is the opposite:
a comma is a short pause, a full stop a long one and a falling pitch, a question mark a rising one.
So the front-end KEEPS a small set of punctuation marks and gives each its own symbol.

    "Boss को बता देना, kal मैं leave पर रहूँगा!"
      -> normalize        NFC, lower-case, unify punctuation (। becomes .), drop what cannot be spoken
      -> characters       one id per Unicode code point, exactly as in ../stt/text.py
      -> + a language id   the same letters are pronounced differently per language (see below)

Characters, not phonemes. Classic TTS first converts text to phonemes with a pronunciation dictionary and
rules (G2P, grapheme-to-phoneme). Devanagari is close to phonetic, so the model can learn the mapping
itself; AI4Bharat's Indic TTS and Supertonic both feed raw characters. The known cost for Hindi is schwa
deletion: कमल is written ka-ma-la but said "kamal". A character model has to learn that from examples.

Why a language id. "to" is English "too" in an English sentence and Hindi "toh" in "main to ja raha hoon".
Roman-script Hinglish and English share an alphabet but not its pronunciation, so they get different ids.
Hindi and Marathi share Devanagari the same way (च, ज, ळ, and where the schwa drops).
"""

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

PAD = 0
LANGS = ["hi", "mr", "en", "hing"]  # hing = Hindi written in Roman letters, with English words mixed in
PUNCT = ",.?!"
_UNIFY = str.maketrans({"।": ".", "॥": ".", ";": ",", ":": ",", "—": ",", "–": ",", "…": ".", "|": "."})
_MARATHI = set("आहे आहेत आहात नाही आणि मी मला तुम्ही कसे काय होते होता माझे आम्ही आता मध्ये".split())  # words Hindi does not have
_HINGLISH = set("hai hain nahi nahin kya kyu kyun mein main mera meri tum aap bhai yaar kar raha rahi tha thi hoga ho gaya "
                "ka ki ke ko se par aur bhi toh bahut accha theek abhi kal aaj kaise kuch".split())


def normalize(text: str) -> str:
    """Keep letters, matras, spaces and , . ? !  Everything else that is not speakable is dropped."""
    text = unicodedata.normalize("NFC", text).lower().translate(_UNIFY)
    kept = (c if unicodedata.category(c)[0] in "LM" or c in PUNCT else " " for c in text)
    text = " ".join("".join(kept).split())
    return re.sub(r"\s+([,.?!])", r"\1", text)  # no space before punctuation


def guess_lang(text: str) -> str:
    """hi / mr / en / hing from the script and a few marker words. Good enough for a demo; in the agent the
    LLM already knows which language it is replying in and passes it along."""
    words = re.findall(r"[a-z]+|[ऀ-ॿ]+", text.lower())
    devanagari = [w for w in words if w[0] > "z"]
    if len(devanagari) > len(words) / 2:
        return "mr" if _MARATHI & set(devanagari) else "hi"
    return "hing" if len(_HINGLISH & set(words)) >= max(1, len(words) // 6) or devanagari else "en"


class Vocab:
    """Characters the model can read. Index 0 is padding; unknown characters are skipped."""

    def __init__(self, chars: list[str]) -> None:
        self.chars = ["<pad>", *chars]
        self.index = {c: i for i, c in enumerate(self.chars)}

    @classmethod
    def build(cls, texts: list[str], min_count: int = 5) -> "Vocab":
        counts = Counter("".join(texts))
        return cls(sorted(c for c, n in counts.items() if n >= min_count))

    def __len__(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.index[c] for c in normalize(text) if c in self.index]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.chars[1:], ensure_ascii=False))


def demo() -> None:
    for raw in ["Boss को बता देना; kal मैं leave पर रहूँगा!", "नमस्ते। आप कैसे हैं?", "तुम्ही कसे आहात?",
                "kya yaar, traffic mein stuck ho gaya.", "The meeting is at five."]:
        print(f"{guess_lang(raw):5s} {raw}\n      -> {normalize(raw)}")
    vocab = Vocab.build(["नमस्ते, kal milte hain."] * 5)
    print(f"\n'{normalize('नमस्ते, kal?')}' -> {vocab.encode('नमस्ते, kal?')}  ('?' was never seen, so it is skipped)")
    print("\nकमल is 3 code points:", [unicodedata.name(c).split()[-1] for c in "कमल"],
          "-> written ka-ma-la, spoken ka-mal: the last schwa is deleted")


if __name__ == "__main__":
    demo()
