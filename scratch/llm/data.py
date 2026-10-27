"""Lesson 3 - data: from raw text on the internet to the integer arrays the model trains on.

    python data.py            # downloads ~90 MB, trains the tokenizer, writes data/train.bin and data/val.bin

Three open datasets, one per language we care about (all Apache-2.0 / MIT):
    Hindi    short children's stories   TinyStories-Regional/hindi-generated_4o-mini_2M
    Marathi  short children's stories   TinyStories-Regional/marathi-generated_4o-mini_2M
    Hinglish everyday chat, Latin script Abhishekcr448/Hinglish-Everyday-Conversations-1M

Why tiny stories: a small model cannot learn the whole internet, but it CAN learn grammar and simple
reasoning from text written with a small vocabulary. That is the TinyStories idea.

Every document becomes:   <|bos|> <|language tag|> the text <|eos|>
"""

import argparse
import csv
import io
import json
import random
import re
import urllib.request
from pathlib import Path

import numpy as np

from tokenizer import BPE

DATA = Path(__file__).parent / "data"
HF = "https://huggingface.co/datasets"
SOURCES = {  # language tag -> (url, megabytes to take from the start of the file)
    "<|hi|>": (f"{HF}/TinyStories-Regional/hindi-generated_4o-mini_2M/resolve/main/train.json", 48),
    "<|mr|>": (f"{HF}/TinyStories-Regional/marathi-generated_4o-mini_2M/resolve/main/train.json", 32),
    "<|hing|>": (f"{HF}/Abhishekcr448/Hinglish-Everyday-Conversations-1M/resolve/main/hinglish_conversations.csv", 12),
}
STORY = re.compile(r'"story":\s*("(?:[^"\\]|\\.)*")')  # one JSON string after the key "story"


def fetch(url: str, megabytes: int, path: Path) -> str:
    """The files are 5 GB each. An HTTP Range request takes only the first few megabytes."""
    if not path.exists():
        req = urllib.request.Request(url, headers={"Range": f"bytes=0-{megabytes * 1_000_000 - 1}"})
        path.write_bytes(urllib.request.urlopen(req).read())
    return path.read_bytes().decode("utf-8", errors="ignore")  # the cut may land in the middle of a letter


def documents() -> list[tuple[str, str]]:
    docs = []
    for tag, (url, mb) in SOURCES.items():
        raw = fetch(url, mb, DATA / f"raw_{tag.strip('<|>')}.txt")
        if tag == "<|hing|>":
            raw = raw[: raw.rfind("\n")]  # drop the last line: the download cut it in half
            texts = [f"{r[0]}\n{r[1]}" for r in csv.reader(io.StringIO(raw)) if len(r) == 2][1:]  # [1:] skips the header
        else:
            texts = [json.loads(m) for m in STORY.findall(raw)]  # a half-downloaded last story does not match
        print(f"{tag:9s} {len(texts):7d} documents  {sum(map(len, texts)) / 1e6:5.1f} M characters")
        docs += [(tag, t.strip()) for t in texts]
    random.Random(0).shuffle(docs)  # fixed seed: the same split every run (reproducibility)
    return docs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--tokenizer-chars", type=int, default=4_000_000, help="how much text the tokenizer learns from")
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    docs = documents()

    sample, size = [], 0
    for _, text in docs:  # docs are shuffled, so the first N characters are a fair mix of the three languages
        sample.append(text)
        size += len(text)
        if size > args.tokenizer_chars:
            break
    tok = BPE.train("\n".join(sample), args.vocab)
    tok.save(DATA / "tokenizer.json")
    print(f"tokenizer: {tok.vocab_size} tokens, e.g. {[tok.piece(256 + i) for i in (300, 900, 1800, 3000)]}")

    bos, eos = tok.special["<|bos|>"], tok.special["<|eos|>"]
    n_val = len(docs) // 50  # hold out 2% of the documents. The model never trains on them: they measure
    for name, part in (("val", docs[:n_val]), ("train", docs[n_val:])):  # generalization, not memorization.
        ids: list[int] = []
        for tag, text in part:
            ids += [bos, tok.special[tag], *tok.encode(text), eos]
        np.array(ids, dtype=np.uint16).tofile(DATA / f"{name}.bin")  # uint16 is enough: vocab < 65536
        print(f"{name}: {len(part)} documents, {len(ids) / 1e6:.2f} M tokens")


if __name__ == "__main__":
    main()
