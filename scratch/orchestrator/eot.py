"""Lesson 9 - a turn detector of our own: has the user finished the sentence?

    python eot.py --texts a.json b.txt ...   # train (Kaggle GPU, a few minutes)
    python eot.py                            # check, and show it on the turn.py cases if runs/eot.pt exists

Lesson 2 asked the scratch LLM "would the next token end the message?". That LLM learned Roman Hinglish chat,
but the recognizer writes Devanagari, so every Hindi answer looked unfinished and the agent waited the full
1.5 s. LiveKit's turn detector is a small language model fine-tuned for exactly this one question. Ours:

    characters -> embedding 64 -> GRU 128 -> last state -> P(finished)                       ~70 k parameters

Training data is made from sentences we already have, with no labelling: a whole sentence is "finished", the
same sentence cut after a random word is "not finished". Hindi and Marathi put the verb last ("... घर जा रहा
हूँ"), so the ending carries most of the signal, and a left-to-right GRU reads it last.

Text is normalized the way the recognizer writes: no punctuation, lower case, single spaces.
"""

import argparse
import json
import random
import re
import time
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).parent
CKPT = HERE / "runs/eot.pt"


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split())


class Net(nn.Module):
    def __init__(self, chars: str, dim: int = 64, hidden: int = 128) -> None:
        super().__init__()
        self.chars = chars
        self.ids = {c: i + 1 for i, c in enumerate(chars)}  # 0 = padding / unknown
        self.emb = nn.Embedding(len(chars) + 1, dim, padding_idx=0)
        self.gru = nn.GRU(dim, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    def encode(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        rows = [[self.ids.get(c, 0) for c in normalize(t)][-120:] or [0] for t in texts]  # the end matters: keep the last 120
        x = torch.zeros(len(rows), max(map(len, rows)), dtype=torch.long)
        for i, r in enumerate(rows):
            x[i, : len(r)] = torch.tensor(r)
        return x, torch.tensor([len(r) for r in rows])

    def forward(self, x: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(self.emb(x), lens.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return self.out(h[-1])[:, 0]


def load(path: Path = CKPT):
    """-> completeness(text) -> P(finished), the signature turn.Endpointer takes."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    net = Net(ckpt["chars"])
    net.load_state_dict(ckpt["model"])
    net.eval()

    @torch.no_grad()
    def completeness(text: str) -> float:
        return float(torch.sigmoid(net(*net.encode([text])))[0])

    return completeness


def examples(sentences: list[str], rng: random.Random) -> list[tuple[str, float]]:
    out = []
    for s in sentences:
        words = normalize(s).split()
        if len(words) < 2:
            continue
        out.append((" ".join(words), 1.0))
        out.append((" ".join(words[: rng.randrange(1, len(words))]), 0.0))
    return out


def read_texts(paths: list[str]) -> list[str]:
    """.json: an STT index (list of {"text": ...}); anything else: one sentence per line."""
    out = []
    for p in paths:
        if p.endswith(".json"):
            out += [e["text"] for e in json.loads(Path(p).read_text())]
        else:
            out += [line.strip() for line in open(p, encoding="utf-8") if line.strip()]
    return out


def train(args) -> None:
    rng = random.Random(0)
    torch.manual_seed(0)
    train_s, dev_s = read_texts(args.texts), read_texts(args.dev)
    chars = "".join(sorted({c for s in train_s for c in normalize(s)}))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = Net(chars).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3)
    dev = examples(dev_s, random.Random(1))
    print(f"{sum(p.numel() for p in net.parameters())} parameters  {len(train_s)} sentences  vocab {len(chars)}  "
          f"dev {len(dev)} examples", flush=True)

    @torch.no_grad()
    def accuracy() -> float:
        net.eval()
        right = 0
        for i in range(0, len(dev), 512):
            texts, ys = zip(*dev[i : i + 512])
            x, lens = net.encode(list(texts))
            right += int(((net(x.to(device), lens) > 0).float().cpu() == torch.tensor(ys)).sum())
        net.train()
        return right / len(dev)

    t0, best = time.perf_counter(), 0.0
    for epoch in range(args.epochs):
        data = examples(train_s, rng)
        rng.shuffle(data)
        for i in range(0, len(data), 256):
            texts, ys = zip(*data[i : i + 256])
            x, lens = net.encode(list(texts))
            loss = nn.functional.binary_cross_entropy_with_logits(net(x.to(device), lens), torch.tensor(ys, device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc = accuracy()
        print(f"epoch {epoch + 1}  loss {loss.item():.3f}  dev accuracy {acc:.3f}  {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
        if acc > best:
            best = acc
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": net.state_dict(), "chars": chars, "dev_acc": acc, "epoch": epoch + 1}, args.out)


def check() -> None:
    sents = ["मैं घर जा रहा हूँ", "आज मौसम बहुत अच्छा है", "वो कल दिल्ली गया था"] * 50
    rng = random.Random(0)
    ex = examples(sents, rng)
    assert len(ex) == 300 and sum(y for _, y in ex) == 150
    assert all(t != normalize(s) for (t, y), s in zip(ex[1::2], sents) if y == 0)  # a cut sentence is never the whole one
    net = Net("".join(sorted({c for s in sents for c in normalize(s)})))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-2)
    for _ in range(150):  # it must be able to learn this tiny set
        texts, ys = zip(*ex)
        loss = nn.functional.binary_cross_entropy_with_logits(net(*net.encode(list(texts))), torch.tensor(ys))
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1, loss.item()
    assert normalize("Haan, theek hai!") == "haan theek hai"
    print("eot: all checks passed")
    if CKPT.exists():
        completeness = load()
        for text in ("मेरा ऑर्डर नंबर है", "हाँ ठीक है", "कल मैं ऑफिस", "कल मैं ऑफिस नहीं आऊँगा", "मुझे लगता है कि",
                     "तुम कहाँ हो", "अगर तुम फ्री हो तो", "चलो फिर मिलते हैं"):
            print(f"  {completeness(text):.2f}  {text}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", nargs="+")
    ap.add_argument("--dev", nargs="+")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--out", default=str(CKPT))
    a = ap.parse_args()
    train(a) if a.texts else check()
