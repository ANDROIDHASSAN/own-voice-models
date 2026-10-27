"""Lesson 10 - x-ray: watch CTC work on a real clip, frame by frame.

    python xray.py --lang hi              # a Hindi clip from the dev split
    python xray.py --lang mr --clip 7

Shows three things:
    1. the raw frame-by-frame output BEFORE the two collapse rules ("." is the blank)
    2. how sure the model is, and how much of the time it says blank
    3. forced alignment: when each word of the reference was spoken (timestamps from a model that was
       never told any timing)

A trained CTC model is "spiky": it stays on blank most of the time and emits each character in one or
two frames, right when it becomes sure. That is why the picture in (1) is mostly dots.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from ctc import align, greedy
from data import open_split
from model import load
from text import BLANK, Vocab

HERE = Path(__file__).parent
FRAME_SEC = 0.04  # one output frame = 4 mel frames = 40 ms (with the default subsample of 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--clip", type=int, default=0)
    ap.add_argument("--ckpt", default=str(HERE / "runs/ckpt.pt"))
    args = ap.parse_args()

    model, ckpt = load(args.ckpt, "cpu")
    vocab = Vocab(ckpt["chars"])
    mel, index = open_split("dev")
    entry = [e for e in index if e["lang"] == args.lang][args.clip]
    x = torch.from_numpy(mel[entry["start"] : entry["start"] + entry["frames"]].astype(np.float32))[None]
    with torch.no_grad():
        log_probs, lens = model(x, torch.tensor([x.shape[1]]))
    log_probs = log_probs[0]

    best = log_probs.argmax(-1).tolist()
    frame_sec = FRAME_SEC * model.cfg.subsample / 4  # a model trained with --subsample 2 has 20 ms frames
    print(f"clip: {entry['frames'] / 100:.1f} s -> {len(best)} output frames of {frame_sec * 1000:.0f} ms\n")
    print("1. raw output, one symbol per frame ('.' = blank, '_' = space):")
    raw = "".join("." if i == BLANK else "_" if vocab.chars[i] == " " else vocab.chars[i] for i in best)
    for i in range(0, len(raw), 100):
        print("   " + raw[i : i + 100])
    print("\n   after merging repeats and dropping blanks:")
    print("   hyp:", vocab.decode(greedy(log_probs[None], lens)[0]))
    print("   ref:", entry["text"])

    confidence = log_probs.max(-1).values.exp()
    print(f"\n2. blank in {sum(i == BLANK for i in best) / len(best):.0%} of frames; "
          f"mean confidence {confidence.mean():.2f}; least sure frame {confidence.min():.2f}")

    print("\n3. forced alignment of the reference (start - end, word):")
    frames = align(log_probs, vocab.encode(entry["text"]))
    space = vocab.index.get(" ")
    word, start, last = [], None, 0
    for t, s in enumerate([*frames, space]):  # a final space flushes the last word
        if s == space and t > 0 and (t == len(frames) or frames[t - 1] != space):
            if word:
                print(f"   {start * frame_sec:5.2f} - {(last + 1) * frame_sec:5.2f}  {''.join(word)}")
            word, start = [], None
        elif s not in (BLANK, space):
            if start is None:
                start = t
            if t == 0 or frames[t - 1] != s:
                word.append(vocab.chars[s])
            last = t


if __name__ == "__main__":
    main()
