"""Lesson 7 - inference: a .wav file, or your own voice, to text.

    python transcribe.py clip.wav
    python transcribe.py --mic 5            # record 5 seconds from the microphone, then transcribe
    python transcribe.py clip.wav --beam 8  # prefix beam search instead of greedy

Inference is training without the loss:

    samples -> log-mel -> model -> per-frame log-probabilities -> CTC decode -> characters -> text

The whole clip goes through in ONE forward pass. There is no token-by-token loop as in the LLM, which is
why a CTC model is fast: the cost is one pass over the audio, however long the sentence is.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from audio import SR, log_mel, read_wav
from ctc import beam_search, greedy
from model import Encoder, load
from text import Vocab

HERE = Path(__file__).parent


@torch.no_grad()
def transcribe(model: Encoder, vocab: Vocab, wave: np.ndarray, beam: int = 0) -> str:
    """wave: mono float32 samples at 16 kHz."""
    device = next(model.parameters()).device
    mel = log_mel(wave)[None].to(device)
    log_probs, lens = model(mel, torch.tensor([mel.shape[1]], device=device))
    labels = beam_search(log_probs[0].cpu(), beam) if beam else greedy(log_probs.cpu(), lens.cpu())[0]
    return vocab.decode(labels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", nargs="?")
    ap.add_argument("--mic", type=float, metavar="SECONDS")
    ap.add_argument("--beam", type=int, default=0)
    ap.add_argument("--ckpt", default=str(HERE / "runs/ckpt.pt"))
    args = ap.parse_args()

    model, ckpt = load(args.ckpt)
    vocab = Vocab(ckpt["chars"])
    if args.mic:
        import sounddevice  # only needed for the microphone

        print(f"speak for {args.mic:g} seconds ...")
        wave = sounddevice.rec(int(args.mic * SR), samplerate=SR, channels=1, dtype="float32")[:, 0]
        sounddevice.wait()
    elif args.wav:
        wave, rate = read_wav(args.wav)
        if rate != SR:
            raise SystemExit(f"{args.wav} is {rate} Hz; the model was trained on {SR} Hz audio. Resample it first.")
    else:
        ap.error("give a .wav file or --mic SECONDS")

    t0 = time.perf_counter()
    text = transcribe(model, vocab, wave, args.beam)
    took = time.perf_counter() - t0
    # Real-time factor: seconds of compute per second of audio. Below 1 means faster than the person spoke.
    print(f"{text}\n({len(wave) / SR:.1f} s of audio in {took * 1000:.0f} ms, real-time factor {took / (len(wave) / SR):.3f})")


if __name__ == "__main__":
    main()
