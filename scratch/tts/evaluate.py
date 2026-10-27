"""Lesson 9 - evaluation: can a machine understand what our model says?

    python evaluate.py
    python evaluate.py --griffin-lim     # the same sentences without the neural vocoder

The real test of a TTS voice is people listening (MOS, mean opinion score). We have no panel of listeners,
so we measure the part that CAN be measured automatically, intelligibility:

    held-out sentence -> our TTS -> audio -> a strong recognizer -> text -> character error rate

The recognizer is the judge (Omnilingual ASR, 300 M, the model the real agent uses). The judge makes its own
mistakes, so a number alone means little. The SAME sentences spoken by the teacher go through the SAME
judge, and the two are printed side by side: the gap between them is what the student lost.

What this does not measure: naturalness. A robotic voice can be perfectly intelligible. For that, listen
to runs/eval_*.wav.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

import vocoder as voc
from audio import RUNS_DIR, SR, resample, write_wav
from data import open_split
from model import load
from speak import speak
from text import Vocab
from train import speaker_of

sys.path.append(str(Path(__file__).parent.parent))
from stt.text import error_rate, normalize  # noqa: E402  (the recognizer's normalization: no punctuation)

HERE = Path(__file__).parent
JUDGE = HERE / "../../models/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-v2-int8-2026-02-05"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--griffin-lim", action="store_true")
    ap.add_argument("--n", type=int, default=40, help="dev sentences per language and voice")
    ap.add_argument("--model", default="acoustic", help="checkpoint in runs/: acoustic, or flow")
    ap.add_argument("--vocoder", default="vocoder", help="checkpoint in runs/: vocoder, or vocoder_gan")
    ap.add_argument("--steps", type=int, default=4, help="flow model: Euler steps")
    args = ap.parse_args()
    import sherpa_onnx

    judge = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
        model=str(JUDGE / "model.int8.onnx"), tokens=str(JUDGE / "tokens.txt"), num_threads=4)

    def heard(wave: np.ndarray) -> str:
        stream = judge.create_stream()
        stream.accept_waveform(16000, resample(torch.from_numpy(wave), SR, 16000).numpy())
        judge.decode_stream(stream)
        return normalize(stream.result.text)

    torch.set_num_threads(2)
    acoustic, ckpt = load(RUNS_DIR / f"{args.model}.pt", "cpu")
    vocoder = None if args.griffin_lim else voc.load(RUNS_DIR / f"{args.vocoder}.pt", "cpu")[0]
    vocab = Vocab(ckpt["chars"])
    pcm, _, index = open_split("dev")
    size = acoustic.n_params() + (vocoder.n_params() if vocoder else 0)
    print(f"{args.model} + {'Griffin-Lim' if vocoder is None else args.vocoder}: {size / 1e6:.2f} M parameters, CPU, 2 threads\n")
    print(f"{'':24s}{'clips':>6s}{'student CER':>13s}{'teacher CER':>13s}{'real-time factor':>18s}")
    # One row per language AND voice: Rasa's dev clips come after the teacher's, and must be measured on their own.
    for lang, who in sorted({(e["lang"], speaker_of(e)) for e in index}):
        entries = [e for e in index if e["lang"] == lang and speaker_of(e) == who][: args.n]
        refs, student, teacher, compute, audio = [], [], [], 0.0, 0.0
        for i, e in enumerate(entries):
            voice = {}
            if acoustic.cfg.flow:  # the clip's own voice and style, read with the asked-for number of steps
                voice = {"speaker": ckpt["speakers"].index(who), "style": ckpt["styles"].index(e.get("style", "neutral")), "steps": args.steps}
            t0 = time.perf_counter()
            wave = speak(acoustic, vocoder, vocab, e["text"], lang, **voice)
            compute += time.perf_counter() - t0
            audio += len(wave) / SR
            if i == 0:
                write_wav(RUNS_DIR / f"eval_{args.model}_{lang}_{who}.wav", wave)
            refs.append(normalize(e["text"]))
            student.append(heard(wave))
            teacher.append(heard(pcm[e["start"] : e["start"] + e["samples"]].astype(np.float32) / 32768))
        print(f"{lang + ' ' + who:24s}{len(entries):6d}{error_rate(refs, student, 'char'):13.3f}"
              f"{error_rate(refs, teacher, 'char'):13.3f}{compute / audio:18.3f}")
    print("\n(Roman-script Hinglish has no fixed spelling, so the judge's text rarely matches letter for letter:"
          "\n read the hing row as student vs teacher, not as an absolute error. Marathi's \"teacher\" is the real recordings.)")


if __name__ == "__main__":
    main()
