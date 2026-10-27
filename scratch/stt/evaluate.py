"""Lesson 8 - evaluation: the number we report, and an honest yardstick next to it.

    python evaluate.py                 # our model on the full test split
    python evaluate.py --baseline 50   # also run the agent's 300 M model on 50 test clips per language

Dev was looked at after every epoch to pick the best checkpoint, so the model is (slightly) tuned to it.
The TEST split was never looked at. Its score is the one to quote.

A WER means little alone. "40% WER" is terrible for Whisper and fine for a 5 M model trained on 10 hours.
So the same test clips also go through Omnilingual ASR, the 300 M parameter model the real agent uses
(../../speech.py), and the two are printed side by side: same clips, same text normalization, same metric.
"""

import argparse
import tarfile
import time
from pathlib import Path

import torch

from audio import SR, read_wav
from data import fetch, open_split
from model import load
from text import Vocab, error_rate, normalize
from train import evaluate
from transcribe import transcribe

HERE = Path(__file__).parent
BASELINE = HERE / "../../models/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-v2-int8-2026-02-05"


def test_clips(lang: str, n: int):
    """The first n test clips of a language as (reference text, waveform), read straight from the archive."""
    tsv, tar = fetch(lang, "test")
    rows = {r[1]: normalize(r[3]) for r in (line.rstrip("\n").split("\t") for line in open(tsv, encoding="utf-8"))}
    with tarfile.open(tar, "r|gz") as archive:
        for member in archive:
            text = rows.get(Path(member.name).name)
            if text and member.isfile():
                yield text, read_wav(archive.extractfile(member).read())[0]
                n -= 1
                if n == 0:
                    return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "runs/ckpt.pt"))
    ap.add_argument("--baseline", type=int, default=0, metavar="N", help="compare with the 300 M model on N clips per language")
    ap.add_argument("--beam", type=int, default=8)
    args = ap.parse_args()

    model, ckpt = load(args.ckpt)
    vocab = Vocab(ckpt["chars"])
    device = str(next(model.parameters()).device)
    mel, index = open_split("test")
    scores, _ = evaluate(model, mel, index, list(range(len(index))), vocab, device, 16000)
    print(f"{model.n_params() / 1e6:.2f} M parameters, checkpoint from epoch {ckpt['epoch']} (dev CER {ckpt['dev_cer']:.3f})")
    print("\nfull test split, greedy decoding")
    for lang, (cer, wer) in sorted(scores.items()):
        clips = [e for e in index if e["lang"] == lang]
        print(f"  {lang}  {len(clips):4d} clips  {sum(e['frames'] for e in clips) / 360000:.2f} h  CER {cer:.3f}  WER {wer:.3f}")

    if not args.baseline:
        return
    import sherpa_onnx  # only needed for the comparison

    big = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
        model=str(BASELINE / "model.int8.onnx"), tokens=str(BASELINE / "tokens.txt"), num_threads=2)
    cpu = model.to("cpu")
    torch.set_num_threads(2)  # the same 2 threads the baseline gets
    print(f"\nfirst {args.baseline} test clips per language, CPU, 2 threads   (CER / WER / real-time factor)")
    print(f"  {'':4s}{'ours, greedy':24s}{f'ours, beam {args.beam}':24s}{'Omnilingual ASR 300 M':24s}")
    for lang in sorted(scores):
        refs, hyps, seconds, audio = [], {"greedy": [], "beam": [], "big": []}, {"greedy": 0.0, "beam": 0.0, "big": 0.0}, 0.0
        for ref, wave in test_clips(lang, args.baseline):
            refs.append(ref)
            audio += len(wave) / SR
            for name, run in (("greedy", lambda: transcribe(cpu, vocab, wave)), ("beam", lambda: transcribe(cpu, vocab, wave, args.beam))):
                t0 = time.perf_counter()
                hyps[name].append(run())
                seconds[name] += time.perf_counter() - t0
            t0 = time.perf_counter()
            stream = big.create_stream()
            stream.accept_waveform(SR, wave)
            big.decode_stream(stream)
            hyps["big"].append(normalize(stream.result.text))
            seconds["big"] += time.perf_counter() - t0
        cells = [f"{error_rate(refs, hyps[k], 'char'):.3f} / {error_rate(refs, hyps[k]):.3f} / {seconds[k] / audio:.3f}" for k in hyps]
        print(f"  {lang}  " + "".join(f"{c:24s}" for c in cells))


if __name__ == "__main__":
    main()
