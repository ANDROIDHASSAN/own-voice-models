"""Lesson 7 - how robust is the front end? Noise, chatter in the room, and what the STT writes when nobody spoke.

    python robust.py --clips DIR --refs refs.tsv      # DIR: 16 kHz Hindi test wavs, refs: "name<TAB>text" lines

Three numbers decide whether a voice agent works outside a quiet room:

    false starts     noise only, nobody talking to the agent: how often does the VAD say "speech started"?
                     Every false start is the agent interrupting itself or answering a fan.
    noise-only text  what the recognizer writes for that noise. A CTC model does not invent sentences the way
                     an attention decoder (Whisper) can, but it does write fragments. Its confidence should
                     say "this is not speech".
    CER vs SNR       the same sentences with noise mixed in at 20, 10 and 5 dB signal-to-noise ratio.

The noises are made here, nothing is downloaded: white (hiss), pink, brown (like a fan or traffic rumble) and
babble (four other speakers at once, the hardest: it IS speech, just not for us).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

import models
import nvad
from models import lesson
from vad import VAD, frames

SR = 16000


def colored(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """White noise shaped in frequency: pink falls 3 dB per octave, brown 6 dB."""
    spectrum = np.fft.rfft(rng.standard_normal(n))
    f = np.maximum(np.fft.rfftfreq(n, 1 / SR), 20.0)
    spectrum *= {"white": 1.0, "pink": f**-0.5, "brown": f**-1.0}[kind]
    x = np.fft.irfft(spectrum, n)
    return (x / np.sqrt(np.mean(x**2))).astype(np.float32)


def babble(waves: list[np.ndarray], n: int, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros(n, dtype=np.float32)
    for w in waves:
        w = np.tile(w, n // len(w) + 1)
        start = rng.integers(0, len(w) - n + 1)
        x += w[start : start + n] / (np.sqrt(np.mean(w**2)) + 1e-9)
    return x / np.sqrt(np.mean(x**2))


def mix(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    gain = np.sqrt(np.mean(speech**2) / (np.mean(noise**2) * 10 ** (snr_db / 10)))
    return speech + gain * np.resize(noise, len(speech))  # wraps around if the clip is longer than the noise


(_audio,) = lesson("stt", "audio")
confidence = lambda model, wave: models.confidence(model, _audio.log_mel, wave)  # the gate load_stt uses


def vad_events(make_vad, wave: np.ndarray) -> list[tuple[str, float]]:
    """(kind, seconds) of every START / END the VAD emits for this audio, fed 20 ms at a time."""
    vad, out = make_vad(), []
    for i, frame in enumerate(frames(wave)):
        event = vad.feed(frame)
        if event:
            out.append((event.kind, (i + 1) * 0.02))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--musan", help="held-out MUSAN noise + music, int16 .npy at 16 kHz (the Kaggle run's noise_test_audio.npy)")
    ap.add_argument("--ckpt", nargs="+", default=[str(Path(__file__).parent.parent / f"stt/runs/kaggle/{n}.pt") for n in ("kathbath", "noisy")])
    args = ap.parse_args()

    audio, model_mod, text = lesson("stt", "audio", "model", "text")
    refs = dict(line.rstrip("\n").split("\t") for line in open(args.refs, encoding="utf-8"))
    waves = {p.name: audio.read_wav(p)[0] for p in sorted(Path(args.clips).rglob("*.wav")) if p.name in refs}
    names = list(waves)
    rng = np.random.default_rng(0)
    test, others = names[4:], [waves[n] for n in names[:4]]  # 4 speakers make the babble, the rest are tested

    n = SR * 30
    noises = {k: colored(k, n, rng) for k in ("white", "pink", "brown")} | {"babble": babble(others, n, rng)}
    if args.musan:
        m = np.load(args.musan).astype(np.float32) / 32768
        noises["musan"] = m / np.sqrt(np.mean(m**2))  # never seen in training: music, traffic, machines ...
    vads = {"energy": VAD}
    if nvad.CKPT.exists():
        vads["neural"] = nvad.NeuralVAD
    quiet = np.random.default_rng(1).normal(0, 1e-4, SR // 2).astype(np.float32)

    print("1. Noise only, at a normal room level (-35 dBFS). False starts per minute: the agent hears 'speech'.\n")
    print(f"{'noise':8s}" + "".join(f"{v:>10s}" for v in vads))
    for kind, noise in noises.items():
        x = np.concatenate([quiet, noise * 10 ** (-35 / 20)])
        row = [sum(k == "start" for k, _ in vad_events(make, x)) / (len(noise) / SR / 60) for make in vads.values()]
        print(f"{kind:8s}" + "".join(f"{r:10.1f}" for r in row))

    print(f"\n2. Speech found: {len(test)} clips, each between 1 s of noise before and 1.5 s after. A clip counts as")
    print("   found if a START falls inside it; extra = STARTs outside the speech (noise taken for speech).\n")
    print(f"{'noise':8s}{'SNR':>5s}" + "".join(f"{v + ' found':>14s}{'extra':>7s}" for v in vads))
    for kind, noise in noises.items():
        for snr in (10, 5):
            cells = []
            for make in vads.values():
                found = extra = 0
                for name in test:
                    w = waves[name]
                    start = rng.integers(0, len(noise) // 2)
                    pad = np.zeros(SR, dtype=np.float32)
                    clip = np.concatenate([pad, w, np.zeros(SR * 3 // 2, dtype=np.float32)])
                    noisy = clip + np.sqrt(np.mean(w**2) / (np.mean(noise**2) * 10 ** (snr / 10))) * np.resize(noise[start:], len(clip))
                    starts = [t - 0.5 for k, t in vad_events(make, np.concatenate([quiet, noisy]).astype(np.float32)) if k == "start"]
                    inside = [t for t in starts if 1.0 - 0.3 <= t <= 1.0 + len(w) / SR]
                    found += bool(inside)
                    extra += len(starts) - len(inside)
                cells.append(f"{found:>8d}/{len(test):<5d}{extra:>7d}")
            print(f"{kind:8s}{snr:>4d} " + "".join(cells))

    # Short answers: "हाँ", "नहीं", "दिल्ली" last half a second. v1 of the neural VAD never fired on them, and
    # the test above (whole read sentences) could not show it. Real speech: the first 0.4-0.8 s of each clip's
    # speech, cut where its own energy says speech starts.
    print(f"\n2b. Short answers: a 0.4-0.8 s piece of each of the {len(test)} clips, alone (1 s of silence or noise around it)\n")
    print(f"{'noise':8s}{'SNR':>5s}" + "".join(f"{v + ' found':>14s}" for v in vads))
    for kind, snr in ((None, None), ("musan" if "musan" in noises else "pink", 10), ("brown", 10)):
        cells = []
        for make in vads.values():
            found = 0
            for name in test:
                w = waves[name]
                onset = int(np.argmax(np.abs(w) > 0.1 * np.abs(w).max()))
                piece = w[onset : onset + int(SR * rng.uniform(0.4, 0.8))]
                clip = np.concatenate([np.zeros(SR, dtype=np.float32), piece, np.zeros(SR, dtype=np.float32)])
                if kind:
                    noise = noises[kind]
                    clip = clip + np.sqrt(np.mean(piece**2) / (np.mean(noise**2) * 10 ** (snr / 10))) * np.resize(noise[rng.integers(0, len(noise) // 2):], len(clip))
                starts = [t - 0.5 for k, t in vad_events(make, np.concatenate([quiet, clip]).astype(np.float32)) if k == "start"]
                found += any(0.7 <= t <= 1.0 + len(piece) / SR + 0.1 for t in starts)
            cells.append(f"{found:>8d}/{len(test):<5d}")
        print(f"{kind or 'clean':8s}{snr or '':>5} " + "".join(cells))

    for path in args.ckpt:
        model, ckpt = model_mod.load(path, "cpu")
        vocab = text.Vocab(ckpt["chars"])
        print(f"\n3. CER, {Path(path).name}: {len(test)} Hindi test clips, clean and with noise at SNR 20 / 10 / 5 dB\n")
        hyps = [vocab.decode(confidence(model, waves[name])[0]) for name in test]
        print(f"{'clean':8s}  {text.error_rate([refs[n] for n in test], hyps, 'char'):.3f}")
        noise_rng = np.random.default_rng(2)  # the same noise slices for every checkpoint
        for kind, noise in noises.items():
            row = []
            for snr in (20, 10, 5):
                hyps = [vocab.decode(confidence(model, mix(waves[name], noise[noise_rng.integers(0, len(noise) // 2):], snr).astype(np.float32))[0])
                        for name in test]
                row.append(text.error_rate([refs[n] for n in test], hyps, "char"))
            print(f"{kind:8s}  " + "   ".join(f"{snr} dB {r:.3f}" for snr, r in zip((20, 10, 5), row)))


if __name__ == "__main__":
    sys.exit(main())
