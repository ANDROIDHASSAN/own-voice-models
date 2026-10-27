"""Lesson 1 - audio, the other way round: from a spectrogram back to sound.

    python audio.py

Speech recognition (../stt) went  waveform -> log-mel -> text  and threw information away at every step.
Speech synthesis has to go  text -> log-mel -> waveform  and put it back. The mel spectrogram is the
meeting point: the model in lesson 5 writes one, and a vocoder (lesson 7) turns it into samples.

Why the last step is hard. The spectrogram kept the MAGNITUDE of each frequency and dropped the PHASE
(where in its cycle each wave is). Magnitudes say which notes are playing; without phases you cannot add
the waves up into a waveform. Here we do it the classic way, with no learning at all:

    Griffin-Lim (1984)   guess random phases, then repeat: build a waveform, analyse it again, keep its
                         phases, put the known magnitudes back. The guess gets more self-consistent each round.

It works, and it sounds metallic. That gap is the reason neural vocoders exist.

Synthesis runs at 22,050 Hz, not the 16,000 Hz of recognition. For understanding words 8 kHz of bandwidth is
enough; for a voice to sound natural the "s" and "sh" energy above 8 kHz has to be there.
"""

import math
import os
import struct
import sys
from functools import cache
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent))  # lessons build on each other: reuse ../stt/audio.py
from stt.audio import mel_filterbank, read_wav  # noqa: E402, F401

# Lesson 10: TTS_SR=24000 switches everything to 24 kHz with 100 mel bands (the Vocos / F5-TTS setting). 22.05 kHz
# cuts at 11 kHz, which removes the air of "s", "sh" and breath; Rasa is recorded at 48 kHz, so 24 kHz keeps up to
# 12 kHz of it. Each rate has its own data and checkpoint folders, so the 22.05 kHz models stay usable.
SR = int(os.environ.get("TTS_SR", 22050))
N_FFT = 1024  # 46 ms window: longer than in recognition, for finer frequency detail (pitch harmonics)
HOP = 256  # 11.6 ms, 86 frames per second (at 24 kHz: 10.7 ms, 94 per second)
N_MELS = 100 if SR == 24000 else 80
_HERE = Path(__file__).parent
DATA_DIR, RUNS_DIR = (_HERE / "data", _HERE / "runs") if SR == 22050 else (_HERE / f"data{SR // 1000}k", _HERE / f"runs{SR // 1000}k")


def resample(wave: torch.Tensor, sr_in: int, sr_out: int, width: int = 32) -> torch.Tensor:
    """Change the sampling rate by windowed-sinc interpolation.

    A sampled signal defines a smooth curve: put a sinc function on every sample and add them up. Resampling
    reads that curve at new time points. When going DOWN in rate, frequencies above the new Nyquist limit
    cannot be represented; left in, they fold back as false low tones (aliasing). So the sinc is stretched
    to cut everything above the lower of the two Nyquist limits: the low-pass filter and the interpolation
    are the same operation."""
    if sr_in == sr_out:
        return wave
    cutoff = min(1.0, sr_out / sr_in) * 0.95  # as a fraction of the input Nyquist; 0.95 leaves room for the roll-off
    reach = math.ceil(width / cutoff)  # how many input samples each output sample looks at, per side
    t = torch.arange(int(len(wave) * sr_out / sr_in), dtype=torch.float64) * sr_in / sr_out  # output times, in input samples
    centre = t.floor().long()
    out = torch.empty(len(t), dtype=torch.float32)
    offsets = torch.arange(-reach, reach + 1)
    padded = torch.nn.functional.pad(wave, (reach, reach))
    for i in range(0, len(t), 100_000):  # in chunks: the (outputs x taps) table would not fit in memory at once
        idx = centre[i : i + 100_000, None] + offsets  # (n, taps) input sample indices
        x = (idx - t[i : i + 100_000, None]) * cutoff
        taps = torch.sinc(x) * torch.cos(math.pi * x / (2 * width)).pow(2) * (x.abs() < width) * cutoff  # sinc x Hann window
        out[i : i + 100_000] = (padded[idx + reach] * taps.float()).sum(1)
    return out


def write_wav(path: str | Path, wave: np.ndarray | torch.Tensor, sr: int = SR) -> None:
    """16-bit PCM: the same header ../stt/audio.py reads, written by hand."""
    pcm = (np.clip(np.asarray(wave, dtype=np.float32), -1, 1) * 32767).astype("<i2").tobytes()
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
    Path(path).write_bytes(header + b"data" + struct.pack("<I", len(pcm)) + pcm)


def trim_silence(wave: torch.Tensor, threshold_db: float = -40.0, keep: float = 0.05) -> torch.Tensor:
    """Cut leading and trailing silence, keeping 50 ms of margin. Silence the model has to "speak" for an
    unpredictable time at the edges only makes durations harder to learn."""
    frames = wave[: len(wave) // HOP * HOP].view(-1, HOP)
    loud = (10 * torch.log10(frames.pow(2).mean(1) + 1e-10) > threshold_db).nonzero()
    if len(loud) == 0:
        return wave
    margin = int(keep * SR)
    return wave[max(0, int(loud[0]) * HOP - margin) : int(loud[-1] + 1) * HOP + margin]


def stft(wave: torch.Tensor) -> torch.Tensor:
    """(samples,) -> complex (frames, 513). ../stt/audio.py wrote this loop out; here we call the library."""
    return torch.stft(wave, N_FFT, HOP, window=torch.hann_window(N_FFT, device=wave.device), return_complex=True).T


@cache
def _bank(device: torch.device) -> torch.Tensor:
    """The (513, 80) mel matrix, built once per device: the vocoder calls log_mel 32 times a step."""
    return mel_filterbank(N_MELS, N_FFT, SR).T.to(device)


def log_mel(wave: torch.Tensor) -> torch.Tensor:
    """(samples,) at 22.05 kHz -> (frames, 80). Magnitude (not power) and log, the convention of TTS vocoders."""
    mel = stft(wave).abs() @ _bank(wave.device)
    return torch.log(mel.clamp(min=1e-5))


def griffin_lim(log_mel_frames: torch.Tensor, rounds: int = 32) -> torch.Tensor:
    """Log-mel (frames, 80) -> waveform, with no trained model.

    Step 1: undo the mel filterbank. 80 bands cannot give back 513 bins exactly, so use the pseudo-inverse:
            the least-squares best guess of the linear spectrogram.
    Step 2: Griffin-Lim for the phase."""
    bank = mel_filterbank(N_MELS, N_FFT, SR)
    magnitude = (log_mel_frames.exp() @ torch.linalg.pinv(bank).T).clamp(min=0).T  # (513, frames)
    window = torch.hann_window(N_FFT)
    phase = torch.exp(2j * math.pi * torch.rand_like(magnitude))  # random starting phases
    for _ in range(rounds):
        wave = torch.istft(magnitude * phase, N_FFT, HOP, window=window)  # magnitudes + current phases -> waveform
        again = torch.stft(wave, N_FFT, HOP, window=window, return_complex=True)  # what does that waveform really contain?
        phase = again / again.abs().clamp(min=1e-8)  # keep its phases, discard its magnitudes
    return torch.istft(magnitude * phase, N_FFT, HOP, window=window)


def demo() -> None:
    t = torch.arange(SR) / SR
    # A vowel-like sound: a 150 Hz pitch with harmonics, loudest around 700 Hz.
    wave = sum(math.exp(-((150 * k - 700) / 500) ** 2) * torch.sin(2 * math.pi * 150 * k * t) for k in range(1, 30)) * 0.1
    mel = log_mel(wave)
    print(f"1 s at {SR} Hz = {len(wave)} samples -> log-mel {tuple(mel.shape)}  ({SR / HOP:.0f} frames / second)")

    for rounds in (1, 8, 32):
        rebuilt = griffin_lim(mel, rounds)
        error = (log_mel(rebuilt)[2:-2] - mel[2 : len(log_mel(rebuilt)) - 2]).abs().mean()
        print(f"Griffin-Lim, {rounds:2d} rounds: mel of the rebuilt audio is off by {error:.3f} (log units)")

    high = torch.sin(2 * math.pi * 9000 * torch.arange(44100) / 44100)  # a 9 kHz tone recorded at 44.1 kHz
    naive, proper = high[::2], resample(high, 44100, 22050)  # 9 kHz fits under the new 11 kHz limit: both keep it
    print(f"\n44.1 kHz -> 22.05 kHz, 9 kHz tone:  naive (every 2nd sample) power {naive.pow(2).mean():.2f}, sinc {proper.pow(2).mean():.2f}")
    high = torch.sin(2 * math.pi * 15000 * torch.arange(44100) / 44100)  # 15 kHz is ABOVE the new limit
    naive, proper = high[::2], resample(high, 44100, 22050)
    peak = torch.fft.rfft(naive).abs().argmax().item() * 22050 / len(naive)
    print(f"                       15 kHz tone: naive keeps power {naive.pow(2).mean():.2f} as a FALSE {peak:.0f} Hz tone (aliasing); "
          f"sinc removes it: {proper.pow(2).mean():.4f}")


if __name__ == "__main__":
    demo()
