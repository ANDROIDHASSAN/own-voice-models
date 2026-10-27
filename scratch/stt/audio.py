"""Lesson 1 - audio: from a .wav file to the picture a speech model actually looks at.

    python audio.py

Sound is air pressure changing over time. A microphone measures that pressure 16,000 times a second
(the sampling rate), so one second of speech is 16,000 numbers between -1 and 1: the waveform.

A model could read those raw numbers, but nothing in them looks like "a" or "k". What makes a vowel a vowel is
WHICH FREQUENCIES are loud, and that is invisible in the waveform. So we change the picture:

    waveform (16,000 numbers / second)
      -> cut into short overlapping frames        25 ms long, a new one every 10 ms
      -> Fourier transform of each frame          how loud is each frequency in this 25 ms?
      -> mel filterbank                           squeeze 201 frequencies into 80 bands spaced like the ear
      -> log                                      loudness is heard in ratios, not differences
    = log-mel spectrogram: 100 frames / second, 80 numbers each

That 2-D array (time x frequency) is the input of the model in lesson 5. Whisper, Conformer and Squeezeformer
all start from this same picture.
"""

import math
import struct
from pathlib import Path

import numpy as np
import torch

SR = 16000  # sampling rate: samples per second. Speech needs up to ~8 kHz, and you need 2 samples per cycle (Nyquist)
N_FFT = 400  # frame (window) length in samples: 400 / 16000 = 25 ms, short enough that the sound is "steady" inside it
HOP = 160  # step between frames: 10 ms, so frames overlap and we get 100 frames per second
N_MELS = 80


def read_wav(src: str | Path | bytes) -> tuple[np.ndarray, int]:
    """A .wav file is a small header followed by the samples, nothing else. The header says how to read them:
    sampling rate, number of channels, and the sample format (16-bit integers, or 32-bit floats).
    Returns (mono float32 samples in -1..1, sampling rate).

    Python's own `wave` module refuses float files, and our dataset is stored as float, so we parse it here."""
    raw = src if isinstance(src, bytes) else Path(src).read_bytes()
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("not a WAV file")
    fmt, data, pos = None, None, 12
    while pos + 8 <= len(raw):  # the file is a list of chunks: 4-byte name, 4-byte size, then the bytes
        name, size = raw[pos : pos + 4], struct.unpack("<I", raw[pos + 4 : pos + 8])[0]
        body = raw[pos + 8 : pos + 8 + size]
        if name == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])  # format tag, channels, rate, byte rate, block align, bits
            if fmt[0] == 0xFFFE:  # "extensible": the real format tag sits further inside the chunk
                fmt = (struct.unpack("<H", body[24:26])[0], *fmt[1:])
        elif name == b"data":
            data = body
        pos += 8 + size + (size & 1)  # chunks are padded to an even number of bytes
    if fmt is None or data is None:
        raise ValueError("WAV file has no fmt or data chunk")
    tag, channels, rate, _, _, bits = fmt
    if tag == 3 and bits == 32:  # IEEE float: already -1..1
        x = np.frombuffer(data, dtype="<f4")
    elif tag == 1 and bits == 16:  # PCM integers: divide by the largest value to get -1..1
        x = np.frombuffer(data, dtype="<i2") / 32768.0
    elif tag == 1 and bits == 32:
        x = np.frombuffer(data, dtype="<i4") / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample format: tag {tag}, {bits} bits")
    x = x[: len(x) // channels * channels].reshape(-1, channels).mean(axis=1)  # stereo -> mono: average the channels
    return x.astype(np.float32), rate


def hz_to_mel(hz: float) -> float:
    """The mel scale follows human hearing: we tell 100 Hz from 200 Hz easily, but 7100 Hz from 7200 Hz barely.
    So mel is roughly linear below 1 kHz and logarithmic above it."""
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_filterbank(n_mels: int = N_MELS, n_fft: int = N_FFT, sr: int = SR) -> torch.Tensor:
    """(n_mels, n_fft / 2 + 1) matrix. Row i is a triangle over the FFT bins: band i's weights.
    The triangles are evenly spaced on the MEL scale, so in Hz they are narrow at the bottom (where speech
    carries most information) and wide at the top. Multiplying by this matrix turns 201 bins into 80 bands."""
    mels = torch.linspace(hz_to_mel(0.0), hz_to_mel(sr / 2), n_mels + 2)  # n_mels triangles need n_mels + 2 corner points
    corners = 700.0 * (10 ** (mels / 2595.0) - 1.0)  # back to Hz
    bins = torch.linspace(0, sr / 2, n_fft // 2 + 1)  # the frequency of every FFT bin
    left, centre, right = corners[:-2, None], corners[1:-1, None], corners[2:, None]
    rising = (bins - left) / (centre - left)
    falling = (right - bins) / (right - centre)
    return torch.minimum(rising, falling).clamp(min=0.0)


def power_spectrogram(wave: torch.Tensor) -> torch.Tensor:
    """Short-time Fourier transform, written out. wave: (samples,) -> (frames, n_fft / 2 + 1).

    The Fourier transform answers "which frequencies is this signal made of?" but for the WHOLE signal at
    once. Speech changes every few milliseconds, so we take it of one short frame at a time.

    Hann window: each frame is faded in and out before the transform. Cutting a frame out with hard edges
    creates a click at both ends, which shows up as fake frequencies (spectral leakage)."""
    wave = torch.nn.functional.pad(wave[None], (N_FFT // 2, N_FFT // 2), mode="reflect")[0]  # so frame t is centred on t * HOP
    frames = wave.unfold(0, N_FFT, HOP)  # (frames, N_FFT): overlapping slices
    spectrum = torch.fft.rfft(frames * torch.hann_window(N_FFT), dim=-1)  # complex: magnitude and phase per frequency
    return spectrum.abs().pow(2)  # keep the energy, drop the phase: the ear mostly ignores phase


def log_mel(wave: torch.Tensor | np.ndarray) -> torch.Tensor:
    """wave: (samples,) at 16 kHz -> log-mel spectrogram (frames, 80), 100 frames per second."""
    wave = torch.as_tensor(wave, dtype=torch.float32)
    mel = power_spectrogram(wave) @ mel_filterbank().T
    return torch.log(mel.clamp(min=1e-10))  # log: a sound 10x louder adds a constant instead of multiplying


def demo() -> None:
    t = torch.arange(SR) / SR
    tone = torch.sin(2 * math.pi * 440.0 * t)  # one second of a pure 440 Hz tone (the note A)
    power = power_spectrogram(tone)
    peak_hz = power.mean(0).argmax().item() * SR / N_FFT
    mel = log_mel(tone)
    print(f"1 s of audio = {len(tone)} samples -> {tuple(power.shape)} power spectrogram -> {tuple(mel.shape)} log-mel")
    print(f"a 440 Hz tone is loudest in the FFT bin at {peak_hz:.0f} Hz (bins are {SR / N_FFT:.0f} Hz apart)")
    assert abs(peak_hz - 440.0) <= SR / N_FFT

    # A chirp: a tone sliding from 200 Hz to 4000 Hz. In the spectrogram it must show up as a rising line.
    sweep = torch.sin(2 * math.pi * (200.0 * t + (4000.0 - 200.0) / 2 * t * t))
    picture = log_mel(sweep)[::5, ::4]  # every 5th frame, every 4th band: small enough to print
    shades = " .:-=+*#%@"
    scaled = ((picture - picture.min()) / (picture.max() - picture.min()) * (len(shades) - 1)).long()
    print("\nlog-mel of a rising tone (time left to right, low frequencies at the bottom):")
    for band in reversed(range(scaled.shape[1])):
        print("  " + "".join(shades[v] for v in scaled[:, band].tolist()))
    centres = mel_filterbank().argmax(dim=1) * SR / N_FFT
    print(f"\nmel band 10 is centred near {centres[10]:.0f} Hz, band 40 near {centres[40]:.0f} Hz, band 70 near "
          f"{centres[70]:.0f} Hz: the bands get wider as frequency rises")


if __name__ == "__main__":
    demo()
