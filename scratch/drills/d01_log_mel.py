"""Drill 1 - log-mel from scratch, numpy only (interview whiteboard version of stt/audio.py).

    python d01_log_mel.py              # checks YOUR functions below
    python d01_log_mel.py --solution   # checks solutions/d01_log_mel.py

Write the three functions without looking at stt/audio.py. Settings: 16 kHz, 25 ms window (400 samples),
10 ms hop (160), Hann window, centre padding by reflection, 80 mel bands, natural log with a 1e-10 floor.
Time budget in an interview: about 15 minutes.
"""

import sys
from pathlib import Path

import numpy as np

SR, N_FFT, HOP, N_MELS = 16000, 400, 160, 80


def hz_to_mel(hz):
    """Mel of a frequency in Hz (the 2595 / 700 formula). Must work on floats and numpy arrays."""
    raise NotImplementedError


def mel_filterbank(n_mels=N_MELS, n_fft=N_FFT, sr=SR):
    """(n_mels, n_fft // 2 + 1) matrix of triangles, evenly spaced on the mel scale from 0 to sr / 2."""
    raise NotImplementedError


def log_mel(wave):
    """(samples,) float waveform -> (frames, n_mels) log-mel. Frame t is centred on sample t * HOP."""
    raise NotImplementedError


def check(m) -> None:
    sys.path.insert(0, str(Path(__file__).parents[1] / "stt"))
    import torch

    import audio as ref

    assert abs(m.hz_to_mel(1000.0) - ref.hz_to_mel(1000.0)) < 1e-6, "hz_to_mel"
    fb = np.asarray(m.mel_filterbank())
    assert fb.shape == (80, 201), f"filterbank shape {fb.shape}, want (80, 201)"
    assert np.allclose(fb, ref.mel_filterbank().numpy(), atol=1e-5), "filterbank values"

    rng = np.random.default_rng(0)
    wave = rng.standard_normal(SR).astype(np.float32) * 0.1
    got = np.asarray(m.log_mel(wave))
    want = ref.log_mel(torch.from_numpy(wave)).numpy()
    assert got.shape == want.shape == (101, 80), f"log_mel shape {got.shape}, want {want.shape}"
    assert np.allclose(got, want, atol=1e-3), f"log_mel max diff {np.abs(got - want).max():.4f}"

    t = np.arange(SR) / SR
    tone = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    band = int(np.asarray(m.log_mel(tone)).mean(0).argmax())
    print(f"PASS  hz_to_mel, filterbank (80, 201), log_mel (101, 80); a 440 Hz tone peaks in mel band {band}")


if __name__ == "__main__":
    if "--solution" in sys.argv:
        sys.path.insert(0, str(Path(__file__).parent / "solutions"))
        import d01_log_mel as mod
    else:
        mod = sys.modules[__name__]
    check(mod)
