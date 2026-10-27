"""Solution to drill 1. Same maths as stt/audio.py, in numpy."""

import numpy as np

SR, N_FFT, HOP, N_MELS = 16000, 400, 160, 80


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_filterbank(n_mels=N_MELS, n_fft=N_FFT, sr=SR):
    mels = np.linspace(hz_to_mel(0.0), hz_to_mel(sr / 2), n_mels + 2)
    corners = 700.0 * (10 ** (mels / 2595.0) - 1.0)
    bins = np.linspace(0, sr / 2, n_fft // 2 + 1)
    left, centre, right = corners[:-2, None], corners[1:-1, None], corners[2:, None]
    return np.maximum(0.0, np.minimum((bins - left) / (centre - left), (right - bins) / (right - centre)))


def log_mel(wave):
    wave = np.pad(np.asarray(wave, dtype=np.float32), N_FFT // 2, mode="reflect")
    n = 1 + (len(wave) - N_FFT) // HOP
    frames = np.stack([wave[i * HOP : i * HOP + N_FFT] for i in range(n)])
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)  # periodic, like torch.hann_window
    power = np.abs(np.fft.rfft(frames * hann, axis=-1)) ** 2
    return np.log(np.maximum(power @ mel_filterbank().T, 1e-10))
