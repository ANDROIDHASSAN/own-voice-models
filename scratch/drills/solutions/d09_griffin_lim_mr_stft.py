"""Solution for d09_griffin_lim_mr_stft.py"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tts"))
from audio import N_FFT, HOP  # noqa: E402


def griffin_lim_step(magnitude: torch.Tensor, phase: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    """One Griffin-Lim iteration.  audio.py:116-119."""
    window = torch.hann_window(n_fft)
    wave = torch.istft(magnitude * phase, n_fft, hop, window=window)
    again = torch.stft(wave, n_fft, hop, window=window, return_complex=True)
    return again / again.abs().clamp(min=1e-8)


def spectral_convergence(fake: torch.Tensor, real: torch.Tensor, n_fft: int) -> torch.Tensor:
    """Relative L2 spectral convergence term.  vocoder.py:107."""
    hop = n_fft // 4
    window = torch.hann_window(n_fft, device=fake.device)
    f = torch.stft(fake, n_fft, hop, window=window, return_complex=True).abs().clamp(min=1e-5)
    r = torch.stft(real, n_fft, hop, window=window, return_complex=True).abs().clamp(min=1e-5)
    return (r - f).pow(2).sum().sqrt() / r.pow(2).sum().sqrt()
