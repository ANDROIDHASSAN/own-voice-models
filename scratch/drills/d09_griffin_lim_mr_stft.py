"""Drill 9 - vocoder fundamentals: one step of Griffin-Lim and the spectral convergence term
from the multi-resolution STFT loss (tts/audio.py, tts/vocoder.py).

    python d09_griffin_lim_mr_stft.py              # checks YOUR functions below
    python d09_griffin_lim_mr_stft.py --solution   # checks solutions/d09_griffin_lim_mr_stft.py

Write:
  griffin_lim_step(magnitude, phase, n_fft, hop)
      One Griffin-Lim iteration (audio.py:116-119, one round):
        - Build complex spectrum: magnitude * phase
        - iSTFT to waveform, then re-STFT
        - Return the new phases (unit complex) of that re-analysis
      magnitude: (F, T) real, phase: (F, T) complex unit, returns (F, T) complex unit.

  spectral_convergence(fake, real, n_fft)
      The relative L2 spectral convergence term from vocoder.py:107:
        sqrt( sum((|R| - |F|)^2) ) / sqrt( sum(|R|^2) )
      fake, real: (N,) 1-D waveforms. Use hop = n_fft // 4 and a Hann window.
      Return a scalar tensor.

torch only. About 10 minutes.
"""

import sys
from pathlib import Path

import torch


def griffin_lim_step(magnitude: torch.Tensor, phase: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    raise NotImplementedError


def spectral_convergence(fake: torch.Tensor, real: torch.Tensor, n_fft: int) -> torch.Tensor:
    raise NotImplementedError


def check(m) -> None:
    import math
    sys.path.insert(0, str(Path(__file__).parent.parent / "tts"))
    from audio import griffin_lim, log_mel, SR, N_FFT, HOP
    from vocoder import spectral_loss

    # --- griffin_lim_step ---
    torch.manual_seed(7)
    t = torch.arange(SR) / SR
    wave = sum(
        math.exp(-((150 * k - 700) / 500) ** 2) * torch.sin(2 * math.pi * 150 * k * t)
        for k in range(1, 30)
    ) * 0.1  # same vowel-like wave as audio.py demo

    window = torch.hann_window(N_FFT)
    spec = torch.stft(wave, N_FFT, HOP, window=window, return_complex=True)  # (F, T)
    magnitude = spec.abs()
    rand_phase = torch.exp(2j * math.pi * torch.rand_like(magnitude))

    new_phase = m.griffin_lim_step(magnitude, rand_phase, N_FFT, HOP)
    assert new_phase.shape == magnitude.shape, "phase shape must match magnitude"
    assert new_phase.is_complex(), "must be complex"
    true_phase = spec / spec.abs().clamp(min=1e-8)
    rand_err = (rand_phase - true_phase).abs().mean().item()
    step_err = (new_phase - true_phase).abs().mean().item()
    assert step_err < rand_err, (
        f"one GL step from true magnitude should improve phase: rand {rand_err:.4f}, step {step_err:.4f}"
    )

    # 32-round Griffin-Lim mel error should match audio.py demo (~0.398)
    mel = log_mel(wave)
    rebuilt = griffin_lim(mel, rounds=32)
    ref_err = (log_mel(rebuilt)[2:-2] - mel[2 : len(log_mel(rebuilt)) - 2]).abs().mean().item()
    assert abs(ref_err - 0.398) < 0.05, f"Griffin-Lim 32-round mel error should be ~0.398, got {ref_err:.3f}"

    # --- spectral_convergence ---
    torch.manual_seed(3)
    real_w = torch.randn(SR)
    fake_w = real_w + 0.1 * torch.randn(SR)

    sc = m.spectral_convergence(fake_w, real_w, 1024)
    assert sc.ndim == 0, "should return a scalar"
    assert 0 < sc.item() < 1.0, f"small perturbation should give a small convergence: {sc.item():.4f}"

    full_loss = spectral_loss(fake_w.unsqueeze(0), real_w.unsqueeze(0))
    assert full_loss.item() > sc.item(), "full multi-resolution loss > single-scale convergence"

    sc_zero = m.spectral_convergence(real_w, real_w, 1024)
    assert sc_zero.item() < 1e-6, f"identical waves: convergence should be 0, got {sc_zero.item()}"

    print("PASS  griffin_lim_step improves phase, spectral_convergence matches vocoder.py formula")


if __name__ == "__main__":
    if "--solution" in sys.argv:
        sys.path.insert(0, str(Path(__file__).parent / "solutions"))
        import d09_griffin_lim_mr_stft as mod
    else:
        mod = sys.modules[__name__]
    check(mod)
