"""Lesson 7 - the vocoder: a small network that turns a mel spectrogram into a waveform.

    python vocoder.py                  # train
    python vocoder.py --resume

Lesson 1 ended with the problem: a mel spectrogram has no phase, and Griffin-Lim's guessed phase sounds
metallic. A neural vocoder LEARNS to produce a waveform that sounds right for a given mel.

Two designs:
    HiFi-GAN (2020)   upsample the mel 256x with transposed convolutions until it is a waveform. Every layer
                      runs at a higher sample rate than the last, so most of the compute happens at 22 kHz.
    Vocos (2023)      never leave the frame rate. For each of the 86 frames per second, predict the
                      magnitude AND the phase of all 513 frequency bins, then let one inverse STFT (no
                      learning, lesson 1) produce the samples. The paper measures about 13x HiFi-GAN's speed
                      on GPU, and far more on CPU.

We build the Vocos design: it is the reason a vocoder can run in a browser tab.

    mel (T, 80) -> conv -> N x ConvNeXt block (T, dim) -> linear -> magnitude (T, 513), phase (T, 513) -> iSTFT -> samples

The loss here is purely spectral: the generated and the real waveform should have the same spectrogram at
three different window sizes ("multi-resolution STFT loss": a short window checks timing, a long one checks
pitch). We do not compare waveforms sample by sample, because two waveforms with different phases can sound
identical. Published vocoders add a GAN on top (a second network that learns to tell real from generated
audio) to remove the last of the buzz; that is the listed next step in README.md.
"""

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from audio import DATA_DIR, HOP, N_FFT, N_MELS, RUNS_DIR, SR, log_mel, mel_filterbank
from model import Stepper, pick_device

HERE = Path(__file__).parent
DATA, RUNS = DATA_DIR, RUNS_DIR


@dataclass
class Config:
    n_mels: int = N_MELS
    dim: int = 256
    n_layer: int = 6


class ConvNeXtBlock(nn.Module):
    """A depthwise convolution over 7 neighbouring frames, then a small MLP on every frame, with a residual.
    The same recipe as the ConvModule + FeedForward of ../stt/model.py, without attention: a vocoder only
    needs local context (a few frames decide the waveform), so it can do without the global view."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.up, self.down = nn.Linear(dim, 3 * dim), nn.Linear(3 * dim, dim)
        self.scale = nn.Parameter(torch.full((dim,), 0.1))  # each block starts as a small correction

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, dim)
        y = self.norm(self.depthwise(x.transpose(1, 2)).transpose(1, 2))
        return x + self.scale * self.down(F.gelu(self.up(y)))


class Vocoder(nn.Module):
    def __init__(self, cfg: Config, mean: float = 0.0, std: float = 1.0) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))
        self.register_buffer("window", torch.hann_window(N_FFT), persistent=False)
        self.embed = nn.Conv1d(cfg.n_mels, cfg.dim, 7, padding=3)
        self.norm_in = nn.LayerNorm(cfg.dim)
        self.blocks = nn.ModuleList(ConvNeXtBlock(cfg.dim) for _ in range(cfg.n_layer))
        self.norm_out = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, N_FFT + 2)  # 513 log-magnitudes + 513 phases per frame

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: (B, T, 80) raw log-mel -> waveform (B, (T - 1) * 256)."""
        x = self.embed(((mel - self.mean) / self.std).transpose(1, 2)).transpose(1, 2)
        x = self.norm_in(x)
        for block in self.blocks:
            x = block(x)
        log_mag, phase = self.head(self.norm_out(x)).float().chunk(2, dim=-1)  # fp32 from here: exp and iSTFT
        mag = log_mag.exp().clamp(max=100.0)  # the clamp stops one wild early prediction from blowing up the iSTFT
        # A complex number from magnitude and angle. cos / sin make the phase wrap around: 0 and 2 pi are the same.
        spectrum = torch.complex(mag * phase.cos(), mag * phase.sin())
        return torch.istft(spectrum.transpose(1, 2), N_FFT, HOP, window=self.window)


def spectral_loss(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """Multi-resolution STFT loss + mel loss. fake, real: (B, samples)."""
    loss = 0.0
    for n_fft in (512, 1024, 2048):
        window = torch.hann_window(n_fft, device=fake.device)
        f = torch.stft(fake, n_fft, n_fft // 4, window=window, return_complex=True).abs().clamp(min=1e-5)
        r = torch.stft(real, n_fft, n_fft // 4, window=window, return_complex=True).abs().clamp(min=1e-5)
        convergence = (r - f).pow(2).sum().sqrt() / r.pow(2).sum().sqrt()  # relative error: dominated by the loud parts
        log_magnitude = (r.log() - f.log()).abs().mean()  # the same on a log scale: the quiet parts count too
        loss = loss + convergence + log_magnitude
        if n_fft == N_FFT:
            bank = mel_filterbank(N_MELS, N_FFT, SR).to(fake.device)
            mel = lambda s: torch.log((s.transpose(1, 2) @ bank.T).clamp(min=1e-5))
            loss = loss + (mel(r) - mel(f)).abs().mean()  # the acoustic model's own currency
    return loss


class PeriodDiscriminator(nn.Module):
    """HiFi-GAN's multi-period idea: fold the waveform into rows of `period` samples, so 2-D convolutions see
    samples one pitch-period apart side by side. A buzzy or smeared periodic structure is easy to spot there."""

    def __init__(self, period: int, ch: int = 32) -> None:
        super().__init__()
        self.period = period
        sizes = [1, ch, ch * 4, ch * 8, ch * 8]
        self.convs = nn.ModuleList(nn.utils.parametrizations.weight_norm(nn.Conv2d(a, b, (5, 1), (3 if i < 3 else 1, 1), (2, 0)))
                                   for i, (a, b) in enumerate(zip(sizes, sizes[1:])))
        self.out = nn.utils.parametrizations.weight_norm(nn.Conv2d(sizes[-1], 1, (3, 1), 1, (1, 0)))

    def forward(self, wave: torch.Tensor) -> list[torch.Tensor]:  # wave (B, N) -> every layer's output, score last
        x = F.pad(wave, (0, -wave.shape[1] % self.period), mode="reflect")[:, None]
        x = x.view(x.shape[0], 1, -1, self.period)
        feats = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            feats.append(x)
        return feats + [self.out(x)]


class ResolutionDiscriminator(nn.Module):
    """Vocos / UnivNet's multi-resolution idea: look at the magnitude spectrogram at one window size.
    Small windows judge timing (clicks, attacks), big ones judge pitch (harmonics)."""

    def __init__(self, n_fft: int, ch: int = 32) -> None:
        super().__init__()
        self.n_fft = n_fft
        wn = nn.utils.parametrizations.weight_norm
        self.convs = nn.ModuleList([wn(nn.Conv2d(1, ch, (3, 9), padding=(1, 4)))] +
                                   [wn(nn.Conv2d(ch, ch, (3, 9), (1, 2), (1, 4))) for _ in range(3)] +
                                   [wn(nn.Conv2d(ch, ch, 3, padding=1))])
        self.out = wn(nn.Conv2d(ch, 1, 3, padding=1))

    def forward(self, wave: torch.Tensor) -> list[torch.Tensor]:
        window = torch.hann_window(self.n_fft, device=wave.device)
        x = torch.stft(wave, self.n_fft, self.n_fft // 4, window=window, return_complex=True).abs()[:, None]  # (B, 1, F, T)
        feats = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            feats.append(x)
        return feats + [self.out(x)]


class Discriminators(nn.Module):
    """Lesson 10: the critic that removes the last of the buzz. It learns to tell real audio from the vocoder's;
    the vocoder learns to fool it. A spectral loss alone accepts any phase that gives the right magnitudes,
    and a slightly wrong phase is exactly what sounds metallic. Vocos trains with these two kinds."""

    def __init__(self) -> None:
        super().__init__()
        self.all = nn.ModuleList([PeriodDiscriminator(p) for p in (2, 3, 5, 7, 11)] +
                                 [ResolutionDiscriminator(n) for n in (512, 1024, 2048)])

    def forward(self, wave: torch.Tensor) -> list[list[torch.Tensor]]:
        return [d(wave) for d in self.all]


def critic_loss(disc: Discriminators, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """Hinge loss, as in Vocos: real scores pushed above +1, fake below -1. The fake is detached, so this
    backward trains only the critic."""
    return sum(F.relu(1 - r[-1]).mean() + F.relu(1 + f[-1]).mean() for r, f in zip(disc(real), disc(fake.detach())))


def fooling_loss(disc: Discriminators, fake: torch.Tensor, real: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """For the vocoder: (adversarial, feature matching). Feature matching (HiFi-GAN): the vocoder's audio should
    also look like real audio INSIDE the critic, layer by layer, which steadies the adversarial game."""
    with torch.no_grad():
        real_out = disc(real)
    fake_out = disc(fake)
    adv = sum(F.relu(1 - f[-1]).mean() for f in fake_out)
    fm = sum((a - b).abs().mean() for r, f in zip(real_out, fake_out) for a, b in zip(r[:-1], f[:-1]))
    return adv, fm


def save(model: Vocoder, path, **extra) -> None:
    torch.save({"config": asdict(model.cfg), "model": model.state_dict(), **extra}, path)


def load(path, device: str | None = None) -> tuple[Vocoder, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = Vocoder(Config(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device or pick_device()).eval(), ckpt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, help="default 5e-4; 1e-4 with --gan (fine-tuning a trained vocoder: 5e-4 kicked its dev loss up)")
    ap.add_argument("--crop", type=int, default=64, help="training crop in mel frames: 64 frames = 0.74 s")
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--gan", action="store_true", help="lesson 10: fine-tune runs/vocoder.pt against the critics -> runs/vocoder_gan.pt")
    ap.add_argument("--recon", type=float, default=15.0, help="GAN mode: weight of the spectral loss next to the adversarial one")
    args = ap.parse_args()
    name = "vocoder_gan" if args.gan else "vocoder"
    args.lr = args.lr or (1e-4 if args.gan else 5e-4)

    torch.manual_seed(0)
    np.random.seed(0)
    device = pick_device()
    RUNS.mkdir(exist_ok=True)
    meta = json.loads((DATA / "meta.json").read_text())
    train = np.memmap(DATA / "train.pcm", dtype=np.int16, mode="r")
    dev = np.memmap(DATA / "dev.pcm", dtype=np.int16, mode="r")

    def batch_from(pcm: np.ndarray, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """n random crops. A vocoder never needs whole sentences: it learns frame-local sound, so short random
        windows are enough, and every step sees fresh ones."""
        samples = (args.crop - 1) * HOP
        starts = np.random.randint(0, len(pcm) - samples, n)
        wave = torch.from_numpy(np.stack([pcm[s : s + samples] for s in starts]).astype(np.float32) / 32768).to(device)
        return torch.stack([log_mel(w) for w in wave]), wave

    start, best = 0, float("inf")
    if args.resume:
        model, ckpt = load(RUNS / f"{name}_last.pt", device)
        start, best = ckpt["step"], ckpt["best"]
    elif args.gan:  # start from the spectral-loss vocoder: a GAN from scratch is slow and unstable (Vocos sec. 2)
        model, _ = load(RUNS / "vocoder.pt", device)
    else:
        model = Vocoder(Config(), meta["mean"], meta["std"]).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.8, 0.99), weight_decay=0.0)
    if args.resume:
        opt.load_state_dict(ckpt["opt"])
    disc = opt_d = None
    if args.gan:
        disc = Discriminators().to(device)
        opt_d = torch.optim.AdamW(disc.parameters(), lr=args.lr, betas=(0.8, 0.99), weight_decay=0.0)
        if args.resume:
            disc.load_state_dict(ckpt["disc"])
            opt_d.load_state_dict(ckpt["opt_d"])
    print(f"{device}  {model.n_params() / 1e6:.2f} M parameters  {len(train) / SR / 3600:.1f} h of audio  {args.steps} steps", flush=True)

    stepper, critic_step = Stepper(device), Stepper(device)  # one loss scale per optimizer
    np.random.seed(1)
    dev_batches =[batch_from(dev, args.batch) for _ in range(8)]  # fixed, so dev numbers are comparable across evals
    np.random.seed(start)
    t0 = time.perf_counter()
    for step in range(start, args.steps):
        lr = args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * step / args.steps)))  # cosine decay, no warmup needed
        for group in opt.param_groups + (opt_d.param_groups if opt_d else []):
            group["lr"] = lr
        mel, wave = batch_from(train, args.batch)
        with stepper.autocast():
            fake = model(mel)  # fp32 out: the head is cast before the iSTFT
        extra = ""
        if disc is None:
            loss = spectral_loss(fake, wave)  # outside autocast: logs of tiny magnitudes need fp32
        else:
            with stepper.autocast():
                d_loss = critic_loss(disc, fake, wave)  # 1. the critic learns, on a detached fake
            critic_step(d_loss, opt_d, disc.parameters(), 10.0)
            with stepper.autocast():
                adv, fm = fooling_loss(disc, fake, wave)  # 2. the vocoder learns to fool the updated critic
            recon = spectral_loss(fake, wave)
            loss = args.recon * recon + adv.float() + 2.0 * fm.float()  # HiFi-GAN's weighting: feature matching x2
            extra = f"  spectral {recon.item():.3f}  adv {adv.item():.3f}  fm {fm.item():.3f}  critic {d_loss.item():.3f}"
        grad_norm = stepper(loss, opt, model.parameters(), 10.0)
        if step % 100 == 0:
            print(f"step {step:6d}  loss {loss.item():.3f}{extra}  grad-norm {grad_norm:.2f}  lr {lr:.2e}  "
                  f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            model.eval()
            with torch.no_grad():
                dev_loss = sum(spectral_loss(model(m), w).item() for m, w in dev_batches) / len(dev_batches)
            model.train()
            print(f"== step {step + 1}  dev loss {dev_loss:.3f}", flush=True)
            # With a GAN the spectral dev loss may RISE while the sound improves (the critic trades a little
            # spectral accuracy for realism), so the GAN run keeps its latest weights, not the lowest loss.
            if dev_loss < best or disc is not None:
                best = min(best, dev_loss)
                save(model, RUNS / f"{name}.pt", step=step + 1, dev_loss=dev_loss)
            gan_state = {"disc": disc.state_dict(), "opt_d": opt_d.state_dict()} if disc is not None else {}
            save(model, RUNS / f"{name}_last.pt", step=step + 1, best=best, opt=opt.state_dict(), **gan_state)


if __name__ == "__main__":
    main()
