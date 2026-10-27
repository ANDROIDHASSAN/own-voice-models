"""Lesson 5 - the acoustic model: characters in, a mel spectrogram out, all frames at once.

What the whole thing does, in one sentence:
    given the characters of a sentence, decide how long each one lasts and draw the spectrogram of it being spoken.

The data flow for a sentence of L characters that turns into T mel frames (d = d_model):

    character ids (L) + language id
      -> embeddings                     (L, d)
      -> text encoder, N blocks         (L, d)    each character now knows its neighbours ("c" before "h" ...)
      -> duration predictor             (L,)      frames per character
      -> length regulator               (T, d)    copy each character's vector `duration` times
      -> decoder, N blocks              (T, d)    smooth the copies into speech: transitions, pitch movement
      -> linear                         (T, 80)   the mel spectrogram

This is the FastSpeech family (Ren 2019; FastPitch is what AI4Bharat found best for Indian languages):
NON-AUTOREGRESSIVE. Tacotron generated frame t from frame t - 1, with attention deciding where in the text
it was; one bad attention step and it skipped or repeated words. Here the durations fix the alignment before
a single frame is drawn, so the model cannot skip or repeat, and all frames come out in one parallel pass.

During training the true durations come from MAS (lesson 4). For that the encoder also outputs `mu`, its
rough guess of each character's mel frame. Three losses are added up:

    mel loss       |decoder output - real mel|              the thing we care about
    prior loss     (mu - real mel)^2 along the alignment    makes the guesses good, so MAS aligns well
    duration loss  (predicted - MAS) log-durations          teaches the predictor what MAS found

The encoder and decoder blocks are the same Block as the recognizer (../stt/model.py): attention +
depthwise convolution + feed-forward. Text and speech need the same two things, local shape and context.

Lesson 10 - why the L1 model sounds robotic, and the fix (Config.flow = True).
    One sentence can be said in many ways. An L1 (or L2) loss is smallest for the AVERAGE of all of them,
    and the average of many sharp spectrograms is a blurred one: flat pitch, smeared harmonics. We measured
    it on our own model: 17% less frame-to-frame movement and 14% less harmonic contrast than real speech.

    Conditional flow matching (Lipman 2023; Matcha-TTS, Mehta 2024) learns to SAMPLE one sharp spectrogram:
        training   take noise x0 and the real mel x1, a random time t, the point between them
                       x_t = (1 - (1 - s) t) x0 + t x1          (s = sigma_min, tiny)
                   and teach the decoder the straight-line velocity  u = x1 - (1 - s) x0  at that point.
        speaking   start from noise and take a few Euler steps along the predicted velocity:  x += v / steps.
    Each run gives a different, sharp, natural-sounding reading instead of the blurred average.

    Expressiveness comes from what the decoder is conditioned on: the text (upsampled encoder output), the
    voice (speaker id) and the style or emotion (style id). Classifier-free guidance (Ho 2022; used by
    F5-TTS) strengthens the style: in training the style is hidden 20% of the time, so the decoder also
    learns an "any style" velocity; at synthesis  v = v_none + w (v_style - v_none)  with w > 1 pushes
    further toward the asked-for emotion.
"""

import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from align import mas

sys.path.append(str(Path(__file__).parent.parent))
from stt.model import Block, RMSNorm, Stepper, frame_mask, pick_device  # noqa: E402, F401


@dataclass
class Config:
    vocab_size: int = 120
    n_langs: int = 4
    n_mels: int = 80
    d_model: int = 192
    n_layer: int = 4  # blocks in the encoder, and again in the decoder
    n_head: int = 2
    conv_kernel: int = 7
    dropout: float = 0.1
    max_frames: int = 2048  # longest sequence a block can see: 2048 mel frames = 23 s
    # Lesson 10 (expressive). All off by default, so the first, L1-trained checkpoint still loads unchanged.
    flow: bool = False  # decoder trained with conditional flow matching instead of L1
    n_speakers: int = 0  # voices; 0 = one voice, no table
    n_styles: int = 0  # speaking styles / emotions, id 0 = "none" (needed for classifier-free guidance)
    sigma_min: float = 1e-4  # Matcha-TTS: the width left around the data at t = 1


class TimeEmbedding(nn.Module):
    """Where on the way from noise (t = 0) to speech (t = 1) we are: sinusoids of t, then a small MLP.
    The same trick as the positions of the original transformer, applied to a single number."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.d = d
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # t: (B,) in [0, 1] -> (B, d)
        freqs = torch.exp(-math.log(10000) * torch.arange(self.d // 2, device=t.device) / (self.d // 2))
        angles = 1000 * t[:, None] * freqs[None]
        return self.mlp(torch.cat((angles.sin(), angles.cos()), dim=-1))


class DurationPredictor(nn.Module):
    """Two small convolutions over the character vectors -> one number per character: log(frames).
    Log, because durations are ratios: mistaking 2 frames for 4 is as bad as 20 for 40."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(cfg.d_model, cfg.d_model, 3, padding=1)
        self.conv2 = nn.Conv1d(cfg.d_model, cfg.d_model, 3, padding=1)
        self.norm1, self.norm2 = RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.out = nn.Linear(cfg.d_model, 1)

    def forward(self, h: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        h = h * real[..., None]
        h = self.drop(self.norm1(F.relu(self.conv1(h.transpose(1, 2))).transpose(1, 2))) * real[..., None]
        h = self.drop(self.norm2(F.relu(self.conv2(h.transpose(1, 2))).transpose(1, 2)))
        return self.out(h).squeeze(-1)


def length_regulate(h: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
    """The bridge between text time and audio time. h: (B, L, d), durations: (B, L) whole frames.
    Character i's vector is repeated durations[i] times; clips are padded to the longest. Returns (B, T, d)."""
    return nn.utils.rnn.pad_sequence([torch.repeat_interleave(row, d, dim=0) for row, d in zip(h, durations)], batch_first=True)


class Acoustic(nn.Module):
    def __init__(self, cfg: Config, mean: float = 0.0, std: float = 1.0) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("mean", torch.tensor(mean))  # one number each: all 80 bands share a scale, so that
        self.register_buffer("std", torch.tensor(std))  # "louder in band 30 than band 60" stays visible in the target
        self.chars = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.langs = nn.Embedding(cfg.n_langs, cfg.d_model)
        self.encoder = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.enc_norm = RMSNorm(cfg.d_model)
        self.prior = nn.Linear(cfg.d_model, cfg.n_mels)  # mu: each character's guess of its own mel frame
        self.duration = DurationPredictor(cfg)
        self.decoder = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.dec_norm = RMSNorm(cfg.d_model)
        self.to_mel = nn.Linear(cfg.d_model, cfg.n_mels)
        if cfg.n_speakers:
            self.speakers = nn.Embedding(cfg.n_speakers, cfg.d_model)
        if cfg.n_styles:
            self.styles = nn.Embedding(cfg.n_styles, cfg.d_model)
        if cfg.flow:
            self.time = TimeEmbedding(cfg.d_model)
            self.noisy_in = nn.Linear(cfg.n_mels, cfg.d_model)  # the point x_t on the way from noise to speech
            self.style_dec = nn.Embedding(cfg.n_styles, cfg.d_model) if cfg.n_styles else None

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def condition(self, speaker: torch.Tensor | None, style: torch.Tensor | None) -> torch.Tensor | int:
        """Voice and style as one vector per sentence (B, 1, d), added to every position."""
        out = 0
        if self.cfg.n_speakers and speaker is not None:
            out = out + self.speakers(speaker)[:, None, :]
        if self.cfg.n_styles and style is not None:
            out = out + self.styles(style)[:, None, :]
        return out

    def encode(self, chars: torch.Tensor, char_lens: torch.Tensor, lang: torch.Tensor,
               speaker: torch.Tensor | None = None, style: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        real = frame_mask(char_lens, chars.shape[1])
        # the language id (and voice, and style) is added to every character: an angry "nahi" is longer and louder
        h = self.chars(chars) + self.langs(lang)[:, None, :] + self.condition(speaker, style)
        for block in self.encoder:
            h = block(h, real)
        return self.enc_norm(h), real

    def decode(self, h: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
        x = length_regulate(h, durations)
        real = frame_mask(durations.sum(1), x.shape[1])
        for block in self.decoder:
            x = block(x, real)
        return self.to_mel(self.dec_norm(x))

    def velocity(self, x_t: torch.Tensor, t: torch.Tensor, frames: torch.Tensor, real: torch.Tensor,
                 speaker: torch.Tensor | None, style: torch.Tensor | None) -> torch.Tensor:
        """The flow decoder: which way to move x_t (B, T, 80) at time t (B,), given the upsampled text
        `frames` (B, T, d). Style here is separate from the encoder's, so it can be hidden for guidance."""
        x = self.noisy_in(x_t) + frames + self.time(t)[:, None, :]
        if self.cfg.n_speakers and speaker is not None:
            x = x + self.speakers(speaker)[:, None, :]
        if self.style_dec is not None and style is not None:
            x = x + self.style_dec(style)[:, None, :]
        for block in self.decoder:
            x = block(x, real)
        return self.to_mel(self.dec_norm(x))

    def forward(self, chars, char_lens, lang, mel, mel_lens, speaker=None, style=None) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
        """Training. chars: (B, L), mel: (B, T, 80) raw log-mel. Returns (total loss, the three parts, MAS durations)."""
        target = (mel - self.mean) / self.std
        h, real = self.encode(chars, char_lens, lang, speaker, style)
        mu = self.prior(h)  # (B, L, 80)

        # score[b, l, t] = -|mu_l - mel_t|^2 / 2, expanded as a^2 - 2ab + b^2 so the (B, L, T, 80) table never exists
        # fp32 even under mixed precision (autocast would run the @ in fp16): MAS compares sums of 80 squares
        with torch.no_grad(), torch.autocast(mu.device.type, enabled=False):
            m = mu.float()
            score = -0.5 * (m.pow(2).sum(-1)[:, :, None] - 2 * m @ target.transpose(1, 2) + target.pow(2).sum(-1)[:, None, :])
            durations = mas(score, char_lens, mel_lens).to(chars.device)

        frames = frame_mask(mel_lens, mel.shape[1])[..., None]
        n = frames.sum() * self.cfg.n_mels
        prior_loss = ((length_regulate(mu, durations) - target).pow(2) * frames).sum() / n
        if self.cfg.flow:
            x0, t = torch.randn_like(target), torch.rand(len(target), device=target.device)
            x_t = (1 - (1 - self.cfg.sigma_min) * t[:, None, None]) * x0 + t[:, None, None] * target
            u = target - (1 - self.cfg.sigma_min) * x0  # the straight line's velocity: the same at every t
            if style is not None and self.training:  # hide the style 20% of the time -> learns "any style" too
                style_dec = torch.where(torch.rand(len(style), device=style.device) < 0.2, torch.zeros_like(style), style)
            else:
                style_dec = style
            v = self.velocity(x_t, t, length_regulate(h, durations), frames[..., 0], speaker, style_dec)
            mel_loss = ((v - u).pow(2) * frames).sum() / n  # "mel" in the logs: here the flow-matching loss
        else:
            mel_loss = ((self.decode(h, durations) - target).abs() * frames).sum() / n
        # detach: the duration loss trains only the predictor, it must not bend the encoder to make durations easy
        log_dur = self.duration(h.detach(), real)
        dur_loss = ((log_dur - durations.clamp(min=1).float().log()).pow(2) * real).sum() / real.sum()
        parts = {"mel": mel_loss.item(), "prior": prior_loss.item(), "dur": dur_loss.item()}
        return mel_loss + prior_loss + dur_loss, parts, durations

    @torch.no_grad()
    def synthesize(self, chars: torch.Tensor, lang: int, speed: float = 1.0, speaker: int = 0, style: int = 0,
                   steps: int = 10, temperature: float = 0.667, guidance: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        """One sentence. chars: (L,) ids. Returns (log-mel (T, 80), durations (L,)).
        `speed` scales every predicted duration: 1.2 = 20% faster, with no change in pitch.
        Flow model only: `steps` Euler steps (Matcha-TTS: 4 is nearly as good as 10), `temperature` scales the
        starting noise (lower = calmer and safer, higher = livelier), `guidance` > 1 exaggerates the style."""
        device = chars.device
        chars = chars[None]
        ids = lambda n, on: torch.tensor([n], device=device) if on else None
        spk, sty = ids(speaker, self.cfg.n_speakers), ids(style, self.cfg.n_styles)
        h, real = self.encode(chars, torch.tensor([chars.shape[1]], device=device), torch.tensor([lang], device=device), spk, sty)
        durations = (self.duration(h, real).exp() / speed).round().clamp(min=1).long()
        if not self.cfg.flow:
            return (self.decode(h, durations) * self.std + self.mean)[0], durations[0]
        frames = length_regulate(h, durations)
        mask = frame_mask(durations.sum(1), frames.shape[1])
        x = torch.randn(1, frames.shape[1], self.cfg.n_mels, device=device) * temperature
        for i in range(steps):
            t = torch.full((1,), i / steps, device=device)
            v = self.velocity(x, t, frames, mask, spk, sty)
            if guidance != 1.0 and sty is not None:
                v_none = self.velocity(x, t, frames, mask, spk, torch.zeros_like(sty))
                v = v_none + guidance * (v - v_none)
            x = x + v / steps
        return (x * self.std + self.mean)[0], durations[0]


def save(model: nn.Module, path, **extra) -> None:
    torch.save({"config": asdict(model.cfg), "model": model.state_dict(), **extra}, path)


def load(path, device: str | None = None) -> tuple[Acoustic, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = Acoustic(Config(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device or pick_device()).eval(), ckpt
