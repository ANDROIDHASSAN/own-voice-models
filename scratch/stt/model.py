"""Lesson 5 - the model: an encoder that turns a spectrogram into per-frame character probabilities.

What the whole thing does, in one sentence:
    given the log-mel frames of a clip, output for every 40 ms a probability for each character (and blank).

The data flow for one clip of T frames (d = d_model):

    log-mel (T, 80)                  100 frames / second
      -> normalize each band         mean 0, std 1 (statistics measured on the training set)
      -> SpecAugment                 training only: hide random stripes so the model cannot rely on any one cue
      -> conv subsampling            (T / 4, d)   25 frames / second: 4x fewer positions for attention
      -> N x block                   (T / 4, d)   attention + convolution + feed-forward, each with a residual
      -> final norm -> linear        (T / 4, vocab)
      -> log-softmax                 log-probabilities for CTC (lesson 4)

This is ENCODER-ONLY. The LLM in ../llm was decoder-only: causal, one token at a time. Here every frame may
look at the whole clip, past and future, and all frames are predicted in one pass. There is no decoder and
no loop: CTC turns the frame outputs into text.

The block is a Conformer block (Gulati 2020) in the simplified layout of Squeezeformer (Kim 2022):

    attention      which other moments of the clip matter for this frame?      global context
    convolution    what do the neighbouring frames look like?                  local patterns (a consonant is ~3 frames)
    feed-forward   think about this frame alone

Speech needs both: a phoneme is a local shape, a word's identity depends on context seconds away.
RMSNorm, RoPE and SwiGLU are the same pieces as in ../llm/model.py; read them there first.
"""

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 112  # characters + blank
    n_mels: int = 80
    d_model: int = 192
    n_layer: int = 8
    n_head: int = 4
    conv_kernel: int = 15  # frames the convolution sees at once: 15 x 40 ms = 0.6 s
    dropout: float = 0.1
    subsample: int = 4  # time reduction of the front-end: 4 -> 25 output frames / second, 2 -> 50
    max_frames: int = 3000  # longest clip after subsampling: 60 s at 50 frames / second


class RMSNorm(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.gain


def rope_table(length: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (10000 ** (torch.arange(0, head_dim, 2) / head_dim))
    angles = torch.outer(torch.arange(length), freqs)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


def spec_augment(x: torch.Tensor, lens: torch.Tensor, freq_masks: int = 2, freq_width: int = 15,
                 time_masks: int = 2, time_ratio: float = 0.05) -> torch.Tensor:
    """SpecAugment (Park 2019): the standard data augmentation for speech. On every training clip, blank out
    a few random horizontal stripes (frequency bands) and vertical stripes (moments in time).

    The model can no longer lean on one band or one frame, so it has to use the rest: the same idea as
    dropout, applied to the input picture. With little data this is the difference between memorizing
    the training clips and learning sounds. x: (B, T, n_mels), already normalized, so 0 means "average"."""
    x = x.clone()
    for b in range(x.shape[0]):
        for _ in range(freq_masks):
            width = int(torch.randint(0, freq_width + 1, ()))
            start = int(torch.randint(0, x.shape[2] - width + 1, ()))
            x[b, :, start : start + width] = 0.0
        n = int(lens[b])
        for _ in range(time_masks):
            width = int(torch.randint(0, int(n * time_ratio) + 1, ()))
            start = int(torch.randint(0, n - width + 1, ()))
            x[b, start : start + width] = 0.0
    return x


class Subsample(nn.Module):
    """100 frames a second is far more than text needs: a character lasts 60 ms or more. Two stride-2
    convolutions shrink time (and frequency) by 4, which makes attention 16x cheaper, and each output
    position already summarizes a small time-frequency patch of the spectrogram."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.time_strides = (2, cfg.subsample // 2)  # frequency is always halved twice; time once or twice
        self.conv2 = nn.Conv2d(32, 32, 3, stride=(cfg.subsample // 2, 2), padding=1)
        self.out = nn.Linear(32 * math.ceil(cfg.n_mels / 4), cfg.d_model)

    @staticmethod
    def out_len(n: torch.Tensor, stride: int = 2) -> torch.Tensor:
        return (n - 1) // stride + 1  # length after a convolution with kernel 3, padding 1 and this stride

    def forward(self, x: torch.Tensor, lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.silu(self.conv1(x[:, None]))  # (B, 32, T / 2, 40): treat the spectrogram as a 1-channel image
        lens = self.out_len(lens)
        x = x * frame_mask(lens, x.shape[2])[:, None, :, None]  # padding must stay silent: see Encoder.forward
        x = F.silu(self.conv2(x))  # (B, 32, T / 4, 20)
        lens = self.out_len(lens, self.time_strides[1])
        return self.out(x.permute(0, 2, 1, 3).flatten(2)), lens  # (B, T / 4, 32 * 20) -> (B, T / 4, d)


def frame_mask(lens: torch.Tensor, T: int) -> torch.Tensor:
    """(B, T) bool: True where a frame is real audio, False where it is padding added to fill the batch."""
    return torch.arange(T, device=lens.device)[None, :] < lens[:, None]


class SelfAttention(nn.Module):
    """The same scaled dot-product attention as the LLM, with two differences:
    no causal mask (a frame may look at the future: the whole clip is already recorded), and a padding mask
    (no frame may look at padding)."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.n_head, self.head_dim = cfg.n_head, cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        cos, sin = rope_table(cfg.max_frames, self.head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, self.cos[:T], self.sin[:T]), apply_rope(k, self.cos[:T], self.sin[:T])
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, T, T)
        att = att.masked_fill(~real[:, None, None, :], float("-inf"))  # padding mask: no attention TO padded frames
        y = self.drop(F.softmax(att, dim=-1)) @ v
        return self.drop(self.proj(y.transpose(1, 2).reshape(B, T, C)))


class ConvModule(nn.Module):
    """The part a plain transformer lacks. A depthwise convolution slides a small window along time and, for
    each of the d channels separately, mixes the 15 neighbouring frames. "Depthwise" means the channels are
    not mixed here (the linear layers around it do that), so it costs 15 x d weights instead of 15 x d x d.

    The gate in front (GLU) lets the module decide, per frame and channel, how much to let through."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.expand = nn.Linear(cfg.d_model, 2 * cfg.d_model)
        self.depthwise = nn.Conv1d(cfg.d_model, cfg.d_model, cfg.conv_kernel, padding=cfg.conv_kernel // 2, groups=cfg.d_model)
        self.norm = RMSNorm(cfg.d_model)
        self.project = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        value, gate = self.expand(x).chunk(2, dim=-1)
        y = (value * torch.sigmoid(gate)) * real[..., None]  # zero the padding so the window does not read it
        y = self.depthwise(y.transpose(1, 2)).transpose(1, 2)  # Conv1d wants (B, channels, T)
        return self.drop(self.project(F.silu(self.norm(y))))


class FeedForward(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        hidden = int(4 * cfg.d_model * 2 / 3)
        self.up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.gate = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.norm1, self.attn = RMSNorm(cfg.d_model), SelfAttention(cfg)
        self.norm2, self.conv = RMSNorm(cfg.d_model), ConvModule(cfg)
        self.norm3, self.mlp = RMSNorm(cfg.d_model), FeedForward(cfg)

    def forward(self, x: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), real)
        x = x + self.conv(self.norm2(x), real)
        return x + self.mlp(self.norm3(x))


class Encoder(nn.Module):
    def __init__(self, cfg: Config, mean: list[float] | None = None, std: list[float] | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        # Saved inside the checkpoint, so the model always normalizes the way it was trained.
        self.register_buffer("mean", torch.tensor(mean if mean is not None else [0.0] * cfg.n_mels))
        self.register_buffer("std", torch.tensor(std if std is not None else [1.0] * cfg.n_mels))
        self.subsample = Subsample(cfg)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

        for name, p in self.named_parameters():
            if p.dim() == 2:  # the linear layers; convolutions keep PyTorch's default initialization
                last = name.endswith(("proj.weight", "down.weight", "project.weight"))
                nn.init.normal_(p, std=0.02 / math.sqrt(3 * cfg.n_layer) if last else 0.02)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, mel: torch.Tensor, lens: torch.Tensor, augment: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """mel: (B, T, 80) raw log-mel, padded to the longest clip. lens: (B,) real frame counts.
        Returns (log_probs (B, T / 4, vocab), output lengths (B,)).

        Clips in a batch have different lengths, so the short ones are padded. A clip must get the SAME
        output whether it is alone or padded next to a longer one. That needs care in three places: padded
        input frames are set to 0, the convolutions never read padded positions, and attention never looks
        at them. check.py tests exactly this."""
        x = (mel - self.mean) / self.std
        if augment:
            x = spec_augment(x, lens)
        x = x * frame_mask(lens, x.shape[1])[..., None]
        x, lens = self.subsample(x, lens)
        real = frame_mask(lens, x.shape[1])
        x = self.drop(x)
        for block in self.blocks:
            x = block(x, real)
        return F.log_softmax(self.head(self.norm(x)), dim=-1), lens


def pick_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


class Stepper:
    """One optimizer step, in fp16 mixed precision on CUDA (a T4 is ~8x faster in fp16) and plain fp32 elsewhere.
    Mixed precision: matrix multiplies run in fp16, while softmax, log-softmax, sums and the losses stay fp32
    (autocast decides per op). Small fp16 gradients would round to zero, so the loss is scaled up before
    backward and the gradients scaled back down before clipping (GradScaler); a step whose gradients
    overflowed is skipped and the scale lowered."""

    def __init__(self, device: str) -> None:
        self.on = device == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.on)

    def autocast(self):
        return torch.autocast("cuda", dtype=torch.float16, enabled=self.on)

    def __call__(self, loss: torch.Tensor, opt: torch.optim.Optimizer, params, clip: float) -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(opt)  # true gradients, so the clipping threshold means what it says
        grad_norm = torch.nn.utils.clip_grad_norm_(params, clip)
        self.scaler.step(opt)
        self.scaler.update()
        return grad_norm


def save(model: Encoder, path, **extra) -> None:
    torch.save({"config": asdict(model.cfg), "model": model.state_dict(), **extra}, path)


def load(path, device: str | None = None) -> tuple[Encoder, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)  # tensors only: never unpickle code
    model = Encoder(Config(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device or pick_device()).eval(), ckpt
