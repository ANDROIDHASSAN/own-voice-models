"""Lesson 4 - the model: a decoder-only transformer (a GPT), every layer written out.

What the whole thing does, in one sentence:
    given the tokens so far, output a probability for every possible next token.

The data flow for one sequence of T tokens (d = d_model):

    token ids (T)
      -> token embedding            (T, d)       each id becomes a learned vector
      -> N x transformer block      (T, d)       each block = attention + feed-forward, both with a residual
      -> final norm                 (T, d)
      -> unembedding (lm head)      (T, vocab)   logits: one score per vocabulary entry, at every position

Design choices here are the ones modern LLMs (Llama, Gemma, Mistral) use, not the 2017 originals:
    pre-norm + RMSNorm   instead of post-norm LayerNorm
    RoPE                 instead of learned / sinusoidal position embeddings
    SwiGLU               instead of ReLU / GELU feed-forward
    weight tying         input embedding and output head share one matrix
"""

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:  # these are hyperparameters: chosen by us, not learned
    vocab_size: int = 4096
    ctx: int = 256  # context window: the most tokens the model can look at in one go
    n_layer: int = 6  # depth: how many transformer blocks
    n_head: int = 6  # attention heads per block
    d_model: int = 384  # width: the size of every token's vector (the residual stream)
    dropout: float = 0.1
    # Two ideas from Karpathy's nanochat, off by default so they can be measured one at a time (see COMPARE.md).
    qk_norm: bool = False  # normalize queries and keys before the dot product
    zero_init: bool = False  # start every block as a no-op: its output projections begin at exactly zero


class RMSNorm(nn.Module):
    """Normalization keeps the numbers flowing through a deep network at a steady size, which keeps training
    stable. RMSNorm divides each vector by its root-mean-square and applies a learned gain. It is LayerNorm
    without the mean-subtraction and the bias: cheaper, and works just as well."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.gain


def rope_table(ctx: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention by itself has no idea of word order: shuffle the tokens and it computes the same thing.
    Position has to be injected. RoPE (rotary position embedding) does it by ROTATING each pair of numbers in
    the query and key vectors by an angle proportional to the token's position. Each pair rotates at a
    different frequency, like the hands of a clock. After rotation, the dot product of a query at position m
    and a key at position n depends only on (m - n): the model sees relative position."""
    freqs = 1.0 / (10000 ** (torch.arange(0, head_dim, 2) / head_dim))
    angles = torch.outer(torch.arange(ctx), freqs)  # (ctx, head_dim / 2)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., 0::2], x[..., 1::2]  # split the vector into (even, odd) pairs
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)  # 2D rotation of each pair


class CausalSelfAttention(nn.Module):
    """The heart of the transformer. Every token asks a question (query), every token offers a label (key)
    and some content (value). A token collects the values of the tokens whose key matches its query.

        score      = query . key / sqrt(head_dim)       scaled dot-product
        weights    = softmax(score)                     each row sums to 1
        output     = weights @ values

    "Self": queries, keys and values all come from the same sequence.
    "Causal": a token may only look at itself and earlier tokens, never the future. That is what makes the
              model autoregressive: it can be trained on every position at once and still generate left to right.
    "Multi-head": do this n_head times in parallel on smaller vectors, so different heads can track
                  different relations (who is the subject, which quote is open, ...)."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.n_head, self.head_dim = cfg.n_head, cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)  # one matrix makes Q, K and V together
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)  # mixes the heads back into one vector
        self.drop = nn.Dropout(cfg.dropout)
        self.qk_norm = cfg.qk_norm
        cos, sin = rope_table(cfg.ctx, self.head_dim)
        self.register_buffer("cos", cos, persistent=False)  # a buffer moves with the model but is not trained
        self.register_buffer("sin", sin, persistent=False)
        self.record = False  # xray.py sets this to keep the attention weights for inspection
        self.last_att: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, past: tuple | None = None) -> tuple[torch.Tensor, tuple]:
        B, T, C = x.shape  # batch, time (tokens), channels (d_model)
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)  # each (B, H, T, hd)

        P = past[0].shape[2] if past else 0  # tokens already processed and sitting in the KV cache
        q = apply_rope(q, self.cos[P : P + T], self.sin[P : P + T])
        k = apply_rope(k, self.cos[P : P + T], self.sin[P : P + T])
        if self.qk_norm:
            # QK norm: give every query and key the same length, so the score measures only DIRECTION
            # (cosine similarity). No single token can win attention just by having a huge vector.
            q, k = (t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6) for t in (q, k))
        if past:  # KV cache: reuse the keys and values of earlier tokens instead of recomputing them
            k, v = torch.cat((past[0], k), dim=2), torch.cat((past[1], v), dim=2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, T, P + T) attention scores
        # Causal mask: row i is new token i (absolute position P + i); it may see columns 0 .. P + i.
        mask = torch.ones(T, P + T, dtype=torch.bool, device=x.device).tril(diagonal=P)
        att = att.masked_fill(~mask, float("-inf"))  # -inf becomes exactly 0 after softmax
        att = F.softmax(att, dim=-1)
        if self.record:
            self.last_att = att.detach()
        y = self.drop(att) @ v  # (B, H, T, hd): the weighted mix of values
        y = y.transpose(1, 2).reshape(B, T, C)  # glue the heads back side by side
        return self.drop(self.proj(y)), (k, v)


class FeedForward(nn.Module):
    """Attention moves information BETWEEN tokens. The feed-forward layer then thinks about each token on its
    own: widen the vector, apply a non-linearity, narrow it back. Most of a transformer's parameters, and
    most of its stored "facts", live here.

    SwiGLU: one branch (gate) decides how much of the other branch (up) to let through.
    The hidden size is 2/3 of the usual 4x so the three matrices cost the same as the classic two."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        hidden = int(4 * cfg.d_model * 2 / 3)
        self.up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.gate = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))  # silu(x) = x * sigmoid(x), a.k.a. swish


class Block(nn.Module):
    """One transformer block. Note the shape of both lines:  x = x + something(norm(x))

    "x + ..." is the residual (skip) connection. The block does not replace x, it ADDS a correction to it.
    Gradients flow straight down the "+" path, which is why 6 or 60 layers can be trained at all.
    Normalizing BEFORE the sub-layer (pre-norm) keeps that straight path completely clean."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.norm1, self.attn = RMSNorm(cfg.d_model), CausalSelfAttention(cfg)
        self.norm2, self.mlp = RMSNorm(cfg.d_model), FeedForward(cfg)

    def forward(self, x: torch.Tensor, past: tuple | None = None) -> tuple[torch.Tensor, tuple]:
        a, present = self.attn(self.norm1(x), past)
        x = x + a
        return x + self.mlp(self.norm2(x)), present


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """The loss for next-token prediction, written out (it equals F.cross_entropy).

    log_softmax turns logits into log-probabilities. We pick the log-probability the model gave to the token
    that actually came next and negate it: confident and right -> near 0, confident and wrong -> large.
    exp(loss) is the perplexity: "the model is as unsure as if it were choosing between this many tokens"."""
    log_probs = logits - logits.logsumexp(dim=-1, keepdim=True)
    return -log_probs.gather(-1, targets.unsqueeze(-1)).mean()


class GPT(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)  # a lookup table: row i is the vector of token i
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying: "reading" and "writing" a token share one matrix

        # Initialization: small random numbers. Layers that write into the residual stream start smaller
        # still, so that at step 0 a deep stack does not blow the signal up.
        for name, p in self.named_parameters():
            if p.dim() == 2:
                last = name.endswith(("proj.weight", "down.weight"))
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer) if last else 0.02)
                if last and cfg.zero_init:
                    nn.init.zeros_(p)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, cache: list | None = None):
        """idx: (B, T) token ids. With `cache`, idx holds only the NEW tokens.
        Returns (logits, loss, new_cache)."""
        x = self.drop(self.tok_emb(idx))
        new_cache = []
        for i, block in enumerate(self.blocks):
            x, present = block(x, cache[i] if cache else None)
            new_cache.append(present)
        logits = self.lm_head(self.norm(x))
        # Teacher forcing: targets are the same tokens shifted left by one, so EVERY position is a training
        # example (predict token t+1 from tokens 0..t). The causal mask is what makes this honest.
        loss = cross_entropy(logits, targets) if targets is not None else None
        return logits, loss, new_cache


def pick_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def save(model: GPT, path, **extra) -> None:
    torch.save({"config": asdict(model.cfg), "model": model.state_dict(), **extra}, path)


def load(path, device: str | None = None) -> tuple[GPT, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)  # tensors only: never unpickle code
    model = GPT(Config(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device or pick_device()).eval(), ckpt
