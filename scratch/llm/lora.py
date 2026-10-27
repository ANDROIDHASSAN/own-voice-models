"""Lesson 7 - fine-tuning with LoRA: teach a trained model something new by training under 1% of it.

    python lora.py          # fine-tunes runs/ckpt.pt on persona.txt, writes runs/ckpt_persona.pt

Pre-training (train.py) teaches the language. Fine-tuning takes that model and adapts it to a task, a domain
or a personality with far less data. Full fine-tuning updates every weight. LoRA (low-rank adaptation) does
not touch the original weights at all:

    original layer:   y = W x                    W is frozen
    with LoRA:        y = W x + (alpha / r) * B A x

A is (r x in) and B is (out x r) with a small rank r, so B A is a full-size update built from very few
numbers. The bet is that the CHANGE a task needs is simple (low rank) even though W itself is not.
B starts at zero, so at step 0 the model behaves exactly as before.

After training, B A can be added into W (merging): the model is the same size and speed as before.
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from generate import reply
from model import GPT, load, save
from tokenizer import BPE
from train import estimate_loss, get_batch

HERE = Path(__file__).parent


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16) -> None:
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)  # small random
        self.B = nn.Parameter(torch.zeros(base.out_features, r))  # zero: the adapter starts as a no-op
        self.scale = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

    def merge(self) -> nn.Linear:
        with torch.no_grad():
            self.base.weight += self.scale * (self.B @ self.A)
        return self.base


def add_lora(model: GPT, r: int = 8, alpha: int = 16) -> list[nn.Parameter]:
    """Freeze the whole model, then wrap the attention matrices. Returns the only trainable parameters."""
    for p in model.parameters():
        p.requires_grad_(False)
    for block in model.blocks:
        block.attn.qkv = LoRALinear(block.attn.qkv, r, alpha)
        block.attn.proj = LoRALinear(block.attn.proj, r, alpha)
    return [p for p in model.parameters() if p.requires_grad]


def merge_lora(model: GPT) -> None:
    for block in model.blocks:
        block.attn.qkv, block.attn.proj = block.attn.qkv.merge(), block.attn.proj.merge()


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    tok = BPE.load(HERE / "data/tokenizer.json")
    model, _ = load(HERE / "runs/ckpt.pt")
    device = str(next(model.parameters()).device)
    val = np.memmap(HERE / "data/val.bin", dtype=np.uint16, mode="r")
    train = np.memmap(HERE / "data/train.bin", dtype=np.uint16, mode="r")

    pairs = [p.strip() for p in (HERE / "persona.txt").read_text().split("\n\n") if p.strip()]
    docs = [tok.encode(f"<|bos|><|hing|>{p}<|eos|>") for p in pairs]
    width = max(map(len, docs))
    # Padding: documents differ in length, a batch must be a rectangle. Pad with <|eos|> and tell the loss to
    # ignore those positions (target -100 is skipped below).
    eos = tok.special["<|eos|>"]
    x = torch.tensor([d[:-1] + [eos] * (width - len(d)) for d in docs], device=device)
    y = torch.tensor([d[1:] + [-100] * (width - len(d)) for d in docs], device=device)

    def persona_loss() -> torch.Tensor:
        logits, _, _ = model(x)
        keep = y != -100
        return nn.functional.cross_entropy(logits[keep], y[keep])

    question = "tera naam kya hai?"
    total = model.n_params()
    with torch.no_grad():
        print(f"before  persona loss {persona_loss().item():.3f}  general val loss {estimate_loss(model, val, 32, device):.3f}")
    print(f"        {question} -> {reply(model, tok, question, chat=True, temperature=0, max_new=30)}")

    trainable = add_lora(model)
    model.to(device)  # the new A and B matrices are created on the CPU
    n = sum(p.numel() for p in trainable)
    print(f"LoRA: training {n:,} of {total:,} parameters ({100 * n / total:.2f}%)")
    opt = torch.optim.AdamW(trainable, lr=2e-3, weight_decay=0.0)
    model.train()
    for step in range(600):
        # Replay (rehearsal): add a batch of the ORIGINAL data to every step, so the model keeps what it knew
        # while it learns the new thing. Without it the general val loss climbs: catastrophic forgetting.
        loss = persona_loss() + model(*get_batch(train, 16, model.cfg.ctx, device))[1]
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"step {step:3d}  loss {loss.item():.3f}")

    adapter = {k: v for k, v in model.state_dict().items() if k.endswith((".A", ".B"))}
    torch.save(adapter, HERE / "runs/persona_adapter.pt")  # the adapter alone: this is all you need to ship
    merge_lora(model)
    model.eval()
    save(model, HERE / "runs/ckpt_persona.pt")
    with torch.no_grad():
        # If the general val loss went up, the model forgot some of what it knew: catastrophic forgetting.
        print(f"after   persona loss {persona_loss().item():.3f}  general val loss {estimate_loss(model, val, 32, device):.3f}")
    print(f"        {question} -> {reply(model, tok, question, chat=True, temperature=0, max_new=30)}")
    size = lambda f: (HERE / "runs" / f).stat().st_size / 1e6
    print(f"adapter {size('persona_adapter.pt'):.2f} MB   full model {size('ckpt_persona.pt'):.1f} MB")
    assert math.isfinite(loss.item())


if __name__ == "__main__":
    main()
