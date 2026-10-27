"""Lesson 5 - training: the loop that turns random weights into a language model.

    python train.py                    # full run
    python train.py --iters 200        # quick smoke run
    python train.py --resume           # continue from runs/ckpt.pt

This is self-supervised learning: nobody labelled anything. The label for each position is simply the next
token of the text itself. Every step:

    batch -> forward -> loss -> backward (gradients) -> clip -> optimizer step -> next learning rate

Watch two numbers. Train loss falling means the model is learning the training text. Val loss (on text it
never trained on) falling too means it is learning the LANGUAGE. If train keeps falling while val rises,
the model has started memorizing: overfitting.
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from generate import reply
from model import GPT, Config, load, pick_device, save
from tokenizer import BPE

HERE = Path(__file__).parent
DATA, RUNS = HERE / "data", HERE / "runs"


def get_batch(data: np.ndarray, batch: int, ctx: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """`batch` random windows of `ctx` tokens. y is x shifted by one: y[i] is the token that follows x[i]."""
    starts = np.random.randint(0, len(data) - ctx - 1, batch)
    x = np.stack([data[s : s + ctx] for s in starts]).astype(np.int64)
    y = np.stack([data[s + 1 : s + ctx + 1] for s in starts]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


@torch.no_grad()
def estimate_loss(model: GPT, data: np.ndarray, batch: int, device: str, n: int = 20) -> float:
    """One batch is noisy, so average a few. eval() switches dropout off for the measurement."""
    model.eval()
    loss = sum(model(*get_batch(data, batch, model.cfg.ctx, device))[1].item() for _ in range(n)) / n
    model.train()
    return loss


def lr_at(it: int, base: float, warmup: int, total: int) -> float:
    """Learning-rate schedule. Warmup: start near zero and ramp up, because the first gradients of a random
    network are wild. Then cosine decay: glide down to 10% so the model settles instead of bouncing."""
    if it < warmup:
        return base * (it + 1) / warmup
    progress = (it - warmup) / max(1, total - warmup)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--name", default="ckpt", help="checkpoint name in runs/, so experiments do not overwrite each other")
    ap.add_argument("--qk-norm", action="store_true")
    ap.add_argument("--zero-init", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = pick_device()
    RUNS.mkdir(exist_ok=True)
    tok = BPE.load(DATA / "tokenizer.json")
    train = np.memmap(DATA / "train.bin", dtype=np.uint16, mode="r")  # memmap: read from disk on demand
    val = np.memmap(DATA / "val.bin", dtype=np.uint16, mode="r")

    start, best = 0, float("inf")
    if args.resume:
        model, ckpt = load(RUNS / f"{args.name}.pt", device)
        start, best = ckpt["iter"], ckpt["val_loss"]
    else:
        model = GPT(Config(vocab_size=tok.vocab_size, qk_norm=args.qk_norm, zero_init=args.zero_init)).to(device)
    model.train()
    tokens_per_iter = args.batch * model.cfg.ctx
    print(f"{device}  {model.n_params() / 1e6:.1f} M parameters  {len(train) / 1e6:.1f} M train tokens  "
          f"{args.iters * tokens_per_iter / len(train):.1f} epochs")
    # A random model spreads its guess evenly over the vocabulary, so the first loss should be near ln(vocab).
    print(f"expected starting loss ~ ln({tok.vocab_size}) = {math.log(tok.vocab_size):.2f}")

    # AdamW: Adam keeps a running average of each weight's gradient (momentum) and of its squared gradient
    # (so every weight gets its own step size). The "W" is weight decay: gently pull weights towards zero,
    # a regularizer. Decay only the matrices; gains and other 1-D parameters are left alone.
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95))

    t0 = time.perf_counter()
    for it in range(start, args.iters):
        for group in opt.param_groups:
            group["lr"] = lr_at(it, args.lr, args.warmup, args.iters)
        x, y = get_batch(train, args.batch, model.cfg.ctx, device)
        _, loss, _ = model(x, y)  # forward pass
        opt.zero_grad(set_to_none=True)  # clear the old gradients
        loss.backward()  # backward pass: a gradient for every weight (lesson 1, on tensors)
        # Gradient clipping: if the gradient is huge, scale it down to length 1. One bad batch cannot wreck the model.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()  # update the weights

        if it % 50 == 0:
            speed = (it - start + 1) * tokens_per_iter / (time.perf_counter() - t0)
            print(f"iter {it:5d}  loss {loss.item():.3f}  grad-norm {grad_norm:.2f}  lr {opt.param_groups[0]['lr']:.2e}  "
                  f"{speed / 1000:.1f}k tok/s", flush=True)
        if (it + 1) % args.eval_every == 0 or it + 1 == args.iters:
            tr, va = estimate_loss(model, train, args.batch, device), estimate_loss(model, val, args.batch, device)
            print(f"== iter {it + 1}  train {tr:.3f}  val {va:.3f}  val perplexity {math.exp(va):.1f}")
            if va < best:  # keep the checkpoint that is best on unseen text (a simple form of early stopping)
                best = va
                save(model, RUNS / f"{args.name}.pt", iter=it + 1, val_loss=va)
            model.eval()
            print("   hi  :", reply(model, tok, "एक दिन", "hi", max_new=40).replace("\n", " "))
            print("   hing:", reply(model, tok, "kya kar raha hai?", "hing", chat=True, max_new=30), flush=True)
            model.train()


if __name__ == "__main__":
    main()
