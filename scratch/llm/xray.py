"""Lesson 9 - x-ray: look inside the model while it reads a prompt.

    python xray.py --lang hi --prompt "एक दिन राजू ने अपनी माँ से कहा"

Shows three things:
    1. how the tokenizer cut the prompt
    2. the model's probability for the next token (its "belief")
    3. per layer, which earlier tokens the last position is attending to

A model you can inspect is a model you can explain. This is the glass-box view of everything in model.py.
"""

import argparse
from pathlib import Path

import torch

from model import load
from tokenizer import BPE

HERE = Path(__file__).parent


def show(tok: BPE, token_id: int) -> str:
    return repr(tok.piece(token_id))[1:-1]  # repr makes spaces and newlines visible


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="एक दिन राजू ने अपनी माँ से कहा")
    ap.add_argument("--lang", default="hi", choices=["hi", "mr", "hing"])
    ap.add_argument("--ckpt", default=str(HERE / "runs/ckpt.pt"))
    args = ap.parse_args()
    model, _ = load(args.ckpt, "cpu")
    tok = BPE.load(HERE / "data/tokenizer.json")
    ids = tok.encode(f"<|bos|><|{args.lang}|>{args.prompt}")

    print(f"1. TOKENS  {len(args.prompt.encode())} bytes -> {len(ids)} tokens")
    print("   " + " | ".join(show(tok, i) for i in ids))

    for block in model.blocks:
        block.attn.record = True
    logits, _, _ = model(torch.tensor([ids]))
    probs = torch.softmax(logits[0, -1], dim=-1)
    # Entropy: how spread out the belief is. 0 bits = certain; log2(vocab) = no idea.
    entropy = -(probs * torch.log2(probs.clamp(min=1e-12))).sum().item()
    print(f"\n2. NEXT TOKEN  (entropy {entropy:.2f} bits of a possible {torch.log2(torch.tensor(float(len(probs)))).item():.1f})")
    for p, i in zip(*probs.topk(8)):
        print(f"   {p.item():6.1%} {'#' * int(p.item() * 40):40s} {show(tok, int(i))}")

    print("\n3. ATTENTION of the last position (averaged over heads), top 4 tokens per layer")
    for n, block in enumerate(model.blocks):
        att = block.attn.last_att[0, :, -1].mean(0)  # (heads, T) -> (T)
        top = ", ".join(f"{show(tok, ids[int(i)])}@{int(i)} {w.item():.0%}" for w, i in zip(*att.topk(min(4, len(ids)))))
        print(f"   layer {n}: {top}")


if __name__ == "__main__":
    main()
