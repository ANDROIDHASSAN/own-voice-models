"""Lesson 6 - generation: turning "a probability for every next token" into text.

    python generate.py --lang hi --prompt "एक बार की बात है"
    python generate.py --lang hing --prompt "kya chal raha hai bhai?" --chat

The model only ever answers one question: what comes next? Text comes from asking it again and again,
feeding each answer back in. That loop is autoregressive decoding. HOW we pick from the probabilities is the
sampling strategy, and it changes the output a lot:

    greedy            always the most likely token. Safe, repetitive, often dull.
    temperature       divide the logits by T before softmax. T < 1 sharpens (safer), T > 1 flattens (wilder).
    top-k             keep only the k most likely tokens, drop the rest.
    top-p (nucleus)   keep the smallest set of tokens whose probabilities add up to p. Adapts to how sure
                      the model is: few candidates when confident, many when not.
    repetition penalty  push down tokens that were already used, so it does not loop.
"""

import argparse
import time
from pathlib import Path

import torch

from model import GPT, load
from tokenizer import BPE

HERE = Path(__file__).parent


@torch.no_grad()  # no gradients needed at inference: saves memory and time
def stream(model: GPT, ids: list[int], max_new: int = 120, temperature: float = 0.8, top_k: int = 50,
           top_p: float = 0.95, repetition_penalty: float = 1.1, stop: int | None = None,
           use_cache: bool = True):
    """Yields the newly generated token ids one at a time, as they are sampled. A voice agent starts
    speaking the first sentence while the rest is still being generated (../orchestrator)."""
    device = next(model.parameters()).device
    out: list[int] = []
    cache = None
    x = torch.tensor([ids], device=device)
    # ponytail: stops at the context window instead of sliding it. Enough for short voice replies;
    # sliding needs the KV cache rebuilt because every position (and so every RoPE angle) shifts.
    for _ in range(min(max_new, model.cfg.ctx - len(ids))):
        if use_cache:
            # With the KV cache the model only processes the newest token; earlier keys/values are reused.
            # Without it, every step re-reads the whole sequence: O(T^2) work per token instead of O(T).
            logits, _, cache = model(x, cache=cache)
        else:
            logits, _, _ = model(torch.tensor([ids + out], device=device))
        logits = logits[0, -1]  # we only need the prediction after the LAST token

        if repetition_penalty != 1.0 and (seen := list(set(ids + out))):
            s = logits[seen]
            logits[seen] = torch.where(s > 0, s / repetition_penalty, s * repetition_penalty)
        if temperature == 0:
            nxt = int(logits.argmax())  # greedy
        else:
            logits = logits / temperature
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
                logits[logits < kth] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            if top_p < 1.0:
                sorted_p, order = probs.sort(descending=True)
                drop = sorted_p.cumsum(0) - sorted_p > top_p  # tokens that start after the mass already reached p
                probs[order[drop]] = 0
                probs = probs / probs.sum()
            nxt = int(torch.multinomial(probs, 1))  # roll the weighted dice
        if nxt == stop:
            break
        out.append(nxt)
        yield nxt
        x = torch.tensor([[nxt]], device=device)


def sample(model: GPT, ids: list[int], *args, **kw) -> list[int]:
    """Returns only the newly generated token ids."""
    return list(stream(model, ids, *args, **kw))


def reply(model: GPT, tok: BPE, prompt: str, lang: str = "hing", chat: bool = False, **kw) -> str:
    """Builds the same layout the training documents had: <|bos|><|lang|>text. In chat mode the prompt is
    the first line and the model writes the second, exactly like the Hinglish question/answer pairs."""
    ids = [tok.special["<|bos|>"], tok.special[f"<|{lang}|>"], *tok.encode(prompt + ("\n" if chat else ""))]
    new = sample(model, ids, stop=tok.special["<|eos|>"], **kw)
    return tok.decode(new) if chat else prompt + tok.decode(new)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="")
    ap.add_argument("--lang", default="hi", choices=["hi", "mr", "hing"])
    ap.add_argument("--chat", action="store_true", help="treat the prompt as a message and generate the answer")
    ap.add_argument("--ckpt", default=str(HERE / "runs/ckpt.pt"))
    ap.add_argument("--max-new", type=int, default=150)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    args = ap.parse_args()
    model, _ = load(args.ckpt)
    tok = BPE.load(HERE / "data/tokenizer.json")
    t = time.perf_counter()
    text = reply(model, tok, args.prompt, args.lang, args.chat, max_new=args.max_new,
                 temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
    dt = time.perf_counter() - t
    print(text)
    print(f"\n[{len(tok.encode(text))} tokens in {dt:.2f} s]")


if __name__ == "__main__":
    main()
