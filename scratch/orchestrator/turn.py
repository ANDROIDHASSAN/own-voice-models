"""Lesson 2 - turn detection: has the person finished, or are they just thinking?

    python turn.py

The VAD says "silence for 500 ms". That is not the same as "your turn":

    "mera order number hai ... [600 ms] ... chaar paanch do"       a pause in the MIDDLE of a turn
    "haan theek hai."  [600 ms]                                     the END of a turn

Answer after every pause and the agent interrupts people mid-sentence (a false cutoff). Wait 2 seconds after
every pause and it feels dead. This one decision is most of what makes a voice agent feel good or bad, and
it is why LiveKit, Pipecat and others train dedicated end-of-turn models and publish benchmarks for them.

Two signals, combined the way LiveKit's turn detector does it:

    acoustic   how long has it been silent?                              (from the VAD)
    semantic   does what was said so far LOOK finished?                  (from a language model)

The semantic signal here comes from our own LLM (../llm). It was trained on chats laid out as
"message, newline, reply". So after the partial transcript we ask it one question:

    P(next token is "\n" or . ? !)  =  "how likely is it that this message is complete?"

"mera order number hai" -> the model expects more words, P is low.  "haan theek hai" -> P is high.

The endpointer turns that into a waiting time:

    looks finished     answer after  min_delay  of silence   (fast)
    looks unfinished   wait up to    max_delay               (patient)

The metric to judge it by (LiveKit's eot-bench): false cutoffs at a given latency budget.
"""

from dataclasses import dataclass
from typing import Callable


@dataclass
class Endpointer:
    """Decides how much silence to wait for after an utterance, given how finished it looks."""

    completeness: Callable[[str], float] | None = None  # text -> probability the turn is complete; None = acoustic only
    min_delay: float = 0.3  # seconds of silence before answering a turn that looks finished
    max_delay: float = 1.5  # ... and one that does not
    threshold: float = 0.5

    def delay(self, transcript: str) -> float:
        """Seconds of silence to wait before answering. The threshold does not decide WHETHER to answer
        (the agent always does eventually), only how patient to be: a wrong guess costs a second, not a turn."""
        if self.completeness is None:
            return self.min_delay
        return self.min_delay if self.completeness(transcript) >= self.threshold else self.max_delay


def lm_completeness(model, tok, lang: str = "hing") -> Callable[[str], float]:
    """Builds the semantic signal from the scratch LLM: P("\\n" comes next | <bos><lang> text so far)."""
    import torch

    # A recognizer writes no punctuation, but the LLM learned from punctuated text, where a message ends
    # "... hai.\n". So "the next token closes the sentence" counts as finished too.
    enders = [tok.encode(s)[0] for s in ("\n", ".", "?", "!")]
    device = next(model.parameters()).device

    @torch.no_grad()
    def completeness(text: str) -> float:
        ids = [tok.special["<|bos|>"], tok.special[f"<|{lang}|>"], *tok.encode(text.strip())][-model.cfg.ctx :]
        logits, _, _ = model(torch.tensor([ids], device=device))
        return torch.softmax(logits[0, -1], dim=-1)[enders].sum().item()

    return completeness


def false_cutoffs(endpointer: Endpointer, cases: list[tuple[str, bool, float]]) -> tuple[float, float]:
    """cases: (text so far, is this really the end of the turn?, seconds of silence that actually followed).
    Returns (false-cutoff rate on mid-turn pauses, mean response latency on real turn ends)."""
    holds = [(t, s) for t, end, s in cases if not end]
    ends = [t for t, end, _ in cases if end]
    cut = sum(endpointer.delay(t) < silence for t, silence in holds) / max(1, len(holds))  # answered before they resumed
    return cut, sum(endpointer.delay(t) for t in ends) / max(1, len(ends))


def demo() -> None:
    import sys
    from pathlib import Path

    llm = Path(__file__).parent.parent / "llm"
    sys.path.insert(0, str(llm))
    from model import load  # noqa: E402
    from tokenizer import BPE  # noqa: E402

    model, _ = load(llm / "runs/ckpt.pt", "cpu")
    completeness = lm_completeness(model, BPE.load(llm / "data/tokenizer.json"))

    # (what was said so far, is it really the end, how long the pause lasted before they went on)
    cases = [("mera order number hai", False, 0.8), ("haan theek hai", True, 0.0),
             ("kal main office", False, 0.6), ("kal main office nahi aaunga", True, 0.0),
             ("mujhe lagta hai ki", False, 1.0), ("tum kahan ho?", True, 0.0),
             ("agar tum free ho to", False, 0.7), ("chalo phir milte hain", True, 0.0),
             ("main soch raha tha ki hum", False, 0.9), ("kya kar rahe ho aaj?", True, 0.0)]
    print("P(complete)  truth       text")
    for text, end, _ in cases:
        print(f"   {completeness(text):.3f}     {'END ' if end else 'hold'}        {text}")

    print("\nendpointer                      false cutoffs   mean latency on real ends")
    for name, ep in [("VAD only, 0.3 s", Endpointer(None, 0.3)), ("VAD only, 1.5 s", Endpointer(None, 1.5)),
                     ("VAD + our LLM, 0.3 / 1.5 s", Endpointer(completeness))]:
        cut, latency = false_cutoffs(ep, cases)
        print(f"{name:30s}  {cut:6.0%}          {latency:.2f} s")
    print("\n(10 hand-written cases: an illustration of the trade-off, not a benchmark.)")


if __name__ == "__main__":
    demo()
