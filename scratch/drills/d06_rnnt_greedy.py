"""Drill 6 - transducer (RNN-T) greedy decoding: the loop every streaming ASR runs. Theory only in this repo
(stt/GLOSSARY.md #89), so this drill uses toy networks passed in as functions.

    python d06_rnnt_greedy.py              # checks YOUR function below
    python d06_rnnt_greedy.py --solution   # checks solutions/d06_rnnt_greedy.py

A transducer has three parts:
  encoder     enc[t]                 one vector per audio frame (already computed: a list here)
  predictor   predictor(token, state) -> (g, state)   embeds the LAST emitted token; start with token=blank, state=None
  joiner      joiner(enc[t], g) -> list of scores over the vocabulary

Write rnnt_greedy(enc, predictor, joiner, blank=0, max_symbols=5) -> list of token ids:
  for each frame t: repeatedly take argmax of joiner(enc[t], g).
     blank  -> move to the next frame (g unchanged)
     token  -> append it, update g with predictor(token, state), stay on the SAME frame
  at most `max_symbols` tokens per frame, so a bad model cannot loop forever.
About 15 minutes.
"""

import sys
from pathlib import Path


def rnnt_greedy(enc, predictor, joiner, blank=0, max_symbols=5):
    raise NotImplementedError


def check(m) -> None:
    # Toy model: frame t "plans" some tokens; the predictor state is the history, the joiner emits the next
    # planned token for this frame, or blank once the plan is done.
    def run(plans, max_symbols=5):
        starts = [sum(len(p) for p in plans[:t]) for t in range(len(plans))]
        enc = list(zip(plans, starts))
        calls = {"pred": 0}

        def predictor(token, state):
            calls["pred"] += 1
            hist = () if state is None else state
            hist = hist if token == 0 else hist + (token,)
            return hist, hist

        def joiner(e, g):
            plan, start = e
            k = len(g) - start
            nxt = plan[k] if 0 <= k < len(plan) else 0
            return [1.0 if v == nxt else 0.0 for v in range(10)]

        return m.rnnt_greedy(enc, predictor, joiner, 0, max_symbols), calls["pred"]

    out, preds = run([[3], [], [4, 4, 5], [], [7]])
    assert out == [3, 4, 4, 5, 7], f"got {out}: several tokens may come out of ONE frame, and a repeat is a real repeat"
    assert preds == 1 + 5, f"predictor called {preds} times: once to start, then once per emitted token (not per frame)"
    assert run([[], [], []])[0] == [], "all-blank frames emit nothing"
    assert run([[1, 2, 3, 4, 5, 6, 7]], max_symbols=3)[0] == [1, 2, 3], "cap the symbols per frame"
    print("PASS  multi-token frames, real repeats (no CTC-style merge), predictor only on emission, max_symbols cap")


if __name__ == "__main__":
    if "--solution" in sys.argv:
        sys.path.insert(0, str(Path(__file__).parent / "solutions"))
        import d06_rnnt_greedy as mod
    else:
        mod = sys.modules[__name__]
    check(mod)
