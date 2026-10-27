"""Checks for the orchestration logic. The three models are replaced by fakes with known delays, so the
state machine, barge-in and cancellation are tested on their own, in a few seconds, with no checkpoint.

    python check.py
"""

import asyncio
import re
import time
from pathlib import Path

import numpy as np

from models import romanize, sentences
from pipeline import Agent
from turn import Endpointer, false_cutoffs
from vad import FRAME, SR, VAD, frames

HERE = Path(__file__).parent
rng = np.random.default_rng(0)
noise = lambda seconds: (0.003 * rng.standard_normal(int(seconds * SR))).astype(np.float32)
voice = lambda seconds: (noise(seconds) + 0.2 * np.sin(2 * np.pi * 180 * np.arange(int(seconds * SR)) / SR)).astype(np.float32)

# VAD: one utterance in, exactly one start and one end; a short pause does not split it; the first syllable is kept.
vad, found = VAD(), []
for i, f in enumerate(frames(np.concatenate((noise(1), voice(1), noise(0.2), voice(0.8), noise(1.5))))):
    if e := vad.feed(f):
        found.append((round(i * 0.02, 2), e))
assert [e.kind for _, e in found] == ["start", "end"]
assert 1.0 <= found[0][0] <= 1.2 and abs(len(found[1][1].audio) / SR - (2.0 + 0.5 + 0.3)) < 0.15, "utterance = speech + hangover + pre-roll"
assert not any(VAD().feed(f) for f in frames(noise(3))), "steady room noise must never trigger"
assert not any(VAD().feed(f) for f in frames(30 * noise(3))), "a loud room (a fan at -40 dB) must not trigger either"
fan_on = VAD()  # a quiet room, then a fan is switched on and stays on: at most one false utterance while the floor adapts
assert sum(bool(fan_on.feed(f)) for f in frames(np.concatenate((noise(2), 30 * noise(8))))) <= 2 and not fan_on.speaking

# Endpointer: finished-looking text is answered fast, unfinished text is given time.
ep = Endpointer(lambda text: 0.9 if text.endswith("hai") else 0.01, min_delay=0.3, max_delay=1.5)
assert ep.delay("theek hai") == 0.3 and ep.delay("mujhe lagta hai ki") == 1.5 and Endpointer().delay("x") == 0.3
assert false_cutoffs(ep, [("lagta hai ki", False, 0.8), ("theek hai", True, 0.0)]) == (0.0, 0.3)
assert false_cutoffs(Endpointer(None, 0.3), [("lagta hai ki", False, 0.8)])[0] == 1.0

# Sentence chunker: sentences come out as soon as they close, short ones are merged, nothing is lost.
pieces = ["Haan", " bhai.", " Sab", " theek hai", "? Tum", " batao", "।", " accha"]
assert list(sentences(iter(pieces))) == ["Haan bhai. Sab theek hai?", "Tum batao। accha"]
first = next(sentences(iter(["Ek lamba sa vaakya yahan khatm.", " baaki"])))
assert first == "Ek lamba sa vaakya yahan khatm."

# Romanizer: matras, halant, and the dropped final "a".
assert romanize("कमल नमस्ते क्या हाल है") == "kamal namaste kyaa haal hai" and romanize("ok 5") == "ok 5"
assert romanize("सड़क ज़रूर") == "sarak zaroor", "a nukta consonant keeps its vowel"


# Pipeline with fake models: STT 50 ms, LLM streams 3 sentences 100 ms apart, TTS 50 ms per sentence, 1 s of audio each.
def make(endpointer: Endpointer):
    log, played = [], []

    def llm(text):
        for sentence in ("Pehla vaakya yahan hai. ", "Doosra vaakya yahan hai. ", "Teesra vaakya yahan hai."):
            time.sleep(0.1)
            yield sentence

    stt = lambda audio: (time.sleep(0.05), "kya haal hai")[1]
    tts = lambda sentence, lang=None: (time.sleep(0.05), (np.zeros(SR, dtype=np.float32), SR))[1]
    agent = Agent(stt, llm, tts, lambda s, r: played.append(time.perf_counter()), lambda: log.append("stop"),
                  endpointer=endpointer, on_event=log.append)
    return agent, log, played


async def drive(agent: Agent, wave: np.ndarray) -> None:
    for frame in frames(wave):
        await agent.feed(frame)
        await asyncio.sleep(FRAME / SR)


async def scenario_full():
    agent, log, played = make(Endpointer(min_delay=0.3))
    await drive(agent, np.concatenate((noise(0.5), voice(1.0), noise(5.0))))
    turn = agent.turns[0]
    assert len(agent.turns) == 1 and turn.reply.count("vaakya") == 3 and not turn.interrupted, log
    assert log.index("[THINKING]") < log.index("[SPEAKING]") < len(log) - 1 and log[-1] == "[LISTENING]", log
    assert len(played) == 3 and played[1] - played[0] >= 0.95, "sentence 2 must wait for sentence 1 to finish playing"
    # Streaming: the first audio arrives after ONE sentence of LLM + TTS, not after all three.
    assert turn.llm_first_ms < 200 and turn.first_audio_ms < 500 + 350, turn
    return turn


async def scenario_barge_in():
    agent, log, played = make(Endpointer(min_delay=0.3))
    await drive(agent, np.concatenate((noise(0.5), voice(1.0), noise(1.2), voice(1.0), noise(4.5))))  # talks again mid-reply
    assert "barge-in: stopped" in log and "stop" in log, log
    assert len(agent.turns) == 2 and agent.turns[0].interrupted and not agent.turns[1].interrupted
    assert agent.turns[0].reply.count("vaakya") < 3, "the interrupted reply must not keep going"
    assert agent.turns[1].reply.count("vaakya") == 3 and agent.state == "LISTENING"
    assert agent.turns[1].first_audio_ms < 1000, "the new reply must not wait for the audio that was cut"


async def scenario_last_sentence():
    # A one-sentence reply: the reply task is finished while its 1 s of audio is still playing. Still interruptible.
    agent, log, _ = make(Endpointer(min_delay=0.3))
    agent.llm = lambda text: iter(["Bas ek hi vaakya hai."])
    langs = []
    agent.lang, agent.tts = (lambda s: "hing"), (lambda s, lang=None: (langs.append(lang), (np.zeros(3 * SR, dtype=np.float32), SR))[1])
    await drive(agent, np.concatenate((noise(0.5), voice(1.0), noise(1.2), voice(0.8), noise(0.2))))
    assert "barge-in: stopped" in log and agent.turns[0].interrupted and langs == ["hing"], log


async def scenario_blip():
    # 0.3 s of sound while the agent speaks (a cough, its own voice in the mic): no interruption, and no answer to it.
    agent, log, _ = make(Endpointer(min_delay=0.3))
    agent.llm = lambda text: iter(["Bas ek hi vaakya hai."])
    agent.tts = lambda s, lang=None: (np.zeros(3 * SR, dtype=np.float32), SR)
    await drive(agent, np.concatenate((noise(0.5), voice(1.0), noise(1.2), voice(0.3), noise(3.0))))
    assert "barge-in: stopped" not in log and "ignored: too short to interrupt" in log and len(agent.turns) == 1, log


async def scenario_background():
    # After the user has spoken, a voice 20 dB quieter (a TV across the room) is not a turn.
    agent, log, _ = make(Endpointer(min_delay=0.3))
    await drive(agent, np.concatenate((noise(0.5), voice(1.0), noise(5.0), 0.1 * voice(1.5), noise(2.0))))
    assert len(agent.turns) == 1 and any("background" in str(e) for e in log), log


async def scenario_error():
    # A model that raises: the agent reports it and goes back to listening instead of hanging in THINKING.
    agent, log, _ = make(Endpointer(min_delay=0.3))
    agent.llm = lambda text: (_ for _ in ()).throw(ConnectionError("llama-server is not running"))
    await drive(agent, np.concatenate((noise(0.5), voice(1.0), noise(1.5))))
    assert any(str(e).startswith("error:") for e in log) and agent.state == "LISTENING", log


async def scenario_resume():
    # The text looks unfinished, so the endpointer waits 1.5 s. The user resumes after 0.9 s: ONE turn, not two.
    agent, log, _ = make(Endpointer(lambda text: 0.0, min_delay=0.3, max_delay=1.5))
    heard = []
    agent.stt = lambda audio: (heard.append(len(audio) / SR), "mera number hai")[1]
    await drive(agent, np.concatenate((noise(0.5), voice(0.8), noise(0.9), voice(0.8), noise(5.0))))
    assert len(agent.turns) == 1 and "barge-in: stopped" not in log, log
    assert heard[-1] > 2.0, "the second utterance must be recognized together with the first"


turn = asyncio.run(scenario_full())
asyncio.run(scenario_barge_in())
asyncio.run(scenario_resume())
asyncio.run(scenario_last_sentence())
asyncio.run(scenario_blip())
asyncio.run(scenario_background())
asyncio.run(scenario_error())

# Glossary: every `file.py: name` it points to exists.
glossary = HERE / "GLOSSARY.md"
terms = refs = 0
if glossary.exists():
    found_refs = re.findall(r"`(\w+\.py): ([^`]+)`", glossary.read_text())
    missing = [(f, name) for f, name in found_refs if name not in (HERE / f).read_text()]
    assert not missing, f"glossary points at code that is not there: {missing}"
    terms, refs = len(re.findall(r"^\| \d+ \|", glossary.read_text(), flags=re.M)), len(found_refs)
    assert terms >= 100, terms

print(f"all checks passed ({terms} glossary terms, {refs} code references); "
      f"fake-model turn: first audio {turn.first_audio_ms:.0f} ms after the user stopped")
