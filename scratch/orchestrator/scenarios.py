"""Real-call scenarios against a running live.py, over its WebSocket: real Hindi speech (FLEURS), real-time pacing.

    python live.py &                          # then:
    python scenarios.py CLIPS_DIR MUSAN.npy   # 16 kHz FLEURS wavs, held-out MUSAN audio (see robust.py)

Each scenario prints PASS / FAIL and the event log: talking over the greeting, a turn in noise, a TV in the
room, a cough while the agent speaks, a real barge-in. What simulated single turns (agent.py --wav) never hit.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

import aiohttp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "stt"))
import audio  # noqa: E402

CLIPS = sorted(Path(sys.argv[1]).rglob("*.wav"))
MUSAN = np.load(sys.argv[2]).astype(np.float32) / 32768
SR, F = 16000, 320
rng = np.random.default_rng(7)
speech = [audio.read_wav(p)[0] for p in CLIPS[4:12]]
# someone else talking the whole time; each sentence at the same loudness, like a TV (FLEURS clips differ by 30 dB)
tv = np.concatenate([w / np.sqrt(np.mean(w**2)) * 0.05 for w in (audio.read_wav(p)[0] for p in CLIPS[30:40])])


def rms(x):
    return float(np.sqrt(np.mean(x**2)) + 1e-9)


class Call:
    def __init__(self, query: str, background: np.ndarray | None = None, bg_db: float = -99):
        self.query, self.events, self.audio_s = query, [], 0.0
        self.bg = background
        self.bg_gain = 0.0 if background is None else rms(speech[0]) * 10 ** (bg_db / 20) / rms(background)
        self.pos = 0
        self.speaking_since = None

    def frame(self, x: np.ndarray | None = None) -> np.ndarray:
        f = np.random.default_rng(self.pos).normal(0, 1e-4, F).astype(np.float32) if x is None else x
        if self.bg is not None:
            f = f + self.bg_gain * np.resize(self.bg[self.pos % len(self.bg):], F)
        self.pos += F
        return f.astype(np.float32)

    async def run(self, script):
        async with aiohttp.ClientSession() as s, s.ws_connect(f"http://127.0.0.1:8791/ws?{self.query}") as ws:
            self.ws = ws

            async def reader():
                async for m in ws:
                    if m.type == aiohttp.WSMsgType.BINARY:
                        self.audio_s += (len(m.data) - 4) / 4 / int.from_bytes(m.data[:4], "little")
                    else:
                        msg = json.loads(m.data)
                        self.events.append((round(time.perf_counter() - self.t0, 2), msg))

            self.t0 = time.perf_counter()
            task = asyncio.create_task(reader())
            await script(self)
            await ws.close()
            await asyncio.sleep(0.2)
            task.cancel()
        return self

    async def say(self, x: np.ndarray):
        for i in range(0, len(x) - F + 1, F):
            await self.ws.send_bytes(self.frame(x[i : i + F]).tobytes())
            await asyncio.sleep(0.02)

    async def wait(self, seconds: float):
        for _ in range(int(seconds / 0.02)):
            await self.ws.send_bytes(self.frame().tobytes())
            await asyncio.sleep(0.02)

    def ev(self, key):
        return [m[key] for _, m in self.events if key in m]


def report(name, ok, call, extra=""):
    lines = [str(e) for e in call.ev("event") if not e.startswith("connected")]
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {extra}\n      " + " | ".join(lines)[:600], flush=True)
    for t in call.ev("turn"):
        print(f"      first audio {t['first_audio_ms']} ms (endpoint {t['endpoint_ms']}, stt {t['stt_ms']}, tts {t['tts_first_ms']})")
    return ok


async def main():
    results = []

    # 1. The student talks over the teacher's greeting (crashed the call before the fix).
    async def s1(c):
        await c.wait(1.5)
        await c.say(speech[0][: SR * 2])
        await c.wait(8)
    c = await Call("llm=teacher&speaker=rasa_hi_female").run(s1)
    ev = c.ev("event")
    results.append(report("talk over the greeting", "barge-in: stopped" in ev and any(e.startswith("user :") for e in ev), c))

    # 2. A whole turn with MUSAN noise (music / traffic, never trained on) playing at 15 dB under the voice.
    async def s2(c):
        await c.wait(1.5)
        await c.say(speech[1])
        await c.wait(8)
    c = await Call("llm=scratch", MUSAN, -15).run(s2)
    ev = c.ev("event")
    users = [e for e in ev if e.startswith("user :")]
    results.append(report("turn in MUSAN noise (15 dB)", len(users) == 1 and "barge-in: stopped" not in ev, c, f"{len(users)} user turns"))

    # 3. A TV / another person talking the whole time, 15 dB under the user: one turn, the agent not cut by it.
    async def s3(c):
        await c.wait(2.0)
        await c.say(speech[2])
        await c.wait(8)
    c = await Call("llm=scratch", tv, -15).run(s3)
    ev = c.ev("event")
    users = [e for e in ev if e.startswith("user :")]
    results.append(report("turn with a TV talking (15 dB under)", len(users) == 1 and "barge-in: stopped" not in ev, c, f"{len(users)} user turns"))

    # 4. The user speaks once, then only the TV for 30 s: the agent must not answer the TV.
    async def s4(c):
        await c.wait(1.0)
        await c.say(speech[6])
        await c.wait(30)
    c = await Call("llm=scratch", tv, -15).run(s4)
    users = [e for e in c.ev("event") if e.startswith("user :")]
    results.append(report("user once, then 30 s of TV", len(users) == 1, c, f"{len(users) - 1} answers to the TV"))

    # 4b. Known limit, reported, not counted: TV from the very start, before the user has said anything.
    async def s4b(c):
        await c.wait(20)
    c = await Call("llm=scratch", tv, -15).run(s4b)
    users = [e for e in c.ev("event") if e.startswith("user :")]
    print(f"INFO  TV only from the start, no user yet: {len(users)} answers (no loudness reference before the user speaks)")

    # 5. A 0.3 s cough while the agent speaks: it keeps talking and does not answer the cough.
    async def s5(c):
        await c.wait(1.0)
        await c.say(speech[3])
        for _ in range(400):
            await c.wait(0.02)
            if "[SPEAKING]" in c.ev("event"):
                break
        await c.wait(0.5)
        await c.say(0.3 * rng.standard_normal(int(0.3 * SR)).astype(np.float32) * rms(speech[3]) / 0.3)
        await c.wait(6)
    c = await Call("llm=scratch").run(s5)
    ev = c.ev("event")
    results.append(report("0.3 s cough while it speaks", "barge-in: stopped" not in ev and len([e for e in ev if e.startswith("user :")]) == 1, c))

    # 6. Real barge-in: 2 s of speech over the agent: it stops and answers the new one.
    async def s6(c):
        await c.wait(1.0)
        await c.say(speech[4])
        for _ in range(400):
            await c.wait(0.02)
            if "[SPEAKING]" in c.ev("event"):
                break
        await c.wait(0.3)
        await c.say(speech[5][: SR * 2])
        await c.wait(7)
    c = await Call("llm=scratch").run(s6)
    ev = c.ev("event")
    heard = [e for e in ev if e.startswith("user :")]
    results.append(report("real barge-in (2 s over the agent)", "barge-in: stopped" in ev and len(heard) >= 2, c))

    print(f"\n{sum(results)}/{len(results)} scenarios passed")


asyncio.run(main())
