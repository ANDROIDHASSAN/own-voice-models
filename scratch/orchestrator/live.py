"""A live call in the browser, like a LiveKit room, but every part is ours: VAD, end-of-turn, STT, LLM, TTS.

    python live.py           # then open http://127.0.0.1:8791 and click "Call shuru"

The browser streams the microphone as 16 kHz float32 over a WebSocket, 20 ms at a time. Here each frame goes
into the same Agent as agent.py (lesson 4): the VAD finds speech, the endpointer decides the user has finished,
STT -> LLM -> TTS answer sentence by sentence, and talking over the agent interrupts it (barge-in). The reply
audio goes back over the same socket and the browser plays it; "stop" makes the browser cut it at once.

Messages to the browser: binary = [uint32 sample rate][float32 samples] to play; JSON = {"stop"}, an event
line, or a finished turn's latencies. One call at a time: the models are not shared between callers.
"""

import asyncio
import struct
from pathlib import Path

import numpy as np
from aiohttp import WSMsgType, web

import eot
import nvad
from nvad import NeuralVAD
from pipeline import Agent
from teacher import Teacher
from turn import Endpointer
from vad import FRAME, VAD
from web import CHAT, COMPLETE, STT, TTS, TTS_TEXT, romanize  # loads the models once

PAGE = Path(__file__).with_name("live.html")
EOT = eot.load() if eot.CKPT.exists() else None  # lesson 9's turn detector, once trained (runs/eot.pt)
# The neural VAD (v3 + energy hybrid) beats the energy VAD everywhere robust.py measures except short answers
# in a quiet room (35/40 vs 40/40). v1 was off: it never fired on one-word answers ("दिल्ली": p 0.52, one frame).
USE_NVAD = nvad.CKPT.exists()
BUSY = asyncio.Lock()  # ponytail: one caller at a time, a second one is told to wait


async def call(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    if BUSY.locked():
        await ws.send_json({"event": "Ek call pehle se chal rahi hai. Use band karke try karo."})
        await ws.close()
        return ws
    q = request.query
    voice = {"speaker": q.get("speaker"), "style": q.get("style", "conv"), "guidance": float(q.get("guidance", 1))}
    llm_name = q.get("llm", "scratch")
    teacher = Teacher() if llm_name == "teacher" else None
    llm = teacher or {"scratch": CHAT, "echo": lambda text: iter([text + "."])}[llm_name]
    lang = (lambda sentence: "hing") if llm_name == "scratch" else TTS_TEXT.guess_lang
    # The teacher picks the mood per sentence (praise = happy); otherwise the caller's style for every sentence.
    style = lambda: teacher.style if teacher else voice["style"]
    # A teacher waits for an answer it asked for, so a short pause already means "done": no 1.5 s of patience.
    # Otherwise our turn detector (eot.py, trained on what the STT writes) if it is trained, else the scratch
    # LLM's guess (it reads Roman Hinglish and finds short Devanagari answers "unfinished").
    endpointer = Endpointer(None if teacher else EOT or (lambda text: COMPLETE(romanize(text))))
    sends: set[asyncio.Task] = set()

    def send(data) -> None:  # the Agent calls play / stop / on_event synchronously: queue the socket write
        task = asyncio.create_task(ws.send_bytes(data) if isinstance(data, bytes) else ws.send_json(data))
        sends.add(task)
        task.add_done_callback(sends.discard)

    def event(line: str) -> None:
        send({"event": line})
        if line == "[SPEAKING]" and agent.turns:
            t = agent.turns[-1]
            send({"turn": {k: round(getattr(t, k)) for k in ("endpoint_ms", "stt_ms", "llm_first_ms", "tts_first_ms", "first_audio_ms")}})

    agent = Agent(STT, llm, lambda s, lang: TTS(s, lang, voice["speaker"], style(), voice["guidance"]),
                  play=lambda samples, rate: send(struct.pack("<I", rate) + samples.astype(np.float32).tobytes()),
                  stop=lambda: send({"stop": True}), lang=lang, on_event=event, endpointer=endpointer,
                  vad=NeuralVAD() if USE_NVAD else VAD())
    async with BUSY:
        send({"event": f"connected · {llm_name} · awaaz {voice['speaker']} · VAD {type(agent.vad).__name__} · "
                       f"turn {'eot' if EOT and not teacher else 'teacher' if teacher else 'scratch LLM'}"})
        if teacher:
            agent.say(teacher.intro())
        buf = np.zeros(0, dtype=np.float32)
        async for msg in ws:
            if msg.type != WSMsgType.BINARY:
                continue
            buf = np.concatenate([buf, np.frombuffer(msg.data, dtype=np.float32)])
            while len(buf) >= FRAME:
                frame, buf = buf[:FRAME], buf[FRAME:]
                await agent.feed(frame)
        if agent._task:
            agent._task.cancel()
    return ws


def main() -> None:
    app = web.Application()
    app.router.add_get("/", lambda _: web.FileResponse(PAGE))
    app.router.add_get("/voices", lambda _: web.json_response({"speakers": getattr(TTS, "speakers", []), "styles": getattr(TTS, "styles", [])}))
    app.router.add_get("/ws", call)
    print("open http://127.0.0.1:8791", flush=True)
    web.run_app(app, host="127.0.0.1", port=8791, print=None)


if __name__ == "__main__":
    main()
