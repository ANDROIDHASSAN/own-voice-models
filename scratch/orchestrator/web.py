"""A temporary browser page to try the whole thing: speak, and hear our own STT -> LLM -> TTS answer.

    python web.py            # then open http://127.0.0.1:8790   (python web.py 8800: another port)

Push-to-talk, not streaming: the browser records one utterance, resamples it to 16 kHz and posts it. The server
feeds it through the same Agent as agent.py --wav (real-time speed, so the latencies are the real ones) and
sends back what was heard, the reply, every stage's latency and the reply audio. A second panel speaks typed
text with the flow voice, any speaker and style. Only 127.0.0.1: nothing here is meant for a network.
"""

import asyncio
import base64
import io
import json
import sys
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from agent import run_file
from models import gemma, lesson, load_llm, load_stt, load_tts, romanize
from pipeline import Agent
from turn import Endpointer, lm_completeness

PAGE = Path(__file__).with_name("web.html")


def pitch(x: np.ndarray, rate: int) -> int:
    """Median pitch of the loud 85 ms windows, by autocorrelation: shows on the page that the voice really changed."""
    found = []
    for i in range(0, len(x) - 2048, 1024):
        f = x[i : i + 2048]
        if np.sqrt((f**2).mean()) > 0.02:
            a = np.correlate(f, f, "full")[2047:]
            found.append(rate / (rate // 400 + np.argmax(a[rate // 400 : rate // 70])))
    return round(float(np.median(found))) if found else 0


def wav_b64(samples: np.ndarray, rate: int) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())
    return base64.b64encode(buf.getvalue()).decode()


print("loading models ...")
STT, TTS = load_stt(), load_tts()
CHAT, LM, TOK = load_llm()
COMPLETE = lm_completeness(LM, TOK)
(TTS_TEXT,) = lesson("tts", "text")
LOCK = threading.Lock()  # ponytail: one turn at a time; the models are not shared between threads


def turn(samples: np.ndarray, llm_name: str, voice: dict) -> dict:
    llm = {"scratch": CHAT, "echo": lambda text: iter([text + "."]), "gemma": gemma()}[llm_name]
    lang = (lambda sentence: "hing") if llm_name == "scratch" else TTS_TEXT.guess_lang
    played, used = [], []

    def tts(sentence: str, lang: str | None) -> tuple[np.ndarray, int]:
        out = TTS(sentence, lang, **voice)
        used.append(f"flow 24 kHz, {voice['speaker']}, {voice['style']}" if out[1] == 24000 else "purana 22 kHz (Roman / Hinglish)")
        return out

    agent = Agent(STT, llm, tts, lambda s, rate: played.append((s, rate)), lambda: None, lang=lang,
                  endpointer=Endpointer(lambda text: COMPLETE(romanize(text))))
    # The VAD takes its noise floor from the first frame (vad.py). A browser recording starts the moment the
    # button is pressed, often already loud, so give it half a second of quiet room first, like a real mic has.
    quiet = np.random.default_rng(0).normal(0, 1e-3, 8000).astype(np.float32)
    asyncio.run(run_file(agent, np.concatenate([quiet, samples]), None))
    t = agent.turns[-1] if agent.turns else None
    return {"user": t.user if t else "", "reply": t.reply if t else "(VAD ko koi awaaz nahi mili: zor se ya paas se bolo)",
            "ms": {k: round(getattr(t, k)) for k in ("endpoint_ms", "stt_ms", "llm_first_ms", "tts_first_ms", "first_audio_ms")} if t else {},
            "llm": llm_name, "tts": sorted(set(used)), "seconds": round(len(samples) / 16000, 1),
            "audio": [wav_b64(s, rate) for s, rate in played]}


class Handler(BaseHTTPRequestHandler):
    def send(self, body: bytes, kind: str = "application/json", code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/voices":
            return self.send(json.dumps({"speakers": getattr(TTS, "speakers", []), "styles": getattr(TTS, "styles", [])}).encode())
        self.send(PAGE.read_bytes(), "text/html; charset=utf-8")

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers["Content-Length"]))
        try:
            with LOCK:
                if self.path.startswith("/turn"):
                    q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
                    voice = {"speaker": q.get("speaker"), "style": q.get("style", "conv"), "guidance": float(q.get("guidance", 1))}
                    out = turn(np.frombuffer(body, dtype=np.float32).copy(), q.get("llm", "scratch"), voice)
                else:  # /speak
                    req = json.loads(body)
                    samples, rate = TTS(req["text"], None, req.get("speaker"), req.get("style", "conv"), float(req.get("guidance", 1)))
                    out = {"audio": [wav_b64(samples, rate)], "rate": rate, "pitch": pitch(samples, rate)}
                    print(f"speak {req.get('speaker')} {req.get('style')} g{req.get('guidance')} -> {rate} Hz {len(samples) / rate:.2f} s "
                          f"{req['text'][:40]!r}", flush=True)
            self.send(json.dumps(out, ensure_ascii=False).encode())
        except Exception as e:  # show it on the page instead of a silent hang
            self.send(json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(), code=500)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8790
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"open http://127.0.0.1:{port}", flush=True)
    server.serve_forever()
