"""Times every stage on this machine and checks speech by round trip: text -> TTS -> STT -> text.

    python bench.py            # STT + TTS only
    python bench.py --llm      # also the LLM (llama-server must be running, see README)

Exits non-zero if a round trip comes back too different from what was said.
"""

import difflib
import json
import os
import sys
import time
import unicodedata
import urllib.request

from speech import lang_of, synth, transcribe

LLM_URL = os.getenv("LLM_URL", "http://127.0.0.1:8080/v1")
SENTENCES = {
    "en": "Hello, I am a voice agent that runs fully on open source models.",
    "hi": "नमस्ते, मैं एक वॉइस एजेंट हूँ और मैं आपकी मदद कर सकता हूँ।",
    "mr": "नमस्कार, मी एक व्हॉइस एजंट आहे आणि मी तुम्हाला मदत करू शकतो.",
}
MIN_SIMILARITY = {"en": 0.8, "hi": 0.7, "mr": 0.5}  # Marathi is read by a Hindi voice, so it gets more slack


def normalize(text: str) -> str:
    # keep letters, digits and combining marks (Devanagari vowel signs are marks, not letters)
    return "".join(c for c in unicodedata.normalize("NFC", text.lower()) if unicodedata.category(c)[0] in "LNM" or c == " ")


def speech() -> bool:
    synth("warm up"), transcribe(synth("warm up")[0], 44100)  # first call loads the models
    ok = True
    for lang, said in SENTENCES.items():
        assert lang_of(said) == lang, f"lang_of misread the {lang} sentence"
        t = time.perf_counter()
        samples, rate = synth(said)
        tts_s = time.perf_counter() - t
        t = time.perf_counter()
        heard = transcribe(samples, rate)
        stt_s = time.perf_counter() - t
        audio_s = len(samples) / rate
        score = difflib.SequenceMatcher(None, normalize(said), normalize(heard)).ratio()
        ok &= score >= MIN_SIMILARITY[lang]
        print(f"{lang}  tts {tts_s * 1000:5.0f} ms (RTF {tts_s / audio_s:.2f})  stt {stt_s * 1000:5.0f} ms (RTF {stt_s / audio_s:.2f})"
              f"  audio {audio_s:.1f} s  round-trip {score:.2f}\n    said : {said}\n    heard: {heard}")
    return ok


def llm() -> None:
    for lang, said in SENTENCES.items():
        body = {"model": "local", "stream": True, "max_tokens": 80,
                "messages": [{"role": "system", "content": "Reply in one short sentence, in the user's language."},
                             {"role": "user", "content": said}]}
        req = urllib.request.Request(f"{LLM_URL}/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        t = time.perf_counter()
        first, reply = None, ""
        for line in urllib.request.urlopen(req):
            if line.startswith(b"data: {"):
                delta = json.loads(line[6:])["choices"][0]["delta"].get("content") or ""
                first = first or (delta and time.perf_counter() - t)
                reply += delta
        total = time.perf_counter() - t
        print(f"{lang}  llm first token {first * 1000:5.0f} ms  total {total * 1000:5.0f} ms\n    reply: {reply.strip()}")


if __name__ == "__main__":
    passed = speech()
    if "--llm" in sys.argv:
        llm()
    sys.exit(0 if passed else 1)
