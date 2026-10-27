"""Lesson 5 - run it: talk to the three models you built.

    python agent.py                         # microphone and speakers (use headphones, see below)
    python agent.py --llm gemma             # same pipeline, Gemma as the brain (llama-server must be running)
    python agent.py --wav question.wav      # no microphone: play a 16 kHz file INTO the pipeline, save the reply
    python agent.py --wav q.wav --barge-in 1.0   # ... and start talking again 1 s into the agent's reply

At the end it prints one line per turn with the latency of every stage. The number to look at is
"first audio": from the moment you stopped speaking to the moment the agent starts.

Headphones: the microphone hears the speakers. Without echo cancellation the agent would hear ITSELF,
decide the user is talking, and interrupt its own reply. Phones and browsers solve this with AEC (acoustic
echo cancellation: subtract what you are playing from what you record); a browser gives it to you for free
with getUserMedia({audio: {echoCancellation: true}}). This script has none, so it needs headphones.
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

import numpy as np

from models import gemma, lesson, load_llm, load_stt, load_tts, romanize
from pipeline import Agent
from turn import Endpointer, lm_completeness
from vad import FRAME, SR, frames

HERE = Path(__file__).parent


def report(agent: Agent) -> None:
    print(f"\n{'endpoint':>9s}{'STT':>7s}{'LLM first':>11s}{'TTS first':>11s}{'first audio':>13s}   (ms)")
    for t in agent.turns:
        print(f"{t.endpoint_ms:9.0f}{t.stt_ms:7.0f}{t.llm_first_ms:11.0f}{t.tts_first_ms:11.0f}{t.first_audio_ms:13.0f}   "
              f"{'[interrupted] ' if t.interrupted else ''}{t.user[:40]!r} -> {t.reply[:50]!r}")


async def run_mic(agent: Agent) -> None:
    import sounddevice

    loop, queue = asyncio.get_running_loop(), asyncio.Queue()
    # The audio driver calls this from its own thread; hand each frame over to the event loop.
    callback = lambda indata, *_: loop.call_soon_threadsafe(queue.put_nowait, indata[:, 0].copy())
    with sounddevice.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=FRAME, callback=callback):
        print("listening ... (Ctrl+C to stop)")
        while True:
            await agent.feed(await queue.get())


async def run_file(agent: Agent, wave: np.ndarray, barge_in: float | None) -> None:
    """Feed a recording at real-time speed (the pipeline measures wall-clock time), then silence."""
    silence = np.zeros(FRAME, dtype=np.float32)
    for frame in frames(wave):
        await agent.feed(frame)
        await asyncio.sleep(FRAME / SR)
    spoke_at, idle = None, 0
    for _ in range(int(30 / 0.02)):  # up to 30 s for the reply
        await agent.feed(silence)
        await asyncio.sleep(FRAME / SR)
        if agent.state == "SPEAKING" and spoke_at is None:
            spoke_at = time.perf_counter()
        if barge_in is not None and spoke_at and time.perf_counter() - spoke_at >= barge_in:
            barge_in = None
            for frame in frames(wave[: SR * 2]):  # the user talks over the agent
                await agent.feed(frame)
                await asyncio.sleep(FRAME / SR)
        # Done when nothing has happened for 2 s: no reply in progress, nobody speaking, nothing waiting to be answered.
        idle = 0 if agent.busy or agent.vad.speaking or agent.state != "LISTENING" else idle + 1
        if agent.turns and idle >= 100:
            break


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="scratch", choices=["scratch", "gemma", "echo"], help="echo repeats what it heard")
    ap.add_argument("--wav", help="a 16 kHz recording to use instead of the microphone")
    ap.add_argument("--barge-in", type=float, metavar="SECONDS", help="with --wav: interrupt the reply after this long")
    ap.add_argument("--out", default="reply.wav")
    args = ap.parse_args()

    print("loading models ...")
    stt, tts = load_stt(), load_tts()
    chat, lm, tok = load_llm()
    complete = lm_completeness(lm, tok)
    endpointer = Endpointer(lambda text: complete(romanize(text)))  # the LLM reads Roman script, the recognizer writes Devanagari
    llm = {"scratch": chat, "gemma": gemma(), "echo": lambda text: iter([text + "."])}[args.llm]
    (tts_audio, tts_text), (stt_audio,) = lesson("tts", "audio", "text"), lesson("stt", "audio")
    # The scratch LLM always answers in Roman Hinglish. For the others, the first sentence of a reply decides.
    lang = (lambda sentence: "hing") if args.llm == "scratch" else tts_text.guess_lang

    played: list[np.ndarray] = []
    if args.wav:
        play, stop = lambda samples, rate: played.append(samples), lambda: None
    else:
        import sounddevice

        play, stop = sounddevice.play, sounddevice.stop
    agent = Agent(stt, llm, tts, play, stop, lang=lang, endpointer=endpointer)

    try:
        if args.wav:
            wave, rate = stt_audio.read_wav(args.wav)
            if rate != SR:
                sys.exit(f"{args.wav} is {rate} Hz, need {SR}")
            asyncio.run(run_file(agent, wave, args.barge_in))
            if played:
                tts_audio.write_wav(args.out, np.concatenate(played))
                print(f"reply written to {args.out}")
        else:
            asyncio.run(run_mic(agent))
    except KeyboardInterrupt:
        pass
    report(agent)


if __name__ == "__main__":
    main()
