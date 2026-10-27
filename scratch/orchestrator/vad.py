"""Lesson 1 - VAD: is anyone speaking right now?

    python vad.py

A voice agent's microphone is always open. Voice activity detection (VAD) is the gate in front of
everything else: it decides which stretch of audio is a spoken utterance. Without it the recognizer would
transcribe silence, and the agent would never know when to answer.

This is the classic energy VAD, the kind every phone had before neural ones:

    1. cut the audio into 20 ms frames and measure each frame's loudness in decibels
    2. track the NOISE FLOOR: the loudness of the room when nobody speaks (it follows quiet frames quickly
       and loud frames slowly, so a fan is learned as "floor" and speech is not)
    3. a frame is "voiced" if it is clearly louder than the floor

One frame is too jumpy to act on ("s" and "t" are nearly silent, a door slam is loud), so on top sits a
small state machine with HYSTERESIS, different rules for starting and stopping:

    SILENCE --(voiced for `start_ms`)--> SPEECH --(unvoiced for `end_ms`)--> SILENCE
                                            emits START                         emits END

    pre-roll   the START event comes `start_ms` late by design, so we keep the last few hundred ms and
               hand them over too. Otherwise every utterance would lose its first consonant.

What an energy VAD cannot do: tell speech from any other loud sound (music, a horn, typing). Silero VAD,
which the real agent uses, is a small neural network trained for exactly that. This one is the baseline
that makes it clear what the network is for.
"""

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

SR = 16000
FRAME = 320  # 20 ms at 16 kHz


@dataclass
class Event:
    kind: str  # "start" or "end"
    audio: np.ndarray | None = None  # with "end": the whole utterance, pre-roll included
    silence_ms: float = 0.0  # with "end": how much trailing silence was waited out


class VAD:
    def __init__(self, start_ms: int = 100, end_ms: int = 500, margin_db: float = 12.0, pre_roll_ms: int = 300) -> None:
        self.start_frames, self.end_frames = start_ms // 20, end_ms // 20
        self.margin_db = margin_db
        self.floor_db: float | None = None  # set from the first frame: a room with a loud fan starts at the fan's level
        self.speaking = False
        self.run = 0  # consecutive frames that disagree with the current state
        self.pre_roll: deque[np.ndarray] = deque(maxlen=pre_roll_ms // 20)
        self.utterance: list[np.ndarray] = []

    def is_voiced(self, frame: np.ndarray) -> bool:
        """One frame's decision. Also updates the noise floor."""
        level = 10 * math.log10(float(np.mean(frame.astype(np.float64) ** 2)) + 1e-10)
        if self.floor_db is None:
            self.floor_db = level
        # Asymmetric tracking: fall to a quieter level fast, rise slowly, and during speech hardly at all.
        # Speech is short bursts, so it never drags the floor up; a fan that is switched on does, in a second or two.
        rate = 0.2 if level < self.floor_db else 0.005 if self.speaking else 0.02
        self.floor_db += rate * (level - self.floor_db)
        return level > self.floor_db + self.margin_db

    def feed(self, frame: np.ndarray) -> Event | None:
        """Give it one 20 ms frame (320 float samples). Returns an Event when the state changes, else None."""
        voiced = self.voiced = self.is_voiced(frame)  # the raw per-frame answer, for callers that count talk time
        if not self.speaking:
            self.pre_roll.append(frame)
            self.run = self.run + 1 if voiced else 0
            if self.run >= self.start_frames:  # enough voiced frames in a row: this is speech, not a click
                self.speaking, self.run = True, 0
                self.utterance = list(self.pre_roll)
                self.pre_roll.clear()
                return Event("start")
            return None
        self.utterance.append(frame)
        self.run = self.run + 1 if not voiced else 0
        if self.run >= self.end_frames:  # a long enough pause: the utterance is over
            self.speaking, silence, self.run = False, self.run * 20.0, 0
            audio, self.utterance = np.concatenate(self.utterance), []
            return Event("end", audio, silence)
        return None


def frames(wave: np.ndarray):
    for i in range(0, len(wave) - FRAME + 1, FRAME):
        yield wave[i : i + FRAME]


def demo() -> None:
    rng = np.random.default_rng(0)
    t = np.arange(SR) / SR
    voice = lambda seconds: 0.2 * np.sin(2 * np.pi * 180 * t[: int(seconds * SR)]) * (1 + np.sin(2 * np.pi * 4 * t[: int(seconds * SR)]))
    silence = lambda seconds: np.zeros(int(seconds * SR))
    # room noise, then: 1 s of "speech", a 0.2 s pause (inside a sentence), 0.8 s more speech, then silence
    wave = np.concatenate((silence(1.0), voice(1.0), silence(0.2), voice(0.8), silence(1.5)))
    wave = (wave + 0.003 * rng.standard_normal(len(wave))).astype(np.float32)

    vad = VAD()
    print("time    event")
    for i, frame in enumerate(frames(wave)):
        event = vad.feed(frame)
        if event and event.kind == "start":
            print(f"{i * 0.02:5.2f}s  START   (speech really began at 1.00 s; the pre-roll covers the delay)")
        elif event:
            print(f"{i * 0.02:5.2f}s  END     utterance of {len(event.audio) / SR:.2f} s handed to the recognizer, "
                  f"after {event.silence_ms:.0f} ms of silence")
    print(f"\nnoise floor learned: {vad.floor_db:.0f} dB.  The 0.2 s pause did not end the utterance: it is shorter than end_ms.")


if __name__ == "__main__":
    demo()
