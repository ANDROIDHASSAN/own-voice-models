"""Lesson 4 - the pipeline: the state machine that turns three models into a conversation.

    microphone frames (20 ms)
        -> VAD            start / end of an utterance                     (lesson 1)
        -> STT            utterance -> text                               (../stt)
        -> endpointer     wait a little, or a lot, before answering       (lesson 2)
        -> LLM            text -> reply, streamed piece by piece          (../llm)
        -> sentences      regroup the stream into sentences
        -> TTS            each sentence -> audio, as soon as it is ready  (../tts)
        -> speaker

The agent is always in one of three states:

    LISTENING --utterance ended, endpointer delay passed--> THINKING --first audio ready--> SPEAKING
        ^                                                       |                              |
        +-----------------------  reply finished playing, or the user started talking  -------+

Three ideas make it feel like a conversation instead of a walkie-talkie:

    streaming     nothing waits for the stage before it to FINISH. The first sentence is being spoken while
                  the LLM writes the second. What the user feels is "time to first audio", not total time.
    barge-in      if the user starts talking while the agent thinks or speaks, the agent stops at once:
                  playback is cut and the rest of the reply is thrown away.
    cancellation  every reply gets a number. A stage only acts if its number is still the current one, so
                  a reply that was interrupted cannot speak up later.

The models are slow, blocking functions, so each call runs in a worker thread and the event loop stays
free to keep reading the microphone. That is the difference between an agent that can be interrupted and
one that is deaf while it thinks.
"""

import asyncio
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import numpy as np

from models import sentences
from turn import Endpointer
from vad import VAD


@dataclass
class Turn:
    """What happened in one exchange, and how long each stage took (milliseconds)."""
    user: str = ""
    reply: str = ""
    endpoint_ms: float = 0.0  # silence waited after the user stopped (VAD hangover + endpointer delay)
    stt_ms: float = 0.0
    llm_first_ms: float = 0.0  # time to the first piece of text from the LLM (TTFT)
    tts_first_ms: float = 0.0  # time to synthesize the first sentence
    first_audio_ms: float = 0.0  # user stopped speaking -> agent starts speaking: the number that matters
    interrupted: bool = False


@dataclass
class Agent:
    stt: Callable[[np.ndarray], str]
    llm: Callable[[str], Iterator[str]]
    tts: Callable[[str, str | None], tuple[np.ndarray, int]]  # (sentence, language) -> (samples, rate)
    play: Callable[[np.ndarray, int], None]  # start playing these samples (returns immediately)
    stop: Callable[[], None]  # cut playback now
    lang: Callable[[str], str] | None = None  # reply text -> language for the TTS; None lets the TTS guess
    vad: VAD = field(default_factory=VAD)
    endpointer: Endpointer = field(default_factory=Endpointer)
    on_event: Callable[[str], None] = print
    state: str = "LISTENING"
    turns: list[Turn] = field(default_factory=list)
    _generation: int = 0  # bumped on every barge-in: older replies see they are stale and stop
    _pending: list[np.ndarray] = field(default_factory=list)  # utterances not answered yet (the user may go on)
    _task: asyncio.Task | None = None
    _speaking_until: float = 0.0  # when the audio handed to play() so far will have finished
    min_interrupt: float = 0.5  # seconds the user must talk over the agent before it stops
    _barge_at: float | None = None  # when speech over the agent started, until it is long enough or ends
    _barge_voiced: float = 0.0  # seconds of voiced frames in it so far
    user_db: float | None = None  # the user's usual loudness (dB), learned from the turns they speak
    background_db: float = 10.0  # this much quieter than the user = someone else in the room

    @property
    def busy(self) -> bool:
        """A reply is being prepared, or its audio is still coming out of the speaker."""
        return bool(self._task and not self._task.done()) or time.perf_counter() < self._speaking_until

    async def feed(self, frame: np.ndarray) -> None:
        """Call this for every 20 ms microphone frame."""
        event = self.vad.feed(frame)
        now = time.perf_counter()
        if event is None:
            if self.state != "LISTENING" and not self.busy:
                self._set("LISTENING")
            # Talking over the agent counts only once it has lasted `min_interrupt` of VOICED frames (LiveKit's
            # min_interruption_duration, 0.5 s): a cough, a door, or the agent's own voice leaking from the
            # speaker into the mic is shorter, and must not cut the reply. (Not the VAD's state: that stays
            # "speaking" for its 0.5 s hangover after any sound.)
            if self._barge_at is not None:
                self._barge_voiced += 0.02 if self.vad.voiced and self._near(frame) else 0.0
                if self._barge_voiced >= self.min_interrupt:
                    self._barge_at = None
                    self._interrupt()
        elif event.kind == "start":
            if self.busy and self.state == "SPEAKING":
                self._barge_at, self._barge_voiced = now, self.vad.start_frames * 0.02  # the frames that made START
            elif self.busy:  # still thinking: the user carried on after a pause, so wait for all of it
                self._interrupt()
        elif self._barge_at is not None:  # it ended before it was long enough to interrupt: not a turn, drop it
            self._barge_at = None
            self.on_event("ignored: too short to interrupt")
        elif not self._near(event.audio):
            self.on_event("ignored: far quieter than the user, background talk")
        else:
            # The loudest recent turn, forgetting 1 dB per turn: background talk heard BEFORE the user spoke
            # (a TV already on) must not drag the reference down the way an average does.
            level = _loudness(event.audio)
            self.user_db = level if self.user_db is None else max(level, self.user_db - 1.0)
            self._pending.append(event.audio)
            self._task = asyncio.create_task(self._respond(self._generation, event.silence_ms))
            self._task.add_done_callback(self._finished)

    def _near(self, audio: np.ndarray) -> bool:
        """The user sits at the microphone; a TV or other people are further away and come in quieter. Once the
        user has spoken, anything more than `background_db` below their usual level is background, not them.
        (Measured with a TV 15 dB under the user: without this it was answered and cut the agent off.)"""
        return self.user_db is None or _loudness(audio) >= self.user_db - self.background_db

    def say(self, text: str) -> None:
        """Speak without being asked: a greeting, a teacher's first question. Interruptible like any reply."""
        self._task = asyncio.create_task(self._say(self._generation, text))
        self._task.add_done_callback(self._finished)

    async def _say(self, generation: int, text: str) -> None:
        for sentence in sentences(iter([text])):
            samples, rate = await asyncio.to_thread(self.tts, sentence, self.lang(sentence) if self.lang else None)
            await asyncio.sleep(max(0.0, self._speaking_until - time.perf_counter()))
            if generation != self._generation:
                return
            self._set("SPEAKING")
            self.on_event(f"agent: {sentence}")
            self.play(samples, rate)
            self._speaking_until = time.perf_counter() + len(samples) / rate

    def _interrupt(self) -> None:
        self._generation += 1  # everything belonging to the old reply is now stale
        if self._task:
            self._task.cancel()
        self.stop()
        self._speaking_until = 0.0  # the speaker is silent now: the next reply must not wait for the cut audio
        if self.state != "LISTENING":
            if self.turns:  # say() speaks without a turn (a greeting): nothing to mark
                self.turns[-1].interrupted = True
            self.on_event("barge-in: stopped")
        self._set("LISTENING")

    def _finished(self, task: asyncio.Task) -> None:
        """A model call that raises must not leave the agent stuck in THINKING, silently."""
        if not task.cancelled() and task.exception():
            self.on_event(f"error: {task.exception()!r}")
            self._pending.clear()
            self._set("LISTENING")

    async def _respond(self, generation: int, silence_ms: float) -> None:
        turn, t_end = Turn(), time.perf_counter()
        audio = np.concatenate(self._pending)
        turn.user = await asyncio.to_thread(self.stt, audio)
        turn.stt_ms = (time.perf_counter() - t_end) * 1000
        if not turn.user.strip():
            self._pending.clear()
            return

        # The endpointer chooses how much silence this transcript deserves. The VAD already waited
        # `silence_ms` of it, and recognizing took time too; sleep only for what is left.
        delay = await asyncio.to_thread(self.endpointer.delay, turn.user)
        wait = delay - silence_ms / 1000 - (time.perf_counter() - t_end)
        if wait > 0:
            await asyncio.sleep(wait)  # if the user resumes during this sleep, feed() cancels this task
        turn.endpoint_ms = silence_ms + (time.perf_counter() - t_end) * 1000 - turn.stt_ms
        self._pending.clear()
        self.turns.append(turn)
        self._set("THINKING")
        self.on_event(f"user : {turn.user}")

        t_llm, lang = time.perf_counter(), None
        pieces = _while(self.llm(turn.user), lambda: generation == self._generation)
        async for sentence in _aiter(sentences(_timed(pieces, turn, t_llm))):
            if generation != self._generation:
                return
            lang = lang or (self.lang(sentence) if self.lang else None)  # one language for the whole reply
            t_tts = time.perf_counter()
            samples, rate = await asyncio.to_thread(self.tts, sentence, lang)
            if generation != self._generation:
                return
            if not turn.reply:
                turn.tts_first_ms = (time.perf_counter() - t_tts) * 1000
                turn.first_audio_ms = (time.perf_counter() - t_end) * 1000 + silence_ms
                self._set("SPEAKING")
            turn.reply += sentence + " "
            self.on_event(f"agent: {sentence}")
            # Sentences are queued back to back: wait until the previous one has finished playing.
            await asyncio.sleep(max(0.0, self._speaking_until - time.perf_counter()))
            if generation != self._generation:
                return
            self.play(samples, rate)
            self._speaking_until = time.perf_counter() + len(samples) / rate

    def _set(self, state: str) -> None:
        if state != self.state:
            self.state = state
            self.on_event(f"[{state}]")


def _loudness(audio: np.ndarray) -> float:
    """dB of the louder half of the 20 ms frames: the speech itself, not the pauses and pre-roll around it."""
    n = len(audio) // 320 * 320
    if n == 0:
        return 10 * np.log10(float(np.mean(audio**2)) + 1e-10)
    energy = np.sort(np.mean(audio[:n].reshape(-1, 320) ** 2, axis=1))
    return 10 * float(np.log10(energy[len(energy) // 2 :].mean() + 1e-10))


def _timed(pieces: Iterator[str], turn: Turn, t0: float) -> Iterator[str]:
    for i, piece in enumerate(pieces):
        if i == 0:
            turn.llm_first_ms = (time.perf_counter() - t0) * 1000
        yield piece


def _while(iterator: Iterator[str], alive: Callable[[], bool]) -> Iterator[str]:
    """Stop pulling from a model's iterator once its reply is stale (the model stops generating with it)."""
    for item in iterator:
        if not alive():
            return
        yield item


async def _aiter(iterator: Iterator[str]):
    """Step a blocking iterator from the event loop: each next() runs in a worker thread."""
    done = object()
    while (item := await asyncio.to_thread(next, iterator, done)) is not done:
        yield item
