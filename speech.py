"""Local speech-to-text and text-to-speech on CPU, both through sherpa-onnx. No API keys.

STT: Meta Omnilingual ASR 300M (CTC, INT8). One model for Hindi, Marathi and English.
TTS: Supertonic 3 (INT8). One voice for English and Hindi; it has no Marathi.

`transcribe` and `synth` are plain functions (bench.py times them); LocalSTT / LocalTTS wrap them for LiveKit.
Neither model streams, so AgentSession adds the VAD (STT) and the sentence splitter (TTS) around them.
"""

import asyncio
import os
import re
from functools import cache
from pathlib import Path

import numpy as np
import sherpa_onnx
from livekit import rtc
from livekit.agents import APIConnectOptions, stt, tts, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr

MODELS = Path(__file__).parent / "models"
STT_DIR = MODELS / "sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-v2-int8-2026-02-05"
TTS_DIR = MODELS / "sherpa-onnx-supertonic-3-tts-int8-2026-05-11"
THREADS = int(os.getenv("SPEECH_THREADS", "2"))  # 2 = what a free-tier VM or a mid-range phone can spare
TTS_STEPS = int(os.getenv("TTS_STEPS", "5"))  # denoising steps: fewer is faster, more is cleaner
TTS_VOICE = int(os.getenv("TTS_VOICE", "0"))
TTS_SPEED = float(os.getenv("TTS_SPEED", "1.05"))

# Common words that tell Marathi and Hindi apart (both are written in Devanagari).
MARATHI_WORDS = {"आहे", "आहेत", "मी", "मला", "तुम्ही", "तुम्हाला", "नाही", "आणि", "काय", "कसे", "कशी", "करू", "शकता", "माझे", "माझा", "आम्ही", "होते", "आता", "बद्दल", "मध्ये"}
HINDI_WORDS = {"है", "हैं", "मैं", "मुझे", "आप", "आपको", "नहीं", "और", "क्या", "कैसे", "कर", "सकते", "मेरा", "मेरे", "हम", "था", "अभी", "बारे", "में", "हूँ"}


def lang_of(text: str) -> str:
    """en, hi or mr, from the script and a few marker words. Ties go to Hindi."""
    words = set(re.findall(r"[ऀ-ॿ]+", text))
    if not words:
        return "en"
    return "mr" if len(words & MARATHI_WORDS) > len(words & HINDI_WORDS) else "hi"


@cache
def _recognizer() -> sherpa_onnx.OfflineRecognizer:
    return sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
        model=str(STT_DIR / "model.int8.onnx"), tokens=str(STT_DIR / "tokens.txt"), num_threads=THREADS)


@cache
def _synthesizer() -> sherpa_onnx.OfflineTts:
    f = lambda name: str(TTS_DIR / name)
    supertonic = sherpa_onnx.OfflineTtsSupertonicModelConfig(
        duration_predictor=f("duration_predictor.int8.onnx"), text_encoder=f("text_encoder.int8.onnx"),
        vector_estimator=f("vector_estimator.int8.onnx"), vocoder=f("vocoder.int8.onnx"),
        tts_json=f("tts.json"), unicode_indexer=f("unicode_indexer.bin"), voice_style=f("voice.bin"))
    return sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(supertonic=supertonic, num_threads=THREADS)))


def transcribe(samples: np.ndarray, sample_rate: int) -> str:
    """samples: mono float32 in [-1, 1]."""
    stream = _recognizer().create_stream()
    stream.accept_waveform(sample_rate, samples)
    _recognizer().decode_stream(stream)
    return stream.result.text.strip()


def synth(text: str) -> tuple[np.ndarray, int]:
    """Returns (mono float32 samples, sample rate)."""
    config = sherpa_onnx.GenerationConfig()
    config.sid, config.speed, config.num_steps = TTS_VOICE, TTS_SPEED, TTS_STEPS
    # ponytail: Supertonic 3 has no Marathi, so Marathi is read with the Hindi voice (same script, Hindi accent).
    # Swap in a real Marathi voice here when an open one with a usable license exists.
    config.extra = {"lang": "en" if lang_of(text) == "en" else "hi"}
    audio = _synthesizer().generate(text, config)
    return np.asarray(audio.samples, dtype=np.float32), audio.sample_rate


class LocalSTT(stt.STT):
    def __init__(self) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        _recognizer()  # load now, not on the caller's first sentence

    async def _recognize_impl(self, buffer: utils.AudioBuffer, *, language: NotGivenOr[str] = NOT_GIVEN,
                              conn_options: APIConnectOptions) -> stt.SpeechEvent:
        frame = rtc.combine_audio_frames(buffer)
        samples = np.frombuffer(frame.data, dtype=np.int16).reshape(-1, frame.num_channels).mean(axis=1) / 32768
        text = await asyncio.to_thread(transcribe, samples.astype(np.float32), frame.sample_rate)
        return stt.SpeechEvent(type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                               alternatives=[stt.SpeechData(language=lang_of(text), text=text)])


class LocalTTS(tts.TTS):
    def __init__(self) -> None:
        super().__init__(capabilities=tts.TTSCapabilities(streaming=False),
                         sample_rate=_synthesizer().sample_rate, num_channels=1)

    def synthesize(self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> tts.ChunkedStream:
        return _LocalChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class _LocalChunkedStream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        samples, sample_rate = await asyncio.to_thread(synth, self.input_text)
        output_emitter.initialize(request_id=utils.shortuuid(), sample_rate=sample_rate, num_channels=1, mime_type="audio/pcm")
        output_emitter.push((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())
        output_emitter.flush()
