"""Lesson 3 - the three models behind three plain functions.

The pipeline (lesson 4) does not know what a transformer is. It needs exactly three things:

    stt(samples)   -> text                  16 kHz mono float32 in, one string out
    llm(text)      -> pieces of text        an ITERATOR: words arrive while the model is still writing
    tts(text)      -> (samples, rate)       one sentence in, audio out

Anything with those shapes can be plugged in. That is the whole trick of a framework like LiveKit Agents:
the models are plugins behind small interfaces, and the orchestration never changes when you swap one.
Here the plugins are the models from ../stt, ../llm and ../tts, plus Gemma (the real agent's LLM, served by
llama-server) as a second LLM, to show a swap.
"""

import importlib
import json
import os
import re
import sys
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np

SCRATCH = Path(__file__).parent.parent


def lesson(folder: str, *names: str) -> list:
    """Import ../<folder>/<name>.py for each name. Every lesson folder has its own model.py and text.py and
    imports them by their short names, so only one folder can be "the current one" at a time: put it first
    on the path, import, then forget the short names again so the next folder can use them."""
    path = str(SCRATCH / folder)
    _forget()  # a caller may already have imported another folder's model.py under the short name
    sys.path.insert(0, path)
    try:
        return [importlib.import_module(name) for name in names]
    finally:
        sys.path.remove(path)
        _forget()


def _forget() -> None:
    """Drop the short-named lesson modules (model, text, audio, ...) of llm, stt and tts from the import cache."""
    folders = tuple(str(SCRATCH / f) + "/" for f in ("llm", "stt", "tts"))
    for key, module in list(sys.modules.items()):
        if "." not in key and str(getattr(module, "__file__", "")).startswith(folders):
            del sys.modules[key]


def confidence(net, log_mel: Callable, wave: np.ndarray) -> tuple[list[int], float]:
    """Greedy CTC labels, and the mean probability of the frames that wrote a character (blank frames say nothing)."""
    import torch

    with torch.no_grad():
        mel = log_mel(wave)[None]
        log_probs, _ = net(mel, torch.tensor([mel.shape[1]]))
    p, best = log_probs[0].exp().max(-1)
    merged = torch.unique_consecutive(best)
    keep = best != 0
    return merged[merged != 0].tolist(), float(p[keep].mean()) if keep.any() else 0.0


def load_stt(min_confidence: float = 0.5) -> Callable[[np.ndarray], str]:
    """Below `min_confidence` the transcript is dropped (""), and the agent does not answer. Measured with
    robust.py: real speech, clean or at 5-10 dB SNR, scores 0.49-0.95; hiss, pink noise and a fan 0.24-0.40.
    Background chatter scores up to 0.63: it IS speech, which only a better VAD can tell apart."""
    audio, model, text = lesson("stt", "audio", "model", "text")
    # Newest first: 158 h + noise (robust.py: babble at 10 dB CER 0.40 -> 0.15), 158 h, the 23 h lesson model.
    found = [p for p in ("stt/runs/kaggle/noisy.pt", "stt/runs/kaggle/kathbath.pt", "stt/runs/ckpt.pt") if (SCRATCH / p).exists()]
    net, ckpt = model.load(SCRATCH / found[0], "cpu")
    vocab = text.Vocab(ckpt["chars"])

    def stt(samples: np.ndarray) -> str:
        labels, conf = confidence(net, audio.log_mel, samples)
        return vocab.decode(labels) if conf >= min_confidence else ""

    return stt


def load_tts() -> Callable[[str, str | None], tuple[np.ndarray, int]]:
    """Lesson 6's voice (22.05 kHz, all four languages) and, when it exists, lesson 10's flow voice (24 kHz, Rasa:
    Hindi and Marathi only). Devanagari goes to the flow voice; Roman-script English / Hinglish, which our LLM
    writes, can only go to the old one."""
    speak, model, vocoder, text, audio = lesson("tts", "speak", "model", "vocoder", "text", "audio")
    acoustic, ckpt = model.load(SCRATCH / "tts/runs/acoustic.pt", "cpu")
    voc, _ = vocoder.load(SCRATCH / "tts/runs/vocoder.pt", "cpu")
    vocab = text.Vocab(ckpt["chars"])
    old = lambda sentence, lang: (speak.speak(acoustic, voc, vocab, sentence, lang), audio.SR)
    final = SCRATCH / "tts/runs24k/final"
    if not (final / "flow.pt").exists():
        return lambda sentence, lang=None: old(sentence, lang)

    os.environ["TTS_SR"] = "24000"  # audio.py reads it at import: the second import below is the 24 kHz one
    try:
        speak24, model24, vocoder24, text24, audio24 = lesson("tts", "speak", "model", "vocoder", "text", "audio")
    finally:
        del os.environ["TTS_SR"]
    flow, ckpt24 = model24.load(final / "flow.pt", "cpu")
    voc24, _ = vocoder24.load(final / "vocoder_gan.pt", "cpu")
    vocab24, speakers = text24.Vocab(ckpt24["chars"]), ckpt24["speakers"]
    styles = ckpt24["styles"]

    def tts(sentence: str, lang: str | None = None, speaker: str | None = None, style: str = "conv",
            guidance: float = 1.0) -> tuple[np.ndarray, int]:
        lang = lang or text.guess_lang(sentence)
        if lang in ("hi", "mr"):
            voice = {"speaker": speakers.index(speaker or f"rasa_{lang}_female"), "style": styles.index(style),
                     "steps": 4, "guidance": guidance}
            return speak24.speak(flow, voc24, vocab24, sentence, lang, **voice), audio24.SR
        return old(sentence, lang)

    tts.speakers, tts.styles = speakers, styles[1:]  # for a UI: styles[0] is "<none>", the guidance-off style
    return tts


# Devanagari -> Roman, by rule. Our recognizer writes Hindi in Devanagari; our LLM chats in Roman-script
# Hinglish. One table and one rule (drop the unwritten "a" at the end of a word) bridge the two.
_CONSONANTS = dict(zip("कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहळ", "k kh g gh n ch chh j jh n t th d dh n t th d dh n p ph b bh m y r l v sh sh s h l".split()))
_MATRAS = dict(zip("ािीुूृेैोौॅॉ", "aa i ee u oo ri e ai o au e o".split()))
_VOWELS = dict(zip("अआइईउऊएऐओऔऋ", "a aa i ee u oo e ai o au ri".split()))


_NUKTA = dict(zip("जफडढकखग", "z f r rh q kh g".split()))  # these letters with a nukta dot below are different sounds


def romanize(text: str) -> str:
    out = []
    for word in text.split():
        roman, chars = [], list(word)
        for i, c in enumerate(chars):
            nxt = chars[i + 1] if i + 1 < len(chars) else ""
            if c in _CONSONANTS:
                if nxt == "़":  # a nukta changes the consonant (ja -> za); the letter AFTER it decides about the vowel
                    roman.append(_NUKTA.get(c, _CONSONANTS[c]))
                    nxt = chars[i + 2] if i + 2 < len(chars) else ""
                else:
                    roman.append(_CONSONANTS[c])
                # every consonant carries an "a" unless a matra, a halant (्) or the end of the word follows
                if nxt and nxt not in _MATRAS and nxt not in "़्":
                    roman.append("a")
            elif c in _MATRAS:
                roman.append(_MATRAS[c])
            elif c in _VOWELS:
                roman.append(_VOWELS[c])
            elif c in "ंँ":
                roman.append("n")
            elif c not in "़्ः":
                roman.append(c)  # Latin letters and digits pass through
        out.append("".join(roman))
    return " ".join(out)


def load_llm(lang: str = "hing") -> tuple[Callable[[str], Iterator[str]], object, object]:
    """The scratch LLM as a streaming chat function. Also returns (model, tokenizer) for the turn detector."""
    import torch

    generate, model, tokenizer = lesson("llm", "generate", "model", "tokenizer")
    checkpoint = SCRATCH / "llm/runs/ckpt_persona.pt"
    net, _ = model.load(checkpoint if checkpoint.exists() else SCRATCH / "llm/runs/ckpt.pt", "cpu")
    tok = tokenizer.BPE.load(SCRATCH / "llm/data/tokenizer.json")

    def chat(message: str) -> Iterator[str]:
        ids = [tok.special["<|bos|>"], tok.special[f"<|{lang}|>"], *tok.encode(romanize(message) + "\n")]
        produced: list[int] = []
        shown = ""
        for token in generate.stream(net, ids, max_new=40, temperature=0.7, stop=tok.special["<|eos|>"]):
            produced.append(token)
            text = tok.decode(produced)
            if text.endswith("�"):  # half of a multi-byte character: wait for the rest
                continue
            if "\n" in text:  # the reply is one line; a newline means the model started a new message
                break
            yield text[len(shown) :]
            shown = text

    torch.set_num_threads(2)
    return chat, net, tok


def gemma(url: str = "http://127.0.0.1:8080/v1/chat/completions") -> Callable[[str], Iterator[str]]:
    """A different LLM behind the same interface: Gemma on llama-server, streamed over server-sent events.
    Start it the way ../../README.md describes. Nothing else in the pipeline changes."""
    system = "You are a friendly voice assistant. Reply in the user's language, in one or two short sentences, no markdown."

    def chat(message: str) -> Iterator[str]:
        body = json.dumps({"stream": True, "max_tokens": 80, "messages": [
            {"role": "system", "content": system}, {"role": "user", "content": message}]}).encode()
        request = urllib.request.Request(url, body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=30) as response:
            for line in response:  # SSE: each event is a line "data: {json}", the last one "data: [DONE]"
                line = line.decode().strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    piece = json.loads(line[6:])["choices"][0]["delta"].get("content")
                    if piece:
                        yield piece

    return chat


def sentences(pieces: Iterator[str], min_chars: int = 12) -> Iterator[str]:
    """Regroup a stream of text pieces into sentences, yielding each one the moment it is complete.

    The TTS needs whole sentences (prosody depends on where a sentence ends), but waiting for the LLM's full
    reply would add its whole generation time to the delay. So: speak sentence 1 while sentence 2 is being
    written. `min_chars` stops "Hi." from being synthesized alone and sounding chopped."""
    buffer = ""
    for piece in pieces:
        buffer += piece
        while end := next((m.end() for m in re.finditer(r"[.?!।]+[\s\"']*", buffer) if m.end() >= min_chars), 0):
            yield buffer[:end].strip()
            buffer = buffer[end:]
    if buffer.strip():
        yield buffer.strip()
