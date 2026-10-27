"""Lesson 8 - inference: a sentence in, a .wav file out.

    python speak.py "नमस्ते, आप कैसे हैं?"
    python speak.py "kya haal hai bhai, sab theek?" --lang hing --speed 1.1
    python speak.py "Hello there." --griffin-lim       # skip the neural vocoder, to hear the difference

    text -> normalize -> character ids -> acoustic model -> mel -> vocoder -> samples -> out.wav

Two forward passes, no loops: the acoustic model draws all mel frames at once, the vocoder turns all of
them into samples at once. Compare the LLM, which needed one forward pass per token. This is why TTS can
be much faster than real time on a CPU.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import vocoder as voc
from audio import RUNS_DIR, SR, griffin_lim, write_wav
from model import Acoustic, load
from text import LANGS, Vocab, guess_lang

HERE = Path(__file__).parent


@torch.no_grad()
def speak(acoustic: Acoustic, vocoder, vocab: Vocab, text: str, lang: str | None = None, speed: float = 1.0, **voice) -> np.ndarray:
    """Returns mono float32 samples at 22.05 kHz. vocoder=None uses Griffin-Lim.
    `voice`: speaker / style ids and steps / temperature / guidance for a flow model (see Acoustic.synthesize)."""
    device = next(acoustic.parameters()).device
    chars = torch.tensor(vocab.encode(text), device=device)
    if len(chars) == 0:
        return np.zeros(0, dtype=np.float32)
    mel, _ = acoustic.synthesize(chars, LANGS.index(lang or guess_lang(text)), speed, **voice)
    wave = vocoder(mel[None])[0] if vocoder else griffin_lim(mel.cpu())
    return wave.clamp(-1, 1).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--lang", choices=LANGS, help="default: guessed from the text")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--griffin-lim", action="store_true")
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--device", default="cpu", help="cpu is the honest number: the target is a phone or a browser")
    ap.add_argument("--model", default="acoustic", help="checkpoint in runs/: acoustic (lesson 6) or flow (lesson 10)")
    ap.add_argument("--vocoder", default="vocoder", help="checkpoint in runs/: vocoder, or vocoder_gan")
    ap.add_argument("--speaker", help="flow model: a voice, e.g. rasa_hi_female (see the list it prints)")
    ap.add_argument("--style", help="flow model: e.g. happy, sad, angry (default neutral; a wrong name prints the list)")
    ap.add_argument("--steps", type=int, default=4, help="flow model: Euler steps. CPU, 3.4 s sentence: 10 -> RTF 0.40, 4 -> 0.17, 2 -> 0.09")
    ap.add_argument("--temperature", type=float, default=0.667)
    ap.add_argument("--guidance", type=float, default=1.0, help="> 1 exaggerates the style, e.g. 2")
    args = ap.parse_args()

    acoustic, ckpt = load(RUNS_DIR / f"{args.model}.pt", args.device)
    vocoder = None if args.griffin_lim else voc.load(RUNS_DIR / f"{args.vocoder}.pt", args.device)[0]
    voice = {}
    if acoustic.cfg.flow:
        speakers, styles = ckpt["speakers"], ckpt["styles"]
        lang = args.lang or guess_lang(args.text)  # default voice: a real voice of this language, else the teacher's
        own = [s for s in speakers if f"_{lang}_" in s] + [s for s in ("openslr_mr" if lang == "mr" else "supertonic",) if s in speakers]
        speaker = args.speaker or (own or speakers)[0]
        style = args.style or ("neutral" if "neutral" in styles else styles[1])
        if speaker not in speakers or style not in styles:
            raise SystemExit(f"voices: {', '.join(speakers)}\nstyles: {', '.join(styles[1:])}")
        voice = {"speaker": speakers.index(speaker), "style": styles.index(style), "steps": args.steps,
                 "temperature": args.temperature, "guidance": args.guidance}
        print(f"voice {speaker}, style {style}")
    t0 = time.perf_counter()
    wave = speak(acoustic, vocoder, Vocab(ckpt["chars"]), args.text, args.lang, args.speed, **voice)
    took = time.perf_counter() - t0
    write_wav(args.out, wave)
    print(f"{args.out}: {len(wave) / SR:.1f} s of speech in {took * 1000:.0f} ms (real-time factor {took / max(len(wave) / SR, 1e-9):.3f}), "
          f"language {args.lang or guess_lang(args.text)}")


if __name__ == "__main__":
    main()
