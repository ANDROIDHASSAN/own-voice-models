# Voice AI stack, from scratch

Building all four parts of a voice agent by hand, so each one is understood from the inside. The agent in the
parent folder (`../agent.py`) uses ready-made models; here the same pieces are built by hand.

| # | Module | What gets built | Status |
|---|---|---|---|
| 1 | [llm/](llm/) | Tokenizer + transformer + training + LoRA + quantization | done |
| 2 | [stt/](stt/) | Waveform → mel spectrogram → Conformer encoder → CTC → text | done (Hindi test CER 0.110 on 158 h; was 0.323 on 23 h) |
| 3 | [tts/](tts/) | Characters → durations (MAS) → mel → Vocos vocoder → audio | done (Hindi CER 0.038, teacher 0.035; 30x real time on CPU). Lesson 10: 24 kHz flow + emotions, 4 real voices, Hindi CER 0.09 (real recording 0.11) |
| 4 | [orchestrator/](orchestrator/) | A mini LiveKit of my own: VAD, end-of-turn, barge-in, streaming | done, runs with all three real models (first audio 1.6–1.8 s; new STT + flow TTS wired in) |

To read from the very beginning: [docs/Awaaz-AI-Shuru-Se.pdf](docs/Awaaz-AI-Shuru-Se.pdf) (21 steps), the full
guide [docs/Voice-AI-Zero-Se.pdf](docs/Voice-AI-Zero-Se.pdf), 60 interview questions in [docs/questions.md](docs/questions.md),
and the list of concepts in [CONCEPTS.md](CONCEPTS.md). The PDF guides are written in Hinglish.

Every module has: files in lesson order, a `GLOSSARY.md` (100+ terms, with pointers into the code), a `check.py`,
and a README with a "how to explain it in an interview" section.

## Three angles that run through every module

- **Indic-first.** The data is Hindi, Marathi and Hinglish. Devanagari and code-switching from day one.
- **Glass-box.** A tool to look inside every model (`xray.py` in the LLM).
- **Phone-size.** A size budget for every model, with measured quantization numbers.

## Two tracks

- **Scratch** (this folder): for learning. Small models, toy quality.
- **Fine-tune**: improving ready-made models (Gemma, Omnilingual ASR) on my own data. Usable quality.
  The LoRA idea is in `llm/lora.py`; the same thing applies to a bigger model.

WebRTC (ICE, DTLS, SRTP, SFU) will not be built from scratch: that is networking, not voice AI. The
orchestrator glossary covers its terms as theory.
