# own-voice-models

A voice agent that speaks English, Hindi and Marathi using open-source models only. Everything runs on the
CPU of the machine it is on. No paid API and no API key.

| Stage | Model | Size on disk | Runs with |
|---|---|---|---|
| Orchestration | LiveKit Agents, Silero VAD, LiveKit multilingual end-of-turn model | small | `livekit-agents` |
| Speech-to-text | Meta Omnilingual ASR 300M, CTC, INT8 | 354 MB | `sherpa-onnx` |
| LLM | Gemma 3 1B (QAT, 4-bit GGUF) | 688 MB | `llama-server` (llama.cpp) |
| Text-to-speech | Supertonic 3, INT8 | 146 MB | `sherpa-onnx` |

`sherpa-onnx` and llama.cpp both run on Android and iOS, so the same model files work on a phone.

## Run it

```bash
sh get_models.sh                                   # STT + TTS models into ./models
uv venv -p 3.12 .venv && uv pip install -p .venv/bin/python -r requirements.txt
.venv/bin/python agent.py download-files           # VAD and end-of-turn weights

llama-server -hf ggml-org/gemma-3-1b-it-qat-GGUF --port 8080 -c 4096   # terminal 1: the LLM
.venv/bin/python agent.py console                                      # terminal 2: talk to it
```

`LLM_URL` points the agent at any other OpenAI-compatible server. `SPEECH_THREADS`, `TTS_STEPS`, `TTS_VOICE`
and `TTS_SPEED` tune the speech models (see `speech.py`).

## Measure it

```bash
.venv/bin/python bench.py --llm
```

It speaks one sentence per language, transcribes its own audio, and fails if the text does not come back.

Measured on an Apple M1 Pro, speech models on 2 CPU threads, sentences of 4 to 5 seconds:

| Language | TTS | STT | LLM first token | Round trip (1.00 = identical) |
|---|---|---|---|---|
| English | 857 ms | 579 ms | 412 ms (first request) | 1.00 |
| Hindi | 778 ms | 542 ms | 50 ms | 1.00 |
| Marathi | 909 ms | 693 ms | 53 ms | 0.96 |

llama-server uses the Mac's GPU by default, so the LLM numbers will be slower on a CPU-only machine.

## Known limits

- **Marathi voice.** Supertonic 3 has no Marathi. Marathi replies are read with the Hindi voice: understandable,
  with a Hindi accent.
- **Hindi in Urdu script.** The STT model cannot be told which language to expect. A short Hindi phrase
  sometimes comes back in Urdu script (seen once on synthesized audio; not yet tested on real voices).
- **Not streaming.** STT runs once the speaker stops and TTS runs one sentence at a time, so a reply starts
  after roughly STT + LLM first sentence + TTS of that sentence.
- **Small LLM.** A 1B model gets facts wrong. Set `LLM_URL` to a larger model when quality matters more than size.

## Built from scratch

[scratch/](scratch/) builds the same four parts by hand to learn how each works inside: a 12 M parameter LLM,
a 4.6 M Conformer + CTC speech recognizer, a FastSpeech / flow-matching TTS with a Vocos vocoder, and a mini
voice-agent orchestrator (VAD, end-of-turn, streaming, barge-in). Each module has measured results, a glossary
and tests.
