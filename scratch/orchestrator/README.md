# The brain's conductor: a mini voice-agent framework, from scratch

Joining three models (../stt, ../llm, ../tts) into one conversation: VAD, end-of-turn, streaming, barge-in and
cancellation. A small version of what LiveKit Agents and Pipecat do, in asyncio and numpy. No framework.

## Reading order

| # | File | What you learn | Run |
|---|---|---|---|
| 1 | [vad.py](vad.py) | Energy VAD: noise floor, hysteresis, pre-roll | `python vad.py` |
| 2 | [turn.py](turn.py) | End-of-turn: silence does not mean "my turn"; completeness from the LLM | `python turn.py` |
| 3 | [models.py](models.py) | Three models behind three plain functions (plugin interface), Devanagari → Roman | (imported) |
| 4 | [pipeline.py](pipeline.py) | State machine, streaming, sentence chunking, barge-in, generation counter | (imported) |
| 5 | [agent.py](agent.py) | A real conversation from the mic or a .wav, with per-stage latency | `python agent.py --wav q.wav` |
| 6 | [live.py](live.py) + [live.html](live.html) | A LiveKit-style live call in the browser: mic over WebSocket, the same Agent, barge-in | `python live.py` → http://127.0.0.1:8791 |
| 6b | [teacher.py](teacher.py) | A lesson script in the LLM's seat: the teacher speaks first, asks questions, checks answers. First audio ~0.7 s (5 questions, simulated student, 5/5 caught) | `python teacher.py` |
| 6c | [robust.py](robust.py) | Noise test: false VAD starts, whether long and short answers are caught, STT CER in noise | `python robust.py --clips DIR --refs refs.tsv --musan noise.npy` |
| 8 | [nvad.py](nvad.py) | My own neural VAD (GRU, 30 k params, streaming), hybrid with the energy VAD | `python nvad.py` |
| 9 | [eot.py](eot.py) | My own turn detector (char GRU, 83 k params): is the sentence complete or not | `python eot.py` |
| 7 | [web.py](web.py) + [web.html](web.html) | Push-to-talk test page, and a TTS-only panel (voice, style, guidance) | `python web.py` → http://127.0.0.1:8790 |

[GLOSSARY.md](GLOSSARY.md) has 115 terms (including theory terms such as WebRTC, SFU, AEC). [check.py](check.py)
swaps the three models for fake models with known delays, so the state machine, barge-in and cancellation are
tested in a few seconds without any checkpoint.

## The journey of one turn

```
mic, 20 ms frames
   │  vad.py         sound 12 dB above the floor for 100 ms → START (with 300 ms pre-roll)
   │                 500 ms of silence → END
   ▼
utterance (samples)
   │  stt            → "मेरा ऑर्डर कब आएगा"                         in a worker thread
   │  turn.py        romanize → LLM: P(next token is "\n" . ? !)
   │                 looks complete → wait 0.3 s;  incomplete → wait up to 1.5 s
   ▼
LISTENING → THINKING
   │  llm            tokens stream in
   │  sentences()    passed on as soon as the first full sentence arrives
   │  tts            speaks sentence 1 while the LLM writes sentence 2
   ▼
THINKING → SPEAKING → (done) → LISTENING
           user talks over it → stop playback, bump the reply number, throw away everything else
```

## Design decisions, and why

| Decision | Why |
|---|---|
| Three states, one generation counter | An interrupted reply must never speak later: every stage checks that its number is still current |
| Models in worker threads | The models block. With the event loop free, the mic keeps being read and barge-in can be heard |
| Sentence chunking | TTS needs a whole sentence (intonation). Waiting for the whole reply = a long silence |
| End-of-turn from the LLM | VAD alone either cuts people off or feels slow. LiveKit's turn detector combines the same two signals |
| Energy VAD, not Silero | A baseline that shows what a neural VAD is for (it mistakes music, horns, typing for speech) |
| Romanizer | STT writes Devanagari, the LLM reads Roman Hinglish. One table and one rule (drop the final "a") |
| Plugin functions | `stt(samples) → text`, `llm(text) → iterator`, `tts(text) → audio`. Plug in Gemma and nothing else changes |

## How to explain it in an interview

**In 30 seconds:**
"I wrote the orchestration layer of a voice agent from scratch in asyncio: an energy VAD with an adaptive
noise floor and hysteresis, an end-of-turn detector that asks my own LLM how likely the transcript is to be
complete and waits 0.3 or 1.5 seconds accordingly, a three-state machine with streaming sentence-level TTS,
and barge-in with a generation counter so an interrupted reply can never speak later. The models run in
worker threads so the agent keeps listening while it thinks, and the logic is tested with fake models of
known latency."

| Question | Short answer | See |
|---|---|---|
| How do you measure voice-agent latency? | User stops talking → agent's first audio. "Time to first audio", not total time | `Turn.first_audio_ms` |
| Where does the latency go? | endpoint wait + STT + LLM's first sentence + TTS of the first sentence | agent.py's report |
| Why streaming? | The second sentence is written while the first is spoken; the user only waits for the first | `sentences` |
| How does barge-in work? | VAD START during THINKING/SPEAKING: stop playback, cancel the task, generation += 1 | `_interrupt` |
| Why doesn't the old reply speak later? | Every stage checks its own generation number | `_respond` |
| Why VAD hysteresis? | A per-frame decision is jumpy: "s" is nearly silent, a door slam is loud. Separate rules for start and end | vad.py |
| Why pre-roll? | START arrives 100 ms late on purpose; the first letter must not be cut | `pre_roll_ms` |
| What goes wrong in end-of-turn? | False cutoff: the user was still talking and the agent spoke | `false_cutoffs` |
| Why is echo a problem? | The mic hears the speaker and the agent interrupts itself. The browser's AEC removes it | agent.py docstring |
| What if a blocking model runs on the event loop? | The agent goes deaf while thinking: barge-in will not work | `asyncio.to_thread` |

**The honest answer:** "Does this have WebRTC?" No. It talks through the mic and speaker in one process. Over a
network (ICE, DTLS, SRTP, SFU) you need a framework like LiveKit; the glossary covers those as theory.

## Measured results

**End-of-turn** (`python turn.py`, 10 hand-written cases; an illustration, not a benchmark)

| Endpointer | False cutoffs | Average latency at a real end |
|---|---|---|
| VAD only, 0.3 s | 100% | 0.30 s |
| VAD only, 1.5 s | 0% | 1.50 s |
| VAD + our LLM, 0.3 / 1.5 s | 0% | 1.02 s |

The LLM gave incomplete sentences 0.39 or less ("mera order number hai" 0.389), but also gave some complete
sentences low scores ("haan theek hai" 0.014): the 12 M model is sure about questions (ending in ?), not about
short answers. Hence an average of 1.02 s, not 0.3 s.

**VAD** (`python vad.py`, synthetic clip): speech starts at 1.00 s, START at 1.08 s, and the pre-roll keeps the
start. A 0.2 s pause does not split the utterance. Learned noise floor: -51 dB.

**With all three real models** (`python agent.py --wav q.wav`, one FLEURS Hindi test clip, M1 Pro CPU;
full output in `runs/e2e.log` and `runs/e2e_barge_in.log`)

| | Endpoint wait | STT | LLM first sentence | TTS first sentence | First audio |
|---|---|---|---|---|---|
| One turn | 1350 ms | 153 ms | 52 ms | 123 ms | 1776 ms |
| Barge-in: first reply, cut at 0.5 s | 1396 ms | 105 ms | 22 ms | 62 ms | 1612 ms |
| Barge-in: new utterance | 1467 ms | 33 ms | 11 ms | 78 ms | 1640 ms |
| `--llm echo` (the new 24 kHz flow TTS speaks) | 1422 ms | 78 ms | 0 ms | 232 ms | 1736 ms |

STT is now the 158-hour model (`stt/runs/kaggle/kathbath.pt`). The same clip used to come out as "एबग बवरदल
जर्ना ति कार्याल के बाहार पटा", now "एक बम गवर्नर जर्नल की कार्यालय के बाहर पटाता" (reference: "एक बम गवर्नर
जनरल के कार्यालय के बाहर फटा था"). There are two TTS models: Devanagari goes to the new flow model (24 kHz, Rasa
female, `conv`), Roman Hinglish to the old model, because Rasa has no Hinglish (`models.py: load_tts`). Our LLM
writes in Roman, so a normal turn still speaks with the old voice; `--llm echo` plays the new flow TTS. The flow
TTS is ~100 ms slower than the old one on the first sentence because of its 4 Euler steps. All the models together
take under 350 ms; the rest of the wait is end-of-turn (1.35–1.47 s): the turn detector and the Hinglish LLM
still find Devanagari text incomplete. The LLM's replies ("haan, mujhe toh bilkul pasand nahi hai ...") show the
limit of a 12 M model: the tone is right, the meaning is not. Sampling is random, so each run replies differently.
In barge-in the reply was cut, the old reply was not spoken again, and the new utterance got an answer.

**With fake models** (`python check.py`): first audio 729 ms after the user stops. Five scenarios pass: full
turn, barge-in, talking again after an interrupt, last sentence, model error.

**In noise: my own neural VAD, turn detector, noise-trained STT** (`python robust.py`, 40 FLEURS Hindi test
clips; noise made in-house: white / pink / brown / babble = 4 other people; the held-out 10 % of MUSAN, never
heard in training. Kaggle T4: STT 8 epochs 46 min, VAD 12k steps 73 min, turn detector 2.4 min.)

| | Energy VAD | Neural v3 + hybrid |
|---|---|---|
| Noise only, false starts / min: pink, brown | 2.0, 2.0 | **0, 0** |
| Noise only, false starts / min: babble, MUSAN | 4.0, 4.8 | **2.0, 1.0** |
| Long sentences caught, noise 5-10 dB (white, pink, brown, MUSAN) | 9-25 / 40 | **33-40 / 40** |
| Short answers (0.4-0.8 s) caught: quiet / MUSAN 10 dB / brown 10 dB | 40 / 10 / 0 | 35 / **19** / **34** |

The v1 VAD was trained only on long sentences and never fired on one-word answers ("दिल्ली": p 0.52, one frame);
v3 added short snippets, chatter-only windows and synthetic noise. Hybrid: a frame is voiced when the network
is above 0.95, or the sound is loud and the network is above 0.5.

| STT CER | 158 h | 158 h + noise |
|---|---|---|
| Clean | 0.097 | **0.086** |
| Babble 10 dB / 5 dB | 0.400 / 0.723 | **0.154 / 0.270** |
| White 5 dB | 0.406 | **0.256** |
| MUSAN 5 dB | 0.258 | **0.155** |

If STT confidence (average probability of the frames that emit characters) is below 0.5, drop the transcript:
hiss / fan 0.24-0.40, real speech 0.49-0.95. Turn detector, Hindi + Marathi dev (FLEURS): accuracy 0.902
(scratch LLM 0.500; it waited 1.5 s on Devanagari every time); complete sentences answered at once 98 %, wrongly
cut on incomplete ones 17.7 %. Live call (`live.py`, simulated): first audio scratch 684 ms (was 1667), teacher
683-758 ms, barge-in worked.

## What is missing (and when to add it)

- **Background conversation / TV.** `scenarios.py` (live call, real speed): 4/6 pass. Talking during the
  greeting, a turn in MUSAN noise, a 0.3 s cough (the agent did not stop), real barge-in: pass. TV 8-12 dB
  quieter: fail, it answered the TV too. The level gate (`pipeline.py: _near`, a voice 10 dB quieter than the
  user = background) only works on large gaps; there is no reference before the user first speaks. LiveKit +
  Silero also answers the TV; a speaker-aware model like Krisp BVC fixes it. That is the next step.
- **Interruption rules** (LiveKit-style): speech shorter than 0.5 s does not stop the agent and gets no reply
  (`min_interrupt`, counted in voiced frames, not from the VAD's 0.5 s hangover).
- **Turn detector for short answers.** It gives "हाँ ठीक है" 0.08: the training data was stories and read
  sentences, not conversation. LiveKit's detector also reads the previous question; ours only reads the user's text.
- **Noise cancellation (denoiser).** The browser's `noiseSuppression` is used; no model of my own yet.
- **Echo cancellation.** Headphones needed. The browser gives `echoCancellation: true` for free.
- **Streaming STT.** STT runs only after the utterance ends (the ../stt encoder sees the whole clip).
