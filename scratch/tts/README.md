# Zubaan (Tongue): a text-to-speech model, from scratch

A TTS that speaks Hindi, Marathi, English and Roman Hinglish: a 4.7 M acoustic model (characters → mel) and a
2.8 M vocoder (mel → audio). Alignment (MAS), duration predictor, length regulator, Griffin-Lim and a
Vocos-style iSTFT vocoder: all written by hand in this folder.

This will not replace Supertonic. Its job is to teach how TTS works on the inside.

## Reading order

| # | File | What you learn | Run |
|---|---|---|---|
| 1 | [audio.py](audio.py) | From mel back to audio: the phase problem, Griffin-Lim, resampling | `python audio.py` |
| 2 | [text.py](text.py) | Text front-end: keeping punctuation, language id, schwa deletion | `python text.py` |
| 3 | [data.py](data.py) | Distillation: a teacher model reads, plus real Marathi recordings | `python data.py` |
| 4 | [align.py](align.py) | MAS: how many frames each character was spoken for, without anyone saying | `python align.py` |
| 5 | [model.py](model.py) | FastSpeech-style: encoder → durations → length regulator → decoder, in one pass | (imported) |
| 6 | [train.py](train.py) | Three losses (mel, prior, duration), overfit test | `python train.py --overfit 16`, then `python train.py` |
| 7 | [vocoder.py](vocoder.py) | Vocos: predict magnitude + phase, one iSTFT | `python vocoder.py` |
| 8 | [speak.py](speak.py) | Sentence → .wav | `python speak.py "नमस्ते, आप कैसे हैं?"` |
| 9 | [evaluate.py](evaluate.py) | Does a machine understand it? Student vs teacher, same judge | `python evaluate.py` |

[GLOSSARY.md](GLOSSARY.md) has 128 terms, each with its place in the code. [check.py](check.py) tests the things
that break everything if they are wrong. All commands run from this folder: `../../.venv/bin/python <file>`.

## The journey of one sentence

```
"नमस्ते, आप कैसे हैं?"  + language id (hi)
   │  text.py       normalize, keep punctuation (comma = short pause), one id per character
   ▼
20 character ids
   │  model.py      embedding + language → encoder (4 blocks)          (20, 192)
   │                duration predictor: frames per character           [3, 7, 5, 9, ...]
   │                length regulator: copy each vector that many times (170, 192)
   │                decoder (4 blocks) → linear                         (170, 80) mel
   ▼
mel, 86 frames / second
   │  vocoder.py    conv → 6 ConvNeXt blocks → magnitude (513) + phase (513) per frame
   │                → inverse STFT (nothing learned)
   ▼
22,050 Hz audio, 2 s
```

Two forward passes, no loop. The LLM needed one pass per token; that is why TTS can run faster than real time even on a CPU.

## Design decisions, and why

| Decision | Why |
|---|---|
| Characters, not phonemes | Devanagari is nearly phonetic; no G2P dictionary needed. The cost: the model has to learn schwa deletion itself (कमल = "kamal") |
| Language id | "to" is "too" in English and "toh" in Hinglish. One alphabet, two pronunciations |
| Non-autoregressive (FastSpeech) | Tacotron built frame by frame; when attention slipped, words were skipped or repeated. With durations fixed up front this cannot happen |
| MAS, no external aligner | The Glow-TTS approach: it finds durations itself during training. No tool like the Montreal Forced Aligner needed |
| Vocos, not HiFi-GAN | Stays at frame rate, with one iSTFT at the end. Cheap enough for a browser / phone CPU |
| Multi-resolution STFT loss | Two waveforms with different phase can sound the same; a sample-by-sample comparison would be wrong |
| Distillation from a teacher | One voice, clean, as much data as wanted. The smallest published TTS models are built this way |
| Real Marathi recordings | Supertonic does not speak Marathi. OpenSLR 64, ~3 hours, CC BY-SA 4.0 |
| Encoder / decoder = the STT's Block | Both text and speech need local shape and context |

**Licence:** the teacher, Supertonic 3, is Open RAIL-M, so the student is its "derivative": its use restrictions
carry over to our weights.

## How to explain it in an interview

**In 30 seconds:**
"I built a non-autoregressive TTS from scratch for Hindi, Marathi, English and Hinglish: a character-level
FastSpeech-style acoustic model that learns its own durations with Monotonic Alignment Search, a duration
predictor and length regulator, and a Vocos-style vocoder that predicts magnitude and phase and uses an
inverse STFT, trained with a multi-resolution spectral loss. Training data is 11 hours distilled from a 99
million parameter teacher plus real Marathi recordings, and I evaluate intelligibility by running the output
through a strong speech recognizer next to the teacher's audio."

| Question | Short answer | See |
|---|---|---|
| Why two parts in TTS? | Text → mel (what to say, for how long) and mel → audio (recovering phase). Separate problems | model.py, vocoder.py |
| Why is mel → audio hard? | Mel has no phase. Griffin-Lim guesses it and sounds metallic | `griffin_lim` |
| Where do durations come from? | MAS: the monotonic path that gives every frame one character, by dynamic programming | `mas` in align.py |
| Autoregressive vs non-autoregressive? | AR: frame t from frame t-1. NAR: all frames at once; fast, no skips/repeats | model.py docstring |
| Weakness of a mel L1 loss? | It averages all possible spectrograms: blurry. Hence GANs or diffusion / flow matching | train.py docstring |
| How do you change speed? | Scale the predicted durations | `synthesize(..., speed)` |
| How do you evaluate TTS? | MOS (people listen). Automatically: CER through an ASR, next to the teacher | evaluate.py |
| What is distillation? | A big model is the teacher, a small one learns to copy it. It cannot beat the teacher | data.py |
| Why 22 kHz when STT is 16 kHz? | 8 kHz is enough to understand; a natural voice needs the part of "s", "sh" above 8 kHz | audio.py |
| Why a GAN for the vocoder? | A second network that tells real from fake; it removes the last buzz | vocoder.py docstring |

## Measured results

Apple M1 Pro, MPS, sharing the GPU with another training run. Data: 8,004 clips = 10.9 hours.

**Overfit test** (`python train.py --overfit 16 --epochs 300`, 16 sentences, 3 minutes): mel loss 0.52 → 0.084,
duration error 1.02 → 0.000, predicted / real length 0.74 → 1.00. The pipeline is wired correctly.

**Acoustic model** (4.72 M, 60 epochs, 26,400 steps, 154 minutes)

| Epoch | Train mel | Dev mel | Dev duration | Length |
|---|---|---|---|---|
| 1 | 0.517 | 0.416 | 1.081 | 0.57 |
| 10 | 0.255 | 0.270 | 0.372 | 0.80 |
| 20 | 0.231 | 0.254 | 0.314 | 0.85 |
| 31 | 0.216 | **0.249** | 0.301 | 0.91 |
| 60 | 0.196 | 0.252 | 0.307 | 0.92 |

After epoch 31 dev stopped improving while train kept falling: that is the limit of 11 hours of data. `acoustic.pt`
is epoch 31. Length 0.91 means the model speaks ~9% faster than the teacher (durations are learned on a log scale,
which slightly underestimates them). `speak.py --speed 0.9` evens it out.

**Vocoder** (2.79 M, 40,000 steps, 45 minutes): dev spectral loss 3.41 (step 2,000) → 2.71 (10k) → 2.44 (40k).

**Is it understandable?** (`python evaluate.py`: 40 dev sentences per language, Omnilingual ASR 300 M as judge, CPU 2 threads)

| Language | Student CER | Teacher CER | Real-time factor |
|---|---|---|---|
| Hindi | 0.038 | 0.035 | 0.034 |
| Marathi | 0.090 | 0.043 (real recordings) | 0.027 |
| English | 0.035 | 0.016 | 0.021 |
| Hinglish | 0.295 | 0.293 | 0.029 |

In Hindi the 7.5 M student is about as intelligible as the 99 M teacher. Marathi has the biggest gap, because
there the "teacher" is a real person and there are only 2.3 hours of data. Hinglish's 0.29 is the judge's error
(Roman Hinglish has no fixed spelling): student and teacher are equal, and that is the number to look at.
Real-time factor 0.03 = ~30 times faster than real time on CPU.

This does not measure naturalness: a robotic voice can still be perfectly intelligible. For that, listen to `runs/eval_*.wav`.

## Lesson 10: from robotic to expressive

**Why it sounds robotic, measured** (10 dev sentences per language; real voice / real mel → our vocoder / full TTS):

| | Pitch movement | Mel GV | Frame-to-frame change | Harmonic clarity |
|---|---|---|---|---|
| Real | 2.67 | 1.78 | 0.526 | 3.85 |
| Real mel → vocoder | 2.61 | 1.75 | 0.495 (−6%) | 3.72 (−3%) |
| Full TTS (L1) | 2.47 (−8%) | 1.69 | 0.435 (−17%) | 3.31 (−14%) |

The biggest cause is the acoustic model: an L1 loss teaches the average of all ways of saying it, which is blurry.
The vocoder's share is small. The data is also a cause: a copy of the teacher cannot be more expressive than the teacher.

**Three changes** (all behind flags; the old `acoustic.pt` still runs as before):

| Change | Paper | Where |
|---|---|---|
| Flow matching (OT-CFM) instead of the L1 decoder: a sharp, realistic mel from noise | Matcha-TTS (Mehta 2024): MOS 3.84, VITS 3.71, FastSpeech2 3.29 | `model.py: velocity`, `train.py --flow` |
| Speaker + style/emotion embedding, classifier-free guidance (`--guidance 2` = more emotion) | F5-TTS, Ho 2022 | `model.py: synthesize` |
| Critics in front of the vocoder: multi-period + multi-resolution, hinge, feature matching | Vocos, HiFi-GAN | `vocoder.py --gan` |
| Expressive data from real people: Rasa (Hindi + Marathi, female + male, 6 emotions + styles, CC BY 4.0) | AI4Bharat 2024 | `data.py --rasa hi mr` |

**Tests** (`python check.py`):
- flow loss on a learnable target 2.21 → 0.77
- sampling brings each clip close to its own target (distance 0.45; 1.2+ from other targets)
- guidance 1 = no guidance
- the critic's loss does not train the vocoder
- the old checkpoint loads

**Smoke tests:** on real data both the flow model (12.45 M) and the GAN (dev loss steady at 2.44 at lr 1e-4) run.
A 3.4 s sentence on CPU: 10 steps RTF 0.40, **4 steps 0.17** (default), 2 steps 0.09.

**Overnight commands** (from this folder; accept Rasa's terms on HF first):

```
uv run --no-project --python ../../.venv/bin/python --with pyarrow python -u data.py --rasa hi mr > runs/data_rasa.log 2>&1
caffeinate -is ../../.venv/bin/python -u train.py --flow --d-model 256 --layers 6 --heads 4 --epochs 8 --name flow > runs/train_flow.log 2>&1
caffeinate -is ../../.venv/bin/python -u vocoder.py --gan --steps 10000 --batch 16 > runs/vocoder_gan.log 2>&1
../../.venv/bin/python evaluate.py --model flow --vocoder vocoder_gan > runs/eval_flow.log 2>&1
../../.venv/bin/python speak.py "आज मैं बहुत खुश हूँ!" --model flow --vocoder vocoder_gan --style happy --guidance 2
```

**Plan A (chosen): 24 kHz, Rasa only.** Prefix every command with `TTS_SR=24000`: 100 mel bands, data in `data24k/`,
checkpoints in `runs24k/` (the 22.05 kHz models stay as they are). Rasa's styles (from the real data): `conv`,
`wiki`, `book`, `news`, `indic`, emotions `happy`, `sad`, `anger`, `fear`, `surprise`, `disgust`, and commands
`alexa`, `bb`, `digi`, `umang`, `proper noun`. Voices: `rasa_hi_female`, `rasa_hi_male`, `rasa_mr_female`,
`rasa_mr_male`. This version has no English / Hinglish (Rasa does not have them). Training runs on Kaggle: [../kaggle/README.md](../kaggle/README.md).

```
TTS_SR=24000 ../../.venv/bin/python speak.py "आज मैं बहुत खुश हूँ!" --model flow --style happy --guidance 2
```

**Plan A's measured result** (Kaggle 2× T4, fp16; 12.47 M flow model, 27,127 clips = 49.3 hours, 30
epochs ~5.5 hours; vocoder 40k steps 30 min; GAN fine-tune 20k steps 1.8 hours). Checkpoints in
`runs24k/final/`. Best flow epoch is 16 (dev mel 0.231; flow dev loss is measured at a random t, so it
jumps between 0.23–0.32 from epoch to epoch). `evaluate.py --model final/flow --vocoder final/vocoder --n 5`
(only 5 dev sentences per voice: a rough number):

| Voice | CER: Mac, epoch 4 | CER: Kaggle, 30 epochs | CER: real recording |
|---|---|---|---|
| rasa_hi_female | 0.19 | **0.094** | 0.112 |
| rasa_hi_male | 0.29 | **0.084** | 0.212 |
| rasa_mr_female | 0.22 | **0.063** | 0.036 |
| rasa_mr_male | 0.21 | **0.136** | 0.109 |

The GAN vocoder does not change CER (0.094 / 0.084 / 0.067 / 0.140): it is not for clarity but for the metallic
sound, and its spectral dev loss went 2.55 → 2.77 (normal for a GAN: sharp detail instead of the average). The
decision is made by listening. RTF on CPU ~0.15. Samples: `runs24k/final/samples/`.

Honestly: ElevenLabs and Bulbul are built on thousands of hours and GPU clusters. This is ~50 hours of data and
two free T4s. Flow matching and real data should take the voice well beyond robotic, but not to that level.
Only `evaluate.py` and listening will confirm the numbers.

## What is missing (and when to add it)

- **The flow model and GAN vocoder are not trained yet.** The code and tests are ready; use the commands above.
- **Pitch / energy predictor.** FastPitch predicts them separately; here the flow decoder learns them itself.
- **Numbers → words.** Sentences with digits were removed from the data; it cannot say "25".
- **Multi-speaker.** One voice (the teacher) and one female Marathi speaker.
