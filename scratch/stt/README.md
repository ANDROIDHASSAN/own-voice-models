# Kaan (Ear): a speech recognizer, from scratch

A 4.6 million parameter model that listens to Hindi, Marathi and English and writes them down. Mel spectrogram,
Conformer encoder, CTC loss, beam search, forced alignment and quantization: all written by hand in this folder.
Even the CTC forward algorithm is hand-written, and check.py matches it against PyTorch's down to the value and gradient.

This will not replace Omnilingual ASR (numbers below). Its job is to teach how ASR works on the inside.

## Reading order

| # | File | What you learn | Run |
|---|---|---|---|
| 1 | [audio.py](audio.py) | Waveform → STFT → mel filterbank → log-mel | (imported) |
| 2 | [text.py](text.py) | Character vocab, blank, normalization, WER / CER | (imported) |
| 3 | [data.py](data.py) | From FLEURS to training arrays | `python data.py` |
| 4 | [ctc.py](ctc.py) | CTC: training on (audio, text) without timings; greedy, beam search, alignment | (imported) |
| 5 | [model.py](model.py) | Conformer encoder: subsampling, attention + convolution + feed-forward, SpecAugment | (imported) |
| 6 | [train.py](train.py) | Overfit test, length bucketing, warmup, blank plateau | `python train.py --overfit 16`, then `python train.py` |
| 7 | [transcribe.py](transcribe.py) | Text from a .wav or the mic | `python transcribe.py file.wav` |
| 8 | [evaluate.py](evaluate.py) | Test CER / WER, and a comparison with a big model | `python evaluate.py --baseline 50` |
| 9 | [shrink.py](shrink.py) | Quantization: fp32 down to int2, and what each costs | `python shrink.py` |
| 10 | [xray.py](xray.py) | What the model is thinking at every 40 ms frame | `python xray.py` |

[GLOSSARY.md](GLOSSARY.md) has 152 terms, each with its place in the code. [check.py](check.py) tests the things
that break everything if they are wrong. All commands run from this folder: `../../.venv/bin/python <file>`.

## The journey of one clip

```
16 kHz waveform (12 s = 192,000 numbers)
   │  audio.py      25 ms window, 10 ms hop, FFT 400 → 80 mel bands → log
   ▼
log-mel (1200, 80)                     100 frames / second
   │  model.py      normalize → SpecAugment (training only) → 2 convs, stride 2 each
   ▼
(300, 192)                             25 frames / second, one frame = 40 ms
   │  8 times:  x = x + attention(norm(x))      sees the whole clip
   │            x = x + convolution(norm(x))    the 15 nearby frames (0.6 s)
   │            x = x + feed_forward(norm(x))
   ▼
(300, 117) log-probabilities           every character + blank at every frame
   │  ctc.py        greedy: top of each frame → merge repeats → drop blanks
   ▼
"नमस्ते दुनिया"
```

## Design decisions, and why

| Decision | Why |
|---|---|
| Characters, not BPE | 23 hours of data would give each BPE token few examples. A Devanagari character is roughly one sound |
| CTC, no decoder | All frames in one pass, no loop: fast, and it cannot hallucinate (every output is tied to a frame) |
| Encoder-only | Every frame looks both forward and back. No streaming, but higher offline accuracy |
| Conformer (attention + conv) | A consonant is a 3-frame local pattern; a word's meaning comes from context seconds away |
| 4x subsampling | Attention cost is quadratic in length. 100 → 25 frames / second = 16x cheaper |
| RoPE, RMSNorm, SwiGLU | The same pieces as in ../llm |
| SpecAugment | Only 23 hours of data: stops the model from relying on any single band or moment |
| Length bucketing | Clips of similar length in one batch, so no compute is wasted on padding |
| FLEURS | CC-BY licence, all three languages, one consistent format |

## How to explain it in an interview

**In 30 seconds:**
"I built a CTC speech recognizer from scratch for Hindi, Marathi and English: log-mel features, a 4.6 million
parameter Conformer-style encoder with 4x convolutional subsampling, SpecAugment, and my own CTC forward
algorithm, verified against PyTorch's to the gradient. It trains on 23 hours of FLEURS on a laptop, reaches
a test character error rate of 32% on Hindi, runs ten times faster than a 300 million parameter baseline,
and quantizes to 4.6 MB in int8 with no measurable loss."

| Question | Short answer | See |
|---|---|---|
| Why a mel spectrogram, not raw audio? | The ear hears more difference at low frequencies; mel bands do the same and cut 16,000 numbers/s to 8,000 | `mel_filterbank` |
| What problem does CTC solve? | We have the text, but not when each character was spoken. CTC maximizes the sum over all valid alignments | `ctc_loss` |
| Why is a blank needed? | To separate double letters like "aa", and for frames where nothing is said | `BLANK` in text.py |
| CTC's weakness? | It treats every frame as independent: no language model for spelling. Hence the high WER | WER below |
| Why the early loss plateau? | The cheapest answer is "blank on every frame". The model gets stuck there, then suddenly learns to spell | train.log epochs 1–11 |
| Why the overfit test first? | If it cannot memorize 16 clips, there is a bug: lengths, masking, labels | `--overfit` |
| Should padding change the output? | No. check.py compares one clip alone and inside a batch | check.py |
| CER vs WER? | One wrong letter makes the whole word wrong. So WER is always well above CER | text.py |
| How can WER exceed 1? | Insertions count too: the model emitted extra words | Marathi 1.09 |
| How is this different from Whisper? | Whisper is encoder-decoder (680k hours, autoregressive, can hallucinate). This is encoder + CTC | model.py docstring |

**The honest answer:** "Will this work in production?" No. In Hindi every third character is wrong. That is the
limit of 23 hours of data and 4.6 M parameters; Omnilingual is 300 M parameters trained on 1,600 languages.

## Measured results

Apple M1 Pro, MPS. 4.61 M parameters, 7,594 clips = 23.5 hours of FLEURS, 60 epochs, 208 minutes.
The best checkpoint is epoch 37 (dev CER 0.406); after that train loss kept falling (0.74 → 0.49) and dev
CER rose (0.42–0.45): overfitting.

**Test split, greedy** (`python evaluate.py`)

| Language | Clips | Hours | CER | WER |
|---|---|---|---|---|
| Hindi | 418 | 1.34 | 0.323 | 0.747 |
| Marathi | 1015 | 3.86 | 0.402 | 1.091 |
| English | 647 | 1.78 | 0.446 | 0.923 |

**More data: + Kathbath Hindi** (`python data.py --kathbath DIR`, 99,345 clips = 158.4 hours; Kaggle T4,
fp16, 15 epochs, 1.3 hours; same FLEURS dev/test). The same 4.61 M model, only 7x the data. Best epoch
14 (dev CER 0.237, was 0.406). Test split, greedy (`python evaluate.py --ckpt runs/kaggle/kathbath.pt`):

| Language | CER (23 h) | CER (158 h) | WER (23 h) | WER (158 h) |
|---|---|---|---|---|
| Hindi | 0.323 | **0.110** | 0.747 | **0.314** |
| Marathi | 0.402 | **0.193** | 1.091 | **0.623** |
| English | 0.446 | **0.369** | 0.923 | **0.860** |

Kathbath is Hindi only, yet Marathi CER halved: both use Devanagari and share similar sounds. English improved
the least, because it got no new data. Dev CER was still falling at epoch 15 (no overfitting), so more epochs
could improve it further. The orchestrator now runs on this model.

**Against a big model** (first 50 test clips per language, CPU, 2 threads; CER / WER / real-time factor)

| | Ours, greedy | Ours, beam 8 | Omnilingual ASR 300 M |
|---|---|---|---|
| Hindi | 0.333 / 0.766 / 0.023 | 0.330 / 0.763 / 0.026 | 0.095 / 0.276 / 0.238 |
| Marathi | 0.400 / 1.098 / 0.029 | 0.397 / 1.105 / 0.031 | 0.115 / 0.453 / 0.312 |
| English | 0.454 / 0.911 / 0.014 | 0.449 / 0.910 / 0.022 | 0.061 / 0.178 / 0.158 |

Our model makes 3 to 7 times more errors and is 7 to 11 times faster. Beam search without a language model
gives almost nothing (0.003 CER): CTC outputs are already peaky.

**Quantization** (`python shrink.py`, dev split)

| Format | Size | Dev CER | Difference |
|---|---|---|---|
| fp32 | 18.5 MB | 0.4060 | |
| fp16 | 9.2 MB | 0.4060 | 0 |
| int8, per-channel | 4.6 MB | 0.4064 | +0.0004 |
| int6, per-channel | 3.5 MB | 0.4069 | +0.0009 |
| int4, per-channel | 2.3 MB | 0.4177 | +0.012 |
| int4, per-tensor | 2.3 MB | 0.4452 | +0.039 |
| int3, per-channel | 1.7 MB | 0.4906 | +0.085 |
| int2, per-channel | 1.2 MB | 0.9846 | broken |

Same story as the LLM: int8 is free, per-channel is required at 4 bits, 2 bits is the end.

## What went wrong

- **Blank plateau.** The first run (1000 warmup steps) was still stuck at dev CER 0.78 after 7 epochs, the
  "all blank" answer. With 3000 warmup steps and lr 7e-4 the plateau still lasted 11 epochs, then CER started
  falling from epoch 12 (0.70 → 0.55 at epoch 16). Had the first run not been stopped it might have got out too;
  that cannot be said for sure.
- **WAV float format.** Python's `wave` module cannot read 32-bit float WAV. `read_wav` parses the chunks itself.
- **CTC gradient check.** Comparing on log-probs failed; PyTorch's CTC backward is only exact behind log-softmax.
  Comparing on logits is the right test.

## What is missing (and when to add it)

- **Language model / KenLM.** With beam search it would fix word spelling. This is where the biggest WER gain is.
- **Streaming.** The encoder sees the whole clip. Streaming needs chunked attention (the orchestrator runs STT only after the utterance ends).
- **Tempo perturbation.** `train.py --stretch 0.1` is written and tested in check.py, but no run was trained with it.
- **More data.** Kathbath (Hindi + Marathi, ~150 hours each) would improve this model the most.
