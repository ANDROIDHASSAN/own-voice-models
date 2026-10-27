# Drills — write the core function yourself

Each `dNN_*.py` is a stub: the real logic in the function body is `raise NotImplementedError`, with the real
shape and contract in the docstring. Fill it in, then check it against your own code in `../stt` or `../tts`
(or, for theory drills, against toy reference implementations).

```
.venv/bin/python dNN_*.py              # checks your version — fails until you implement it
.venv/bin/python dNN_*.py --solution   # checks solutions/dNN_*.py — always passes, read it if stuck
```

Run from this directory. Each one is self-contained (stdlib + torch only, no new dependencies).

| Drill | Builds | Checked against | ~Time |
|---|---|---|---|
| `d01_log_mel.py` | log-mel spectrogram in plain numpy | `stt/audio.py` | 15 min |
| `d02_padding_mask.py` | frame mask + masked bidirectional attention | `stt/model.py` | 10 min |
| `d03_ctc_loss.py` | CTC collapse rule + forward algorithm | `stt/ctc.py` vs `F.ctc_loss` | 20 min |
| `d04_ctc_decoding.py` | greedy decoding + prefix beam search | `stt/ctc.py` | 25 min |
| `d05_wer.py` | edit distance, WER/CER | `stt/text.py` | 10 min |
| `d06_rnnt_greedy.py` | transducer (RNN-T) greedy decoding loop | toy encoder/predictor/joiner (theory: not built in `stt/`) | 15 min |
| `d07_mas.py` | Monotonic Alignment Search | `tts/align.py` | 15 min |
| `d08_length_regulator_flow.py` | length regulator, conditional flow matching pair, Euler sampling | `tts/model.py` | 10 min |
| `d09_griffin_lim_mr_stft.py` | one Griffin-Lim iteration, spectral convergence loss | `tts/audio.py`, `tts/vocoder.py` | 15 min |
| `d10_rvq.py` | vector quantization + residual VQ (RVQ) | toy codebooks (theory: audio codecs not built here) | 10 min |

Run them all:

```
for f in d0*.py; do echo == $f ==; .venv/bin/python "$f" --solution; done
```

Companion videos: `../../../videos/fundamentals/` (F1–F11), one per drill plus a rapid-fire recap.
