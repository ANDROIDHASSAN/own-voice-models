"""Checks that fail if a core idea is implemented wrong. No checkpoint or dataset needed, runs on CPU in seconds.

    python check.py
"""

import math
import re
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from audio import HOP, N_FFT, N_MELS, SR, log_mel, mel_filterbank, power_spectrogram, read_wav
from ctc import align, beam_search, ctc_loss, greedy
from model import Config, Encoder, Subsample, spec_augment
from text import BLANK, Vocab, edit_distance, error_rate, normalize
from train import make_batches, tempo

HERE = Path(__file__).parent
torch.manual_seed(0)

# WAV reader: 16-bit integer and 32-bit float files of the same sound decode to the same samples.
samples = (np.sin(np.arange(800) / 5) * 0.5).astype(np.float32)
header = lambda tag, bits, size: (b"RIFF" + struct.pack("<I", 36 + size) + b"WAVEfmt " +
                                  struct.pack("<IHHIIHH", 16, tag, 1, SR, SR * bits // 8, bits // 8, bits) + b"data" + struct.pack("<I", size))
as_float, rate = read_wav(header(3, 32, 3200) + samples.tobytes())
as_int, _ = read_wav(header(1, 16, 1600) + (samples * 32768).astype("<i2").tobytes())
assert rate == SR and np.allclose(as_float, samples) and np.allclose(as_int, samples, atol=1e-4)

# Spectrogram: our frame-by-frame STFT is the standard one, and gives 100 frames per second.
wave = torch.randn(SR * 2)
reference = torch.stft(wave, N_FFT, HOP, window=torch.hann_window(N_FFT), return_complex=True).abs().pow(2).T
assert torch.allclose(power_spectrogram(wave), reference, atol=1e-3, rtol=1e-4)
assert log_mel(wave).shape == (len(wave) // HOP + 1, N_MELS)
bank = mel_filterbank()
assert bank.shape == (N_MELS, N_FFT // 2 + 1) and (bank.sum(1) > 0).all(), "a mel band with no FFT bin would always output log(0)"
tone = log_mel(torch.sin(2 * math.pi * 1000.0 * torch.arange(SR) / SR)).mean(0)
assert abs((bank.argmax(1) * SR / N_FFT)[tone.argmax()] - 1000.0) < 100, "a 1 kHz tone must light up the band around 1 kHz"

# Text: normalization keeps matras and drops punctuation; the vocabulary round-trips; the metrics are right.
assert normalize("Boss को, बता देना!  OK?") == "boss को बता देना ok"
vocab = Vocab.build(["नमस्ते दुनिया hello"] * 5 + ["z"])
assert "z" not in vocab.index and vocab.chars[BLANK] == "<blank>"
assert vocab.decode(vocab.encode("नमस्ते hello")) == "नमस्ते hello" and vocab.encode("zzz") == []
assert edit_distance("kitten", "sitting") == 3 and edit_distance([], [1, 2]) == 2
assert error_rate(["a b c d"], ["a x c"]) == 0.5 and error_rate(["abcd"], ["abcd"], "char") == 0.0

# CTC loss: our forward algorithm equals PyTorch's, value and gradient, including repeated letters and padding.
logits = torch.randn(4, 60, 12, requires_grad=True)
lp = logits.log_softmax(-1)
targets = torch.randint(1, 12, (4, 14))
targets[0, :4] = 5  # "aaaa": every pair needs a blank between
in_lens, tgt_lens = torch.tensor([60, 60, 47, 31]), torch.tensor([14, 9, 14, 1])
ours = ctc_loss(lp, in_lens, targets, tgt_lens)
theirs = F.ctc_loss(lp.transpose(0, 1), targets, in_lens, tgt_lens, reduction="none")
assert torch.allclose(ours, theirs, atol=1e-3), (ours, theirs)
# Compared at the logits: PyTorch's CTC backward is only exact once it has passed back through the log-softmax.
(g_ours,), (g_theirs,) = torch.autograd.grad(ours.sum(), logits, retain_graph=True), torch.autograd.grad(theirs.sum(), logits)
assert torch.allclose(g_ours, g_theirs, atol=1e-4), "CTC gradient differs from PyTorch"

# Decoding: the two collapse rules; beam search never returns a less likely text than greedy.
path = F.one_hot(torch.tensor([1, 1, 0, 1, 2, 2, 0]), 3).float().log().clamp(min=-50)
assert greedy(path[None], torch.tensor([7])) == [[1, 1, 2]] and greedy(path[None], torch.tensor([2])) == [[1]]
score = lambda lp1, text: -ctc_loss(lp1[None], torch.tensor([len(lp1)]), torch.tensor([text or [0]]), torch.tensor([len(text)])).item()
for clip in torch.randn(5, 12, 4).log_softmax(-1):
    assert score(clip, beam_search(clip, beam=16)) >= score(clip, greedy(clip[None], torch.tensor([12]))[0]) - 1e-4

# Forced alignment: one symbol per frame, and collapsing it gives back the text.
frames = align(lp[0].detach(), targets[0, :14].tolist())
merged = torch.unique_consecutive(torch.tensor(frames))
assert len(frames) == 60 and merged[merged != BLANK].tolist() == targets[0].tolist()

# Model: a clip gets the same output alone and padded inside a batch with a longer clip.
model = Encoder(Config(vocab_size=30, n_layer=2, d_model=32, n_head=2)).eval()
short, long = torch.randn(317, N_MELS), torch.randn(1000, N_MELS)
batch = torch.zeros(2, 1000, N_MELS)
batch[0, :317], batch[0, 317:], batch[1] = short, 99.0, long  # garbage in the padding must not matter
with torch.no_grad():
    alone, n = model(short[None], torch.tensor([317]))
    both, lens = model(batch, torch.tensor([317, 1000]))
assert torch.allclose(alone[0], both[0, : n[0]], atol=1e-4), "padding leaks into the output"
assert lens.tolist() == [Subsample.out_len(Subsample.out_len(torch.tensor(t))).item() for t in (317, 1000)] and both.shape[1] == lens[1]
assert torch.allclose(both.exp().sum(-1), torch.ones(2, both.shape[1]), atol=1e-4), "outputs must be probabilities"

# SpecAugment only zeroes things, and never touches the shape.
x = torch.randn(2, 200, N_MELS)
masked = spec_augment(x, torch.tensor([200, 120]))
assert masked.shape == x.shape and ((masked == x) | (masked == 0)).all() and (masked == 0).any()

# Tempo perturbation: 10% faster = 10% fewer frames; factor 1 changes nothing.
clip = torch.randn(300, N_MELS)
assert tempo(clip, 1.1).shape == (273, N_MELS) and torch.allclose(tempo(clip, 1.0), clip, atol=1e-5)

# Batching: every clip exactly once, and no batch larger than the frame budget.
index = [{"frames": int(f)} for f in torch.randint(100, 2000, (300,))]
groups = make_batches(index, list(range(300)), 16000)
assert sorted(i for g in groups for i in g) == list(range(300))
assert all(len(g) == 1 or len(g) * max(index[i]["frames"] for i in g) <= 16000 for g in groups)

# Glossary: every `file.py: name` it points to exists, so the glossary cannot drift from the code.
glossary = HERE / "GLOSSARY.md"
terms = refs = 0
if glossary.exists():
    found = re.findall(r"`(\w+\.py): ([^`]+)`", glossary.read_text())
    missing = [(f, name) for f, name in found if name not in (HERE / f).read_text()]
    assert not missing, f"glossary points at code that is not there: {missing}"
    terms, refs = len(re.findall(r"^\| \d+ \|", glossary.read_text(), flags=re.M)), len(found)
    assert terms >= 100, terms

print(f"all checks passed ({terms} glossary terms, {refs} code references)")
