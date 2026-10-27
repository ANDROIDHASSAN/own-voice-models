"""Checks that fail if a core idea is implemented wrong. No checkpoint or dataset needed, runs on CPU in seconds.

    python check.py
"""

import itertools
import math
import re
import tempfile
from pathlib import Path

import torch

from align import mas
from audio import HOP, N_MELS, SR, griffin_lim, log_mel, read_wav, resample, trim_silence, write_wav
from model import Acoustic, Config, length_regulate
from text import PAD, Vocab, guess_lang, normalize
from train import make_batches
from vocoder import Config as VocoderConfig
from vocoder import Discriminators, Vocoder, critic_loss, fooling_loss, spectral_loss

HERE = Path(__file__).parent
torch.manual_seed(0)
tone = lambda hz, sr, seconds=1.0: torch.sin(2 * math.pi * hz * torch.arange(int(sr * seconds)) / sr)

# Resampling keeps what fits under the new Nyquist limit and removes what does not (no aliasing).
kept, removed = resample(tone(3000, 44100), 44100, SR), resample(tone(15000, 44100), 44100, SR)
assert len(kept) == SR and abs(kept[1000:-1000].pow(2).mean() - 0.5) < 0.01 and removed[1000:-1000].pow(2).mean() < 1e-4
assert abs(torch.fft.rfft(resample(tone(440, 48000), 48000, SR)).abs().argmax().item() - 440) <= 1, "pitch must survive 48 kHz -> 22.05 kHz"

# WAV writer and reader agree; trimming removes silence but not speech.
with tempfile.TemporaryDirectory() as tmp:
    write_wav(Path(tmp) / "t.wav", tone(440, SR) * 0.5)
    back, rate = read_wav(Path(tmp) / "t.wav")
assert rate == SR and torch.allclose(torch.from_numpy(back), tone(440, SR) * 0.5, atol=1e-4)
padded = torch.cat((torch.zeros(SR), tone(440, SR) * 0.5, torch.zeros(SR)))
assert SR <= len(trim_silence(padded)) <= SR + 0.12 * SR

# Mel and Griffin-Lim: right shapes, and the rebuilt audio has roughly the spectrogram it was built from.
voiced = sum(tone(150 * k, SR) / k for k in range(1, 20)) * 0.1
mel = log_mel(voiced)
rebuilt = griffin_lim(mel)
assert mel.shape == (len(voiced) // HOP + 1, N_MELS) and len(rebuilt) == (len(mel) - 1) * HOP
assert (log_mel(rebuilt)[2:-2] - mel[2:-2]).abs().mean() < 1.0

# Loudness levelling: a whisper and a shout of the same sound come out equally loud, and nothing clips.
from data import _pcm  # noqa: E402

soft, loud = (torch.frombuffer(bytearray(_pcm(tone(200, SR) * g)), dtype=torch.int16).float() / 32768 for g in (0.01, 0.9))
assert abs(soft.pow(2).mean().sqrt() - loud.pow(2).mean().sqrt()) < 0.005 and loud.abs().max() <= 0.99

# Text: punctuation survives (it is prosody), the language guess separates the four cases.
assert normalize("नमस्ते।  आप कैसे हैं ?") == "नमस्ते. आप कैसे हैं?" and normalize("Hi — there; ok (2024)") == "hi, there, ok"
assert [guess_lang(t) for t in ("आप कैसे हैं", "तुम्ही कसे आहात", "kya haal hai bhai", "how are you today")] == ["hi", "mr", "hing", "en"]
vocab = Vocab.build(["नमस्ते, kal milte hain."] * 5 + ["z"])
assert vocab.chars[PAD] == "<pad>" and "z" not in vocab.index and vocab.encode("kalZ!") == vocab.encode("kal")

# MAS: equals brute force over every monotonic alignment, and padding in the batch changes nothing.
def brute(score: torch.Tensor) -> list[int]:
    L, T = score.shape
    cuts = max(itertools.combinations(range(1, T), L - 1),
               key=lambda c: sum(score[l, a:b].sum() for l, (a, b) in enumerate(zip((0, *c), (*c, T)))))
    return [b - a for a, b in zip((0, *cuts), (*cuts, T))]

for _ in range(25):
    L = int(torch.randint(1, 5, ()))
    score = torch.randn(L, int(torch.randint(L, 9, ())))
    assert mas(score[None], torch.tensor([L]), torch.tensor([score.shape[1]]))[0].tolist() == brute(score)
a, b = torch.randn(5, 9), torch.randn(8, 20)
both = torch.full((2, 8, 20), 7.0)  # a tempting score in the padding must be ignored
both[0, :5, :9], both[1] = a, b
found = mas(both, torch.tensor([5, 8]), torch.tensor([9, 20]))
assert found[0, :5].tolist() == brute(a) and found[0, 5:].sum() == 0 and found[1].tolist() == brute(b)

# Length regulator: each vector is repeated `duration` times.
h = torch.arange(6.0).view(1, 3, 2)
assert length_regulate(h, torch.tensor([[2, 1, 3]]))[0, :, 0].tolist() == [0, 0, 2, 4, 4, 4]

# Acoustic model: finite losses, durations that add up, and synthesis as long as its own durations say.
model = Acoustic(Config(vocab_size=40, n_layer=1, d_model=32, n_mels=N_MELS))
chars, char_lens = torch.randint(1, 40, (3, 20)), torch.tensor([20, 14, 9])
target, mel_lens, lang = torch.randn(3, 90, N_MELS), torch.tensor([90, 70, 33]), torch.tensor([0, 1, 3])
loss, parts, durations = model(chars, char_lens, lang, target, mel_lens)
assert torch.isfinite(loss) and durations.sum(1).tolist() == mel_lens.tolist()
assert all((durations[i, :n] >= 1).all() and durations[i, n:].sum() == 0 for i, n in enumerate(char_lens.tolist()))
model.eval()
out, d = model.synthesize(chars[0], 0)
slow, d_slow = model.synthesize(chars[0], 0, speed=0.5)
assert out.shape == (int(d.sum()), N_MELS) and d_slow.sum() > 1.5 * d.sum()
with torch.no_grad():  # the encoder gives a sentence the same vectors alone and padded in a batch
    alone, _ = model.encode(chars[2:, :9], char_lens[2:], lang[2:])
    batched, _ = model.encode(chars, char_lens, lang)
assert torch.allclose(alone[0], batched[2, :9], atol=1e-4), "padding leaks into the text encoder"

# Lesson 10, flow model. The first checkpoint must still load: a default Config builds no new tables.
assert not any(k.startswith(("time.", "speakers.", "styles.", "noisy_in.", "style_dec.")) for k in Acoustic(Config()).state_dict())
flow = Acoustic(Config(vocab_size=40, n_layer=1, d_model=32, n_mels=N_MELS, flow=True, n_speakers=3, n_styles=4))
spk, sty = torch.tensor([0, 2, 1]), torch.tensor([1, 3, 2])
loss, parts, durations = flow(chars, char_lens, lang, target, mel_lens, spk, sty)
assert torch.isfinite(loss) and durations.sum(1).tolist() == mel_lens.tolist()
# It can learn: with a learnable target (one spectrum per clip) the flow-matching loss must fall far below its
# start (~2: the variance of noise + data). A sign error in x_t or in u leaves it stuck. Measured with a fixed
# seed over several draws of (noise, t), because a single draw is very noisy.
learnable = torch.randn(3, 1, N_MELS).expand(3, 90, N_MELS).contiguous()


def flow_loss() -> float:
    flow.eval()
    torch.manual_seed(123)
    with torch.no_grad():
        out = sum(flow(chars, char_lens, lang, learnable, mel_lens, spk, sty)[1]["mel"] for _ in range(8)) / 8
    flow.train()
    return out


first, opt = flow_loss(), torch.optim.Adam(flow.parameters(), 3e-3)
for _ in range(150):
    loss, parts, _ = flow(chars, char_lens, lang, learnable, mel_lens, spk, sty)
    opt.zero_grad()
    loss.backward()
    opt.step()
assert flow_loss() < 0.5 * first, (first, flow_loss())
flow.eval()
# ... and SAMPLING goes the right way: starting from noise, each clip ends nearest its own learned spectrum.
# (The loss alone would also fall with u's sign flipped everywhere; only sampling exposes that.)
for i in range(3):
    out, _ = flow.synthesize(chars[i, : char_lens[i]], int(lang[i]), speaker=int(spk[i]), style=int(sty[i]))
    distance = [(out - learnable[j, 0]).abs().mean().item() for j in range(3)]
    assert min(range(3), key=distance.__getitem__) == i and distance[i] < 0.7, distance
torch.manual_seed(1)
plain, d = flow.synthesize(chars[0], 0, speaker=1, style=2)
torch.manual_seed(1)
same, _ = flow.synthesize(chars[0], 0, speaker=1, style=2, guidance=1.0)
torch.manual_seed(1)
pushed, _ = flow.synthesize(chars[0], 0, speaker=1, style=2, guidance=3.0)
assert plain.shape == (int(d.sum()), N_MELS) and torch.equal(plain, same) and not torch.allclose(plain, pushed)
assert torch.isfinite(pushed).all()

# Speakers and styles: old clips get a default voice, style 0 is reserved for "no style" (guidance).
from train import collate, voices  # noqa: E402

clips = [{"lang": "hi", "text": "a"}, {"lang": "mr", "text": "b"}, {"lang": "hi", "text": "c", "speaker": "rasa_hi_female", "style": "happy"}]
speakers, styles = voices(clips)
assert speakers == ["openslr_mr", "rasa_hi_female", "supertonic"] and styles == ["<none>", "happy", "neutral"]
for i, c in enumerate(clips):
    c.update(mel_start=i * 5, frames=5)
*_, spk_ids, sty_ids = collate(torch.zeros(15, N_MELS).numpy(), clips, [0, 1, 2], Vocab.build(["abc"] * 3), "cpu", speakers, styles)
assert spk_ids.tolist() == [2, 0, 1] and sty_ids.tolist() == [2, 2, 1]

# Vocoder: (T - 1) * hop samples out; the loss is zero only for identical audio.
vocoder = Vocoder(VocoderConfig(dim=32, n_layer=1))
wave = vocoder(torch.randn(2, 50, N_MELS))
assert wave.shape == (2, 49 * HOP) and spectral_loss(wave, wave) < 1e-4 < spectral_loss(wave, torch.randn_like(wave) * 0.1)

# GAN critics: every critic scores the clip; the critic's own loss must not train the vocoder (detached fake),
# while the fooling loss must.
critics = Discriminators()
real = torch.randn(2, 49 * HOP) * 0.1
assert all(len(out) >= 2 and torch.isfinite(out[-1]).all() for out in critics(real))
critic_loss(critics, wave, real).backward()
assert all(p.grad is None for p in vocoder.parameters()) and any(p.grad is not None for p in critics.parameters())
adv, fm = fooling_loss(critics, wave, real)
(adv + fm).backward()
assert all(p.grad is not None for p in vocoder.parameters() if p.requires_grad)

# Batching: every clip exactly once, within the frame budget.
index = [{"frames": int(f)} for f in torch.randint(100, 1500, (200,))]
groups = make_batches(index, list(range(200)), 8000)
assert sorted(i for g in groups for i in g) == list(range(200))
assert all(len(g) == 1 or len(g) * max(index[i]["frames"] for i in g) <= 8000 for g in groups)

# Glossary: every `file.py: name` it points to exists.
glossary = HERE / "GLOSSARY.md"
terms = refs = 0
if glossary.exists():
    found = re.findall(r"`(\w+\.py): ([^`]+)`", glossary.read_text())
    missing = [(f, name) for f, name in found if name not in (HERE / f).read_text()]
    assert not missing, f"glossary points at code that is not there: {missing}"
    terms, refs = len(re.findall(r"^\| \d+ \|", glossary.read_text(), flags=re.M)), len(found)
    assert terms >= 100, terms

print(f"all checks passed ({terms} glossary terms, {refs} code references)")
