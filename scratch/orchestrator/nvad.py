"""Lesson 8 - a neural VAD of our own: the network that lesson 1's energy VAD said was missing.

    python nvad.py --stt-data DIR --noise noise.npy     # train (Kaggle GPU, ~20 min)
    python nvad.py                                       # check: streaming == whole-clip, and the gate works

The energy VAD asks "is it loud?". Measured with robust.py, a fan or hiss switching on is "speech" for it,
and so is music. This one asks "is it SPEECH?": per 10 ms mel frame, a probability.

    log-mel (80, the STT's own front end) -> linear 64 -> GRU 64 -> linear 1 -> sigmoid      ~37 k parameters

A GRU and not a bidirectional or attention model: it must run as the audio arrives, frame by frame, keeping
its state, never seeing the future. That is also how Silero VAD works (with LSTMs).

Training data is made, not labelled by hand. Take a clean speech clip (FLEURS + Kathbath, the STT's data),
mark its frames as speech where they are within 35 dB of the clip's loudest frame, put silence around it,
then add MUSAN noise or music at a random SNR and the whole thing at a random loudness. The labels come from
the CLEAN clip, so the network learns "speech is there" even when noise is louder in some bands. A third of
the windows also get babble (another clip, 8-20 dB quieter) whose frames stay labelled "not speech": the
talker the agent should hear is the near, loud one, not the TV.
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from vad import FRAME, VAD

HERE = Path(__file__).parent
CKPT = HERE / "runs/nvad.pt"
OFFSET, SCALE = -8.0, 4.0  # rough centre and spread of our log-mel values: keeps the first layer's input near 0


class Net(nn.Module):
    def __init__(self, n_mels: int = 80, hidden: int = 64) -> None:
        super().__init__()
        self.inp = nn.Linear(n_mels, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    def forward(self, mel: torch.Tensor, state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """mel (B, T, 80) log-mel -> (logits (B, T), state). Pass the state back in to continue a stream."""
        h, state = self.gru(torch.relu(self.inp((mel - OFFSET) / SCALE)), state)
        return self.out(h)[..., 0], state


class NeuralVAD(VAD):
    """Drop-in for vad.VAD: the same START / END state machine and pre-roll, a network for the per-frame call.
    Each 20 ms frame gives two 10 ms mel frames; the frame is voiced if either is above `threshold`."""

    # threshold 0.95, measured with robust.py (v1 net): at 0.5 pink noise and MUSAN music took 2 and 0.3 false
    # starts a minute; at 0.85-0.95 none, same recall on sentences (clean 40/40, white / brown 5 dB 40/40).
    # hybrid 0.5, measured with robust.py (v3 net): short answers in a quiet room 27/40 -> 35/40, no new false
    # starts on pink / brown, MUSAN 0.6 -> 1.0 a minute (energy VAD alone: 4.8).
    def __init__(self, path: Path = CKPT, threshold: float = 0.95, hybrid: float | None = 0.5, **kw) -> None:
        """hybrid: also voiced when the frame is LOUD (energy VAD) and the network says at least this much."""
        super().__init__(**kw)
        self.hybrid = hybrid
        sys.path.insert(0, str(HERE.parent / "stt"))
        from audio import HOP, N_FFT, mel_filterbank

        self.hop, self.n_fft, self.fbank = HOP, N_FFT, mel_filterbank()
        self.net = Net()
        self.net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"])
        self.net.eval()
        self.threshold, self.state = threshold, None
        self.buf = np.zeros(N_FFT - HOP, dtype=np.float32)  # the overlap the next 25 ms window needs
        self.prob = 0.0

    @torch.no_grad()
    def is_voiced(self, frame: np.ndarray) -> bool:
        self.buf = np.concatenate([self.buf, frame.astype(np.float32)])
        windows = torch.from_numpy(self.buf).unfold(0, self.n_fft, self.hop)  # 2 windows of 400 for 320 new samples
        self.buf = self.buf[len(windows) * self.hop :]
        power = torch.fft.rfft(windows * torch.hann_window(self.n_fft), dim=-1).abs().pow(2)
        logits, self.state = self.net(torch.log((power @ self.fbank.T).clamp(min=1e-10))[None], self.state)
        self.prob = float(torch.sigmoid(logits).max())
        loud = super().is_voiced(frame)  # the energy VAD's answer (and it keeps learning the noise floor)
        return self.prob > self.threshold or (self.hybrid is not None and loud and self.prob > self.hybrid)


def frame_labels(clip: np.ndarray, range_db: float = 35.0) -> np.ndarray:
    """Speech frames of a CLEAN clip: within `range_db` of its loudest frame, short gaps (< 150 ms) filled."""
    energy = np.logaddexp.reduce(clip.astype(np.float32), axis=1)  # log of the frame's total power
    speech = energy > energy.max() - range_db / 10 * np.log(10)
    on = np.flatnonzero(speech)
    for a, b in zip(on[:-1], on[1:]):  # a stop consonant or a breath inside a word is still speech
        if 1 < b - a <= 15:
            speech[a:b] = True
    return speech


def synthetic_bank(minutes: float = 5.0) -> np.ndarray:
    """Log-mels of made noise: white, pink, brown and random tilts between. v1 had only MUSAN and took pink
    noise for speech (robust.py: 2 false starts a minute)."""
    sys.path.insert(0, str(HERE.parent / "stt"))
    from audio import log_mel

    rng, n, out = np.random.default_rng(0), int(16000 * 60 * minutes / 8), []
    for tilt in np.linspace(0.0, 1.5, 8):  # spectrum falls as f^-tilt/2 in amplitude: 0 white, 0.5 pink, 1 brown
        spectrum = np.fft.rfft(rng.standard_normal(n)) * np.maximum(np.fft.rfftfreq(n, 1 / 16000), 20.0) ** (-tilt / 2)
        x = np.fft.irfft(spectrum, n).astype(np.float32)
        out.append(log_mel(x / np.sqrt(np.mean(x**2)) * 0.05).numpy())
    return np.concatenate(out).astype(np.float16)


def crowd(mel: np.ndarray, index: list[dict], n: int) -> np.ndarray:
    """3-5 other speakers at once, the room's chatter, as one log-mel track."""
    track = np.full((n, mel.shape[1]), np.log(1e-10), dtype=np.float32)
    for _ in range(random.randint(3, 5)):
        o = index[random.randrange(len(index))]
        other = np.resize(mel[o["start"] : o["start"] + o["frames"]].astype(np.float32), (n, mel.shape[1]))
        track = np.logaddexp(track, other - np.log(np.exp(other).mean() + 1e-30))
    return track


def window(mel: np.ndarray, index: list[dict], bank: np.ndarray, n: int = 600, synth: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """One training example: n frames (6 s) of log-mel and their 0/1 labels."""
    if synth is not None and random.random() < 0.3:
        bank = synth
    silent = np.full((n, mel.shape[1]), np.log(1e-10), dtype=np.float32)
    x, y = silent.copy(), np.zeros(n, dtype=np.float32)
    if random.random() < 0.25:  # a quarter of windows have nobody talking to us: noise, and half the time chatter
        if random.random() < 0.5:
            x = crowd(mel, index, n) + random.uniform(-4, 0)
    else:  # 1-3 utterances, most of them SHORT: v1 saw only whole read sentences and missed one-word answers
        free = 0
        for _ in range(random.randint(1, 3)):
            e = index[random.randrange(len(index))]
            full = mel[e["start"] : e["start"] + e["frames"]].astype(np.float32)[: n - 150]
            labels = frame_labels(full)
            if random.random() < 0.7 and labels.any():  # a 0.25-1.5 s piece starting at a speech frame: "हाँ", "दिल्ली"
                on = random.choice(np.flatnonzero(labels))
                cut = slice(on, on + random.randint(25, 150))
                full, labels = full[cut], labels[cut]
            gap = random.randint(20, 120)
            if free + gap + len(full) > n:
                break
            at = free + gap
            x[at : at + len(full)], y[at : at + len(full)] = full, labels
            free = at + len(full)
        clip = x[y > 0] if y.any() else x
        if random.random() < 0.33:  # babble: someone else, quieter, anywhere in the window, labelled not-speech
            o = index[random.randrange(len(index))]
            other = mel[o["start"] : o["start"] + o["frames"]].astype(np.float32)[:n]
            at2 = random.randrange(0, n - len(other) + 1)
            back = silent.copy()
            back[at2 : at2 + len(other)] = other + (np.log(np.exp(clip).mean() / np.exp(other).mean()) - random.uniform(8, 20) / 10 * np.log(10))
            x = np.logaddexp(x, back)
    start = random.randrange(0, len(bank) - n)
    noise = bank[start : start + n].astype(np.float32)
    if y.any():
        level = np.log(np.exp(x[y > 0]).sum() / np.exp(noise[y > 0]).sum()) - random.uniform(-5, 25) / 10 * np.log(10)
    else:
        level = random.uniform(-6, 3)  # noise alone: anywhere from very quiet to loud
    x = np.logaddexp(x, noise + level) + random.uniform(-25, 10) / 10 * np.log(10)  # the whole thing at a random loudness
    return x.astype(np.float32), y


def train(args) -> None:
    sys.path.insert(0, str(HERE.parent / "stt"))
    import data as stt_data

    stt_data.DATA = Path(args.stt_data)
    mel, index = stt_data.open_split("train")
    dev_mel, dev_index = stt_data.open_split("dev")
    bank = np.load(args.noise, mmap_mode="r")
    split = int(len(bank) * 0.9)  # the last 10 % of the noise is never trained on: dev
    random.seed(0)
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = Net().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3)
    print(f"{sum(p.numel() for p in net.parameters())} parameters, {len(index)} speech clips, noise {len(bank) / 360000:.1f} h", flush=True)

    synth = synthetic_bank()

    def batch(m, idx, b, size):
        xs, ys = zip(*(window(m, idx, b, synth=synth) for _ in range(size)))
        return torch.from_numpy(np.stack(xs)).to(device), torch.from_numpy(np.stack(ys)).to(device)

    random.seed(1)
    dev = [batch(dev_mel, dev_index, bank[split:], 64) for _ in range(8)]
    random.seed(0)
    t0, best = time.perf_counter(), 0.0
    for step in range(1, args.steps + 1):
        x, y = batch(mel, index, bank[:split], 64)
        logits, _ = net(x)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % 500 == 0:
            net.eval()
            with torch.no_grad():
                pred = torch.cat([(net(dx)[0] > 0).float().flatten() for dx, _ in dev])
                true = torch.cat([dy.flatten() for _, dy in dev])
            net.train()
            acc = float((pred == true).float().mean())
            fa = float(pred[true == 0].mean())  # false alarm: noise called speech
            miss = float(1 - pred[true == 1].mean())
            print(f"step {step}  loss {loss.item():.3f}  dev acc {acc:.3f}  false alarm {fa:.3f}  miss {miss:.3f}  "
                  f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)
            if acc > best:
                best = acc
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": net.state_dict(), "dev_acc": acc, "false_alarm": fa, "miss": miss, "step": step}, args.out)


def check() -> None:
    """Streaming (20 ms at a time, carried GRU state) must give the same decisions as the whole clip at once."""
    torch.manual_seed(0)
    net = Net()
    path = HERE / "runs/_nvad_check.pt"
    path.parent.mkdir(exist_ok=True)
    torch.save({"model": net.state_dict()}, path)
    vad = NeuralVAD(path)
    sys.path.insert(0, str(HERE.parent / "stt"))
    from audio import log_mel

    wave = np.random.default_rng(0).standard_normal(16000).astype(np.float32) * 0.1
    streamed = []
    for i in range(0, len(wave), FRAME):
        vad.is_voiced(wave[i : i + FRAME])
        streamed.append(vad.prob)
    with torch.no_grad():
        whole = torch.sigmoid(net(log_mel(wave)[None])[0][0])
    # the streamed windows are not centred like log_mel's, so compare loosely: same behaviour, not same bits
    assert abs(np.mean(streamed) - float(whole.mean())) < 0.05, (np.mean(streamed), float(whole.mean()))
    labels = frame_labels(np.log(np.r_[np.full((10, 80), 1e-10), np.full((20, 80), 1.0), np.full((10, 80), 1e-10)]))
    assert labels[10:30].all() and not labels[:10].any() and not labels[30:].any()
    path.unlink()
    print("nvad: all checks passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stt-data")
    ap.add_argument("--noise")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--out", default=str(CKPT))
    a = ap.parse_args()
    train(a) if a.stt_data else check()
