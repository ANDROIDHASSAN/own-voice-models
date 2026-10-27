"""Lesson 6 - training: the loop that turns random weights into a speech recognizer.

    python train.py --overfit 16       # FIRST: can the model memorize 16 clips? (2 minutes)
    python train.py                    # full run
    python train.py --resume           # continue from runs/ckpt_last.pt

Every step:
    batch of clips -> model -> per-frame log-probabilities -> CTC loss against the transcripts
                   -> backward -> clip -> optimizer step -> next learning rate

The overfit test comes first, always. A model that cannot drive the loss to ~0 on 16 clips has a bug
(wrong lengths, wrong masking, labels and audio mismatched), and no amount of data will fix a bug. If it
can, the pipeline is wired correctly and the only open question is generalization.

What to watch:
    loss      CTC loss per target character. Early on it drops fast and then sits on a PLATEAU: the model has
              learned the cheapest trick, "always say blank". Then it suddenly starts spelling. That is normal.
    dev CER   character error rate on clips the model never trained on. The honest number.
    Train loss falling while dev CER stops improving = overfitting, the main danger with this little data.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ctc import greedy
from data import DATA, open_split
from model import Config, Encoder, Stepper, load, pick_device, save
from text import Vocab, error_rate

HERE = Path(__file__).parent
RUNS = HERE / "runs"


def make_batches(index: list[dict], ids: list[int], max_frames: int) -> list[list[int]]:
    """Group clips of similar length. A batch is padded to its longest clip, so mixing a 3 s clip with a
    19 s one would spend 85% of that row computing on padding. Each batch holds as many clips as fit in
    `max_frames` (clips x longest), so short clips come in big batches and long clips in small ones."""
    batches, current = [], []
    for i in sorted(ids, key=lambda i: index[i]["frames"]):
        if current and (len(current) + 1) * index[i]["frames"] > max_frames:
            batches.append(current)
            current = []
        current.append(i)
    return batches + [current]


def tempo(clip: torch.Tensor, factor: float) -> torch.Tensor:
    """Tempo perturbation: the same clip spoken `factor` times faster, by resampling the mel frames in time.
    With 23 hours of speech every speaker the model hears is one of a few hundred; hearing each clip at a
    slightly different speed every epoch is a cheap way to make it look like more data (Ko et al. 2015 do it
    on the waveform; on the mel it changes tempo without touching pitch)."""
    n = max(1, round(len(clip) / factor))
    return torch.nn.functional.interpolate(clip.T[None], size=n, mode="linear", align_corners=True)[0].T


def add_noise(clip: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Mix noise into a LOG-mel, without going back to audio. Two independent sounds add in power, and a mel
    band is a sum of powers, so: mixed = log(exp(speech) + g * exp(noise)), with g setting the SNR over the clip.
    (Exact on average; the cross term between the two waves averages out within a band.)"""
    gain = clip.exp().sum() / noise.exp().sum() * 10 ** (-snr_db / 10)
    return torch.logaddexp(clip, noise + gain.log())


def noisy(clips: list[torch.Tensor], bank: np.ndarray, p: float) -> list[torch.Tensor]:
    """Each clip, with probability p, gets either recorded noise / music from `bank` (MUSAN log-mels, SNR 0-20 dB)
    or babble: another clip of the batch, another speaker, played in the background (SNR 5-20 dB)."""
    out = []
    for i, c in enumerate(clips):
        if random.random() < p:
            if random.random() < 0.3 and len(clips) > 1:
                other = clips[(i + random.randrange(1, len(clips))) % len(clips)]
                n = other.repeat(len(c) // len(other) + 1, 1)[: len(c)]
                c = add_noise(c, n, random.uniform(5, 20))
            else:
                start = random.randrange(0, len(bank) - len(c))
                c = add_noise(c, torch.from_numpy(bank[start : start + len(c)].astype(np.float32)), random.uniform(0, 20))
        out.append(c)
    return out


def collate(mel: np.ndarray, index: list[dict], ids: list[int], vocab: Vocab, device: str, stretch: float = 0.0,
            bank: np.ndarray | None = None, noise_p: float = 0.0):
    """Stack clips into padded tensors. Returns (mel (B, T, 80), frame counts, targets (B, L), target lengths).
    stretch > 0: each clip gets a random tempo in [1 - stretch, 1 + stretch]; bank: noise, see noisy() (training only)."""
    clips = [torch.from_numpy(mel[index[i]["start"] : index[i]["start"] + index[i]["frames"]].astype(np.float32)) for i in ids]
    if stretch:
        clips = [tempo(c, 1 + stretch * (2 * random.random() - 1)) for c in clips]
    if bank is not None:
        clips = noisy(clips, bank, noise_p)
    labels = [torch.tensor(vocab.encode(index[i]["text"])) for i in ids]
    pad = torch.nn.utils.rnn.pad_sequence
    return (pad(clips, batch_first=True).to(device), torch.tensor([len(c) for c in clips], device=device),
            pad(labels, batch_first=True).to(device), torch.tensor([len(t) for t in labels], device=device))


@torch.no_grad()
def evaluate(model: Encoder, mel: np.ndarray, index: list[dict], ids: list[int], vocab: Vocab, device: str,
             max_frames: int) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[str, str]]]:
    """Greedy-decode the given clips. Returns {language: (CER, WER)} and one (reference, hypothesis) per language."""
    was_training = model.training
    model.eval()  # dropout off for the measurement
    refs, hyps = {}, {}
    for batch in make_batches(index, ids, max_frames):
        x, lens, _, _ = collate(mel, index, batch, vocab, device)
        log_probs, out_lens = model(x, lens)
        for i, labels in zip(batch, greedy(log_probs.cpu(), out_lens.cpu())):
            refs.setdefault(index[i]["lang"], []).append(index[i]["text"])
            hyps.setdefault(index[i]["lang"], []).append(vocab.decode(labels))
    model.train(was_training)  # back to whatever the caller had: evaluate.py and shrink.py call this on an eval model
    scores = {lang: (error_rate(refs[lang], hyps[lang], "char"), error_rate(refs[lang], hyps[lang], "word")) for lang in refs}
    return scores, {lang: (refs[lang][0], hyps[lang][0]) for lang in refs}


def lr_at(step: int, base: float, warmup: int, total: int) -> float:
    """Warmup, then cosine decay to 10%: the same schedule as ../llm/train.py."""
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=7e-4)
    # A long warmup matters more here than for the LLM. With 1000 steps the run was still on the blank plateau
    # after 7 epochs (dev CER 0.78) and was stopped; with 3000 steps and 7e-4 it left it at epoch 12 (see README).
    ap.add_argument("--warmup", type=int, default=3000, help="steps")
    ap.add_argument("--max-frames", type=int, default=16000, help="batch size in mel frames: 16000 = 160 s of audio")
    ap.add_argument("--langs", nargs="+", help="train on a subset of the languages, e.g. --langs hi")
    ap.add_argument("--stretch", type=float, default=0.0, help="tempo perturbation, e.g. 0.1 = random speed 0.9x to 1.1x")
    ap.add_argument("--subsample", type=int, default=4, choices=[2, 4], help="4 = 25 output frames / second, 2 = 50")
    ap.add_argument("--overfit", type=int, default=0, metavar="N", help="sanity check: memorize N training clips")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--name", default="ckpt", help="checkpoint name in runs/, so experiments do not overwrite each other")
    ap.add_argument("--init", help="start from this checkpoint's weights (fine-tuning), with a fresh optimizer and schedule")
    ap.add_argument("--noise", help="MUSAN log-mel bank (.npy, frames x 80) to mix into training clips, see noisy()")
    ap.add_argument("--noise-p", type=float, default=0.7, help="share of training clips that get noise")
    args = ap.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    device = pick_device()
    RUNS.mkdir(exist_ok=True)
    meta = json.loads((DATA / "meta.json").read_text())
    vocab = Vocab(meta["chars"])
    train_mel, train_index = open_split("train")
    dev_mel, dev_index = open_split("dev")
    keep = lambda index: [i for i, e in enumerate(index) if not args.langs or e["lang"] in args.langs]
    train_ids, dev_ids = keep(train_index), keep(dev_index)

    if args.overfit:  # a handful of clips, no regularization, and "dev" is the same clips: pure memorization
        train_ids = dev_ids = train_ids[:: len(train_ids) // args.overfit][: args.overfit]
        dev_mel, dev_index = train_mel, train_index
        args.warmup, args.name = 50, "overfit"
    cfg = Config(vocab_size=len(vocab), subsample=args.subsample, dropout=0.0 if args.overfit else 0.1)

    batches = make_batches(train_index, train_ids, args.max_frames)
    total_steps = args.epochs * len(batches)
    start_epoch, step, best = 0, 0, float("inf")
    if args.resume:
        model, ckpt = load(RUNS / f"{args.name}_last.pt", device)
        start_epoch, step, best = ckpt["epoch"], ckpt["step"], ckpt["best"]
    elif args.init:
        model, init = load(args.init, device)
        if init["chars"] != meta["chars"]:  # output i must mean the same character, or it fine-tunes into garbage
            raise SystemExit(f"{args.init} was trained with a different vocabulary ({len(init['chars'])} vs "
                             f"{len(meta['chars'])} characters in {DATA / 'meta.json'})")
    else:
        model = Encoder(cfg, meta["mean"], meta["std"]).to(device)
    model.train()
    bank = np.load(args.noise, mmap_mode="r") if args.noise else None

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.98))
    if args.resume:
        opt.load_state_dict(ckpt["opt"])  # Adam's running averages: without them the first steps after a resume are wild
    stepper = Stepper(device)

    hours = sum(train_index[i]["frames"] for i in train_ids) / 100 / 3600
    print(f"{device}  {model.n_params() / 1e6:.2f} M parameters  {len(train_ids)} clips = {hours:.1f} h  "
          f"{len(batches)} batches / epoch  {total_steps} steps  vocab {len(vocab)}", flush=True)

    t0 = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        random.shuffle(batches)  # the clips inside a batch stay together, the order of batches changes
        running = 0.0
        for n, batch in enumerate(batches, 1):
            for group in opt.param_groups:
                group["lr"] = lr_at(step, args.lr, args.warmup, total_steps)
            x, lens, targets, tgt_lens = collate(train_mel, train_index, batch, vocab, device, 0 if args.overfit else args.stretch,
                                                 bank, args.noise_p)
            with stepper.autocast():
                log_probs, out_lens = model(x, lens, augment=not args.overfit)
            # PyTorch's CTC (identical to ours in ctc.py, see check.py, but without a Python loop over frames).
            # reduction="mean": each clip's loss is divided by its number of characters, then averaged.
            # zero_infinity: a clip with more characters than output frames has NO valid path; skip it, do not crash.
            # Always in fp32: CTC sums probabilities over every path, far beyond fp16's range.
            loss = F.ctc_loss(log_probs.float().transpose(0, 1), targets, out_lens, tgt_lens, zero_infinity=True)
            grad_norm = stepper(loss, opt, model.parameters(), 5.0)
            step += 1
            running += loss.item()
            if step % 50 == 0:
                print(f"epoch {epoch + 1:3d}  step {step:6d}  loss {running / n:.3f}  grad-norm {grad_norm:.2f}  "
                      f"lr {opt.param_groups[0]['lr']:.2e}  {(time.perf_counter() - t0) / 60:.1f} min", flush=True)

        if args.overfit and (epoch + 1) % 25 and epoch + 1 != args.epochs:
            continue
        scores, samples = evaluate(model, dev_mel, dev_index, dev_ids, vocab, device, args.max_frames)
        cer = sum(s[0] for s in scores.values()) / len(scores)  # average over languages, so each counts equally
        print(f"== epoch {epoch + 1}  train loss {running / len(batches):.3f}  dev CER {cer:.3f}  " +
              "  ".join(f"{lang} CER {c:.3f} WER {w:.3f}" for lang, (c, w) in sorted(scores.items())))
        for lang, (ref, hyp) in sorted(samples.items()):
            print(f"   {lang} ref: {ref[:90]}\n      hyp: {hyp[:90]}", flush=True)
        extra = {"chars": meta["chars"], "epoch": epoch + 1, "step": step}
        if cer < best:  # keep the checkpoint that is best on unseen clips
            best = cer
            save(model, RUNS / f"{args.name}.pt", dev_cer=cer, **extra)
        if not args.overfit:
            save(model, RUNS / f"{args.name}_last.pt", best=best, opt=opt.state_dict(), **extra)


if __name__ == "__main__":
    main()
