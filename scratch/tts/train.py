"""Lesson 6 - training the acoustic model.

    python train.py --overfit 16       # FIRST: can it memorize 16 sentences? (a few minutes)
    python train.py                    # full run
    python train.py --resume

Every step:
    batch of (characters, mel) -> encoder -> MAS finds the durations -> decoder draws the mel
                               -> mel loss + prior loss + duration loss -> backward -> step

What to watch:
    mel     L1 distance between the drawn and the real spectrogram, in standard deviations. The main number.
    prior   how well the per-character guesses match the audio. When it drops, MAS has found real alignments.
    dur     error of the duration predictor in log-frames. 0.1 means it is typically within about 30%.
    length  on dev: total predicted length / true length. Near 1.00 means the speaking rate is right.

A lower mel loss is not automatically better speech. L1 rewards the AVERAGE of all plausible spectrograms,
which is a blurred one; that is the known weakness of this model family. Listen to runs/sample_*.wav
(Griffin-Lim, so metallic, but the words and rhythm are already audible) and trust evaluate.py.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from audio import HOP, N_MELS, RUNS_DIR, SR, griffin_lim, write_wav
from data import DATA, open_split
from model import Acoustic, Config, Stepper, load, pick_device, save
from text import LANGS, Vocab

HERE = Path(__file__).parent
RUNS = RUNS_DIR


def make_batches(index: list[dict], ids: list[int], max_frames: int) -> list[list[int]]:
    """Clips of similar length together, as many as fit in `max_frames` (see ../stt/train.py)."""
    batches, current = [], []
    for i in sorted(ids, key=lambda i: index[i]["frames"]):
        if current and (len(current) + 1) * index[i]["frames"] > max_frames:
            batches.append(current)
            current = []
        current.append(i)
    return batches + [current]


def speaker_of(entry: dict) -> str:
    """Clips from before Rasa carry no speaker: the teacher's one voice, and the OpenSLR Marathi reader."""
    return entry.get("speaker") or ("openslr_mr" if entry["lang"] == "mr" else "supertonic")


def voices(index: list[dict]) -> tuple[list[str], list[str]]:
    """(speakers, styles) seen in the data. Style 0 is "<none>": the hidden style of classifier-free guidance."""
    return sorted({speaker_of(e) for e in index}), ["<none>"] + sorted({e.get("style", "neutral") for e in index})


def collate(mel: np.ndarray, index: list[dict], ids: list[int], vocab: Vocab, device: str,
            speakers: list[str] | None = None, styles: list[str] | None = None):
    """Returns (chars (B, L), char lengths, language ids, mel (B, T, 80), mel lengths, speaker ids, style ids).
    The last two are None when the model has no speaker / style tables."""
    pad = torch.nn.utils.rnn.pad_sequence
    chars = [torch.tensor(vocab.encode(index[i]["text"])) for i in ids]
    mels = [torch.from_numpy(mel[index[i]["mel_start"] : index[i]["mel_start"] + index[i]["frames"]].astype(np.float32)) for i in ids]
    tensor = lambda values: torch.tensor(values, device=device)
    return (pad(chars, batch_first=True).to(device), tensor([len(c) for c in chars]),
            tensor([LANGS.index(index[i]["lang"]) for i in ids]),
            pad(mels, batch_first=True).to(device), tensor([len(m) for m in mels]),
            tensor([speakers.index(speaker_of(index[i])) for i in ids]) if speakers else None,
            tensor([styles.index(index[i].get("style", "neutral")) for i in ids]) if styles else None)


@torch.no_grad()
def evaluate(model: Acoustic, mel: np.ndarray, index: list[dict], ids: list[int], vocab: Vocab, device: str, max_frames: int,
             speakers: list[str] | None = None, styles: list[str] | None = None):
    """Dev losses, plus how far the PREDICTED total length is from the real one."""
    was_training = model.training
    model.eval()
    totals, predicted, actual = {"mel": 0.0, "prior": 0.0, "dur": 0.0}, 0.0, 0.0
    batches = make_batches(index, ids, max_frames)
    for batch in batches:
        chars, char_lens, lang, x, lens, spk, sty = collate(mel, index, batch, vocab, device, speakers, styles)
        _, parts, _ = model(chars, char_lens, lang, x, lens, spk, sty)
        for k in totals:
            totals[k] += parts[k] / len(batches)
        h, real = model.encode(chars, char_lens, lang, spk, sty)
        predicted += (model.duration(h, real).exp().round().clamp(min=1) * real).sum().item()
        actual += lens.sum().item()
    model.train(was_training)
    return totals, predicted / actual


def lr_at(step: int, base: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--max-frames", type=int, default=8000, help="batch size in mel frames: 8000 = 93 s of audio")
    ap.add_argument("--overfit", type=int, default=0, metavar="N")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--name", default="acoustic")
    ap.add_argument("--flow", action="store_true", help="lesson 10: flow-matching decoder + speaker and style tables")
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4, help="blocks in the encoder, and again in the decoder")
    ap.add_argument("--heads", type=int, default=2)
    args = ap.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    device = pick_device()
    RUNS.mkdir(exist_ok=True)
    meta = json.loads((DATA / "meta.json").read_text())
    vocab = Vocab(meta["chars"])
    _, train_mel, train_index = open_split("train")
    _, dev_mel, dev_index = open_split("dev")
    # MAS needs at least one frame per character; a clip with fewer frames than characters cannot be aligned.
    usable = lambda index: [i for i, e in enumerate(index) if len(vocab.encode(e["text"])) < e["frames"] <= 1500]
    train_ids, dev_ids = usable(train_index), usable(dev_index)
    if args.overfit:
        train_ids = dev_ids = train_ids[:: len(train_ids) // args.overfit][: args.overfit]
        dev_mel, dev_index = train_mel, train_index
        args.warmup, args.name = 50, "overfit"

    batches = make_batches(train_index, train_ids, args.max_frames)
    total_steps = args.epochs * len(batches)
    start_epoch, step, best = 0, 0, float("inf")
    speakers, styles = voices(train_index) if args.flow else (None, None)
    if args.resume:
        model, ckpt = load(RUNS / f"{args.name}_last.pt", device)
        start_epoch, step, best = ckpt["epoch"], ckpt["step"], ckpt["best"]
        speakers, styles = ckpt.get("speakers"), ckpt.get("styles")
    else:
        cfg = Config(vocab_size=len(vocab), n_langs=len(LANGS), n_mels=N_MELS, dropout=0.0 if args.overfit else 0.1,
                     d_model=args.d_model, n_layer=args.layers, n_head=args.heads, flow=args.flow,
                     n_speakers=len(speakers or []), n_styles=len(styles or []))
        model = Acoustic(cfg, meta["mean"], meta["std"]).to(device)
    model.train()
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.98))
    if args.resume:
        opt.load_state_dict(ckpt["opt"])
    stepper = Stepper(device)

    hours = sum(train_index[i]["frames"] for i in train_ids) * HOP / SR / 3600
    print(f"{device}  {model.n_params() / 1e6:.2f} M parameters  {len(train_ids)} clips = {hours:.1f} h  "
          f"{len(batches)} batches / epoch  {total_steps} steps  vocab {len(vocab)}", flush=True)

    t0 = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        random.shuffle(batches)
        running = {"mel": 0.0, "prior": 0.0, "dur": 0.0}
        for n, batch in enumerate(batches, 1):
            for group in opt.param_groups:
                group["lr"] = lr_at(step, args.lr, args.warmup, total_steps)
            with stepper.autocast():
                loss, parts, _ = model(*collate(train_mel, train_index, batch, vocab, device, speakers, styles))
            grad_norm = stepper(loss, opt, model.parameters(), 1.0)
            if device == "mps" and step % 50 == 0:
                # Every batch has a different length, and the MPS allocator keeps a cached block per size: measured,
                # the 12 M flow model's process grew to 18 GB on a 16 GB Mac and training slowed 12x from swapping.
                torch.mps.empty_cache()
            step += 1
            for k in running:
                running[k] += parts[k]
            if step % 50 == 0:
                print(f"epoch {epoch + 1:3d}  step {step:6d}  " + "  ".join(f"{k} {v / n:.3f}" for k, v in running.items()) +
                      f"  grad-norm {grad_norm:.2f}  lr {opt.param_groups[0]['lr']:.2e}  {(time.perf_counter() - t0) / 60:.1f} min", flush=True)

        if args.overfit and (epoch + 1) % 50 and epoch + 1 != args.epochs:
            continue
        dev, length = evaluate(model, dev_mel, dev_index, dev_ids, vocab, device, args.max_frames, speakers, styles)
        print(f"== epoch {epoch + 1}  train mel {running['mel'] / len(batches):.3f}  dev " +
              "  ".join(f"{k} {v:.3f}" for k, v in dev.items()) + f"  length {length:.2f}", flush=True)
        extra = {"chars": meta["chars"], "epoch": epoch + 1, "step": step, "speakers": speakers, "styles": styles}
        if dev["mel"] < best:
            best = dev["mel"]
            save(model, RUNS / f"{args.name}.pt", dev_mel=best, **extra)
        if not args.overfit:
            save(model, RUNS / f"{args.name}_last.pt", best=best, opt=opt.state_dict(), **extra)
        model.eval()
        for lang in LANGS:  # one dev sentence per language, through Griffin-Lim, to listen to
            entry = next((dev_index[i] for i in dev_ids if dev_index[i]["lang"] == lang), None)
            if entry:
                who = {"speaker": speakers.index(speaker_of(entry)), "style": styles.index(entry.get("style", "neutral"))} if speakers else {}
                mel, _ = model.synthesize(torch.tensor(vocab.encode(entry["text"]), device=device), LANGS.index(lang), **who)
                write_wav(RUNS / f"sample_{lang}.wav", griffin_lim(mel.cpu()))
        model.train()


if __name__ == "__main__":
    main()
