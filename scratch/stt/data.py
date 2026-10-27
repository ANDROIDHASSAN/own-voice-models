"""Lesson 3 - data: from a public speech corpus to arrays the training loop can read fast.

    python data.py                 # Hindi + Marathi + English, about 6.7 GB of downloads
    python data.py --langs hi      # one language
    python data.py --kathbath ../../../datasets/kathbath/hindi   # + Kathbath Hindi in train (see add_kathbath)

Speech recognition is SUPERVISED learning: every clip needs a human transcript. (The LLM needed no labels;
its label was the next token of the text itself.) That makes speech data expensive, and it is why this
model sees tens of hours while Whisper saw 680,000.

The corpus is FLEURS (Google, CC-BY 4.0): people reading Wikipedia sentences, 16 kHz, with official
train / dev / test splits. A sentence that is in test is never in train, so the test score measures
recognizing NEW sentences, not remembering old ones.

    train   the model learns from these
    dev     we look at these DURING training (when to stop, which checkpoint to keep)
    test    touched once, at the end, for the number we report

Computing a spectrogram is the same every epoch, so we do it once here and store the result:

    data/train.mel     every clip's log-mel frames, back to back, as float16
    data/train.json    for each clip: where its frames start, how many, its text and language
    data/meta.json     the vocabulary, and the mean / std of each mel band (measured on train only)
"""

import argparse
import json
import subprocess
import tarfile
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

from audio import N_MELS, SR, log_mel, read_wav
from text import Vocab, normalize

HERE = Path(__file__).parent
DATA = HERE / "data"
LANGS = {"hi": "hi_in", "mr": "mr_in", "en": "en_us"}
MAX_TRAIN_SEC = 20.0  # longer training clips are dropped: one 40 s clip would set the memory use of its whole batch


def fetch(lang: str, split: str) -> tuple[Path, Path]:
    folder = f"data/{LANGS[lang]}"
    get = lambda name: Path(hf_hub_download("google/fleurs", f"{folder}/{name}", repo_type="dataset", local_dir=DATA / "fleurs"))
    return get(f"{split}.tsv"), get(f"audio/{split}.tar.gz")


def build_split(split: str, langs: list[str]) -> list[dict]:
    """Writes data/<split>.mel and returns the index. Clips are read straight out of the .tar.gz, one at a
    time, so the audio is never unpacked to disk."""
    index, offset = [], 0
    with open(DATA / f"{split}.mel", "wb") as out:
        for lang in langs:
            tsv, tar = fetch(lang, split)
            # tsv columns: sentence id, file name, raw text, lightly cleaned text, spaced characters, samples, gender
            rows = {r[1]: r for r in (line.rstrip("\n").split("\t") for line in open(tsv, encoding="utf-8"))}
            seconds = 0.0
            with tarfile.open(tar, "r|gz") as archive:
                for member in archive:
                    row = rows.get(Path(member.name).name)
                    if row is None or not member.isfile():
                        continue
                    wave, rate = read_wav(archive.extractfile(member).read())
                    text = normalize(row[3])
                    if rate != SR or not text or len(wave) < SR // 2:
                        continue
                    if split == "train" and len(wave) > MAX_TRAIN_SEC * SR:
                        continue
                    mel = log_mel(wave).numpy().astype(np.float16)  # float16 halves the file; the model never notices
                    out.write(mel.tobytes())
                    index.append({"lang": lang, "sentence": row[0], "file": row[1], "text": text,
                                  "start": offset, "frames": len(mel)})
                    offset += len(mel)
                    seconds += len(wave) / SR
            print(f"  {split:5s} {lang}: {sum(e['lang'] == lang for e in index):5d} clips  {seconds / 3600:5.2f} h", flush=True)
    (DATA / f"{split}.json").write_text(json.dumps(index, ensure_ascii=False))
    return index


def _decode(m4a: bytes) -> np.ndarray:
    """Kathbath stores AAC audio in .m4a; ffmpeg turns it into 16 kHz mono float32 samples."""
    with tempfile.NamedTemporaryFile(suffix=".m4a") as f:  # a file, not a pipe: an .m4a may keep its index at the end
        f.write(m4a)
        f.flush()
        pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", f.name, "-f", "f32le", "-ac", "1", "-ar", str(SR), "pipe:1"],
                             capture_output=True, check=True).stdout
    return np.frombuffer(pcm, dtype=np.float32).copy()  # a writable array, as read_wav returns


def add_kathbath(folder: Path, index: list[dict], workers: int) -> None:
    """Appends Kathbath Hindi (AI4Bharat: people reading on their phones, ~92k clips) to the TRAIN split only.
    Dev and test stay FLEURS, so the reported numbers remain comparable with the 23-hour model. Needs pyarrow,
    used only here:  uv run --no-project --python ../../.venv/bin/python --with pyarrow python data.py --kathbath DIR"""
    import pyarrow.parquet as pq

    offset, seconds, start = index[-1]["start"] + index[-1]["frames"], 0.0, len(index)
    with open(DATA / "train.mel", "ab") as out, ProcessPoolExecutor(workers) as pool:
        for shard in sorted(folder.glob("train-*.parquet")):
            for batch in pq.ParquetFile(shard).iter_batches(batch_size=256, columns=["fname", "text", "audio_filepath"]):
                rows = batch.to_pylist()
                for row, wave in zip(rows, pool.map(_decode, [r["audio_filepath"]["bytes"] for r in rows])):
                    text = normalize(row["text"])
                    if not text or not SR // 2 <= len(wave) <= MAX_TRAIN_SEC * SR:
                        continue
                    mel = log_mel(wave).numpy().astype(np.float16)
                    out.write(mel.tobytes())
                    index.append({"lang": "hi", "source": "kathbath", "file": row["fname"], "text": text, "start": offset, "frames": len(mel)})
                    offset += len(mel)
                    seconds += len(wave) / SR
            print(f"  train hi kathbath: {len(index) - start:6d} clips  {seconds / 3600:6.2f} h  ({shard.name})", flush=True)
    (DATA / "train.json").write_text(json.dumps(index, ensure_ascii=False))


def open_split(split: str) -> tuple[np.ndarray, list[dict]]:
    """(frames, 80) array read from disk on demand (memmap), and the index that says which rows are which clip."""
    index = json.loads((DATA / f"{split}.json").read_text())
    return np.memmap(DATA / f"{split}.mel", dtype=np.float16, mode="r").reshape(-1, N_MELS), index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=list(LANGS), choices=list(LANGS))
    ap.add_argument("--kathbath", type=Path, metavar="DIR", help="folder of Kathbath Hindi train-*.parquet: added to train only")
    ap.add_argument("--workers", type=int, default=6, help="ffmpeg processes decoding Kathbath audio")
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)

    for split in ("dev", "test", "train"):
        index = build_split(split, args.langs)
    if args.kathbath:
        add_kathbath(args.kathbath, index, args.workers)

    mel, index = open_split("train")
    # Normalization statistics: every mel band is shifted to mean 0 and scaled to std 1 before the model sees
    # it, so loud and quiet bands start on equal footing. They come from TRAIN only; measuring them on dev
    # or test would leak information about the data we are supposed to be tested on.
    total, total_sq = np.zeros(N_MELS), np.zeros(N_MELS)
    for i in range(0, len(mel), 1_000_000):
        chunk = mel[i : i + 1_000_000].astype(np.float64)
        total, total_sq = total + chunk.sum(0), total_sq + (chunk**2).sum(0)
    mean = total / len(mel)
    std = np.sqrt(total_sq / len(mel) - mean**2)

    vocab = Vocab.build([e["text"] for e in index])
    dropped = sorted(set("".join(e["text"] for e in index)) - set(vocab.chars))
    (DATA / "meta.json").write_text(json.dumps(
        {"langs": args.langs, "chars": vocab.chars[1:], "mean": mean.tolist(), "std": std.tolist()}, ensure_ascii=False))
    print(f"train: {len(index)} clips, {len(mel) / 100 / 3600:.1f} h, {len(mel) * N_MELS * 2 / 1e9:.2f} GB of features")
    print(f"vocabulary: {len(vocab)} symbols including <blank>; {len(dropped)} rare characters dropped: {''.join(dropped)}")


if __name__ == "__main__":
    main()
