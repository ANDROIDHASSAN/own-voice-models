"""Lesson 3 - data: where a small TTS model gets clean speech to learn from.

    python data.py                      # Hindi, English and Hinglish from a teacher model + real Marathi recordings
    python data.py --hi 200 --en 100 --hing 100 --mr 100    # a small set, to try the pipeline

A TTS model needs what a recognizer does not: ONE consistent voice, studio-clean, hours of it. Two sources:

    Teacher (distillation)   A large TTS model reads text we choose, and the small model learns to imitate
                             it. Any amount of data, one voice across languages, no noise. This is how the
                             smallest published TTS models are made (sanoTTS 2026: a 1.5 M student of a 15 M
                             Piper teacher). Our teacher is Supertonic 3, the 99 M model the real agent uses;
                             it reads Hindi, English and Roman-script Hinglish in the same voice.
    Real recordings          Supertonic has no Marathi, so Marathi is OpenSLR 64 (Google, CC BY-SA 4.0):
                             about 3 hours of volunteers reading, recorded for TTS.

The student is therefore a "derivative" of the teacher under its Open RAIL-M licence: its use restrictions
travel with our weights (see README). A model distilled from a teacher can approach it, not beat it.

What gets written:
    data/train.pcm     every clip's samples, back to back, 16-bit at 22.05 kHz   (the vocoder trains on these)
    data/train.mel     every clip's log-mel frames, float16                      (the acoustic model trains on these)
    data/train.json    for each clip: language, text, where its samples and frames start, how many
                       (written as train.jsonl line by line while running, so an interrupted run can resume)
    data/meta.json     the character vocabulary and the mean / std of each mel band
"""

import argparse
import csv
import json
import random
import re
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from audio import DATA_DIR, HOP, N_MELS, SR, log_mel, read_wav, resample, trim_silence
from text import LANGS, Vocab, normalize

HERE = Path(__file__).parent
DATA = DATA_DIR
LLM_DATA, STT_DATA = HERE / "../llm/data", HERE / "../stt/data/fleurs/data"
TEACHER = HERE / "../../models/sherpa-onnx-supertonic-3-tts-int8-2026-05-11"
MARATHI_URL = "https://www.openslr.org/resources/64"
DEV_PER_LANG = 40


def _usable(sentence: str, lo: int = 20, hi: int = 140) -> bool:
    return lo <= len(sentence) <= hi and not re.search(r"\d", sentence)  # digits need a number-to-words step we do not have


def texts_hi(n: int) -> list[str]:
    """Story sentences (simple, everyday words) plus news-style FLEURS sentences (rarer words and names)."""
    with open(LLM_DATA / "raw_hi.txt", "rb") as f:  # the file is a cut-off download: read a prefix, ignore a split character
        stories = f.read(8_000_000).decode("utf-8", errors="ignore")
    found = re.findall(r"[^।\"\\\n]{20,140}।", stories)
    fleurs = [line.split("\t")[2] for line in open(STT_DATA / "hi_in/train.tsv", encoding="utf-8")]
    return _pick([s.strip() for s in found + fleurs if not re.search(r"[a-zA-Z]", s)], n)


def texts_en(n: int) -> list[str]:
    return _pick([line.split("\t")[2] for line in open(STT_DATA / "en_us/train.tsv", encoding="utf-8")], n, hi=200)


def texts_hing(n: int) -> list[str]:
    rows = csv.reader(open(LLM_DATA / "raw_hing.txt", encoding="utf-8"))
    return _pick([cell.strip() for row in rows for cell in row[:2]], n, lo=14)


def _pick(sentences: list[str], n: int, lo: int = 20, hi: int = 140) -> list[str]:
    unique = sorted({s for s in sentences if _usable(s, lo, hi)})
    random.Random(0).shuffle(unique)
    return unique[:n]


_teacher = None


def _speak(job: tuple[str, str]) -> tuple[str, str, bytes]:
    """Runs in a worker process: the teacher reads one sentence. Returns 16-bit samples at 22.05 kHz."""
    global _teacher
    import sherpa_onnx

    lang, text = job
    if _teacher is None:
        f = lambda name: str(TEACHER / name)
        model = sherpa_onnx.OfflineTtsSupertonicModelConfig(
            duration_predictor=f("duration_predictor.int8.onnx"), text_encoder=f("text_encoder.int8.onnx"),
            vector_estimator=f("vector_estimator.int8.onnx"), vocoder=f("vocoder.int8.onnx"),
            tts_json=f("tts.json"), unicode_indexer=f("unicode_indexer.bin"), voice_style=f("voice.bin"))
        _teacher = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(supertonic=model, num_threads=2)))
    config = sherpa_onnx.GenerationConfig()
    config.sid, config.speed, config.num_steps = 0, 1.0, 8  # more denoising steps than the agent uses: cleaner targets
    config.extra = {"lang": "en" if lang == "en" else "hi"}  # Roman Hinglish is read with the Hindi voice
    audio = _teacher.generate(text, config)
    wave = resample(torch.tensor(audio.samples, dtype=torch.float32), audio.sample_rate, SR)
    return lang, text, _pcm(wave)


def _pcm(wave: torch.Tensor) -> bytes:
    """Trim the silence at both ends and bring every clip to the same LOUDNESS: RMS at -20 dBFS, then turned down
    only if a peak would clip. (Peak levelling makes a soft, breathy clip as loud at its peak as a shout, so the
    model hears the same voice at random volumes; loudness levelling is what broadcast and TTS corpora use.)"""
    wave = trim_silence(wave)
    wave = wave * (0.1 / wave.pow(2).mean().sqrt().clamp(min=1e-5))  # 0.1 RMS = -20 dBFS
    wave = wave / wave.abs().max().clamp(min=0.99) * 0.99
    return (wave * 32767).short().numpy().tobytes()


def marathi(n: int, skip: set = frozenset()):
    """Real recordings, read straight out of the zip. 48 kHz -> 22.05 kHz. Sentences in `skip` are already done."""
    if n == 0:  # --mr 0: not even the download (on Kaggle the clips arrive with the uploaded data instead)
        return
    folder = DATA / "openslr64"
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("line_index.tsv", "mr_in_female.zip"):
        if not (folder / name).exists():
            print(f"  downloading {name} ...", flush=True)
            urllib.request.urlretrieve(f"{MARATHI_URL}/{name}", folder / name)
    lines = [line.rstrip("\n").split("\t") for line in open(folder / "line_index.tsv", encoding="utf-8")]
    with zipfile.ZipFile(folder / "mr_in_female.zip") as archive:
        for file_id, text in [(f, t) for f, t in lines if _usable(t, 10, 200)][:n]:
            if ("mr", text, None) in skip:
                continue
            wave, rate = read_wav(archive.read(f"{file_id}.wav"))
            yield "mr", text, _pcm(resample(torch.from_numpy(wave), rate, SR))


RASA = {"hi": "Hindi", "mr": "Marathi"}
DEV_PER_SPEAKER = 20


def _rasa_clip(row: dict) -> tuple[str, str, bytes, str, str] | None:
    """Runs in a worker process: one Rasa row (48 kHz WAV) -> the same 22.05 kHz, trimmed, levelled clip as the rest."""
    wave, rate = read_wav(row["audio"]["bytes"])
    if wave.ndim > 1:
        wave = wave.mean(1)
    return row["lang"], row["text"], _pcm(resample(torch.from_numpy(wave.astype(np.float32)), rate, SR)), row["speaker"], row["style"]


def rasa(lang: str, n: int, skip: set, workers: int, free_disk: bool = False, folder: Path | None = None):
    """Lesson 10 data: AI4Bharat Rasa (CC BY 4.0), studio recordings of one female and one male voice per language,
    ~25 h each, read neutrally AND in the six Ekman emotions (happy, sad, angry, fear, disgust, surprise), plus
    conversation, news and storytelling styles. Real people, so the model can learn real expressiveness instead
    of copying another TTS. Gated: accept the terms on huggingface.co/datasets/ai4bharat/Rasa first.
    Needs pyarrow, used only here:  uv run --no-project --python ../../.venv/bin/python --with pyarrow python data.py --rasa ...
    `n` caps clips per voice AND style: most of Rasa is neutral reading, and a cap per voice alone could fill
    up on neutral before the first emotion shard. `free_disk` deletes each DOWNLOADED shard once it is read.
    `folder`: shards already on disk (Hindi/train-*.parquet or Hindi-train-*.parquet), e.g. a Kaggle dataset."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    if folder:  # local: never deleted, whatever free_disk says (a Kaggle input is read-only anyway)
        shards = sorted(map(str, list(folder.glob(f"{RASA[lang]}/train-*.parquet")) + list(folder.glob(f"{RASA[lang]}-train-*.parquet"))))
        if not shards:
            raise SystemExit(f"no {RASA[lang]} train shards in {folder}")
    else:
        shards = sorted(f for f in HfApi().list_repo_files("ai4bharat/Rasa", repo_type="dataset")
                        if f.startswith(f"{RASA[lang]}/train-") and f.endswith(".parquet"))
    taken = {}
    with ProcessPoolExecutor(workers) as pool:
        for shard in shards:
            path = shard if folder else hf_hub_download("ai4bharat/Rasa", shard, repo_type="dataset", local_dir=DATA / "rasa")
            for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=["text", "gender", "style", "audio"]):
                rows = []
                for r in batch.to_pylist():
                    speaker = f"rasa_{lang}_{(r['gender'] or 'unknown').strip().lower()}"
                    style = (r["style"] or "neutral").strip().lower()
                    if taken.get((speaker, style), 0) >= n or (lang, r["text"], speaker) in skip or not _usable(r["text"] or "", 10, 200):
                        continue
                    taken[speaker, style] = taken.get((speaker, style), 0) + 1
                    rows.append({"lang": lang, "text": r["text"], "audio": r["audio"], "speaker": speaker, "style": style})
                yield from pool.map(_rasa_clip, rows)
            if free_disk and not folder:
                Path(path).unlink()
    print(f"  rasa {lang}: " + "  ".join(f"{s}/{st} {k}" for (s, st), k in sorted(taken.items())), flush=True)


def open_split(split: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """(samples as int16, mel frames (N, 80), index). Both arrays are memory-mapped."""
    index = json.loads((DATA / f"{split}.json").read_text())
    return (np.memmap(DATA / f"{split}.pcm", dtype=np.int16, mode="r"),
            np.memmap(DATA / f"{split}.mel", dtype=np.float16, mode="r").reshape(-1, N_MELS), index)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hi", type=int, default=3000, help="sentences the teacher reads")
    ap.add_argument("--en", type=int, default=1600)
    ap.add_argument("--hing", type=int, default=2500)
    ap.add_argument("--mr", type=int, default=2000, help="real recordings (the corpus has about 1500)")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--rasa", nargs="*", choices=list(RASA), default=[], help="lesson 10: add Rasa expressive recordings, e.g. --rasa hi mr")
    ap.add_argument("--rasa-clips", type=int, default=100_000, help="at most this many clips per Rasa voice and style")
    ap.add_argument("--rasa-dir", type=Path, help="Rasa shards already on disk instead of downloading them")
    ap.add_argument("--free-disk", action="store_true", help="delete each downloaded Rasa shard once read (Kaggle)")
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)

    # Resumable. Generating hours of audio takes hours, and a laptop that sleeps can kill a worker halfway.
    # Every clip is appended to <split>.jsonl the moment its audio is on disk, so a rerun reads what is done,
    # cuts off any half-written tail, and continues with the sentences that are left.
    index = {s: [json.loads(line) for line in open(DATA / f"{s}.jsonl", encoding="utf-8")] if (DATA / f"{s}.jsonl").exists() else []
             for s in ("train", "dev")}
    files = {}
    for split, entries in index.items():
        end = entries[-1] if entries else {"start": 0, "samples": 0, "mel_start": 0, "frames": 0}
        pcm, mel = open(DATA / f"{split}.pcm", "ab"), open(DATA / f"{split}.mel", "ab")
        pcm.truncate((end["start"] + end["samples"]) * 2)
        mel.truncate((end["mel_start"] + end["frames"]) * N_MELS * 2)
        files[split] = (pcm, mel, open(DATA / f"{split}.jsonl", "a", encoding="utf-8"))
    # A sentence is done per voice: Rasa's two speakers read the same texts, and both readings are wanted.
    done = {(e["lang"], e["text"], e.get("speaker")) for entries in index.values() for e in entries}
    seen = {lang: sum(e["lang"] == lang and "speaker" not in e for entries in index.values() for e in entries) for lang in LANGS}
    seen_voice = {}
    for e in (e for entries in index.values() for e in entries if "speaker" in e):
        seen_voice[e["speaker"]] = seen_voice.get(e["speaker"], 0) + 1
    if done:
        print(f"  resuming: {len(done)} clips already on disk", flush=True)

    def add(lang: str, text: str, pcm: bytes, speaker: str | None = None, style: str | None = None) -> None:
        if len(pcm) < SR or not normalize(text):  # under half a second: the teacher produced nothing usable
            return
        if speaker:  # a Rasa voice: the first few of each voice go to dev, so every voice is measured
            seen_voice[speaker] = seen_voice.get(speaker, 0) + 1
            split = "dev" if seen_voice[speaker] <= DEV_PER_SPEAKER else "train"
        else:
            seen[lang] += 1
            split = "dev" if seen[lang] <= DEV_PER_LANG else "train"
        mel = log_mel(torch.frombuffer(bytearray(pcm), dtype=torch.int16).float() / 32768).numpy().astype(np.float16)
        entries, (pcm_file, mel_file, log) = index[split], files[split]
        last = entries[-1] if entries else {"start": 0, "samples": 0, "mel_start": 0, "frames": 0}
        entries.append({"lang": lang, "text": text, "start": last["start"] + last["samples"], "samples": len(pcm) // 2,
                        "mel_start": last["mel_start"] + last["frames"], "frames": len(mel),
                        **({"speaker": speaker, "style": style} if speaker else {})})
        pcm_file.write(pcm)
        mel_file.write(mel.tobytes())
        pcm_file.flush(), mel_file.flush()  # audio first, then the index line that points at it
        log.write(json.dumps(entries[-1], ensure_ascii=False) + "\n")
        log.flush()
        if (sum(seen.values()) + sum(seen_voice.values())) % 500 == 0:
            print(f"  {sum(seen.values()) + sum(seen_voice.values())} clips  " +
                  "  ".join(f"{k} {v}" for k, v in (seen | seen_voice).items()), flush=True)

    for lang, text, pcm in marathi(args.mr, skip=done):
        add(lang, text, pcm)
    jobs = [(lang, t) for lang, texts_of, n in (("hi", texts_hi, args.hi), ("en", texts_en, args.en), ("hing", texts_hing, args.hing))
            if n for t in texts_of(n) if (lang, t, None) not in done]  # n = 0: the source files are not even opened
    random.Random(0).shuffle(jobs)
    # ProcessPoolExecutor, not multiprocessing.Pool: if a worker dies, Pool waits forever for its result,
    # while the executor raises BrokenProcessPool, so the run stops loudly and can simply be started again.
    with ProcessPoolExecutor(args.workers) as pool:  # each worker loads its own copy of the teacher
        for item in pool.map(_speak, jobs, chunksize=8):
            add(*item)
    for lang in args.rasa:
        for item in rasa(lang, args.rasa_clips, done, args.workers, args.free_disk, args.rasa_dir):
            add(*item)
    for split, entries in index.items():
        for f in files[split]:
            f.close()
        (DATA / f"{split}.json").write_text(json.dumps(entries, ensure_ascii=False))

    _, mel, train = open_split("train")
    # One number each (see model.py), from ~2 M frames spread over the WHOLE file: new sources are appended at
    # the end, and statistics from the first frames only would never see them.
    sample = mel[:: max(1, len(mel) // 2_000_000)].astype(np.float32)
    mean, std = float(sample.mean()), float(sample.std())
    vocab = Vocab.build([normalize(e["text"]) for e in train])
    (DATA / "meta.json").write_text(json.dumps({"chars": vocab.chars[1:], "mean": mean, "std": std}, ensure_ascii=False))
    for lang in LANGS:
        clips = [e for e in train if e["lang"] == lang]
        print(f"  train {lang:4s}: {len(clips):5d} clips  {sum(e['samples'] for e in clips) / SR / 3600:5.2f} h")
    print(f"vocabulary: {len(vocab)} symbols; mel mean {mean:.2f} std {std:.2f}; {len(mel) / (SR / HOP) / 3600:.1f} h total")


if __name__ == "__main__":
    main()
