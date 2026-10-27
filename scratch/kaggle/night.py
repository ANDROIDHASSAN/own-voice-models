"""Kaggle jobs. push.py writes  MODE = "..."  above this file and pushes it as its own kernel:

    prep_tts   CPU, once: Rasa (Hindi + Marathi, 24 kHz, 100 mels) -> /kaggle/working/tts_data24k
    prep_stt   CPU, once: FLEURS + Kathbath Hindi -> /kaggle/working/stt_data
    smoke      GPU T4 x2, 15 minutes of `train` in its own kernel: does it run, how fast, how much memory
    train      GPU T4 x2, every night:  GPU 0 = flow TTS,  GPU 1 = 24 kHz vocoder, then STT.  -> tts_runs/, stt_runs/

Data is made on CPU machines (no GPU quota) and comes back in as the prep kernels' output, so a GPU night is
all training. Kaggle Secrets do not reach CLI-pushed runs (measured), so gated data arrives as private datasets.
Kaggle kills a session at 12 h and may lose its output then, so training runs under `timeout` until 11 h 15 min;
every loop already saves a checkpoint after each epoch / eval, and the next night resumes from it.
"""

import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

START = time.time()
DEADLINE = START + 11.25 * 3600
INPUT, OUT, WORK = Path("/kaggle/input"), Path("/kaggle/working"), Path("/tmp/work")
PY = sys.executable
TTS24 = {"TTS_SR": "24000"}


def log(msg: str) -> None:
    print(f"[{(time.time() - START) / 3600:5.2f} h] {msg}", flush=True)


def run(cmd: str, cwd: Path, logfile: Path, env: dict | None = None, gpu: str | None = None,
        until: float | None = None) -> int:
    """One stage, its output to `logfile`. With `until` it is stopped at that time (exit 124 = timed out, expected:
    the last finished epoch is saved)."""
    full = dict(os.environ, **(env or {}))
    if gpu is not None:
        full["CUDA_VISIBLE_DEVICES"] = gpu
    if until:
        cmd = f"timeout {max(60, int(until - time.time()))}s {cmd}"
    log(f"start  {cmd}  -> {logfile.name}")
    with open(logfile, "a") as f:
        code = subprocess.call(cmd, shell=True, cwd=cwd, env=full, stdout=f, stderr=subprocess.STDOUT)
    log(f"end    exit {code}  ({logfile.name})")
    return code


def attached(name: str) -> Path | None:
    """An attached file or folder, wherever Kaggle mounted it (/kaggle/input/<slug>/ or deeper)."""
    found = sorted(INPUT.rglob(name))
    return found[0] if found else None


def need(name: str) -> Path:
    found = attached(name)
    if found is None:
        sys.exit(f"{name} not found under {INPUT}: is that dataset / kernel output attached?")
    return found


def setup_code() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    archive = attached("code.tar.gz")
    if archive:
        with tarfile.open(archive) as tar:
            tar.extractall(WORK, filter="data")
    else:  # Kaggle unpacks uploaded archives itself (seen with probe.py)
        shutil.copytree(need("stt/model.py").parent.parent, WORK, dirs_exist_ok=True)


def keep(src: Path, dst: Path, suffixes: tuple[str, ...]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.glob("*"):
        if f.is_file() and f.suffix in suffixes and "overfit" not in f.name:
            shutil.copy(f, dst / f.name)


def prep_tts() -> None:
    # 700 per voice and style keeps every emotion clip (~500 each per voice) and caps the big neutral styles:
    # ~60 h, ~15 GB of 24 kHz audio + mels, under the 20 GB a Kaggle output can hold.
    rasa = need("Hindi-train-00000-of-00025.parquet").parent
    code = run(f"{PY} -u data.py --hi 0 --en 0 --hing 0 --mr 0 --rasa hi mr --rasa-dir {rasa} --rasa-clips 700 --workers 4",
               WORK / "tts", OUT / "prep_tts.log", env=TTS24)
    if code == 0:
        keep(WORK / "tts/data24k", OUT / "tts_data24k", (".pcm", ".mel", ".json", ".jsonl"))


def prep_stt() -> None:
    kathbath = need("train-00000-of-00032.parquet").parent
    if run(f"{PY} -u data.py --kathbath {kathbath} --workers 4", WORK / "stt", OUT / "prep_stt.log") == 0:
        keep(WORK / "stt/data", OUT / "stt_data", (".mel", ".json"))


def link_data(found: Path, target: Path) -> None:
    """Training only reads its data (memory-mapped), so a symlink to the read-only input is enough."""
    if target.exists() or target.is_symlink():
        target.unlink() if target.is_symlink() else shutil.rmtree(target)
    target.symlink_to(found)


def carry(runs_name: str, runs: Path) -> None:
    """Last run's checkpoints (best AND last) come back in: a new kernel version's output replaces the old one, so
    anything not copied forward here is lost."""
    runs.mkdir(parents=True, exist_ok=True)
    for f in INPUT.rglob(f"{runs_name}/*.pt"):
        shutil.copy(f, runs / f.name)


def memory_log() -> None:
    """Host RAM every 10 min: night 1's flow run was SIGKILLed (exit 137) with no Python error."""
    while True:
        log("ram " + subprocess.run("free -m | awk '/Mem/ {print $3\" MB used, \"$7\" MB available\"}'", shell=True,
                                    capture_output=True, text=True).stdout.strip())
        time.sleep(600)


def train() -> None:
    import torch

    log(f"GPUs: {torch.cuda.device_count()} x {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    if torch.cuda.device_count() < 2:
        sys.exit("expected 2 GPUs (T4 x2); got fewer, stopping instead of running half the plan")
    # 24 kHz TTS data: the prep_tts output, or the same files made on the Mac and uploaded (dataset tts-data24k).
    # Found by train.pcm, which only TTS data has (STT data is mels only).
    link_data(need("train.pcm").parent, WORK / "tts/data24k")
    link_data(need("stt_data"), WORK / "stt/data")
    tts_runs, stt_runs = WORK / "tts/runs24k", WORK / "stt/runs"
    carry("tts_runs", tts_runs)
    carry("stt_runs", stt_runs)
    resume = lambda runs, name: " --resume" if (runs / f"{name}_last.pt").exists() else ""
    threading.Thread(target=memory_log, daemon=True).start()

    def tts() -> None:
        run(f"{PY} -u train.py --flow --d-model 256 --layers 6 --heads 4 --epochs 30 --name flow --max-frames 16000"
            + resume(tts_runs, "flow"), WORK / "tts", OUT / "train_flow.log", env=TTS24, gpu="0", until=DEADLINE)

    def gpu1() -> None:
        # Each stage resumes, and one already finished returns at once: vocoder (40k steps, ~30 min), its GAN
        # fine-tune (needs vocoder.pt), STT (15 epochs, ~1.3 h). Smoke: equal shares.
        stages = [(f"vocoder.py --steps 40000 --batch 32" + resume(tts_runs, "vocoder"), "tts", "train_vocoder.log", TTS24),
                  (f"vocoder.py --gan --steps 20000 --batch 16" + resume(tts_runs, "vocoder_gan"), "tts", "train_vocoder_gan.log", TTS24),
                  (f"train.py --epochs 15 --name kathbath" + resume(stt_runs, "kathbath"), "stt", "train_kathbath.log", None)]
        for i, (cmd, where, logname, env) in enumerate(stages):
            share = (DEADLINE - time.time()) / (len(stages) - i) if DEADLINE - START < 3600 else 4 * 3600
            run(f"{PY} -u {cmd}", WORK / where, OUT / logname, env=env, gpu="1", until=min(DEADLINE, time.time() + share))

    jobs = [threading.Thread(target=tts), threading.Thread(target=gpu1)]
    for t in jobs:
        t.start()
    for t in jobs:
        t.join()
    keep(tts_runs, OUT / "tts_runs", (".pt", ".wav"))
    keep(stt_runs, OUT / "stt_runs", (".pt",))


def noise_bank(out: Path, music_hours: float = 15.0) -> None:
    """MUSAN noise (all) + music (up to `music_hours`) -> log-mel banks with the STT's front end. Files are
    split 90 / 10 by name: the 10 % never trained on are the test bank, plus 3 minutes of their audio as wav
    for robust.py on the Mac."""
    import hashlib

    import numpy as np

    sys.path.insert(0, str(WORK / "stt"))
    from audio import log_mel, read_wav

    musan = need("musan")
    files = sorted(musan.rglob("noise/**/*.wav"))
    music = sorted(musan.rglob("music/**/*.wav"), key=lambda p: hashlib.md5(p.name.encode()).hexdigest())
    files += music[: int(len(music) * music_hours / 42)]  # MUSAN music is ~42 h
    test = lambda p: int(hashlib.md5(p.name.encode()).hexdigest(), 16) % 10 == 0
    for split, keep_files in (("train", [p for p in files if not test(p)]), ("test", [p for p in files if test(p)])):
        mels, waves = [], []
        for p in keep_files:
            wave, rate = read_wav(p)
            if rate == 16000 and len(wave) > 16000:
                mels.append(log_mel(wave).numpy().astype(np.float16))
                if split == "test" and sum(map(len, waves)) < 180 * 16000:
                    waves.append(wave[: 20 * 16000])
        np.save(out / f"noise_{split}.npy", np.concatenate(mels))
        log(f"noise {split}: {len(mels)} files, {sum(map(len, mels)) / 360000:.1f} h")
        if waves:
            np.save(out / "noise_test_audio.npy", (np.concatenate(waves) * 32767).astype(np.int16))


def robust() -> None:
    """Night 3: the front end. GPU 0: STT fine-tuned with noise. GPU 1: neural VAD, then the turn detector."""
    import torch

    log(f"GPUs: {torch.cuda.device_count()} x {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    link_data(need("stt_data"), WORK / "stt/data")
    stt_runs, orch = WORK / "stt/runs", OUT / "orch_runs"
    stt_runs.mkdir(parents=True, exist_ok=True)
    orch.mkdir(parents=True, exist_ok=True)
    shutil.copy(need("stt_runs/kathbath.pt"), stt_runs / "kathbath.pt")
    bank = OUT / "noise"
    bank.mkdir(exist_ok=True)
    noise_bank(bank)

    def gpu0() -> None:
        run(f"{PY} -u train.py --init runs/kathbath.pt --noise {bank / 'noise_train.npy'} --epochs 8 --name noisy "
            f"--lr 3e-4 --warmup 500", WORK / "stt", OUT / "train_noisy.log", gpu="0", until=DEADLINE)

    def gpu1() -> None:
        run(f"{PY} -u nvad.py --stt-data {WORK / 'stt/data'} --noise {bank / 'noise_train.npy'} --steps 8000 --out {orch / 'nvad.pt'}",
            WORK / "orchestrator", OUT / "train_nvad.log", gpu="1", until=DEADLINE)
        texts = f"{need('eot_sentences.txt')} {WORK / 'stt/data/train.json'}"
        run(f"{PY} -u eot.py --texts {texts} --dev {WORK / 'stt/data/dev.json'} --epochs 4 --out {orch / 'eot.pt'}",
            WORK / "orchestrator", OUT / "train_eot.log", gpu="1", until=DEADLINE)

    jobs = [threading.Thread(target=gpu0), threading.Thread(target=gpu1)]
    for t in jobs:
        t.start()
    for t in jobs:
        t.join()
    keep(stt_runs, OUT / "stt_runs", (".pt",))
    (bank / "noise_train.npy").unlink()  # 1+ GB, rebuilt from MUSAN in minutes: not worth keeping as output


def vad() -> None:
    """The neural VAD alone (v2: synthetic noise and chatter-only windows added), on GPU 0."""
    link_data(need("stt_data"), WORK / "stt/data")
    bank, orch = OUT / "noise", OUT / "orch_runs"
    bank.mkdir(exist_ok=True)
    orch.mkdir(exist_ok=True)
    noise_bank(bank)
    run(f"{PY} -u nvad.py --stt-data {WORK / 'stt/data'} --noise {bank / 'noise_train.npy'} --steps 12000 --out {orch / 'nvad.pt'}",
        WORK / "orchestrator", OUT / "train_nvad.log", gpu="0", until=DEADLINE)
    (bank / "noise_train.npy").unlink()


def main() -> None:
    log(subprocess.run("df -h /tmp /kaggle/working | tail -2; nproc; free -g | head -2", shell=True,
                       capture_output=True, text=True).stdout)
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found on this image")
    setup_code()
    if MODE == "smoke":  # noqa: F821 - the whole train path for 15 minutes: CUDA, fp16, memory, real speed
        global DEADLINE
        DEADLINE = START + 15 * 60
    {"prep_tts": prep_tts, "prep_stt": prep_stt, "train": train, "smoke": train, "robust": robust, "vad": vad}[MODE]()  # noqa: F821 - MODE comes from push.py
    log("output: " + ", ".join(f"{p.relative_to(OUT)} {p.stat().st_size / 1e6:.0f} MB" for p in sorted(OUT.rglob("*")) if p.is_file()))


if __name__ == "__main__":
    main()
