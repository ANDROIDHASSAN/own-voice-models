"""Mac -> Kaggle, with the official kaggle CLI. The Kaggle token comes from ../../.env (KAGGLE_API_TOKEN).

    python push.py upload-folder kathbath-hindi ../../../datasets/kathbath/hindi   # gated parquet -> private dataset
    python push.py upload     # code (always) + the TTS base data (first time only), as PRIVATE datasets
    python push.py run prep_tts   # CPU, once: Rasa -> 24 kHz TTS data   (needs dataset rasa-hi-mr)
    python push.py run prep_stt   # CPU, once: FLEURS + Kathbath -> STT data   (needs dataset kathbath-hindi)
    python push.py run train      # GPU T4 x2, every night, in the background; the Mac can sleep
    python push.py run probe      # CPU only, no training: what does the Kaggle machine see?
    python push.py status     # is it running?
    python push.py fetch      # the train kernel's checkpoints + logs -> ../tts/runs24k/kaggle, ../stt/runs/kaggle

No Hugging Face token is needed on Kaggle: Secrets do not reach CLI-pushed runs, so the gated datasets (Rasa,
Kathbath) are uploaded from the Mac as private Kaggle datasets with upload-folder.
"""

import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
SCRATCH = HERE.parent
USER = "hassankazi786"
CODE, BASE, KERNEL = f"{USER}/own-voice-code", f"{USER}/own-voice-tts-base", f"{USER}/own-voice-night"


def kaggle(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    for line in (SCRATCH.parent / ".env").read_text().splitlines():  # names only ever printed, never values
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.removeprefix("export ").split("=", 1)
            env[key.strip()] = value.strip().strip("'\"")
    result = subprocess.run(["kaggle", *args], env=env, capture_output=True, text=True)
    print((result.stdout + result.stderr).strip())
    if check and result.returncode:
        sys.exit(f"kaggle {args[0]} {args[1]} failed")
    return result


def exists(kind: str, ref: str) -> bool:
    if kind == "dataset":
        return kaggle("datasets", "status", ref, check=False).returncode == 0
    return kaggle("kernels", "status", ref, check=False).returncode == 0


def dataset(ref: str, folder: Path, message: str) -> None:
    (folder / "dataset-metadata.json").write_text(json.dumps({"title": ref.split("/")[1], "id": ref, "licenses": [{"name": "other"}]}))
    if exists("dataset", ref):
        kaggle("datasets", "version", "-p", str(folder), "-m", message)
    else:
        kaggle("datasets", "create", "-p", str(folder))  # private unless --public is passed


def upload() -> None:
    with tempfile.TemporaryDirectory() as tmp:  # code: .py and .md of each lesson folder; no data, no checkpoints
        with tarfile.open(Path(tmp) / "code.tar.gz", "w:gz") as tar:
            for sub in ("stt", "tts", "kaggle", "orchestrator"):
                for f in sorted((SCRATCH / sub).glob("*")):
                    if f.suffix in (".py", ".md"):
                        tar.add(f, arcname=f"{sub}/{f.name}")
        dataset(CODE, Path(tmp), "code update")
    if not exists("dataset", BASE):  # 2.8 GB: uploaded once
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("train.pcm", "train.mel", "train.json", "train.jsonl", "dev.pcm", "dev.mel", "dev.json", "dev.jsonl", "meta.json"):
                os.link(SCRATCH / "tts/data" / name, Path(tmp) / name)  # hard links: no 2.8 GB copy
            dataset(BASE, Path(tmp), "base data")


def upload_files(slug: str, folder: str, pattern: str = "*") -> None:
    """Flat files (e.g. the 24 kHz TTS data made on the Mac) -> a PRIVATE dataset, by hard link."""
    with tempfile.TemporaryDirectory(dir=Path(folder).parent) as tmp:
        for f in sorted(Path(folder).glob(pattern)):
            if f.is_file():
                os.link(f, Path(tmp) / f.name)
        dataset(f"{USER}/{slug}", Path(tmp), "data")


def upload_folder(slug: str, folder: str) -> None:
    """A folder of .parquet files (gated data downloaded on the Mac) -> a PRIVATE dataset. Kaggle runs started from
    the CLI get no Secrets (measured: HF_TOKEN -> ConnectionError), so gated data must already be on Kaggle.
    Files in subfolders are flattened with the folder as prefix: rasa/Hindi/train-0.parquet -> Hindi-train-0.parquet."""
    root = Path(folder)
    with tempfile.TemporaryDirectory(dir=root.parent) as tmp:  # same disk: hard links, no copy
        for f in sorted(root.rglob("*.parquet")):
            os.link(f, Path(tmp) / (f.name if f.parent == root else f"{f.parent.name}-{f.name}"))
        dataset(f"{USER}/{slug}", Path(tmp), "data")


# Each job is its own kernel. TTS data for smoke/train: dataset tts-data24k, made on the Mac (prep_tts on
# Kaggle CPU was 4x slower than the Mac). Inputs: the code, its data, and its own previous output (to resume).
JOBS = {
    "probe": (f"{USER}/own-voice-probe", False, [CODE], []),
    "prep_tts": (f"{USER}/own-voice-prep-tts", False, [CODE, f"{USER}/rasa-hi-mr"], []),
    "prep_stt": (f"{USER}/own-voice-prep-stt", False, [CODE, f"{USER}/kathbath-hindi"], []),
    "smoke": (f"{USER}/own-voice-smoke", True, [CODE, f"{USER}/tts-data24k"], [f"{USER}/own-voice-prep-stt"]),
    "train": (KERNEL, True, [CODE, f"{USER}/tts-data24k"], [f"{USER}/own-voice-prep-stt"]),
    # The front end: noise (MUSAN, CC BY 4.0, a public Kaggle copy), story sentences for the turn detector,
    # the STT data and the 158 h STT checkpoint (last night's kernel output) to fine-tune.
    "robust": (f"{USER}/own-voice-robust", True, [CODE, f"{USER}/own-voice-text", "nhattruongdev/musan-noise"],
               [f"{USER}/own-voice-prep-stt", KERNEL]),
    "vad": (f"{USER}/own-voice-vad", True, [CODE, "nhattruongdev/musan-noise"], [f"{USER}/own-voice-prep-stt"]),
}


def run(job: str = "train") -> None:
    slug, gpu, datasets, kernels = JOBS[job]
    script = "probe.py" if job == "probe" else "night.py"
    with tempfile.TemporaryDirectory() as tmp:
        body = (HERE / script).read_text()
        (Path(tmp) / script).write_text(body if job == "probe" else f'MODE = "{job}"\n' + body)
        if job == "train" and exists("kernel", slug):
            kernels = kernels + [slug]  # last night's checkpoints come back in as input, so training resumes
        meta = {"id": slug, "title": slug.split("/")[1], "code_file": script, "language": "python", "kernel_type": "script",
                "is_private": True, "enable_gpu": gpu, "enable_internet": True, "dataset_sources": datasets, "kernel_sources": kernels}
        if gpu:
            meta["machine_shape"] = "NvidiaTeslaT4"
        (Path(tmp) / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
        kaggle("kernels", "push", "-p", tmp, *(["--accelerator", "NvidiaTeslaT4"] if gpu else []))


def fetch() -> None:
    out = SCRATCH / "kaggle/output"
    kaggle("kernels", "output", KERNEL, "-p", str(out), "--force")
    for name, runs in (("tts", "runs24k/kaggle"), ("stt", "runs/kaggle")):
        (SCRATCH / name / runs).mkdir(parents=True, exist_ok=True)
        for f in (out / f"{name}_runs").glob("*"):
            f.replace(SCRATCH / name / runs / f.name)


if __name__ == "__main__":
    if sys.argv[1:2] == ["upload-files"]:
        sys.exit(upload_files(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "*"))
    if sys.argv[1:2] == ["upload-folder"]:
        sys.exit(upload_folder(sys.argv[2], sys.argv[3]))
    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    if command == "run":  # run prep_tts | prep_stt | train | probe
        run(sys.argv[2] if len(sys.argv) > 2 else "train")
    elif command == "status":
        for slug, *_ in {j[0]: j for j in JOBS.values()}.values():
            kaggle("kernels", "status", slug, check=False)
    else:
        {"upload": upload, "fetch": fetch}[command]()
