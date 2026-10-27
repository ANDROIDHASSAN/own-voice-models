"""Kaggle probe, CPU only, no training: what does night.py find on this machine? Run with  python push.py probe."""

import importlib.util
import shutil
import subprocess
from pathlib import Path

print("inputs:", sorted(str(p.relative_to("/kaggle/input")) for p in Path("/kaggle/input").rglob("*") if p.is_file())[:20])
print("ffmpeg:", shutil.which("ffmpeg"), "| timeout:", shutil.which("timeout"))
print("pyarrow:", bool(importlib.util.find_spec("pyarrow")), "| huggingface_hub:", bool(importlib.util.find_spec("huggingface_hub")))
print(subprocess.run("df -h /tmp /kaggle/working | tail -2; nproc", shell=True, capture_output=True, text=True).stdout)
try:
    from kaggle_secrets import UserSecretsClient

    print("HF_TOKEN secret: present, length", len(UserSecretsClient().get_secret("HF_TOKEN")))
except Exception as e:  # noqa: BLE001 - the probe's job is to report this
    print("HF_TOKEN secret: NOT available:", type(e).__name__)
