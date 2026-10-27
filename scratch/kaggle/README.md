# Training on Kaggle

Kaggle's free GPU (T4 ×2, 30 hours/week) instead of the Mac. Everything goes through the official `kaggle` CLI; the
token is in `../../.env` (`KAGGLE_API_TOKEN`). Kaggle Secrets do not reach runs started from the CLI (measured:
`HF_TOKEN` → ConnectionError), so gated data (Rasa, Kathbath) goes up from the Mac as private datasets.

| File | Job |
|---|---|
| [push.py](push.py) | From the Mac: `upload`, `upload-folder`, `run <job>`, `status`, `fetch` |
| [night.py](night.py) | Jobs that run on Kaggle: `prep_tts`, `prep_stt`, `smoke`, `train` |
| [probe.py](probe.py) | `run probe`: on CPU, no training, shows what the machine has |

## Once

```
cd scratch/kaggle
P=../../.venv/bin/python
$P push.py upload-folder kathbath-hindi ../../../datasets/kathbath/hindi   # 15 GB, private
$P push.py upload-folder rasa-hi-mr ../../../datasets/rasa                 # ~32 GB, private (Hindi-, Marathi- prefix)
$P push.py upload                                                          # code
$P push.py run prep_stt     # CPU (uses no GPU quota): FLEURS + Kathbath -> stt_data
$P push.py run prep_tts     # CPU: Rasa -> 24 kHz, 100 mels -> tts_data24k (~15 GB)
$P push.py run smoke        # GPU 15 min: CUDA, fp16, memory, real speed. Read the logs before a full night
```

## Every night

```
$P push.py run train        # GPU 0: flow TTS · GPU 1: 24 kHz vocoder (3 h), then STT · stops at 11 h 15 min
$P push.py status
$P push.py fetch            # morning: tts/runs24k/kaggle, stt/runs/kaggle
```

The next night, `run train` does `--resume` from the previous night's `*_last.pt` (the train kernel takes its own
output as input).

| Kernel | Machine | Input | Output |
|---|---|---|---|
| own-voice-prep-stt | CPU | code, kathbath-hindi, FLEURS (internet) | `stt_data/` |
| own-voice-prep-tts | CPU | code, rasa-hi-mr | `tts_data24k/` |
| own-voice-smoke | T4 ×2, 15 min | code, both prep outputs | logs |
| own-voice-night | T4 ×2, 11 h 15 min | code, both prep outputs, its own previous output | `tts_runs/`, `stt_runs/` |
