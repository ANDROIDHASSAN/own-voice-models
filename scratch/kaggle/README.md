# Kaggle pe training

Mac ki jagah Kaggle ka free GPU (T4 ×2, 30 ghante/hafta). Sab kuch official `kaggle` CLI se; token
`../../.env` mein (`KAGGLE_API_TOKEN`). Kaggle Secrets CLI se chale runs tak nahi pahunchte (naapa: `HF_TOKEN` →
ConnectionError), isliye gated data (Rasa, Kathbath) Mac se private datasets ban ke jaata hai.

| File | Kaam |
|---|---|
| [push.py](push.py) | Mac se: `upload`, `upload-folder`, `run <job>`, `status`, `fetch` |
| [night.py](night.py) | Kaggle pe chalne wale jobs: `prep_tts`, `prep_stt`, `smoke`, `train` |
| [probe.py](probe.py) | `run probe`: CPU pe, bina training, machine pe kya hai |

## Ek baar

```
cd scratch/kaggle
P=../../.venv/bin/python
$P push.py upload-folder kathbath-hindi ../../../datasets/kathbath/hindi   # 15 GB, private
$P push.py upload-folder rasa-hi-mr ../../../datasets/rasa                 # ~32 GB, private (Hindi-, Marathi- prefix)
$P push.py upload                                                          # code
$P push.py run prep_stt     # CPU (GPU quota nahi lagta): FLEURS + Kathbath -> stt_data
$P push.py run prep_tts     # CPU: Rasa -> 24 kHz, 100 mels -> tts_data24k (~15 GB)
$P push.py run smoke        # GPU 15 min: CUDA, fp16, memory, asli speed. Logs dekh ke hi poori raat
```

## Har raat

```
$P push.py run train        # GPU 0: flow TTS · GPU 1: 24 kHz vocoder (3 h), phir STT · 11 h 15 min pe band
$P push.py status
$P push.py fetch            # subah: tts/runs24k/kaggle, stt/runs/kaggle
```

Agli raat `run train` pichhli raat ke `*_last.pt` se `--resume` karta hai (train kernel apna hi output input
ke roop mein leta hai).

| Kernel | Machine | Input | Output |
|---|---|---|---|
| own-voice-prep-stt | CPU | code, kathbath-hindi, FLEURS (internet) | `stt_data/` |
| own-voice-prep-tts | CPU | code, rasa-hi-mr | `tts_data24k/` |
| own-voice-smoke | T4 ×2, 15 min | code, dono prep outputs | logs |
| own-voice-night | T4 ×2, 11 h 15 min | code, dono prep outputs, apna pichhla output | `tts_runs/`, `stt_runs/` |
