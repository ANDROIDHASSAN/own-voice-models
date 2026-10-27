#!/bin/sh
# Downloads the STT and TTS models (~420 MB) into ./models. The LLM is fetched by llama-server on first run.
set -eu
cd "$(dirname "$0")"
mkdir -p models
R=https://github.com/k2-fsa/sherpa-onnx/releases/download
for url in \
  "$R/asr-models/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-v2-int8-2026-02-05.tar.bz2" \
  "$R/tts-models/sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2"; do
  name=$(basename "$url" .tar.bz2)
  [ -d "models/$name" ] || curl -fL "$url" | tar -xj -C models
done
ls models
