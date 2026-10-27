# Interview concepts: is project mein kahan hain

Do mashhoor lists ("25 fundamentals" aur "50 AI engineering concepts") ka har item, aur ye ki tune use is
project mein kahan khud banaya. Jahan nahi banaya, wahan saaf "nahi" likha hai: interview mein jhooth mat bolna,
"ye maine nahi banaya, par ye aise kaam karta hai" kehna theek hai.

Phir teen nayi lists, usi shaili mein: STT, TTS aur voice orchestration ke wo concepts jo poochhe jaate hain.
Har concept ka poora matlab us module ke GLOSSARY.md mein hai.

## 1. 25 fundamentals

| # | Concept | Is project mein |
|---|---|---|
| 1 | Linear algebra | Har jagah: attention = matrix multiply ([llm/model.py](llm/model.py)), mel filterbank = ek matrix ([stt/audio.py](stt/audio.py)), pseudo-inverse ([tts/audio.py](tts/audio.py)) |
| 2 | Probability & statistics | Softmax, sampling ([llm/generate.py](llm/generate.py)); CTC saare paths ki probability jodta hai ([stt/ctc.py](stt/ctc.py)) |
| 3 | Loss functions | Cross-entropy (LLM), CTC (STT), L1 + MSE (TTS), multi-resolution STFT (vocoder) |
| 4 | Gradient descent | Haath se: [llm/autograd.py](llm/autograd.py). AdamW: teeno `train.py` |
| 5 | Backpropagation | Scratch se scalar autograd: [llm/autograd.py](llm/autograd.py) |
| 6 | Neural networks | MLP haath se ([llm/autograd.py](llm/autograd.py)), phir transformers |
| 7 | Activation functions | tanh (autograd), SiLU / SwiGLU (LLM, STT), GELU (vocoder), sigmoid gate ([stt/model.py](stt/model.py)) |
| 8 | Regularization | Dropout, weight decay, SpecAugment ([stt/model.py](stt/model.py)), early stopping (best dev checkpoint) |
| 9 | Normalization | RMSNorm (LLM, STT, TTS), LayerNorm (vocoder), feature mean / std |
| 10 | CNNs | Conv subsampling + depthwise conv ([stt/model.py](stt/model.py)), ConvNeXt ([tts/vocoder.py](tts/vocoder.py)) |
| 11 | RNNs & LSTMs | Nahi banaya. Tacotron / DeepSpeech inpe the; hum transformer + conv pe hain |
| 12 | Embeddings | Token embedding (LLM), character + language embedding ([tts/model.py](tts/model.py)) |
| 13 | Attention | Causal ([llm/model.py](llm/model.py)) aur bidirectional + padding mask ([stt/model.py](stt/model.py)) |
| 14 | Transformer architecture | Decoder-only (LLM), encoder-only (STT), encoder + decoder blocks (TTS) |
| 15 | Tokenization | BPE scratch se ([llm/tokenizer.py](llm/tokenizer.py)); characters ([stt/text.py](stt/text.py), [tts/text.py](tts/text.py)) |
| 16 | Positional encoding | RoPE, tokens pe aur audio frames pe |
| 17 | Pretraining | LLM ka next-token pretraining ([llm/train.py](llm/train.py)). Speech ka self-supervised pretraining (wav2vec) nahi |
| 18 | Fine-tuning & LoRA | [llm/lora.py](llm/lora.py) |
| 19 | Generative models | Autoregressive (LLM), non-autoregressive (TTS). GAN / diffusion / flow nahi banaye |
| 20 | LLM inference & decoding | Temperature, top-k, top-p, KV cache ([llm/generate.py](llm/generate.py)); CTC greedy + beam ([stt/ctc.py](stt/ctc.py)) |
| 21 | RAG | Nahi |
| 22 | Vector databases | Nahi |
| 23 | AI agents & tool use | Voice agent ka state machine ([orchestrator/pipeline.py](orchestrator/pipeline.py)). Tool calling nahi |
| 24 | Multimodal AI | Audio ↔ text dono taraf, teen alag models se (cascaded). Ek multimodal model nahi |
| 25 | AI evaluation & systems | WER / CER, baseline ke saath ([stt/evaluate.py](stt/evaluate.py)); latency per stage ([orchestrator/agent.py](orchestrator/agent.py)) |

## 2. 50 AI engineering concepts

| # | Concept | Is project mein |
|---|---|---|
| 1 | Tokens | [llm/tokenizer.py](llm/tokenizer.py) |
| 2 | Context window | `ctx = 256` ([llm/model.py](llm/model.py)) |
| 3 | Embeddings | `tok_emb` ([llm/model.py](llm/model.py)) |
| 4 | Attention | [llm/model.py](llm/model.py), [stt/model.py](stt/model.py) |
| 5 | Transformers | Teeno models |
| 6 | Temperature | [llm/generate.py](llm/generate.py) |
| 7 | Top-p / sampling | [llm/generate.py](llm/generate.py) |
| 8 | System prompt | Gemma adapter mein ([orchestrator/models.py](orchestrator/models.py)). Scratch LLM system prompt nahi samajhta |
| 9 | Few-shot prompting | Nahi |
| 10 | Chain of thought | Nahi |
| 11 | Structured output | Nahi |
| 12 | Function / tool calling | Nahi |
| 13 | Agents | Voice agent ([orchestrator/pipeline.py](orchestrator/pipeline.py)) |
| 14 | Multi-agent systems | Nahi |
| 15 | MCP | Nahi |
| 16 | RAG | Nahi |
| 17 | Vector databases | Nahi |
| 18 | Hybrid search | Nahi |
| 19 | Chunking | Text stream ko vaakyon mein ([orchestrator/models.py](orchestrator/models.py)); RAG wala chunking nahi |
| 20 | Reranking | Nahi |
| 21 | Hallucination | 12 M model facts gadhta hai (LLM README). CTC ASR mein kam hota hai, kyun: STT glossary #149 |
| 22 | Grounding | Nahi |
| 23 | Evals | Dev / test split, WER / CER, round-trip checks |
| 24 | Golden datasets | FLEURS test split, jise training ne kabhi nahi dekha ([stt/evaluate.py](stt/evaluate.py)) |
| 25 | LLM-as-judge | Nahi. Uski jagah ASR-as-judge: TTS ki awaaz ko ASR se likhwana |
| 26 | Fine-tuning | [llm/lora.py](llm/lora.py) |
| 27 | LoRA | [llm/lora.py](llm/lora.py) |
| 28 | RLHF | Nahi |
| 29 | DPO | Nahi |
| 30 | Distillation | TTS ka data ek teacher model ne bola ([tts/data.py](tts/data.py)) |
| 31 | Speculative decoding | Nahi |
| 32 | KV cache | [llm/model.py](llm/model.py), test [llm/check.py](llm/check.py) |
| 33 | Latency | Har stage ka naap ([orchestrator/pipeline.py](orchestrator/pipeline.py)) |
| 34 | Cost per task | Sab local hai: cost = compute time. Real-time factor har model ka naapa hai |
| 35 | Semantic caching | Nahi |
| 36 | Model routing | LLM swap ek flag se (`--llm gemma`); automatic routing nahi |
| 37 | Guardrails | Nahi |
| 38 | Prompt injection | Nahi |
| 39 | Jailbreaks | Nahi |
| 40 | PII redaction | Nahi |
| 41 | Sandboxing tools | Nahi. Kareeb ki cheez: checkpoints `weights_only=True` se load, pickle ka code nahi chalta |
| 42 | Human-in-the-loop | Nahi |
| 43 | Observability / tracing | Har turn ke stage-wise numbers ([orchestrator/agent.py](orchestrator/agent.py)) |
| 44 | Streaming (SSE) | LLM tokens ka stream, Gemma ke liye SSE ([orchestrator/models.py](orchestrator/models.py)) |
| 45 | Multimodal | Cascaded audio ↔ text |
| 46 | Computer use | Nahi |
| 47 | Memory (short + long term) | Nahi: scratch agent har baari ko akela dekhta hai |
| 48 | State machines for agents | LISTENING / THINKING / SPEAKING ([orchestrator/pipeline.py](orchestrator/pipeline.py)) |
| 49 | Fallbacks + retries | Nahi |
| 50 | Shipping to real users | Abhi local hai. Browser ka raasta har README ke "kya nahi hai" mein |

## 3. STT: 40 concepts ([stt/GLOSSARY.md](stt/GLOSSARY.md) mein 152)

1/ Sampling rate & Nyquist
2/ WAV / PCM
3/ Frames, hop, overlap
4/ Fourier transform / STFT
5/ Hann window & spectral leakage
6/ Power spectrogram
7/ Mel scale & filterbank
8/ Log-mel spectrogram
9/ Feature normalization (CMVN)
10/ Text normalization
11/ Character vs subword vs phoneme targets
12/ Unicode code points & matras
13/ WER
14/ CER
15/ Edit distance
16/ Train / dev / test discipline
17/ Data leakage
18/ The alignment problem
19/ CTC & the blank symbol
20/ Collapse rules
21/ Forward algorithm (alpha)
22/ Log-space arithmetic
23/ Conditional independence assumption
24/ Greedy decoding
25/ Prefix beam search
26/ Forced alignment / Viterbi
27/ Encoder-only architecture
28/ Conv subsampling & frame rate
29/ Padding masks & padding invariance
30/ Conformer block (attention + convolution)
31/ Depthwise convolution
32/ SpecAugment
33/ Length bucketing & dynamic batches
34/ Overfit sanity test
35/ Blank plateau
36/ Real-time factor
37/ Baseline comparison
38/ Quantization
39/ Streaming vs batch ASR
40/ CTC vs transducer vs attention decoder (Whisper)

## 4. TTS: 40 concepts ([tts/GLOSSARY.md](tts/GLOSSARY.md) mein 128)

1/ Two-stage TTS (acoustic model + vocoder)
2/ Mel spectrogram as the meeting point
3/ Phase & why it is lost
4/ Griffin-Lim
5/ Inverse STFT
6/ Resampling & aliasing
7/ Text front-end
8/ Punctuation as prosody
9/ G2P vs raw characters
10/ Schwa deletion
11/ Language id
12/ Code-switching (Hinglish)
13/ TTS data requirements
14/ Knowledge distillation (teacher → student)
15/ Synthetic data
16/ Duration
17/ Monotonic alignment
18/ MAS
19/ Autoregressive vs non-autoregressive TTS
20/ Tacotron attention failures (skip / repeat)
21/ FastSpeech design
22/ Duration predictor
23/ Length regulator
24/ Prior loss
25/ Mel L1 loss & over-smoothing
26/ Stop-gradient
27/ Speed control
28/ Pitch / energy predictors
29/ One-to-many problem
30/ Flow matching / diffusion TTS
31/ Neural vocoder
32/ HiFi-GAN vs Vocos (iSTFT)
33/ Multi-resolution STFT loss
34/ GAN & discriminators
35/ ConvNeXt block
36/ Real-time factor
37/ MOS
38/ ASR round-trip intelligibility
39/ Voice cloning / speaker embedding
40/ Streaming TTS & time to first audio

## 5. Voice orchestration: 40 concepts ([orchestrator/GLOSSARY.md](orchestrator/GLOSSARY.md) mein 115)

1/ Cascaded pipeline (STT → LLM → TTS)
2/ Audio frames (20 ms)
3/ Audio callback & thread handoff
4/ VAD
5/ Noise floor
6/ Hysteresis
7/ Hangover & pre-roll
8/ Energy VAD vs neural VAD
9/ Turn-taking
10/ End-of-turn detection
11/ Pause vs end of turn
12/ False cutoff
13/ Acoustic vs semantic signals
14/ LM-based turn detector
15/ Min / max endpointing delay
16/ Latency vs cutoff trade-off
17/ State machine (listening / thinking / speaking)
18/ Token streaming
19/ SSE
20/ Sentence chunking
21/ Time to first audio
22/ TTFT
23/ Latency budget per stage
24/ Barge-in
25/ Cancellation & generation counter
26/ Event loop vs worker threads
27/ Plugin interfaces & model swap
28/ Utterance merging
29/ Echo & AEC
30/ Fake models for testing
31/ Scenario tests
32/ WebRTC
33/ UDP vs TCP for real-time
34/ Opus
35/ Jitter buffer
36/ ICE / STUN / TURN
37/ SFU
38/ Room / participant / track
39/ Speech-to-speech models
40/ Full-duplex
