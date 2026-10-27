# Voice AI stack, scratch se

Ek voice agent ke chaaron hisse khud banana, taaki har ek andar se samajh aaye. Upar wale folder ka agent
(`../agent.py`) bane-banaye models use karta hai; yahan wahi cheezein haath se banti hain.

| # | Module | Kya banega | Status |
|---|---|---|---|
| 1 | [llm/](llm/) | Tokenizer + transformer + training + LoRA + quantization | ban gaya |
| 2 | [stt/](stt/) | Waveform → mel spectrogram → Conformer encoder → CTC → text | ban gaya (Hindi test CER 0.110 pe 158 h; pehle 0.323 pe 23 h) |
| 3 | [tts/](tts/) | Characters → durations (MAS) → mel → Vocos vocoder → audio | ban gaya (Hindi CER 0.038, teacher 0.035; CPU pe 30x real time). Lesson 10: 24 kHz flow + emotions, 4 asli awaazein, Hindi CER 0.09 (asli recording 0.11) |
| 4 | [orchestrator/](orchestrator/) | Khud ka mini-LiveKit: VAD, end-of-turn, barge-in, streaming | ban gaya, teeno asli models ke saath chala (pehli awaaz 1.6–1.8 s; naya STT + flow TTS jude) |

Shuru se padhna ho to [docs/Awaaz-AI-Shuru-Se.pdf](docs/Awaaz-AI-Shuru-Se.pdf) (21 seedhiyan), poori guide
[docs/Voice-AI-Zero-Se.pdf](docs/Voice-AI-Zero-Se.pdf), interview ke 60 sawaal [docs/questions.md](docs/questions.md),
aur concepts ki list [CONCEPTS.md](CONCEPTS.md).

Har module mein: lesson-order files, `GLOSSARY.md` (100+ terms, code ke pointers ke saath), `check.py`, aur
ek README jisme "interview mein kaise samjhana" hai.

## Teen angle jo har module mein chalenge

- **Indic-first.** Data Hindi, Marathi aur Hinglish. Devanagari aur code-switching pehle din se.
- **Glass-box.** Har model ke andar dekhne ka tool (LLM mein `xray.py`).
- **Phone-size.** Har model ka size budget, quantization ke naape hue numbers ke saath.

## Do track

- **Scratch** (ye folder): seekhne ke liye. Chhote models, quality toy.
- **Fine-tune**: bane-banaye models (Gemma, Omnilingual ASR) ko apne data pe sudharna. Kaam layak quality.
  LoRA ka idea `llm/lora.py` mein hai; wahi bade model pe lagega.

WebRTC (ICE, DTLS, SRTP, SFU) scratch se nahi banega: wo networking hai, voice AI nahi. Orchestrator ki
glossary mein uske terms theory ke roop mein honge.
