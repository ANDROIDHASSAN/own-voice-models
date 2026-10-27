# Kaan: ek speech recognizer, scratch se

4.6 million parameter ka model jo Hindi, Marathi aur English sunke likhta hai. Mel spectrogram, Conformer
encoder, CTC loss, beam search, forced alignment aur quantization: sab is folder mein haath se likha hai.
CTC ka forward algorithm bhi khud likha hai, aur check.py usse PyTorch ke saath value aur gradient tak milata hai.

Ye Omnilingual ASR ki jagah nahi lega (neeche numbers). Iska kaam hai sikhana ki ASR andar se kaise chalta hai.

## Padhne ka order

| # | File | Kya seekhoge | Chalao |
|---|---|---|---|
| 1 | [audio.py](audio.py) | Waveform → STFT → mel filterbank → log-mel | (import hota hai) |
| 2 | [text.py](text.py) | Characters ki vocab, blank, normalization, WER / CER | (import hota hai) |
| 3 | [data.py](data.py) | FLEURS se training arrays tak | `python data.py` |
| 4 | [ctc.py](ctc.py) | CTC: bina timing ke (audio, text) pe training; greedy, beam search, alignment | (import hota hai) |
| 5 | [model.py](model.py) | Conformer encoder: subsampling, attention + convolution + feed-forward, SpecAugment | (import hota hai) |
| 6 | [train.py](train.py) | Overfit test, length bucketing, warmup, blank plateau | `python train.py --overfit 16`, phir `python train.py` |
| 7 | [transcribe.py](transcribe.py) | Ek .wav ya mic se text | `python transcribe.py file.wav` |
| 8 | [evaluate.py](evaluate.py) | Test CER / WER, aur ek bade model se tulna | `python evaluate.py --baseline 50` |
| 9 | [shrink.py](shrink.py) | Quantization: fp32 se int2 tak, har ek ki keemat | `python shrink.py` |
| 10 | [xray.py](xray.py) | Har 40 ms frame pe model kya soch raha hai | `python xray.py` |

[GLOSSARY.md](GLOSSARY.md) mein 152 terms hain, har ek ke saath code ki jagah. [check.py](check.py) wo cheezein
test karta hai jo galat hon to sab galat ho jaata hai. Sab commands is folder se: `../../.venv/bin/python <file>`.

## Ek clip ka safar

```
16 kHz waveform (12 s = 192,000 numbers)
   │  audio.py      25 ms window, 10 ms hop, FFT 400 → 80 mel bands → log
   ▼
log-mel (1200, 80)                     100 frames / second
   │  model.py      normalize → SpecAugment (sirf training) → 2 conv, stride 2 each
   ▼
(300, 192)                             25 frames / second, ek frame = 40 ms
   │  8 baar:   x = x + attention(norm(x))      poori clip dekhta hai
   │            x = x + convolution(norm(x))    paas ke 15 frames (0.6 s)
   │            x = x + feed_forward(norm(x))
   ▼
(300, 117) log-probabilities           har frame pe har character + blank
   │  ctc.py        greedy: har frame ka top → repeat jodo → blank hatao
   ▼
"नमस्ते दुनिया"
```

## Design ke faisle, aur kyun

| Faisla | Kyun |
|---|---|
| Characters, BPE nahi | 23 ghante data mein har BPE token ke kam examples aate. Devanagari character lagbhag ek awaaz hai |
| CTC, decoder nahi | Ek pass mein saare frames, koi loop nahi: tez, aur hallucination nahi kar sakta (har output ek frame se juda hai) |
| Encoder-only | Har frame aage-peeche dono dekhta hai. Streaming nahi, par offline accuracy zyada |
| Conformer (attention + conv) | Ek consonant 3 frame ka local pattern hai, ek shabd ka matlab seconds door ke context se aata hai |
| 4x subsampling | Attention ka kharcha length ka square hai. 100 → 25 frames / second = 16x sasta |
| RoPE, RMSNorm, SwiGLU | Wahi pieces jo ../llm mein hain |
| SpecAugment | Sirf 23 ghante data: model ko kisi ek band ya pal pe bharosa nahi karne dete |
| Length bucketing | Ek batch mein barabar lambi clips, padding pe compute barbaad nahi |
| FLEURS | CC-BY licence, teeno bhashayein, ek jaisa format |

## Interview mein kaise samjhana

**30 second mein:**
"I built a CTC speech recognizer from scratch for Hindi, Marathi and English: log-mel features, a 4.6 million
parameter Conformer-style encoder with 4x convolutional subsampling, SpecAugment, and my own CTC forward
algorithm, verified against PyTorch's to the gradient. It trains on 23 hours of FLEURS on a laptop, reaches
a test character error rate of 32% on Hindi, runs ten times faster than a 300 million parameter baseline,
and quantizes to 4.6 MB in int8 with no measurable loss."

| Sawaal | Chhota jawab | Dekho |
|---|---|---|
| Mel spectrogram kyun, raw audio kyun nahi? | Kaan low frequency mein zyada fark sunta hai; mel bands wahi karte hain aur 16,000 numbers/s ko 8,000 kar dete hain | `mel_filterbank` |
| CTC kya problem solve karta hai? | Text hai, par kaun sa akshar kab bola gaya, ye nahi. CTC saare valid alignments ka jod maximize karta hai | `ctc_loss` |
| Blank kyun chahiye? | "aa" jaise double akshar alag karne ke liye, aur "kuch nahi bola" frames ke liye | `BLANK` in text.py |
| CTC ki kamzori? | Har frame independent maanta hai: spelling ke liye language model nahi. Isliye WER zyada | neeche WER |
| Pehle loss plateau kyun? | Sabse sasta jawab "har frame blank" hai. Model wahan atakta hai, phir achanak spell karna seekhta hai | train.log epoch 1–11 |
| Overfit test kyun pehle? | 16 clips yaad nahi kar sakta to bug hai: lengths, masking, labels | `--overfit` |
| Padding se output badalna chahiye? | Nahi. check.py ek clip akele aur batch mein milata hai | check.py |
| CER aur WER mein fark? | Ek galat akshar poora shabd galat kar deta hai. Isliye WER hamesha CER se kaafi upar | text.py |
| WER 1 se zyada kaise? | Insertions bhi gine jaate hain: model ne zyada shabd bol diye | Marathi 1.09 |
| Whisper se fark? | Whisper encoder-decoder hai (680k ghante, autoregressive, hallucinate kar sakta hai). Ye encoder + CTC hai | model.py docstring |

**Sachcha jawab:** "Ye production mein chalega?" Nahi. Hindi mein har teesra akshar galat hai. 23 ghante data
aur 4.6 M parameters ki yahi seema hai; Omnilingual 1,600 bhashaon ke data pe 300 M parameters hai.

## Naape hue results

Apple M1 Pro, MPS. 4.61 M parameters, 7,594 clips = 23.5 ghante FLEURS, 60 epochs, 208 minute.
Sabse accha checkpoint epoch 37 ka (dev CER 0.406); uske baad train loss girta raha (0.74 → 0.49) aur dev
CER badha (0.42–0.45): overfitting.

**Test split, greedy** (`python evaluate.py`)

| Bhasha | Clips | Ghante | CER | WER |
|---|---|---|---|---|
| Hindi | 418 | 1.34 | 0.323 | 0.747 |
| Marathi | 1015 | 3.86 | 0.402 | 1.091 |
| English | 647 | 1.78 | 0.446 | 0.923 |

**Zyada data: + Kathbath Hindi** (`python data.py --kathbath DIR`, 99,345 clips = 158.4 ghante; Kaggle T4,
fp16, 15 epochs, 1.3 ghante; dev/test wahi FLEURS). Wahi 4.61 M model, sirf data 7 guna. Sabse accha epoch
14 (dev CER 0.237, pehle 0.406). Test split, greedy (`python evaluate.py --ckpt runs/kaggle/kathbath.pt`):

| Bhasha | CER (23 h) | CER (158 h) | WER (23 h) | WER (158 h) |
|---|---|---|---|---|
| Hindi | 0.323 | **0.110** | 0.747 | **0.314** |
| Marathi | 0.402 | **0.193** | 1.091 | **0.623** |
| English | 0.446 | **0.369** | 0.923 | **0.860** |

Kathbath sirf Hindi hai, phir bhi Marathi ka CER aadha hua: dono Devanagari aur milte-julte sound share karte
hain. English sabse kam sudhri, kyunki uska koi naya data nahi aaya. Dev CER epoch 15 tak gir raha tha
(overfitting nahi), to aur epochs se aur sudhar sambhav hai. Isi model se orchestrator ab chalta hai.

**Ek bade model ke saamne** (pehli 50 test clips har bhasha, CPU, 2 threads; CER / WER / real-time factor)

| | Hamara, greedy | Hamara, beam 8 | Omnilingual ASR 300 M |
|---|---|---|---|
| Hindi | 0.333 / 0.766 / 0.023 | 0.330 / 0.763 / 0.026 | 0.095 / 0.276 / 0.238 |
| Marathi | 0.400 / 1.098 / 0.029 | 0.397 / 1.105 / 0.031 | 0.115 / 0.453 / 0.312 |
| English | 0.454 / 0.911 / 0.014 | 0.449 / 0.910 / 0.022 | 0.061 / 0.178 / 0.158 |

Hamara model 3 se 7 guna zyada galti karta hai, aur 7 se 11 guna tez hai. Beam search language model ke bina
lagbhag kuch nahi deta (0.003 CER): CTC ke outputs pehle se peaky hain.

**Quantization** (`python shrink.py`, dev split)

| Format | Size | Dev CER | Fark |
|---|---|---|---|
| fp32 | 18.5 MB | 0.4060 | |
| fp16 | 9.2 MB | 0.4060 | 0 |
| int8, per-channel | 4.6 MB | 0.4064 | +0.0004 |
| int6, per-channel | 3.5 MB | 0.4069 | +0.0009 |
| int4, per-channel | 2.3 MB | 0.4177 | +0.012 |
| int4, per-tensor | 2.3 MB | 0.4452 | +0.039 |
| int3, per-channel | 1.7 MB | 0.4906 | +0.085 |
| int2, per-channel | 1.2 MB | 0.9846 | toot gaya |

LLM jaisi hi kahani: int8 muft, 4-bit pe per-channel zaroori, 2-bit pe khatam.

## Jo galat gaya

- **Blank plateau.** Pehla run (warmup 1000 steps) 7 epochs baad bhi dev CER 0.78 pe atka tha, "sab blank"
  waala jawab. Warmup 3000 aur lr 7e-4 ke saath bhi plateau 11 epochs chala, phir epoch 12 se CER girne laga
  (0.70 → epoch 16 pe 0.55). Pehle run ko roka na hota to shayad wo bhi nikal jaata; ye pakka nahi kaha ja sakta.
- **WAV float format.** Python ka `wave` module 32-bit float WAV nahi padhta. `read_wav` khud chunks padhta hai.
- **CTC gradient check.** Log-probs pe compare kiya to fail hua; PyTorch ka CTC backward log-softmax ke peeche
  hi exact hai. Logits pe compare karna sahi test hai.

## Kya nahi hai (aur kab jodna)

- **Language model / KenLM.** Beam search ke saath shabdon ki spelling sudharta. WER pe sabse bada fayda yahin hai.
- **Streaming.** Encoder poori clip dekhta hai. Streaming ke liye chunked attention chahiye (neeche orchestrator utterance ke baad hi STT chalata hai).
- **Tempo perturbation.** `train.py --stretch 0.1` likha hai aur check.py mein test hai, par us run se train nahi kiya.
- **Zyada data.** Kathbath (Hindi + Marathi, ~150 ghante har ek) is model ko sabse zyada sudharta.
