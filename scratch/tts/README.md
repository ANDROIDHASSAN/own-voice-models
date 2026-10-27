# Zubaan: ek text-to-speech model, scratch se

Hindi, Marathi, English aur Roman Hinglish bolne wala TTS: 4.7 M ka acoustic model (characters → mel) aur
2.8 M ka vocoder (mel → awaaz). Alignment (MAS), duration predictor, length regulator, Griffin-Lim aur
Vocos-style iSTFT vocoder: sab is folder mein haath se likha hai.

Ye Supertonic ki jagah nahi lega. Iska kaam hai sikhana ki TTS andar se kaise chalta hai.

## Padhne ka order

| # | File | Kya seekhoge | Chalao |
|---|---|---|---|
| 1 | [audio.py](audio.py) | Mel se wapas awaaz: phase ki problem, Griffin-Lim, resampling | `python audio.py` |
| 2 | [text.py](text.py) | Text front-end: punctuation rakhna, language id, schwa deletion | `python text.py` |
| 3 | [data.py](data.py) | Distillation: teacher model padhta hai, plus asli Marathi recordings | `python data.py` |
| 4 | [align.py](align.py) | MAS: kaun sa akshar kitne frame bola gaya, bina kisi ke bataye | `python align.py` |
| 5 | [model.py](model.py) | FastSpeech-style: encoder → durations → length regulator → decoder, ek pass mein | (import hota hai) |
| 6 | [train.py](train.py) | Teen loss (mel, prior, duration), overfit test | `python train.py --overfit 16`, phir `python train.py` |
| 7 | [vocoder.py](vocoder.py) | Vocos: magnitude + phase predict karo, ek iSTFT | `python vocoder.py` |
| 8 | [speak.py](speak.py) | Sentence → .wav | `python speak.py "नमस्ते, आप कैसे हैं?"` |
| 9 | [evaluate.py](evaluate.py) | Machine sun ke samjhe? Student banam teacher, ek hi judge | `python evaluate.py` |

[GLOSSARY.md](GLOSSARY.md) mein 128 terms hain, har ek ke saath code ki jagah. [check.py](check.py) wo cheezein
test karta hai jo galat hon to sab galat ho jaata hai. Sab commands is folder se: `../../.venv/bin/python <file>`.

## Ek vaakya ka safar

```
"नमस्ते, आप कैसे हैं?"  + language id (hi)
   │  text.py       normalize, punctuation rakho (comma = chhota pause), ek id har akshar
   ▼
20 character ids
   │  model.py      embedding + language → encoder (4 blocks)          (20, 192)
   │                duration predictor: har akshar kitne frame          [3, 7, 5, 9, ...]
   │                length regulator: har vector utni baar copy         (170, 192)
   │                decoder (4 blocks) → linear                         (170, 80) mel
   ▼
mel, 86 frames / second
   │  vocoder.py    conv → 6 ConvNeXt blocks → magnitude (513) + phase (513) har frame
   │                → inverse STFT (seekhna kuch nahi)
   ▼
22,050 Hz awaaz, 2 s
```

Do forward pass, koi loop nahi. LLM ko har token ke liye ek pass chahiye tha; isliye TTS CPU pe bhi real time se tez ho sakta hai.

## Design ke faisle, aur kyun

| Faisla | Kyun |
|---|---|
| Characters, phonemes nahi | Devanagari lagbhag phonetic hai; G2P dictionary ki zaroorat nahi. Keemat: schwa deletion model ko khud seekhna padta hai (कमल = "kamal") |
| Language id | "to" English mein "too", Hinglish mein "toh". Ek alphabet, do uchcharan |
| Non-autoregressive (FastSpeech) | Tacotron frame-by-frame banata tha, attention fisla to shabd chhoot ya dohra jaate the. Durations pehle tay, to ye ho hi nahi sakta |
| MAS, bahar ka aligner nahi | Glow-TTS ka tareeka: training ke dauraan khud durations dhoondhta hai. Montreal Forced Aligner jaisa tool nahi chahiye |
| Vocos, HiFi-GAN nahi | Frame rate pe hi rehta hai, aakhir mein ek iSTFT. Browser / phone CPU ke liye kaafi sasta |
| Multi-resolution STFT loss | Do waveform alag phase ke saath ek jaisi sun sakti hain; sample-by-sample compare galat hoga |
| Teacher se distillation | Ek awaaz, saaf, jitna chahe data. Sabse chhote published TTS models aise hi bante hain |
| Marathi asli recordings | Supertonic Marathi nahi bolta. OpenSLR 64, ~3 ghante, CC BY-SA 4.0 |
| Encoder / decoder = STT ka Block | Text aur speech dono ko local shape aur context chahiye |

**Licence:** teacher Supertonic 3 Open RAIL-M hai, to student uska "derivative" hai: uski use restrictions
hamare weights ke saath chalti hain.

## Interview mein kaise samjhana

**30 second mein:**
"I built a non-autoregressive TTS from scratch for Hindi, Marathi, English and Hinglish: a character-level
FastSpeech-style acoustic model that learns its own durations with Monotonic Alignment Search, a duration
predictor and length regulator, and a Vocos-style vocoder that predicts magnitude and phase and uses an
inverse STFT, trained with a multi-resolution spectral loss. Training data is 11 hours distilled from a 99
million parameter teacher plus real Marathi recordings, and I evaluate intelligibility by running the output
through a strong speech recognizer next to the teacher's audio."

| Sawaal | Chhota jawab | Dekho |
|---|---|---|
| TTS ke do hisse kyun? | Text → mel (kya bolna, kitni der) aur mel → awaaz (phase wapas laana). Alag problems | model.py, vocoder.py |
| Mel se awaaz mushkil kyun? | Mel mein phase nahi hai. Griffin-Lim andaza lagata hai, metallic lagta hai | `griffin_lim` |
| Duration kahan se aati hai? | MAS: monotonic raasta jo har frame ko ek akshar de, dynamic programming | `mas` in align.py |
| Autoregressive banam non-autoregressive? | AR: frame t, frame t-1 se. NAR: saare frames ek saath; tez, skip/repeat nahi | model.py docstring |
| Mel L1 loss ki kamzori? | Saare sambhav spectrograms ka average: dhundhla. Isliye GAN ya diffusion / flow matching | train.py docstring |
| Speed kaise badlein? | Predicted durations ko scale karo | `synthesize(..., speed)` |
| TTS kaise evaluate karte ho? | MOS (log sunte hain). Automatic: ASR se CER, teacher ke saath | evaluate.py |
| Distillation kya hai? | Bada model teacher, chhota uski nakal seekhta hai. Teacher se behtar nahi ho sakta | data.py |
| 22 kHz kyun, STT 16 kHz? | Samajhne ko 8 kHz kaafi; natural awaaz ke liye "s", "sh" ka 8 kHz se upar wala hissa chahiye | audio.py |
| Vocoder ka GAN kyun? | Ek doosra network jo asli/nakli pehchaanta hai; aakhri buzz hatata hai | vocoder.py docstring |

## Naape hue results

Apple M1 Pro, MPS, ek doosri training ke saath GPU baant ke. Data: 8,004 clips = 10.9 ghante.

**Overfit test** (`python train.py --overfit 16 --epochs 300`, 16 vaakya, 3 minute): mel loss 0.52 → 0.084,
duration error 1.02 → 0.000, predicted / asli length 0.74 → 1.00. Pipeline sahi juda hai.

**Acoustic model** (4.72 M, 60 epochs, 26,400 steps, 154 minute)

| Epoch | Train mel | Dev mel | Dev duration | Length |
|---|---|---|---|---|
| 1 | 0.517 | 0.416 | 1.081 | 0.57 |
| 10 | 0.255 | 0.270 | 0.372 | 0.80 |
| 20 | 0.231 | 0.254 | 0.314 | 0.85 |
| 31 | 0.216 | **0.249** | 0.301 | 0.91 |
| 60 | 0.196 | 0.252 | 0.307 | 0.92 |

Epoch 31 ke baad dev ruk gaya aur train girta raha: 11 ghante data pe yahi seema hai. `acoustic.pt` epoch 31
ka hai. Length 0.91 ka matlab: model teacher se ~9% tez bolta hai (durations log-scale pe seekhi, jo thodi
kam aankti hain). `speak.py --speed 0.9` se barabar hota hai.

**Vocoder** (2.79 M, 40,000 steps, 45 minute): dev spectral loss 3.41 (step 2,000) → 2.71 (10k) → 2.44 (40k).

**Samajh mein aata hai?** (`python evaluate.py`: 40 dev vaakya har bhasha, Omnilingual ASR 300 M judge, CPU 2 threads)

| Bhasha | Student CER | Teacher CER | Real-time factor |
|---|---|---|---|
| Hindi | 0.038 | 0.035 | 0.034 |
| Marathi | 0.090 | 0.043 (asli recordings) | 0.027 |
| English | 0.035 | 0.016 | 0.021 |
| Hinglish | 0.295 | 0.293 | 0.029 |

Hindi mein 7.5 M ka student 99 M ke teacher ke lagbhag barabar samajh aata hai. Marathi mein fark sabse
zyada, kyunki wahan "teacher" asli insaan hai aur data sirf 2.3 ghante. Hinglish ka 0.29 judge ki galti hai
(Roman Hinglish ki koi fixed spelling nahi): student aur teacher barabar hain, wahi number dekhna hai.
Real-time factor 0.03 = CPU pe real time se ~30 guna tez.

Ye naturalness nahi naapta: robotic awaaz bhi saaf samajh aa sakti hai. Uske liye `runs/eval_*.wav` suno.

## Lesson 10: robotic se expressive

**Robotic kyun lagta hai, naap ke** (10 dev vaakya har bhasha; asli awaaz / asli mel → hamara vocoder / poora TTS):

| | Pitch movement | Mel GV | Frame-to-frame badlaav | Harmonics ki saafi |
|---|---|---|---|---|
| Asli | 2.67 | 1.78 | 0.526 | 3.85 |
| Asli mel → vocoder | 2.61 | 1.75 | 0.495 (−6%) | 3.72 (−3%) |
| Poora TTS (L1) | 2.47 (−8%) | 1.69 | 0.435 (−17%) | 3.31 (−14%) |

Sabse badi wajah acoustic model hai: L1 loss saare tareekon ka average sikhata hai, jo dhundhla hota hai.
Vocoder ka hissa chhota hai. Data bhi ek wajah hai: teacher ki nakal teacher se zyada expressive nahi ho sakti.

**Teen badlaav** (sab flags ke peeche hain; purana `acoustic.pt` waisa hi chalta hai):

| Badlaav | Paper | Kahan |
|---|---|---|
| L1 decoder ki jagah flow matching (OT-CFM): noise se ek tez, asli jaisa mel | Matcha-TTS (Mehta 2024): MOS 3.84, VITS 3.71, FastSpeech2 3.29 | `model.py: velocity`, `train.py --flow` |
| Speaker + style/emotion embedding, classifier-free guidance (`--guidance 2` = zyada emotion) | F5-TTS, Ho 2022 | `model.py: synthesize` |
| Vocoder ke saamne critics: multi-period + multi-resolution, hinge, feature matching | Vocos, HiFi-GAN | `vocoder.py --gan` |
| Asli insaanon ka expressive data: Rasa (Hindi + Marathi, female + male, 6 emotions + styles, CC BY 4.0) | AI4Bharat 2024 | `data.py --rasa hi mr` |

**Tests** (`python check.py`):
- flow loss ek seekhne layak target pe 2.21 → 0.77
- sampling har clip ko uske apne target ke paas laata hai (doori 0.45; doosre targets se 1.2+)
- guidance 1 = bina guidance
- critic ki loss vocoder ko train nahi karti
- purana checkpoint load hota hai

**Smoke tests:** asli data pe flow model (12.45 M) aur GAN (lr 1e-4 pe dev loss 2.44 pe sthir) dono chalte hain.
CPU pe 3.4 s ka vaakya: 10 steps RTF 0.40, **4 steps 0.17** (default), 2 steps 0.09.

**Raat ki commands** (is folder se; pehle HF pe Rasa ke terms accept karo):

```
uv run --no-project --python ../../.venv/bin/python --with pyarrow python -u data.py --rasa hi mr > runs/data_rasa.log 2>&1
caffeinate -is ../../.venv/bin/python -u train.py --flow --d-model 256 --layers 6 --heads 4 --epochs 8 --name flow > runs/train_flow.log 2>&1
caffeinate -is ../../.venv/bin/python -u vocoder.py --gan --steps 10000 --batch 16 > runs/vocoder_gan.log 2>&1
../../.venv/bin/python evaluate.py --model flow --vocoder vocoder_gan > runs/eval_flow.log 2>&1
../../.venv/bin/python speak.py "आज मैं बहुत खुश हूँ!" --model flow --vocoder vocoder_gan --style happy --guidance 2
```

**Plan A (chuna hua): 24 kHz, sirf Rasa.** `TTS_SR=24000` har command ke aage: 100 mel bands, data `data24k/`,
checkpoints `runs24k/` (22.05 kHz wale models waise hi rehte hain). Rasa ke styles (asli data se): `conv`,
`wiki`, `book`, `news`, `indic`, emotions `happy`, `sad`, `anger`, `fear`, `surprise`, `disgust`, aur commands
`alexa`, `bb`, `digi`, `umang`, `proper noun`. Awaazein: `rasa_hi_female`, `rasa_hi_male`, `rasa_mr_female`,
`rasa_mr_male`. Is version mein English / Hinglish nahi (Rasa mein nahi). Training Kaggle pe: [../kaggle/README.md](../kaggle/README.md).

```
TTS_SR=24000 ../../.venv/bin/python speak.py "आज मैं बहुत खुश हूँ!" --model flow --style happy --guidance 2
```

**Plan A ka naapa hua result** (Kaggle 2× T4, fp16; 12.47 M flow model, 27,127 clips = 49.3 ghante, 30
epochs ~5.5 ghante; vocoder 40k steps 30 min; GAN fine-tune 20k steps 1.8 ghante). Checkpoints
`runs24k/final/`. Sabse accha flow epoch 16 (dev mel 0.231; flow ka dev loss random t pe naapa jaata hai, isliye
epoch-dar-epoch 0.23–0.32 uchhalta hai). `evaluate.py --model final/flow --vocoder final/vocoder --n 5`
(sirf 5 dev vaakya har awaaz: rough number):

| Awaaz | CER: Mac, epoch 4 | CER: Kaggle, 30 epochs | CER: asli recording |
|---|---|---|---|
| rasa_hi_female | 0.19 | **0.094** | 0.112 |
| rasa_hi_male | 0.29 | **0.084** | 0.212 |
| rasa_mr_female | 0.22 | **0.063** | 0.036 |
| rasa_mr_male | 0.21 | **0.136** | 0.109 |

GAN vocoder CER nahi badalta (0.094 / 0.084 / 0.067 / 0.140): wo saafi ke liye nahi, metallic pan ke liye hai,
aur uski spectral dev loss 2.55 → 2.77 gayi (GAN ka normal: average ki jagah tez detail). Faisla sunke.
CPU pe RTF ~0.15. Samples: `runs24k/final/samples/`.

Sachchi baat: ElevenLabs aur Bulbul hazaaron ghante aur GPU clusters pe bante hain. Ye ~50 ghante data aur
do muft T4 hain. Flow matching aur asli data se awaaz robotic se kaafi aage jaani chahiye, par us level ki nahi.
Number `evaluate.py` aur sunke hi pakke honge.

## Kya nahi hai (aur kab jodna)

- **Flow model aur GAN vocoder abhi train nahi hue.** Code aur tests tayyar hain; upar ki commands se.
- **Pitch / energy predictor.** FastPitch inhe alag predict karta hai; yahan flow decoder khud seekhta hai.
- **Numbers → shabd.** Digits wale vaakya data se hata diye; "25" bolna nahi aata.
- **Multi-speaker.** Ek awaaz (teacher) aur Marathi ki ek female speaker.
