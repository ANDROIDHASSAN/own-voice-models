# Dimaag ka conductor: ek mini voice-agent framework, scratch se

Teen models (../stt, ../llm, ../tts) ko ek baatcheet mein jodna: VAD, end-of-turn, streaming, barge-in aur
cancellation. LiveKit Agents aur Pipecat jo kaam karte hain, uska chhota roop, asyncio aur numpy se. Koi
framework nahi.

## Padhne ka order

| # | File | Kya seekhoge | Chalao |
|---|---|---|---|
| 1 | [vad.py](vad.py) | Energy VAD: noise floor, hysteresis, pre-roll | `python vad.py` |
| 2 | [turn.py](turn.py) | End-of-turn: chup hone ka matlab "meri baari" nahi; LLM se completeness | `python turn.py` |
| 3 | [models.py](models.py) | Teen models teen saade functions ke peeche (plugin interface), Devanagari → Roman | (import hota hai) |
| 4 | [pipeline.py](pipeline.py) | State machine, streaming, sentence chunking, barge-in, generation counter | (import hota hai) |
| 5 | [agent.py](agent.py) | Mic se ya .wav se asli baatcheet, har stage ki latency | `python agent.py --wav q.wav` |
| 6 | [live.py](live.py) + [live.html](live.html) | Browser mein LiveKit jaisi live call: mic WebSocket pe, wahi Agent, barge-in | `python live.py` → http://127.0.0.1:8791 |
| 6b | [teacher.py](teacher.py) | LLM ki seat pe lesson script: teacher pehle bolti hai, sawaal poochti hai, jawab check karti hai. Pehli awaaz ~0.7 s (5 sawaal, simulated student, 5/5 pakde) | `python teacher.py` |
| 6c | [robust.py](robust.py) | Noise test: VAD ke galat start, lambe aur chhote jawab mile ya nahi, noise mein STT CER | `python robust.py --clips DIR --refs refs.tsv --musan noise.npy` |
| 8 | [nvad.py](nvad.py) | Apna neural VAD (GRU, 30 k params, streaming), energy VAD ke saath hybrid | `python nvad.py` |
| 9 | [eot.py](eot.py) | Apna turn detector (char GRU, 83 k params): vaakya poora hua ya nahi | `python eot.py` |
| 7 | [web.py](web.py) + [web.html](web.html) | Push-to-talk test page, aur sirf-TTS panel (awaaz, style, guidance) | `python web.py` → http://127.0.0.1:8790 |

[GLOSSARY.md](GLOSSARY.md) mein 115 terms hain (WebRTC, SFU, AEC jaise theory ke terms bhi). [check.py](check.py)
teen models ki jagah nakli models (pata delay ke saath) lagata hai, taaki state machine, barge-in aur
cancellation bina checkpoint ke, kuch second mein test hon.

## Ek turn ka safar

```
mic, 20 ms frames
   │  vad.py         awaaz floor se 12 dB upar, 100 ms tak → START (300 ms pre-roll ke saath)
   │                 500 ms chup → END
   ▼
utterance (samples)
   │  stt            → "मेरा ऑर्डर कब आएगा"                         worker thread mein
   │  turn.py        romanize → LLM: P(agla token "\n" . ? !)
   │                 lagta hai poora → 0.3 s ruko;  adhoora → 1.5 s tak ruko
   ▼
LISTENING → THINKING
   │  llm            tokens stream hote hain
   │  sentences()    pehla poora vaakya milte hi aage
   │  tts            vaakya 1 bol raha hai jab LLM vaakya 2 likh raha hai
   ▼
THINKING → SPEAKING → (khatam) → LISTENING
           user beech mein bola → playback band, reply ka number badla, baaki sab phenk do
```

## Design ke faisle, aur kyun

| Faisla | Kyun |
|---|---|
| Teen states, ek generation counter | Interrupt hua reply baad mein bol na pade: har stage check karta hai ki uska number abhi bhi current hai |
| Models worker threads mein | Model blocking hain. Event loop free rahe to mic padhta rahe, aur barge-in sun sake |
| Sentence chunking | TTS ko poora vaakya chahiye (intonation). Poore reply ka intezaar = lambi chuppi |
| LLM se end-of-turn | Sirf VAD: ya to beech mein kaat-ta hai ya sust lagta hai. LiveKit ka turn detector bhi yahi do signal jodta hai |
| Energy VAD, Silero nahi | Baseline jo dikhata hai ki neural VAD kis liye hai (music, horn, typing ko speech samajh leta hai) |
| Romanizer | STT Devanagari likhta hai, LLM Roman Hinglish padhta hai. Ek table aur ek rule (aakhri "a" girao) |
| Plugin functions | `stt(samples) → text`, `llm(text) → iterator`, `tts(text) → audio`. Gemma lagao, kuch aur nahi badalta |

## Interview mein kaise samjhana

**30 second mein:**
"I wrote the orchestration layer of a voice agent from scratch in asyncio: an energy VAD with an adaptive
noise floor and hysteresis, an end-of-turn detector that asks my own LLM how likely the transcript is to be
complete and waits 0.3 or 1.5 seconds accordingly, a three-state machine with streaming sentence-level TTS,
and barge-in with a generation counter so an interrupted reply can never speak later. The models run in
worker threads so the agent keeps listening while it thinks, and the logic is tested with fake models of
known latency."

| Sawaal | Chhota jawab | Dekho |
|---|---|---|
| Voice agent ki latency kaise naapte ho? | User chup hua → agent ki pehli awaaz. Total time nahi, "time to first audio" | `Turn.first_audio_ms` |
| Latency kahan jaati hai? | endpoint wait + STT + LLM ka pehla vaakya + TTS ka pehla vaakya | agent.py ki report |
| Streaming kyun? | Pehla vaakya bolte waqt doosra likha ja raha hai; user ko sirf pehle ka intezaar | `sentences` |
| Barge-in kaise? | VAD START jab THINKING/SPEAKING: playback band, task cancel, generation += 1 | `_interrupt` |
| Purana reply baad mein kyun nahi bolta? | Har stage apna generation number check karta hai | `_respond` |
| VAD hysteresis kyun? | Ek frame pe faisla jhatkedaar: "s" lagbhag chup, darwaza zor ka. Start aur end ke alag rules | vad.py |
| Pre-roll kyun? | START jaan-boojh ke 100 ms der se aata hai; pehla akshar na kate | `pre_roll_ms` |
| End-of-turn mein galti kya hoti hai? | False cutoff: user abhi bol raha tha, agent bol pada | `false_cutoffs` |
| Echo kyun problem hai? | Mic speaker ko sunta hai, agent khud ko interrupt karta hai. Browser ka AEC isse hatata hai | agent.py docstring |
| Blocking model event loop mein chalaya to? | Soch ke dauraan agent behra: barge-in kaam nahi karega | `asyncio.to_thread` |

**Sachcha jawab:** "Isme WebRTC hai?" Nahi. Ye ek process mein mic aur speaker se baat karta hai. Network ke
upar (ICE, DTLS, SRTP, SFU) LiveKit jaisa framework chahiye; glossary mein wo theory ke roop mein hain.

## Naape hue results

**End-of-turn** (`python turn.py`, 10 haath se likhe cases; ek misaal, benchmark nahi)

| Endpointer | False cutoffs | Asli end pe average latency |
|---|---|---|
| Sirf VAD, 0.3 s | 100% | 0.30 s |
| Sirf VAD, 1.5 s | 0% | 1.50 s |
| VAD + hamara LLM, 0.3 / 1.5 s | 0% | 1.02 s |

LLM ne adhoore vaakyon ko 0.39 ya kam diya ("mera order number hai" 0.389), par kuch poore vaakyon ko bhi kam
("haan theek hai" 0.014): 12 M ka model sawaal (? wale) pe pakka hai, chhote jawab pe nahi. Isliye average
1.02 s, 0.3 s nahi.

**VAD** (`python vad.py`, banaayi hui clip): speech 1.00 s pe shuru, START 1.08 s pe, pre-roll ne shuru ka
hissa bacha liya. 0.2 s ka pause utterance ko nahi todta. Noise floor -51 dB seekha.

**Asli teeno models ke saath** (`python agent.py --wav q.wav`, ek FLEURS Hindi test clip, M1 Pro CPU;
poora output `runs/e2e.log` aur `runs/e2e_barge_in.log` mein)

| | Endpoint wait | STT | LLM pehla vaakya | TTS pehla vaakya | Pehli awaaz |
|---|---|---|---|---|---|
| Ek turn | 1350 ms | 153 ms | 52 ms | 123 ms | 1776 ms |
| Barge-in: pehla reply, 0.5 s pe kata | 1396 ms | 105 ms | 22 ms | 62 ms | 1612 ms |
| Barge-in: naya utterance | 1467 ms | 33 ms | 11 ms | 78 ms | 1640 ms |
| `--llm echo` (naya 24 kHz flow TTS bolta hai) | 1422 ms | 78 ms | 0 ms | 232 ms | 1736 ms |

STT ab 158 ghante wala hai (`stt/runs/kaggle/kathbath.pt`). Wahi clip pehle "एबग बवरदल जर्ना ति कार्याल के
बाहार पटा" sunta tha, ab "एक बम गवर्नर जर्नल की कार्यालय के बाहर पटाता" (asli: "एक बम गवर्नर जनरल के कार्यालय
के बाहर फटा था"). TTS do hain: Devanagari naye flow model (24 kHz, Rasa female, `conv`) ko, Roman Hinglish
purane model ko, kyunki Rasa mein Hinglish nahi (`models.py: load_tts`). Hamara LLM Roman mein likhta hai, isliye
normal turn mein abhi bhi purani awaaz bolti hai; `--llm echo` naya flow TTS sunata hai. Flow TTS pehle vaakya
pe 4 Euler steps ki wajah se purane se ~100 ms dheema hai. Saare models milke 350 ms se kam lete hain; baaki
intezaar end-of-turn ka hai (1.35–1.47 s): turn detector aur Hinglish LLM ko Devanagari text abhi bhi adhoora lagta hai. LLM ke jawab
("haan, mujhe toh bilkul pasand nahi hai ...") 12 M ke model ki seema dikhate hain: lehja sahi, matlab nahi.
Sampling random hai, to har run ka jawab alag aata hai. Barge-in
mein reply kata, purana reply dobara nahi bola, aur naye utterance ka jawab aaya.

**Nakli models ke saath** (`python check.py`): pehli awaaz user ke chup hone ke 729 ms baad. Paanch scenario
pass: poora turn, barge-in, interrupt ke baad dobara, aakhri vaakya, model error.

**Shor mein: apna neural VAD, turn detector, noise wala STT** (`python robust.py`, 40 FLEURS Hindi test clips;
noise khud banaya: white / pink / brown / babble = 4 doosre log; MUSAN ka held-out 10 %, training mein kabhi nahi
suna. Kaggle T4: STT 8 epoch 46 min, VAD 12k steps 73 min, turn detector 2.4 min.)

| | Energy VAD | Neural v3 + hybrid |
|---|---|---|
| Sirf noise, galat start / min: pink, brown | 2.0, 2.0 | **0, 0** |
| Sirf noise, galat start / min: babble, MUSAN | 4.0, 4.8 | **2.0, 1.0** |
| Lambe vaakya mile, noise 5-10 dB (white, pink, brown, MUSAN) | 9-25 / 40 | **33-40 / 40** |
| Chhote jawab (0.4-0.8 s) mile: shaant / MUSAN 10 dB / brown 10 dB | 40 / 10 / 0 | 35 / **19** / **34** |

v1 VAD sirf lambe vaakyon pe train tha aur ek-shabd jawab pe kabhi nahi chala ("दिल्ली": p 0.52, ek frame);
v3 mein chhote tukde, sirf-chatter windows aur synthetic noise jode. Hybrid: frame tab voiced jab network
0.95 se upar, ya awaaz zor ki ho aur network 0.5 se upar.

| STT CER | 158 h | 158 h + noise |
|---|---|---|
| Saaf | 0.097 | **0.086** |
| Babble 10 dB / 5 dB | 0.400 / 0.723 | **0.154 / 0.270** |
| White 5 dB | 0.406 | **0.256** |
| MUSAN 5 dB | 0.258 | **0.155** |

STT confidence (likhne wale frames ki average probability) 0.5 se kam ho to transcript gira do: hiss / fan
0.24-0.40, asli awaaz 0.49-0.95. Turn detector, Hindi + Marathi dev (FLEURS): accuracy 0.902 (scratch LLM
0.500, wo Devanagari pe har baar 1.5 s rukta tha); poora vaakya turant 98 %, adhoore pe galat kaata 17.7 %.
Live call (`live.py`, simulated): pehli awaaz scratch 684 ms (pehle 1667), teacher 683-758 ms, barge-in chala.

## Kya nahi hai (aur kab jodna)

- **Peeche ki baatcheet / TV.** `scenarios.py` (live call, asli speed): 4/6 pass. Greeting ke beech bolna,
  MUSAN noise mein turn, 0.3 s khansi (agent nahi ruka), asli barge-in: pass. TV 8-12 dB dheema: fail, uski
  baat ka bhi jawab diya. Level gate (`pipeline.py: _near`, user se 10 dB dheemi awaaz = background) sirf
  bade farq pe kaam karta hai; user ke bolne se pehle koi reference nahi. LiveKit + Silero bhi TV ko jawab
  deta hai; ise Krisp BVC jaisa speaker-aware model theek karta hai. Agla kadam wahi.
- **Interruption ke niyam** (LiveKit jaise): 0.5 s se kam boli awaaz agent ko nahi rokti, aur uska jawab bhi
  nahi banta (`min_interrupt`, voiced frames se gina, VAD ke 0.5 s hangover se nahi).
- **Chhote jawab ke liye turn detector.** "हाँ ठीक है" ko 0.08 deta hai: training mein kahaniyan aur padhe
  hue vaakya the, baatcheet nahi. LiveKit ka detector pichhla sawaal bhi padhta hai; hamara sirf user ka text.
- **Noise cancellation (denoiser).** Browser ka `noiseSuppression` chalta hai; apna model abhi nahi.
- **Echo cancellation.** Headphones chahiye. Browser mein `echoCancellation: true` muft milta hai.
- **Streaming STT.** Utterance khatam hone ke baad hi STT chalta hai (../stt encoder poori clip dekhta hai).
