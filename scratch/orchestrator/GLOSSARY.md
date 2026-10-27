# Orchestration glossary: beginner se advanced

Har term ke saamne wo jagah likhi hai jahan tune use khud code mein banaya hai (file ka naam, phir wo naam jo us file mein search karna hai).
`check.py` fail ho jaata hai agar koi pointer galat ho.

"Kahan" mein `theory` ka matlab: ye term is project ke code mein nahi hai, par voice-AI interview mein poocha jaata hai.
Section F (WebRTC) poora theory hai: wo networking hai, aur LiveKit jaisa framework use tumhare liye sambhalta hai.

## A. Real-time audio ki neenv

| # | Term | Seedhi bhasha mein | Kahan |
|---|---|---|---|
| 1 | Real-time audio | Audio jo aate hi process hona chahiye; der hui to baat-cheet toot jaati hai. | `agent.py: def run_mic` |
| 2 | Audio frame | Audio ka chhota tukda (20 ms) jo ek baar mein aage badhaya jaata hai. | `vad.py: FRAME = 320` |
| 3 | Frame size ka trade-off | Chhota frame = kam delay par zyada overhead. 10-20 ms voice ka standard. | `agent.py: blocksize=FRAME` |
| 4 | Audio callback | Sound driver apne thread se har naye frame pe ye function bulata hai. | `agent.py: callback = lambda` |
| 5 | Thread-safe handoff | Driver ke thread se event loop tak data surakshit pahunchana. | `agent.py: call_soon_threadsafe` |
| 6 | Queue | Frames ki line: producer (mic) aur consumer (pipeline) alag raftaar se chal sakte hain. | `agent.py: asyncio.Queue()` |
| 7 | Capture / playback | Mic se lena, speaker pe bajana. Dono alag streams. | `agent.py: sounddevice.InputStream` |
| 8 | Non-blocking playback | Audio bajana shuru karke turant lautna, taaki sunna chalu rahe. | `agent.py: sounddevice.play` |
| 9 | Decibel (dB) | Loudness ka log scale: 10 log10(energy). | `vad.py: 10 * math.log10` |
| 10 | Buffer underrun | Speaker ko samay pe audio na mile to "khat-khat" (glitch). | theory |
| 11 | Jitter | Packets ke aane ke samay ka upar-neeche hona. | theory |
| 12 | Sample-rate mismatch | STT 16 kHz chahta hai, TTS 22 kHz deta hai: har seema pe rate dekhna padta hai. | `agent.py: need {SR}` |

## B. VAD (lesson 1: vad.py)

| # | Term | Seedhi bhasha mein | Kahan |
|---|---|---|---|
| 13 | VAD | Voice activity detection: abhi koi bol raha hai ya nahi. | `vad.py: class VAD` |
| 14 | Energy VAD | Frame ki loudness dekh ke faisla. Sabse purana tareeka. | `vad.py: def is_voiced` |
| 15 | Noise floor | Kamre ki wo awaaz jab koi nahi bol raha. | `vad.py: self.floor_db` |
| 16 | Adaptive noise floor | Floor khud seekhta hai: neeche tezi se, upar dheere. | `vad.py: rate = 0.2 if level` |
| 17 | Margin / threshold | Floor se kitna upar ho to "bolna" maanein. | `vad.py: margin_db` |
| 18 | Hysteresis | Shuru hone aur rukne ke alag niyam, taaki state baar-baar na palte. | `vad.py: HYSTERESIS` |
| 19 | Start debounce | Itne ms lagataar awaaz ho tabhi "shuru". Ek click se trigger nahi. | `vad.py: start_ms` |
| 20 | Hangover / end silence | Itne ms chuppi ho tabhi "khatm". Shabdon ke beech ke gap se nahi. | `vad.py: end_ms` |
| 21 | Pre-roll buffer | Pichhle ~300 ms sambhal ke rakhna, kyunki "shuru" der se pata chalta hai. | `vad.py: self.pre_roll` |
| 22 | VAD event | State badalne ki khabar: start ya end (end ke saath poora utterance). | `vad.py: class Event` |
| 23 | Utterance | Do chuppiyon ke beech bola gaya ek tukda. | `vad.py: self.utterance` |
| 24 | Streaming VAD | Ek-ek frame pe chalta hai, poori recording ka intezaar nahi. | `vad.py: def feed` |
| 25 | Run counter | Lagataar kitne frames state ke khilaaf gaye. Hysteresis isi se banta hai. | `vad.py: self.run` |
| 26 | Energy VAD ki seema | Awaaz aur kisi bhi zor ke shor mein fark nahi kar sakta. | `vad.py: cannot do` |
| 27 | Neural VAD (Silero) | Chhota network jo speech ko shor se alag pehchanta hai. Asli agent yahi use karta hai. | `vad.py: Silero` |
| 28 | VAD ke false positive / negative | Shor ko bolna samajhna / dheemi awaaz chhod dena. | theory |
| 29 | Speech probability | Neural VAD har frame pe 0-1 ka number deta hai; us pe threshold lagta hai. | theory |

## C. Turn-taking (lesson 2: turn.py)

| # | Term | Seedhi bhasha mein | Kahan |
|---|---|---|---|
| 30 | Turn | Ek vyakti ki ek baari. Baat-cheet baariyon ka aana-jaana hai. | `pipeline.py: class Turn` |
| 31 | End-of-turn (EOT) detection | Ye tay karna ki saamne wala bol chuka. | `turn.py: class Endpointer` |
| 32 | Endpointing | Wahi, STT ki bhasha mein: utterance ka ant dhoondhna. | `turn.py: def delay` |
| 33 | Pause vs end | Vaakya ke beech ka rukna aur baari ka ant: dono chuppi hain. | `turn.py: MIDDLE of a turn` |
| 34 | False cutoff | Agent ne beech mein hi jawab de diya. Sabse buri galti. | `turn.py: def false_cutoffs` |
| 35 | Acoustic signal | Kitni der se chuppi hai. | `turn.py: acoustic` |
| 36 | Semantic signal | Jo kaha gaya wo poora lagta hai ya adhoora. | `turn.py: semantic` |
| 37 | LM-based turn detector | Language model se poochna: kya agla token "vaakya khatm" hai? | `turn.py: def lm_completeness` |
| 38 | End tokens | Newline aur . ? !: inki probability ka jod = "poora hone" ki sambhavna. | `turn.py: enders` |
| 39 | Min endpointing delay | Poora lagne wale vaakya ke baad kitni chuppi ka intezaar. | `turn.py: min_delay` |
| 40 | Max endpointing delay | Adhoora lagne wale vaakya ke baad. | `turn.py: max_delay` |
| 41 | Threshold ka matlab | Ye "jawab dein ya nahi" tay nahi karta, sirf kitna sabr rakhna hai. | `turn.py: threshold` |
| 42 | Latency vs cutoff trade-off | Jaldi jawab = zyada false cutoffs. Dono saath kam karna hi kala hai. | `turn.py: latency budget` |
| 43 | eot-bench | LiveKit ka benchmark: 300 / 600 ms budget pe false cutoffs. | `turn.py: eot-bench` |
| 44 | Audio-based turn model | Smart Turn jaisa model jo awaaz ke sur (prosody) se ant pehchanta hai. | theory |
| 45 | Prosody | Sur, lay, zor. "theek hai?" aur "theek hai." ka fark isi mein hai. | theory |
| 46 | Backchannel | "hmm", "haan" jaise shabd jo baari nahi lete. Inpe rukna nahi chahiye. | theory |
| 47 | Adaptive endpointing | Is user ke pichhle pauses dekh ke delay badalna. | theory |

## D. Pipeline aur streaming (lessons 3-4: models.py, pipeline.py)

| # | Term | Seedhi bhasha mein | Kahan |
|---|---|---|---|
| 48 | Cascaded pipeline | STT → LLM → TTS, teen alag models ek ke baad ek. | `pipeline.py: class Agent` |
| 49 | Plugin interface | Har model ek saade function ke peeche. Model badlo, pipeline wahi. | `models.py: plugins` |
| 50 | STT adapter | Scratch recognizer ko `samples -> text` function banana. | `models.py: def load_stt` |
| 51 | TTS adapter | `sentence -> (samples, rate)`. | `models.py: def load_tts` |
| 52 | LLM adapter | `text -> pieces ka iterator`. | `models.py: def load_llm` |
| 53 | Model swap | Wahi interface, doosra LLM (Gemma). Baaki code nahi badla. | `models.py: def gemma` |
| 54 | Token streaming | LLM ke tokens bante hi aage bhejna, poore jawab ka intezaar nahi. | `models.py: generate.stream` |
| 55 | SSE (server-sent events) | HTTP pe streaming: har line `data: {...}`, aakhir mein `[DONE]`. | `models.py: data: [DONE]` |
| 56 | Partial UTF-8 | Ek Devanagari akshar kai bytes ka hai; aadha aaye to rukna padta hai. | `models.py: �` |
| 57 | Sentence chunking | Token stream ko vaakyon mein baantna, har vaakya turant TTS ko. | `models.py: def sentences` |
| 58 | Minimum chunk | "Hi." akela synthesize ho to kata-kata lagta hai; chhote tukde jodo. | `models.py: min_chars` |
| 59 | System prompt | LLM ko diya gaya sthayi nirdesh: chhota bolo, markdown nahi. | `models.py: system =` |
| 60 | Script bridging | STT Devanagari likhta hai, LLM Roman padhta hai: beech mein transliteration. | `models.py: def romanize` |
| 61 | Inherent vowel | Har vyanjan ke saath chhupa "a"; shabd ke ant mein gir jaata hai. | `models.py: carries an "a"` |
| 62 | State machine | Agent hamesha teen mein se ek state mein: LISTENING, THINKING, SPEAKING. | `pipeline.py: self.state` |
| 63 | Barge-in | User beech mein bole to agent turant chup. | `pipeline.py: def _interrupt` |
| 64 | Playback cut | Bajta hua audio usi pal rokna. | `pipeline.py: self.stop()` |
| 65 | Cancellation | Tooti hui reply ka baaki kaam radd karna. | `pipeline.py: self._task.cancel()` |
| 66 | Generation counter | Har reply ka number; purana number wala kaam khud ruk jaata hai. | `pipeline.py: _generation` |
| 67 | Stale result | Thread ko maara nahi ja sakta; uska der se aaya nateeja phenk dete hain. | `pipeline.py: generation != self._generation` |
| 68 | Event loop | Ek thread jo chhote kaamon ko baari-baari chalata hai. | `pipeline.py: asyncio.create_task` |
| 69 | Blocking call in a thread | Model ka bhaari kaam worker thread mein, taaki loop mic sunta rahe. | `pipeline.py: asyncio.to_thread` |
| 70 | Async iterator bridge | Blocking generator ko event loop se ek-ek kadam chalana. | `pipeline.py: async def _aiter` |
| 71 | Utterance merging | User ruk ke phir bola: dono tukde ek hi baari mein. | `pipeline.py: _pending` |
| 72 | Playback clock | Pichhla vaakya kab tak bajega, taaki agla uske baad shuru ho. | `pipeline.py: _speaking_until` |
| 73 | Event hook | Pipeline ke andar kya hua, bahar batane ka raasta (log, UI). | `pipeline.py: on_event` |
| 74 | Interruption ke baad context | Asli agents yaad rakhte hain ki user ne kitna jawab suna tha. Hamara nahi rakhta. | theory |
| 75 | Chat history | Pichhli baariyan LLM ko dobara dena. Hamara scratch LLM ek baari ka hai. | theory |
| 76 | Tool / function calling | LLM ka bahar ke function bulana (order status, calendar). | theory |
| 77 | Filler / thinking sound | Jawab mein der ho to "hmm" bajana. | theory |
| 78 | Speech-to-speech model | Ek hi model audio leke audio deta hai (Moshi, GPT realtime). Pipeline nahi. | theory |
| 79 | Full-duplex | Sunna aur bolna ek saath. Cascaded pipeline ise barge-in se nakal karta hai. | theory |

## E. Latency aur naap-tol (pipeline.py, agent.py, check.py)

| # | Term | Seedhi bhasha mein | Kahan |
|---|---|---|---|
| 80 | Time to first audio | User ruka → agent ne bolna shuru kiya. Yahi mehsoos hota hai. | `pipeline.py: first_audio_ms` |
| 81 | Endpointing delay | VAD hangover + endpointer ka intezaar. Aksar sabse bada hissa. | `pipeline.py: endpoint_ms` |
| 82 | STT latency | Utterance → text. | `pipeline.py: stt_ms` |
| 83 | TTFT | Time to first token: LLM ka pehla tukda. | `pipeline.py: llm_first_ms` |
| 84 | TTS first-chunk latency | Pehle vaakya ka audio banne ka samay. | `pipeline.py: tts_first_ms` |
| 85 | Latency budget | Kul delay ko stages mein baantna; ~800 ms ke andar baat-cheet sahaj lagti hai. | `agent.py: def report` |
| 86 | Streaming ka fayda | Pehla audio ek vaakya ke baad, poore jawab ke baad nahi. | `check.py: Streaming` |
| 87 | Fake models | Models ki jagah tay delay wale nakli functions: logic akela test hota hai. | `check.py: def make` |
| 88 | Scenario test | Poori baat-cheet ka drishya chalake jaanchna: barge-in, resume. | `check.py: scenario_barge_in` |
| 89 | File-driven simulation | Mic ki jagah recording ko real-time raftaar se pipeline mein daalna. | `agent.py: def run_file` |
| 90 | Echo test | LLM ki jagah "jo suna wahi bolo": STT aur TTS ka akela test. | `agent.py: "echo"` |
| 91 | Real-time pacing | Simulation mein bhi 20 ms ka frame 20 ms mein dena, kyunki pipeline ghadi dekhta hai. | `agent.py: FRAME / SR` |
| 92 | P50 / P95 latency | Ausat nahi, percentile dekho: 5% baariyan kitni dheemi hain. | theory |
| 93 | Observability | Har turn ke stage-wise numbers production mein log karna. | theory |

## F. WebRTC aur transport (theory)

| # | Term | Seedhi bhasha mein | Kahan |
|---|---|---|---|
| 94 | Echo | Speaker ki awaaz mic mein wapas aana: agent khud ko sun leta hai. | `agent.py: hear ITSELF` |
| 95 | AEC | Acoustic echo cancellation: jo baja rahe ho use mic se ghata dena. | `agent.py: echoCancellation` |
| 96 | WebRTC | Browser ka real-time audio/video standard: kam delay, UDP pe. | theory |
| 97 | UDP vs TCP | UDP khoya packet dobara nahi bhejta: real-time ke liye der se behtar hai na aana. | theory |
| 98 | Opus | WebRTC ka audio codec: 20 ms frames, 6-510 kbps. | theory |
| 99 | Jitter buffer | Packets ko thoda rok ke sahi kram aur samay pe bajana. | theory |
| 100 | Packet loss concealment | Khoye packet ki jagah andaza bharna. | theory |
| 101 | ICE | Do peers ke beech raasta dhoondhne ki prakriya. | theory |
| 102 | STUN | Server jo batata hai tumhara public IP / port kya hai. | theory |
| 103 | TURN | Seedha raasta na mile to beech ka relay server. | theory |
| 104 | NAT traversal | Ghar ke router ke peeche se bahar wale se judna. ICE + STUN + TURN. | theory |
| 105 | SDP offer / answer | Dono taraf ka "main kya bhej / le sakta hoon" ka parcha. | theory |
| 106 | Signaling | SDP aur ICE candidates ka aadan-pradan. WebRTC ise tay nahi karta; aksar WebSocket. | theory |
| 107 | DTLS | UDP ke upar encryption ka handshake. | theory |
| 108 | SRTP | Encrypted audio/video packets. | theory |
| 109 | SFU | Selective forwarding unit: server jo har participant ka stream doosron ko aage bhejta hai. LiveKit yahi hai. | theory |
| 110 | Room / participant / track | LiveKit ke shabd: kamra, usme log (agent bhi ek participant), unke audio/video streams. | theory |
| 111 | Agent worker / dispatch | Process jo naye room ke liye agent ka job uthata hai. | theory |
| 112 | WebSocket | TCP pe do-tarfa connection. Simple par packet loss pe atakta hai (head-of-line blocking). | theory |
| 113 | AGC / noise suppression | Awaaz ka level barabar karna, background shor hatana. Browser ye bhi deta hai. | theory |
| 114 | Telephony (SIP, 8 kHz) | Phone calls ka raasta; audio 8 kHz, isliye phone pe STT mushkil. | theory |
| 115 | DTMF | Phone ke button ki tones. | theory |
