# TTS glossary: beginner se advanced

Har term ke saamne wo jagah likhi hai jahan tune use khud code mein banaya hai (file ka naam, phir wo naam jo us file mein search karna hai).
`check.py` fail ho jaata hai agar koi pointer galat ho.

"Kahan" mein `theory` ka matlab: ye term is project ke code mein nahi hai, par interview mein poocha jaata hai.
Jo terms STT wale glossary mein aa chuke hain (sampling rate, STFT, mel, padding mask...) wo yahan dobara nahi samjhaye.

## A. Awaaz wapas banana (lesson 1: audio.py)

| # | Term | Seedhi bhasha mein | Kahan |
|---|---|---|---|
| 1 | Speech synthesis / TTS | Text se awaaz banana. STT ka ulta, aur mushkil: jo jaankari phenki thi wo wapas laani hai. | `speak.py: def speak` |
| 2 | 22.05 / 24 kHz | TTS ka sampling rate. Samajhne ke liye 16 kHz kaafi, achha sunne ke liye zyada chahiye. Lesson 10: `TTS_SR=24000` (100 mel bands). | `audio.py: TTS_SR` |
| 3 | Lamba window | 46 ms ka frame: frequency ki baareeki zyada (pitch ke harmonics dikhte hain). | `audio.py: N_FFT = 1024` |
| 4 | Frame rate | 86 mel frames per second. | `audio.py: HOP = 256` |
| 5 | Resampling | Sampling rate badalna (44.1 → 22.05 kHz). | `audio.py: def resample` |
| 6 | Sinc interpolation | Har sample pe ek sinc rakh ke unka jod: samples ke beech ka sahi curve. | `audio.py: torch.sinc` |
| 7 | Anti-aliasing filter | Rate ghatane se pehle oonchi frequencies kaatna. | `audio.py: cutoff` |
| 8 | Aliasing | Na kaato to oonchi frequency jhoothi neechi tone ban jaati hai (15 kHz → 7 kHz). | `audio.py: aliasing` |
| 9 | Windowed sinc | Asli sinc anant lamba hai; use window se kaat-te hain. | `audio.py: width` |
| 10 | WAV likhna | Wahi header jo STT ne padha tha, ab haath se likha. | `audio.py: def write_wav` |
| 11 | Silence trimming | Clip ke shuru / aakhir ki chuppi kaatna. | `audio.py: def trim_silence` |
| 12 | Magnitude spectrogram | TTS mein power nahi, magnitude ka log lete hain (vocoders ka riwaaz). | `audio.py: def log_mel` |
| 13 | Log floor | log(0) se bachne ke liye sabse chhoti value. | `audio.py: clamp(min=1e-5)` |
| 14 | Phase | Har frequency ki lehar apne chakkar mein kahan hai. Spectrogram mein nahi hota. | `audio.py: PHASE` |
| 15 | Phase reconstruction | Magnitude se phase ka andaza lagana: TTS ke aakhri kadam ki asli mushkil. | `audio.py: def griffin_lim` |
| 16 | Griffin-Lim | Random phase se shuru, baar-baar: waveform banao, dobara analyse karo, uska phase rakho. | `audio.py: again / again.abs()` |
| 17 | Iterations | Jitne zyada chakkar, utna self-consistent phase. | `audio.py: rounds` |
| 18 | Inverse STFT | Complex spectrogram se waveform: overlap-add. | `audio.py: torch.istft` |
| 19 | Mel inversion | 80 bands se 513 bins wapas: pseudo-inverse se sabse accha andaza. | `audio.py: torch.linalg.pinv` |
| 20 | Metallic artifact | Griffin-Lim ki pehchaan: galat phase ki jhanjhanahat. | `audio.py: metallic` |
| 21 | Harmonics | Pitch ke guna (150, 300, 450 Hz ...). Awaaz ki "banawat". | `audio.py: harmonics` |
| 22 | Pitch / F0 | Vocal cords kitni baar per second khulte hain. Sur. | theory |
| 23 | Formants | Munh ki banawat se bani oonchi-energy wali pattiyan. Swar inse pehchana jaata hai. | theory |
| 24 | Overlap-add | Frames ko window lagake jodna, taaki jod pe awaaz na toote. | theory |

## B. Text front-end (lesson 2: text.py)

| # | Term | Seedhi bhasha mein | Kahan |
|---|---|---|---|
| 25 | Text front-end | Text ko model ke layak banana: normalize, characters, bhasha. | `text.py: def normalize` |
| 26 | Punctuation as prosody | Comma = chhota rukna, full stop = lamba aur girta sur. Isliye TTS use rakhta hai. | `text.py: PUNCT` |
| 27 | Punctuation unification | । ; : — sab ko , ya . mein badalna. | `text.py: _UNIFY` |
| 28 | Character input | Model seedha akshar padhta hai, phonemes nahi. | `text.py: Characters, not phonemes` |
| 29 | G2P | Grapheme-to-phoneme: likhe ko bolne ki dhwaniyon mein badalna. Purane TTS ka pehla kadam. | `text.py: G2P` |
| 30 | Schwa deletion | Hindi mein chhupa "a" kai jagah nahi bola jaata: कमल = "kamal". | `text.py: schwa` |
| 31 | Language id | Wahi akshar alag bhasha mein alag bolte hain; model ko bhasha batani padti hai. | `text.py: LANGS` |
| 32 | Language guessing | Script aur kuch pehchaan-shabdon se bhasha ka andaza. | `text.py: def guess_lang` |
| 33 | Hinglish | Roman mein likhi Hindi, English shabdon ke saath. Apni alag id. | `text.py: _HINGLISH` |
| 34 | Hindi vs Marathi | Ek script, alag ucharan: marker shabdon se alag karte hain. | `text.py: _MARATHI` |
| 35 | Padding id | Batch bharne wala khaali symbol. | `text.py: PAD = 0` |
| 36 | Vocabulary | Akshar jo model padh sakta hai; an-dekhe chhod diye jaate hain. | `text.py: class Vocab` |
| 37 | Text normalization (numbers) | "2024" ko "do hazaar chaubees" banana. Hamare paas nahi hai. | `data.py: digits` |
| 38 | Homograph | Ek spelling, do ucharan ("read"). G2P ki purani mushkil. | theory |
| 39 | Phoneme set / IPA | Dhwaniyon ki maanak list. | theory |
| 40 | espeak-ng | Open-source G2P jo Piper aur Kokoro use karte hain. | theory |
| 41 | Code-switching in TTS | Ek vaakya mein do bhashayen: ek awaaz mein dono bolna. | theory |

## C. Data (lesson 3: data.py)

| # | Term | Seedhi bhasha mein | Kahan |
|---|---|---|---|
| 42 | TTS data ki zaroorat | Ek hi awaaz, saaf recording, ghanton tak. ASR data se ulta. | `data.py: ONE consistent voice` |
| 43 | Knowledge distillation | Bada model (teacher) bolta hai, chhota (student) uski nakal seekhta hai. | `data.py: Teacher (distillation)` |
| 44 | Teacher model | Supertonic 3 (99 M). | `data.py: TEACHER` |
| 45 | Synthetic data | Model ka banaya training data. Jitna chaho, bilkul saaf. | `data.py: def _speak` |
| 46 | Teacher ke denoising steps | Zyada steps = saaf target. Data ek hi baar banta hai, to yahan kanjoosi nahi. | `data.py: config.num_steps` |
| 47 | Student ki seema | Student teacher ke paas pahunch sakta hai, usse aage nahi. | `data.py: not beat it` |
| 48 | Derivative model | License ki bhasha: teacher ke output pe train hua model uska "derivative" hai. | `data.py: derivative` |
| 49 | Real recordings | Marathi ke liye asli insaani awaaz (OpenSLR 64). | `data.py: def marathi` |
| 50 | Loudness normalization | Har clip ko ek hi LOUDNESS (RMS -20 dBFS) pe laana, peak pe nahi: warna dheemi aur cheekhti clip ek jaisi zor se lagti hain. | `data.py: -20 dBFS` |
| 51 | PCM storage | Samples ko seedha 16-bit integers mein rakhna. | `data.py: .pcm` |
| 52 | Parallel workers | Paanch processes, har ek ke paas teacher ki apni copy. Ek worker mare to `ProcessPoolExecutor` zor se fail hota hai; `multiprocessing.Pool` hamesha intezaar karta reh jaata. | `data.py: ProcessPoolExecutor(args.workers)` |
| 53 | Held-out dev sentences | Har bhasha ke 40 vaakya training se bahar. | `data.py: DEV_PER_LANG` |
| 54 | Sentence selection | Na bahut chhote, na bahut lambe, bina digits ke. | `data.py: def _usable` |
| 55 | Text diversity | Kahaniyon ke aasaan shabd + khabron ke mushkil shabd aur naam. | `data.py: def texts_hi` |
| 56 | Single vs multi-speaker | Ek awaaz seekhna aasaan; kai ke liye speaker embedding chahiye. | theory |
| 57 | Speaker embedding | Awaaz ki pehchaan ka vector. | theory |
| 58 | Voice cloning / zero-shot TTS | Kuch second ke sample se nayi awaaz. | theory |

## D. Alignment aur duration (lesson 4: align.py)

| # | Term | Seedhi bhasha mein | Kahan |
|---|---|---|---|
| 59 | Duration | Ek akshar kitne mel frames tak bola gaya. | `align.py: durations` |
| 60 | TTS ka alignment problem | 52 akshar se 430 frames: kaunsa akshar kitna lamba? Data mein nahi likha. | `align.py: Durations are not in the data` |
| 61 | Monotonic alignment | Akshar kram mein, koi chhoota nahi, har frame ek hi akshar ka. | `align.py: MONOTONIC` |
| 62 | MAS | Monotonic Alignment Search: sabse accha monotonic raasta, dynamic programming se. | `align.py: def mas` |
| 63 | Score table | Har (akshar, frame) jodi ke liye: kitna milta hai. | `model.py: score = -0.5` |
| 64 | Stay / step | Raasta ya usi akshar pe rukta hai ya agle pe jaata hai. CTC jaisa "skip" nahi. | `align.py: step = np.concatenate` |
| 65 | DP table | best[l, t]: frame t tak, akshar l pe khatm hone wala sabse accha raasta. | `align.py: best[:, :, t]` |
| 66 | Backtracking | Ant se peeche chal ke raasta padhna. | `align.py: came_by_step` |
| 67 | Durations from a path | Raaste ne har akshar ko kitne frames diye, wahi uski duration. | `align.py: += 1` |
| 68 | Chicken-and-egg | Acche guess ko accha alignment chahiye aur ulta; phir bhi dono saath sudharte hain. | `align.py: circular` |
| 69 | External aligner | Alag tool (Montreal Forced Aligner) se durations. FastSpeech 2 ka tareeka. | theory |
| 70 | Attention alignment | Tacotron: attention khud seekhta hai kahan dekhna hai; galti pe shabd chhoot-te / dohrate hain. | `model.py: Tacotron` |
| 71 | Forward-sum loss | Alignment ko CTC jaise loss se seekhna (FastPitch ka naya aligner). | theory |

## E. Acoustic model (lesson 5: model.py)

| # | Term | Seedhi bhasha mein | Kahan |
|---|---|---|---|
| 72 | Acoustic model | Text se mel spectrogram banane wala network. | `model.py: class Acoustic` |
| 73 | Two-stage TTS | Pehle text → mel, phir mel → waveform. | `speak.py: acoustic model -> mel -> vocoder` |
| 74 | Non-autoregressive TTS | Saare frames ek saath, pichhle frame pe nirbhar nahi. | `model.py: NON-AUTOREGRESSIVE` |
| 75 | FastSpeech family | Duration predictor + length regulator wala design. | `model.py: FastSpeech` |
| 76 | Text encoder | Har akshar ko uske padosiyon ki jaankari ke saath vector banana. | `model.py: def encode` |
| 77 | Language embedding | Bhasha ka vector, har akshar mein joda jaata hai. | `model.py: self.langs` |
| 78 | Duration predictor | Chhota network: har akshar ke liye log(frames). | `model.py: class DurationPredictor` |
| 79 | Log-duration | Duration ka log predict karna: 2 ko 4 kehna utna hi bura jitna 20 ko 40. | `model.py: .log()` |
| 80 | Length regulator | Har akshar ka vector utni baar dohrana jitni uski duration. Text-time se audio-time. | `model.py: def length_regulate` |
| 81 | Upsampling by repetition | Wahi kaam, PyTorch ki ek line mein. | `model.py: torch.repeat_interleave` |
| 82 | Mel decoder | Dohraye gaye vectors ko chikna spectrogram banana. | `model.py: def decode` |
| 83 | Mel head | Aakhri linear layer: d se 80 bands. | `model.py: self.to_mel` |
| 84 | Prior (mu) | Encoder ka har akshar ke liye ek mel frame ka mota andaza. MAS isi pe chalta hai. | `model.py: self.prior` |
| 85 | Prior loss | Andaze ko asli mel ke kareeb laana, alignment ke saath. | `model.py: prior_loss` |
| 86 | Mel loss (L1) | Bane aur asli spectrogram ka fark. | `model.py: mel_loss` |
| 87 | Duration loss | Predictor ko MAS ki durations sikhana. | `model.py: dur_loss` |
| 88 | Stop-gradient | Duration loss encoder ko na mode, sirf predictor ko sikhaye. | `model.py: h.detach()` |
| 89 | Multi-task loss | Teen losses ka jod ek saath kam karna. | `model.py: mel_loss + prior_loss + dur_loss` |
| 90 | Speed control | Durations ko ek number se bhaag do: tez ya dheema, sur wahi. | `model.py: speed` |
| 91 | Inference path | Training mein MAS ki durations, inference mein predictor ki. | `model.py: def synthesize` |
| 92 | Global mel normalization | Poore mel ka ek mean aur ek std: bands ka aapsi fark bana rahe. | `model.py: self.register_buffer("mean"` |
| 93 | Over-smoothing | L1 loss "ausat" spectrogram ko inaam deta hai, jo dhundhla hota hai. | `train.py: blurred` |
| 94 | Length ratio | Dev pe predicted kul lambai / asli lambai. 1.00 = bolne ki raftaar sahi. | `train.py: length` |
| 95 | Listening sample | Har epoch ke baad ek vaakya Griffin-Lim se bana ke sunna. | `train.py: sample_` |
| 96 | Pitch predictor | FastPitch / FastSpeech 2: har akshar ka sur bhi predict karna. Hamare paas nahi. | theory |
| 97 | Energy predictor | Har frame ki loudness predict karna. | theory |
| 98 | Variance adaptor | Duration + pitch + energy predictors ka samooh. | theory |
| 99 | One-to-many problem | Ek vaakya kai tarah se bola ja sakta hai; model ko ek chunna hai. | theory |
| 100 | Flow matching | Matcha-TTS / Supertonic: shor se mel tak seedha raasta seekhna. Over-smoothing ka ilaaj. | theory |
| 101 | Diffusion TTS | Shor ko kai kadmon mein saaf karke spectrogram banana. | theory |
| 102 | VITS | Acoustic model aur vocoder ek saath (end-to-end). Piper yahi hai. | theory |
| 103 | Codec language model | Audio ko tokens bana ke LLM jaise predict karna (VALL-E, Orpheus). | theory |

## F. Vocoder (lesson 7: vocoder.py)

| # | Term | Seedhi bhasha mein | Kahan |
|---|---|---|---|
| 104 | Vocoder | Mel spectrogram se waveform banane wala model. | `vocoder.py: class Vocoder` |
| 105 | Neural vocoder | Phase ko seekh ke banana, Griffin-Lim ki tarah andaze se nahi. | `vocoder.py: LEARNS` |
| 106 | HiFi-GAN | Mel ko 256 guna upsample karke waveform. Standard, par bhaari. | `vocoder.py: HiFi-GAN` |
| 107 | Vocos / iSTFT vocoder | Frame rate pe hi raho, magnitude + phase predict karo, iSTFT baaki kare. | `vocoder.py: Vocos` |
| 108 | ConvNeXt block | Depthwise conv + chhota MLP + residual. Attention nahi. | `vocoder.py: class ConvNeXtBlock` |
| 109 | Layer scale | Har block shuru mein bahut chhota sudhaar jodta hai. | `vocoder.py: self.scale` |
| 110 | Magnitude + phase head | Har frame ke liye 513 log-magnitudes aur 513 phases. | `vocoder.py: N_FFT + 2` |
| 111 | Phase wrapping | Phase ek kon hai: 0 aur 2π ek hi. cos / sin se ye apne aap sambhalta hai. | `vocoder.py: torch.complex` |
| 112 | Magnitude clamp | Shuru ki ek paagal prediction iSTFT ko uda na de. | `vocoder.py: clamp(max=100.0)` |
| 113 | Spectral loss | Waveform nahi, spectrograms ki tulna. Alag phase wali do awaazein ek jaisi sun sakti hain. | `vocoder.py: def spectral_loss` |
| 114 | Multi-resolution STFT loss | Teen window sizes: chhoti timing dekhti hai, lambi pitch. | `vocoder.py: (512, 1024, 2048)` |
| 115 | Spectral convergence | Zor wale hisson ka relative fark. | `vocoder.py: convergence` |
| 116 | Log-magnitude loss | Dheeme hisson ko bhi barabar wazan. | `vocoder.py: log_magnitude` |
| 117 | Random crops | Vocoder ko poore vaakya nahi chahiye; chhote random tukde kaafi. | `vocoder.py: def batch_from` |
| 118 | GAN | Do networks: ek banata hai, doosra asli-nakli pehchanta hai. Aakhri "buzz" isi se jaati hai. | `vocoder.py: GAN` |
| 119 | Discriminator (MPD / MRD) | GAN ka jaanchne wala: alag periods / resolutions pe dekhta hai. | `vocoder.py: class Discriminators` |
| 120 | Feature matching loss | Generator ko discriminator ki beech ki layers ke features milane ko kehna. | `vocoder.py: def fooling_loss` |
| 121 | Mismatch: GT vs predicted mel | Vocoder asli mel pe train hota hai, par use acoustic model ka mel milta hai. | theory |

## G. Inference aur evaluation (lesson 8: speak.py)

| # | Term | Seedhi bhasha mein | Kahan |
|---|---|---|---|
| 122 | Real-time factor | Compute ke second / audio ke second. TTS ke liye 0.1 se kam aam hai. | `speak.py: real-time factor` |
| 123 | Vocoder ablation | Wahi mel, Griffin-Lim banam neural vocoder: fark sunna. | `speak.py: --griffin-lim` |
| 124 | MOS | Mean opinion score: log 1-5 mein rate karte hain. TTS ka asli paimana. | theory |
| 125 | Intelligibility (ASR round-trip) | Banayi awaaz ko ASR se likhwa ke CER dekhna: shabd samajh aa rahe hain? | theory |
| 126 | UTMOS | Model jo MOS ka andaza lagata hai, insaanon ke bina. | theory |
| 127 | Streaming TTS | Vaakya-dar-vaakya (ya chunk) audio dena, poore jawab ka intezaar nahi. | theory |
| 128 | Time to first audio | Text milne se pehla sample bajne tak. Voice agent ke liye sabse zaroori. | theory |

## H. Expressive awaaz (lesson 10: model.py, vocoder.py, data.py)

| # | Term | Seedhi bhasha mein | Kahan |
|---|---|---|---|
| 129 | Over-smoothing | L1/L2 loss saare sambhav tareekon ka AVERAGE sikhata hai: dhundhla mel, flat sur = robotic. Humne naapa: 17% kam frame-to-frame badlaav. | `model.py: Lesson 10` |
| 130 | Flow matching | Noise se speech tak seedhi line pe chalna sikhana: har point pe "kis taraf jaana hai" (velocity). | `model.py: def velocity` |
| 131 | OT-CFM | Optimal-transport conditional flow matching: x_t = (1-(1-s)t)x0 + t·x1, target u = x1-(1-s)x0. Matcha-TTS ka loss. | `model.py: u = target - (1 - self.cfg.sigma_min) * x0` |
| 132 | sigma_min | t=1 pe data ke aas-paas chhoda gaya bahut chhota phailaav (1e-4). | `model.py: sigma_min` |
| 133 | Time embedding | t (0 se 1) ko sinusoids + MLP se vector banana, taaki decoder jaane wo raaste mein kahan hai. | `model.py: class TimeEmbedding` |
| 134 | Euler steps (NFE) | Noise se mel tak kitne kadam. Zyada = behtar par dheema. Humare CPU pe 4 steps: RTF 0.17. | `model.py: x = x + v / steps` |
| 135 | Temperature (sampling) | Shuruaati noise ka size. Kam = shaant aur surakshit, zyada = zinda par galti ka khatra. Matcha: 0.667. | `model.py: temperature` |
| 136 | Speaker embedding | Har awaaz ka ek vector; model ko batata hai kaun bol raha hai. | `model.py: self.speakers` |
| 137 | Style / emotion embedding | Khushi, gussa, dukh... har style ka vector. Encoder (durations) aur decoder (sur) dono ko milta hai. | `model.py: self.styles` |
| 138 | Classifier-free guidance | Training mein 20% baar style chhupao; bolte waqt v = v_none + w(v_style - v_none). w > 1 = emotion zyada. | `model.py: guidance` |
| 139 | Hinge loss (GAN) | Critic ko asli pe +1 se upar, nakli pe -1 se neeche score dena sikhana. Vocos yahi use karta hai. | `vocoder.py: def critic_loss` |
| 140 | GAN fine-tuning | Pehle se train vocoder ko critic ke saamne aage train karna; zero se GAN asthir hota hai. | `vocoder.py: --gan` |
| 141 | Expressive dataset | Ek hi insaan alag emotions aur styles mein padhe. Rasa: Hindi aur Marathi, 2 awaazein, 6 emotions + baat-cheet, news, kahani. | `data.py: def rasa` |
| 142 | Gated dataset | HF pe terms accept kiye bina download nahi hota (403). | `data.py: Gated` |
| 143 | Distillation ki seema | Teacher (Supertonic) se seekha model teacher jaisa hi expressive ho sakta hai, zyada nahi. Isliye asli insaan ka data. | `data.py: Real people` |
