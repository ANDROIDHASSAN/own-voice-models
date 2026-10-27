# STT glossary: beginner se advanced

Har term ke saamne wo jagah likhi hai jahan tune use khud code mein banaya hai (file ka naam, phir wo naam jo us file mein search karna hai).
File kholo, naam search karo, wahi line hai. `check.py` fail ho jaata hai agar koi pointer galat ho.

"Kahan" mein `theory` ka matlab: ye term is project ke code mein nahi hai, par interview mein poocha jaata hai.
Jo terms LLM wale module mein aa chuke hain (attention, residual, AdamW...) wo yahan dobara nahi samjhaye, sirf wahi jo speech mein naya hai ya alag tarah se lagta hai.

## A. Awaaz ki neenv (lesson 1: audio.py)

| # | Term | Seedhi bhasha mein | Kahan |
|---|---|---|---|
| 1 | Waveform | Hawa ke dabaav ka time ke saath record. Numbers ki ek lambi list, -1 se 1 ke beech. | `audio.py: def read_wav` |
| 2 | Sample | Us list ka ek number: ek pal ka dabaav. | `audio.py: np.frombuffer` |
| 3 | Sampling rate | Ek second mein kitne samples. Hamara 16,000 (16 kHz). | `audio.py: SR = 16000` |
| 4 | Nyquist limit | Sampling rate ka aadha hi sabse oonchi frequency hai jo pakdi ja sakti hai. 16 kHz pe 8 kHz tak. | `audio.py: Nyquist` |
| 5 | WAV / RIFF | Audio file ka sabse seedha format: chhota header, phir samples. | `audio.py: b"RIFF"` |
| 6 | Chunk (WAV) | WAV file ke tukde: naam, size, bytes. `fmt ` mein format, `data` mein samples. | `audio.py: name == b"fmt "` |
| 7 | PCM 16-bit | Har sample ek 16-bit integer (-32768 se 32767). 32768 se bhaag do to -1..1. | `audio.py: "<i2"` |
| 8 | Float audio | Har sample seedha 32-bit float. FLEURS isi mein hai. | `audio.py: "<f4"` |
| 9 | Mono / stereo | Ek channel ya do. Speech model ko ek chahiye, to channels ka average. | `audio.py: .mean(axis=1)` |
| 10 | Frequency | Ek second mein kitni baar dabaav upar-neeche hota hai (Hz). Patli awaaz = zyada. | `audio.py: peak_hz` |
| 11 | Frame / window | Waveform ka chhota tukda (25 ms) jisme awaaz "sthir" maani jaati hai. | `audio.py: N_FFT = 400` |
| 12 | Hop length | Agla frame kitne samples baad shuru. 160 = 10 ms, yaani 100 frames/second. | `audio.py: HOP = 160` |
| 13 | Overlap | Frames ek doosre pe chadhe hote hain (25 ms lamba, 10 ms pe naya). Kuch chhoot-ta nahi. | `audio.py: wave.unfold` |
| 14 | Fourier transform | Signal ko uski frequencies mein todna: "ismein kaunsi frequency kitni hai". | `audio.py: torch.fft.rfft` |
| 15 | FFT | Fourier transform nikalne ka tez algorithm. | `audio.py: torch.fft.rfft` |
| 16 | STFT | Short-time Fourier transform: har chhote frame ka alag FFT. Time aur frequency dono milte hain. | `audio.py: def power_spectrogram` |
| 17 | FFT bin | FFT ka ek output: ek frequency ki patti. 400 samples pe bins 40 Hz door hain. | `audio.py: SR / N_FFT` |
| 18 | Hann window | Frame ke kinaaron ko dheere se zero karna, taaki kata hua kinara jhoothi frequency na banaye. | `audio.py: torch.hann_window` |
| 19 | Spectral leakage | Wo jhoothi frequencies jo frame ko seedha kaatne se aati hain. | `audio.py: spectral leakage` |
| 20 | Magnitude / phase | FFT ka har number complex hai: kitna zor (magnitude) aur lehar kahan se shuru (phase). | `audio.py: spectrum.abs()` |
| 21 | Power spectrogram | Magnitude ka square, har frame aur har bin ke liye. Phase phenk diya. | `audio.py: spectrum.abs().pow(2)` |
| 22 | Mel scale | Frequency ka wo scale jo kaan jaisa hai: neeche baareek, upar mota. | `audio.py: def hz_to_mel` |
| 23 | Mel filterbank | 80 triangles jo 201 bins ko 80 bands mein samet dete hain. | `audio.py: def mel_filterbank` |
| 24 | Mel bands | Filterbank ke baad ke 80 numbers. | `audio.py: N_MELS = 80` |
| 25 | Log compression | Energy ka log lena. Kaan anupaat sunta hai, antar nahi. | `audio.py: torch.log(mel.clamp` |
| 26 | Log-mel spectrogram | Upar ka sab milake: (frames, 80) ki tasveer. Model ka input. | `audio.py: def log_mel` |
| 27 | Reflect padding | Shuru aur aakhir mein signal ko aaine ki tarah badhana, taaki frame t theek t pe centred ho. | `audio.py: mode="reflect"` |
| 28 | Chirp | Aisi tone jiski frequency badhti jaati hai. Spectrogram check karne ka tareeka. | `audio.py: sweep = torch.sin` |
| 29 | MFCC | Log-mel pe ek aur transform (DCT). Purane HMM systems ka feature; neural models log-mel hi lete hain. | theory |
| 30 | Pre-emphasis | Oonchi frequencies ko thoda badhana. Purana trick, ab zaroori nahi. | theory |
| 31 | Resampling | Sampling rate badalna (jaise 48 kHz se 16 kHz). Pehle low-pass filter, warna aliasing. | theory |
| 32 | Aliasing | Nyquist se oonchi frequency ka galat, neechi frequency ban ke dikhna. | theory |

## B. Text aur scoring (lesson 2: text.py)

| # | Term | Seedhi bhasha mein | Kahan |
|---|---|---|---|
| 33 | Transcript | Clip mein jo bola gaya uska likha hua roop. Training ka label. | `data.py: "text": text` |
| 34 | Text normalization | Transcript se wo sab hatana jo suna nahi ja sakta: punctuation, capital letters. | `text.py: def normalize` |
| 35 | Unicode NFC | Ek hi dikhne wale akshar ko ek hi tarah store karna. | `text.py: unicodedata.normalize("NFC"` |
| 36 | Code point | Unicode ka ek number. "कि" do code points hai: क aur ि. | `text.py: code points of` |
| 37 | Matra | Devanagari mein swar ka chinh (ा ि ी ...). Alag code point hai. | `text.py: unicodedata.category` |
| 38 | Character vocabulary | Model jo symbols likh sakta hai unki list. Hamari lagbhag 110. | `text.py: class Vocab` |
| 39 | Grapheme | Likhne ki ikai (akshar). Hum graphemes predict karte hain, phonemes nahi. | `text.py: def encode` |
| 40 | Phoneme | Bolne ki ikai (dhwani). "phone" mein 5 graphemes, 3 phonemes. | theory |
| 41 | Blank symbol | CTC ka extra symbol: "is frame mein kuch naya nahi". Id 0. | `text.py: BLANK = 0` |
| 42 | Rare character cutoff | Jo character 5 se kam baar aaya use vocabulary se hata dena: seekha nahi ja sakta. | `text.py: min_count` |
| 43 | Edit distance | Ek text ko doosre mein badalne ke liye kam se kam kitne insert / delete / substitute. | `text.py: def edit_distance` |
| 44 | WER | Word error rate: shabdon pe edit distance / reference ke shabd. | `text.py: def error_rate` |
| 45 | CER | Character error rate: wahi, characters pe. Lambe shabdon wali bhashaon ke liye zyada nyaay-sangat. | `text.py: unit: str = "word"` |
| 46 | Substitution / deletion / insertion | WER ki teen galtiyan: galat shabd, chhoota shabd, faltu shabd. | `text.py: diagonal + (r != h)` |
| 47 | WER > 100% | Ho sakta hai: model itne faltu shabd jode ki galtiyan reference se zyada ho jaayen. | `text.py: It can exceed 1.0` |
| 48 | Subword (BPE) for ASR | Bade models characters ki jagah subwords likhte hain. Kam data pe characters behtar. | theory |

## C. Data (lesson 3: data.py)

| # | Term | Seedhi bhasha mein | Kahan |
|---|---|---|---|
| 49 | Supervised learning | Har input ke saath insaan ka diya sahi jawab. Speech ke liye: audio + transcript. | `data.py: SUPERVISED` |
| 50 | Corpus | Data ka sangrah. Hamara FLEURS. | `data.py: google/fleurs` |
| 51 | Read speech | Likha hua padhke bola gaya. Saaf hota hai; asli baat-cheet (spontaneous) mushkil. | `data.py: reading Wikipedia sentences` |
| 52 | Train / dev / test split | Seekhne ke liye / training ke dauraan dekhne ke liye / aakhir mein ek baar naapne ke liye. | `data.py: ("dev", "test", "train")` |
| 53 | Sentence-disjoint split | Test ka koi vaakya train mein nahi. Warna model ratta maar ke pass ho jaata. | `data.py: "sentence": row[0]` |
| 54 | Feature extraction | Audio se log-mel banana. Har epoch same hai, to ek baar karke disk pe rakh do. | `data.py: def build_split` |
| 55 | Streaming a tar | Archive ko kholke ek-ek file padhna, disk pe failaye bina. | `data.py: tarfile.open` |
| 56 | float16 storage | Features ko aadhe size mein rakhna. Model ko fark nahi padta. | `data.py: np.float16` |
| 57 | Memmap | Badi file ko RAM mein laaye bina array ki tarah padhna. | `data.py: np.memmap` |
| 58 | Index | Har clip ke liye: frames kahan se shuru, kitne, text, bhasha. | `data.py: def open_split` |
| 59 | Max duration filter | Bahut lambe training clips hatana: ek lamba clip poore batch ki memory tay karta hai. | `data.py: MAX_TRAIN_SEC` |
| 60 | Feature mean / std | Har mel band ka ausat aur failaav, sirf train pe naapa. | `data.py: mean = total / len(mel)` |
| 61 | Data leakage | Test ki jaankari ka training mein ghus jaana (jaise stats test pe naapna). Score jhootha ho jaata hai. | `data.py: leak information` |
| 62 | Hours of speech | Speech data ghanton mein naapa jaata hai. Hamara ~27 h; Whisper ka 680,000 h. | `data.py: seconds / 3600` |
| 63 | Pseudo-labeling | Bade model se bina-label audio ka transcript banwa ke us pe chhota model train karna. | theory |
| 64 | Speed perturbation | Audio ko 0.9x / 1.1x speed pe chalake data badhana. | theory |

## D. CTC (lesson 4: ctc.py)

| # | Term | Seedhi bhasha mein | Kahan |
|---|---|---|---|
| 65 | Alignment problem | Kaunsa frame kis akshar ka hai, ye kisi ne nahi bataya. | `ctc.py: Which frame belongs to which character` |
| 66 | CTC | Connectionist Temporal Classification: bina alignment ke audio-text pe train karne ka tareeka. | `ctc.py: def ctc_loss` |
| 67 | Path | Har frame ke liye ek symbol: model ka frame-dar-frame output. | `ctc.py: frame-by-frame output` |
| 68 | Collapse rules | Pehle repeats milao, phir blanks hatao. Path se text. | `ctc.py: torch.unique_consecutive` |
| 69 | Blank ka doosra kaam | Asli double letters ko alag rakhna: "l-l" → "ll", "ll" → "l". | `ctc.py: can_skip` |
| 70 | Extended label sequence | Target ke aage, beech aur peeche blank: "hi" → -h-i-. 2L+1 states. | `ctc.py: def _extend` |
| 71 | Marginalization | Sahi text dene wale SAARE paths ki probability jodna. | `ctc.py: torch.logsumexp` |
| 72 | Forward algorithm | Wo jod dynamic programming se, T × (2L+1) steps mein. | `ctc.py: alpha` |
| 73 | Alpha | Frame t tak, state s mein pahunchne wale saare adhoore paths ki log-probability. | `ctc.py: stay = alpha` |
| 74 | Stay / step / skip | State mein aane ke teen raaste: wahi rukna, ek aage, ya blank chhod ke do aage. | `ctc.py: step = torch.cat` |
| 75 | Emission | Kisi frame pe kisi symbol ki log-probability. | `ctc.py: emit = log_probs.gather` |
| 76 | Log space | Probabilities ki jagah unke log. Chhoti probabilities ka guna zero ban jaata hai (underflow). | `ctc.py: def _log_add` |
| 77 | log(0) ka jugaad | Asli -inf se gradient NaN ho jaata hai, isliye bahut bada negative number. | `ctc.py: NEG = -1e30` |
| 78 | Input length masking | Batch mein chhota clip apne asli frames ke baad update hona band kar deta hai. | `ctc.py: in_lens` |
| 79 | Conditional independence | CTC ki maanyata: har frame ka output baaki outputs se azaad hai (audio diya ho to). | `ctc.py: INDEPENDENTLY` |
| 80 | Greedy / best-path decoding | Har frame ka sabse sambhav symbol, phir collapse. | `ctc.py: def greedy` |
| 81 | Prefix beam search | Sabse sambhav TEXTS rakhna, paths nahi. Ek text ke kai paths ki probability judti hai. | `ctc.py: def beam_search` |
| 82 | Beam width | Har frame ke baad kitne candidate texts zinda rakhne hain. | `ctc.py: beam: int = 8` |
| 83 | p_blank / p_non_blank | Har prefix ke do numbers: blank pe khatm hone wale paths, aur akshar pe khatm hone wale. | `ctc.py: p_b, p_nb` |
| 84 | Pruning | Bahut kam probability wale symbols ko dekhna hi nahi. | `ctc.py: frame.topk` |
| 85 | Forced alignment | Diye hue text ke liye sabse sambhav path: kaunsa akshar kab bola gaya. | `ctc.py: def align` |
| 86 | Viterbi | Forward algorithm mein "sum" ki jagah "max", saath mein peeche jaane ka record. | `ctc.py: moves[s] = max` |
| 87 | Spiky output | Trained CTC zyada-tar blank bolta hai aur har akshar ek-do frame mein. | `xray.py: spiky` |
| 88 | LM fusion | Beam search mein language model ka score jodna, taaki spelling aur grammar sudhre. | theory |
| 89 | RNN-T / transducer | CTC ka bada bhai: output pichhle outputs pe bhi nirbhar. Streaming ASR ka standard. | theory |
| 90 | Attention encoder-decoder | Whisper ka tareeka: decoder ek-ek token likhta hai, encoder pe attention karke. | theory |

## E. Model (lesson 5: model.py)

| # | Term | Seedhi bhasha mein | Kahan |
|---|---|---|---|
| 91 | Acoustic model / encoder | Spectrogram ko har frame ke liye character probabilities mein badalne wala network. | `model.py: class Encoder` |
| 92 | Encoder-only | Koi decoder nahi, koi loop nahi: ek pass mein saare frames. | `model.py: ENCODER-ONLY` |
| 93 | Feature normalization | Har mel band ko mean 0, std 1 pe laana, model ke andar. | `model.py: (mel - self.mean) / self.std` |
| 94 | Buffer in checkpoint | Mean / std weights ke saath save hote hain, to model hamesha waise hi normalize karega jaise train hua. | `model.py: self.register_buffer("mean"` |
| 95 | Subsampling | 100 frames/s ko 25 pe laana. Attention 16 guna sasta. | `model.py: class Subsample` |
| 96 | Strided convolution | Aisa convolution jo har doosri position chhod deta hai: output aadha. | `model.py: stride=2` |
| 97 | Output length formula | Convolution ke baad kitne frames bachenge: (n - 1) // 2 + 1. CTC ko ye sahi chahiye. | `model.py: def out_len` |
| 98 | Frame rate | Model ke output ki raftaar: 25 frames/s, ek frame = 40 ms. | `xray.py: FRAME_SEC = 0.04` |
| 99 | Padding | Batch ke chhote clips ko lambe ke barabar karne ke liye joda gaya khaali hissa. | `train.py: pad_sequence` |
| 100 | Padding mask | Bool array: kaunsa frame asli hai, kaunsa padding. | `model.py: def frame_mask` |
| 101 | Attention padding mask | Koi frame padding pe attention na de. | `model.py: ~real[:, None, None, :]` |
| 102 | Padding invariance | Clip akela ho ya batch mein, output same. Na ho to batch size badalte hi accuracy badlegi. | `model.py: SAME` |
| 103 | Bidirectional attention | Har frame poora clip dekh sakta hai, aage ka bhi. LLM mein ye mana tha. | `model.py: class SelfAttention` |
| 104 | Conformer | Transformer + convolution ka block. Speech ka standard encoder (2020). | `model.py: class Block` |
| 105 | Squeezeformer layout | Conformer ka saral roop: attention → conv → feed-forward, seedhe pre-norm residuals. | `model.py: x = x + self.conv` |
| 106 | Convolution module | Aas-paas ke 15 frames ko milana: local pattern pakadta hai. | `model.py: class ConvModule` |
| 107 | Depthwise convolution | Har channel ka apna alag chhota filter. 15 × d weights, 15 × d × d nahi. | `model.py: groups=cfg.d_model` |
| 108 | Pointwise layer | Har frame pe akele lagne wali linear layer: channels ko milati hai. | `model.py: self.expand` |
| 109 | GLU gate | Aadhe channels sigmoid se guzar ke baaki aadhon ko rokte / jaane dete hain. | `model.py: torch.sigmoid(gate)` |
| 110 | Kernel size | Convolution ek baar mein kitne frames dekhta hai. 15 × 40 ms = 0.6 s. | `model.py: conv_kernel: int = 15` |
| 111 | Receptive field | Ek output kitne input ko dekh paata hai. Conv se local, attention se poora clip. | `model.py: global context` |
| 112 | RoPE on frames | LLM wala hi rotary position, yahan tokens ki jagah frames pe. | `model.py: def apply_rope` |
| 113 | SpecAugment | Training mein spectrogram ki random pattiyan chhupa dena. Speech ka sabse zaroori augmentation. | `model.py: def spec_augment` |
| 114 | Frequency masking | Kuch mel bands ko poore clip mein zero karna. | `model.py: freq_masks` |
| 115 | Time masking | Kuch pal ke saare bands zero karna. | `model.py: time_masks` |
| 116 | CTC head | Aakhri linear layer: d numbers se vocabulary ke scores. | `model.py: self.head = nn.Linear` |
| 117 | Log-softmax | Scores ko log-probabilities mein. CTC ko yahi chahiye. | `model.py: F.log_softmax` |
| 118 | Safe checkpoint loading | Sirf tensors load karna, pickle ka code nahi chalana. | `model.py: weights_only=True` |
| 119 | Streaming encoder | Aisa encoder jo sirf peeche (aur thoda aage) dekhta hai, taaki bolte-bolte likh sake. Hamara nahi hai. | theory |
| 120 | Lookahead | Streaming model kitna aage ka audio dekhta hai. Moonshine v2 mein 80 ms. | theory |
| 121 | Self-supervised pretraining | wav2vec 2.0 / HuBERT: bina transcript ke audio se seekhna, phir thode labels pe fine-tune. | theory |
| 122 | Vector quantization | Continuous vector ko codebook ke sabse kareebi entry se badalna. Codecs aur wav2vec mein. | theory |

## F. Training (lesson 6: train.py)

| # | Term | Seedhi bhasha mein | Kahan |
|---|---|---|---|
| 123 | Overfit sanity test | Pehle 16 clips ratwa ke dekho. Na ratein to code mein bug hai, data mein nahi. | `train.py: --overfit` |
| 124 | Length bucketing | Ek jaisi lambai ke clips ek batch mein, taaki padding kam ho. | `train.py: def make_batches` |
| 125 | Dynamic batch size | Batch clips ki ginti se nahi, kul frames se tay: chhote clips ke bade batch. | `train.py: max_frames` |
| 126 | Collate | Alag lambai ke clips ko ek padded tensor mein jodna, saath mein asli lambaiyan. | `train.py: def collate` |
| 127 | CTC loss per character | Har clip ka loss uske characters se bhaag, phir ausat. Lambe clips haavi nahi hote. | `train.py: F.ctc_loss` |
| 128 | Infeasible target | Jab text mein output frames se zyada characters hon: koi valid path nahi, loss infinity. | `train.py: zero_infinity=True` |
| 129 | Blank plateau | Shuru mein model "hamesha blank" seekh leta hai aur loss atak jaata hai, phir achanak likhne lagta hai. | `train.py: PLATEAU` |
| 130 | Dev CER se checkpoint chunna | Jo checkpoint an-dekhe clips pe sabse accha, wahi rakho. | `train.py: if cer < best` |
| 131 | Macro average | Har bhasha ka CER barabar wazan se jodna, taaki zyada data wali bhasha haavi na ho. | `train.py: each counts equally` |
| 132 | Resume | Ruki hui training wahin se chalana. Optimizer ki state bhi chahiye. | `train.py: opt.load_state_dict` |
| 133 | Shuffling batches | Har epoch batches ka order badalna. | `train.py: random.shuffle(batches)` |
| 134 | Augmentation sirf train mein | SpecAugment training mein on, evaluation mein off. | `train.py: augment=not args.overfit` |
| 135 | Epoch | Poore training data ka ek chakkar. | `train.py: for epoch in range` |

## G. Inference, evaluation, size (lessons 7-10)

| # | Term | Seedhi bhasha mein | Kahan |
|---|---|---|---|
| 136 | Non-autoregressive inference | Poora clip ek pass mein. LLM jaisa token-dar-token loop nahi. | `transcribe.py: def transcribe` |
| 137 | Real-time factor (RTF) | Compute ke second / audio ke second. 1 se kam = bolne se tez. | `transcribe.py: real-time factor` |
| 138 | Microphone capture | Mic se seedha 16 kHz mono float samples lena. | `transcribe.py: sounddevice.rec` |
| 139 | Test set discipline | Test split ko training ke dauraan kabhi nahi dekhna. | `evaluate.py: open_split("test")` |
| 140 | Baseline | Wo jaana-maana model jiske saamne apna number rakhte hain. | `evaluate.py: from_omnilingual_asr_ctc` |
| 141 | Fair comparison | Same clips, same normalization, same metric, same threads. | `evaluate.py: torch.set_num_threads(2)` |
| 142 | Post-training quantization | Train hue weights ko kam bits mein rakhna aur nuksaan naapna. | `shrink.py: def quant_roundtrip` |
| 143 | Per-channel scale | Har output filter ka apna scale. Conv aur linear dono pe. | `shrink.py: per_channel` |
| 144 | Confidence | Model apne chune hue symbol ko kitni probability deta hai. | `xray.py: confidence` |
| 145 | Word timestamps | Har shabd kab se kab tak bola gaya, forced alignment se. | `xray.py: align(log_probs` |
| 146 | Endpointing | Ye tay karna ki bolne wala ruk gaya. Orchestrator ka kaam (module 4). | theory |
| 147 | VAD | Voice activity detection: is pal koi bol raha hai ya nahi. | theory |
| 148 | Streaming vs batch ASR | Bolte-bolte likhna, ya poora turn sunke. Hamara batch hai. | theory |
| 149 | Hallucination (ASR) | Decoder wale models chup audio pe bhi vaakya likh dete hain. CTC mein ye bahut kam. | theory |
| 150 | Diarization | "Kisne kab bola". ASR se alag kaam. | theory |
| 151 | Code-switching | Ek hi vaakya mein do bhashayen (Hinglish). ASR ke liye mushkil: do scripts, ek awaaz. | theory |
| 152 | Knowledge distillation | Bade model (teacher) ke outputs se chhote model (student) ko sikhana. | theory |
