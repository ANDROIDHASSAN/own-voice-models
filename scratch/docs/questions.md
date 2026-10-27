# Interview sawaal: beginner se advanced

Har sawaal ke do jawab. `S:` simple, apni bhasha mein, taaki samajh aaye. `B:` wo jo interview mein bolna hai.
`docs/build.py` isi file se guide ka chapter 9 banata hai.

## LLM

### Beginner
Q: LLM kya hai?
S: Ek machine jo "agla shabd batao" khel bahut accha khelti hai.
B: A neural network trained to predict the next token. Text generation is predicting, sampling and repeating.

Q: Token kya hota hai?
S: Text ka tukda. Aam shabd poora, mushkil shabd 2-3 tukdon mein.
B: A subword unit. It keeps the vocabulary small while still covering any word. I built a BPE tokenizer for mine.

Q: Model seekhta kaise hai?
S: Jawab deta hai, galti naapta hai, har knob ko thoda sahi taraf ghumata hai. Baar-baar.
B: Forward pass, compute the loss, backpropagate to get a gradient for every weight, and take a small step against it.

Q: Embedding kya hai?
S: Har token ka "pata": numbers ki ek list. Milte-julte shabdon ke pate paas hote hain.
B: A learned vector per token. Similar tokens end up with similar vectors.

Q: Temperature kya karta hai?
S: Kam = safe aur ek jaisa, zyada = creative par galat bhi.
B: It divides the logits before softmax. Low sharpens the distribution, high flattens it.

### Intermediate
Q: Attention kaise kaam karta hai?
S: Har shabd baaki shabdon ko dekhta hai aur kaam wale se jaankari le leta hai.
B: Each token makes a query, key and value. The query-key dot product, after softmax, weights a sum of the values.

Q: Causal mask kyun?
S: Taaki model agla shabd dekh ke nakal na kare.
B: It stops a position attending to the future, so I can train on every position at once without leaking the answer.

Q: Position ka pata kaise chalta hai?
S: Har shabd ke vector ko uski jagah ke hisaab se ghuma dete hain.
B: I use RoPE: queries and keys are rotated by an angle that depends on position, so attention sees relative position.

Q: Overfitting kaise pakdoge?
S: Purane paper pe pass, naye pe fail.
B: Training loss keeps falling while validation loss rises. I keep the checkpoint with the best validation loss.

Q: KV cache kya hai?
S: Purane shabdon ka hisaab yaad rakhna, taaki har baar shuru se na padhna pade.
B: Caching each layer's keys and values so every new token costs one position, not the whole sequence. I test that cached and uncached outputs match.

### Advanced
Q: Pre-norm aur RMSNorm kyun?
S: Deep network mein numbers ka size sambhalne ke liye, saste tareeke se.
B: Pre-norm keeps the residual path clean so gradients flow; RMSNorm is LayerNorm without mean subtraction, cheaper and equally stable.

Q: LoRA kaise kaam karta hai?
S: Poori kitaab dobara nahi likhte, haashiye mein chhote notes chipkate hain.
B: Freeze the weights and learn a low-rank update B·A beside them. B starts at zero so step zero is the original model. I trained 0.9% of the parameters.

Q: Quantization se kya khota hai?
S: Numbers ko chhota likhne se thodi baareeki jaati hai.
B: int8 per-channel cost me nothing in validation loss; at 4 bits per-channel mattered, and 2 bits broke the model. I measured each on the same batches.

Q: Model galat baat kyun bolta hai?
S: Use sach nahi pata, sirf ye ki kaunsa shabd accha lagta hai agla.
B: It is trained for likely text, not true text, and has no way to verify. Grounding with retrieval is the usual fix.

Q: Pehla loss kitna hona chahiye, aur kyun dekhte ho?
S: ln(vocabulary). Hamara 8.37 aaya, hona tha 8.32.
B: A random model spreads probability evenly, so the first loss should be about ln(vocab size). If it is not, the setup has a bug.

## STT

### Beginner
Q: Awaaz computer mein kaise hoti hai?
S: Hawa ke dabaav ki naap, ek second mein 16,000 baar. Numbers ki lambi list.
B: A waveform: amplitude samples at a sampling rate. I use 16 kHz, which captures speech up to 8 kHz.

Q: Spectrogram kya hai?
S: Awaaz ki tasveer: kab kaunsi frequency kitni zor se baji.
B: The magnitude of a short-time Fourier transform over time. It makes phonetic content visible, which the raw waveform does not.

Q: Mel scale kyun?
S: Kaan neeche ki frequencies mein baareek fark sunta hai, upar nahi.
B: It spaces frequency bands like human hearing: fine at the bottom, coarse at the top. 80 mel bands is the standard input.

Q: WER kya hai?
S: 100 shabdon mein kitne galat.
B: Word error rate: the edit distance between hypothesis and reference in words, divided by the reference length. It can exceed 100%.

Q: STT ko train karne ke liye kya chahiye?
S: Awaaz aur uska likha hua roop, saath-saath.
B: Paired audio and transcripts. It is supervised learning, which is why speech data is expensive.

### Intermediate
Q: CTC kya hal karta hai?
S: Kaunsa frame kis akshar ka hai, ye bataye bina training.
B: The alignment problem. The model emits a symbol per frame, and the loss sums the probability of all paths that collapse to the transcript.

Q: Blank symbol kyun?
S: "Abhi kuch naya nahi" kehne ke liye, aur do "l" ko alag rakhne ke liye.
B: It lets the model emit nothing on a frame, and it separates genuine repeated characters, which would otherwise be merged.

Q: Greedy aur beam search?
S: Greedy sabse accha raasta leta hai; beam sabse accha text.
B: Greedy takes the best symbol per frame. Prefix beam search keeps the best texts and adds up the paths that collapse to each.

Q: Subsampling kyun karte ho?
S: 100 frames har second zyada hain; 25 kaafi hain aur attention sasta ho jaata hai.
B: Two stride-2 convolutions reduce 100 frames per second to 25. Attention cost falls 16 times and each frame covers a useful patch.

Q: SpecAugment kya hai?
S: Training mein tasveer ki kuch pattiyan chhupa dena.
B: Masking random time and frequency stripes of the spectrogram. It is the main regulariser when speech data is small.

### Advanced
Q: CTC ki kamzori kya hai?
S: Har frame akela faisla leta hai; use yaad nahi ki pehle kya likha.
B: Conditional independence between frames: no internal language model. Transducers and attention decoders remove it at the cost of speed and simplicity.

Q: CTC, RNN-T aur Whisper mein fark?
S: CTC sabse saral, RNN-T streaming ke liye, Whisper decoder wala.
B: CTC is encoder-only and non-autoregressive. RNN-T adds a prediction network over past outputs. Whisper is an attention encoder-decoder, most accurate but not streaming.

Q: Padding se kya bug aa sakta hai?
S: Batch ki khaali jagah agar model dekh le to wahi clip alag jawab dega.
B: If convolutions or attention read padded frames, a clip's output depends on its batch. I mask in three places and test that alone and batched outputs match.

Q: Training plateau pe atki, kya kiya?
S: Model sirf blank bol raha tha. Warmup lamba kiya.
B: It was the CTC blank plateau. A controlled experiment showed the warmup was too short; with 3000 steps it left the plateau.

Q: Forced alignment kya hai?
S: Diye hue text ke liye pata karna ki kaunsa shabd kab bola gaya.
B: The Viterbi version of the CTC recursion: the single best path for a known transcript. It gives word timestamps without timing labels.

## TTS

### Beginner
Q: TTS ke do hisse kaunse hain?
S: Pehle text se tasveer (mel), phir tasveer se awaaz.
B: An acoustic model that predicts a mel spectrogram, and a vocoder that turns it into a waveform.

Q: TTS, STT se mushkil kyun?
S: STT jaankari phenkta hai; TTS ko wo wapas banani padti hai.
B: It is one-to-many: one sentence can be spoken many ways, and the model must invent pitch, timing and phase.

Q: Vocoder kya hai?
S: Mel ki tasveer se asli awaaz banane wala.
B: A model that generates the waveform from a mel spectrogram, which means reconstructing the phase the spectrogram dropped.

Q: Duration kya hai?
S: Ek akshar kitni der bola jaata hai.
B: The number of mel frames each character or phoneme lasts. Predicting it is what lets a model generate all frames in parallel.

Q: TTS mein punctuation kyun rakhte ho?
S: Comma rukna hai, sawaaliya nishaan uthta sur.
B: Punctuation carries prosody: pauses and pitch movement. Recognition drops it; synthesis needs it.

### Intermediate
Q: Autoregressive aur non-autoregressive TTS?
S: Ek-ek frame banana, ya sab ek saath.
B: Tacotron generates frame by frame with attention and can skip or repeat words. FastSpeech-style models fix durations first and generate in parallel.

Q: MAS kya hai?
S: Training ke dauraan khud pata karna ki har akshar ko kitne frames mile.
B: Monotonic alignment search: dynamic programming for the best monotonic text-to-frame alignment, used to get durations without an external aligner.

Q: Length regulator?
S: Har akshar ka vector utni baar dohrana jitni uski duration.
B: It expands the text-rate sequence to frame rate by repeating each encoder output by its duration.

Q: Griffin-Lim kya hai, aur kyun kaafi nahi?
S: Phase ka andaza lagane ka purana tareeka; jhanjhanata hai.
B: An iterative phase estimate from magnitudes only. It is consistent but not natural, which is why neural vocoders exist.

Q: Characters ya phonemes?
S: Hum seedhe akshar dete hain; Devanagari lagbhag jaisa likha waisa bola jaata hai.
B: I use raw characters, like AI4Bharat's Indic TTS. The cost in Hindi is schwa deletion, which the model must learn from examples.

### Advanced
Q: L1 loss se awaaz dhundhli kyun hoti hai?
S: Wo saare sambhav jawabon ka ausat banata hai.
B: L1 rewards the average of all plausible spectrograms, which is over-smoothed. Flow matching, diffusion or a GAN loss fix that.

Q: HiFi-GAN aur Vocos mein fark?
S: Ek mel ko phula ke waveform banata hai, doosra frame rate pe rehke phase predict karta hai.
B: HiFi-GAN upsamples with transposed convolutions, so most compute is at audio rate. Vocos predicts magnitude and phase per frame and uses one inverse STFT.

Q: Multi-resolution STFT loss kyun?
S: Chhoti window timing dekhti hai, lambi pitch.
B: Comparing spectrograms at several window sizes covers the time-frequency trade-off, and avoids comparing waveforms whose phases differ but sound the same.

Q: Distillation kya hai, aur tumne kahan kiya?
S: Bada model bolta hai, chhota uski nakal seekhta hai.
B: Training a student on a teacher's outputs. My TTS training audio was generated by a 99 million parameter teacher; the student cannot exceed it.

Q: TTS ko naapte kaise ho?
S: Log sunke number dete hain; ya machine se likhwa ke dekhte hain.
B: MOS from listeners for naturalness. For intelligibility I run the audio through a strong recognizer and compare its CER with the teacher's on the same sentences.

## Orchestration

### Beginner
Q: Voice agent ka pipeline kya hai?
S: Suno, likho, socho, bolo.
B: VAD, speech-to-text, turn detection, an LLM, sentence chunking and text-to-speech: a cascaded pipeline.

Q: VAD kya hai?
S: Mic pe koi bol raha hai ya nahi.
B: Voice activity detection: it marks where an utterance starts and ends, so the recognizer only gets speech.

Q: Latency kya hai aur kyun zaroori?
S: User ke rukne aur agent ke bolne ke beech ka samay.
B: Time to first audio. Above roughly a second a conversation stops feeling natural.

Q: Barge-in kya hai?
S: User beech mein bole to agent turant chup.
B: Interrupting the agent while it speaks: playback is cut and the rest of the reply is cancelled.

Q: Streaming kyun?
S: Poore jawab ka intezaar karoge to der hogi.
B: Each stage starts on partial output of the one before, so the first sentence is spoken while the rest is still being generated.

### Intermediate
Q: End-of-turn detection mushkil kyun?
S: Beech ka rukna aur baat ka ant dono chuppi hain.
B: Silence alone cannot tell a pause from the end. I combine silence duration with a semantic signal from the LLM.

Q: Hysteresis kya hai VAD mein?
S: Shuru hone aur rukne ke alag niyam, taaki faisla baar-baar na palte.
B: Different thresholds for starting and stopping: 100 ms of speech to start, 500 ms of silence to end, plus a pre-roll buffer.

Q: Latency kahan-kahan jaati hai?
S: Chuppi ka intezaar, STT, LLM ka pehla tukda, TTS ka pehla vaakya.
B: Endpointing delay, STT time, LLM time to first token and TTS time for the first sentence. I log each per turn.

Q: Model calls thread mein kyun?
S: Warna sochte waqt agent behra ho jaayega.
B: They block. Running them in worker threads keeps the event loop reading the microphone, which is what makes barge-in possible.

Q: Sentence chunking kyun?
S: TTS ko poora vaakya chahiye, par poore jawab ka intezaar nahi.
B: Prosody needs whole sentences, so I regroup the token stream into sentences and synthesize each as soon as it closes.

### Advanced
Q: Cancellation sahi kaise karte ho?
S: Har jawab ka number; purana number wala kaam khud ruk jaata hai.
B: A generation counter. Threads cannot be killed, so every stage checks that its generation is still current before acting.

Q: False cutoff aur latency ka trade-off?
S: Jaldi jawab doge to tokne ka khatra badhta hai.
B: A shorter endpointing delay cuts latency but raises false cutoffs. I use a short delay when the text looks finished and a long one when it does not.

Q: Echo ka kya karte ho?
S: Speaker ki awaaz mic mein jaati hai; agent khud ko tok dega.
B: It needs acoustic echo cancellation. Browsers and WebRTC provide it; my local script does not, so it needs headphones.

Q: WebRTC kyun, WebSocket kyun nahi?
S: WebRTC der se aaye packet ko chhod deta hai; WebSocket atak jaata hai.
B: WebRTC runs on UDP with a jitter buffer, loss concealment and echo cancellation. WebSocket is TCP, so one lost packet stalls everything behind it.

Q: Orchestrator ko test kaise kiya?
S: Asli models ki jagah nakli functions, tay delay ke saath.
B: Fake models with known delays, and scenario tests: a full turn, a barge-in, a pause that is not the end, and a model that raises.
