# LLM glossary: beginner se advanced

Har term ke saamne wo jagah likhi hai jahan tune use khud code mein banaya hai (file ka naam, phir wo naam jo us file mein search karna hai).
File kholo, naam search karo, wahi line hai. `check.py` fail ho jaata hai agar koi pointer galat ho.

"Kahan" mein `theory` ka matlab: ye term is project ke code mein nahi hai, par interview mein poocha jaata hai.

## A. Neural network ki neenv (lesson 1: autograd.py)

| # | Term | Seedhi bhasha mein | Kahan |
|---|---|---|---|
| 1 | Parameter / weight | Wo numbers jo model seekhta hai. Model ka "gyaan" inhi mein hai. | `autograd.py: self.w` |
| 2 | Bias | Har neuron mein juda ek extra number, jo output ko upar-neeche khiskata hai. | `autograd.py: self.b` |
| 3 | Neuron | Inputs ko weights se guna karke jodta hai, phir activation lagata hai. | `autograd.py: class Neuron` |
| 4 | Layer | Ek saath kaam karne wale neurons ka group. | `autograd.py: self.layers` |
| 5 | MLP | Multi-layer perceptron: ek ke baad ek layers. Transformer ke har block mein ek hota hai. | `autograd.py: class MLP` |
| 6 | Activation function | Non-linear function. Iske bina kitni bhi layers ek hi linear layer ke barabar hain. | `autograd.py: def tanh` |
| 7 | Forward pass | Input se output tak numbers ko chalana. | `autograd.py: def __call__` |
| 8 | Loss | Ek number jo batata hai prediction kitni galat hai. Training ise kam karti hai. | `autograd.py: loss = sum` |
| 9 | MSE | Mean squared error: (prediction - target) ka square. Numbers predict karne ke liye. | `autograd.py: mean squared error` |
| 10 | Gradient | "Is weight ko thoda badhaun to loss kitna badlega." Har weight ka apna. | `autograd.py: self.grad` |
| 11 | Computation graph | Kaunsa number kis number se bana, uska record. Backprop isi pe chalta hai. | `autograd.py: self._parents` |
| 12 | Chain rule | Calculus ka niyam: jude hue functions ka derivative = unke derivatives ka guna. | `autograd.py: self._backward` |
| 13 | Backpropagation | Loss se peeche ki taraf chain rule lagakar har weight ka gradient nikalna. | `autograd.py: def backward` |
| 14 | Topological order | Graph ko us order mein chalna ki koi node apne dependents se pehle na aaye. | `autograd.py: def visit` |
| 15 | Autograd | Gradients apne aap nikalne wala system. PyTorch ka dil. | `autograd.py: class Value` |
| 16 | Gradient descent / SGD | Har weight ko uske gradient ke ulte taraf thoda khiskana. | `autograd.py: p.data -= lr * p.grad` |
| 17 | Learning rate | Har step kitna bada ho. Zyada: loss phat jaata hai. Kam: seekhna dheema. | `autograd.py: lr = 0.05` |
| 18 | Zero grad | Gradients jud-te jaate hain, isliye har step se pehle saaf karna padta hai. | `autograd.py: p.grad = 0.0` |
| 19 | Random initialization | Weights random se shuru, warna saare neurons ek jaisa hi seekhenge (symmetry). | `autograd.py: random.uniform` |
| 20 | Gradient check | Backprop ka gradient "nudge karke dekho" wale gradient se milana. | `autograd.py: numeric` |
| 21 | Divergence | Loss ghatne ki jagah badhkar infinity ho jaana. Aksar learning rate zyada hone se. | `autograd.py: divergence` |
| 22 | Tensor | Numbers ka multi-dimensional array. PyTorch mein sab kuch tensor hai. | `model.py: torch.Tensor` |

## B. Text se numbers (lesson 2-3: tokenizer.py, data.py)

| # | Term | Seedhi bhasha mein | Kahan |
|---|---|---|---|
| 23 | Token | Text ka wo tukda jo model ek unit maanta hai. Akshar se bada, shabd se chhota. | `tokenizer.py: def encode` |
| 24 | Tokenizer | Text ko token ids mein aur ids ko wapas text mein badalta hai. | `tokenizer.py: class BPE` |
| 25 | Vocabulary | Saare possible tokens ki list. Hamari 4096 ki hai. | `tokenizer.py: self.vocab` |
| 26 | Vocab size | Vocabulary mein kitne tokens. Bada: chhoti sequence par bada embedding table. | `tokenizer.py: def vocab_size` |
| 27 | UTF-8 / bytes | Text ko bytes mein likhne ka tareeka. Ek Devanagari akshar = 3 bytes. | `tokenizer.py: encode("utf-8")` |
| 28 | Byte-level BPE | 256 bytes se shuru karna, taaki koi bhi script ya emoji encode ho sake. | `tokenizer.py: range(256)` |
| 29 | BPE | Byte-pair encoding: sabse zyada saath aane wale pair ko baar-baar ek token bana dena. | `tokenizer.py: def train` |
| 30 | Merge | Do tokens ko jodkar ek naya token banana. | `tokenizer.py: def merge` |
| 31 | Merge rank | Merges usi order mein lagte hain jis order mein seekhe gaye. | `tokenizer.py: self.rank` |
| 32 | Pre-tokenization | Pehle text ko shabdon mein kaatna, taaki merge shabd ke bahar na jaye. | `tokenizer.py: CHUNK` |
| 33 | Subword | Shabd ka hissa. Naya shabd bhi purane tukdon se ban jaata hai. | `tokenizer.py: def piece` |
| 34 | OOV (out of vocabulary) | Aisa shabd jo vocabulary mein nahi. Byte-level BPE mein ye problem hoti hi nahi. | `check.py: unseen` |
| 35 | Special token | Haath se daale gaye control tokens, text se nahi aate. | `tokenizer.py: SPECIALS` |
| 36 | BOS / EOS | Document ki shuruaat / ant. EOS aate hi generation rukta hai. | `data.py: bos, eos` |
| 37 | Control token / conditional generation | Language tag dekar model se usi bhasha mein likhwana. | `generate.py: def reply` |
| 38 | Encode / decode | Text → ids, aur ids → text. Round trip exact hona chahiye. | `tokenizer.py: def decode` |
| 39 | Compression ratio | Bytes per token. Achha tokenizer kam tokens mein zyada text rakhta hai. | `tokenizer.py: compression` |
| 40 | Corpus | Training ka poora text. | `data.py: def documents` |
| 41 | Dataset | Corpus ka source. Hamare teen: Hindi, Marathi, Hinglish. | `data.py: SOURCES` |
| 42 | TinyStories | Chhote shabdon wali kahaniyan, jinse chhota model bhi grammar seekh leta hai. | `data.py: TinyStories` |
| 43 | Code-switching | Ek hi vaakya mein do bhashayein (Hinglish). | `data.py: Hinglish` |
| 44 | Train / validation split | Kuch data alag rakhna jis pe model kabhi train nahi hota. | `data.py: n_val` |
| 45 | Shuffle | Data ka order random karna, taaki split aur batches fair hon. | `data.py: shuffle` |
| 46 | Reproducibility / seed | Fixed seed se har baar wahi result. | `train.py: manual_seed` |
| 47 | Token ids ki binary file | Tokens ko uint16 array mein disk pe rakhna. | `data.py: np.uint16` |
| 48 | Memory map | Poori file RAM mein laaye bina disk se padhna. | `train.py: np.memmap` |
| 49 | WordPiece / SentencePiece / Unigram | BPE ke rishtedaar tokenizers (BERT, T5, Gemma). | theory |

## C. Transformer ke andar (lesson 4: model.py)

| # | Term | Seedhi bhasha mein | Kahan |
|---|---|---|---|
| 50 | Language model | Model jo agle token ki probability deta hai. | `model.py: class GPT` |
| 51 | Transformer | Attention + feed-forward blocks ka stack. 2017 ka "Attention Is All You Need". | `model.py: self.blocks` |
| 52 | Decoder-only | Sirf left-to-right padhne wala transformer. GPT, Llama, Gemma sab yahi. | `model.py: decoder-only` |
| 53 | Autoregressive | Har naya token pichhle tokens pe depend karta hai. | `generate.py: autoregressive` |
| 54 | Hyperparameter | Jo hum chunte hain (layers, width), model nahi seekhta. | `model.py: class Config` |
| 55 | Embedding | Token id ko seekhe hue vector mein badalna. Milte-julte tokens ke vector paas hote hain. | `model.py: nn.Embedding` |
| 56 | d_model (width) | Har token ke vector ki lambai. Hamari 384. | `model.py: d_model` |
| 57 | Context window | Model ek baar mein kitne tokens dekh sakta hai. Hamara 256. | `model.py: ctx: int` |
| 58 | Depth (n_layer) | Kitne blocks. Zyada depth = zyada steps mein sochna. | `model.py: n_layer` |
| 59 | Residual stream | Wo vector jo blocks ke beech behta hai; har block isme kuch jodta hai. | `model.py: residual stream` |
| 60 | Residual / skip connection | `x = x + f(x)`. Gradient seedha neeche tak pahunchta hai, isliye deep model train hota hai. | `model.py: x = x + a` |
| 61 | Vanishing / exploding gradient | Deep network mein gradient ka bahut chhota ya bahut bada ho jaana. | `model.py: class Block` |
| 62 | Self-attention | Har token baaki tokens se zaroori jaankari uthata hai. | `model.py: class CausalSelfAttention` |
| 63 | Query, Key, Value | Query = sawaal, Key = label, Value = content. Query jis key se mile uski value milti hai. | `model.py: self.qkv` |
| 64 | Dot product | Do vectors kitne milte hain uska number. Attention score yahi hai. | `model.py: q @ k.transpose` |
| 65 | Scaled dot-product | Score ko sqrt(head_dim) se bhaag dena, warna softmax ek hi token pe atak jaata hai. | `model.py: math.sqrt(self.head_dim)` |
| 66 | Softmax | Scores ko probabilities mein badalta hai jo milkar 1 hoti hain. | `model.py: F.softmax` |
| 67 | Attention weights | Softmax ke baad ke numbers: kis token pe kitna dhyaan. | `model.py: self.last_att` |
| 68 | Causal mask | Future tokens ko chhupana. Isi se ek hi pass mein har position pe training ho paati hai. | `model.py: mask = ` |
| 69 | Multi-head attention | Kai chhote attention ek saath, har ek alag rishta pakadta hai. | `model.py: n_head` |
| 70 | Head dimension | d_model / n_head. Har head ke vector ki lambai. | `model.py: self.head_dim` |
| 71 | Output projection | Saare heads ko wapas ek vector mein milana. | `model.py: self.proj` |
| 72 | Positional encoding | Attention ko order nahi pata, isliye position alag se batani padti hai. | `model.py: def rope_table` |
| 73 | RoPE | Query aur key ko position ke hisaab se ghumana; dot product sirf doori pe depend karta hai. | `model.py: def apply_rope` |
| 74 | Absolute vs relative position | "Main 5th token hun" banam "wo mujhse 3 peeche hai". RoPE relative deta hai. | `model.py: relative position` |
| 75 | Feed-forward network | Har token pe alag se chalne wala MLP. Zyadatar parameters yahin. | `model.py: class FeedForward` |
| 76 | SiLU / Swish | `x * sigmoid(x)`. ReLU ka chikna roop. | `model.py: F.silu` |
| 77 | SwiGLU | Gate wala feed-forward: ek branch doosri ko control karti hai. Llama, Gemma mein. | `model.py: self.gate` |
| 78 | ReLU / GELU | Purane activations. Original transformer ReLU, GPT-2 GELU. | theory |
| 79 | Normalization | Vectors ka size sthir rakhna, taaki training na bigde. | `model.py: class RMSNorm` |
| 80 | LayerNorm vs RMSNorm | RMSNorm = LayerNorm bina mean aur bias ke. Sasta, utna hi achha. | `model.py: self.gain` |
| 81 | Pre-norm vs post-norm | Norm sub-layer se pehle (aaj) ya baad mein (2017). Pre-norm zyada stable. | `model.py: self.norm1` |
| 82 | Transformer block | Norm → attention → add, norm → feed-forward → add. | `model.py: class Block` |
| 83 | Logits | Softmax se pehle ke raw scores, har vocab entry ka ek. | `model.py: logits = self.lm_head` |
| 84 | LM head / unembedding | Aakhri vector ko vocab-size scores mein badalne wali layer. | `model.py: self.lm_head` |
| 85 | Weight tying | Input embedding aur output head ek hi matrix. Parameters bachte hain. | `model.py: self.lm_head.weight = self.tok_emb.weight` |
| 86 | Weight initialization | Shuruaati random values ka size. Galat ho to training shuru hi nahi hoti. | `model.py: nn.init.normal_` |
| 87 | Parameter count | Model ka size. Hamara 12.2 million. | `model.py: def n_params` |
| 88 | Buffer | Model ke saath chalne wala tensor jo train nahi hota (RoPE table). | `model.py: register_buffer` |
| 89 | Batch / time / channel (B, T, C) | Tensor ke teen dimension: kitni sequences, kitne tokens, vector ki lambai. | `model.py: B, T, C` |
| 90 | Encoder / encoder-decoder | BERT sirf encoder (dono taraf dekhta), T5 dono. STT mein encoder milega. | theory |
| 91 | Cross-attention | Ek sequence doosri sequence pe attention. Whisper aur TTS mein. | theory |
| 92 | Flash attention / GQA / MQA | Attention ko tez ya sasta karne ke tareeke. | theory |

## D. Training (lesson 5: train.py)

| # | Term | Seedhi bhasha mein | Kahan |
|---|---|---|---|
| 93 | Self-supervised learning | Label khud text se aata hai: agla token. Kisi insaan ne label nahi kiya. | `train.py: self-supervised` |
| 94 | Pre-training | Bahut saare text pe bhasha seekhna. | `train.py: def main` |
| 95 | Next-token prediction | Training ka poora lakshya. | `train.py: def get_batch` |
| 96 | Teacher forcing | Training mein model ko sahi pichhle tokens dena, uske apne guess nahi. | `model.py: Teacher forcing` |
| 97 | Cross-entropy loss | Sahi token ko di gayi probability ka negative log. | `model.py: def cross_entropy` |
| 98 | Log-softmax / logsumexp | Softmax ka log, numerically stable tareeke se. | `model.py: logsumexp` |
| 99 | Perplexity | exp(loss). "Model itne tokens ke beech confuse hai." | `train.py: math.exp(va)` |
| 100 | Starting loss = ln(vocab) | Random model ka loss. Isse pata chalta hai setup sahi hai. | `train.py: math.log(tok.vocab_size)` |
| 101 | Batch / batch size | Ek step mein kitni sequences ek saath. | `train.py: --batch` |
| 102 | Iteration / step | Ek batch pe ek update. | `train.py: for it in range` |
| 103 | Epoch | Poore data pe ek chakkar. | `train.py: epochs` |
| 104 | Optimizer | Gradient se weights update karne ka niyam. | `train.py: opt.step()` |
| 105 | Adam / AdamW | Momentum + har weight ka apna step size. W = sahi tareeke ka weight decay. | `train.py: AdamW` |
| 106 | Momentum / betas | Pichhle gradients ka chalta hua average. | `train.py: betas` |
| 107 | Weight decay | Weights ko zero ki taraf kheenchna. Ek regularizer. | `train.py: weight_decay` |
| 108 | Regularization | Jo cheez overfitting rokti hai (dropout, weight decay). | `model.py: dropout` |
| 109 | Dropout | Training mein kuch numbers random zero karna, taaki model kisi ek pe na tike. | `model.py: nn.Dropout` |
| 110 | LR schedule | Training ke dauran learning rate badalna. | `train.py: def lr_at` |
| 111 | Warmup | Shuru mein learning rate dheere badhana. | `train.py: warmup` |
| 112 | Cosine decay | Learning rate ko cosine curve pe ghatana. | `train.py: math.cos` |
| 113 | Gradient clipping | Bahut bade gradient ko chhota kar dena. | `train.py: clip_grad_norm_` |
| 114 | Gradient norm | Saare gradients ki kul lambai. Training ki sehat ka meter. | `train.py: grad_norm` |
| 115 | Train loss vs val loss | Train: seekhe hue data pe. Val: andekhe data pe. | `train.py: def estimate_loss` |
| 116 | Overfitting | Train loss girta rahe, val loss badhne lage: model ratta maar raha hai. | `train.py: overfitting` |
| 117 | Underfitting | Dono loss oonche: model chhota hai ya kam train hua. | theory |
| 118 | Generalization | Andekhe text pe achha karna. Asli lakshya. | `data.py: generalization` |
| 119 | Checkpoint | Training ke beech save kiya gaya model. | `model.py: def save` |
| 120 | Early stopping | Sabse achhe val loss wala checkpoint rakhna. | `train.py: if va < best` |
| 121 | train() / eval() mode | Dropout on / off. | `train.py: model.eval()` |
| 122 | Device (CPU / GPU / MPS) | Hisaab kahan hota hai. MPS = Mac ka GPU. | `model.py: def pick_device` |
| 123 | Throughput (tokens/sec) | Training ki speed. | `train.py: tok/s` |
| 124 | Gradient accumulation | Kai chhote batches ke gradient jodkar ek bada batch banana. | theory |
| 125 | Mixed precision (AMP) | fp16 mein train karna, tez aur kam memory. | theory |
| 126 | Scaling laws / compute budget | Data, parameters aur compute ka rishta. | theory |

## E. Generation (lesson 6: generate.py)

| # | Term | Seedhi bhasha mein | Kahan |
|---|---|---|---|
| 127 | Inference | Train hue model ko use karna. | `generate.py: torch.no_grad` |
| 128 | Decoding / sampling | Probabilities se ek token chunna. | `generate.py: def sample` |
| 129 | Greedy decoding | Hamesha sabse likely token. | `generate.py: argmax` |
| 130 | Temperature | Logits ko T se bhaag. Kam = safe, zyada = creative. | `generate.py: temperature` |
| 131 | Top-k | Sirf k sabse likely tokens rakhna. | `generate.py: top_k` |
| 132 | Top-p / nucleus | Utne tokens rakhna jinki probability milkar p ho. | `generate.py: top_p` |
| 133 | Multinomial sampling | Probabilities ke hisaab se pasa phenkna. | `generate.py: torch.multinomial` |
| 134 | Repetition penalty | Pehle aa chuke tokens ko dabana. | `generate.py: repetition_penalty` |
| 135 | Stop token | Jis token pe generation ruk jaye. | `generate.py: stop` |
| 136 | KV cache | Purane tokens ke keys / values yaad rakhna, dobara na nikalna. | `model.py: past` |
| 137 | Prompt | Wo text jisse generation shuru hota hai. | `generate.py: prompt` |
| 138 | Prompt format / chat template | Prompt ka wahi dhaancha jo training data ka tha. | `generate.py: def reply` |
| 139 | Hallucination | Model ka poore vishwas se galat baat kehna. Wo sach nahi, agla token predict karta hai. | theory |
| 140 | Beam search | Kai raaste saath rakhkar sabse achha chunna. STT mein milega. | theory |
| 141 | Time to first token (TTFT) | Pehla token aane mein laga samay. Voice agent mein sabse zaroori. | theory |
| 142 | Streaming | Tokens bante hi bhejna. | theory |

## F. Fine-tuning aur chhota karna (lesson 7-9: lora.py, shrink.py, xray.py)

| # | Term | Seedhi bhasha mein | Kahan |
|---|---|---|---|
| 143 | Fine-tuning | Train hue model ko naye kaam pe thoda aur train karna. | `lora.py: def main` |
| 144 | Transfer learning | Ek kaam ka gyaan doosre kaam mein lagana. | `lora.py: Pre-training` |
| 145 | Full fine-tuning | Saare weights update karna. Mehenga. | `lora.py: Full fine-tuning` |
| 146 | PEFT | Parameter-efficient fine-tuning: thode se weights train karna. | `lora.py: def add_lora` |
| 147 | LoRA | Frozen W ke saath chhota B·A jodna. | `lora.py: class LoRALinear` |
| 148 | Rank (r) | A aur B kitne patle hain. Chhota r = kam parameters. | `lora.py: r: int` |
| 149 | Alpha / scale | LoRA ke update ki taakat: alpha / r. | `lora.py: self.scale` |
| 150 | Frozen weights | Jo train nahi hote. | `lora.py: requires_grad_(False)` |
| 151 | Adapter | Sirf naye weights wali chhoti file. | `lora.py: persona_adapter.pt` |
| 152 | Merge | B·A ko W mein jod dena; speed wapas wahi. | `lora.py: def merge` |
| 153 | Padding | Chhoti sequences ko lamba karna taaki batch chaukor bane. | `lora.py: Padding` |
| 154 | Ignore index | Padding wali positions ko loss se bahar rakhna. | `lora.py: -100` |
| 155 | Catastrophic forgetting | Naya seekhte hue purana bhool jaana. | `lora.py: catastrophic forgetting` |
| 156 | Quantization | Weights ko kam bits mein rakhna. | `shrink.py: def quant_roundtrip` |
| 157 | fp32 / fp16 / int8 | 4, 2, 1 byte per weight. | `shrink.py: w.half()` |
| 158 | Scale (quantization) | Float aur integer ke beech ka guna-bhaag. | `shrink.py: scale` |
| 159 | Per-tensor vs per-channel | Poori matrix ka ek scale banam har row ka apna. | `shrink.py: per_channel` |
| 160 | PTQ vs QAT | Training ke baad quantize banam training ke dauran. Gemma QAT hai. | `shrink.py: post-training quantization` |
| 161 | GGUF | llama.cpp ka file format, quantized models ke liye. | theory |
| 162 | Distillation | Bade model se chhote model ko sikhana. | theory |
| 163 | Entropy | Model kitna unsure hai, bits mein. | `xray.py: entropy` |
| 164 | Interpretability | Model ke andar dekhkar samajhna ki wo kya kar raha hai. | `xray.py: last_att` |
| 165 | RAG / instruction tuning / RLHF | Bade LLMs ko kaam ka banane ke tareeke. | theory |
| 166 | QK norm | Query aur key ko barabar lambai ka karna; score sirf disha (cosine similarity) naapta hai. | `model.py: qk_norm` |
| 167 | Zero-init projection | Har block ka output shuru mein zero, to naya model pehle "kuch nahi" karta hai. | `model.py: zero_init` |
| 168 | Ablation study | Ek cheez badlo, baaki same rakho, aur naapo. Dekho COMPARE.md. | `train.py: --qk-norm` |
| 169 | Replay / rehearsal | Fine-tune ke saath purana data bhi dikhana, taaki model bhoole nahi. | `lora.py: Replay` |
