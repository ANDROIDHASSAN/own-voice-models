# Chhotu GPT: ek LLM, scratch se

Ek 12 million parameter ka language model jo Hindi, Marathi aur Hinglish likhta hai. Tokenizer, attention,
transformer, training loop, sampling, LoRA aur quantization: sab is folder mein haath se likha hai.
PyTorch sirf tensors aur autograd ke liye hai, aur autograd bhi lesson 1 mein khud banaya hai.

Ye model Gemma ki jagah nahi lega. Iska kaam hai sikhana ki LLM andar se kaise chalta hai.

## Padhne ka order

Har file ek lesson hai. Upar ka docstring padho, phir code, phir chalao.

| # | File | Kya seekhoge | Chalao |
|---|---|---|---|
| 1 | [autograd.py](autograd.py) | Network seekhta kaise hai: gradient, backprop, gradient descent | `python autograd.py` |
| 2 | [tokenizer.py](tokenizer.py) | Text ko numbers mein badalna: BPE | `python tokenizer.py` |
| 3 | [data.py](data.py) | Internet ke text se training arrays tak | `python data.py` |
| 4 | [model.py](model.py) | Transformer: embedding, attention, RoPE, feed-forward, residual | (import hota hai) |
| 5 | [train.py](train.py) | Training loop: loss, AdamW, LR schedule, clipping, val loss | `python train.py` |
| 6 | [generate.py](generate.py) | Probabilities se text: temperature, top-k, top-p, KV cache | `python generate.py --lang hi --prompt "एक दिन"` |
| 7 | [lora.py](lora.py) | 1% se kam weights train karke fine-tune | `python lora.py` |
| 8 | [shrink.py](shrink.py) | Quantization: phone ke size tak, aur uski keemat | `python shrink.py` |
| 9 | [xray.py](xray.py) | Model ke andar jhaankna: tokens, belief, attention | `python xray.py` |

[GLOSSARY.md](GLOSSARY.md) mein 169 terms hain, beginner se advanced, har ek ke saath code ki wo jagah jahan wo
bana hai. [check.py](check.py) wo cheezein test karta hai jo galat hon to sab galat ho jaata hai.
[docs/LLM-Zero-Se-Interview-Tak.pdf](docs/LLM-Zero-Se-Interview-Tak.pdf) pehle padho agar kuch nahi aata: 10 seedhiyan,
kahaniyan, haath se hisaab, aur 25 interview sawaal.
[docs/Chhotu-GPT-Andar-Se.pdf](docs/Chhotu-GPT-Andar-Se.pdf) poori study guide hai: diagrams, charts, saare terms
aur interview ke sawaal-jawab (`python docs/build.py` se dobara banti hai).
[COMPARE.md](COMPARE.md) batata hai ki Karpathy, Raschka aur minimind ke repos ke saamne ye kahan khada hai,
aur unke do ideas ka hamare model pe naapa hua asar.

Sab commands is folder se, project ke venv ke saath: `../../.venv/bin/python <file>`.

## Ek token ka safar

```
"एक दिन"
   │  tokenizer.py     BPE: bytes → merges
   ▼
[4091, 4093, 460, 459]              <|bos|> <|hi|> एक ▁दिन
   │  model.py         nn.Embedding
   ▼
4 vectors, har ek 384 numbers ka                    ← residual stream
   │
   │  6 baar:   x = x + attention(norm(x))          tokens ek doosre se baat karte hain
   │            x = x + feed_forward(norm(x))       har token akele sochta hai
   ▼
4 vectors → lm_head → 4 × 4096 logits
   │  generate.py      aakhri position ke logits → temperature → top-k / top-p → ek token chuno
   ▼
agla token, wapas input mein jodo, dobara
```

## Design ke faisle, aur kyun

| Faisla | Kyun |
|---|---|
| Byte-level BPE | Devanagari, Latin, emoji: sab encode hota hai, unknown token kabhi nahi aata |
| Vocab 4096 | Chhota model, chhota data. Bada vocab = embedding table mein zyada parameters, har token ke kam examples |
| Language tags `<|hi|>` `<|mr|>` `<|hing|>` | Ek hi model, reply ki bhasha tag se chunte hain |
| RoPE | Relative position deta hai, aur alag position-embedding table nahi chahiye |
| RMSNorm + pre-norm | Llama / Gemma wala setup. Sasta aur deep stack mein stable |
| SwiGLU | Aaj ke LLMs ka feed-forward. Hidden size 2/3 karke parameter count wahi rakha |
| Weight tying | 4096 × 384 ki ek matrix bachti hai (12 M mein se 1.6 M) |
| TinyStories data | Chhota model chhoti vocabulary se grammar seekh sakta hai, poore internet se nahi |

## Interview mein kaise samjhana

**30 second mein:**
"I built a decoder-only transformer from scratch in PyTorch: my own BPE tokenizer, multi-head causal
self-attention with RoPE, RMSNorm, SwiGLU, weight tying, and a training loop with AdamW, warmup plus cosine
decay and gradient clipping. I trained a 12 million parameter model on Hindi, Marathi and Hinglish, added a KV
cache for inference, LoRA for fine-tuning and int8 quantization, and wrote tests for causality and cache
correctness."

**Sawaal jo aate hain, aur jawab code mein kahan hai:**

| Sawaal | Chhota jawab | Dekho |
|---|---|---|
| Attention kaise kaam karta hai? | Query aur key ka dot product → softmax → values ka weighted sum | `CausalSelfAttention.forward` |
| sqrt(d) se kyun bhaag dete hain? | Bade dot products softmax ko ek token pe atka dete hain, gradient mar jaata hai | `math.sqrt(self.head_dim)` |
| Causal mask kyun? | Taaki har position pe ek saath train ho sake bina future dekhe | `mask = ` in model.py |
| Position kaise pata chalti hai? | RoPE: Q aur K ko position ke angle se ghumate hain | `apply_rope` |
| Residual connection kyun? | Gradient ka seedha raasta; block sirf correction jodta hai | `Block.forward` |
| Loss kya hai? | Sahi agle token ki negative log-probability | `cross_entropy` |
| Pehla loss kitna hona chahiye? | ln(vocab) = 8.32. Hamara 8.37 aaya | train.py ka pehla print |
| Overfitting kaise pakdoge? | Train loss gire, val loss badhe | `estimate_loss` |
| KV cache kya bachata hai? | Har naye token pe poori sequence dobara nahi padhni padti | `past` in model.py |
| KV cache sahi hai, kaise pata? | Token-by-token aur full pass ke logits barabar hain | check.py |
| LoRA mein B zero se kyun shuru? | Taaki step 0 pe model bilkul pehle jaisa rahe | `LoRALinear` |
| Temperature kya karta hai? | Logits ko bhaag deta hai: kam = sharp, zyada = flat | `sample` |
| Quantization se kya khota hai? | Neeche table dekho: int8 tak lagbhag kuch nahi | shrink.py |

**Ek sawaal jiska sachcha jawab dena:** "Ye production mein use hoga?" Nahi. 12 M parameters aur 11 M tokens se
model kahaniyan aur chhoti Hinglish baat likh leta hai, facts ya reasoning nahi. Asli agent Gemma use karta hai;
ye model wahi architecture chhote scale pe hai.

## Naape hue results

Apple M1 Pro, Mac ke GPU (MPS) pe. 12.2 M parameters, 10.9 M training tokens, 4000 iterations (4.5 epochs),
lagbhag 50 minute.

**Training**

| Iteration | Train loss | Val loss | Val perplexity |
|---|---|---|---|
| 0 | 8.37 (ln 4096 = 8.32) | | |
| 500 | 2.64 | 2.74 | 15.5 |
| 1000 | 2.24 | 2.36 | 10.5 |
| 2000 | 1.94 | 2.07 | 8.0 |
| 3000 | 1.77 | 1.92 | 6.8 |
| 4000 | 1.69 | 1.88 | 6.5 |

Val loss aakhir tak gir raha tha, aur train se sirf 0.19 upar hai: model abhi overfit nahi hua. Zyada
iterations ya zyada data se aur sudhrega.

**Jo likhta hai** (temperature 0.8, prompt bold mein)

- Hindi: **एक दिन राजू ने अपनी माँ से कहा** कि वह जंगल में घूमने जाएगा। सभी जानवर बहुत उत्साहित थे। राजू ने सोचा, "क्यों न मैं भी उनके साथ खेलूं?"
- Marathi: **एकदा एक लहान मुलगा** होता, ज्याचे नाव आर्यन होते. आर्यनला समुद्रकिनाऱ्यावर खेळायला खूप आवडत असे.
- Hinglish: **kya chal raha hai bhai?** → kuchh nahin yaar, bas aaram karna hai.

Grammar theek hai, kahani ka logic kabhi-kabhi toot-ta hai (Raju akela jaata hai, phir "sabhi jaanwar" aa jaate hain).

**Quantization** (`python shrink.py`, same validation batches)

| Format | Size | Val loss | Fark |
|---|---|---|---|
| fp32 | 48.8 MB | 1.8448 | |
| fp16 | 24.4 MB | 1.8448 | 0 |
| int8, per-channel | 12.2 MB | 1.8448 | 0 |
| int8, per-tensor | 12.2 MB | 1.8451 | +0.0003 |
| int4, per-channel | 6.1 MB | 1.8728 | +0.03 |
| int4, per-tensor | 6.1 MB | 1.9647 | +0.12 |
| int3, per-channel | 4.6 MB | 2.0325 | +0.19 |
| int3, per-tensor | 4.6 MB | 3.4534 | +1.6 |
| int2, per-channel | 3.0 MB | 8.5676 | toot gaya |

int8 tak muft hai. 4-bit pe per-channel ka fayda saaf dikhta hai (0.03 banam 0.12). 2-bit pe model random se
bhi kharab hai. Size weights ka hisaab hai (parameters × bits / 8), file format ka nahi.

**LoRA fine-tune** (`python lora.py`, 20 sawaal-jawab, 600 steps, rank 8)

| | Pehle | Baad mein |
|---|---|---|
| Train hue parameters | | 110,592 of 12.2 M (0.91%) |
| Persona loss | 4.58 | 0.32 |
| General val loss | 1.829 | 1.888 |
| "tera naam kya hai?" | mera toh bas normal hi hai. | mera naam Chhotu GPT hai. |
| Adapter file | | 0.45 MB (poora model 48.8 MB) |

Do sachchi baatein. General val loss 0.06 badha: thoda bhoola (catastrophic forgetting), replay ke bawajood.
Aur persona aam baat mein ris gaya: "kya chal raha hai bhai?" ka jawab aaya "kuchh nahi yaar, bas phone pe
scratch se banaya gaya." 20 examples pe 600 steps ka matlab ratta hai, samajh nahi. Asli fine-tune ko
hazaaron examples chahiye.

## Kya nahi hai (aur kab jodna)

- **Sliding context.** Generation 256 tokens pe ruk jaata hai. Voice replies ke liye kaafi; lambi baat ke liye window khiskani hogi.
- **Mixed precision, gradient accumulation, multi-GPU.** Is size pe zaroorat nahi.
- **Instruction tuning.** Model sawaal-jawab ka sirf Hinglish chat format jaanta hai, system prompt nahi.
