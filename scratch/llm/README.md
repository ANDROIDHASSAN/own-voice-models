# Chhotu GPT: an LLM, from scratch

A 12 million parameter language model that writes Hindi, Marathi and Hinglish. Tokenizer, attention,
transformer, training loop, sampling, LoRA and quantization: all written by hand in this folder.
PyTorch is used only for tensors and autograd, and autograd itself is built by hand in lesson 1.

This model will not replace Gemma. Its job is to teach how an LLM works on the inside.

## Reading order

Each file is a lesson. Read the docstring at the top, then the code, then run it.

| # | File | What you learn | Run |
|---|---|---|---|
| 1 | [autograd.py](autograd.py) | How a network learns: gradients, backprop, gradient descent | `python autograd.py` |
| 2 | [tokenizer.py](tokenizer.py) | Turning text into numbers: BPE | `python tokenizer.py` |
| 3 | [data.py](data.py) | From internet text to training arrays | `python data.py` |
| 4 | [model.py](model.py) | Transformer: embedding, attention, RoPE, feed-forward, residual | (imported) |
| 5 | [train.py](train.py) | Training loop: loss, AdamW, LR schedule, clipping, val loss | `python train.py` |
| 6 | [generate.py](generate.py) | Text from probabilities: temperature, top-k, top-p, KV cache | `python generate.py --lang hi --prompt "एक दिन"` |
| 7 | [lora.py](lora.py) | Fine-tuning by training less than 1% of the weights | `python lora.py` |
| 8 | [shrink.py](shrink.py) | Quantization: down to phone size, and what it costs | `python shrink.py` |
| 9 | [xray.py](xray.py) | Looking inside the model: tokens, belief, attention | `python xray.py` |

[GLOSSARY.md](GLOSSARY.md) has 169 terms, beginner to advanced, each with the place in the code where it is
built. [check.py](check.py) tests the things that break everything if they are wrong.
Read [docs/LLM-Zero-Se-Interview-Tak.pdf](docs/LLM-Zero-Se-Interview-Tak.pdf) first if you know nothing yet: 10 steps,
stories, hand calculations, and 25 interview questions.
[docs/Chhotu-GPT-Andar-Se.pdf](docs/Chhotu-GPT-Andar-Se.pdf) is the full study guide: diagrams, charts, every term
and interview Q&A (rebuilt with `python docs/build.py`). Both PDFs are in Hinglish.
[COMPARE.md](COMPARE.md) shows where this stands next to the Karpathy, Raschka and minimind repos, and the
measured effect of two of their ideas on our model.

All commands run from this folder with the project venv: `../../.venv/bin/python <file>`.

## The journey of one token

```
"एक दिन"
   │  tokenizer.py     BPE: bytes → merges
   ▼
[4091, 4093, 460, 459]              <|bos|> <|hi|> एक ▁दिन
   │  model.py         nn.Embedding
   ▼
4 vectors, 384 numbers each                         ← residual stream
   │
   │  6 times:  x = x + attention(norm(x))          tokens talk to each other
   │            x = x + feed_forward(norm(x))       each token thinks on its own
   ▼
4 vectors → lm_head → 4 × 4096 logits
   │  generate.py      logits at the last position → temperature → top-k / top-p → pick one token
   ▼
next token, append it to the input, repeat
```

## Design decisions, and why

| Decision | Why |
|---|---|
| Byte-level BPE | Devanagari, Latin, emoji: everything encodes, there is never an unknown token |
| Vocab 4096 | Small model, small data. A bigger vocab = more parameters in the embedding table and fewer examples per token |
| Language tags `<|hi|>` `<|mr|>` `<|hing|>` | One model; the tag picks the reply language |
| RoPE | Gives relative position, and needs no separate position-embedding table |
| RMSNorm + pre-norm | The Llama / Gemma setup. Cheap and stable in a deep stack |
| SwiGLU | The feed-forward of today's LLMs. Hidden size cut to 2/3 to keep the parameter count the same |
| Weight tying | Saves one 4096 × 384 matrix (1.6 M of the 12 M) |
| TinyStories data | A small model can learn grammar from a small vocabulary, not from the whole internet |

## How to explain it in an interview

**In 30 seconds:**
"I built a decoder-only transformer from scratch in PyTorch: my own BPE tokenizer, multi-head causal
self-attention with RoPE, RMSNorm, SwiGLU, weight tying, and a training loop with AdamW, warmup plus cosine
decay and gradient clipping. I trained a 12 million parameter model on Hindi, Marathi and Hinglish, added a KV
cache for inference, LoRA for fine-tuning and int8 quantization, and wrote tests for causality and cache
correctness."

**Questions that come up, and where the answer is in the code:**

| Question | Short answer | See |
|---|---|---|
| How does attention work? | Dot product of query and key → softmax → weighted sum of values | `CausalSelfAttention.forward` |
| Why divide by sqrt(d)? | Large dot products pin softmax on one token and the gradient dies | `math.sqrt(self.head_dim)` |
| Why a causal mask? | So every position can train at once without seeing the future | `mask = ` in model.py |
| How does the model know position? | RoPE: rotate Q and K by an angle that depends on position | `apply_rope` |
| Why residual connections? | A straight path for the gradient; each block only adds a correction | `Block.forward` |
| What is the loss? | Negative log-probability of the correct next token | `cross_entropy` |
| What should the first loss be? | ln(vocab) = 8.32. Ours came out at 8.37 | first print of train.py |
| How do you catch overfitting? | Train loss falls, val loss rises | `estimate_loss` |
| What does the KV cache save? | Re-reading the whole sequence for every new token | `past` in model.py |
| How do you know the KV cache is right? | Token-by-token and full-pass logits are equal | check.py |
| Why does LoRA's B start at zero? | So at step 0 the model is exactly what it was | `LoRALinear` |
| What does temperature do? | Divides the logits: lower = sharper, higher = flatter | `sample` |
| What does quantization lose? | See the table below: almost nothing down to int8 | shrink.py |

**One question to answer honestly:** "Will this be used in production?" No. With 12 M parameters and 11 M tokens,
the model writes stories and short Hinglish chat, not facts or reasoning. The real agent uses Gemma; this model
is the same architecture at a small scale.

## Measured results

Apple M1 Pro, on the Mac's GPU (MPS). 12.2 M parameters, 10.9 M training tokens, 4000 iterations (4.5 epochs),
about 50 minutes.

**Training**

| Iteration | Train loss | Val loss | Val perplexity |
|---|---|---|---|
| 0 | 8.37 (ln 4096 = 8.32) | | |
| 500 | 2.64 | 2.74 | 15.5 |
| 1000 | 2.24 | 2.36 | 10.5 |
| 2000 | 1.94 | 2.07 | 8.0 |
| 3000 | 1.77 | 1.92 | 6.8 |
| 4000 | 1.69 | 1.88 | 6.5 |

Val loss was still falling at the end and is only 0.19 above train: the model has not overfit yet. More
iterations or more data would improve it further.

**What it writes** (temperature 0.8, prompt in bold)

- Hindi: **एक दिन राजू ने अपनी माँ से कहा** कि वह जंगल में घूमने जाएगा। सभी जानवर बहुत उत्साहित थे। राजू ने सोचा, "क्यों न मैं भी उनके साथ खेलूं?"
- Marathi: **एकदा एक लहान मुलगा** होता, ज्याचे नाव आर्यन होते. आर्यनला समुद्रकिनाऱ्यावर खेळायला खूप आवडत असे.
- Hinglish: **kya chal raha hai bhai?** → kuchh nahin yaar, bas aaram karna hai.

The grammar is fine; the story logic sometimes breaks (Raju goes alone, then "all the animals" show up).

**Quantization** (`python shrink.py`, same validation batches)

| Format | Size | Val loss | Difference |
|---|---|---|---|
| fp32 | 48.8 MB | 1.8448 | |
| fp16 | 24.4 MB | 1.8448 | 0 |
| int8, per-channel | 12.2 MB | 1.8448 | 0 |
| int8, per-tensor | 12.2 MB | 1.8451 | +0.0003 |
| int4, per-channel | 6.1 MB | 1.8728 | +0.03 |
| int4, per-tensor | 6.1 MB | 1.9647 | +0.12 |
| int3, per-channel | 4.6 MB | 2.0325 | +0.19 |
| int3, per-tensor | 4.6 MB | 3.4534 | +1.6 |
| int2, per-channel | 3.0 MB | 8.5676 | broken |

Down to int8 it is free. At 4 bits the benefit of per-channel is clear (0.03 vs 0.12). At 2 bits the model is
worse than random. Size is computed from the weights (parameters × bits / 8), not from a file format.

**LoRA fine-tune** (`python lora.py`, 20 question-answer pairs, 600 steps, rank 8)

| | Before | After |
|---|---|---|
| Trained parameters | | 110,592 of 12.2 M (0.91%) |
| Persona loss | 4.58 | 0.32 |
| General val loss | 1.829 | 1.888 |
| "tera naam kya hai?" | mera toh bas normal hi hai. | mera naam Chhotu GPT hai. |
| Adapter file | | 0.45 MB (full model 48.8 MB) |

Two honest notes. General val loss rose by 0.06: it forgot a little (catastrophic forgetting), despite replay.
And the persona leaked into ordinary chat: "kya chal raha hai bhai?" got the reply "kuchh nahi yaar, bas phone pe
scratch se banaya gaya." 600 steps on 20 examples means memorizing, not understanding. A real fine-tune needs
thousands of examples.

## What is missing (and when to add it)

- **Sliding context.** Generation stops at 256 tokens. Enough for voice replies; long conversations need a sliding window.
- **Mixed precision, gradient accumulation, multi-GPU.** Not needed at this size.
- **Instruction tuning.** The model only knows the Hinglish chat format for Q&A, not a system prompt.
