# Famous "LLM from scratch" repos ke saamne hum kahan hain

Ye table yaad se nahi, in repos ke source code ko padhkar bana hai (12 Oct 2026 ko GitHub se).
Stars us din ke hain.

## Kaunsa repo kya sikhata hai

| Repo | Stars | Ek line mein | Hamare yahan |
|---|---|---|---|
| [karpathy/micrograd](https://github.com/karpathy/micrograd) | 17.9k | Scalar autograd engine: backprop ka sabse chhota roop | [autograd.py](autograd.py), wahi idea |
| [karpathy/makemore](https://github.com/karpathy/makemore) | 4.5k | Character-level models: bigram se transformer tak | nahi (hum seedha transformer pe gaye) |
| [karpathy/minbpe](https://github.com/karpathy/minbpe) | 10.8k | BPE tokenizer, saaf code | [tokenizer.py](tokenizer.py) |
| [karpathy/minGPT](https://github.com/karpathy/minGPT) | 24.9k | GPT ka padhne layak version | [model.py](model.py) |
| [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | 63.7k | GPT-2 ko train karne ka sabse seedha code | [model.py](model.py) + [train.py](train.py) |
| [karpathy/llama2.c](https://github.com/karpathy/llama2.c) | 20.2k | TinyStories pe Llama-2, inference ek C file mein | wahi data idea (TinyStories) |
| [karpathy/llm.c](https://github.com/karpathy/llm.c) | 31.1k | GPT-2 training pure C / CUDA mein | nahi |
| [karpathy/nanochat](https://github.com/karpathy/nanochat) | 58.6k | Poora ChatGPT clone: tokenizer → pretrain → chat fine-tune → eval → UI | sabse kareeb ka lakshya |
| [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) | 106k | Kitaab ke saath GPT-2, step by step | padhne ka order isi jaisa |
| [jingyaogong/minimind](https://github.com/jingyaogong/minimind) | 63.6k | 64M ka Llama-style model 2 ghante mein | architecture lagbhag yahi |

## Architecture: kisne kya chuna

| Cheez | nanoGPT / Raschka (GPT-2, 2019) | minimind (Llama-style) | nanochat (2025-26) | Chhotu GPT (hum) |
|---|---|---|---|---|
| Position | Seekhi hui table (`wpe` / `pos_emb`) | RoPE, base 1e6 | RoPE, base 1e5 | RoPE, base 1e4 |
| Norm | LayerNorm | RMSNorm (gain ke saath) | RMSNorm (koi parameter nahi) | RMSNorm (gain ke saath) |
| Norm kahan | pre-norm | pre-norm | pre-norm, embedding ke baad bhi | pre-norm |
| Feed-forward | 4x, GELU | SwiGLU (gate / up / down) | 4x, relu² | SwiGLU |
| Weight tying | haan (nanoGPT) | haan | **nahi** | haan |
| QK norm | nahi | **haan** | **haan** | switch (`qk_norm`), default off |
| Output projections ka init | chhota normal | (dekha nahi) | **zero** | switch (`zero_init`), default off |
| Bias | optional | nahi | nahi | nahi |
| Attention | SDPA (flash) | SDPA (flash) | Flash Attention 3 | haath se likha, taaki weights dikhen |
| KV heads | barabar | GQA option | GQA option | barabar |
| Optimizer | AdamW | AdamW | Muon (matrices) + AdamW (baaki) | AdamW |
| Extra | - | MoE option, YaRN | value embeddings, sliding window, residual scalars | language control tokens |

## Is se kya seekha

1. **Hum 2019 ke GPT-2 pe nahi, aaj ke Llama-style pe hain.** RoPE, RMSNorm, SwiGLU, no bias: minimind se lagbhag
   same. nanoGPT aur Raschka jaan-boojhkar GPT-2 dikhate hain, kyunki wo original hai.
2. **Dono naye repos QK norm use karte hain, hum nahi karte the.** Ye sabse saaf gap tha. Ab switch hai, aur
   neeche measure kiya hai.
3. **nanochat ki kai cheezein bade scale ke liye hain.** Muon optimizer, Flash Attention 3, fp8, sliding window:
   ye H100 GPUs aur 2048 context ke liye hain. 12 M model aur 256 context pe inka fayda naapna mushkil hai.
4. **Hamara BPE training mein minbpe se kam ginti karta hai.** minbpe har merge pe poora text dobara ginta hai
   (seekhne ke liye saaf). Hum har alag shabd ko ek baar ginte hain aur merge pe sirf wahi shabd chhoote hain
   jinme wo pair hai. Speed ka fark naapa nahi hai; hamara poora `data.py` (download + tokenizer training
   + 11 M tokens encode) 40 second mein chala.
5. **Jo unke paas hai aur hamare paas nahi:** chat fine-tuning ka poora stage, evaluation benchmarks, web UI
   (nanochat); kitaab jaisi step-by-step notebooks (Raschka); C mein inference (llama2.c).

## Ablation: nanochat ke do ideas, hamare model pe naape hue

Ek baar mein ek cheez badlo, baaki sab same (seed, data, 500 iterations), aur val loss dekho.
Isi ko ablation study kehte hain.

| Run | Val loss @ 500 | Baseline se fark | Command |
|---|---|---|---|
| Baseline | 2.816 | | `python train.py --iters 500 --name abl1` |
| + QK norm | 2.778 | -0.038 | `... --qk-norm` |
| + zero-init projections | 2.767 | -0.049 | `... --zero-init` |
| + dono | 2.731 | -0.085 | `... --qk-norm --zero-init` |

**Kya keh sakte hain:** dono ideas alag-alag madad karte hain, aur saath mein unka fayda lagbhag jud jaata hai
(0.038 + 0.049 = 0.087, mila 0.085).

**Kya nahi keh sakte:** ye ek seed aur sirf 500 iterations hain. Fark chhota hai, aur alag seed pe kitna
hilta hai wo naapa nahi. Ye bhi nahi pata ki 4000 iterations ke baad fark bachta hai ya shuru ki tezi bhar hai.
Pakka karne ke liye: 3 seeds, aur poori lambai ka run.

Isiliye README ka main model abhi baseline hi hai. Agla kadam:

```bash
python train.py --qk-norm --zero-init --name ckpt_v2      # ~50 minute, phir val loss 1.879 se milao
```

## Aage kya uthana hai, order mein

1. `ckpt_v2` ka poora run (upar). Jeete to yahi default bane.
2. Zyada data. Abhi har source ka sirf pehla hissa liya hai (Hindi 48 MB of 5.4 GB). Val loss abhi bhi gir
   raha tha, to data aur iterations sabse sasta sudhaar hain.
3. Chat fine-tuning stage, nanochat jaisa: system / user / assistant tokens ke saath.
4. Evaluation jo loss se aage jaye: fixed prompts ka set, har checkpoint pe wahi.

