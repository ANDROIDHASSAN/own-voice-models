"""Checks that fail if a core idea is implemented wrong. No checkpoint needed, runs on CPU in seconds.

    python check.py
"""

import re
from pathlib import Path

import torch
import torch.nn.functional as F

from generate import sample
from lora import add_lora, merge_lora
from model import GPT, Config, cross_entropy
from tokenizer import BPE, SPECIALS

HERE = Path(__file__).parent
torch.manual_seed(0)
model = GPT(Config(vocab_size=300, ctx=32, n_layer=2, n_head=2, d_model=32, dropout=0.0)).eval()
x = torch.randint(0, 300, (2, 16))

# Tokenizer: any text, including text it never saw, survives encode -> decode.
tok = BPE.train("नमस्ते दुनिया hello world मी मराठी बोलतो " * 30, 256 + len(SPECIALS) + 40)
for text in ["नमस्ते bhai! कसं काय?", "<|bos|><|mr|>unseen शब्द 123\n<|eos|>", ""]:
    assert tok.decode(tok.encode(text), skip_special=False) == text, text

# Causality: changing a later token must not change the prediction at any earlier position.
with torch.no_grad():
    a, _, _ = model(x)
    x2 = x.clone()
    x2[:, 10] = (x2[:, 10] + 1) % 300
    b, _, _ = model(x2)
assert torch.allclose(a[:, :10], b[:, :10], atol=1e-5), "the future leaked into the past"
assert not torch.allclose(a[:, 10:], b[:, 10:], atol=1e-5)

# Our cross-entropy is the standard one.
t = torch.randint(0, 300, (2, 16))
assert torch.allclose(cross_entropy(a, t), F.cross_entropy(a.view(-1, 300), t.view(-1)), atol=1e-5)

# KV cache: feeding tokens one at a time gives the same logits and the same greedy text as the full pass.
with torch.no_grad():
    cache, steps = None, []
    for i in range(x.shape[1]):
        logit, _, cache = model(x[:, i : i + 1], cache=cache)
        steps.append(logit)
assert torch.allclose(torch.cat(steps, dim=1), a, atol=1e-4), "KV cache changes the output"
prompt = x[0, :5].tolist()
assert sample(model, prompt, 20, temperature=0, use_cache=True) == sample(model, prompt, 20, temperature=0, use_cache=False)

# LoRA: a fresh adapter changes nothing (B = 0); a trained one, merged, equals the unmerged one.
params = add_lora(model, r=4)
with torch.no_grad():
    assert torch.allclose(model(x)[0], a, atol=1e-6), "a new LoRA adapter must be a no-op"
    for p in params:
        p.add_(torch.randn_like(p) * 0.1)
    with_adapter = model(x)[0]
    merge_lora(model)
    assert torch.allclose(model(x)[0], with_adapter, atol=1e-4), "merging LoRA changed the output"
assert not torch.allclose(with_adapter, a, atol=1e-3)

# Glossary: every `file.py: name` it points to exists, so the glossary cannot drift from the code.
refs = re.findall(r"`(\w+\.py): ([^`]+)`", (HERE / "GLOSSARY.md").read_text())
missing = [(f, name) for f, name in refs if name not in (HERE / f).read_text()]
assert not missing, f"glossary points at code that is not there: {missing}"
terms = len(re.findall(r"^\| \d+ \|", (HERE / "GLOSSARY.md").read_text(), flags=re.M))
assert terms >= 100, terms

print(f"all checks passed ({terms} glossary terms, {len(refs)} code references)")
