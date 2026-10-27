"""Builds the study guide: docs/Chhotu-GPT-Andar-Se.html and .pdf.

    ../../../.venv/bin/python build.py

Every chart is drawn from this project's own files: runs/train.log, runs/shrink.log, runs/abl*.log,
the trained checkpoint, the tokenizer and GLOSSARY.md. Nothing in a chart is typed in by hand.
"""

import html
import re
import subprocess
import sys
import time
from pathlib import Path

import torch

DOCS = Path(__file__).parent
LLM = DOCS.parent
sys.path.insert(0, str(LLM))

from data import documents  # noqa: E402
from generate import sample  # noqa: E402
from model import load  # noqa: E402
from tokenizer import BPE  # noqa: E402

INK, SEC, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # validated categorical slots 1-3 (dataviz palette)
KINDS = {  # box styles for diagrams: (fill, stroke, text)
    "plain": ("#f4f6fa", "#d5dbe6", INK), "blue": (BLUE, BLUE, "#ffffff"), "soft": ("#e6f0fc", "#9ec5f4", INK),
    "orange": ("#fdeee7", ORANGE, INK), "aqua": ("#e3f6ee", AQUA, INK), "ghost": ("#ffffff", "#c3c2b7", SEC),
}
esc = html.escape


# ---------- tiny SVG toolkit ----------

def svg(w: int, h: int, body: str, label: str) -> str:
    marker = ('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
              f'orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="{MUTED}"/></marker></defs>')
    return f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{esc(label)}" xmlns="http://www.w3.org/2000/svg">{marker}{body}</svg>'


def text(x, y, s, size=13, fill=INK, anchor="middle", weight=400, extra="") -> str:
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}" {extra}>{esc(str(s))}</text>'


def box(x, y, w, h, label, sub=None, kind="plain", size=13) -> str:
    fill, stroke, ink = KINDS[kind]
    out = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>'
    cy = y + h / 2 + size * 0.35 - (8 if sub else 0)
    out += text(x + w / 2, cy, label, size, ink, weight=600)
    if sub:
        out += text(x + w / 2, cy + 16, sub, 10.5, "#ffffff" if kind == "blue" else SEC)
    return out


def arrow(x1, y1, x2, y2, dashed=False) -> str:
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{MUTED}" stroke-width="1.6" marker-end="url(#ah)"{dash}/>'


def curve(d, dashed=False) -> str:
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    return f'<path d="{d}" fill="none" stroke="{MUTED}" stroke-width="1.6" marker-end="url(#ah)"{dash}/>'


def line_chart(series, xmax, ymin, ymax, xticks, yticks, xlabel, ylabel, w=720, h=300, yfmt="{:g}") -> str:
    """series: (name, color, [(x, y)], markers). One y-axis. Direct label at the end of each line."""
    L, R, T, B = 54, 118, 14, 42
    X = lambda v: L + (w - L - R) * v / xmax
    Y = lambda v: T + (h - T - B) * (1 - (v - ymin) / (ymax - ymin))
    out = f'<clipPath id="plot{id(series)}"><rect x="{L}" y="{T}" width="{w - L - R}" height="{h - T - B}"/></clipPath>'
    for t in yticks:
        out += f'<line x1="{L}" y1="{Y(t)}" x2="{w - R}" y2="{Y(t)}" stroke="{GRID}" stroke-width="1"/>'
        out += text(L - 8, Y(t) + 4, yfmt.format(t), 11, MUTED, "end")
    for t in xticks:
        out += text(X(t), h - B + 17, f"{t:g}", 11, MUTED)
    out += f'<line x1="{L}" y1="{h - B}" x2="{w - R}" y2="{h - B}" stroke="{AXIS}" stroke-width="1"/>'
    out += text((L + w - R) / 2, h - 6, xlabel, 11.5, SEC)
    out += text(14, (T + h - B) / 2, ylabel, 11.5, SEC, extra=f'transform="rotate(-90 14 {(T + h - B) / 2})"')
    for name, color, pts, markers in series:
        path = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in pts)
        out += f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" clip-path="url(#plot{id(series)})"/>'
        if markers:
            for x, y in pts:
                if ymin <= y <= ymax:
                    out += f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4.5" fill="{color}" stroke="#fff" stroke-width="2"/>'
    # direct labels at the line ends, pushed apart so two close lines do not print on top of each other
    ends = sorted((Y(pts[-1][1]), X(pts[-1][0]), name, color) for name, color, pts, _ in series)
    for i in range(1, len(ends)):
        if ends[i][0] - ends[i - 1][0] < 15:
            ends[i] = (ends[i - 1][0] + 15, *ends[i][1:])
    for ly, lx, name, color in ends:
        if name:  # an empty name means the caller draws a legend instead
            out +=f'<circle cx="{lx + 12}" cy="{ly}" r="4" fill="{color}"/>' + text(lx + 20, ly + 4, name, 11.5, SEC, "start")
    return svg(w, h, out, f"{ylabel} vs {xlabel}")


def hbars(rows, color=BLUE, fmt="{:.1f}", maxv=None, unit="") -> str:
    """rows: (label, value) or (label, value, color). Horizontal bars with the value written at the end."""
    maxv = maxv or max(r[1] for r in rows)
    out = '<div class="hbars">'
    for r in rows:
        c = r[2] if len(r) > 2 else color
        width = max(0.6, 100 * r[1] / maxv)
        out += (f'<div class="hb"><div class="hb-l">{esc(str(r[0]))}</div><div class="hb-t"><div class="hb-f" '
                f'style="width:{width:.1f}%;background:{c}"></div></div><div class="hb-v">{fmt.format(r[1])}{unit}</div></div>')
    return out + "</div>"


def legend(items) -> str:
    return '<div class="legend">' + "".join(f'<span><i style="background:{c}"></i>{esc(n)}</span>' for n, c in items) + "</div>"


def chips(tok: BPE, ids: list[int]) -> str:
    return '<div class="chips">' + "".join(
        f'<span class="chip c{i % 2}">{esc(tok.piece(t)).replace(" ", "&#9141;")}</span>' for i, t in enumerate(ids)) + "</div>"


# ---------- diagrams (concepts, drawn by hand) ----------

def d_pipeline() -> str:
    steps = [("Text", '"एक दिन"', "ghost"), ("Tokenizer", "BPE", "soft"), ("Token ids", "4 numbers", "ghost"),
             ("Embedding", "4 × 384", "soft"), ("6 Blocks", "attention + FF", "blue"), ("Logits", "4 × 4096", "soft"),
             ("Sampling", "ek token chuno", "orange")]
    w, gap, x, out = 88, 16, 4, ""
    for i, (a, b, kind) in enumerate(steps):
        out += box(x, 30, w, 62, a, b, kind, 12.5)
        if i < len(steps) - 1:
            out += arrow(x + w + 1, 61, x + w + gap - 2, 61)
        x += w + gap
    out += curve(f"M {x - gap - w / 2} 94 C {x - gap - w / 2} 140, 240 140, 240 96", dashed=True)
    out += text(430, 152, "naya token wapas input mein judta hai, phir dobara (autoregressive loop)", 11.5, SEC)
    return svg(728, 162, out, "Text se agle token tak ka flow")


def d_graph() -> str:
    out = box(20, 20, 70, 40, "a = 2", kind="ghost") + box(20, 90, 70, 40, "b = 3", kind="ghost")
    out += box(170, 55, 80, 40, "a × b", "= 6", "soft") + box(20, 160, 70, 40, "c = 1", kind="ghost")
    out += box(330, 100, 80, 40, "+ c", "= 7", "soft") + box(490, 100, 90, 40, "loss", "= 7", "blue")
    out += arrow(90, 40, 168, 68) + arrow(90, 110, 168, 84) + arrow(250, 80, 328, 112) + arrow(90, 180, 328, 130) + arrow(410, 120, 488, 120)
    out += curve("M 488 150 C 420 190, 300 190, 250 100", dashed=True) + curve("M 168 100 C 130 130, 110 125, 92 118", dashed=True)
    out += text(385, 196, "backward: gradient ulti disha mein behta hai (chain rule)", 11.5, SEC)
    out += text(610, 60, "forward →", 11.5, SEC, "start") + text(610, 150, "← backward", 11.5, SEC, "start")
    return svg(720, 210, out, "Computation graph: forward aur backward pass")


def d_loop() -> str:
    steps = [("1. Batch", "48 × 256 tokens", "ghost"), ("2. Forward", "logits nikalo", "soft"), ("3. Loss", "kitna galat?", "orange"),
             ("4. Backward", "har weight ka gradient", "soft"), ("5. Update", "AdamW step", "blue")]
    w, gap, x, out = 124, 22, 6, ""
    for i, (a, b, kind) in enumerate(steps):
        out += box(x, 24, w, 58, a, b, kind, 12.5)
        if i < len(steps) - 1:
            out += arrow(x + w + 1, 53, x + w + gap - 2, 53)
        x += w + gap
    out += curve(f"M {x - gap - w / 2} 84 C {x - gap - w / 2} 132, 68 132, 68 86", dashed=True)
    out += text(364, 142, "4000 baar dohrao. Har chakkar mein weights thode behtar.", 11.5, SEC)
    return svg(728, 152, out, "Training loop ke paanch step")


def d_arch() -> str:
    out = text(120, 16, "Poora model", 12.5, SEC, weight=600) + text(520, 16, "Ek block ke andar (zoom)", 12.5, SEC, weight=600)
    stack = [("Token ids", "(T)", "ghost"), ("Embedding", "4096 × 384, 1.57 M", "soft"), ("Block × 6", "har ek 1.77 M", "blue"),
             ("RMSNorm", "", "plain"), ("LM head", "embedding wali hi matrix", "soft"), ("Logits", "(T, 4096)", "ghost")]
    y = 28
    for i, (a, b, kind) in enumerate(stack):
        out += box(30, y, 180, 46, a, b or None, kind, 12.5)
        if i < len(stack) - 1:
            out += arrow(120, y + 47, 120, y + 60)
        y += 62
    out += curve("M 212 175 C 290 175, 290 60, 362 60", dashed=True)
    out += f'<rect x="364" y="28" width="340" height="364" rx="12" fill="#fbfcfe" stroke="{GRID}" stroke-width="1.3"/>'
    out += box(470, 44, 120, 34, "x (andar aaya)", kind="ghost", size=12)
    out += box(470, 98, 120, 34, "RMSNorm", kind="plain", size=12)
    out += box(450, 150, 160, 44, "Self-attention", "tokens aapas mein baat", "blue", 12.5)
    out += f'<circle cx="530" cy="222" r="13" fill="#fff" stroke="{MUTED}" stroke-width="1.4"/>' + text(530, 227, "+", 16, SEC, weight=600)
    out += box(470, 252, 120, 34, "RMSNorm", kind="plain", size=12)
    out += box(450, 304, 160, 44, "Feed-forward", "SwiGLU, har token akela", "aqua", 12.5)
    out += f'<circle cx="530" cy="372" r="13" fill="#fff" stroke="{MUTED}" stroke-width="1.4"/>' + text(530, 377, "+", 16, SEC, weight=600)
    for y1, y2 in ((79, 96), (133, 148), (195, 208), (236, 250), (287, 302), (349, 358)):
        out += arrow(530, y1, 530, y2)
    out += curve("M 470 61 C 400 61, 400 222, 515 222") + curve("M 517 228 C 400 240, 400 372, 515 372")
    out += text(392, 146, "skip", 11, SEC, extra='transform="rotate(-90 392 146)"') + text(392, 306, "skip", 11, SEC, extra='transform="rotate(-90 392 306)"')
    out += text(650, 226, "x = x +", 10.5, SEC) + text(650, 240, "attn(norm(x))", 10.5, SEC)
    out += text(650, 376, "x = x +", 10.5, SEC) + text(650, 390, "ff(norm(x))", 10.5, SEC)
    return svg(720, 404, out, "Transformer ka architecture aur ek block ka zoom")


def d_attention() -> str:
    out = box(10, 76, 96, 48, "Token ka", "vector (384)", "ghost", 12.5)
    out += box(134, 10, 136, 44, "Query", "main kya dhoondh raha?", "soft", 12.5)
    out += box(134, 78, 136, 44, "Key", "mere paas kya hai?", "soft", 12.5)
    out += box(134, 146, 136, 44, "Value", "meri asli jaankari", "aqua", 12.5)
    out += arrow(107, 92, 132, 36) + arrow(107, 100, 132, 100) + arrow(107, 108, 132, 164)
    out += box(326, 40, 112, 50, "Q · K", "÷ √head_dim", "plain", 13)
    out += arrow(271, 32, 324, 56) + arrow(271, 100, 324, 78)
    out += box(470, 40, 100, 50, "Mask", "future = −∞", "orange", 13) + arrow(439, 65, 468, 65)
    out += box(602, 40, 106, 50, "Softmax", "weights, jod = 1", "plain", 13) + arrow(571, 65, 600, 65)
    out += box(540, 142, 168, 50, "weights × Values", "nayi jaankari is token ke liye", "blue", 12.5)
    out += arrow(655, 91, 640, 140) + arrow(271, 168, 538, 168)
    out += text(360, 214, "Ye sab 6 heads mein ek saath hota hai, phir sab jud kar wapas ek vector banta hai.", 11.5, SEC)
    return svg(720, 224, out, "Attention ke andar: query, key, value, mask, softmax")


def d_mask() -> str:
    toks = ["एक", "दिन", "राजू", "ने", "कहा"]
    c, x0, y0, out = 44, 96, 40, ""
    for j, t in enumerate(toks):
        out += text(x0 + j * c + c / 2, 30, t, 13, SEC)
        out += text(x0 - 10, y0 + j * c + c / 2 + 5, t, 13, SEC, "end")
    for i in range(5):
        for j in range(5):
            ok = j <= i
            out += (f'<rect x="{x0 + j * c + 1}" y="{y0 + i * c + 1}" width="{c - 2}" height="{c - 2}" rx="5" '
                    f'fill="{"#cde2fb" if ok else "#f4f3ef"}"/>')
            out += text(x0 + j * c + c / 2, y0 + i * c + c / 2 + 5, "✓" if ok else "×", 15, BLUE if ok else AXIS, weight=600)
    out += text(350, 80, "Row = jo token soch raha hai", 12, SEC, "start")
    out += text(350, 102, "Column = jise wo dekh sakta hai", 12, SEC, "start")
    out += text(350, 140, '"राजू" sirf "एक दिन राजू" dekhta hai.', 12, INK, "start")
    out += text(350, 162, '"ने" aur "कहा" uske liye future hain.', 12, INK, "start")
    out += text(350, 200, "Isi wajah se ek hi pass mein har position", 12, SEC, "start")
    out += text(350, 220, "pe training ho jaati hai, bina cheating ke.", 12, SEC, "start")
    return svg(720, 270, out, "Causal mask ka grid")


def d_rope() -> str:
    import math
    out = text(360, 16, "Har token ka vector uski position ke hisaab se ghoomta hai", 12, SEC)
    for row, (name, speed) in enumerate((("tez ghoomne wala pair", 0.9), ("dheema pair", 0.22))):
        y = 62 + row * 84
        out += text(8, y + 4, name, 11.5, SEC, "start")
        for p in range(6):
            cx = 190 + p * 92
            ang = p * speed - math.pi / 2
            out += f'<circle cx="{cx}" cy="{y}" r="26" fill="#f4f6fa" stroke="#d5dbe6" stroke-width="1.3"/>'
            out += f'<line x1="{cx}" y1="{y}" x2="{cx + 22 * math.cos(ang):.1f}" y2="{y + 22 * math.sin(ang):.1f}" stroke="{BLUE}" stroke-width="2.5" stroke-linecap="round"/>'
            out += f'<circle cx="{cx}" cy="{y}" r="3" fill="{BLUE}"/>'
            if row == 1:
                out += text(cx, y + 46, f"position {p}", 11, MUTED)
    return svg(720, 200, out, "RoPE: position ke saath ghoomte vector")


def d_lora() -> str:
    out = box(20, 70, 80, 44, "x", "input", "ghost")
    out += box(210, 20, 230, 60, "W  (frozen)", "384 × 384 = 147,456 numbers", "plain", 13.5)
    out += box(170, 124, 110, 54, "A", "8 × 384", "orange", 13.5) + box(370, 124, 110, 54, "B", "384 × 8, zero se shuru", "orange", 13.5)
    out += arrow(101, 84, 208, 54) + arrow(101, 100, 168, 146) + arrow(281, 151, 368, 151)
    out += f'<circle cx="560" cy="92" r="15" fill="#fff" stroke="{MUTED}" stroke-width="1.4"/>' + text(560, 98, "+", 18, SEC, weight=600)
    out += arrow(441, 52, 545, 84) + arrow(481, 146, 547, 102) + arrow(576, 92, 628, 92) + box(630, 70, 76, 44, "y", "output", "ghost")
    out += text(325, 200, "Sirf A aur B train hote hain: 6,144 numbers, W ka 4%. Baad mein W + B·A jodkar ek matrix.", 11.5, SEC)
    return svg(720, 212, out, "LoRA: frozen W ke saath chhota B A")


# ---------- charts (this project's measured data) ----------

def c_loss() -> tuple[str, str]:
    log = (LLM / "runs/train.log").read_text()
    batch = [(int(a), float(b)) for a, b in re.findall(r"^iter\s+(\d+)\s+loss ([\d.]+)", log, flags=re.M)]
    lr = [(int(a), float(b) * 1e4) for a, b in re.findall(r"^iter\s+(\d+).*?lr ([\d.e+-]+)", log, flags=re.M)]
    val = [(int(a), float(c)) for a, _, c in re.findall(r"== iter (\d+)\s+train ([\d.]+)\s+val ([\d.]+)", log)]
    assert len(batch) > 50 and len(val) == 8, (len(batch), len(val))
    loss = line_chart([("train (har batch)", BLUE, batch, False), ("val (andekha text)", ORANGE, val, True)], 4000, 1.5, 3.5,
                      [0, 1000, 2000, 3000, 4000], [1.5, 2.0, 2.5, 3.0, 3.5], "iteration", "loss", yfmt="{:.1f}")
    rate = line_chart([("learning rate", BLUE, lr, False)], 4000, 0, 6.5, [0, 1000, 2000, 3000, 4000], [0, 2, 4, 6],
                      "iteration", "learning rate (× 0.0001)", h=210)
    return loss, rate


def c_quant() -> tuple[str, str]:
    rows = re.findall(r"^(\S+(?: \S+)*?)\s+([\d.]+) MB\s+val loss ([\d.]+)", (LLM / "runs/shrink.log").read_text(), flags=re.M)
    table = {name: (float(mb), float(loss)) for name, mb, loss in rows}
    base = table["fp32 (trained)"][1]
    bars = []
    for bits in (8, 4, 3):
        bars.append((f"int{bits} · har row ka scale", table[f"int{bits} per-channel"][1] - base, BLUE))
        bars.append((f"int{bits} · ek hi scale", table[f"int{bits} per-tensor"][1] - base, ORANGE))
    damage = legend([("per-channel (har row ka apna scale)", BLUE), ("per-tensor (poori matrix ka ek scale)", ORANGE)]) + hbars(bars, fmt="+{:.3f}")
    size = hbars([("fp32", table["fp32 (trained)"][0]), ("fp16", table["fp16"][0]), ("int8", table["int8 per-channel"][0]),
                  ("int4", table["int4 per-channel"][0])], fmt="{:.1f}", unit=" MB")
    return damage, size


def c_ablation() -> str:
    names = ["Baseline", "+ QK norm", "+ zero-init", "+ dono"]
    vals = [float(re.search(r"val ([\d.]+)", (LLM / f"runs/abl{i}.log").read_text().split("== iter")[1]).group(1)) for i in (1, 2, 3, 4)]
    low = 2.6  # bars start at 2.6 so the small gap is visible; the axis note in the guide says so
    return hbars([(f"{n}  ({v:.3f})", v - low) for n, v in zip(names, vals)], fmt="", maxv=max(vals) - low + 0.02)


def model_figures(tok: BPE) -> dict:
    model, _ = load(LLM / "runs/ckpt.pt", "cpu")
    out = {}

    # parameter budget, counted from the real model
    groups = {"Feed-forward (6 blocks)": 0, "Attention (6 blocks)": 0, "Embedding (= LM head)": 0, "Norm gains": 0}
    for name, p in model.named_parameters():
        key = ("Embedding (= LM head)" if "tok_emb" in name else "Attention (6 blocks)" if ".attn." in name
               else "Feed-forward (6 blocks)" if ".mlp." in name else "Norm gains")
        groups[key] += p.numel()
    assert sum(groups.values()) == model.n_params()
    out["PARAMS"] = hbars([(k, v / 1e6) for k, v in groups.items()], fmt="{:.3f}", unit=" M")
    out["N_PARAMS"] = f"{model.n_params():,}"

    # what the model believes comes next, for two prompts
    @torch.no_grad()
    def beliefs(prompt: str, lang: str):
        ids = tok.encode(f"<|bos|><|{lang}|>{prompt}")
        return ids, model(torch.tensor([ids]))[0][0, -1]

    for key, prompt, lang in (("NEXT_HI", "एक दिन राजू ने अपनी माँ से कहा", "hi"), ("NEXT_HING", "aaj office mein", "hing")):
        _, logits = beliefs(prompt, lang)
        p, i = torch.softmax(logits, -1).topk(6)
        out[key] = hbars([(tok.piece(int(t)).strip() or "␣", float(v) * 100) for v, t in zip(p, i)], fmt="{:.1f}", unit="%", maxv=100)

    # temperature, on the real logits of the Hinglish prompt
    _, logits = beliefs("aaj office mein", "hing")
    top = torch.softmax(logits, -1).topk(6).indices
    cols = ""
    for temp, note in ((0.5, "safe, ek hi jawab"), (1.0, "model ka asli bharosa"), (1.5, "creative, risky")):
        probs = torch.softmax(logits / temp, -1)[top]
        cols += (f'<div><div class="mini-h">temperature {temp}</div><div class="mini-s">{note}</div>'
                 + hbars([(tok.piece(int(t)).strip(), float(v) * 100) for v, t in zip(probs, top)], fmt="{:.0f}", unit="%", maxv=100) + "</div>")
    out["TEMP"] = f'<div class="cols3">{cols}</div>'

    # a real attention map: layer 3, averaged over heads
    ids, _ = beliefs("एक दिन राजू ने अपनी माँ से कहा", "hi")
    for b in model.blocks:
        b.attn.record = True
    with torch.no_grad():
        model(torch.tensor([ids]))
    att = model.blocks[3].attn.last_att[0].mean(0)
    labels = [tok.piece(t).strip() for t in ids]
    n, c, x0, y0, body = len(ids), 40, 86, 74, ""
    for j, t in enumerate(labels):
        body += text(x0 + j * c + c / 2, y0 - 8, t, 11.5, SEC, "start", extra=f'transform="rotate(-40 {x0 + j * c + c / 2} {y0 - 8})"')
        body += text(x0 - 8, y0 + j * c + c / 2 + 4, t, 11.5, SEC, "end")
    for i in range(n):
        for j in range(n):
            if j > i:
                body += f'<rect x="{x0 + j * c + 1}" y="{y0 + i * c + 1}" width="{c - 2}" height="{c - 2}" rx="4" fill="#f6f5f1"/>'
                continue
            v = float(att[i, j])
            shade = [round(a + (b - a) * min(1, v * 1.6)) for a, b in zip((238, 244, 252), (24, 79, 149))]  # one hue, light to dark
            body += f'<rect x="{x0 + j * c + 1}" y="{y0 + i * c + 1}" width="{c - 2}" height="{c - 2}" rx="4" fill="rgb({shade[0]},{shade[1]},{shade[2]})"/>'
            if v >= 0.12:
                body += text(x0 + j * c + c / 2, y0 + i * c + c / 2 + 4, f"{v * 100:.0f}", 10.5, "#fff" if v > 0.3 else INK)
    body += text(x0 + n * c + 16, y0 + 20, "Number = kitna % dhyaan", 11.5, SEC, "start")
    body += text(x0 + n * c + 16, y0 + 40, "Gehra neela = zyada", 11.5, SEC, "start")
    body += text(x0 + n * c + 16, y0 + 60, "Khaali kona = mask", 11.5, SEC, "start")
    body += text(x0 + n * c + 16, y0 + 80, "(future chhupa hua)", 11.5, SEC, "start")
    out["HEAT"] = svg(720, y0 + n * c + 10, body, "Layer 3 ka asli attention map")
    out["HEAT_LAST"] = ", ".join(f"{labels[int(i)]} {float(w) * 100:.0f}%" for w, i in zip(*att[-1].topk(3)))
    for b in model.blocks:
        b.attn.record = False

    # KV cache, timed on this machine (CPU, greedy, same output both ways)
    prompt = tok.encode("<|bos|><|hi|>एक दिन")
    timings = {}
    for use in (True, False):
        t = time.perf_counter()
        new = sample(model, prompt, 150, temperature=0, repetition_penalty=1.0, use_cache=use)
        timings[use] = (time.perf_counter() - t, len(new))
    assert timings[True][1] == timings[False][1]
    on, off = (timings[u][1] / timings[u][0] for u in (True, False))
    out |= {"KV_X": f"{on / off:.1f}", "KV_N": str(timings[True][1])}
    out["KV_BARS"] = hbars([("KV cache ke saath", on, BLUE), ("bina cache", off, ORANGE)], fmt="{:.0f}", unit=" tokens/s")
    return out


def data_figures(tok: BPE) -> dict:
    docs = documents()
    names = {"<|hi|>": "Hindi kahaniyan", "<|mr|>": "Marathi kahaniyan", "<|hing|>": "Hinglish chat"}
    count, chars, ratio = {}, {}, {}
    for tag, name in names.items():
        mine = [t for g, t in docs if g == tag]
        count[name], chars[name] = len(mine), sum(map(len, mine)) / 1e6
        some = mine[:300]
        ratio[name] = sum(len(t) for t in some) / sum(len(tok.encode(t)) for t in some)
    order = list(names.values())
    examples = ""
    for lang, s in (("Hindi", "एक दिन राजू ने अपनी माँ से कहा"), ("Marathi", "आर्यनला समुद्रकिनाऱ्यावर खेळायला आवडत असे"),
                    ("Hinglish", "kya chal raha hai bhai, weekend ka plan?")):
        ids = tok.encode(s)
        examples += f'<div class="tokrow"><div class="tokmeta"><b>{lang}</b><span>{len(s.encode())} bytes → {len(ids)} tokens</span></div>{chips(tok, ids)}</div>'
    return {
        "DATA_DOCS": hbars([(n, count[n]) for n in order], fmt="{:,.0f}"),
        "DATA_CHARS": hbars([(n, chars[n]) for n in order], fmt="{:.1f}", unit=" M"),
        "COMPRESS": hbars([(n, ratio[n]) for n in order], fmt="{:.2f}", unit=" akshar / token"),
        "TOKCHIPS": examples,
    }


def table(head: list[str], rows: list[list], right: tuple = ()) -> str:
    th = "".join(f'<th class="{"r" if i in right else ""}">{h}</th>' for i, h in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td class="{"r" if i in right else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def d_machine() -> str:
    import math
    out = box(10, 62, 120, 50, "Input", '"2 + 3 = ?"', "ghost")
    out += f'<rect x="200" y="20" width="320" height="134" rx="14" fill="#f4f6fa" stroke="#d5dbe6" stroke-width="1.4"/>'
    out += text(360, 44, "Model = knobs wali machine", 13, INK, weight=600)
    for i, turn in enumerate((0.4, 2.1, -0.9, 1.3, 2.8)):
        cx = 240 + i * 60
        out += f'<circle cx="{cx}" cy="96" r="21" fill="#fff" stroke="{BLUE}" stroke-width="2"/>'
        out += f'<line x1="{cx}" y1="96" x2="{cx + 15 * math.sin(turn):.1f}" y2="{96 - 15 * math.cos(turn):.1f}" stroke="{BLUE}" stroke-width="3" stroke-linecap="round"/>'
    out += text(360, 142, "har knob = ek parameter (weight). Hamare model mein 1.2 crore.", 11, SEC)
    out += box(590, 62, 120, 50, "Output", '"5"', "soft") + arrow(131, 87, 198, 87) + arrow(521, 87, 588, 87)
    return svg(720, 170, out, "Model ek knobs wali machine hai")


def d_embed() -> str:
    """An illustration, not measured data: where similar words would sit if vectors had only 2 numbers."""
    groups = [(BLUE, [("kutta", 110, 70), ("billi", 160, 95), ("sher", 95, 120)], "jaanwar"),
              (ORANGE, [("roti", 420, 60), ("chawal", 470, 90), ("daal", 405, 105)], "khaana"),
              (AQUA, [("daudna", 250, 190), ("chalna", 305, 205), ("koodna", 240, 225)], "kaam")]
    out = f'<line x1="40" y1="250" x2="560" y2="250" stroke="{AXIS}"/><line x1="40" y1="20" x2="40" y2="250" stroke="{AXIS}"/>'
    out += text(300, 268, "vector ka pehla number", 11, MUTED) + text(22, 135, "doosra number", 11, MUTED, extra='transform="rotate(-90 22 135)"')
    for color, words, name in groups:
        for w, x, y in words:
            out += f'<circle cx="{x}" cy="{y}" r="6" fill="{color}" stroke="#fff" stroke-width="2"/>' + text(x + 11, y + 4, w, 12.5, INK, "start")
        out += text(words[0][1] + 20, words[0][2] - 22, name, 11, SEC, weight=600)
    out += text(590, 110, "Milte-julte shabd", 12, INK, "start") + text(590, 128, "paas-paas baithte hain.", 12, INK, "start")
    out += text(590, 160, "Kisi ne ye sikhaya nahi;", 11.5, SEC, "start") + text(590, 177, "training mein khud hua.", 11.5, SEC, "start")
    return svg(760, 280, out, "Embedding ka naksha: milte-julte shabd paas")


def zero_figures(tok: BPE) -> dict:
    """Worked examples for the beginner guide. Every number is computed here, none typed by hand."""
    import math

    from model import Config
    from tokenizer import SPECIALS

    out = {"MACHINE": d_machine(), "EMBED": d_embed()}

    # gradient descent on the simplest loss there is: (w - 3)^2
    w, lr, steps = 0.0, 0.25, []
    for i in range(6):
        loss, grad = (w - 3) ** 2, 2 * (w - 3)
        steps.append((i, w, loss, grad))
        w -= lr * grad
    out["GD_TABLE"] = table(["Kadam", "Knob (w)", "Galti (loss)", "Dhalaan (gradient)", "Kya karna hai"],
                            [[i, f"{w:.2f}", f"{l:.2f}", f"{g:+.2f}", "knob badhao" if g < 0 else "knob ghatao"] for i, w, l, g in steps], right=(1, 2, 3))
    valley = [(x / 10, (x / 10 - 3) ** 2) for x in range(0, 61)]
    out["GD_CHART"] = legend([("galti ki ghaati", BLUE), ("hamare kadam", ORANGE)]) + line_chart(
        [("", BLUE, valley, False), ("", ORANGE, [(w, l) for _, w, l, _ in steps], True)],
        6, 0, 9.5, [0, 1, 2, 3, 4, 5, 6], [0, 3, 6, 9], "knob ki value (w)", "galti (loss)", h=200)

    # softmax by hand
    names, scores = ["roti", "chai", "kursi"], [2.0, 1.0, 0.1]
    exps = [math.exp(s) for s in scores]
    out["SOFTMAX"] = table(["Shabd", "Score (logit)", "e^score", "÷ jod (" + f"{sum(exps):.2f}" + ")", "Probability"],
                           [[n, f"{s:.1f}", f"{e:.2f}", f"{e:.2f} ÷ {sum(exps):.2f}", f"<b>{100 * e / sum(exps):.0f}%</b>"] for n, s, e in zip(names, scores, exps)], right=(1, 2, 4))

    # cross-entropy by hand
    out["CE"] = table(["Sahi shabd ko model ne diya", "Loss = −ln(p)", "Matlab"],
                      [[f"{p:.0%}" if p >= 0.01 else f"{100 * p:.2f}%", f"<b>{-math.log(p):.2f}</b>", m] for p, m in
                       ((0.9, "bahut achha"), (0.5, "theek-thaak"), (0.1, "kharab"), (0.01, "bahut kharab"), (1 / 4096, "bina seekha model (1/4096)"))], right=(0, 1))

    # attention by hand: "peeti" looks back over three words. The vectors are made up; the arithmetic is real.
    query = (1.0, 2.0)
    words = [("billi", (2.0, 1.0), (1.0, 0.0)), ("doodh", (1.0, 3.0), (0.0, 1.0)), ("peeti", (0.5, 0.5), (0.5, 0.5))]
    raw = [(query[0] * k[0] + query[1] * k[1]) / math.sqrt(2) for _, k, _ in words]
    wts = [math.exp(r) / sum(math.exp(x) for x in raw) for r in raw]
    mix = [sum(wt * v[i] for wt, (_, _, v) in zip(wts, words)) for i in (0, 1)]
    out["ATTN_TABLE"] = table(["Shabd", "Key", "Query · Key", "÷ √2", "Softmax = dhyaan", "Value"],
                              [[n, f"({k[0]:g}, {k[1]:g})", f"{query[0] * k[0] + query[1] * k[1]:g}", f"{r:.2f}", f"<b>{100 * wt:.0f}%</b>", f"({v[0]:g}, {v[1]:g})"]
                               for (n, k, v), r, wt in zip(words, raw, wts)], right=(2, 3, 4))
    out["ATTN_MIX"] = f"({mix[0]:.2f}, {mix[1]:.2f})"
    out["ATTN_TOP"] = f"{100 * max(wts):.0f}%"

    # quantization by hand
    weights = [0.31, -0.07, 0.52, -0.44]
    scale = max(abs(x) for x in weights) / 127
    out["QUANT_TABLE"] = table(["Asli weight (float)", "÷ scale, gol karo (int8)", "Wapas × scale", "Galti"],
                               [[f"{x:+.2f}", f"<b>{round(x / scale):+d}</b>", f"{round(x / scale) * scale:+.4f}", f"{abs(round(x / scale) * scale - x):.4f}"] for x in weights], right=(0, 1, 2, 3))
    out["QUANT_SCALE"] = f"{scale:.5f}"

    # BPE by hand: train our real tokenizer on six tiny words and show what it merged, in order
    tiny = BPE.train("aam aam aam kaam kaam naam", 256 + len(SPECIALS) + 4)
    show = lambda i: "⎵" if tiny.piece(i) == " " else tiny.piece(i).replace(" ", "⎵")
    out["BPE_STEPS"] = table(["Merge", "Kaunsa joda sabse zyada aaya", "Naya token"],
                             [[i + 1, f'<span class="chip c0">{esc(show(a))}</span> + <span class="chip c1">{esc(show(b))}</span>', f'<span class="chip c0">{esc(show(256 + i))}</span>']
                              for i, (a, b) in enumerate(tiny.merges)])
    ids = tiny.encode("aam kaam naam")
    out["BPE_RESULT"] = " ".join(f'<span class="chip c{i % 2}">{esc(show(t))}</span>' for i, t in enumerate(ids)) + f" &nbsp; ({len(ids)} tokens, pehle 13 akshar the)"

    # counting the parameters of one block
    c = Config()
    hidden = int(4 * c.d_model * 2 / 3)
    parts = [("Attention: Q, K, V banane wali matrix", f"{c.d_model} × {3 * c.d_model}", c.d_model * 3 * c.d_model),
             ("Attention: heads ko wapas jodne wali matrix", f"{c.d_model} × {c.d_model}", c.d_model * c.d_model),
             ("Feed-forward: teen matrix", f"3 × {c.d_model} × {hidden}", 3 * c.d_model * hidden),
             ("Do norm ke gains", f"2 × {c.d_model}", 2 * c.d_model)]
    block = sum(p[2] for p in parts)
    total = c.n_layer * block + c.vocab_size * c.d_model + c.d_model
    rows = [[a, b, f"{n:,}"] for a, b, n in parts] + [["<b>Ek block</b>", "", f"<b>{block:,}</b>"], [f"× {c.n_layer} blocks", "", f"{c.n_layer * block:,}"],
                                                       ["Embedding table (LM head bhi yahi)", f"{c.vocab_size} × {c.d_model}", f"{c.vocab_size * c.d_model:,}"],
                                                       ["Aakhri norm", f"{c.d_model}", f"{c.d_model:,}"], ["<b>Poora model</b>", "", f"<b>{total:,}</b>"]]
    out["PARAM_WALK"] = table(["Hissa", "Size", "Parameters"], rows, right=(2,))
    out["LORA_SAVE"] = f"{2 * 8 * c.d_model:,}"
    return out


def glossary() -> tuple[str, int, int]:
    out, n, coded = "", 0, 0
    for block in (LLM / "GLOSSARY.md").read_text().split("\n## ")[1:]:
        title, *lines = block.split("\n")
        rows = ""
        for num, term, meaning, where in re.findall(r"^\| (\d+) \| (.+?) \| (.+?) \| (.+?) \|$", "\n".join(lines), flags=re.M):
            n += 1
            ref = re.fullmatch(r"`(\w+\.py): (.+)`", where)
            coded += bool(ref)
            cell = f'<span class="where"><b>{esc(ref.group(1))}</b> {esc(ref.group(2))}</span>' if ref else '<span class="theory">theory</span>'
            rows += f'<tr><td class="num">{num}</td><td class="term">{esc(term)}</td><td>{esc(meaning)}</td><td>{cell}</td></tr>'
        out += (f'<h3 class="gl-h">{esc(title)}</h3><table class="gl"><thead><tr><th>#</th><th>Term</th><th>Seedhi bhasha mein</th>'
                f'<th>Code mein kahan</th></tr></thead><tbody>{rows}</tbody></table>')
    return out, n, coded


def main() -> None:
    tok = BPE.load(LLM / "data/tokenizer.json")
    loss, rate = c_loss()
    damage, size = c_quant()
    gl, n_terms, n_coded = glossary()
    parts = {
        "FLOW": d_pipeline(), "GRAPH": d_graph(), "LOOP": d_loop(), "ARCH": d_arch(), "ATTN": d_attention(), "MASK": d_mask(),
        "ROPE": d_rope(), "LORA": d_lora(), "LOSS": loss, "LR": rate, "QUANT": damage, "SIZE": size, "ABL": c_ablation(),
        "GLOSSARY": gl, "N_TERMS": str(n_terms), "N_CODED": str(n_coded), "N_THEORY": str(n_terms - n_coded),
        **model_figures(tok), **data_figures(tok),
    }
    assert parts["N_PARAMS"] == re.search(r"<b>([\d,]+)</b></td></tr></tbody>", zero_figures(tok)["PARAM_WALK"]).group(1), "hand count != real model"
    reference = (DOCS / "template.html").read_text()
    parts["STYLE"] = re.search(r"<style>.*?</style>", reference, flags=re.S).group(0)  # both guides share one stylesheet
    parts |= zero_figures(tok)
    # (template, output name, strict). The reference guide must use every figure; the beginner guide picks what it needs.
    for template, name in (("template.html", "Chhotu-GPT-Andar-Se"), ("template_zero.html", "LLM-Zero-Se-Interview-Tak")):
        page = (DOCS / template).read_text()
        for key, value in parts.items():
            page = page.replace("{{" + key + "}}", value)
        assert "{{" not in page, (template, re.findall(r"\{\{\w+\}\}", page))
        out = DOCS / f"{name}.html"
        out.write_text(page)
        subprocess.run(["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "--headless=new", "--no-pdf-header-footer",
                        "--virtual-time-budget=4000", f"--print-to-pdf={out.with_suffix('.pdf')}", out.as_uri()], check=True, capture_output=True)
        print(f"{n_terms} terms ({n_coded} point at code)  ->  {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
