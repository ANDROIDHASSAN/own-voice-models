"""Builds the whole-project guide: docs/Voice-AI-Zero-Se.html and .pdf.

    ../../.venv/bin/python build.py

Every number, chart and table in it is read from this project's own files (training logs, checkpoints, the
dataset, the GLOSSARY.md files, each lesson's docstring). Where a model has not finished training yet, the
guide says so instead of showing a number. Run it again after a training run and the figures update.
"""

import ast
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

DOCS = Path(__file__).parent
SCRATCH = DOCS.parent
sys.path.insert(0, str(SCRATCH / "orchestrator"))
from models import lesson, romanize  # noqa: E402


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


kit = _load(SCRATCH / "llm/docs/build.py", "llm_guide")  # the SVG toolkit and stylesheet of the LLM guide
svg, text, box, arrow, table, esc = kit.svg, kit.text, kit.box, kit.arrow, kit.table, kit.esc
BLUE, ORANGE, AQUA, MUTED, SEC, INK, AXIS = kit.BLUE, kit.ORANGE, kit.AQUA, kit.MUTED, kit.SEC, kit.INK, kit.AXIS
PENDING = '<div class="box warn"><span class="k">Abhi nahi naapa</span><p>{}</p></div>'


# ---------- diagrams ----------

def d_pipeline() -> str:
    names = [("Mic", "20 ms frames", "ghost"), ("VAD", "bol raha hai?", "aqua"), ("STT", "awaaz → text", "blue"), ("Turn", "baari khatm?", "aqua"),
             ("LLM", "text → jawab", "blue"), ("Vaakya", "tukde jodna", "aqua"), ("TTS", "text → awaaz", "blue"), ("Speaker", "", "ghost")]
    out, x = "", 6
    for i, (name, sub, kind) in enumerate(names):
        out += box(x, 30, 80, 54, name, sub or None, kind, 12.5)
        if i < len(names) - 1:
            out += arrow(x + 82, 57, x + 92, 57)
        x += 94
    out += f'<path d="M 705 86 C 705 136, 150 136, 148 88" fill="none" stroke="{ORANGE}" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#ah)"/>'
    out += text(425, 112, "barge-in: user beech mein bole to agent turant chup", 11, ORANGE)
    out += text(380, 160, "neela = model jo tumne scratch se train kiya   ·   hara = orchestrator ka logic (koi model nahi)", 11, SEC)
    return svg(760, 170, out, "voice agent pipeline")


def d_stt() -> str:
    rows = [("log-mel", "(T, 80) · 100 frames/s", "ghost"), ("normalize + SpecAugment", "training mein random pattiyan chhupao", "plain"),
            ("conv subsampling ×4", "(T/4, 192) · 25 frames/s", "soft"), ("8 × block", "attention + convolution + feed-forward", "blue"),
            ("linear + log-softmax", "(T/4, 117) · har frame pe har akshar ki probability", "soft"), ("CTC", "training: loss · inference: collapse", "orange")]
    out = ""
    for i, (name, sub, kind) in enumerate(rows):
        out += box(10, 8 + i * 58, 380, 44, name, sub, kind)
        if i < len(rows) - 1:
            out += arrow(200, 53 + i * 58, 200, 64 + i * 58)
    return svg(400, 8 + len(rows) * 58, out, "STT model")


def d_tts() -> str:
    rows = [("akshar + bhasha ki id", "(L,)", "ghost"), ("text encoder · 4 blocks", "(L, 192) har akshar apne padosi jaanta hai", "blue"),
            ("duration predictor", "har akshar kitne frames?", "orange"), ("length regulator", "(T, 192) har vector ko utni baar dohrao", "soft"),
            ("decoder · 4 blocks", "(T, 192) dohraye vectors ko chikna karo", "blue"), ("linear", "(T, 80) mel spectrogram", "soft"),
            ("vocoder", "magnitude + phase → iSTFT → samples", "blue")]
    out = ""
    for i, (name, sub, kind) in enumerate(rows):
        out += box(10, 8 + i * 56, 310, 43, name, sub, kind, 12.5)
        if i < len(rows) - 1:
            out += arrow(165, 52 + i * 56, 165, 62 + i * 56)
    out += box(334, 64, 100, 43, "MAS", "asli durations", "aqua", 12)
    out += arrow(384, 108, 322, 136, dashed=True)
    out += text(384, 50, "sirf training mein", 9.5, SEC)
    return svg(440, 8 + len(rows) * 56, out, "TTS model")


def d_states() -> str:
    out = box(40, 90, 150, 56, "LISTENING", "mic sun raha hai", "aqua") + box(290, 90, 150, 56, "THINKING", "STT → LLM → TTS", "soft") + box(540, 90, 150, 56, "SPEAKING", "audio baj raha hai", "blue")
    out += arrow(192, 118, 286, 118) + arrow(442, 118, 536, 118)
    out += text(240, 108, "baari khatm", 10.5, SEC) + text(490, 108, "pehla audio", 10.5, SEC)
    out += f'<path d="M 615 148 C 615 200, 115 200, 115 150" fill="none" stroke="{MUTED}" stroke-width="1.6" marker-end="url(#ah)"/>'
    out += text(365, 206, "jawab poora baj gaya", 10.5, SEC)
    out += f'<path d="M 365 88 C 365 52, 130 52, 130 86" fill="none" stroke="{ORANGE}" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#ah)"/>'
    out += f'<path d="M 615 88 C 615 22, 100 22, 100 86" fill="none" stroke="{ORANGE}" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#ah)"/>'
    out += text(365, 16, "barge-in: user bola, to seedha LISTENING pe wapas", 10.5, ORANGE)
    return svg(730, 216, out, "agent state machine")


def d_streaming() -> str:
    """Two timelines for the same reply of three sentences: wait for everything vs stream sentence by sentence."""
    def bar(x, y, w, label, color):
        return f'<rect x="{x}" y="{y}" width="{w}" height="20" rx="4" fill="{color}"/>' + text(x + w / 2, y + 14, label, 10, "#fff", weight=600)

    out = text(8, 22, "Bina streaming", 12, INK, "start", 600)
    out += bar(120, 8, 60, "STT", AQUA) + bar(182, 8, 270, "LLM: poora jawab", BLUE) + bar(454, 8, 150, "TTS: poora jawab", ORANGE)
    out += f'<line x1="606" y1="2" x2="606" y2="34" stroke="{INK}" stroke-width="1.5"/>' + text(612, 22, "pehla audio", 10.5, INK, "start")
    out += text(8, 82, "Streaming", 12, INK, "start", 600)
    out += bar(120, 68, 60, "STT", AQUA) + bar(182, 68, 88, "LLM v1", BLUE) + bar(272, 68, 88, "LLM v2", BLUE) + bar(362, 68, 88, "LLM v3", BLUE)
    out += bar(272, 92, 50, "TTS v1", ORANGE) + bar(362, 92, 50, "TTS v2", ORANGE) + bar(452, 92, 50, "TTS v3", ORANGE)
    out += f'<line x1="324" y1="90" x2="324" y2="120" stroke="{INK}" stroke-width="1.5"/>' + text(330, 128, "pehla audio: ek vaakya ke baad", 10.5, INK, "start")
    out += text(120, 148, "samay →   (v1 = pehla vaakya)", 10.5, SEC, "start")
    return svg(730, 154, out, "streaming versus waiting")


def d_mas() -> str:
    """The alignment grid of align.py's demo, with the path MAS found."""
    import torch

    (align,) = lesson("tts", "align")
    torch.manual_seed(0)
    sounds = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]])
    true = torch.tensor([3, 1, 5, 2])
    mel = torch.repeat_interleave(sounds, true, dim=0) + 0.3 * torch.randn(11, 3)
    score = -((sounds[:, None, :] - mel[None, :, :]) ** 2).sum(-1)
    found = align.mas(score[None], torch.tensor([4]), torch.tensor([11]))[0].tolist()
    path = [l for l, d in enumerate(found) for _ in range(d)]
    out, cell = "", 46
    lo, hi = score.min().item(), score.max().item()
    for l in range(4):
        out += text(70, 46 + l * cell, f"akshar {l + 1}", 11.5, SEC, "end")
        for t in range(11):
            on = path[t] == l
            shade = int(235 - 110 * (score[l, t].item() - lo) / (hi - lo))
            fill = BLUE if on else f"rgb({shade},{shade + 8},{min(255, shade + 18)})"
            out += f'<rect x="{80 + t * cell}" y="{20 + l * cell}" width="{cell - 3}" height="{cell - 3}" rx="5" fill="{fill}"/>'
            out += text(80 + t * cell + (cell - 3) / 2, 47 + l * cell, f"{score[l, t]:.1f}", 10, "#fff" if on else INK)
    for t in range(11):
        out += text(80 + t * cell + (cell - 3) / 2, 20 + 4 * cell + 12, f"f{t + 1}", 10.5, MUTED)
    out += text(80 + 5.5 * cell, 20 + 4 * cell + 32, f"neela = MAS ka raasta.  Durations: {found}  (asli: {true.tolist()})", 11.5, INK)
    return svg(80 + 11 * cell + 10, 20 + 4 * cell + 42, out, "monotonic alignment search")


def d_ctc() -> str:
    out, labels = "", ["-", "h", "-", "i", "-"]
    path = [0, 1, 1, 2, 3, 3, 4]  # one valid path for "hi" over 7 frames
    for s, label in enumerate(labels):
        out += text(34, 39 + s * 34, "blank" if label == "-" else f'"{label}"', 11.5, SEC, "end")
        for t in range(7):
            on = path[t] == s
            out += f'<circle cx="{70 + t * 70}" cy="{35 + s * 34}" r="11" fill="{BLUE if on else "#eef1f6"}" stroke="{BLUE if on else "#d5dbe6"}"/>'
            if on:
                out += text(70 + t * 70, 39 + s * 34, label, 11, "#fff", weight=600)
    for t in range(6):
        out += f'<line x1="{81 + t * 70}" y1="{35 + path[t] * 34}" x2="{59 + (t + 1) * 70}" y2="{35 + path[t + 1] * 34}" stroke="{BLUE}" stroke-width="2"/>'
    for t in range(7):
        out += text(70 + t * 70, 198, f"frame {t + 1}", 10.5, MUTED)
    out += text(300, 222, 'ye raasta:  - h h - i i -   →  repeats milao  →  - h - i -   →  blank hatao  →  "hi"', 11.5, INK)
    return svg(560, 232, out, "one CTC path for hi")


def card(x, y, w, h, title, lines, color) -> str:
    """A coloured card with a title strip: the building block of the journey and architecture pictures."""
    out = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="#ffffff" stroke="{color}" stroke-width="1.6"/>'
    out += f'<path d="M {x} {y + 26} L {x} {y + 10} Q {x} {y} {x + 10} {y} L {x + w - 10} {y} Q {x + w} {y} {x + w} {y + 10} L {x + w} {y + 26} Z" fill="{color}"/>'
    out += text(x + w / 2, y + 18, title, 11.5, "#ffffff", weight=600)
    for i, line in enumerate(lines):
        out += text(x + w / 2, y + 44 + i * 15, line, 10.2, SEC)
    return out


def d_journey() -> str:
    """One turn of the conversation, as the candidate lives it (top) and as the agent runs it (bottom)."""
    steps = [("1  Bolta hai", ["“mera naam", "Rahul hai”"], ["VAD: awaaz shuru", "frames jama karo"], BLUE),
             ("2  Rukta hai", ["chup", ""], ["VAD: 0.5 s chuppi", "utterance band"], BLUE),
             ("3  Intezaar", ["kuch nahi", "sunai deta"], ["STT: awaaz → text", "Turn: poora hua?"], AQUA),
             ("4  Intezaar", ["(yahi delay", "mehsoos hota hai)"], ["LLM: pehla vaakya", "TTS: uska audio"], AQUA),
             ("5  Sunta hai", ["agent ka", "jawab"], ["baaki vaakya bante", "aur bajte rehte hain"], ORANGE),
             ("6  Tokta hai", ["“ruko, ek", "aur baat”"], ["barge-in: audio band", "jawab radd, phir suno"], "#a8431b")]
    out = text(10, 20, "Candidate", 12, INK, "start", 600) + text(10, 146, "Agent ke andar", 12, INK, "start", 600)
    for i, (title, top, bottom, color) in enumerate(steps):
        x = 10 + i * 122
        out += card(x, 30, 112, 82, title, top, color) + card(x, 156, 112, 82, title.split("  ")[1], bottom, color)
        out += arrow(x + 56, 114, x + 56, 130)
        if i < len(steps) - 1:
            out += arrow(x + 113, 197, x + 121, 197)
    out += f'<line x1="254" y1="254" x2="496" y2="254" stroke="{INK}" stroke-width="1.5"/><line x1="254" y1="248" x2="254" y2="260" stroke="{INK}" stroke-width="1.5"/>'
    out += f'<line x1="496" y1="248" x2="496" y2="260" stroke="{INK}" stroke-width="1.5"/>'
    out += text(375, 272, "time to first audio: candidate ruka → agent bola. Poora project isi ko chhota karne ke baare mein hai.", 10.8, INK)
    return svg(745, 280, out, "one turn from the candidate's side and the agent's side")


def d_architecture() -> str:
    out = card(20, 6, 700, 54, "CANDIDATE", ["microphone (16 kHz) aur speaker (22 kHz)"], "#52514e")
    out += card(20, 76, 700, 70, "ORCHESTRATOR  ·  scratch/orchestrator  ·  koi model nahi, sirf logic",
                ["VAD (kab bola)  ·  turn detection (kab ruka)  ·  state machine  ·  streaming  ·  barge-in  ·  latency ka naap"], AQUA)
    for i, (title, lines, color) in enumerate([("STT  ·  sunna", ["{{P_STT}}", "encoder + CTC"], BLUE), ("LLM  ·  sochna", ["{{P_LLM}}", "decoder-only GPT"], "#184f95"),
                                               ("TTS  ·  bolna", ["{{P_TTS}} + {{P_VOC}}", "acoustic + vocoder"], ORANGE)]):
        out += card(20 + i * 238, 162, 224, 70, title, lines, color)
    for i, (title, lines) in enumerate([("data", ["FLEURS 23.5 h", "hi · mr · en"]), ("data", ["kahaniyan + chat", "hi · mr · hinglish"]), ("data", ["teacher ki awaaz", "+ asli Marathi"])]):
        out += card(20 + i * 238, 248, 224, 70, title, lines, "#898781")
        out += arrow(132 + i * 238, 246, 132 + i * 238, 234)
    out += arrow(370, 62, 370, 74) + arrow(370, 148, 370, 160)
    out += card(20, 334, 700, 52, "CHECKS  ·  har folder mein check.py", ["wo cheezein jo galat hon to sab galat ho jaata hai: padding, CTC, alignment, barge-in ..."], "#0b6e4b")
    return svg(740, 392, out, "architecture in layers")


def d_mirror() -> str:
    out = text(10, 24, "STT", 13, BLUE, "start", 600) + text(10, 92, "TTS", 13, ORANGE, "start", 600)
    for i, (name, sub) in enumerate([("waveform", "16,000 numbers / s"), ("log-mel", "tasveer: samay × frequency"), ("text", "akshar")]):
        out += box(70 + i * 220, 4, 180, 40, name, sub, "soft" if i != 1 else "blue")
        out += box(70 + (2 - i) * 220, 72, 180, 40, name, sub.replace("16,000", "22,050"), "orange" if i != 1 else "blue")
    out += arrow(252, 24, 288, 24) + arrow(472, 24, 508, 24) + arrow(252, 92, 288, 92) + arrow(472, 92, 508, 92)
    out += text(270, 60, "jaankari phenko →", 10.5, SEC) + text(490, 128, "→ jaankari wapas banao", 10.5, SEC)
    return svg(740, 134, out, "recognition and synthesis are mirror images")


def d_vad() -> str:
    """A timeline: room noise, speech with a short pause inside, silence. Where the VAD fires."""
    out = f'<line x1="30" y1="60" x2="710" y2="60" stroke="#e1e0d9"/>'
    import math

    for x in range(30, 710, 3):
        speaking = 150 <= x < 330 or 362 <= x < 500
        amp = 26 * abs(math.sin(x * 0.9)) * (0.6 + 0.4 * math.sin(x * 0.13)) if speaking else 2.5
        out += f'<line x1="{x}" y1="{60 - amp:.1f}" x2="{x}" y2="{60 + amp:.1f}" stroke="{BLUE if speaking else MUTED}" stroke-width="1.6"/>'
    for x, label, color in ((168, "START", "#0b6e4b"), (590, "END", "#a8431b")):
        out += f'<line x1="{x}" y1="14" x2="{x}" y2="106" stroke="{color}" stroke-width="1.8"/>' + text(x, 122, label, 11, color, weight=600)
    out += f'<rect x="96" y="96" width="72" height="8" rx="3" fill="{AQUA}" opacity=".55"/>' + text(132, 140, "pre-roll: pehla akshar bachao", 10, SEC)
    out += f'<rect x="500" y="96" width="90" height="8" rx="3" fill="{ORANGE}" opacity=".55"/>' + text(545, 140, "hangover: 0.5 s chuppi", 10, SEC)
    out += text(346, 20, "chhota rukna: baari khatm NAHI", 10, SEC) + text(70, 20, "kamre ka shor", 10, SEC)
    return svg(740, 148, out, "what the VAD does on a timeline")


def d_distill() -> str:
    out = box(10, 30, 150, 50, "Text", "kahaniyan, khabrein, chat", "ghost") + arrow(162, 55, 198, 55)
    out += box(200, 18, 170, 74, "TEACHER", "Supertonic 3 · 99 M", "blue") + arrow(372, 55, 408, 55)
    out += box(410, 30, 130, 50, "Saaf awaaz", "ek hi voice, ghanton", "soft") + arrow(542, 55, 578, 55)
    out += box(580, 18, 150, 74, "STUDENT", "{{P_TTS}} + {{P_VOC}}", "orange")
    out += text(370, 116, "Student wahi vaakya, wahi awaaz dekhta hai aur nakal karna seekhta hai. Teacher se aage nahi ja sakta, paas pahunch sakta hai.", 10.5, SEC)
    return svg(740, 124, out, "knowledge distillation")


# ---------- beginner diagrams (the zero-to-interview guide) ----------

def _wave(xs, f) -> str:
    return " ".join(f"{x:.1f},{f(x):.1f}" for x in xs)


def d_sampling() -> str:
    """A smooth wave and the 20 dots a microphone writes down for it."""
    import math

    f = lambda x: 80 - 52 * math.sin((x - 40) / 640 * 4 * math.pi)
    out = f'<line x1="40" y1="80" x2="680" y2="80" stroke="#e1e0d9"/>'
    out += f'<polyline points="{_wave(range(40, 681, 4), f)}" fill="none" stroke="#9ec5f4" stroke-width="2"/>'
    for i in range(21):
        x = 40 + i * 32
        out += f'<line x1="{x}" y1="80" x2="{x}" y2="{f(x):.1f}" stroke="{BLUE}" stroke-width="1.4"/><circle cx="{x}" cy="{f(x):.1f}" r="4.5" fill="{BLUE}"/>'
    out += text(360, 160, "Halki neeli lehar = asli awaaz.  Gehre bindu = mic ne jo numbers likhe (samples).  Bindu jitne ghane, awaaz utni saaf.", 11, SEC)
    return svg(720, 168, out, "sampling a wave")


def d_sines() -> str:
    """Two pure tones add up to one messy wave; Fourier pulls them apart again."""
    import math

    rows = [("200 Hz (mota sur)", lambda x: 22 * math.sin(x / 60 * 2 * math.pi), BLUE),
            ("600 Hz (patla sur)", lambda x: 11 * math.sin(x / 20 * 2 * math.pi), ORANGE),
            ("dono jode = jo mic sunta hai", lambda x: 22 * math.sin(x / 60 * 2 * math.pi) + 11 * math.sin(x / 20 * 2 * math.pi), INK)]
    out = ""
    for i, (name, g, color) in enumerate(rows):
        y = 40 + i * 70
        out += text(10, y - 26, name, 11, SEC, "start")
        out += f'<polyline points="{_wave(range(10, 431, 2), lambda x, g=g, y=y: y - g(x))}" fill="none" stroke="{color}" stroke-width="1.8"/>'
        if i < 2:
            out += text(440, y + 4, "+" if i == 0 else "=", 16, MUTED, "middle", 600)
    out += arrow(460, 180, 520, 180) + text(490, 168, "Fourier", 11, INK, weight=600)
    out += f'<line x1="540" y1="210" x2="710" y2="210" stroke="{AXIS}"/>'
    out += f'<rect x="575" y="130" width="26" height="80" rx="3" fill="{BLUE}"/><rect x="655" y="170" width="26" height="40" rx="3" fill="{ORANGE}"/>'
    out += text(588, 226, "200 Hz", 10, MUTED) + text(668, 226, "600 Hz", 10, MUTED) + text(625, 120, "kaunsa sur kitna zor se", 10.5, SEC)
    return svg(720, 236, out, "adding tones and taking them apart")


def d_frames() -> str:
    import math

    f = lambda x: 70 - (30 * math.sin(x / 9) + 14 * math.sin(x / 3.1)) * (0.4 + 0.6 * abs(math.sin(x / 120)))
    out = f'<polyline points="{_wave(range(20, 701, 2), f)}" fill="none" stroke="#9ec5f4" stroke-width="1.4"/>'
    colors = [BLUE, ORANGE, AQUA, "#184f95"]
    for i, c in enumerate(colors):
        x = 40 + i * 64
        out += f'<rect x="{x}" y="{118 + i * 14}" width="160" height="10" rx="3" fill="{c}" opacity=".75"/>' + text(x + 80, 115 + i * 14 + 10, f"frame {i + 1}", 9.5, "#fff", weight=600)
    out += f'<line x1="40" y1="190" x2="104" y2="190" stroke="{INK}" stroke-width="1.4" marker-end="url(#ah)"/>' + text(72, 204, "hop 10 ms", 10.5, INK)
    out += text(300, 204, "har frame 25 ms lamba; agla 10 ms baad shuru, isliye frames ek doosre pe chadhe hain", 10.5, SEC, "start")
    return svg(720, 212, out, "overlapping frames")


def d_mel_bank() -> str:
    """The real mel filterbank of stt/audio.py, with 12 bands instead of 80 so each triangle is visible."""
    (audio,) = lesson("stt", "audio")
    bank = audio.mel_filterbank(12, 400, 16000)  # (12, 201)
    X = lambda b: 40 + b * 650 / 200
    out = f'<line x1="40" y1="150" x2="690" y2="150" stroke="{AXIS}"/>'
    for i, row in enumerate(bank):
        color = [BLUE, ORANGE, AQUA][i % 3]
        pts = " ".join(f"{X(b):.1f},{150 - 110 * v:.1f}" for b, v in enumerate(row.tolist()) if v > 0 or (0 < b < 200 and (row[b - 1] > 0 or row[b + 1] > 0)))
        out += f'<polyline points="{pts}" fill="{color}" fill-opacity=".14" stroke="{color}" stroke-width="1.5"/>'
    for hz in (0, 1000, 2000, 4000, 8000):
        out += text(X(hz / 40), 166, f"{hz // 1000} kHz" if hz else "0", 10, MUTED)
    out += text(365, 186, "Har triangle = ek mel band. Neeche patle (baareek sunna), upar chaude (mota sunna). Asli model mein aise 80 hain.", 10.5, SEC)
    return svg(730, 194, out, "mel filterbank")


def d_phase() -> str:
    import math

    out = f'<line x1="20" y1="60" x2="700" y2="60" stroke="#e1e0d9"/>'
    out += f'<polyline points="{_wave(range(20, 701, 3), lambda x: 60 - 36 * math.sin(x / 110 * 2 * math.pi))}" fill="none" stroke="{BLUE}" stroke-width="2"/>'
    out += f'<polyline points="{_wave(range(20, 701, 3), lambda x: 60 - 36 * math.sin(x / 110 * 2 * math.pi + 1.6))}" fill="none" stroke="{ORANGE}" stroke-width="2" stroke-dasharray="6 4"/>'
    out += text(360, 122, "Dono lehron ka sur aur zor barabar (magnitude same), bas shuru alag jagah se (phase alag). Spectrogram dono ko ek jaisa dikhata hai.", 10.5, SEC)
    return svg(720, 130, out, "same magnitude, different phase")


def d_frame_probs() -> str:
    """An illustration with made-up probabilities: what the recognizer outputs for the word 'kal'."""
    syms = ["blank", "k", "a", "l"]
    probs = [[.10, .80, .05, .05], [.20, .70, .05, .05], [.85, .05, .05, .05], [.15, .05, .75, .05], [.10, .05, .10, .75], [.30, .02, .08, .60]]
    out = ""
    for s, name in enumerate(syms):
        out += text(70, 52 + s * 34, name, 11.5, SEC, "end")
    for t, col in enumerate(probs):
        best = max(range(4), key=col.__getitem__)
        for s, p in enumerate(col):
            on = s == best
            out += f'<rect x="{84 + t * 74}" y="{32 + s * 34}" width="66" height="28" rx="5" fill="{BLUE if on else "#f4f6fa"}" fill-opacity="{1 if on else 1}"/>'
            out += text(117 + t * 74, 51 + s * 34, f"{p:.2f}", 11, "#fff" if on else SEC, weight=600 if on else 400)
        out += text(117 + t * 74, 22, f"frame {t + 1}", 10, MUTED)
        out += text(117 + t * 74, 188, "-" if best == 0 else syms[best], 14, INK, weight=600)
    out += text(560, 188, "→  k k - a l l  →  k - a l  →  kal", 12, INK, "start", 600)
    out += text(560, 206, "(repeats milao, blank hatao)", 10, SEC, "start")
    return svg(830, 214, out, "per-frame probabilities for kal")


def d_length_reg() -> str:
    out = ""
    for i, (c, d) in enumerate((("k", 2), ("a", 1), ("l", 3))):
        out += box(20 + i * 70, 30, 56, 44, c, f"{d} frame", "blue")
    out += arrow(232, 52, 292, 52) + text(262, 42, "dohrao", 10.5, SEC)
    for j, c in enumerate("kkalll"):
        out += box(300 + j * 66, 30, 56, 44, c, None, "soft")
    out += text(360, 102, "Akshar ka vector utni baar copy hota hai jitni uski duration. Text ki ghadi se awaaz ki ghadi.", 10.5, SEC)
    return svg(720, 110, out, "length regulator")


# ---------- figures read from the project's own files ----------

def c_wave_and_mel() -> tuple[str, str]:
    """A real clip from the recognizer's dev split: 25 ms of its waveform cannot exist any more (only the
    mel was stored), so the waveform comes from the TTS dev audio and the mel from the STT dev features."""
    wave_svg = PENDING.format("TTS ka data abhi ban raha hai; waveform ki tasveer uske baad aayegi.")
    pcm_path = SCRATCH / "tts/data/dev.pcm"
    if pcm_path.exists() and pcm_path.stat().st_size > 200_000:
        pcm = np.memmap(pcm_path, dtype=np.int16, mode="r")[30000:30551].astype(np.float32) / 32768  # 551 samples = 25 ms
        pts = " ".join(f"{20 + i * 1.25:.1f},{70 - v * 150:.1f}" for i, v in enumerate(pcm))
        body = f'<line x1="20" y1="70" x2="710" y2="70" stroke="#e1e0d9"/><polyline points="{pts}" fill="none" stroke="{BLUE}" stroke-width="1.6"/>'
        body += text(365, 136, "25 ms ki awaaz = 551 numbers (22,050 samples har second). Yahi ek 'frame' hai.", 11.5, SEC)
        wave_svg = svg(730, 142, body, "25 ms of a waveform")

    mel_svg = PENDING.format("STT ka data nahi mila.")
    if (SCRATCH / "stt/data/dev.json").exists():
        index = json.loads((SCRATCH / "stt/data/dev.json").read_text())
        entry = next(e for e in index if e["lang"] == "hi" and 300 < e["frames"] < 600)
        mel = np.memmap(SCRATCH / "stt/data/dev.mel", dtype=np.float16, mode="r").reshape(-1, 80)
        clip = mel[entry["start"] : entry["start"] + 300].astype(np.float32)  # the first 3 seconds
        grid = clip.reshape(150, 2, 40, 2).mean(axis=(1, 3))  # 150 time steps x 40 bands
        grid = (grid - np.percentile(grid, 5)) / (np.percentile(grid, 99) - np.percentile(grid, 5))
        ramp = ["#f4f7fb", "#cfe0f6", "#9ec5f4", "#5b9be6", "#2a78d6", "#184f95", "#0c1a2e"]
        body = ""
        for t in range(150):
            for b in range(40):
                level = min(len(ramp) - 1, max(0, int(grid[t, b] * len(ramp))))
                if level:
                    body += f'<rect x="{40 + t * 4.5:.1f}" y="{8 + (39 - b) * 4.2:.1f}" width="4.6" height="4.3" fill="{ramp[level]}"/>'
        body += text(34, 20, "8 kHz", 10, MUTED, "end") + text(34, 174, "0", 10, MUTED, "end")
        body += text(40, 192, "0 s", 10, MUTED, "start") + text(715, 192, "3 s", 10, MUTED, "end")
        body += text(378, 208, f"Asli Hindi clip ka log-mel (dev split):  “{entry['text'][:46]} ...”", 11, SEC)
        mel_svg = svg(730, 214, body, "log-mel spectrogram of a real clip")
    return wave_svg, mel_svg


def _epochs(log: Path) -> list[tuple[int, float, float, dict]]:
    """(epoch, train loss, dev CER, {lang: (cer, wer)}) for every evaluation line in an STT training log."""
    out = []
    if log.exists():
        for line in log.read_text().splitlines():
            if m := re.match(r"== epoch (\d+)\s+train loss ([\d.]+)\s+dev CER ([\d.]+)(.*)", line):
                langs = {l: (float(c), float(w)) for l, c, w in re.findall(r"(\w+) CER ([\d.]+) WER ([\d.]+)", m.group(4))}
                out.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), langs))
    return out


def stt_figures() -> dict:
    first, final = _epochs(SCRATCH / "stt/runs/train_warmup1000.log"), _epochs(SCRATCH / "stt/runs/train.log")
    out = {"STT_CURVE": PENDING.format("STT ki training ka log nahi mila."), "STT_STATUS": "", "STT_SAMPLES": ""}
    if final:
        last = max(e[0] for e in final + first)
        series = [("warmup 3000", BLUE, [(e, c) for e, _, c, _ in final], False)]
        if first:
            series.append(("warmup 1000", ORANGE, [(e, c) for e, _, c, _ in first], False))
        ticks = [t for t in (1, 5, 10, 20, 30, 40, 50, 60) if t <= max(last, 10)]
        out["STT_CURVE"] = kit.line_chart(series, max(last, 10), 0, 1, ticks, [0, 0.25, 0.5, 0.75, 1], "epoch", "dev CER", h=225, yfmt="{:.2f}")
        epoch, loss, cer, langs = min(final, key=lambda e: e[2])
        done = final[-1][0]
        out["STT_STATUS"] = (f"Training {done} epochs chali. Sabse accha dev CER <b>{cer:.3f}</b> epoch {epoch} pe aaya; uske baad train loss "
                             "girta raha aur dev CER badha (overfitting), isliye wahi checkpoint rakha." if done >= 60 else
                             f"Ye guide tab bani jab training epoch {done} pe thi (60 mein se). Ab tak ka sabse accha dev CER: "
                             f"<b>{cer:.3f}</b> (epoch {epoch}).")
    log = (SCRATCH / "stt/runs/train.log").read_text() if (SCRATCH / "stt/runs/train.log").exists() else ""
    samples = re.findall(r"   (\w\w) ref: (.*)\n      hyp: (.*)", log)[-3:]
    out["STT_SAMPLES"] = "".join(f'<div class="sample"><b>{l}</b> asli: {esc(r)}<br><b>{l}</b> model: {esc(h)}</div>' for l, r, h in samples)
    return out


def measured_tables() -> dict:
    """The STT test results and the quantization table, parsed from the logs evaluate.py and shrink.py wrote."""
    out = {"STT_TEST": PENDING.format("STT ka test abhi nahi chala."), "STT_SHRINK": PENDING.format("Quantization abhi nahi naapa.")}
    names = {"hi": "Hindi", "mr": "Marathi", "en": "English"}
    log = SCRATCH / "stt/runs/eval.log"
    if log.exists():
        txt = log.read_text()
        ours = {l: (c, w) for l, c, w in re.findall(r"^\s+(\w\w)\s+\d+ clips.*?CER ([\d.]+)\s+WER ([\d.]+)", txt, flags=re.M)}
        big = {l: (c, w, r1, r3) for l, r1, c, w, r3 in re.findall(r"^\s+(\w\w)\s+[\d.]+ / [\d.]+ / ([\d.]+)\s+[\d.]+ / [\d.]+ / [\d.]+\s+([\d.]+) / ([\d.]+) / ([\d.]+)", txt, flags=re.M)}
        rows = [[names[l], ours[l][0], ours[l][1], big[l][0], big[l][1]] for l in ("hi", "mr", "en") if l in ours and l in big]
        out["STT_TEST"] = table(["Bhasha", "Hamara CER", "Hamara WER", "300 M model CER", "300 M model WER"], rows, right=(1, 2, 3, 4))
        speed = [f"{names[l]}: hamara {float(big[l][2]):.3f}, bada {float(big[l][3]):.3f}" for l in ("hi", "mr", "en") if l in big]
        out["STT_SPEED"] = "Real-time factor (CPU, 2 threads; kam = tez): " + " · ".join(speed)
    log = SCRATCH / "stt/runs/shrink.log"
    if log.exists():
        rows = [[f, s, c, "—" if not d else "0" if float(d) == 0 else d] for f, s, c, d in re.findall(r"^(fp\d\d|int\d, [\w-]+)\s+([\d.]+ MB)\s+([\d.]+)\s*([+-][\d.]+)?", log.read_text(), flags=re.M)]
        out["STT_SHRINK"] = table(["Format", "Size", "Dev CER", "Badlaav"], rows, right=(1, 2, 3))
    out.setdefault("STT_SPEED", "")
    return out


def tts_figures() -> dict:
    out = {"TTS_DATA": PENDING.format("TTS ka data abhi ban raha hai (teacher vaakya padh raha hai)."),
           "TTS_RESULTS": PENDING.format("Acoustic model aur vocoder abhi train nahi hue. Training ke baad ye guide dobara banao: "
                                         "<code>python docs/build.py</code>.")}
    index, pcm = SCRATCH / "tts/data/train.json", SCRATCH / "tts/data/train.pcm"
    if index.exists() and index.stat().st_mtime >= pcm.stat().st_mtime:  # the index is written last: older means a run is in progress
        entries = json.loads(index.read_text())
        names = {"hi": "Hindi (teacher)", "en": "English (teacher)", "hing": "Hinglish, Roman (teacher)", "mr": "Marathi (asli recording)"}
        rows = [[names[l], f"{sum(e['lang'] == l for e in entries):,}", f"{sum(e['samples'] for e in entries if e['lang'] == l) / 22050 / 3600:.2f} h"]
                for l in names if any(e["lang"] == l for e in entries)]
        out["TTS_DATA"] = table(["Bhasha", "Clips", "Audio"], rows, right=(1, 2))
    evaluation, log = SCRATCH / "tts/runs/eval.log", SCRATCH / "tts/runs/train.log"
    if evaluation.exists():
        names = {"hi": "Hindi", "mr": "Marathi (teacher = asli recording)", "en": "English", "hing": "Hinglish, Roman"}
        # rows look like "hi  40 ..." (first evaluate.py) or "hi rasa_hi_female  20 ..." (one row per voice)
        found = re.findall(r"^(\w+)(?: (\w+))?\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)$", evaluation.read_text(), flags=re.M)
        rows = [[names[l] + (f" · {v}" if v.startswith("rasa") else ""), s, t, r] for l, v, s, t, r in found]
        assert rows, f"no result rows parsed from {evaluation}"
        out["TTS_RESULTS"] = (table(["Bhasha", "Student CER", "Teacher CER", "Real-time factor"], rows, right=(1, 2, 3)) +
                              "<p>40 dev vaakya har bhasha, judge Omnilingual ASR 300 M, CPU 2 threads. Hinglish ka 0.29 judge ki galti hai "
                              "(Roman Hinglish ki fixed spelling nahi): wahan student aur teacher ka fark dekho. Real-time factor 0.03 = real time se ~30 guna tez.</p>")
    epochs = [(int(e), float(m)) for e, m in re.findall(r"^== epoch (\d+)\s+train mel [\d.]+\s+dev mel ([\d.]+)", log.read_text() if log.exists() else "", flags=re.M)]
    if epochs:
        best = min(epochs, key=lambda e: e[1])
        chart = kit.line_chart([("dev mel", ORANGE, epochs, False)], max(e for e, _ in epochs), 0.2, 0.45, [1, 10, 20, 30, 40, 50, 60],
                               [0.2, 0.3, 0.4], "epoch", "dev mel loss", h=200, yfmt="{:.1f}")
        out["TTS_RESULTS"] = (f'<figure><div class="fig-t">Acoustic model ki training: dev mel loss (kam = asli awaaz ke kareeb)</div>{chart}'
                              f"<figcaption>Sabse accha epoch {best[0]} pe ({best[1]:.3f}); uske baad sirf train loss gira. "
                              "11 ghante data ki yahi seema hai.</figcaption></figure>") + out["TTS_RESULTS"]
    return out


def e2e_table() -> str:
    """One real turn and one barge-in, through all three trained models, from the logs agent.py --wav wrote."""
    rows = []
    for name, label in (("e2e", "Ek turn"), ("e2e_barge_in", "Barge-in")):
        log = SCRATCH / f"orchestrator/runs/{name}.log"
        for i, (*ms, cut) in enumerate(re.findall(r"^\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\[interrupted\])?", log.read_text() if log.exists() else "", flags=re.M)):
            rows.append([label + (": kata hua jawab" if cut else ": naya jawab" if i else ""), *ms])
    if not rows:
        return PENDING.format("Asli models ke saath end-to-end test abhi nahi chala: <code>python agent.py --wav q.wav</code>.")
    return (table(["ms", "Intezaar", "STT", "LLM", "TTS", "Pehli awaaz"], rows, right=(1, 2, 3, 4, 5)) +
            "<p>Ek FLEURS Hindi clip, M1 Pro CPU. Teeno model milke 150 ms se kam lete hain; baaki intezaar turn detection ka hai.</p>")


def turn_table() -> str:
    """The LLM-based end-of-turn signal, measured live on ten short Hinglish fragments."""
    (turn,) = [_load(SCRATCH / "orchestrator/turn.py", "turn_lesson")]
    model, tokenizer = lesson("llm", "model", "tokenizer")
    net, _ = model.load(SCRATCH / "llm/runs/ckpt.pt", "cpu")
    complete = turn.lm_completeness(net, tokenizer.BPE.load(SCRATCH / "llm/data/tokenizer.json"))
    cases = [("mera order number hai", False), ("kal main office", False), ("mujhe lagta hai ki", False), ("agar tum free ho to", False),
             ("main soch raha tha ki hum", False), ("haan theek hai", True), ("kal main office nahi aaunga", True), ("tum kahan ho?", True),
             ("chalo phir milte hain", True), ("kya kar rahe ho aaj?", True)]
    rows = [[esc(t), "khatm" if end else "rukna", f"{complete(t):.3f}"] for t, end in cases]
    return table(["Ab tak jo bola", "Sach", "P(poora)"], rows, right=(2,))


def sizes() -> dict:
    import torch

    (stt,), (tts, voc), (llm,) = lesson("stt", "model"), lesson("tts", "model", "vocoder"), lesson("llm", "model")
    count = lambda m: sum(p.numel() for p in m.parameters())
    n = {"STT": count(stt.Encoder(stt.Config(vocab_size=117))), "TTS": count(tts.Acoustic(tts.Config())), "VOC": count(voc.Vocoder(voc.Config())),
         "LLM": count(llm.GPT(llm.Config(**torch.load(SCRATCH / "llm/runs/ckpt.pt", map_location="cpu", weights_only=True)["config"])))}
    out = {f"P_{k}": f"{v / 1e6:.1f} M" for k, v in n.items()}
    out["P_ALL"] = f"{sum(n.values()) / 1e6:.0f} M"
    out["MB_INT8"] = f"{sum(n.values()) / 1e6:.0f} MB"
    return out


def file_map() -> str:
    """Every lesson file with the first line of its docstring: the map of the project, read from the code itself."""
    out = ""
    for folder, title in (("llm", "LLM: text ka dimaag"), ("stt", "STT: sunna"), ("tts", "TTS: bolna"), ("orchestrator", "Orchestrator: baat-cheet")):
        rows = []
        for path in sorted((SCRATCH / folder).glob("*.py")):
            doc = " ".join((ast.get_docstring(ast.parse(path.read_text())) or "").split("\n\n")[0].split())  # the first paragraph
            doc = doc if doc.startswith("Lesson") and len(doc) < 120 else re.split(r"(?<=[a-z])\. ", doc)[0]
            m = re.match(r"Lesson (\d+) - (.*)", doc)
            rows.append((int(m.group(1)) if m else 99, path.name, m.group(2) if m else doc, len(path.read_text().splitlines())))
        body = [[f"<code>{name}</code>", str(n) if n < 99 else "", esc(what.rstrip(".")), str(lines)] for n, name, what, lines in sorted(rows)]
        out += f"<h3>{title} &nbsp;<code>scratch/{folder}/</code></h3>" + table(["File", "Lesson", "Kya karti hai (file ki pehli line se)", "Lines"], body, right=(3,))
    return out


def questions() -> tuple[str, int]:
    """The interview bank in docs/questions.md as HTML: topic -> level -> question with a simple and a spoken answer."""
    out, n = "", 0
    colors = {"LLM": "#184f95", "STT": BLUE, "TTS": ORANGE, "Orchestration": AQUA}
    for topic in (DOCS / "questions.md").read_text().split("\n## ")[1:]:
        name, *levels = topic.split("\n### ")
        name = name.strip()
        for level in levels:
            title, body = level.split("\n", 1)
            head = (f'<h3><span style="background:{colors[name]};color:#fff;border-radius:6px;padding:.6mm 2.5mm">{esc(name)}</span>'
                    f' &nbsp;{esc(title.strip())}</h3>')
            for i, (q, s, b) in enumerate(re.findall(r"Q: (.+)\nS: (.+)\nB: (.+)", body)):
                n += 1
                item = f'<div class="qa"><div class="q">{n}. {esc(q)}</div><p class="hi">{esc(s)}</p><div class="en">{esc(b)}</div></div>'
                # the heading travels with its first question, so it is never left alone at the foot of a page
                out += f'<div style="break-inside:avoid">{head}{item}</div>' if i == 0 else item
    return out, n


def glossaries() -> tuple[str, int]:
    out, n = "", 0
    for folder, title in (("stt", "STT"), ("tts", "TTS"), ("orchestrator", "Orchestration")):
        out += f'<h3 class="gl-h">{title} glossary</h3>'
        for block in (SCRATCH / folder / "GLOSSARY.md").read_text().split("\n## ")[1:]:
            head, *lines = block.split("\n")
            rows = ""
            for num, term, meaning, where in re.findall(r"^\| (\d+) \| (.+?) \| (.+?) \| (.+?) \|$", "\n".join(lines), flags=re.M):
                n += 1
                ref = re.fullmatch(r"`(\w+\.py): (.+)`", where)
                cell = f'<span class="where"><b>{esc(ref.group(1))}</b> {esc(ref.group(2))}</span>' if ref else '<span class="theory">theory</span>'
                rows += f'<tr><td class="num">{num}</td><td class="term">{esc(term)}</td><td>{esc(meaning)}</td><td>{cell}</td></tr>'
            out += (f'<h3 class="gl-h" style="font-size:10.5pt">{esc(head)}</h3><table class="gl"><thead><tr><th>#</th><th>Term</th>'
                    f'<th>Seedhi bhasha mein</th><th>Code mein kahan</th></tr></thead><tbody>{rows}</tbody></table>')
    return out, n


def main() -> None:
    wave, mel = c_wave_and_mel()
    gl, n_terms = glossaries()
    qa, n_qa = questions()
    parts = {"QA": qa, "N_QA": str(n_qa), "SAMPLING": d_sampling(), "SINES": d_sines(), "FRAMES": d_frames(), "MELBANK": d_mel_bank(),
             "PHASE": d_phase(), "FRAMEPROBS": d_frame_probs(), "LENREG": d_length_reg(), "MACHINE": kit.d_machine(), "LOOP": kit.d_loop(),
             **measured_tables(),"JOURNEY": d_journey(), "ARCH": d_architecture(), "MIRROR": d_mirror(), "VAD": d_vad(), "DISTILL": d_distill(),
             "PIPE": d_pipeline(), "STT_ARCH": d_stt(), "TTS_ARCH": d_tts(), "STATES": d_states(), "STREAM": d_streaming(), "MAS": d_mas(),
             "CTC": d_ctc(), "WAVE": wave, "MEL": mel, "TURN": turn_table(), "E2E": e2e_table(), "FILES": file_map(), "GLOSSARY": gl, "N_TERMS": str(n_terms),
             "ROMAN": esc(romanize("मेरा नाम कमल है")), **stt_figures(), **tts_figures(), **sizes()}
    style = re.search(r"<style>.*?</style>", (SCRATCH / "llm/docs/template.html").read_text(), flags=re.S).group(0)
    extra = re.findall(r"<style>.*?</style>", (SCRATCH / "llm/docs/template_zero.html").read_text(), flags=re.S)[-1]
    parts["STYLE"] = style + extra
    # (template, output). The project guide shows everything; the beginner guide teaches from zero and picks its figures.
    for template, name in (("template.html", "Voice-AI-Zero-Se"), ("template_zero.html", "Awaaz-AI-Shuru-Se")):
        page = (DOCS / template).read_text()
        for key, value in parts.items():
            page = page.replace("{{" + key + "}}", value)
        assert "{{" not in page, (template, re.findall(r"\{\{\w+\}\}", page))
        out = DOCS / f"{name}.html"
        out.write_text(page)
        subprocess.run(["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "--headless=new", "--no-pdf-header-footer",
                        "--virtual-time-budget=4000", f"--print-to-pdf={out.with_suffix('.pdf')}", out.as_uri()], check=True, capture_output=True)
        print(f"->  {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
