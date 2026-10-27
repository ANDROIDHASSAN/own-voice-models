"""A teacher that asks questions: a lesson script in the LLM's seat.

The pipeline only needs llm(text) -> pieces of text (models.py). A 12 M model cannot run a lesson, so here a
plain state machine takes that seat: it checks the answer the STT heard, praises or gives a hint, and asks the
next question. Same plug, different brain, exactly like swapping LLM plugins in LiveKit. Everything it says
is Devanagari, so the flow TTS (24 kHz, Rasa voices) speaks all of it, and `style` tells it how: happy for
praise, conv for questions.

Answers are matched loosely: the recognizer has a character error rate of about 0.11 on Hindi, so "दिल्ली"
may arrive as "दिली". A heard word counts if it is close enough to any accepted spelling.
"""

from collections.abc import Iterator
from difflib import SequenceMatcher

from models import romanize

LESSON = [  # (question, accepted answers, hint)
    ("भारत की राजधानी कौन सी है?", ["दिल्ली", "नई दिल्ली", "देल्ही", "डेल्ही"], "ये शहर लाल किले के लिए मशहूर है।"),
    ("भारत का राष्ट्रीय पक्षी कौन सा है?", ["मोर"], "ये पक्षी बारिश में नाचता है।"),
    ("एक हफ़्ते में कितने दिन होते हैं?", ["सात", "७", "7"], "सोमवार से रविवार तक गिनो।"),
    ("गंगा नदी किस पर्वत से निकलती है?", ["हिमालय"], "ये दुनिया का सबसे ऊँचा पर्वत है।"),
    ("दो और तीन मिलाकर कितने होते हैं?", ["पाँच", "पांच", "५", "5"], "अपनी उँगलियों पर गिनो।"),
]


def _roman(text: str) -> str:
    return romanize(text).replace(" ", "").lower()


def close(heard: str, answers: list[str], threshold: float = 0.6) -> bool:
    """Compared in Roman script without spaces, so a wrong matra or a split word costs little: measured on our
    STT, "सaत" -> saat = saat, "ही माले" -> heemaale ~ himaalay (0.62). The first letter must match too, else
    "आठ" (aath) passes for "सात" (saat) at 0.75."""
    words = heard.replace("।", " ").replace(".", " ").split()
    candidates = [_roman(" ".join(words[i : i + n])) for n in (1, 2) for i in range(len(words) - n + 1)]
    return any(w[:1] == _roman(a)[:1] and SequenceMatcher(None, w, _roman(a)).ratio() >= threshold
               for w in candidates for a in answers)


class Teacher:
    def __init__(self) -> None:
        self.q, self.tries, self.score, self.style = 0, 0, 0, "conv"

    def intro(self) -> str:
        return f"नमस्ते! मैं आपकी टीचर हूँ। आज हम भारत के बारे में {len(LESSON)} आसान सवाल करेंगे। पहला सवाल। {LESSON[0][0]}"

    def __call__(self, heard: str) -> Iterator[str]:
        if self.q >= len(LESSON):
            self.style = "conv"
            yield "क्लास ख़त्म हो गई। फिर से खेलने के लिए कॉल दोबारा शुरू करो।"
            return
        question, answers, hint = LESSON[self.q]
        if close(heard, answers):
            self.score += 1
            self.style = "happy"
            yield "बहुत बढ़िया! बिल्कुल सही जवाब।"
        elif self.tries == 0:
            self.tries = 1
            self.style = "conv"
            yield f"हम्म, एक बार और सोचो। संकेत: {hint}"
            return
        else:
            self.style = "sad"
            yield f"कोई बात नहीं। सही जवाब है {answers[0]}।"
        self.q, self.tries = self.q + 1, 0
        self.style = "conv"
        if self.q < len(LESSON):
            yield f"अगला सवाल। {LESSON[self.q][0]}"
        else:
            self.style = "happy"
            yield f"शाबाश! आपने {len(LESSON)} में से {self.score} सवाल सही किए।"


def demo() -> None:
    t = Teacher()
    assert close("नई दिली", LESSON[0][1]) and close("मोर है", LESSON[1][1]) and not close("कबूतर", LESSON[1][1])
    assert close("सaत", LESSON[2][1]) and close("ही माले", LESSON[3][1]) and close("पाज", LESSON[4][1])  # heard by our STT
    assert not any(close(w, LESSON[2][1]) for w in ("आठ", "छह", "दस")) and not close("चार", LESSON[4][1])
    assert list(t("दिल्ली"))[-1] == "अगला सवाल। " + LESSON[1][0]
    assert "संकेत" in list(t("कबूतर"))[0]  # first wrong answer: a hint, same question
    assert list(t("तोता"))[0].startswith("कोई बात नहीं")  # second wrong answer: reveal, move on
    for q, answers, _ in LESSON[2:]:
        out = list(t(answers[0]))
    assert out[-1] == f"शाबाश! आपने {len(LESSON)} में से {len(LESSON) - 1} सवाल सही किए।"
    print("teacher: all checks passed")


if __name__ == "__main__":
    demo()
