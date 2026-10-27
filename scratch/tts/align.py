"""Lesson 4 - alignment: how long is each character spoken? Nobody told us that either.

    python align.py

Recognition had the alignment problem (which frame is which character) and CTC summed over all answers.
Synthesis has the same problem from the other side: to generate 430 mel frames from 52 characters, the
model must know that this "a" lasts 9 frames and that comma 21. Durations are not in the data.

Monotonic Alignment Search (MAS, from Glow-TTS, Kim 2020) finds them during training, with no outside tool:

    1. the text encoder guesses, for every character, what its sound looks like: one mel frame, `mu`
    2. score[i, t] = how well character i's guess matches mel frame t  (negative squared distance)
    3. find the best MONOTONIC path through that table: every frame is assigned to exactly one character,
       characters are used in order, none is skipped
    4. duration of character i = number of frames the path gave it

Step 3 is dynamic programming, the same shape as the Viterbi alignment in ../stt/ctc.py but simpler: from
cell (i, t) the path may only stay on the same character or move to the next one.

It looks circular: good guesses need good alignments and good alignments need good guesses. It still
converges, because even a random encoder makes silence look different from speech, and every improvement
of one side improves the other.
"""

import numpy as np
import torch

NEG = -1e9


def mas(score: torch.Tensor, text_lens: torch.Tensor, mel_lens: torch.Tensor) -> torch.Tensor:
    """score: (B, L, T), higher = character l fits frame t better. Returns durations (B, L), whole frames,
    each real character at least 1, and each row summing to that clip's number of frames.

    best[l, t] = score of the best path that ends with frame t assigned to character l
               = score[l, t] + max(best[l, t - 1]      stay on this character
                                 , best[l - 1, t - 1])  step to it from the previous one"""
    score = score.detach().cpu().numpy().astype(np.float64)
    text_lens, mel_lens = text_lens.cpu().numpy(), mel_lens.cpu().numpy()
    B, L, T = score.shape
    best = np.full((B, L, T), NEG)
    best[:, 0, 0] = score[:, 0, 0]  # a path must start at (first character, first frame)
    for t in range(1, T):
        stay = best[:, :, t - 1]
        step = np.concatenate((np.full((B, 1), NEG), stay[:, :-1]), axis=1)
        best[:, :, t] = score[:, :, t] + np.maximum(stay, step)

    durations = np.zeros((B, L), dtype=np.int64)
    rows = np.arange(B)
    char = text_lens - 1  # ... and must end at (last character, last frame). Walk back from there.
    for t in range(T - 1, -1, -1):
        live = t < mel_lens  # clips shorter than the batch have not started yet (we are walking backwards)
        durations[rows[live], char[live]] += 1
        came_by_step = (char > 0) & (t > 0) & ((best[rows, np.maximum(char - 1, 0), max(t - 1, 0)] > best[rows, char, max(t - 1, 0)]) | (char == t))
        char = np.where(live & came_by_step, char - 1, char)
    return torch.from_numpy(durations)


def demo() -> None:
    torch.manual_seed(0)
    # A 4-character "utterance". Each character has a true sound (a 3-number vector) and a true duration.
    sounds = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]])
    true = torch.tensor([3, 1, 5, 2])
    mel = torch.repeat_interleave(sounds, true, dim=0) + 0.3 * torch.randn(11, 3)  # 11 noisy frames
    score = -((sounds[:, None, :] - mel[None, :, :]) ** 2).sum(-1)  # (4 characters, 11 frames)

    print("score table (rows = characters, columns = frames, higher = better match):")
    for row in score:
        print("  " + " ".join(f"{v:6.2f}" for v in row.tolist()))
    found = mas(score[None], torch.tensor([4]), torch.tensor([11]))[0]
    print(f"\ntrue durations : {true.tolist()}\nMAS durations  : {found.tolist()}   (sum = {found.sum().item()} frames)")
    assert found.tolist() == true.tolist()


if __name__ == "__main__":
    demo()
