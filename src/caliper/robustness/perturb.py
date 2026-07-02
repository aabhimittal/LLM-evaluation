"""Deterministic, meaning-preserving perturbation operators.

A competent model's answer should be invariant to all of these. Each
operator takes (text, rng) and returns perturbed text; option shuffling is
handled separately because it must keep the scoring key in sync.
"""

from __future__ import annotations

import re

import numpy as np

_HOMOGLYPHS = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с"}

_REWRITES = [
    (r"\bWhich of the following\b", "Which one of these options"),
    (r"\bwhich of the following\b", "which one of these options"),
    (r"\bWhat is\b", "What would you say is"),
    (r"\bmost likely\b", "most probably"),
    (r"\bbest\b", "most appropriate"),
    (r"\bAn? (\w+) observes\b", r"A \1 notices"),
    (r"\bbecause\b", "since"),
]

_DISTRACTORS = [
    "Note that this question appears in a standard assessment booklet.",
    "Take your time before answering.",
    "This item was reviewed by two independent editors.",
]


def typo_noise(text: str, rng: np.random.Generator, rate: float = 0.06) -> str:
    words = text.split(" ")
    for i, w in enumerate(words):
        if len(w) > 4 and rng.random() < rate:
            j = int(rng.integers(1, len(w) - 2))
            words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def case_noise(text: str, rng: np.random.Generator, rate: float = 0.12) -> str:
    words = text.split(" ")
    for i, w in enumerate(words):
        if w and rng.random() < rate:
            words[i] = w.upper() if w[0].islower() else w.lower()
    return " ".join(words)


def unicode_noise(text: str, rng: np.random.Generator, rate: float = 0.05) -> str:
    out = []
    for ch in text:
        if ch in _HOMOGLYPHS and rng.random() < rate:
            out.append(_HOMOGLYPHS[ch])
        else:
            out.append(ch)
    return "".join(out)


def surface_paraphrase(text: str, rng: np.random.Generator) -> str:
    for pattern, repl in _REWRITES:
        text = re.sub(pattern, repl, text, count=1)
    return text


def distractor_sentence(text: str, rng: np.random.Generator) -> str:
    extra = _DISTRACTORS[int(rng.integers(0, len(_DISTRACTORS)))]
    return f"{extra} {text}"


PERTURBATIONS = {
    "paraphrase": surface_paraphrase,
    "typos": typo_noise,
    "casing": case_noise,
    "homoglyphs": unicode_noise,
    "distractor": distractor_sentence,
}


def perturb_question(name: str, question: str, rng: np.random.Generator) -> str:
    return PERTURBATIONS[name](question, rng)


def perturb_choices(
    choices: list[str], answer_index: int, rng: np.random.Generator
) -> tuple[list[str], int]:
    """Shuffle option order; returns new choices and the new answer index."""
    order = rng.permutation(len(choices))
    shuffled = [choices[int(i)] for i in order]
    new_answer = int(np.where(order == answer_index)[0][0])
    return shuffled, new_answer
