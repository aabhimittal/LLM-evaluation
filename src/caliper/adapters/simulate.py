"""Simulated models with known latent properties.

These back the test suite and the token-free demo mode: a
:class:`SimulatedSubject` has a *known* ability, calibration skew,
robustness and contamination status, so every estimator in Caliper can be
checked against ground truth. A :class:`SimulatedJudge` has injectable
position and verbosity biases, so the bias audit can be validated too.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from caliper.adapters.base import ChatMessage, ModelAdapter
from caliper.prompts import LETTERS
from caliper.types import Item, ItemBank


def _rng(*parts) -> np.random.Generator:
    blob = "|".join(str(p) for p in parts)
    seed = int(hashlib.sha256(blob.encode()).hexdigest()[:12], 16)
    return np.random.default_rng(seed)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def three_pl(theta: float, a: float, b: float, c: float) -> float:
    return c + (1.0 - c) / (1.0 + np.exp(-a * (theta - b)))


class SimulatedSubject(ModelAdapter):
    """A test-taker with known ability ``theta``.

    Parameters
    ----------
    theta: latent ability on the IRT scale (0 = average calibration model).
    calibration_skew: 1.0 = well calibrated verbalized confidence;
        <1 = overconfident, >1 = underconfident.
    robustness: probability of giving the same answer under a surface
        perturbation of the question.
    contaminated: if True, reproduces benchmark continuations and options
        verbatim (a memorizing model).
    """

    def __init__(
        self,
        theta: float = 0.5,
        bank: ItemBank | None = None,
        seed: int = 0,
        calibration_skew: float = 1.0,
        robustness: float = 0.92,
        contaminated: bool = False,
        name: str | None = None,
    ):
        self.theta = theta
        self.bank = bank if bank is not None else ItemBank.bundled()
        self.seed = seed
        self.calibration_skew = calibration_skew
        self.robustness = robustness
        self.contaminated = contaminated
        self.name = name or f"simulated(theta={theta:+.2f})"

    # -- adapter interface ------------------------------------------------

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
    ) -> str:
        prompt = messages[-1]["content"]
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        wants_confidence = "Confidence:" in system or "Confidence:" in prompt

        if "Complete the following text" in prompt:
            return self._completion_probe(prompt)
        if "Reproduce its original multiple-choice options" in prompt:
            return self._option_recall(prompt)
        if "Reply with only the letter" in prompt or "Reply with only the letter" in system:
            return self._answer_mcq(prompt, wants_confidence)
        rng = _rng(self.seed, "generic", prompt)
        return self._generic_response(prompt, rng)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Cheap deterministic embedding: hashed bag-of-words, l2-normalized.

        Good enough for cosine similarity of near-duplicate texts, which is
        all the offline demo needs.
        """
        dim = 256
        out = np.zeros((len(texts), dim))
        for i, text in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                j = int(hashlib.sha256(tok.encode()).hexdigest()[:8], 16) % dim
                out[i, j] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms

    # -- behaviours --------------------------------------------------------

    def _match_item(self, prompt: str) -> tuple[Item | None, float]:
        """Find the bank item this prompt is about, with a match score."""
        question_part = prompt.split("\n")[0] if "\n" in prompt else prompt
        prompt_toks = _tokens(question_part) or _tokens(prompt)
        best, best_score = None, 0.0
        for item in self.bank:
            item_toks = _tokens(item.question)
            if not item_toks:
                continue
            inter = len(prompt_toks & item_toks)
            union = len(prompt_toks | item_toks)
            score = inter / union if union else 0.0
            if score > best_score:
                best, best_score = item, score
        return best, best_score

    def _answer_mcq(self, prompt: str, wants_confidence: bool) -> str:
        item, score = self._match_item(prompt)
        n_choices = len(re.findall(r"^[A-H]\.\s", prompt, flags=re.MULTILINE)) or 4
        if item is None or score < 0.3:
            rng = _rng(self.seed, "unknown-mcq", prompt)
            idx = int(rng.integers(0, n_choices))
            return self._render_answer(idx, 1.0 / n_choices, wants_confidence, rng)

        p = three_pl(self.theta, item.a, item.b, item.c)
        rng = _rng(self.seed, "mcq", item.id)
        correct = rng.random() < p

        perturbed = score < 0.98
        if perturbed:
            rng_p = _rng(self.seed, "perturb", item.id, prompt)
            if rng_p.random() > self.robustness:
                correct = not correct

        # Where does the correct choice sit in *this* prompt? (option order
        # may have been shuffled by the robustness suite)
        answer_text = item.choices[item.answer_index]
        idx = self._locate_choice(prompt, answer_text, item.answer_index)
        if not correct:
            wrong = [i for i in range(n_choices) if i != idx]
            idx = int(wrong[int(rng.integers(0, len(wrong)))]) if wrong else idx
        return self._render_answer(idx, p, wants_confidence, rng)

    @staticmethod
    def _locate_choice(prompt: str, choice_text: str, fallback: int) -> int:
        for m in re.finditer(r"^([A-H])\.\s(.+)$", prompt, flags=re.MULTILINE):
            if m.group(2).strip() == choice_text.strip():
                return LETTERS.index(m.group(1))
        return fallback

    def _render_answer(
        self, idx: int, p: float, wants_confidence: bool, rng: np.random.Generator
    ) -> str:
        letter = LETTERS[idx]
        if not wants_confidence:
            return letter
        # Verbalized confidence: distort true correctness probability by the
        # calibration skew, plus a little noise.
        conf = float(np.clip(p**self.calibration_skew + rng.normal(0, 0.04), 0.02, 0.99))
        return f"{letter}\nConfidence: {round(conf * 100)}"

    def _completion_probe(self, prompt: str) -> str:
        prefix = prompt.split("\n\n", 1)[-1]
        for item in self.bank:
            if item.question.startswith(prefix.strip()) and len(prefix.strip()) > 20:
                remainder = item.question[len(prefix.strip()):].strip()
                if self.contaminated:
                    return remainder
                # A clean model continues plausibly on-topic but not verbatim:
                # keep only a few content words from the true remainder.
                rng = _rng(self.seed, "completion", item.id)
                words = remainder.split()
                kept = [w for w in words if len(w) > 4 and rng.random() < 0.25]
                filler = ["which", "relates", "to", "the", "topic", "in", "question"]
                return " ".join(kept[:6] + filler)
        return "…and the rest follows naturally from the context above."

    def _option_recall(self, prompt: str) -> str:
        question = prompt.split("\n\n", 1)[-1]
        item, score = self._match_item(question)
        if item is not None and score > 0.5 and self.contaminated:
            return "\n".join(item.choices)
        rng = _rng(self.seed, "recall", prompt)
        fillers = [
            "A plausible but generic first option",
            "Another generic distractor",
            "A third invented alternative",
            "A final made-up choice",
        ]
        rng.shuffle(fillers)
        return "\n".join(fillers)

    def _generic_response(self, prompt: str, rng: np.random.Generator) -> str:
        quality = "high" if self.theta > 0 else "modest"
        n_sentences = 2 + int(rng.integers(0, 3))
        base = (
            f"Here is a {quality}-quality answer addressing the question. "
        )
        return base + " ".join(
            "It considers the key factors and explains the reasoning step by step."
            for _ in range(n_sentences)
        )


class SimulatedJudge(ModelAdapter):
    """A pairwise judge with a hidden quality signal and injectable biases.

    The judge considers the response containing more distinct content tokens
    overlapping the prompt as 'truly better' (accuracy controls how often it
    follows that signal); ``position_bias`` shifts verdicts toward slot A and
    ``verbosity_bias`` toward the longer response.
    """

    def __init__(
        self,
        accuracy: float = 0.9,
        position_bias: float = 0.0,
        verbosity_bias: float = 0.0,
        seed: int = 0,
        name: str = "simulated-judge",
    ):
        self.accuracy = accuracy
        self.position_bias = position_bias
        self.verbosity_bias = verbosity_bias
        self.seed = seed
        self.name = name

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
    ) -> str:
        prompt = messages[-1]["content"]
        a = _between(prompt, "[RESPONSE A]", "[/RESPONSE A]")
        b = _between(prompt, "[RESPONSE B]", "[/RESPONSE B]")
        question = _between(prompt, "[PROMPT]", "[/PROMPT]")
        rng = _rng(self.seed, "judge", prompt, temperature, seed)

        q_toks = _tokens(question)
        signal_a = len(_tokens(a) & q_toks) + 0.01 * len(_tokens(a))
        signal_b = len(_tokens(b) & q_toks) + 0.01 * len(_tokens(b))
        truly_better = "A" if signal_a >= signal_b else "B"

        winner = truly_better if rng.random() < self.accuracy else (
            "B" if truly_better == "A" else "A"
        )
        if rng.random() < self.position_bias:
            winner = "A"
        if rng.random() < self.verbosity_bias:
            winner = "A" if len(a) > len(b) else "B"
        return f'{{"winner": "{winner}", "reason": "simulated verdict"}}'


def _between(text: str, start: str, end: str) -> str:
    try:
        return text.split(start, 1)[1].split(end, 1)[0].strip()
    except IndexError:
        return ""
