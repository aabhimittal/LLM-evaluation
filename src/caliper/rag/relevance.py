"""Answer relevance and context precision.

- **Answer relevance** (question round-trip): a relevant answer should let you
  reconstruct the original question. We ask the model to generate questions
  *from its answer*, embed them and the original question, and report the mean
  cosine similarity — with a CI over the generated questions.
- **Context precision**: what fraction of the retrieved passages the model
  judges genuinely useful for the question. Retrieving junk alongside the
  answer wastes the context window and invites distraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from caliper.adapters.base import ModelAdapter
from caliper.rag.prompts import format_context_relevance, format_question_generation


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


@dataclass
class AnswerRelevance:
    score: float
    ci95: tuple[float, float]
    generated_questions: list[str] = field(default_factory=list)


def answer_relevance(
    adapter: ModelAdapter,
    question: str,
    answer: str,
    *,
    n_questions: int = 3,
    seed: int = 0,
    n_boot: int = 500,
) -> AnswerRelevance:
    """Mean cosine between the original question and answer-derived questions.

    Requires the adapter to support ``embed`` (HF feature-extraction, or the
    simulated hashed embedding). Falls back to a score of 0 if unavailable.
    """
    gen = [
        adapter.ask(
            format_question_generation(answer),
            temperature=0.7,
            max_tokens=48,
            seed=seed + k,
        ).strip()
        for k in range(n_questions)
    ]
    gen = [g for g in gen if g]
    if not gen:
        return AnswerRelevance(score=0.0, ci95=(0.0, 0.0))
    try:
        vecs = adapter.embed([question, *gen])
    except NotImplementedError:
        return AnswerRelevance(score=0.0, ci95=(0.0, 0.0), generated_questions=gen)
    q_vec = vecs[0]
    sims = [_cosine(q_vec, vecs[i + 1]) for i in range(len(gen))]
    arr = np.asarray(sims, dtype=float)
    boot = np.random.default_rng(seed)
    means = [
        float(np.mean(arr[boot.integers(0, len(arr), size=len(arr))]))
        for _ in range(n_boot)
    ]
    ci = (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))
    return AnswerRelevance(score=float(arr.mean()), ci95=ci, generated_questions=gen)


def parse_relevant(text: str) -> bool | None:
    upper = text.upper()
    if "IRRELEVANT" in upper or upper.strip().startswith("NO"):
        return False
    if "RELEVANT" in upper or upper.strip().startswith("YES"):
        return True
    return None


@dataclass
class ContextPrecision:
    score: float                     # fraction of passages judged relevant
    per_passage: list[bool] = field(default_factory=list)


def context_precision(
    adapter: ModelAdapter,
    question: str,
    contexts: list[str],
    *,
    seed: int = 0,
) -> ContextPrecision:
    """Fraction of retrieved passages the model judges relevant to the question."""
    flags: list[bool] = []
    for i, passage in enumerate(contexts):
        reply = adapter.ask(
            format_context_relevance(question, passage),
            system="Reply with a single word: RELEVANT or IRRELEVANT.",
            temperature=0.0,
            max_tokens=8,
            seed=seed + i,
        )
        verdict = parse_relevant(reply)
        flags.append(bool(verdict))
    score = float(np.mean([1.0 if f else 0.0 for f in flags])) if flags else 0.0
    return ContextPrecision(score=score, per_passage=flags)
