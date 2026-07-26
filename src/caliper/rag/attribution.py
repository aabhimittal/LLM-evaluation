"""Was the answer earned by retrieval, or did the model already 'know' it?

A high faithfulness score does not prove the retrieved passages did any work.
A model can recite an answer from parametric memory that happens to agree with
the context and score perfectly — the retrieval pipeline could be switched off
and the benchmark would not notice. That is a silent, expensive failure mode:
you pay for a retriever that is decorative.

This probe answers the counterfactual directly by re-asking the same question
under altered context conditions and comparing the answers:

- **closed book** — no context at all. Similarity to the full-context answer is
  ``parametric_leakage``: how much of the answer the model produced anyway.
- **foreign context** — passages from a *different* question. A grounded model
  should change its answer completely; ``context_sensitivity`` is
  ``1 - similarity``. Near zero means the model ignores what it retrieves.
- **polluted context** — the real passages plus an irrelevant one.
  ``distractor_stability`` is the similarity to the clean answer: does one bad
  retrieval hit derail the response?

This is the retrieval analogue of Caliper's contamination probes: same spirit
(ask what the score would be if the thing under test were removed), same
caveat (these are probes, not proof).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from caliper.adapters.base import ModelAdapter
from caliper.rag.prompts import (
    RAG_ANSWER_SYSTEM,
    format_closed_book,
    format_rag_answer,
)
from caliper.rag.stats import bootstrap_ci, cosine
from caliper.rag.types import RagBank


@dataclass
class AttributionReport:
    n_samples: int
    parametric_leakage: float                     # sim(full, closed-book); high = not earned
    parametric_leakage_ci95: tuple[float, float]
    context_sensitivity: float                    # 1 - sim(full, foreign); high = uses context
    context_sensitivity_ci95: tuple[float, float]
    distractor_stability: float                   # sim(full, polluted); high = robust
    distractor_stability_ci95: tuple[float, float]
    per_sample: list[dict] = field(default_factory=list)

    @property
    def earned_by_retrieval(self) -> float:
        """Headline 0-1 score: high when the answer genuinely depends on retrieval."""
        return float(
            np.clip(0.5 * self.context_sensitivity + 0.5 * (1.0 - self.parametric_leakage),
                    0.0, 1.0)
        )


def _answer(adapter: ModelAdapter, question: str, contexts: list[str], seed: int) -> str:
    prompt = (
        format_closed_book(question)
        if not contexts
        else format_rag_answer(question, contexts)
    )
    return adapter.ask(
        prompt, system=RAG_ANSWER_SYSTEM, temperature=0.0, max_tokens=400, seed=seed
    )


def probe_attribution(
    adapter: ModelAdapter,
    bank: RagBank | None = None,
    *,
    n_samples: int = 10,
    seed: int = 0,
    n_boot: int = 500,
) -> AttributionReport:
    """Re-answer each question under ablated / swapped / polluted context."""
    bank = bank if bank is not None else RagBank.bundled()
    rng = np.random.default_rng(seed)
    n = min(n_samples, len(bank.samples))
    idx = rng.choice(len(bank.samples), size=n, replace=False)
    samples = [bank.samples[int(i)] for i in idx]

    leakage: list[float] = []
    sensitivity: list[float] = []
    stability: list[float] = []
    per_sample: list[dict] = []

    for i, sample in enumerate(samples):
        other = samples[(i + 1) % len(samples)]
        if other is sample or not other.contexts or not sample.contexts:
            continue

        full = _answer(adapter, sample.question, sample.contexts, seed + i)
        closed = _answer(adapter, sample.question, [], seed + i)
        foreign = _answer(adapter, sample.question, other.contexts, seed + i)
        polluted = _answer(
            adapter, sample.question, [*sample.contexts, other.contexts[0]], seed + i
        )

        try:
            vecs = adapter.embed([full, closed, foreign, polluted])
        except NotImplementedError:
            continue
        def _clamp(x: float) -> float:
            return float(min(max(x, 0.0), 1.0))

        sim_closed = _clamp(cosine(vecs[0], vecs[1]))
        sim_foreign = _clamp(cosine(vecs[0], vecs[2]))
        sim_polluted = _clamp(cosine(vecs[0], vecs[3]))

        leakage.append(sim_closed)
        sensitivity.append(1.0 - sim_foreign)
        stability.append(sim_polluted)
        per_sample.append({
            "sample_id": sample.id,
            "parametric_leakage": sim_closed,
            "context_sensitivity": 1.0 - sim_foreign,
            "distractor_stability": sim_polluted,
        })

    return AttributionReport(
        n_samples=len(per_sample),
        parametric_leakage=float(np.mean(leakage)) if leakage else 0.0,
        parametric_leakage_ci95=bootstrap_ci(leakage, seed=seed, n_boot=n_boot),
        context_sensitivity=float(np.mean(sensitivity)) if sensitivity else 0.0,
        context_sensitivity_ci95=bootstrap_ci(sensitivity, seed=seed + 1, n_boot=n_boot),
        distractor_stability=float(np.mean(stability)) if stability else 0.0,
        distractor_stability_ci95=bootstrap_ci(stability, seed=seed + 2, n_boot=n_boot),
        per_sample=per_sample,
    )
