"""RAG grounding suite: faithfulness + relevance across a bank, with CIs.

For each sample the model answers from the retrieved context; we then measure
faithfulness (claim-level, with localized hallucinations), answer relevance
and context precision. Aggregates carry a **bootstrap CI over samples** — the
Caliper house rule that no number ships without its uncertainty.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np

from caliper.adapters.base import ModelAdapter
from caliper.rag.faithfulness import evaluate_faithfulness
from caliper.rag.prompts import RAG_ANSWER_SYSTEM, format_rag_answer
from caliper.rag.relevance import answer_relevance, context_precision
from caliper.rag.types import RagBank


def _bootstrap_ci(values: list[float], seed: int, n_boot: int) -> tuple[float, float]:
    """95% bootstrap percentile interval over a list of per-sample values."""
    if not values:
        return (0.0, 0.0)
    arr = np.asarray(values, dtype=float)
    boot = np.random.default_rng(seed)
    means = [
        float(np.mean(arr[boot.integers(0, len(arr), size=len(arr))]))
        for _ in range(n_boot)
    ]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


@dataclass
class RagReport:
    n_samples: int
    faithfulness: float
    faithfulness_ci95: tuple[float, float]
    answer_relevance: float
    answer_relevance_ci95: tuple[float, float]
    context_precision: float
    context_precision_ci95: tuple[float, float]
    mean_verifier_agreement: float
    n_claims: int
    n_unsupported_claims: int
    unsupported_examples: list[dict] = field(default_factory=list)
    per_sample: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["faithfulness_ci95"] = list(self.faithfulness_ci95)
        d["answer_relevance_ci95"] = list(self.answer_relevance_ci95)
        d["context_precision_ci95"] = list(self.context_precision_ci95)
        return d

    def to_json(self, indent: int = 2) -> str:
        import json

        return json.dumps(self.to_dict(), indent=indent)


def evaluate_rag(
    adapter: ModelAdapter,
    bank: RagBank | None = None,
    *,
    n_samples: int = 20,
    n_verify_samples: int = 3,
    seed: int = 0,
    n_boot: int = 500,
    progress: Callable[[str], None] | None = None,
) -> RagReport:
    """Score a model's grounding on a RAG bank."""
    bank = bank if bank is not None else RagBank.bundled()
    say = progress or (lambda _msg: None)
    rng = np.random.default_rng(seed)
    n = min(n_samples, len(bank.samples))
    idx = rng.choice(len(bank.samples), size=n, replace=False)
    samples = [bank.samples[int(i)] for i in idx]

    faiths: list[float] = []
    rels: list[float] = []
    precs: list[float] = []
    agreements: list[float] = []
    total_claims = 0
    unsupported_examples: list[dict] = []
    per_sample: list[dict] = []

    for i, sample in enumerate(samples):
        say(f"sample {i + 1}/{n}: {sample.id}")
        answer = adapter.ask(
            format_rag_answer(sample.question, sample.contexts),
            system=RAG_ANSWER_SYSTEM,
            temperature=0.0,
            max_tokens=400,
            seed=seed + i,
        )
        faith = evaluate_faithfulness(
            adapter, answer, sample.contexts,
            n_samples=n_verify_samples, seed=seed + i, n_boot=n_boot,
        )
        rel = answer_relevance(
            adapter, sample.question, answer, seed=seed + i, n_boot=n_boot
        )
        prec = context_precision(adapter, sample.question, sample.contexts, seed=seed + i)

        faiths.append(faith.supported_fraction)
        rels.append(rel.score)
        precs.append(prec.score)
        if faith.verdicts:
            agreements.append(faith.mean_agreement)
        total_claims += faith.n_claims
        for claim in faith.unsupported_claims:
            if len(unsupported_examples) < 25:
                unsupported_examples.append({"sample_id": sample.id, "claim": claim})
        per_sample.append({
            "sample_id": sample.id,
            "faithfulness": faith.supported_fraction,
            "n_claims": faith.n_claims,
            "n_unsupported": len(faith.unsupported_claims),
            "answer_relevance": rel.score,
            "context_precision": prec.score,
        })

    return RagReport(
        n_samples=n,
        faithfulness=float(np.mean(faiths)) if faiths else 0.0,
        faithfulness_ci95=_bootstrap_ci(faiths, seed, n_boot),
        answer_relevance=float(np.mean(rels)) if rels else 0.0,
        answer_relevance_ci95=_bootstrap_ci(rels, seed + 1, n_boot),
        context_precision=float(np.mean(precs)) if precs else 0.0,
        context_precision_ci95=_bootstrap_ci(precs, seed + 2, n_boot),
        mean_verifier_agreement=float(np.mean(agreements)) if agreements else 1.0,
        n_claims=total_claims,
        n_unsupported_claims=sum(p["n_unsupported"] for p in per_sample),
        unsupported_examples=unsupported_examples,
        per_sample=per_sample,
    )
