"""Faithfulness: is every claim in the answer supported by the context?

Standard tools (Ragas, TruLens) report a single faithfulness number from one
LLM-judge pass. Caliper instead:

- decomposes the answer into **atomic claims**;
- verifies each claim against the context with the judge sampled ``n_samples``
  times, recording **agreement** (self-consistency of the verifier);
- reports the supported fraction **with a bootstrap CI over claims** and, most
  usefully, the **list of unsupported claims** — hallucinations localized to
  the exact sentence, not hidden inside an aggregate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from caliper.adapters.base import ModelAdapter
from caliper.rag.prompts import (
    NLI_SYSTEM,
    format_claim_decomposition,
    format_nli_verify,
)

_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def decompose_claims(adapter: ModelAdapter, answer: str, *, max_claims: int = 12) -> list[str]:
    """Split an answer into atomic factual claims (one per line)."""
    raw = adapter.ask(
        format_claim_decomposition(answer),
        system="Return only the claims, one per line.",
        temperature=0.0,
        max_tokens=400,
    )
    claims = []
    for line in raw.splitlines():
        text = _BULLET.sub("", line).strip()
        if len(text) >= 3:
            claims.append(text)
    return claims[:max_claims]


def parse_support(text: str) -> bool | None:
    """Parse a SUPPORTED / NOT_SUPPORTED verdict. None if unparseable."""
    upper = text.upper()
    not_sup = "NOT_SUPPORTED" in upper or "NOT SUPPORTED" in upper or "UNSUPPORTED" in upper
    if not_sup:
        return False
    if "SUPPORTED" in upper or upper.strip().startswith("YES"):
        return True
    return None


@dataclass
class ClaimVerdict:
    claim: str
    supported: bool
    agreement: float           # fraction of samples agreeing with the majority
    votes: list[bool] = field(default_factory=list)


def verify_claim(
    adapter: ModelAdapter,
    claim: str,
    contexts: list[str],
    *,
    n_samples: int = 3,
    seed: int = 0,
) -> ClaimVerdict:
    """Ask the verifier ``n_samples`` times and take the majority verdict.

    Sampling (at temperature > 0) exposes an *unreliable* verifier: a claim on
    which the judge cannot make up its mind shows low ``agreement``.
    """
    prompt = format_nli_verify(claim, contexts)
    votes: list[bool] = []
    for k in range(n_samples):
        reply = adapter.ask(
            prompt, system=NLI_SYSTEM, temperature=0.5, max_tokens=8, seed=seed + k
        )
        verdict = parse_support(reply)
        if verdict is not None:
            votes.append(verdict)
    if not votes:
        return ClaimVerdict(claim=claim, supported=False, agreement=0.0, votes=[])
    supported = sum(votes) >= len(votes) / 2
    agreement = sum(v == supported for v in votes) / len(votes)
    return ClaimVerdict(claim=claim, supported=supported, agreement=agreement, votes=votes)


@dataclass
class FaithfulnessReport:
    n_claims: int
    supported_fraction: float
    ci95: tuple[float, float]
    unsupported_claims: list[str] = field(default_factory=list)
    mean_agreement: float = 1.0
    verdicts: list[ClaimVerdict] = field(default_factory=list)


def _bootstrap_fraction_ci(flags: list[float], seed: int, n_boot: int) -> tuple[float, float]:
    if not flags:
        return (0.0, 0.0)
    arr = np.asarray(flags, dtype=float)
    boot = np.random.default_rng(seed)
    means = [
        float(np.mean(arr[boot.integers(0, len(arr), size=len(arr))]))
        for _ in range(n_boot)
    ]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def evaluate_faithfulness(
    adapter: ModelAdapter,
    answer: str,
    contexts: list[str],
    *,
    n_samples: int = 3,
    seed: int = 0,
    n_boot: int = 500,
) -> FaithfulnessReport:
    """Decompose ``answer``, verify every claim against ``contexts``."""
    claims = decompose_claims(adapter, answer)
    verdicts = [
        verify_claim(adapter, claim, contexts, n_samples=n_samples, seed=seed + i)
        for i, claim in enumerate(claims)
    ]
    flags = [1.0 if v.supported else 0.0 for v in verdicts]
    fraction = float(np.mean(flags)) if flags else 0.0
    ci = _bootstrap_fraction_ci(flags, seed, n_boot)
    unsupported = [v.claim for v in verdicts if not v.supported]
    mean_agreement = float(np.mean([v.agreement for v in verdicts])) if verdicts else 1.0
    return FaithfulnessReport(
        n_claims=len(claims),
        supported_fraction=fraction,
        ci95=ci,
        unsupported_claims=unsupported,
        mean_agreement=mean_agreement,
        verdicts=verdicts,
    )
