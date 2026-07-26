"""Audit the faithfulness verifier — and correct faithfulness for its errors.

Every RAG evaluator (Ragas, TruLens, this one) measures faithfulness with an
LLM verifier. That verifier is itself a fallible instrument: it misses real
support (imperfect **sensitivity**) and hallucinates support that is not there
(imperfect **specificity**). A raw faithfulness score silently inherits both
errors, and the bias does not vanish with more samples — it is systematic.

Caliper measures the verifier against **known-truth controls** and then applies
the classic epidemiological correction for an imperfect test
(Rogan & Gladen, 1978). If a verifier with sensitivity *se* and specificity
*sp* reports apparent faithfulness *p_obs*, the bias-corrected estimate is

    p_true = (p_obs + sp - 1) / (se + sp - 1)

which is exactly unbiased when the controls are representative. When
``se + sp <= 1`` the verifier carries no usable signal and the correction is
undefined — Caliper reports that rather than returning a confident number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from caliper.adapters.base import ModelAdapter
from caliper.rag.faithfulness import parse_support
from caliper.rag.prompts import NLI_SYSTEM, format_nli_verify
from caliper.rag.stats import bootstrap_ci
from caliper.rag.types import RagBank


@dataclass
class VerifierAudit:
    """How good is the fact-checker doing the grading?"""

    n_probes: int
    sensitivity: float                    # P(says SUPPORTED | truly supported)
    sensitivity_ci95: tuple[float, float]
    specificity: float                    # P(says NOT_SUPPORTED | truly unsupported)
    specificity_ci95: tuple[float, float]
    order_flip_rate: float                # verdict changed when passages were reordered
    unparseable_rate: float
    youden_j: float                       # se + sp - 1; <= 0 means no usable signal
    usable: bool
    failed_positive_controls: list[str] = field(default_factory=list)
    failed_negative_controls: list[str] = field(default_factory=list)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if len(s.strip()) > 20]


def _controls(samples: list, i: int, per_sample: int) -> tuple[list[str], list[str]]:
    """Build positive and negative control claims for sample ``i``.

    Positives are sentences from the sample's *own* passages (supported by
    construction). Negatives are sentences from *other* samples' passages —
    true statements about the world that these passages do not entail. Drawing
    several of each per sample keeps the sensitivity/specificity estimates
    tight enough that the bias correction reduces error instead of adding
    noise.
    """
    sample = samples[i]
    positives: list[str] = []
    for passage in sample.contexts:
        positives.extend(_sentences(passage))
    negatives: list[str] = []
    for step in range(1, len(samples)):
        other = samples[(i + step) % len(samples)]
        if other is sample:
            continue
        for passage in other.contexts:
            negatives.extend(_sentences(passage))
        if len(negatives) >= per_sample:
            break
    return positives[:per_sample], negatives[:per_sample]


def _ask_verifier(
    adapter: ModelAdapter, claim: str, contexts: list[str], seed: int
) -> bool | None:
    reply = adapter.ask(
        format_nli_verify(claim, contexts),
        system=NLI_SYSTEM,
        temperature=0.0,
        max_tokens=8,
        seed=seed,
    )
    return parse_support(reply)


def audit_verifier(
    adapter: ModelAdapter,
    bank: RagBank | None = None,
    *,
    n_probes: int = 12,
    controls_per_sample: int = 3,
    seed: int = 0,
    n_boot: int = 500,
) -> VerifierAudit:
    """Estimate the verifier's sensitivity and specificity from known controls.

    Controls are built from the bank itself, so no extra labelled data is
    needed:

    - **positive control** — a sentence lifted verbatim from a sample's own
      context. It is supported by construction; a verifier that says
      NOT_SUPPORTED has failed.
    - **negative control** — a sentence lifted from a *different* sample's
      context. It is a true statement about the world but is *not* entailed by
      these passages, which is exactly the distinction faithfulness rests on.

    Reordering the passages probes a second pathology: a verifier whose verdict
    depends on where in the context window the evidence sits.
    """
    bank = bank if bank is not None else RagBank.bundled()
    rng = np.random.default_rng(seed)
    n = min(n_probes, len(bank.samples))
    idx = rng.choice(len(bank.samples), size=n, replace=False)
    samples = [bank.samples[int(i)] for i in idx]

    pos_hits: list[float] = []
    neg_hits: list[float] = []
    flips: list[float] = []
    unparseable = 0
    total = 0
    failed_pos: list[str] = []
    failed_neg: list[str] = []

    for i, sample in enumerate(samples):
        if not sample.contexts:
            continue
        positives, negatives = _controls(samples, i, controls_per_sample)
        probes = [(c, True) for c in positives] + [(c, False) for c in negatives]

        for j, (claim, truly_supported) in enumerate(probes):
            total += 1
            verdict = _ask_verifier(adapter, claim, sample.contexts, seed + 100 * i + j)
            if verdict is None:
                unparseable += 1
                continue
            if truly_supported:
                pos_hits.append(1.0 if verdict else 0.0)
                if not verdict:
                    failed_pos.append(claim[:160])
            else:
                neg_hits.append(1.0 if not verdict else 0.0)
                if verdict:
                    failed_neg.append(claim[:160])

            # Order sensitivity: same claim, same evidence, passages reversed.
            if len(sample.contexts) > 1:
                flipped = _ask_verifier(
                    adapter, claim, list(reversed(sample.contexts)), seed + 100 * i + j
                )
                if flipped is not None:
                    flips.append(1.0 if flipped != verdict else 0.0)

    sensitivity = float(np.mean(pos_hits)) if pos_hits else 0.0
    specificity = float(np.mean(neg_hits)) if neg_hits else 0.0
    youden = sensitivity + specificity - 1.0
    return VerifierAudit(
        n_probes=total,
        sensitivity=sensitivity,
        sensitivity_ci95=bootstrap_ci(pos_hits, seed=seed, n_boot=n_boot),
        specificity=specificity,
        specificity_ci95=bootstrap_ci(neg_hits, seed=seed + 1, n_boot=n_boot),
        order_flip_rate=float(np.mean(flips)) if flips else 0.0,
        unparseable_rate=(unparseable / total) if total else 0.0,
        youden_j=youden,
        usable=youden > 0.05,
        failed_positive_controls=failed_pos[:10],
        failed_negative_controls=failed_neg[:10],
    )


def correct_prevalence(
    observed: float, sensitivity: float, specificity: float
) -> float | None:
    """Rogan-Gladen correction for a test with known error rates.

    Returns ``None`` when ``sensitivity + specificity <= 1`` — a verifier no
    better than chance cannot be corrected, and pretending otherwise would
    manufacture precision that does not exist.
    """
    denom = sensitivity + specificity - 1.0
    if denom <= 1e-6:
        return None
    corrected = (observed + specificity - 1.0) / denom
    return float(min(max(corrected, 0.0), 1.0))


def corrected_faithfulness(
    observed: float, audit: VerifierAudit
) -> tuple[float | None, tuple[float, float] | None]:
    """Bias-corrected faithfulness, with the correction propagated to the CI.

    The interval is obtained by correcting each endpoint of the verifier's own
    sensitivity/specificity intervals — a conservative envelope that widens as
    the verifier gets less trustworthy, which is the honest behaviour.
    """
    if not audit.usable:
        return None, None
    point = correct_prevalence(observed, audit.sensitivity, audit.specificity)
    if point is None:
        return None, None
    corners = []
    for se in audit.sensitivity_ci95:
        for sp in audit.specificity_ci95:
            value = correct_prevalence(observed, se, sp)
            if value is not None:
                corners.append(value)
    ci = (min(corners), max(corners)) if corners else None
    return point, ci
