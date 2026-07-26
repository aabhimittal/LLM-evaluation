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
from caliper.rag.attribution import AttributionReport, probe_attribution
from caliper.rag.audit import VerifierAudit, audit_verifier, corrected_faithfulness
from caliper.rag.faithfulness import evaluate_faithfulness
from caliper.rag.prompts import RAG_ANSWER_SYSTEM, format_rag_answer
from caliper.rag.relevance import answer_relevance, context_precision
from caliper.rag.stats import bootstrap_ci
from caliper.rag.types import RagBank


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
    # Optional probes (see caliper.rag.audit / caliper.rag.attribution)
    verifier: VerifierAudit | None = None
    faithfulness_corrected: float | None = None
    faithfulness_corrected_ci95: tuple[float, float] | None = None
    attribution: AttributionReport | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["faithfulness_ci95"] = list(self.faithfulness_ci95)
        d["answer_relevance_ci95"] = list(self.answer_relevance_ci95)
        d["context_precision_ci95"] = list(self.context_precision_ci95)
        if self.faithfulness_corrected_ci95 is not None:
            d["faithfulness_corrected_ci95"] = list(self.faithfulness_corrected_ci95)
        if self.verifier is not None:
            d["verifier"]["sensitivity_ci95"] = list(self.verifier.sensitivity_ci95)
            d["verifier"]["specificity_ci95"] = list(self.verifier.specificity_ci95)
        if self.attribution is not None:
            for key in ("parametric_leakage_ci95", "context_sensitivity_ci95",
                        "distractor_stability_ci95"):
                d["attribution"][key] = list(getattr(self.attribution, key))
            d["attribution"]["earned_by_retrieval"] = self.attribution.earned_by_retrieval
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
    with_audit: bool = False,
    with_attribution: bool = False,
    progress: Callable[[str], None] | None = None,
) -> RagReport:
    """Score a model's grounding on a RAG bank.

    ``with_audit`` measures the verifier's own sensitivity/specificity against
    known-truth controls and adds a **bias-corrected** faithfulness estimate.
    ``with_attribution`` re-answers each question with the context ablated,
    swapped and polluted, to test whether the answer was earned by retrieval.
    Both add model calls, so they are opt-in.
    """
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

    faithfulness = float(np.mean(faiths)) if faiths else 0.0

    audit = None
    corrected = corrected_ci = None
    if with_audit:
        say("auditing the verifier against known-truth controls…")
        audit = audit_verifier(adapter, bank, n_probes=min(n, 12), seed=seed,
                               n_boot=n_boot)
        corrected, corrected_ci = corrected_faithfulness(faithfulness, audit)

    attribution = None
    if with_attribution:
        say("attribution probe: ablating, swapping and polluting the context…")
        attribution = probe_attribution(adapter, bank, n_samples=min(n, 10),
                                        seed=seed, n_boot=n_boot)

    return RagReport(
        n_samples=n,
        faithfulness=faithfulness,
        faithfulness_ci95=bootstrap_ci(faiths, seed=seed, n_boot=n_boot),
        answer_relevance=float(np.mean(rels)) if rels else 0.0,
        answer_relevance_ci95=bootstrap_ci(rels, seed=seed + 1, n_boot=n_boot),
        context_precision=float(np.mean(precs)) if precs else 0.0,
        context_precision_ci95=bootstrap_ci(precs, seed=seed + 2, n_boot=n_boot),
        mean_verifier_agreement=float(np.mean(agreements)) if agreements else 1.0,
        n_claims=total_claims,
        n_unsupported_claims=sum(p["n_unsupported"] for p in per_sample),
        unsupported_examples=unsupported_examples,
        per_sample=per_sample,
        verifier=audit,
        faithfulness_corrected=corrected,
        faithfulness_corrected_ci95=corrected_ci,
        attribution=attribution,
    )
