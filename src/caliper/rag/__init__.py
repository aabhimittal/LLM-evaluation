"""RAG grounding evaluation: faithfulness and relevance, with uncertainty.

Unlike Ragas / TruLens, which report a single faithfulness/relevance number,
this instrument decomposes the answer into claims, verifies each against the
retrieved context, and ships every metric with a bootstrap confidence interval
plus the list of claims that were *not* supported (localized hallucinations).
"""

from caliper.rag.attribution import AttributionReport, probe_attribution
from caliper.rag.audit import (
    VerifierAudit,
    audit_verifier,
    correct_prevalence,
    corrected_faithfulness,
)
from caliper.rag.faithfulness import (
    ClaimVerdict,
    FaithfulnessReport,
    decompose_claims,
    evaluate_faithfulness,
    verify_claim,
)
from caliper.rag.relevance import (
    AnswerRelevance,
    ContextPrecision,
    answer_relevance,
    context_precision,
)
from caliper.rag.suite import RagReport, evaluate_rag
from caliper.rag.types import RagBank, RagSample

__all__ = [
    "AnswerRelevance",
    "AttributionReport",
    "ClaimVerdict",
    "ContextPrecision",
    "FaithfulnessReport",
    "RagBank",
    "RagReport",
    "RagSample",
    "VerifierAudit",
    "answer_relevance",
    "audit_verifier",
    "context_precision",
    "correct_prevalence",
    "corrected_faithfulness",
    "decompose_claims",
    "evaluate_faithfulness",
    "evaluate_rag",
    "probe_attribution",
    "verify_claim",
]
