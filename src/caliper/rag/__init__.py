"""RAG grounding evaluation: faithfulness and relevance, with uncertainty.

Unlike Ragas / TruLens, which report a single faithfulness/relevance number,
this instrument decomposes the answer into claims, verifies each against the
retrieved context, and ships every metric with a bootstrap confidence interval
plus the list of claims that were *not* supported (localized hallucinations).
"""

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
    "ClaimVerdict",
    "ContextPrecision",
    "FaithfulnessReport",
    "RagBank",
    "RagReport",
    "RagSample",
    "answer_relevance",
    "context_precision",
    "decompose_claims",
    "evaluate_faithfulness",
    "evaluate_rag",
    "verify_claim",
]
