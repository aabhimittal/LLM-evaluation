"""Optional bridges to Ragas and TruLens.

Caliper's native RAG instrument is dependency-light and offline-testable. If
you also want the *standard* Ragas / TruLens numbers — for comparison, or
because a reviewer expects them — install the extra::

    pip install "llm-caliper[rag]"

These bridges are thin adapters: they hand your samples and answers to the
external library and return its scores. They are intentionally not imported by
default and are not exercised in CI (they need heavy deps and, usually, an
LLM API key).
"""

from __future__ import annotations

from caliper.rag.types import RagBank


def _require(pkg: str):
    try:
        return __import__(pkg)
    except ImportError as e:  # pragma: no cover - depends on optional extra
        raise ImportError(
            f"{pkg} is not installed. Install the RAG bridge extra with "
            "`pip install \"llm-caliper[rag]\"`."
        ) from e


def evaluate_with_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    references: list[str] | None = None,
):  # pragma: no cover - optional dependency
    """Score (question, answer, contexts) triples with Ragas.

    Returns the Ragas ``EvaluationResult``. Faithfulness and answer/context
    relevancy are computed by Ragas' own LLM-judged metrics — no uncertainty
    intervals, unlike :func:`caliper.rag.evaluate_rag`.
    """
    _require("ragas")
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    data = {"question": questions, "answer": answers, "contexts": contexts}
    if references is not None:
        data["ground_truth"] = references
    dataset = Dataset.from_dict(data)
    return evaluate(
        dataset, metrics=[faithfulness, answer_relevancy, context_precision]
    )


def evaluate_with_trulens(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
):  # pragma: no cover - optional dependency
    """Score triples with TruLens' RAG-triad feedback functions.

    Returns a list of per-sample dicts with groundedness, answer relevance and
    context relevance from ``trulens_eval``.
    """
    _require("trulens_eval")
    from trulens_eval import Feedback
    from trulens_eval.feedback.provider import OpenAI

    provider = OpenAI()
    groundedness = Feedback(provider.groundedness_measure_with_cot_reasons)
    answer_rel = Feedback(provider.relevance)
    context_rel = Feedback(provider.context_relevance)

    out = []
    for q, a, ctx in zip(questions, answers, contexts):
        joined = "\n".join(ctx)
        out.append({
            "groundedness": groundedness(joined, a),
            "answer_relevance": answer_rel(q, a),
            "context_relevance": context_rel(q, joined),
        })
    return out


def bank_to_triples(bank: RagBank) -> tuple[list[str], list[list[str]], list[str]]:
    """Split a :class:`RagBank` into parallel question / contexts / reference lists."""
    questions = [s.question for s in bank.samples]
    contexts = [s.contexts for s in bank.samples]
    references = [s.reference_answer for s in bank.samples]
    return questions, contexts, references
