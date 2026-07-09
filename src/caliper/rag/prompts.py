"""Canonical RAG prompt templates.

The evaluator and the simulated backend share these templates, so the
simulator can recognize what it is being asked (answer / decompose / verify /
generate-question / rate-context) with no network access — the same trick the
MCQ prompts in :mod:`caliper.prompts` use. Each template embeds a stable
**marker phrase** that :class:`caliper.adapters.simulate.SimulatedRAGSubject`
keys off.
"""

from __future__ import annotations

RAG_ANSWER_SYSTEM = (
    "You are a retrieval-augmented assistant. Answer the question using ONLY "
    "the provided context. Write the answer as a few short factual sentences."
)

NLI_SYSTEM = (
    "You are a strict fact-checker. Decide whether the context supports the "
    "claim. Reply with a single word: SUPPORTED or NOT_SUPPORTED."
)


def _join_contexts(contexts: list[str]) -> str:
    return "\n".join(f"[{i + 1}] {c.strip()}" for i, c in enumerate(contexts))


def format_rag_answer(question: str, contexts: list[str]) -> str:
    return (
        "Answer the question using only the provided context.\n\n"
        f"Context:\n{_join_contexts(contexts)}\n\n"
        f"Question: {question}"
    )


def format_claim_decomposition(answer: str) -> str:
    return (
        "Break the following answer into atomic factual claims, one per line. "
        "Each claim must be a single self-contained statement.\n\n"
        f"Answer:\n{answer}"
    )


def format_nli_verify(claim: str, contexts: list[str]) -> str:
    return (
        "Reply SUPPORTED or NOT_SUPPORTED: does the context entail the claim?\n\n"
        f"Context:\n{_join_contexts(contexts)}\n\n"
        f"Claim: {claim}"
    )


def format_question_generation(answer: str) -> str:
    return (
        "Generate a question that this answer would answer. Output only the "
        "question.\n\n"
        f"Answer: {answer}"
    )


def format_context_relevance(question: str, passage: str) -> str:
    return (
        "Reply RELEVANT or IRRELEVANT: is this passage useful for answering "
        "the question?\n\n"
        f"Question: {question}\n\n"
        f"Passage: {passage}"
    )
