import numpy as np

from caliper.adapters import SimulatedRAGSubject
from caliper.rag import (
    RagBank,
    RagSample,
    decompose_claims,
    evaluate_faithfulness,
    evaluate_rag,
)
from caliper.rag.faithfulness import parse_support
from caliper.rag.relevance import parse_relevant


def _tiny_bank() -> RagBank:
    # Question and contexts deliberately share topic vocabulary (as in real RAG
    # samples), so a faithful, on-topic answer can reconstruct the question.
    samples = [
        RagSample(
            id=f"t/{i}",
            question=(
                f"How does the reaction mechanism {i} operate under normal "
                f"laboratory conditions?"
            ),
            contexts=[
                f"The reaction mechanism {i} operates through steady predictable "
                f"stages under normal laboratory conditions according to the record.",
                f"A second passage confirms the reaction mechanism {i} was documented "
                f"carefully under standard normal conditions by independent researchers.",
            ],
            reference_answer=f"Reaction mechanism {i} operates in predictable stages.",
        )
        for i in range(12)
    ]
    return RagBank(samples=samples, name="tiny", source="test")


def test_bundled_bank_loads():
    bank = RagBank.bundled()
    assert len(bank) >= 10
    assert all(s.contexts for s in bank.samples)


def test_parse_helpers():
    assert parse_support("SUPPORTED") is True
    assert parse_support("NOT_SUPPORTED") is False
    assert parse_support("not supported by the passage") is False
    assert parse_support("banana") is None
    assert parse_relevant("RELEVANT") is True
    assert parse_relevant("IRRELEVANT") is False


def test_decompose_splits_sentences():
    subject = SimulatedRAGSubject(hallucination_rate=0.0, seed=0)
    answer = "The sky is blue. Water is wet. Fire is hot."
    claims = decompose_claims(subject, answer)
    assert len(claims) == 3


def test_faithful_answer_is_fully_supported():
    bank = _tiny_bank()
    sample = bank.samples[0]
    subject = SimulatedRAGSubject(hallucination_rate=0.0, seed=1)
    answer = subject.ask(
        "Answer the question using only the provided context.\n\n"
        "Context:\n[1] " + sample.contexts[0] + "\n[2] " + sample.contexts[1] + "\n\n"
        f"Question: {sample.question}"
    )
    report = evaluate_faithfulness(subject, answer, sample.contexts, seed=0)
    assert report.n_claims > 0
    assert report.supported_fraction == 1.0
    assert report.unsupported_claims == []


def test_higher_hallucination_lowers_faithfulness():
    bank = _tiny_bank()
    clean = SimulatedRAGSubject(hallucination_rate=0.0, seed=2)
    liar = SimulatedRAGSubject(hallucination_rate=0.6, seed=2)
    r_clean = evaluate_rag(clean, bank, n_samples=12, seed=0, n_boot=200)
    r_liar = evaluate_rag(liar, bank, n_samples=12, seed=0, n_boot=200)
    assert r_clean.faithfulness > r_liar.faithfulness
    assert r_liar.n_unsupported_claims > 0
    assert r_clean.n_unsupported_claims == 0


def test_faithfulness_recovers_injected_rate():
    bank = _tiny_bank()
    rate = 0.4
    subject = SimulatedRAGSubject(hallucination_rate=rate, seed=3)
    report = evaluate_rag(subject, bank, n_samples=12, seed=0, n_boot=200)
    # supported_fraction should recover 1 - hallucination_rate within tolerance
    assert abs(report.faithfulness - (1.0 - rate)) < 0.15


def test_faithfulness_ci_contains_point_estimate():
    bank = _tiny_bank()
    subject = SimulatedRAGSubject(hallucination_rate=0.3, seed=4)
    report = evaluate_rag(subject, bank, n_samples=12, seed=0, n_boot=300)
    lo, hi = report.faithfulness_ci95
    assert lo <= report.faithfulness <= hi


def test_higher_answer_relevance_scores_higher():
    bank = _tiny_bank()
    # hold hallucination fixed at 0 so the relevance signal is clean
    relevant = SimulatedRAGSubject(hallucination_rate=0.0, answer_relevance=0.95, seed=5)
    vague = SimulatedRAGSubject(hallucination_rate=0.0, answer_relevance=0.05, seed=5)
    r_rel = evaluate_rag(relevant, bank, n_samples=12, seed=0, n_boot=200)
    r_vague = evaluate_rag(vague, bank, n_samples=12, seed=0, n_boot=200)
    assert r_rel.answer_relevance > r_vague.answer_relevance


def test_context_precision_recovers_injected_value():
    bank = _tiny_bank()
    subject = SimulatedRAGSubject(hallucination_rate=0.1, context_precision=0.5, seed=6)
    report = evaluate_rag(subject, bank, n_samples=12, seed=0, n_boot=200)
    assert abs(report.context_precision - 0.5) < 0.25


def test_report_serializes_to_json():
    bank = _tiny_bank()
    subject = SimulatedRAGSubject(hallucination_rate=0.2, seed=7)
    report = evaluate_rag(subject, bank, n_samples=6, seed=0, n_boot=100)
    payload = report.to_json()
    assert '"faithfulness"' in payload
    assert isinstance(report.to_dict()["faithfulness_ci95"], list)


def test_verifier_is_self_consistent_on_clean_subject():
    bank = _tiny_bank()
    subject = SimulatedRAGSubject(hallucination_rate=0.0, seed=8)
    report = evaluate_rag(subject, bank, n_samples=8, seed=0, n_boot=100)
    # the token-overlap verifier is deterministic, so agreement should be high
    assert report.mean_verifier_agreement == 1.0
    assert np.isfinite(report.faithfulness)
