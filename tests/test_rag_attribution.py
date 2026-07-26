"""Was the answer earned by retrieval? Ground-truth tests for the probe.

``context_reliance`` controls how much of the simulated model's answer is read
off the retrieved passages rather than recalled from parametric memory; the
attribution probe has to detect that difference without being told.
"""

from caliper.adapters import SimulatedRAGSubject
from caliper.rag import RagBank, evaluate_rag, probe_attribution


def _bank() -> RagBank:
    return RagBank.bundled()


def test_grounded_model_shows_high_context_sensitivity():
    subject = SimulatedRAGSubject(hallucination_rate=0.0, context_reliance=1.0, seed=1)
    report = probe_attribution(subject, _bank(), n_samples=8, seed=0, n_boot=200)
    assert report.context_sensitivity > 0.5
    assert report.parametric_leakage < 0.5
    assert report.earned_by_retrieval > 0.5


def test_model_ignoring_context_is_caught():
    """A model answering from memory scores near zero however fluent it is."""
    subject = SimulatedRAGSubject(hallucination_rate=0.0, context_reliance=0.0, seed=1)
    report = probe_attribution(subject, _bank(), n_samples=8, seed=0, n_boot=200)
    assert report.parametric_leakage > 0.95      # closed-book answer is identical
    assert report.context_sensitivity < 0.05     # swapping the context changes nothing
    assert report.earned_by_retrieval < 0.1


def test_catches_the_faithful_but_unearned_answer():
    """The failure mode faithfulness alone cannot see.

    A model that recalls the correct answer from memory produces claims the
    passages *do* support, so faithfulness stays high and a Ragas/TruLens-style
    score would call it excellent. Only the attribution probe reveals that the
    retriever contributed nothing.
    """
    bank = _bank()
    subject = SimulatedRAGSubject(
        hallucination_rate=0.0, context_reliance=0.0, bank=bank, seed=1
    )
    report = evaluate_rag(subject, bank, n_samples=10, seed=0, n_boot=200,
                          with_attribution=True)
    assert report.faithfulness > 0.8              # looks great by the usual metric
    assert report.attribution.earned_by_retrieval < 0.1   # but nothing was earned
    assert report.attribution.parametric_leakage > 0.95


def test_attribution_is_monotone_in_context_reliance():
    bank = _bank()
    scores = []
    for reliance in (0.0, 0.5, 1.0):
        subject = SimulatedRAGSubject(
            hallucination_rate=0.0, context_reliance=reliance, seed=2
        )
        report = probe_attribution(subject, bank, n_samples=8, seed=0, n_boot=100)
        scores.append(report.earned_by_retrieval)
    assert scores[0] <= scores[1] <= scores[2]
    assert scores[2] > scores[0]


def test_intervals_bracket_point_estimates():
    subject = SimulatedRAGSubject(hallucination_rate=0.0, context_reliance=1.0, seed=3)
    report = probe_attribution(subject, _bank(), n_samples=8, seed=0, n_boot=300)
    for value, (lo, hi) in (
        (report.parametric_leakage, report.parametric_leakage_ci95),
        (report.context_sensitivity, report.context_sensitivity_ci95),
        (report.distractor_stability, report.distractor_stability_ci95),
    ):
        assert lo <= value <= hi


def test_scores_stay_in_unit_range():
    subject = SimulatedRAGSubject(hallucination_rate=0.3, context_reliance=0.6, seed=4)
    report = probe_attribution(subject, _bank(), n_samples=6, seed=0, n_boot=100)
    for value in (report.parametric_leakage, report.context_sensitivity,
                  report.distractor_stability, report.earned_by_retrieval):
        assert 0.0 <= value <= 1.0


def test_attribution_wires_into_suite_and_serializes():
    subject = SimulatedRAGSubject(hallucination_rate=0.2, context_reliance=1.0, seed=5)
    report = evaluate_rag(subject, _bank(), n_samples=6, seed=0,
                          n_boot=100, with_attribution=True)
    assert report.attribution is not None
    payload = report.to_dict()
    assert isinstance(payload["attribution"]["context_sensitivity_ci95"], list)
    assert 0.0 <= payload["attribution"]["earned_by_retrieval"] <= 1.0


def test_attribution_is_opt_in():
    subject = SimulatedRAGSubject(hallucination_rate=0.2, seed=6)
    report = evaluate_rag(subject, _bank(), n_samples=4, seed=0, n_boot=50)
    assert report.attribution is None
