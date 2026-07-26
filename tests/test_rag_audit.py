"""The verifier is an instrument too — these tests check that it gets measured.

Each test injects known verifier error rates into the simulated fact-checker
and asserts that the audit recovers them, and that the Rogan-Gladen correction
removes the bias they cause.
"""

import numpy as np

from caliper.adapters import SimulatedRAGSubject
from caliper.rag import RagBank, audit_verifier, evaluate_rag
from caliper.rag.audit import correct_prevalence, corrected_faithfulness


def _bank() -> RagBank:
    # The bundled bank has topically distinct samples, which is what makes a
    # sentence from another sample a valid negative control.
    return RagBank.bundled()


def test_perfect_verifier_audits_clean():
    subject = SimulatedRAGSubject(hallucination_rate=0.2, seed=0)
    audit = audit_verifier(subject, _bank(), n_probes=8, seed=0, n_boot=100)
    assert audit.sensitivity == 1.0
    assert audit.specificity == 1.0
    assert audit.youden_j == 1.0
    assert audit.usable
    assert audit.unparseable_rate == 0.0
    assert audit.failed_positive_controls == []
    assert audit.failed_negative_controls == []


def test_audit_recovers_injected_error_rates():
    subject = SimulatedRAGSubject(
        hallucination_rate=0.3, verifier_sensitivity=0.8,
        verifier_specificity=0.9, seed=1,
    )
    audit = audit_verifier(subject, _bank(), n_probes=12, seed=0, n_boot=200)
    assert abs(audit.sensitivity - 0.8) < 0.15
    assert abs(audit.specificity - 0.9) < 0.15
    lo, hi = audit.sensitivity_ci95
    assert lo <= audit.sensitivity <= hi


def test_audit_surfaces_failed_controls():
    subject = SimulatedRAGSubject(
        hallucination_rate=0.0, verifier_sensitivity=0.3,
        verifier_specificity=0.3, seed=2,
    )
    audit = audit_verifier(subject, _bank(), n_probes=8, seed=0, n_boot=100)
    # A bad verifier misses real support and rubber-stamps foreign claims.
    assert audit.failed_positive_controls
    assert audit.failed_negative_controls
    assert not audit.usable  # se + sp <= 1: no usable signal


def test_correct_prevalence_closed_form():
    # true p = 0.7, se = 0.8, sp = 0.9 -> apparent = 0.7*0.8 + 0.3*0.1 = 0.59
    assert abs(correct_prevalence(0.59, 0.8, 0.9) - 0.7) < 1e-9
    # a perfect test needs no correction
    assert abs(correct_prevalence(0.42, 1.0, 1.0) - 0.42) < 1e-9
    # results stay inside [0, 1]
    assert correct_prevalence(0.01, 0.8, 0.9) == 0.0


def test_correction_undefined_for_coin_flip_verifier():
    assert correct_prevalence(0.5, 0.5, 0.5) is None
    assert correct_prevalence(0.5, 0.3, 0.4) is None


def test_corrected_faithfulness_skips_unusable_verifier():
    subject = SimulatedRAGSubject(
        verifier_sensitivity=0.4, verifier_specificity=0.4, seed=3
    )
    audit = audit_verifier(subject, _bank(), n_probes=8, seed=0, n_boot=100)
    point, ci = corrected_faithfulness(0.6, audit)
    assert point is None and ci is None


def test_bias_correction_beats_raw_estimate():
    """The headline claim: correcting for verifier error reduces the error.

    A verifier that rubber-stamps (high sensitivity, poor specificity) inflates
    faithfulness badly. Averaged over seeds, the corrected estimate must land
    closer to the truth than the raw one.
    """
    bank = _bank()
    true_faithfulness = 0.5
    raw_errors, corrected_errors = [], []
    for seed in range(6):
        subject = SimulatedRAGSubject(
            hallucination_rate=1.0 - true_faithfulness,
            verifier_sensitivity=0.95, verifier_specificity=0.55, seed=seed,
        )
        report = evaluate_rag(subject, bank, n_samples=12, seed=seed,
                              n_boot=100, with_audit=True)
        raw_errors.append(abs(report.faithfulness - true_faithfulness))
        assert report.faithfulness_corrected is not None
        corrected_errors.append(
            abs(report.faithfulness_corrected - true_faithfulness)
        )
    assert np.mean(corrected_errors) < np.mean(raw_errors)


def test_audit_fields_serialize():
    subject = SimulatedRAGSubject(
        hallucination_rate=0.2, verifier_sensitivity=0.9,
        verifier_specificity=0.9, seed=4,
    )
    report = evaluate_rag(subject, _bank(), n_samples=6, seed=0,
                          n_boot=100, with_audit=True)
    payload = report.to_dict()
    assert isinstance(payload["verifier"]["sensitivity_ci95"], list)
    assert isinstance(payload["faithfulness_corrected_ci95"], list)
    assert '"sensitivity"' in report.to_json()


def test_audit_is_opt_in():
    subject = SimulatedRAGSubject(hallucination_rate=0.2, seed=5)
    report = evaluate_rag(subject, _bank(), n_samples=4, seed=0, n_boot=50)
    assert report.verifier is None
    assert report.faithfulness_corrected is None
