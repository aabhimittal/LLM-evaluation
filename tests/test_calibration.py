import numpy as np

from caliper.adapters import SimulatedSubject
from caliper.calibration import ece, evaluate_calibration, risk_coverage


def test_ece_perfect_and_worst():
    conf = np.array([0.05, 0.15, 0.85, 0.95] * 25)
    correct_perfect = (np.random.default_rng(0).random(100) < conf).astype(float)
    value, bins = ece(conf, correct_perfect)
    assert value < 0.2
    correct_inverted = 1.0 - np.round(conf)
    worst, _ = ece(conf, correct_inverted)
    assert worst > 0.6
    assert all(b["n"] > 0 for b in bins)


def test_risk_coverage_known_case():
    # Confident answers correct, unconfident wrong -> risk rises with coverage.
    conf = np.array([0.9, 0.8, 0.2, 0.1])
    correct = np.array([1.0, 1.0, 0.0, 0.0])
    aurc, points = risk_coverage(conf, correct)
    assert points[0]["risk"] == 0.0
    assert points[-1]["risk"] == 0.5
    assert 0.0 < aurc < 0.5


def test_overconfident_model_has_higher_ece(small_bank):
    calibrated = SimulatedSubject(theta=0.0, bank=small_bank, calibration_skew=1.0, seed=6)
    overconfident = SimulatedSubject(theta=-1.0, bank=small_bank, calibration_skew=0.2, seed=6)
    r_cal = evaluate_calibration(calibrated, small_bank, n_items=40, seed=0)
    r_over = evaluate_calibration(overconfident, small_bank, n_items=40, seed=0)
    assert r_over.ece > r_cal.ece
    assert r_over.overconfidence > r_cal.overconfidence
