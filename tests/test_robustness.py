import numpy as np

from caliper.adapters import SimulatedSubject
from caliper.robustness import evaluate_robustness, free_text_consistency
from caliper.robustness.perturb import PERTURBATIONS, perturb_choices


def test_perturbations_are_deterministic():
    text = "Which of the following statements about photosynthesis is most likely correct?"
    for name, fn in PERTURBATIONS.items():
        out1 = fn(text, np.random.default_rng(7))
        out2 = fn(text, np.random.default_rng(7))
        assert out1 == out2, name
        assert out1.strip(), name


def test_perturb_choices_keeps_answer_key_in_sync():
    choices = ["alpha", "beta", "gamma", "delta"]
    for seed in range(10):
        shuffled, new_idx = perturb_choices(choices, 2, np.random.default_rng(seed))
        assert sorted(shuffled) == sorted(choices)
        assert shuffled[new_idx] == "gamma"


def test_fragile_model_scores_lower(small_bank):
    fragile = SimulatedSubject(theta=0.5, bank=small_bank, robustness=0.5, seed=4)
    solid = SimulatedSubject(theta=0.5, bank=small_bank, robustness=1.0, seed=4)
    r_fragile = evaluate_robustness(fragile, small_bank, n_items=8, seed=0)
    r_solid = evaluate_robustness(solid, small_bank, n_items=8, seed=0)
    assert r_fragile.overall_consistency < r_solid.overall_consistency
    lo, hi = r_solid.ci95
    assert lo <= r_solid.overall_consistency <= hi


def test_free_text_consistency_in_range(small_bank):
    subject = SimulatedSubject(theta=0.5, bank=small_bank, seed=4)
    score = free_text_consistency(
        subject, ["Explain why the sky is blue in simple terms."], seed=0
    )
    assert 0.0 <= score <= 1.0
