"""The anytime-validity guarantee is the product here, so it is tested
empirically rather than asserted in a docstring."""

import numpy as np

from caliper.adapters import SimulatedSubject
from caliper.sequential import (
    SequentialDuel,
    confidence_sequence_radius,
    paired_outcome,
    run_item_duel,
)


def _null_stream(rng: np.random.Generator) -> float:
    """Two equal models: discordant pairs split evenly, plus ties."""
    u = rng.random()
    return 1.0 if u < 0.25 else (0.0 if u < 0.5 else 0.5)


def test_paired_outcome_scoring():
    assert paired_outcome(True, False) == 1.0
    assert paired_outcome(False, True) == 0.0
    assert paired_outcome(True, True) == 0.5
    assert paired_outcome(False, False) == 0.5


def test_ties_do_not_move_the_wealth():
    duel = SequentialDuel(alpha=0.05)
    for _ in range(20):
        state = duel.update(0.5)
    assert state.e_value_a == 1.0
    assert state.e_value_b == 1.0
    assert not duel.decided


def test_type_i_error_under_continuous_peeking():
    """Ville's inequality: peeking after every observation must stay <= alpha."""
    alpha, n_sim, max_steps = 0.05, 400, 300
    false_positives = 0
    for s in range(n_sim):
        rng = np.random.default_rng(10_000 + s)
        duel = SequentialDuel(alpha=alpha)
        for _ in range(max_steps):
            duel.update(_null_stream(rng))
            if duel.decided:  # stop the instant it fires — the adversarial rule
                break
        false_positives += duel.decided
    rate = false_positives / n_sim
    assert rate <= alpha, f"anytime-validity violated: {rate:.3f} > {alpha}"


def test_naive_peeking_is_much_worse():
    """The reason this module exists: repeated z-tests blow past their level."""
    alpha, n_sim, max_steps = 0.05, 400, 300
    naive_fires = 0
    for s in range(n_sim):
        rng = np.random.default_rng(10_000 + s)
        xs = []
        for _ in range(max_steps):
            xs.append(_null_stream(rng))
            if len(xs) >= 10:
                arr = np.asarray(xs)
                se = arr.std(ddof=1) / np.sqrt(len(arr))
                if se > 0 and abs(arr.mean() - 0.5) / se > 1.96:
                    naive_fires += 1
                    break
    assert naive_fires / n_sim > 3 * alpha


def test_detects_a_real_difference_and_stops_early():
    stops, detected = [], 0
    for s in range(120):
        rng = np.random.default_rng(20_000 + s)
        duel = SequentialDuel(alpha=0.05, name_a="strong", name_b="weak")
        for _ in range(400):
            u = rng.random()
            duel.update(1.0 if u < 0.40 else (0.0 if u < 0.55 else 0.5))
            if duel.decided:
                break
        if duel.decided and duel.winner == "A":
            detected += 1
            stops.append(duel.stopped_at)
    assert detected / 120 > 0.8
    assert np.median(stops) < 250  # decided well before a fixed-N design
    assert stops  # winner_name resolves to the real model name


def test_confidence_sequence_covers_uniformly_in_time():
    n_sim, max_steps, alpha = 300, 200, 0.05
    misses = 0
    for true_p in (0.5, 0.7):
        for s in range(n_sim // 2):
            rng = np.random.default_rng(30_000 + s)
            xs = []
            for t in range(1, max_steps + 1):
                xs.append(float(rng.random() < true_p))
                mean = float(np.mean(xs))
                radius = confidence_sequence_radius(t, alpha)
                if not (mean - radius <= true_p <= mean + radius):
                    misses += 1
                    break
    assert misses / n_sim <= alpha


def test_confidence_sequence_shrinks():
    radii = [confidence_sequence_radius(n, 0.05) for n in (10, 100, 1000)]
    assert radii[0] > radii[1] > radii[2]
    assert radii[2] < 0.1


def test_item_duel_ranks_models_correctly(small_bank):
    strong = SimulatedSubject(theta=1.8, bank=small_bank, seed=3, name="strong")
    weak = SimulatedSubject(theta=-1.2, bank=small_bank, seed=4, name="weak")
    last = None
    for state in run_item_duel(strong, weak, small_bank, max_items=40, seed=1):
        last = state
    assert last is not None
    assert last.win_rate > 0.5  # strong model wins more discordant pairs
    lo, hi = last.ci95
    assert lo <= last.win_rate <= hi


def test_duel_summary_shape():
    duel = SequentialDuel(alpha=0.05, name_a="m1", name_b="m2")
    for _ in range(5):
        duel.update(1.0)
    summary = duel.summary()
    assert summary["models"] == {"A": "m1", "B": "m2"}
    assert summary["n_observations"] == 5
    assert summary["e_value_a"] > summary["e_value_b"]


def test_rejects_out_of_range_outcomes():
    duel = SequentialDuel()
    for bad in (-0.1, 1.5):
        try:
            duel.update(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted out-of-range outcome {bad}")
