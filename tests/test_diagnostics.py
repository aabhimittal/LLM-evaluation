import numpy as np

from caliper.diagnostics import (
    adaptive_efficiency,
    bank_health,
    items_needed,
    minimum_detectable_difference,
)

# Aliased: pytest would otherwise collect `test_information` as a test case.
from caliper.diagnostics import test_information as information_curve
from caliper.types import Item, ItemBank


def _narrow_bank(center: float, n: int = 40) -> ItemBank:
    """A bank whose items all cluster at one difficulty."""
    return ItemBank(
        items=[
            Item(
                id=f"n/{i}",
                question=f"question {i}",
                choices=["a", "b", "c", "d"],
                answer_index=0,
                a=1.2,
                b=center,
                source="synthetic",
            )
            for i in range(n)
        ],
        name="narrow",
    )


def test_information_peaks_near_bank_difficulty():
    bank = _narrow_bank(1.0)
    thetas, info, se = information_curve(bank)
    assert abs(thetas[int(np.argmax(info))] - 1.0) < 0.4
    assert np.all(se > 0)
    assert se[int(np.argmax(info))] == se.min()


def test_test_length_reduces_information():
    bank = _narrow_bank(0.0, n=60)
    _, full, _ = information_curve(bank)
    _, short, _ = information_curve(bank, test_length=10)
    assert np.all(short <= full + 1e-9)
    assert short.max() < full.max()


def test_bank_health_finds_saturation_ceiling():
    """An easy-only bank must go blind above its difficulty range."""
    easy = _narrow_bank(-1.5, n=60)
    health = bank_health(easy, se_target=0.3)
    assert health.usable_range is not None
    assert health.ceiling is not None
    assert health.ceiling < 1.5
    assert "aturated" in health.verdict or "degrades" in health.verdict


def test_bank_health_reports_no_ceiling_when_precision_holds():
    wide = ItemBank(
        items=[
            Item(id=f"w/{i}", question="q", choices=["a", "b", "c", "d"],
                 answer_index=0, a=1.5, b=float(b))
            for i, b in enumerate(np.linspace(-4, 4, 200))
        ],
        name="wide",
    )
    health = bank_health(wide, se_target=0.5)
    assert health.usable_range is not None
    assert health.saturated_fraction < 0.3


def test_bank_health_serializes(small_bank):
    payload = bank_health(small_bank, test_length=20).to_dict()
    assert payload["n_items"] == len(small_bank)
    assert payload["test_length"] == 20
    assert "verdict" in payload
    assert payload["best_achievable_se"] > 0


def test_more_items_detect_smaller_differences(small_bank):
    coarse = minimum_detectable_difference(small_bank, n_items=8)
    fine = minimum_detectable_difference(small_bank, n_items=32)
    assert fine.minimum_detectable_difference < coarse.minimum_detectable_difference
    assert fine.se_per_model < coarse.se_per_model


def test_items_needed_is_monotone(small_bank):
    """A tighter target needs at least as many items — or is unreachable."""
    loose = items_needed(2.2, small_bank)
    tight = items_needed(1.8, small_bank)
    assert loose is not None
    assert tight is None or tight >= loose


def test_items_needed_returns_none_when_unreachable(small_bank):
    assert items_needed(0.001, small_bank) is None


def test_adaptive_beats_random_selection(small_bank):
    result = adaptive_efficiency(small_bank, theta=0.0, n_items=10, n_trials=50)
    assert result["efficiency_ratio"] > 1.0
    assert result["adaptive_se"] < result["random_se"]
    assert result["random_items_for_same_precision"] > result["n_items"]
