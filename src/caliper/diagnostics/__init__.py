"""Diagnostics for the *benchmark*, not the model.

Three questions nobody asks of a benchmark before quoting scores from it:

1. **Where can this benchmark actually measure?** The test information function
   gives the precision a bank can deliver at each ability level. Above some
   ability the curve collapses and every frontier model returns the same score
   with overlapping error bars — the benchmark is **saturated** and further
   results from it are noise. Caliper reports that ceiling as a number.

2. **How many items do I need?** Classical power analysis, in IRT units:
   the smallest ability gap detectable with a given item budget, and the budget
   required for a target gap.

3. **Is adaptive selection actually paying off?** Information accumulated by
   Fisher-information selection versus random selection, as an efficiency
   ratio — how many random items one adaptive item is worth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from caliper.irt.model import item_information
from caliper.types import ItemBank

__all__ = [
    "BankHealth",
    "PowerAnalysis",
    "adaptive_efficiency",
    "bank_health",
    "items_needed",
    "minimum_detectable_difference",
    "test_information",
]


def test_information(
    bank: ItemBank,
    thetas: np.ndarray | None = None,
    test_length: int | None = None,
) -> tuple:
    """Fisher information available across the ability scale.

    Returns ``(thetas, information, standard_error)`` with
    ``standard_error = 1/sqrt(information)``.

    ``test_length`` is the decisive argument. With ``None`` the whole bank is
    summed, which answers "what could this bank ever measure". With a number,
    only the ``test_length`` most informative items at each ability are
    counted — what an adaptive test of that length actually achieves, and the
    honest basis for a saturation claim.
    """
    if thetas is None:
        thetas = np.linspace(-4.0, 4.0, 161)
    thetas = np.asarray(thetas, dtype=float)
    per_item = np.vstack(
        [item_information(thetas, item.a, item.b, item.c) for item in bank]
    )  # (n_items, n_thetas)
    if test_length is None or test_length >= per_item.shape[0]:
        info = per_item.sum(axis=0)
    else:
        # Best `test_length` items separately at each ability point.
        info = np.sort(per_item, axis=0)[-test_length:, :].sum(axis=0)
    with np.errstate(divide="ignore"):
        se = 1.0 / np.sqrt(np.maximum(info, 1e-12))
    return thetas, info, se


@dataclass
class BankHealth:
    n_items: int
    test_length: int | None       # item budget the verdict is stated for
    peak_theta: float             # ability the bank measures most precisely
    peak_information: float
    usable_range: tuple[float, float] | None   # where SE <= se_target
    ceiling: float | None         # ability above which measurement fails
    floor: float | None
    se_target: float
    grid_limit: float
    best_se: float                # best precision achievable anywhere
    difficulty_span: tuple[float, float]
    mean_discrimination: float
    saturated_fraction: float     # fraction of the scanned scale that is unusable
    curve: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_items": self.n_items,
            "test_length": self.test_length,
            "peak_theta": round(self.peak_theta, 3),
            "peak_information": round(self.peak_information, 2),
            "usable_theta_range": (
                [round(x, 3) for x in self.usable_range] if self.usable_range else None
            ),
            "ceiling": round(self.ceiling, 3) if self.ceiling is not None else None,
            "floor": round(self.floor, 3) if self.floor is not None else None,
            "se_target": self.se_target,
            "best_achievable_se": round(self.best_se, 4),
            "difficulty_span": [round(x, 3) for x in self.difficulty_span],
            "mean_discrimination": round(self.mean_discrimination, 3),
            "unusable_fraction_of_scale": round(self.saturated_fraction, 3),
            "verdict": self.verdict,
        }

    @property
    def verdict(self) -> str:
        budget = (
            f"a {self.test_length}-item test" if self.test_length
            else f"the full {self.n_items}-item bank"
        )
        if self.usable_range is None:
            return (
                f"With {budget} this bank never reaches SE <= {self.se_target}; "
                f"the best it achieves is SE {self.best_se:.3f} at theta "
                f"{self.peak_theta:+.2f}. Lengthen the test or add more "
                "discriminating items."
            )
        if self.ceiling is None:
            return (
                f"With {budget}, precision holds past theta "
                f"{self.grid_limit:+.1f} — no saturation ceiling within the "
                "scanned range."
            )
        pct = 100 * norm.cdf(self.ceiling)
        head = "Saturated" if self.ceiling < 1.0 else "Usable"
        return (
            f"{head}: with {budget}, measurement degrades past theta "
            f"{self.ceiling:+.2f} (~{pct:.0f}th percentile of the calibration "
            "population). Models above that line cannot be told apart by this "
            "bank — differences reported there are noise."
        )


def bank_health(
    bank: ItemBank | None = None,
    se_target: float = 0.3,
    test_length: int | None = None,
    grid_limit: float = 4.0,
    grid: int = 161,
) -> BankHealth:
    """Where on the ability scale can this bank actually measure?

    ``test_length`` states the verdict for a realistic item budget (e.g. 40)
    rather than for the whole bank at once.
    """
    bank = bank if bank is not None else ItemBank.bundled()
    thetas, info, se = test_information(
        bank, np.linspace(-grid_limit, grid_limit, grid), test_length=test_length
    )
    usable = se <= se_target
    usable_range: tuple[float, float] | None
    if usable.any():
        lo, hi = float(thetas[usable][0]), float(thetas[usable][-1])
        usable_range = (lo, hi)
        # A boundary that coincides with the grid edge is a limit of the scan,
        # not a real ceiling/floor.
        ceiling = None if usable[-1] else hi
        floor = None if usable[0] else lo
    else:
        usable_range = ceiling = floor = None

    peak_idx = int(np.argmax(info))
    difficulties = np.array([it.b for it in bank])
    return BankHealth(
        n_items=len(bank),
        test_length=test_length,
        peak_theta=float(thetas[peak_idx]),
        peak_information=float(info[peak_idx]),
        usable_range=usable_range,
        ceiling=ceiling,
        floor=floor,
        se_target=se_target,
        grid_limit=grid_limit,
        best_se=float(se.min()),
        difficulty_span=(float(difficulties.min()), float(difficulties.max())),
        mean_discrimination=float(np.mean([it.a for it in bank])),
        saturated_fraction=float(1.0 - usable.mean()),
        curve=[
            {"theta": float(t), "information": float(i), "se": float(s)}
            for t, i, s in zip(thetas[::4], info[::4], se[::4])
        ],
    )


@dataclass
class PowerAnalysis:
    n_items: int
    theta: float
    se_per_model: float
    minimum_detectable_difference: float
    alpha: float
    power: float
    interpretation: str = ""

    def to_dict(self) -> dict:
        return {
            "n_items_per_model": self.n_items,
            "at_theta": self.theta,
            "se_per_model": round(self.se_per_model, 4),
            "minimum_detectable_theta_gap": round(self.minimum_detectable_difference, 4),
            "alpha": self.alpha,
            "power": self.power,
            "interpretation": self.interpretation,
        }


def _se_at(bank: ItemBank, theta: float, n_items: int, adaptive: bool = True) -> float:
    """SE achievable with ``n_items`` from the bank at ``theta``.

    Adaptive selection takes the ``n_items`` most informative items at that
    ability; random selection takes the average item.
    """
    infos = np.array([item_information(theta, it.a, it.b, it.c) for it in bank])
    n_items = min(n_items, len(infos))
    if n_items <= 0:
        return float("inf")
    total = np.sort(infos)[-n_items:].sum() if adaptive else infos.mean() * n_items
    return float(1.0 / np.sqrt(max(total, 1e-12)))


def minimum_detectable_difference(
    bank: ItemBank | None = None,
    n_items: int = 40,
    theta: float = 0.0,
    alpha: float = 0.05,
    power: float = 0.8,
    adaptive: bool = True,
) -> PowerAnalysis:
    """Smallest ability gap two models must have for this design to detect it.

    Uses the standard two-sample formula on the IRT scale:
    ``delta = (z_{1-alpha/2} + z_{power}) * sqrt(SE_A^2 + SE_B^2)``.
    """
    bank = bank if bank is not None else ItemBank.bundled()
    se = _se_at(bank, theta, n_items, adaptive)
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    delta = float(z * np.sqrt(2.0) * se)
    percentile_gap = 100 * (norm.cdf(theta + delta / 2) - norm.cdf(theta - delta / 2))
    return PowerAnalysis(
        n_items=n_items,
        theta=theta,
        se_per_model=se,
        minimum_detectable_difference=delta,
        alpha=alpha,
        power=power,
        interpretation=(
            f"With {n_items} items per model at theta {theta:+.1f}, only gaps "
            f"larger than {delta:.2f} logits (~{percentile_gap:.0f} percentile "
            "points) are detectable. Smaller reported differences are noise."
        ),
    )


def items_needed(
    target_difference: float,
    bank: ItemBank | None = None,
    theta: float = 0.0,
    alpha: float = 0.05,
    power: float = 0.8,
    adaptive: bool = True,
    max_items: int = 100_000,
) -> int | None:
    """Item budget required to detect a given ability gap. None if unreachable."""
    bank = bank if bank is not None else ItemBank.bundled()
    for n in range(2, min(max_items, len(bank) if adaptive else max_items) + 1):
        result = minimum_detectable_difference(
            bank, n_items=n, theta=theta, alpha=alpha, power=power, adaptive=adaptive
        )
        if result.minimum_detectable_difference <= target_difference:
            return n
    return None


def adaptive_efficiency(
    bank: ItemBank | None = None,
    theta: float = 0.0,
    n_items: int = 30,
    n_trials: int = 200,
    seed: int = 0,
) -> dict:
    """How much is one adaptive item worth, in randomly-chosen items?

    Compares information accumulated by Fisher-information selection against
    random sampling from the same bank, and converts the gap into the number
    of random items needed to match the adaptive standard error.
    """
    bank = bank if bank is not None else ItemBank.bundled()
    infos = np.array([item_information(theta, it.a, it.b, it.c) for it in bank])
    n_items = min(n_items, len(infos))

    adaptive_info = float(np.sort(infos)[-n_items:].sum())
    rng = np.random.default_rng(seed)
    random_info = float(
        np.mean([
            infos[rng.choice(len(infos), size=n_items, replace=False)].sum()
            for _ in range(n_trials)
        ])
    )
    mean_info = float(infos.mean())
    equivalent_random = adaptive_info / mean_info if mean_info > 0 else float("inf")
    return {
        "theta": theta,
        "n_items": n_items,
        "adaptive_information": adaptive_info,
        "random_information": random_info,
        "efficiency_ratio": adaptive_info / random_info if random_info > 0 else float("inf"),
        "adaptive_se": float(1 / np.sqrt(max(adaptive_info, 1e-12))),
        "random_se": float(1 / np.sqrt(max(random_info, 1e-12))),
        "random_items_for_same_precision": equivalent_random,
        "interpretation": (
            f"At theta {theta:+.1f}, {n_items} adaptively chosen items carry the "
            f"information of {equivalent_random:.0f} randomly chosen ones "
            f"({adaptive_info / random_info:.2f}x per item)."
        ),
    }
