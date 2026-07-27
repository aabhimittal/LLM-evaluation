"""Anytime-valid sequential model comparison.

The standard way to compare two models is: run N items, compute a p-value,
declare a winner. Everyone actually peeks at the results as they come in and
stops when they look good — which invalidates the p-value and inflates false
positives dramatically (see ``tests/test_sequential.py``, where naive peeking
fires on ~30% of null comparisons at a nominal 5% level).

Caliper instead uses **e-values / test martingales**. Wealth starts at 1 and
is bet on each observation with a predictable betting fraction; under the null
the wealth process is a nonnegative martingale, so by Ville's inequality

    P( ∃t : K_t ≥ 1/α ) ≤ α

*for every stopping rule at once*. You may look after every single item, stop
the moment ``K_t ≥ 1/α``, and the type-I error is still ≤ α. Paired with a
Robbins normal-mixture **confidence sequence** for the win rate, you get an
interval that is valid at all times simultaneously.

Practical payoff: evaluations stop as soon as the answer is decided instead of
burning a fixed budget of API calls.

References: Ville (1939); Robbins (1970); Howard et al., *Time-uniform,
nonasymptotic confidence sequences* (2021); Waudby-Smith & Ramdas,
*Estimating means of bounded random variables by betting* (2023).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

import numpy as np

from caliper.adapters.base import ModelAdapter
from caliper.runner import administer_item
from caliper.types import ItemBank

__all__ = [
    "DuelState",
    "SequentialDuel",
    "confidence_sequence_radius",
    "paired_outcome",
    "run_item_duel",
    "run_judge_duel",
]

# |lambda| < 2 is what validity requires, but betting that close to the edge is
# ruinous: one early loss at lambda=1.9 multiplies wealth by 0.05 and the
# process never recovers. Truncating at 1.0 bounds the worst step to a halving
# and costs almost nothing in growth rate (it is near-Kelly for realistic
# effect sizes). Validity is unaffected — any predictable lambda is valid.
_LAMBDA_CAP = 1.0

# Pseudo-observations at the null mean, so the first few items cannot produce
# a wild bet from a near-zero variance estimate.
_PRIOR_WEIGHT = 4.0
_VARIANCE_FLOOR = 0.05


def confidence_sequence_radius(
    n: int, alpha: float = 0.05, sigma: float = 0.5, rho: float | None = None
) -> float:
    """Radius of a Robbins normal-mixture confidence sequence for a mean.

    Valid *uniformly over time* for observations in [0, 1] (sub-Gaussian with
    ``sigma = 1/2`` by Hoeffding's lemma):

        P( ∃n : |mean_n - mu| ≥ radius(n) ) ≤ alpha

    ``rho`` tunes which sample size the boundary is tightest at.
    """
    if n <= 0:
        return float("inf")
    if rho is None:
        rho = sigma**2
    v = n * sigma**2
    # Two-sided: split alpha across the two boundaries.
    one_sided = alpha / 2.0
    return float(
        np.sqrt(2.0 * (v + rho) * np.log(np.sqrt((v + rho) / rho) / one_sided)) / n
    )


@dataclass
class DuelState:
    """Snapshot after one paired observation."""

    step: int
    outcome: float          # 1 = A wins, 0.5 = tie, 0 = B wins
    win_rate: float         # running mean of outcomes
    ci95: tuple[float, float]
    e_value_a: float        # wealth betting on "A better"
    e_value_b: float        # wealth betting on "B better"
    decided: bool
    winner: str | None      # "A", "B" or None
    detail: dict = field(default_factory=dict)


class SequentialDuel:
    """Anytime-valid paired comparison of two models.

    Feed outcomes in [0, 1] one at a time (1 = A wins, 0.5 = tie, 0 = B wins);
    after each one, ``decided`` tells you whether you may stop. Peeking is
    free — that is the whole point.
    """

    def __init__(self, alpha: float = 0.05, name_a: str = "A", name_b: str = "B"):
        self.alpha = alpha
        self.name_a = name_a
        self.name_b = name_b
        self.threshold = 2.0 / alpha  # two one-sided tests, each at alpha/2
        self.outcomes: list[float] = []
        self.e_a = 1.0
        self.e_b = 1.0
        self.history: list[DuelState] = []
        self._decided_at: int | None = None
        self._winner: str | None = None

    # -- betting ---------------------------------------------------------

    def _bet(self, direction: int) -> float:
        """Predictable betting fraction (regularised aGRAPA), from past data only.

        The mean and variance are shrunk toward the null with pseudo-counts, so
        a lucky first item cannot trigger an all-in bet. ``direction`` restricts
        each e-process to the side it is testing.
        """
        if not self.outcomes:
            return 0.25 * direction
        past = np.asarray(self.outcomes)
        n = len(past)
        # Shrink the mean toward the null value 0.5.
        mean = (_PRIOR_WEIGHT * 0.5 + past.sum()) / (_PRIOR_WEIGHT + n)
        edge = mean - 0.5
        var = (
            _PRIOR_WEIGHT * 0.25 + float(np.sum((past - mean) ** 2))
        ) / (_PRIOR_WEIGHT + n)
        lam = edge / max(var, _VARIANCE_FLOOR)
        lam = max(lam, 0.0) if direction > 0 else min(lam, 0.0)
        return float(np.clip(lam, -_LAMBDA_CAP, _LAMBDA_CAP))

    def update(self, outcome: float) -> DuelState:
        """Record one paired outcome and return the current state."""
        if not 0.0 <= outcome <= 1.0:
            raise ValueError("outcome must lie in [0, 1]")
        # Bets are computed *before* seeing this outcome (predictability).
        lam_a = self._bet(+1)
        lam_b = self._bet(-1)
        centered = outcome - 0.5
        self.e_a *= 1.0 + lam_a * centered
        self.e_b *= 1.0 + lam_b * centered
        self.outcomes.append(float(outcome))

        n = len(self.outcomes)
        mean = float(np.mean(self.outcomes))
        radius = confidence_sequence_radius(n, self.alpha)
        ci = (max(0.0, mean - radius), min(1.0, mean + radius))

        if self._winner is None:
            if self.e_a >= self.threshold:
                self._winner, self._decided_at = "A", n
            elif self.e_b >= self.threshold:
                self._winner, self._decided_at = "B", n

        state = DuelState(
            step=n,
            outcome=float(outcome),
            win_rate=mean,
            ci95=ci,
            e_value_a=float(self.e_a),
            e_value_b=float(self.e_b),
            decided=self._winner is not None,
            winner=self._winner,
            detail={"lambda_a": lam_a, "lambda_b": lam_b, "threshold": self.threshold},
        )
        self.history.append(state)
        return state

    # -- reporting --------------------------------------------------------

    @property
    def decided(self) -> bool:
        return self._winner is not None

    @property
    def winner(self) -> str | None:
        return self._winner

    @property
    def winner_name(self) -> str | None:
        if self._winner == "A":
            return self.name_a
        if self._winner == "B":
            return self.name_b
        return None

    @property
    def stopped_at(self) -> int | None:
        return self._decided_at

    def summary(self) -> dict:
        n = len(self.outcomes)
        mean = float(np.mean(self.outcomes)) if n else 0.5
        radius = confidence_sequence_radius(n, self.alpha) if n else float("inf")
        return {
            "models": {"A": self.name_a, "B": self.name_b},
            "n_observations": n,
            "win_rate_a": mean,
            "confidence_sequence_95": [
                max(0.0, mean - radius), min(1.0, mean + radius)
            ],
            "e_value_a": float(self.e_a),
            "e_value_b": float(self.e_b),
            "threshold": self.threshold,
            "decided": self.decided,
            "winner": self.winner_name,
            "stopped_at": self.stopped_at,
            "alpha": self.alpha,
            "guarantee": (
                "Type-I error <= alpha under ANY stopping rule (Ville's "
                "inequality); the interval is valid at all sample sizes at once."
            ),
        }


def paired_outcome(a_correct: bool, b_correct: bool) -> float:
    """Paired scoring: 1 if only A is right, 0 if only B is right, else 0.5.

    Discordant-pairs scoring (the sequential analogue of McNemar's test) —
    items both models get right or wrong carry no information about which is
    better, so they leave the wealth process untouched.
    """
    if a_correct and not b_correct:
        return 1.0
    if b_correct and not a_correct:
        return 0.0
    return 0.5


def run_item_duel(
    adapter_a: ModelAdapter,
    adapter_b: ModelAdapter,
    bank: ItemBank | None = None,
    *,
    alpha: float = 0.05,
    max_items: int = 200,
    seed: int = 0,
    stop_when_decided: bool = True,
) -> Iterator[DuelState]:
    """Duel two models on shared bank items, yielding after each item.

    Items are presented in a fixed random order to both models, so the
    comparison is paired. Consuming the iterator runs the duel; it ends as
    soon as the winner is decided (or the budget runs out).
    """
    bank = bank if bank is not None else ItemBank.bundled()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(bank.items))[:max_items]
    duel = SequentialDuel(
        alpha=alpha, name_a=adapter_a.name, name_b=adapter_b.name
    )
    for idx in order:
        item = bank.items[int(idx)]
        res_a = administer_item(adapter_a, item)
        res_b = administer_item(adapter_b, item)
        state = duel.update(paired_outcome(res_a.correct, res_b.correct))
        state.detail["item_id"] = item.id
        state.detail["a_correct"] = res_a.correct
        state.detail["b_correct"] = res_b.correct
        yield state
        if stop_when_decided and state.decided:
            return


def run_judge_duel(
    judge,  # caliper.judge.PairwiseJudge
    prompts: Iterable[str],
    responses_a: Iterable[str],
    responses_b: Iterable[str],
    *,
    alpha: float = 0.05,
    name_a: str = "A",
    name_b: str = "B",
    stop_when_decided: bool = True,
) -> Iterator[DuelState]:
    """Duel two models on open-ended prompts using a debiased pairwise judge.

    Each comparison contributes its *debiased* win probability directly as a
    [0, 1] observation, so judge uncertainty flows into the e-process instead
    of being rounded away.
    """
    duel = SequentialDuel(alpha=alpha, name_a=name_a, name_b=name_b)
    for prompt, ra, rb in zip(prompts, responses_a, responses_b):
        verdict = judge.compare(prompt, ra, rb)
        state = duel.update(verdict.p_a_wins)
        state.detail["prompt"] = prompt
        state.detail["judge_confidence"] = verdict.confidence
        state.detail["position_flip"] = verdict.position_flip
        yield state
        if stop_when_decided and state.decided:
            return
