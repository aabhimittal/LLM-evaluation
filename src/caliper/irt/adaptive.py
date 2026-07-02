"""Computerized adaptive testing over a calibrated item bank.

Each round picks the unseen item with (near-)maximal Fisher information at
the current ability estimate — "randomesque" selection among the top-k
controls item exposure — then re-estimates ability. The session stops when
the ability standard error drops below the target or the item budget runs
out. This is why ~30-50 well-chosen items give CIs comparable to running
the full benchmark.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from caliper.adapters.base import ModelAdapter
from caliper.irt.model import AbilityEstimate, estimate_ability, item_information
from caliper.runner import administer_item
from caliper.types import Item, ItemBank, TurnResult


@dataclass
class AdaptiveState:
    """Snapshot after each administered item (streamed to UIs)."""

    step: int
    item: Item
    result: TurnResult
    estimate: AbilityEstimate
    done: bool


@dataclass
class AdaptiveSession:
    bank: ItemBank
    se_target: float = 0.30
    max_items: int = 50
    min_items: int = 8
    exposure_k: int = 5
    seed: int = 0
    estimate: AbilityEstimate = field(
        default_factory=lambda: AbilityEstimate(theta=0.0, se=1.0, n_items=0)
    )

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)
        self._seen: set[str] = set()
        self._responses: list[tuple[Item, bool]] = []
        self.history: list[AbilityEstimate] = []

    @property
    def done(self) -> bool:
        n = len(self._responses)
        if n >= min(self.max_items, len(self.bank)):
            return True
        return n >= self.min_items and self.estimate.se <= self.se_target

    def next_item(self) -> Item:
        unseen = [it for it in self.bank if it.id not in self._seen]
        if not unseen:
            raise StopIteration("item bank exhausted")
        info = np.array(
            [item_information(self.estimate.theta, it.a, it.b, it.c) for it in unseen]
        )
        k = min(self.exposure_k, len(unseen))
        top = np.argsort(info)[-k:]
        choice = unseen[int(self._rng.choice(top))]
        return choice

    def record(self, item: Item, correct: bool) -> AbilityEstimate:
        self._seen.add(item.id)
        self._responses.append((item, correct))
        a = np.array([it.a for it, _ in self._responses])
        b = np.array([it.b for it, _ in self._responses])
        c = np.array([it.c for it, _ in self._responses])
        y = np.array([1.0 if ok else 0.0 for _, ok in self._responses])
        self.estimate = estimate_ability(a, b, c, y)
        self.history.append(self.estimate)
        return self.estimate

    @property
    def accuracy(self) -> float:
        if not self._responses:
            return 0.0
        return float(np.mean([ok for _, ok in self._responses]))


def run_adaptive(
    adapter: ModelAdapter,
    bank: ItemBank | None = None,
    *,
    se_target: float = 0.30,
    max_items: int = 50,
    min_items: int = 8,
    seed: int = 0,
    with_confidence: bool = False,
) -> Iterator[AdaptiveState]:
    """Drive a full adaptive session against a model, yielding each step.

    Consuming the iterator runs the evaluation; the last yielded state holds
    the final ability estimate. ``with_confidence`` also elicits verbalized
    confidence per item so a calibration curve comes for free.
    """
    bank = bank if bank is not None else ItemBank.bundled()
    session = AdaptiveSession(
        bank=bank, se_target=se_target, max_items=max_items, min_items=min_items, seed=seed
    )
    step = 0
    while not session.done:
        item = session.next_item()
        result = administer_item(adapter, item, with_confidence=with_confidence)
        estimate = session.record(item, result.correct)
        step += 1
        yield AdaptiveState(
            step=step, item=item, result=result, estimate=estimate, done=session.done
        )
