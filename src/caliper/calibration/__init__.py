"""Confidence calibration: does the model know what it doesn't know?

Elicits verbalized confidence alongside each answer and reports:

- **ECE** (expected calibration error, equal-width bins)
- **Brier score**
- **Risk-coverage curve** and its area (AURC): if the model only answers
  when confident, how does its error rate fall as it abstains more?
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from caliper.adapters.base import ModelAdapter
from caliper.runner import administer_item
from caliper.types import ItemBank


@dataclass
class CalibrationReport:
    n_items: int
    ece: float
    brier: float
    aurc: float
    accuracy: float
    mean_confidence: float
    bins: list[dict] = field(default_factory=list)          # reliability diagram data
    risk_coverage: list[dict] = field(default_factory=list)  # curve points

    @property
    def overconfidence(self) -> float:
        """Positive = confidence exceeds accuracy."""
        return self.mean_confidence - self.accuracy


def ece(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> tuple[float, list[dict]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(confidence)
    err = 0.0
    bins = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidence >= lo) & (confidence < hi if i < n_bins - 1 else confidence <= hi)
        if not mask.any():
            continue
        acc = float(correct[mask].mean())
        conf = float(confidence[mask].mean())
        weight = mask.sum() / total
        err += weight * abs(acc - conf)
        bins.append({"lo": float(lo), "hi": float(hi), "n": int(mask.sum()),
                     "accuracy": acc, "confidence": conf})
    return float(err), bins


def risk_coverage(confidence: np.ndarray, correct: np.ndarray) -> tuple[float, list[dict]]:
    """Selective-prediction curve: sort by confidence desc, sweep coverage."""
    order = np.argsort(-confidence)
    sorted_correct = correct[order]
    n = len(sorted_correct)
    points = []
    risks = []
    for k in range(1, n + 1):
        cov = k / n
        risk = float(1.0 - sorted_correct[:k].mean())
        risks.append(risk)
        points.append({"coverage": cov, "risk": risk,
                       "threshold": float(confidence[order][k - 1])})
    aurc = float(np.mean(risks))
    return aurc, points


def evaluate_calibration(
    adapter: ModelAdapter,
    bank: ItemBank | None = None,
    n_items: int = 40,
    seed: int = 0,
) -> CalibrationReport:
    bank = bank if bank is not None else ItemBank.bundled()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(bank.items), size=min(n_items, len(bank.items)), replace=False)
    confs, corrects = [], []
    for i in idx:
        item = bank.items[int(i)]
        result = administer_item(adapter, item, with_confidence=True)
        if result.parsed_answer is None or result.confidence is None:
            continue
        confs.append(result.confidence)
        corrects.append(1.0 if result.correct else 0.0)
    if not confs:
        return CalibrationReport(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    confidence = np.asarray(confs)
    correct = np.asarray(corrects)
    ece_value, bins = ece(confidence, correct)
    aurc, curve = risk_coverage(confidence, correct)
    brier = float(np.mean((confidence - correct) ** 2))
    return CalibrationReport(
        n_items=len(confs),
        ece=ece_value,
        brier=brier,
        aurc=aurc,
        accuracy=float(correct.mean()),
        mean_confidence=float(confidence.mean()),
        bins=bins,
        risk_coverage=curve[:: max(1, len(curve) // 50)],
    )
