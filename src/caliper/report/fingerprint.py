"""Assemble the full measurement fingerprint of a model."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from scipy.stats import norm

from caliper.adapters.base import ModelAdapter
from caliper.calibration import CalibrationReport, evaluate_calibration
from caliper.contamination import ContaminationReport, evaluate_contamination
from caliper.irt import run_adaptive
from caliper.irt.model import AbilityEstimate
from caliper.robustness import RobustnessReport, evaluate_robustness
from caliper.types import ItemBank


@dataclass
class AbilitySection:
    estimate: AbilityEstimate
    accuracy: float
    trajectory: list[dict] = field(default_factory=list)  # per-step theta/se


@dataclass
class Fingerprint:
    model_name: str
    bank_name: str
    bank_calibration: str
    created_at: str
    ability: AbilitySection
    robustness: RobustnessReport
    calibration: CalibrationReport
    contamination: ContaminationReport

    def dimensions(self) -> dict[str, float]:
        """Normalized 0-1 scores for the radar. Higher is always better."""
        return {
            "Ability": float(norm.cdf(self.ability.estimate.theta)),
            "Robustness": self.robustness.overall_consistency,
            "Calibration": 1.0 - min(2.0 * self.calibration.ece, 1.0),
            "Selective risk": 1.0 - min(2.0 * self.calibration.aurc, 1.0),
            "Cleanliness": 1.0 - self.contamination.risk,
        }

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "bank_name": self.bank_name,
            "bank_calibration": self.bank_calibration,
            "created_at": self.created_at,
            "dimensions": self.dimensions(),
            "ability": {
                "theta": self.ability.estimate.theta,
                "se": self.ability.estimate.se,
                "ci95": list(self.ability.estimate.ci95),
                "n_items": self.ability.estimate.n_items,
                "accuracy": self.ability.accuracy,
                "trajectory": self.ability.trajectory,
            },
            "robustness": asdict(self.robustness),
            "calibration": asdict(self.calibration),
            "contamination": asdict(self.contamination),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def run_fingerprint(
    adapter: ModelAdapter,
    bank: ItemBank | None = None,
    *,
    adaptive_max_items: int = 40,
    robustness_items: int = 10,
    calibration_items: int = 30,
    contamination_items: int = 12,
    seed: int = 0,
    progress: Callable[[str], None] | None = None,
) -> Fingerprint:
    """Run all suites against one model. ~150 model calls at default budgets."""
    bank = bank if bank is not None else ItemBank.bundled()
    say = progress or (lambda _msg: None)

    say("adaptive ability estimation…")
    trajectory: list[dict] = []
    last = None
    for state in run_adaptive(adapter, bank, max_items=adaptive_max_items, seed=seed):
        trajectory.append(
            {"step": state.step, "theta": state.estimate.theta, "se": state.estimate.se,
             "item_id": state.item.id, "correct": state.result.correct}
        )
        last = state
    accuracy = (
        float(sum(t["correct"] for t in trajectory)) / len(trajectory) if trajectory else 0.0
    )
    ability = AbilitySection(
        estimate=last.estimate if last else AbilityEstimate(0.0, 1.0, 0),
        accuracy=accuracy,
        trajectory=trajectory,
    )

    say("metamorphic robustness…")
    robustness = evaluate_robustness(adapter, bank, n_items=robustness_items, seed=seed)
    say("confidence calibration…")
    calibration = evaluate_calibration(adapter, bank, n_items=calibration_items, seed=seed)
    say("contamination probes…")
    contamination = evaluate_contamination(adapter, bank, n_items=contamination_items, seed=seed)

    return Fingerprint(
        model_name=adapter.name,
        bank_name=bank.name,
        bank_calibration=bank.calibration,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        ability=ability,
        robustness=robustness,
        calibration=calibration,
        contamination=contamination,
    )
