"""Shared datatypes for Caliper."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Item:
    """A single multiple-choice test item with IRT parameters.

    IRT parameters follow the 3PL model with a fixed guessing floor:
    ``P(correct | theta) = c + (1 - c) * sigmoid(a * (theta - b))``
    where ``a`` is discrimination, ``b`` is difficulty and ``c`` is the
    guessing parameter (1 / number of choices for MCQ).
    """

    id: str
    question: str
    choices: list[str]
    answer_index: int
    a: float = 1.0
    b: float = 0.0
    source: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def c(self) -> float:
        return 1.0 / len(self.choices) if self.choices else 0.0

    @property
    def answer_letter(self) -> str:
        return "ABCDEFGH"[self.answer_index]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Item":
        return cls(**{k: d[k] for k in (
            "id", "question", "choices", "answer_index", "a", "b", "source", "tags"
        ) if k in d})


@dataclass
class ItemBank:
    """A collection of calibrated items."""

    items: list[Item]
    name: str = "item-bank"
    calibration: str = "uncalibrated"

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def by_id(self, item_id: str) -> Item:
        for item in self.items:
            if item.id == item_id:
                return item
        raise KeyError(item_id)

    def save(self, path: str | Path) -> None:
        payload = {
            "name": self.name,
            "calibration": self.calibration,
            "items": [item.to_dict() for item in self.items],
        }
        Path(path).write_text(json.dumps(payload, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ItemBank":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            items=[Item.from_dict(d) for d in payload["items"]],
            name=payload.get("name", "item-bank"),
            calibration=payload.get("calibration", "uncalibrated"),
        )

    @classmethod
    def bundled(cls) -> "ItemBank":
        """Load the item bank that ships with the package."""
        path = Path(__file__).parent / "data" / "item_bank.json"
        return cls.load(path)


@dataclass
class TurnResult:
    """Outcome of asking a model one item."""

    item_id: str
    raw_response: str
    parsed_answer: int | None
    correct: bool
    confidence: float | None = None
