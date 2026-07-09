"""Datatypes for RAG grounding evaluation.

A :class:`RagSample` is one retrieval-augmented question: the user question,
the passages a retriever returned, and (optionally) a reference answer. This
is a different shape from the MCQ :class:`caliper.types.Item`, so it gets its
own minimal, JSON-serializable datamodel that mirrors ``ItemBank``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class RagSample:
    """One retrieval-augmented question.

    Parameters
    ----------
    id: stable identifier.
    question: the user question posed to the RAG system.
    contexts: the retrieved passages supplied to the model as grounding.
    reference_answer: optional gold answer (metadata; not required to score
        faithfulness, which is judged against ``contexts``).
    supported_facts: optional gold list of facts the contexts actually
        support, for datasets that carry it.
    tags: free-form labels.
    """

    id: str
    question: str
    contexts: list[str]
    reference_answer: str = ""
    supported_facts: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RagSample":
        return cls(**{k: d[k] for k in (
            "id", "question", "contexts", "reference_answer",
            "supported_facts", "tags",
        ) if k in d})


@dataclass
class RagBank:
    """A collection of RAG samples."""

    samples: list[RagSample]
    name: str = "rag-bank"
    source: str = "unknown"

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def by_id(self, sample_id: str) -> RagSample:
        for sample in self.samples:
            if sample.id == sample_id:
                return sample
        raise KeyError(sample_id)

    def save(self, path: str | Path) -> None:
        payload = {
            "name": self.name,
            "source": self.source,
            "samples": [s.to_dict() for s in self.samples],
        }
        Path(path).write_text(json.dumps(payload, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RagBank":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            samples=[RagSample.from_dict(d) for d in payload["samples"]],
            name=payload.get("name", "rag-bank"),
            source=payload.get("source", "unknown"),
        )

    @classmethod
    def bundled(cls) -> "RagBank":
        """Load the small demo RAG bank that ships with the package."""
        path = Path(__file__).parent / "data" / "rag_bank.json"
        return cls.load(path)
