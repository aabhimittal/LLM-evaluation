"""Administer items to a model."""

from __future__ import annotations

from caliper.adapters.base import ModelAdapter
from caliper.parsing import parse_confidence, parse_mcq_answer
from caliper.prompts import MCQ_CONFIDENCE_SYSTEM, MCQ_SYSTEM, format_mcq
from caliper.types import Item, TurnResult


def administer_item(
    adapter: ModelAdapter,
    item: Item,
    *,
    with_confidence: bool = False,
    question_override: str | None = None,
    choices_override: list[str] | None = None,
    answer_index_override: int | None = None,
) -> TurnResult:
    """Ask one MCQ item and score the reply.

    Overrides support the robustness suite, which perturbs the question or
    reshuffles the choices while keeping the scoring key in sync.
    """
    prompt = format_mcq(
        item, question_override=question_override, choices_override=choices_override
    )
    system = MCQ_CONFIDENCE_SYSTEM if with_confidence else MCQ_SYSTEM
    raw = adapter.ask(prompt, system=system, temperature=0.0, max_tokens=48)
    n_choices = len(choices_override or item.choices)
    parsed = parse_mcq_answer(raw, n_choices)
    answer_index = (
        answer_index_override if answer_index_override is not None else item.answer_index
    )
    return TurnResult(
        item_id=item.id,
        raw_response=raw,
        parsed_answer=parsed,
        correct=parsed == answer_index,
        confidence=parse_confidence(raw) if with_confidence else None,
    )
