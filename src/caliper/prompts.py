"""Canonical prompt templates.

Runners and the simulated backends share these, so the simulator can
recognize what it is being asked without any network access.
"""

from __future__ import annotations

from caliper.types import Item

LETTERS = "ABCDEFGH"

MCQ_SYSTEM = (
    "You are taking a multiple-choice test. Read the question and reply with "
    "only the letter of the correct choice."
)

MCQ_CONFIDENCE_SYSTEM = (
    "You are taking a multiple-choice test. Reply on the first line with only "
    "the letter of the correct choice. On the second line write "
    "'Confidence: N' where N is a number from 0 to 100 giving the probability "
    "that your answer is correct."
)

JUDGE_SYSTEM = (
    "You are an impartial judge comparing two AI responses to the same prompt. "
    "Judge which response is better on correctness, helpfulness and clarity. "
    "Ignore response order and length; longer is not better. Reply with only a "
    "JSON object: {\"winner\": \"A\" or \"B\" or \"tie\", \"reason\": \"...\"}."
)


def format_mcq(item: Item, question_override: str | None = None,
               choices_override: list[str] | None = None) -> str:
    question = question_override if question_override is not None else item.question
    choices = choices_override if choices_override is not None else item.choices
    lines = [question, ""]
    for i, choice in enumerate(choices):
        lines.append(f"{LETTERS[i]}. {choice}")
    lines.append("")
    lines.append("Reply with only the letter of the correct choice.")
    return "\n".join(lines)


def format_judge(prompt: str, response_a: str, response_b: str) -> str:
    return (
        f"[PROMPT]\n{prompt}\n[/PROMPT]\n\n"
        f"[RESPONSE A]\n{response_a}\n[/RESPONSE A]\n\n"
        f"[RESPONSE B]\n{response_b}\n[/RESPONSE B]\n\n"
        "Which response is better?"
    )


def format_completion_probe(prefix: str) -> str:
    return (
        "Complete the following text exactly as it continues in its original "
        "source. Output only the continuation, nothing else.\n\n"
        f"{prefix}"
    )


def format_option_recall(question: str) -> str:
    return (
        "The following question comes from a public benchmark. Reproduce its "
        "original multiple-choice options verbatim, one per line. If you do "
        "not know them, invent plausible options.\n\n"
        f"{question}"
    )
