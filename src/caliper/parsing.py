"""Robust parsers for model output."""

from __future__ import annotations

import json
import re

from caliper.prompts import LETTERS


def parse_mcq_answer(text: str, n_choices: int) -> int | None:
    """Extract a choice index from a free-form reply. None if unparseable."""
    valid = LETTERS[:n_choices]
    text = text.strip()
    if not text:
        return None
    # Common shapes: "B", "B.", "(B)", "Answer: B", "**B**", "The answer is B."
    patterns = [
        rf"^\(?([{valid}])\)?[.):\s]*$",
        rf"^\**\(?([{valid}])\)?\**[.):\s]",
        rf"answer\s*(?:is|:)?\s*\(?([{valid}])\)?\b",
        rf"^\s*\(?([{valid}])\)?\b",
    ]
    first_line = text.splitlines()[0]
    for pattern in patterns:
        m = re.search(pattern, first_line, flags=re.IGNORECASE)
        if m:
            return valid.index(m.group(1).upper())
    m = re.search(rf"\b([{valid}])\b", text)
    if m:
        return valid.index(m.group(1).upper())
    return None


def parse_confidence(text: str) -> float | None:
    """Extract 'Confidence: N' (0-100) as a probability in [0, 1]."""
    m = re.search(r"confidence\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%?", text, flags=re.IGNORECASE)
    if not m:
        return None
    value = float(m.group(1))
    if value > 1.0:
        value /= 100.0
    return min(max(value, 0.0), 1.0)


def parse_judge_verdict(text: str) -> str | None:
    """Return 'A', 'B' or 'tie' from a judge reply, or None."""
    m = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            winner = str(obj.get("winner", "")).strip().upper()
            if winner in ("A", "B"):
                return winner
            if winner == "TIE":
                return "tie"
        except (json.JSONDecodeError, AttributeError):
            pass
    m = re.search(r"\b(?:winner|better)\b.*?\b(A|B|tie)\b", text, flags=re.IGNORECASE)
    if m:
        w = m.group(1).upper()
        return "tie" if w == "TIE" else w
    return None
