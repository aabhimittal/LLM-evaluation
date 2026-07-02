"""Caliper: measurement-science evaluation for LLMs.

Instead of a single leaderboard number, Caliper produces a statistical
*fingerprint* of a model: ability with confidence intervals (adaptive IRT),
judge verdicts with uncertainty and bias audits, metamorphic robustness,
confidence calibration, and benchmark-contamination risk.
"""

from caliper.types import Item, ItemBank, TurnResult

__version__ = "0.1.0"

__all__ = ["Item", "ItemBank", "TurnResult", "__version__"]
