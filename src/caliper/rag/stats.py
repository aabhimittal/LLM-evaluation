"""Small shared statistics helpers for the RAG instrument."""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values: list[float],
    *,
    seed: int = 0,
    n_boot: int = 500,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of ``values``.

    Returns ``(0.0, 0.0)`` for an empty sample. This is the one place the
    resampling idiom lives, so every RAG metric reports its uncertainty the
    same way.
    """
    if not values:
        return (0.0, 0.0)
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = [
        float(np.mean(arr[rng.integers(0, len(arr), size=len(arr))]))
        for _ in range(n_boot)
    ]
    tail = (1.0 - level) / 2.0 * 100.0
    return (float(np.percentile(means, tail)), float(np.percentile(means, 100.0 - tail)))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0
