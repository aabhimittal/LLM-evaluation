"""Bradley-Terry ranking with bootstrap confidence intervals.

Match outcomes (possibly fractional, e.g. debiased win probabilities from
:class:`~caliper.judge.pairwise.PairwiseJudge`) are fit with a Bradley-Terry
model ``P(a beats b) = sigmoid(s_a - s_b)`` by penalized MLE. Uncertainty
comes from a nonparametric bootstrap over matches — the same approach used
by Chatbot Arena — reported on the familiar Elo-like scale
``rating = 1000 + s * 400 / ln(10)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

ELO_SCALE = 400.0 / np.log(10.0)
ELO_BASE = 1000.0


@dataclass
class Match:
    a: str
    b: str
    score_a: float  # 1 = A wins, 0.5 = tie, 0 = B wins (fractional allowed)


@dataclass
class RatingTable:
    models: list[str]
    strength: dict[str, float]           # BT log-strengths (sum-zero)
    rating: dict[str, float]             # Elo-like scale
    ci95: dict[str, tuple[float, float]]  # bootstrap CI on the Elo scale
    n_matches: int

    def sorted_models(self) -> list[str]:
        return sorted(self.models, key=lambda m: -self.rating[m])


def fit_bradley_terry(
    matches: list[Match], models: list[str] | None = None, l2: float = 0.01
) -> dict[str, float]:
    """Penalized MLE of sum-zero BT strengths."""
    if models is None:
        models = sorted({m.a for m in matches} | {m.b for m in matches})
    index = {m: i for i, m in enumerate(models)}
    ia = np.array([index[m.a] for m in matches])
    ib = np.array([index[m.b] for m in matches])
    y = np.array([m.score_a for m in matches], dtype=float)

    def nll(s):
        z = np.clip(s[ia] - s[ib], -35, 35)
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)) + l2 * np.sum(s**2)

    res = minimize(nll, x0=np.zeros(len(models)), method="L-BFGS-B")
    s = res.x - res.x.mean()
    return {m: float(s[index[m]]) for m in models}


def bootstrap_ratings(
    matches: list[Match],
    n_boot: int = 200,
    seed: int = 0,
) -> RatingTable:
    """BT fit plus bootstrap-over-matches CIs on the Elo scale."""
    models = sorted({m.a for m in matches} | {m.b for m in matches})
    point = fit_bradley_terry(matches, models)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {m: [] for m in models}
    n = len(matches)
    for _ in range(n_boot):
        resampled = [matches[i] for i in rng.integers(0, n, size=n)]
        s = fit_bradley_terry(resampled, models)
        for m in models:
            samples[m].append(ELO_BASE + ELO_SCALE * s[m])
    rating = {m: ELO_BASE + ELO_SCALE * point[m] for m in models}
    ci95 = {
        m: (
            float(np.percentile(samples[m], 2.5)),
            float(np.percentile(samples[m], 97.5)),
        )
        for m in models
    }
    return RatingTable(
        models=models, strength=point, rating=rating, ci95=ci95, n_matches=n
    )
