"""3PL Item Response Theory model (numpy/scipy, no ML frameworks).

``P(correct | theta) = c + (1 - c) * sigmoid(a * (theta - b))``

- ``theta``: latent ability of the model under test
- ``a``: item discrimination, ``b``: item difficulty
- ``c``: guessing floor, fixed at 1/n_choices for MCQ items

Ability is estimated by MAP under a standard-normal prior; its standard
error comes from the curvature of the log-posterior at the mode. Item
parameters are calibrated from a correctness matrix by alternating MAP
(a joint-mode approximation of marginal maximum likelihood that is simple,
dependency-free and accurate enough for adaptive item selection).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize, minimize_scalar

THETA_BOUNDS = (-5.0, 5.0)
_EPS = 1e-9


def p_correct(theta, a, b, c):
    """3PL response probability. Broadcasts over numpy arrays."""
    z = np.clip(a * (np.asarray(theta) - b), -35.0, 35.0)
    return c + (1.0 - c) / (1.0 + np.exp(-z))


def item_information(theta, a, b, c):
    """Fisher information of one item at ability ``theta`` (3PL form)."""
    p = p_correct(theta, a, b, c)
    q = 1.0 - p
    return (a**2) * (q / np.maximum(p, _EPS)) * ((p - c) / (1.0 - c)) ** 2


@dataclass
class AbilityEstimate:
    theta: float
    se: float
    n_items: int
    log_posterior: float = 0.0

    @property
    def ci95(self) -> tuple[float, float]:
        return (self.theta - 1.96 * self.se, self.theta + 1.96 * self.se)


def _neg_log_posterior(theta: float, a, b, c, y, prior_sd: float) -> float:
    p = p_correct(theta, a, b, c)
    p = np.clip(p, _EPS, 1.0 - _EPS)
    ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    return -(ll - 0.5 * theta**2 / prior_sd**2)


def estimate_ability(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    y: np.ndarray,
    prior_sd: float = 1.0,
) -> AbilityEstimate:
    """MAP ability estimate with standard error from posterior curvature.

    Parameters are per administered item; ``y`` is the 0/1 correctness vector.
    """
    a, b, c, y = (np.asarray(x, dtype=float) for x in (a, b, c, y))
    if len(y) == 0:
        return AbilityEstimate(theta=0.0, se=prior_sd, n_items=0)
    res = minimize_scalar(
        _neg_log_posterior,
        bounds=THETA_BOUNDS,
        args=(a, b, c, y, prior_sd),
        method="bounded",
        options={"xatol": 1e-6},
    )
    theta = float(res.x)
    # Observed information: numeric second derivative of the neg log posterior.
    h = 1e-4
    f = _neg_log_posterior
    d2 = (
        f(theta + h, a, b, c, y, prior_sd)
        - 2 * f(theta, a, b, c, y, prior_sd)
        + f(theta - h, a, b, c, y, prior_sd)
    ) / h**2
    se = float(1.0 / np.sqrt(max(d2, _EPS)))
    return AbilityEstimate(
        theta=theta, se=se, n_items=int(len(y)), log_posterior=float(-res.fun)
    )


@dataclass
class CalibrationResult:
    a: np.ndarray
    b: np.ndarray
    thetas: np.ndarray
    n_iter: int
    converged: bool
    history: list[float] = field(default_factory=list)


def _fit_one_item(theta: np.ndarray, y: np.ndarray, c: float,
                  a0: float, b0: float) -> tuple[float, float]:
    """MAP fit of (a, b) for one item given respondent abilities."""
    mask = ~np.isnan(y)
    th, yy = theta[mask], y[mask]

    def nll(params):
        log_a, b = params
        a = np.exp(log_a)
        p = np.clip(p_correct(th, a, b, c), _EPS, 1 - _EPS)
        ll = np.sum(yy * np.log(p) + (1 - yy) * np.log(1 - p))
        # Priors: log a ~ N(0, 0.5^2), b ~ N(0, 1.5^2) — keep params sane.
        return -(ll - 0.5 * (log_a / 0.5) ** 2 - 0.5 * (b / 1.5) ** 2)

    res = minimize(nll, x0=[np.log(max(a0, 0.05)), b0], method="L-BFGS-B",
                   bounds=[(-2.5, 1.6), (-5.0, 5.0)])
    log_a, b = res.x
    return float(np.exp(log_a)), float(b)


def fit_items(
    responses: np.ndarray,
    n_choices: int | np.ndarray = 4,
    max_iter: int = 40,
    tol: float = 1e-3,
) -> CalibrationResult:
    """Calibrate item parameters from a (respondents x items) 0/1 matrix.

    NaN entries mean 'not administered'. Alternates MAP ability estimation
    and per-item (a, b) fits; the theta scale is re-standardized every
    iteration for identifiability.
    """
    X = np.asarray(responses, dtype=float)
    n_resp, n_items = X.shape
    c = np.full(n_items, 1.0 / n_choices) if np.isscalar(n_choices) else 1.0 / np.asarray(
        n_choices, dtype=float
    )

    # Initialize abilities from standardized raw scores, items from p-values.
    raw = np.nanmean(X, axis=1)
    thetas = (raw - np.nanmean(raw)) / (np.nanstd(raw) + _EPS)
    pvals = np.clip(np.nanmean(X, axis=0), 0.02, 0.98)
    b = -np.log((pvals - c.clip(max=pvals - 0.01)) / (1 - pvals + _EPS)).clip(-3, 3)
    a = np.ones(n_items)

    history: list[float] = []
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        # E-like step: abilities given items.
        new_thetas = np.empty(n_resp)
        for i in range(n_resp):
            mask = ~np.isnan(X[i])
            est = estimate_ability(a[mask], b[mask], c[mask], X[i][mask])
            new_thetas[i] = est.theta
        # Standardize for identifiability.
        new_thetas = (new_thetas - new_thetas.mean()) / (new_thetas.std() + _EPS)

        # M-like step: items given abilities.
        new_a, new_b = a.copy(), b.copy()
        for j in range(n_items):
            new_a[j], new_b[j] = _fit_one_item(new_thetas, X[:, j], float(c[j]), a[j], b[j])

        delta = float(
            np.max(np.abs(new_a - a)) + np.max(np.abs(new_b - b))
            + np.max(np.abs(new_thetas - thetas))
        )
        history.append(delta)
        a, b, thetas = new_a, new_b, new_thetas
        if delta < tol:
            converged = True
            break

    return CalibrationResult(a=a, b=b, thetas=thetas, n_iter=it,
                             converged=converged, history=history)
