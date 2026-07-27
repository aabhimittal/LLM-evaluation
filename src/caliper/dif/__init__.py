"""Differential Item Functioning — auditing the benchmark itself.

Every other module in Caliper measures a *model*. This one measures the
*benchmark*, and asks a question the field mostly hasn't:

    Is this item unfair between model families, after controlling for ability?

An item shows **DIF** when two groups of equal ability have different chances
of answering it correctly. In education this is the standard tool for finding
culturally biased test questions; applied to LLM benchmarks it surfaces items
that reward a training recipe, a tokenizer, a formatting habit or plain
contamination rather than the capability the benchmark claims to measure.

The estimator is **Mantel-Haenszel**: models are matched into strata by total
score (an ability proxy), a 2x2 table of group x correctness is built inside
each stratum, and the common odds ratio across strata is tested. Effect sizes
use the ETS delta scale, the classification used operationally by testing
programs:

    |delta| < 1.0            A — negligible
    1.0 <= |delta| < 1.5     B — moderate
    |delta| >= 1.5           C — large

Sign convention: **positive delta favors the focal group**.

Reference: Holland & Thayer (1988); Dorans & Holland (1993).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import chi2

from caliper.types import ItemBank

__all__ = ["DIFItem", "DIFReport", "detect_dif", "mantel_haenszel"]


@dataclass
class DIFItem:
    item_id: str
    index: int
    odds_ratio: float
    delta: float           # ETS delta scale; positive favors the focal group
    chi2: float
    p_value: float
    classification: str    # "A", "B" or "C"
    favors: str            # "focal", "reference" or "neither"
    n_reference: int
    n_focal: int
    p_reference: float     # raw accuracy, for context
    p_focal: float

    @property
    def flagged(self) -> bool:
        return self.classification in ("B", "C")


@dataclass
class DIFReport:
    items: list[DIFItem]
    n_items: int
    n_reference: int
    n_focal: int
    n_strata: int
    reference_name: str = "reference"
    focal_name: str = "focal"
    flagged: list[DIFItem] = field(default_factory=list)

    @property
    def flag_rate(self) -> float:
        return len(self.flagged) / self.n_items if self.n_items else 0.0

    def worst(self, k: int = 10) -> list[DIFItem]:
        return sorted(self.items, key=lambda it: -abs(it.delta))[:k]

    def to_dict(self) -> dict:
        return {
            "reference_group": self.reference_name,
            "focal_group": self.focal_name,
            "n_items": self.n_items,
            "n_reference_models": self.n_reference,
            "n_focal_models": self.n_focal,
            "n_strata": self.n_strata,
            "flagged_items": len(self.flagged),
            "flag_rate": self.flag_rate,
            "worst_items": [
                {
                    "item_id": it.item_id,
                    "delta": round(it.delta, 3),
                    "classification": it.classification,
                    "favors": it.favors,
                    "p_value": round(it.p_value, 5),
                    "accuracy_reference": round(it.p_reference, 3),
                    "accuracy_focal": round(it.p_focal, 3),
                }
                for it in self.worst(10)
            ],
            "interpretation": (
                "Items classified B or C behave differently for the two model "
                "groups at matched ability. Inspect them before trusting the "
                "benchmark to compare these families."
            ),
        }


def mantel_haenszel(
    correct: np.ndarray, is_focal: np.ndarray, strata: np.ndarray
) -> tuple[float, float, float]:
    """Mantel-Haenszel common odds ratio, chi-square and p-value for one item.

    ``correct`` and ``is_focal`` are 0/1 per respondent; ``strata`` holds the
    matching stratum id. Strata that cannot contribute (a single respondent,
    or no variation) are dropped.
    """
    num = den = 0.0
    a_total = expected = variance = 0.0
    used = 0
    for stratum in np.unique(strata):
        mask = strata == stratum
        total = int(mask.sum())
        if total < 2:
            continue
        foc = is_focal[mask].astype(bool)
        cor = correct[mask].astype(bool)
        n_ref, n_foc = int((~foc).sum()), int(foc.sum())
        m_correct = int(cor.sum())
        m_wrong = total - m_correct
        if n_ref == 0 or n_foc == 0 or m_correct == 0 or m_wrong == 0:
            continue  # no contrast available in this stratum
        a = float((~foc & cor).sum())      # reference & correct
        b = float((~foc & ~cor).sum())     # reference & wrong
        c = float((foc & cor).sum())       # focal & correct
        d = float((foc & ~cor).sum())      # focal & wrong
        num += a * d / total
        den += b * c / total
        a_total += a
        expected += n_ref * m_correct / total
        variance += (
            n_ref * n_foc * m_correct * m_wrong / (total**2 * (total - 1))
        )
        used += 1

    if used == 0 or variance <= 0:
        return float("nan"), 0.0, 1.0

    # The chi-square is well defined even when the odds ratio degenerates,
    # so compute it first and never let separation hide a real effect.
    stat = (abs(a_total - expected) - 0.5) ** 2 / variance
    p_value = float(chi2.sf(stat, df=1))

    if num <= 0.0 or den <= 0.0:
        # Perfect separation in one direction: Haldane-Anscombe correction
        # keeps the odds ratio finite and pointing the right way.
        num += 0.5
        den += 0.5
    odds_ratio = num / den
    return float(odds_ratio), float(stat), p_value


def _make_strata(scores: np.ndarray, n_strata: int) -> np.ndarray:
    """Bin respondents into ability strata by matching score (equal-count bins)."""
    n_strata = max(2, min(n_strata, len(np.unique(scores))))
    quantiles = np.linspace(0, 100, n_strata + 1)[1:-1]
    edges = np.percentile(scores, quantiles)
    return np.digitize(scores, edges)


def _matching_scores(X: np.ndarray, studied: int, anchor: np.ndarray) -> np.ndarray:
    """Ability proxy for matching, purified for the item under study.

    Two standard corrections are applied:

    * **rest score** — the studied item is removed from its own matching
      criterion, otherwise a DIF item contaminates the ability it is matched
      on and the effect attenuates toward zero;
    * **anchor purification** — only items believed to be DIF-free (the
      ``anchor`` mask) contribute, so a handful of biased items cannot skew
      the ability scale for everything else.
    """
    use = anchor.copy()
    use[studied] = False
    if use.sum() < 2:  # fall back to leave-one-out on all items
        use = np.ones(X.shape[1], dtype=bool)
        use[studied] = False
    return np.nanmean(X[:, use], axis=1)


def detect_dif(
    responses: np.ndarray,
    groups: np.ndarray,
    bank: ItemBank | None = None,
    *,
    n_strata: int = 4,
    alpha: float = 0.05,
    purify: bool = True,
    reference_name: str = "reference",
    focal_name: str = "focal",
) -> DIFReport:
    """Screen every item for differential functioning between two model groups.

    Parameters
    ----------
    responses:
        ``(n_models, n_items)`` matrix of 0/1 correctness (NaN = not administered).
    groups:
        Length ``n_models``; 0 = reference group, 1 = focal group.
    bank:
        Optional item bank, used only to label items.
    n_strata:
        Number of ability-matching strata. More strata match ability more
        tightly but leave fewer models per cell.
    purify:
        Run the second purification pass, rebuilding the ability scale from
        items the first pass considered clean.

    Notes
    -----
    Matching conditions on observed ability, so a plain accuracy gap between
    strong and weak model families does not, by itself, register as DIF —
    only a gap that survives ability matching does.
    """
    X = np.asarray(responses, dtype=float)
    groups = np.asarray(groups).astype(int)
    if X.ndim != 2:
        raise ValueError("responses must be a 2-D (models x items) matrix")
    if len(groups) != X.shape[0]:
        raise ValueError("groups must have one entry per model (row)")
    if set(np.unique(groups)) - {0, 1}:
        raise ValueError("groups must contain only 0 (reference) and 1 (focal)")

    n_items = X.shape[1]

    def screen(anchor: np.ndarray) -> list[DIFItem]:
        found: list[DIFItem] = []
        for j in range(n_items):
            column = X[:, j]
            observed = ~np.isnan(column)
            strata = _make_strata(_matching_scores(X, j, anchor), n_strata)
            odds_ratio, stat, p_value = mantel_haenszel(
                column[observed], groups[observed], strata[observed]
            )
            if np.isnan(odds_ratio) or odds_ratio <= 0:
                delta, classification, favors = 0.0, "A", "neither"
                odds_ratio = float("nan")
            else:
                # ETS scale: alpha_MH puts the reference group in the numerator,
                # so delta = -2.35 ln(alpha_MH) is POSITIVE when the item
                # favors the focal group.
                delta = -2.35 * np.log(odds_ratio)
                significant = p_value < alpha
                magnitude = abs(delta)
                if not significant or magnitude < 1.0:
                    classification = "A"
                elif magnitude < 1.5:
                    classification = "B"
                else:
                    classification = "C"
                favors = (
                    "neither" if classification == "A"
                    else ("focal" if delta > 0 else "reference")
                )
            ref_mask = observed & (groups == 0)
            foc_mask = observed & (groups == 1)
            found.append(
                DIFItem(
                    item_id=(
                        bank.items[j].id if bank and j < len(bank.items) else f"item/{j}"
                    ),
                    index=j,
                    odds_ratio=float(odds_ratio),
                    delta=float(delta),
                    chi2=float(stat),
                    p_value=float(p_value),
                    classification=classification,
                    favors=favors,
                    n_reference=int(ref_mask.sum()),
                    n_focal=int(foc_mask.sum()),
                    p_reference=(
                        float(np.nanmean(column[ref_mask])) if ref_mask.any() else 0.0
                    ),
                    p_focal=(
                        float(np.nanmean(column[foc_mask])) if foc_mask.any() else 0.0
                    ),
                )
            )
        return found

    # Pass 1: every item anchors the ability scale. Pass 2: rebuild the scale
    # from items pass 1 believes are clean (two-stage purification).
    items = screen(np.ones(n_items, dtype=bool))
    if purify:
        anchor = np.array([not it.flagged for it in items])
        if anchor.sum() >= 2 and anchor.sum() < n_items:
            items = screen(anchor)

    return DIFReport(
        items=items,
        n_items=n_items,
        n_reference=int((groups == 0).sum()),
        n_focal=int((groups == 1).sum()),
        n_strata=len(np.unique(_make_strata(np.nanmean(X, axis=1), n_strata))),
        reference_name=reference_name,
        focal_name=focal_name,
        flagged=[it for it in items if it.flagged],
    )
