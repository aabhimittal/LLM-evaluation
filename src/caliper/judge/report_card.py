"""Validate the judge itself against gold (human) preferences.

Papers routinely report "GPT-4 agrees with humans 80% of the time" and stop
there. Raw agreement is a bad summary: if 70% of gold labels favor A, a judge
that always says A scores 70% while knowing nothing. This module reports the
things that actually decide whether a judge is usable:

* **Cohen's kappa** — agreement corrected for chance
* **agreement on decisive pairs** — ties excluded, where the judge earns its keep
* **bias vs. gold** — does the judge favor slot A, or longer answers, *more
  than the humans do*? (Humans have a verbosity preference too; the judge's
  bias is only the excess.)
* **confidence calibration** — does the judge's own vote agreement predict
  whether it is right? A judge that is confidently wrong is worse than one
  that admits uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from caliper.judge.pairwise import PairwiseVerdict

__all__ = ["JudgeReportCard", "cohen_kappa", "judge_report_card"]


def _label(score: float, tie_band: float) -> str:
    if abs(score - 0.5) <= tie_band:
        return "tie"
    return "A" if score > 0.5 else "B"


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Cohen's kappa between two categorical labelings."""
    if not labels_a or len(labels_a) != len(labels_b):
        return 0.0
    categories = sorted(set(labels_a) | set(labels_b))
    index = {c: i for i, c in enumerate(categories)}
    n = len(labels_a)
    confusion = np.zeros((len(categories), len(categories)))
    for x, y in zip(labels_a, labels_b):
        confusion[index[x], index[y]] += 1
    observed = np.trace(confusion) / n
    expected = float(
        np.sum(confusion.sum(axis=0) * confusion.sum(axis=1)) / (n * n)
    )
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return float((observed - expected) / (1.0 - expected))


@dataclass
class JudgeReportCard:
    n: int
    accuracy: float                  # exact label match, ties included
    accuracy_decisive: float         # gold-decisive pairs only
    kappa: float
    n_decisive: int
    judge_position_rate: float       # share of decisive verdicts going to slot A
    gold_position_rate: float
    excess_position_bias: float      # judge minus gold
    judge_verbosity_rate: float      # share of decisive verdicts to the longer reply
    gold_verbosity_rate: float
    excess_verbosity_bias: float
    confidence_auc: float            # does judge confidence predict correctness?
    mean_confidence_correct: float
    mean_confidence_wrong: float
    position_flip_rate: float
    disagreements: list[dict] = field(default_factory=list)

    @property
    def grade(self) -> str:
        """Letter grade from chance-corrected agreement, penalised for instability.

        A high position-flip rate means individual verdicts are unreliable even
        when averaging still recovers the right ranking — that costs a notch,
        because it means single comparisons must never be quoted.
        """
        if self.kappa >= 0.6 and abs(self.excess_position_bias) < 0.1:
            base = 0
        elif self.kappa >= 0.4:
            base = 1
        elif self.kappa >= 0.2:
            base = 2
        else:
            base = 3
        unstable = self.position_flip_rate > 0.25
        if unstable:
            base = min(base + 1, 3)
        grades = [
            "A — trustworthy for ranking",
            "B — usable, report intervals and audit periodically",
            "C — weak agreement; do not rank models on this judge alone",
            "D — no better than chance; replace the judge or the rubric",
        ]
        note = (
            f" (verdicts flip with presentation order {self.position_flip_rate:.0%} "
            "of the time — never quote a single comparison)"
            if unstable else ""
        )
        return grades[base] + note

    def to_dict(self) -> dict:
        return {
            "n_comparisons": self.n,
            "accuracy": round(self.accuracy, 4),
            "accuracy_on_decisive_pairs": round(self.accuracy_decisive, 4),
            "cohens_kappa": round(self.kappa, 4),
            "n_decisive": self.n_decisive,
            "position_bias": {
                "judge_picks_slot_a": round(self.judge_position_rate, 4),
                "humans_pick_slot_a": round(self.gold_position_rate, 4),
                "excess": round(self.excess_position_bias, 4),
            },
            "verbosity_bias": {
                "judge_picks_longer": round(self.judge_verbosity_rate, 4),
                "humans_pick_longer": round(self.gold_verbosity_rate, 4),
                "excess": round(self.excess_verbosity_bias, 4),
            },
            "confidence": {
                "auc_predicting_correctness": round(self.confidence_auc, 4),
                "mean_when_correct": round(self.mean_confidence_correct, 4),
                "mean_when_wrong": round(self.mean_confidence_wrong, 4),
            },
            "position_flip_rate": round(self.position_flip_rate, 4),
            "grade": self.grade,
        }


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC via the rank (Mann-Whitney) identity; 0.5 = uninformative."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    # Average ranks over ties.
    values = np.concatenate([pos, neg])
    for value in np.unique(values):
        mask = values == value
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    rank_sum = ranks[: len(pos)].sum()
    return float((rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def judge_report_card(
    verdicts: list[PairwiseVerdict],
    gold: list[str],
    tie_band: float = 0.1,
) -> JudgeReportCard:
    """Score a judge's verdicts against gold labels.

    ``gold`` holds ``"A"``, ``"B"`` or ``"tie"`` per comparison, in the same
    order as ``verdicts``. ``tie_band`` is the half-width around 0.5 within
    which a debiased win probability counts as a tie.
    """
    if len(verdicts) != len(gold):
        raise ValueError("verdicts and gold must be the same length")
    if not verdicts:
        return JudgeReportCard(
            0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0
        )

    judge_labels = [_label(v.p_a_wins, tie_band) for v in verdicts]
    gold_labels = [str(g).strip().lower() for g in gold]
    gold_labels = [
        "tie" if g == "tie" else ("A" if g.upper() == "A" else "B") for g in gold_labels
    ]

    correct = np.array([j == g for j, g in zip(judge_labels, gold_labels)], dtype=float)
    accuracy = float(correct.mean())
    kappa = cohen_kappa(judge_labels, gold_labels)

    decisive = [i for i, g in enumerate(gold_labels) if g != "tie"]
    accuracy_decisive = (
        float(np.mean([correct[i] for i in decisive])) if decisive else 0.0
    )

    # Position bias measured only where the judge committed to a side.
    judge_sided = [i for i, j in enumerate(judge_labels) if j != "tie"]
    judge_position = (
        float(np.mean([judge_labels[i] == "A" for i in judge_sided]))
        if judge_sided else 0.5
    )
    gold_position = (
        float(np.mean([gold_labels[i] == "A" for i in decisive])) if decisive else 0.5
    )

    def picks_longer(labels: list[str], indices: list[int]) -> float:
        picks = []
        for i in indices:
            v = verdicts[i]
            la, lb = len(v.response_a), len(v.response_b)
            if la == lb or labels[i] == "tie":
                continue
            longer = "A" if la > lb else "B"
            picks.append(labels[i] == longer)
        return float(np.mean(picks)) if picks else 0.5

    judge_verbosity = picks_longer(judge_labels, judge_sided)
    gold_verbosity = picks_longer(gold_labels, decisive)

    confidences = np.array([v.confidence for v in verdicts])
    auc = _auc(confidences, correct.astype(int))
    conf_correct = float(confidences[correct == 1].mean()) if (correct == 1).any() else 0.0
    conf_wrong = float(confidences[correct == 0].mean()) if (correct == 0).any() else 0.0

    disagreements = [
        {
            "prompt": verdicts[i].prompt[:160],
            "judge": judge_labels[i],
            "gold": gold_labels[i],
            "p_a_wins": round(verdicts[i].p_a_wins, 3),
            "judge_confidence": round(verdicts[i].confidence, 3),
            "position_flip": verdicts[i].position_flip,
        }
        for i in range(len(verdicts))
        if correct[i] == 0
    ][:15]

    return JudgeReportCard(
        n=len(verdicts),
        accuracy=accuracy,
        accuracy_decisive=accuracy_decisive,
        kappa=kappa,
        n_decisive=len(decisive),
        judge_position_rate=judge_position,
        gold_position_rate=gold_position,
        excess_position_bias=judge_position - gold_position,
        judge_verbosity_rate=judge_verbosity,
        gold_verbosity_rate=gold_verbosity,
        excess_verbosity_bias=judge_verbosity - gold_verbosity,
        confidence_auc=auc,
        mean_confidence_correct=conf_correct,
        mean_confidence_wrong=conf_wrong,
        position_flip_rate=float(np.mean([v.position_flip for v in verdicts])),
        disagreements=disagreements,
    )
