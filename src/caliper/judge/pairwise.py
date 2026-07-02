"""Pairwise judging with debiasing and per-verdict uncertainty.

Every comparison is run in *both* presentation orders and sampled several
times at nonzero temperature. That yields:

- a debiased win probability (averaging over orders cancels position bias),
- a per-verdict confidence (vote agreement across samples),
- a position-flip flag (did the majority verdict change with the order?),

and across many comparisons an audit of the judge itself: position-flip
rate and the correlation between verbosity and verdicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from caliper.adapters.base import ModelAdapter
from caliper.parsing import parse_judge_verdict
from caliper.prompts import JUDGE_SYSTEM, format_judge


@dataclass
class PairwiseVerdict:
    prompt: str
    response_a: str
    response_b: str
    p_a_wins: float          # debiased win probability for response A
    confidence: float        # vote agreement in [0, 1]
    position_flip: bool      # majority verdict changed with presentation order
    n_votes: int
    votes: list[dict] = field(default_factory=list)
    unparseable: int = 0

    @property
    def winner(self) -> str:
        if abs(self.p_a_wins - 0.5) < 1e-9:
            return "tie"
        return "A" if self.p_a_wins > 0.5 else "B"


@dataclass
class JudgeAudit:
    n_comparisons: int
    position_flip_rate: float
    verbosity_bias: float    # correlation(sign of length diff, outcome)
    mean_confidence: float


class PairwiseJudge:
    def __init__(
        self,
        adapter: ModelAdapter,
        n_samples: int = 3,
        temperature: float = 0.6,
    ):
        self.adapter = adapter
        self.n_samples = n_samples
        self.temperature = temperature
        self._verdicts: list[PairwiseVerdict] = []

    def compare(self, prompt: str, response_a: str, response_b: str) -> PairwiseVerdict:
        votes: list[dict] = []
        unparseable = 0
        scores = {"first": [], "swapped": []}
        for order in ("first", "swapped"):
            if order == "first":
                pa, pb = response_a, response_b
            else:
                pa, pb = response_b, response_a
            user = format_judge(prompt, pa, pb)
            for s in range(self.n_samples):
                raw = self.adapter.chat(
                    [
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": user},
                    ],
                    temperature=self.temperature,
                    max_tokens=200,
                    seed=s,
                )
                verdict = parse_judge_verdict(raw)
                if verdict is None:
                    unparseable += 1
                    continue
                # Map slot verdict back to the identity of response A.
                if verdict == "tie":
                    score_for_a = 0.5
                elif (verdict == "A") == (order == "first"):
                    score_for_a = 1.0
                else:
                    score_for_a = 0.0
                scores[order].append(score_for_a)
                votes.append({"order": order, "sample": s, "verdict": verdict,
                              "score_for_a": score_for_a})

        all_scores = scores["first"] + scores["swapped"]
        p_a = float(np.mean(all_scores)) if all_scores else 0.5
        # Confidence: agreement of votes with the majority direction.
        if all_scores:
            majority = 1.0 if p_a >= 0.5 else 0.0
            agreement = float(np.mean([
                1.0 if (s > 0.5) == (majority > 0.5) or s == 0.5 else 0.0
                for s in all_scores
            ]))
        else:
            agreement = 0.0
        flip = bool(
            scores["first"] and scores["swapped"]
            and (np.mean(scores["first"]) > 0.5) != (np.mean(scores["swapped"]) > 0.5)
        )
        verdict = PairwiseVerdict(
            prompt=prompt,
            response_a=response_a,
            response_b=response_b,
            p_a_wins=p_a,
            confidence=agreement,
            position_flip=flip,
            n_votes=len(all_scores),
            votes=votes,
            unparseable=unparseable,
        )
        self._verdicts.append(verdict)
        return verdict

    def audit(self) -> JudgeAudit:
        """Bias audit over every comparison this judge has made."""
        if not self._verdicts:
            return JudgeAudit(0, 0.0, 0.0, 0.0)
        flips = float(np.mean([v.position_flip for v in self._verdicts]))
        conf = float(np.mean([v.confidence for v in self._verdicts]))
        len_signs, outcomes = [], []
        for v in self._verdicts:
            diff = len(v.response_a) - len(v.response_b)
            if diff == 0 or v.p_a_wins == 0.5:
                continue
            len_signs.append(np.sign(diff))
            outcomes.append(np.sign(v.p_a_wins - 0.5))
        if len(len_signs) >= 2 and np.std(len_signs) > 0 and np.std(outcomes) > 0:
            verbosity = float(np.corrcoef(len_signs, outcomes)[0, 1])
        else:
            verbosity = 0.0
        return JudgeAudit(
            n_comparisons=len(self._verdicts),
            position_flip_rate=flips,
            verbosity_bias=verbosity,
            mean_confidence=conf,
        )
