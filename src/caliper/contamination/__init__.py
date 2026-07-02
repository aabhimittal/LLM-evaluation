"""Benchmark-contamination probes.

A model that saw the benchmark during training behaves differently from one
that merely knows the subject:

- **Continuation probe**: given the first ~60% of a question, a contaminated
  model completes the remainder near-verbatim. We compare similarity of its
  continuation against the true remainder vs. a control remainder from a
  different item; the *gap* is the signal, which controls for generic
  completion fluency.
- **Option-recall probe**: asked to reproduce a question's original
  multiple-choice options, only a memorizing model can recover the exact
  distractors (the wrong options carry no semantic signal).
- ``ngram_overlap`` is a utility for screening a user-supplied corpus.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from caliper.adapters.base import ModelAdapter
from caliper.prompts import format_completion_probe, format_option_recall
from caliper.types import ItemBank


@dataclass
class ContaminationReport:
    n_items: int
    continuation_gap: float      # mean(sim to true remainder) - mean(sim to control)
    exact_continuation_rate: float
    option_recall_rate: float    # fraction of true distractors reproduced
    risk: float                  # combined score in [0, 1]
    details: list[dict] = field(default_factory=list)


def _norm_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def token_f1(a: str, b: str) -> float:
    ta, tb = _norm_tokens(a), _norm_tokens(b)
    if not ta or not tb:
        return 0.0
    common = {}
    for t in ta:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in tb:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    precision = overlap / len(tb)
    recall = overlap / len(ta)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def ngram_overlap(text: str, corpus: list[str], n: int = 8) -> float:
    """Fraction of ``n``-grams of ``text`` appearing verbatim in the corpus."""
    toks = _norm_tokens(text)
    if len(toks) < n:
        return 0.0
    grams = {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}
    corpus_grams: set[str] = set()
    for doc in corpus:
        dt = _norm_tokens(doc)
        corpus_grams.update(" ".join(dt[i : i + n]) for i in range(max(0, len(dt) - n + 1)))
    return len(grams & corpus_grams) / len(grams)


def evaluate_contamination(
    adapter: ModelAdapter,
    bank: ItemBank | None = None,
    n_items: int = 20,
    seed: int = 0,
    min_words: int = 15,
) -> ContaminationReport:
    bank = bank if bank is not None else ItemBank.bundled()
    rng = np.random.default_rng(seed)
    eligible = [it for it in bank if len(it.question.split()) >= min_words]
    if len(eligible) < 4:
        return ContaminationReport(0, 0.0, 0.0, 0.0, 0.0)
    idx = rng.choice(len(eligible), size=min(n_items, len(eligible)), replace=False)
    items = [eligible[int(i)] for i in idx]

    sims_true, sims_control, exact = [], [], []
    recall_scores = []
    details = []
    for k, item in enumerate(items):
        words = item.question.split()
        cut = max(8, int(len(words) * 0.6))
        prefix = " ".join(words[:cut])
        remainder = " ".join(words[cut:])
        completion = adapter.ask(format_completion_probe(prefix), temperature=0.0, max_tokens=80)

        # Control: remainder of a different eligible item (fluency baseline).
        other = items[(k + 1) % len(items)]
        ow = other.question.split()
        control_remainder = " ".join(ow[max(8, int(len(ow) * 0.6)):])

        s_true = token_f1(completion, remainder)
        s_ctrl = token_f1(completion, control_remainder)
        sims_true.append(s_true)
        sims_control.append(s_ctrl)
        is_exact = _norm_tokens(completion)[: len(_norm_tokens(remainder))] == _norm_tokens(
            remainder
        ) and len(_norm_tokens(remainder)) > 3
        exact.append(1.0 if is_exact else 0.0)

        # Option recall: can the model reproduce the original distractors?
        recall = adapter.ask(format_option_recall(item.question), temperature=0.0, max_tokens=120)
        recalled = 0
        for choice in item.choices:
            best = max(
                (token_f1(line, choice) for line in recall.splitlines() if line.strip()),
                default=0.0,
            )
            if best > 0.8:
                recalled += 1
        recall_scores.append(recalled / len(item.choices))
        details.append({"item_id": item.id, "sim_true": s_true, "sim_control": s_ctrl,
                        "exact": bool(is_exact), "option_recall": recall_scores[-1]})

    gap = float(np.mean(sims_true) - np.mean(sims_control))
    exact_rate = float(np.mean(exact))
    recall_rate = float(np.mean(recall_scores))
    # Combine: gap saturates around 0.5; exact matches and option recall are
    # strong signals on their own.
    risk = float(np.clip(2.0 * max(gap, 0.0), 0, 1) * 0.4
                 + exact_rate * 0.3 + np.clip(recall_rate * 1.5, 0, 1) * 0.3)
    return ContaminationReport(
        n_items=len(items),
        continuation_gap=gap,
        exact_continuation_rate=exact_rate,
        option_recall_rate=recall_rate,
        risk=risk,
        details=details[:20],
    )
