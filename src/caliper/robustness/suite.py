"""Run metamorphic perturbations against a model and score invariance."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from caliper.adapters.base import ModelAdapter
from caliper.robustness.perturb import PERTURBATIONS, perturb_choices, perturb_question
from caliper.runner import administer_item
from caliper.types import ItemBank


@dataclass
class RobustnessReport:
    n_items: int
    overall_consistency: float           # fraction of perturbed answers matching baseline
    ci95: tuple[float, float]            # bootstrap over items
    by_perturbation: dict[str, float] = field(default_factory=dict)
    flips: list[dict] = field(default_factory=list)  # worst offenders, for inspection


def evaluate_robustness(
    adapter: ModelAdapter,
    bank: ItemBank | None = None,
    n_items: int = 15,
    perturbations: list[str] | None = None,
    include_option_shuffle: bool = True,
    seed: int = 0,
    n_boot: int = 500,
) -> RobustnessReport:
    """Answer each sampled item unperturbed, then under each perturbation.

    Consistency compares the *semantic* answer (the chosen option's text) so
    option shuffling counts as consistent when the model picks the same
    underlying choice from a different slot.
    """
    bank = bank if bank is not None else ItemBank.bundled()
    names = perturbations if perturbations is not None else list(PERTURBATIONS)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(bank.items), size=min(n_items, len(bank.items)), replace=False)
    items = [bank.items[int(i)] for i in idx]

    per_item_scores: list[float] = []
    by_pert: dict[str, list[float]] = {name: [] for name in names}
    if include_option_shuffle:
        by_pert["option_shuffle"] = []
    flips: list[dict] = []

    for item in items:
        base = administer_item(adapter, item)
        base_choice = (
            item.choices[base.parsed_answer] if base.parsed_answer is not None else None
        )
        scores = []
        for name in names:
            q = perturb_question(name, item.question, np.random.default_rng(seed + hash(name) % 1000))
            res = administer_item(adapter, item, question_override=q)
            choice = item.choices[res.parsed_answer] if res.parsed_answer is not None else None
            same = float(choice is not None and choice == base_choice)
            by_pert[name].append(same)
            scores.append(same)
            if not same:
                flips.append({"item_id": item.id, "perturbation": name,
                              "baseline": base_choice, "perturbed": choice})
        if include_option_shuffle:
            shuffled, new_answer = perturb_choices(
                item.choices, item.answer_index, np.random.default_rng(seed + 13)
            )
            res = administer_item(
                adapter, item, choices_override=shuffled, answer_index_override=new_answer
            )
            choice = shuffled[res.parsed_answer] if res.parsed_answer is not None else None
            same = float(choice is not None and choice == base_choice)
            by_pert["option_shuffle"].append(same)
            scores.append(same)
            if not same:
                flips.append({"item_id": item.id, "perturbation": "option_shuffle",
                              "baseline": base_choice, "perturbed": choice})
        per_item_scores.append(float(np.mean(scores)))

    overall = float(np.mean(per_item_scores)) if per_item_scores else 0.0
    boot = np.random.default_rng(seed)
    if per_item_scores:
        arr = np.asarray(per_item_scores)
        means = [
            float(np.mean(arr[boot.integers(0, len(arr), size=len(arr))]))
            for _ in range(n_boot)
        ]
        ci = (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))
    else:
        ci = (0.0, 0.0)
    return RobustnessReport(
        n_items=len(items),
        overall_consistency=overall,
        ci95=ci,
        by_perturbation={k: float(np.mean(v)) if v else 0.0 for k, v in by_pert.items()},
        flips=flips[:20],
    )


def free_text_consistency(
    adapter: ModelAdapter,
    prompts: list[str],
    perturbation: str = "paraphrase",
    seed: int = 0,
) -> float:
    """Semantic self-consistency for open-ended prompts.

    Generates a response to each prompt and to its perturbed twin, then
    scores mean cosine similarity of their embeddings (HF feature-extraction
    task, or the adapter's own embedding fallback).
    """
    rng = np.random.default_rng(seed)
    sims = []
    for prompt in prompts:
        twin = perturb_question(perturbation, prompt, rng)
        r1 = adapter.ask(prompt, temperature=0.0, max_tokens=200)
        r2 = adapter.ask(twin, temperature=0.0, max_tokens=200)
        vecs = adapter.embed([r1, r2])
        denom = np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1])
        sims.append(float(np.dot(vecs[0], vecs[1]) / denom) if denom > 0 else 0.0)
    return float(np.mean(sims)) if sims else 0.0
