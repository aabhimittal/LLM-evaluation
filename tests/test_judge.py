from caliper.adapters import SimulatedJudge
from caliper.judge import PairwiseJudge


PROMPTS = [
    f"Explain how {topic} works in the context of energy transfer"
    for topic in ("photosynthesis", "combustion", "wind turbines", "solar panels",
                  "hydro dams", "geothermal wells", "tidal power", "nuclear fission",
                  "muscle contraction", "battery storage", "heat pumps", "fuel cells")
]


def _run(judge: PairwiseJudge) -> None:
    for prompt in PROMPTS:
        good = f"{prompt}: a precise explanation covering energy transfer mechanisms."
        bad = "Not sure."
        judge.compare(prompt, good, bad)


def test_debiased_judge_prefers_better_response():
    judge = PairwiseJudge(SimulatedJudge(accuracy=0.95, seed=0))
    verdict = judge.compare(
        PROMPTS[0],
        f"{PROMPTS[0]}: detailed correct explanation about energy transfer.",
        "Irrelevant.",
    )
    assert verdict.p_a_wins > 0.5
    assert verdict.winner == "A"
    assert 0.0 <= verdict.confidence <= 1.0


def test_swapped_inputs_swap_winner():
    judge = PairwiseJudge(SimulatedJudge(accuracy=0.95, seed=0))
    good = f"{PROMPTS[1]}: detailed correct explanation about energy transfer."
    bad = "Irrelevant."
    forward = judge.compare(PROMPTS[1], good, bad)
    backward = judge.compare(PROMPTS[1], bad, good)
    assert forward.p_a_wins > 0.5 > backward.p_a_wins


def test_position_bias_is_detected():
    biased = PairwiseJudge(SimulatedJudge(accuracy=0.8, position_bias=0.5, seed=1))
    clean = PairwiseJudge(SimulatedJudge(accuracy=0.8, position_bias=0.0, seed=1))
    _run(biased)
    _run(clean)
    assert biased.audit().position_flip_rate > clean.audit().position_flip_rate


def test_audit_empty_judge():
    judge = PairwiseJudge(SimulatedJudge())
    audit = judge.audit()
    assert audit.n_comparisons == 0
    assert audit.position_flip_rate == 0.0
