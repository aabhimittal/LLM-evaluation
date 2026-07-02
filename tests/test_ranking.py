import numpy as np

from caliper.judge import Match, bootstrap_ratings, fit_bradley_terry


def _synthetic_matches(true: dict[str, float], n: int, seed: int = 0) -> list[Match]:
    rng = np.random.default_rng(seed)
    names = list(true)
    matches = []
    for _ in range(n):
        a, b = rng.choice(names, size=2, replace=False)
        p = 1 / (1 + np.exp(-(true[a] - true[b])))
        matches.append(Match(str(a), str(b), float(rng.random() < p)))
    return matches


def test_bt_recovers_ordering():
    true = {"strong": 1.2, "middle": 0.0, "weak": -1.2}
    strengths = fit_bradley_terry(_synthetic_matches(true, 300))
    assert strengths["strong"] > strengths["middle"] > strengths["weak"]
    assert abs(sum(strengths.values())) < 1e-6  # sum-zero constraint


def test_bootstrap_ratings_cis():
    true = {"strong": 1.0, "weak": -1.0}
    table = bootstrap_ratings(_synthetic_matches(true, 200), n_boot=100, seed=1)
    assert table.sorted_models() == ["strong", "weak"]
    for m in table.models:
        lo, hi = table.ci95[m]
        assert lo <= table.rating[m] <= hi
    # strong's CI should sit above weak's
    assert table.ci95["strong"][0] > table.ci95["weak"][1]


def test_fractional_scores_supported():
    matches = [Match("a", "b", 0.75)] * 40 + [Match("b", "a", 0.25)] * 40
    strengths = fit_bradley_terry(matches)
    assert strengths["a"] > strengths["b"]
