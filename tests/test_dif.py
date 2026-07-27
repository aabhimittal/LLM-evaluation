import numpy as np

from caliper.dif import detect_dif, mantel_haenszel
from caliper.irt import p_correct


def _dataset(n_models=100, n_items=80, n_dif=8, boost=1.5, seed=4, focal_theta=0.8):
    """Two model groups; the focal group is stronger AND has injected DIF."""
    rng = np.random.default_rng(seed)
    half = n_models // 2
    groups = np.array([0] * half + [1] * (n_models - half))
    theta = np.where(
        groups == 1, rng.normal(focal_theta, 1, n_models), rng.normal(0.0, 1, n_models)
    )
    a = rng.lognormal(0, 0.3, n_items)
    b = rng.normal(0, 1, n_items)
    c = np.full(n_items, 0.25)
    P = p_correct(theta[:, None], a[None, :], b[None, :], c[None, :])
    dif_items = {int(j) for j in rng.choice(n_items, n_dif, replace=False)}
    for j in dif_items:
        P[groups == 1, j] = p_correct(theta[groups == 1], a[j], b[j] - boost, c[j])
    return (rng.random(P.shape) < P).astype(float), groups, dif_items


def test_mantel_haenszel_no_association():
    """Group assignment independent of both stratum and outcome."""
    rng = np.random.default_rng(0)
    n = 800
    correct = (rng.random(n) < 0.5).astype(float)
    is_focal = (rng.random(n) < 0.5).astype(float)
    strata = rng.integers(0, 4, size=n)
    odds_ratio, _, p_value = mantel_haenszel(correct, is_focal, strata)
    assert 0.5 < odds_ratio < 2.0
    assert p_value > 0.05


def test_mantel_haenszel_detects_strong_association():
    """Focal always right, reference always wrong: perfect separation."""
    n_per_stratum = 20
    correct, is_focal, strata = [], [], []
    for s in range(5):
        for _ in range(n_per_stratum // 2):
            correct += [0.0, 1.0]        # reference wrong, focal right
            is_focal += [0.0, 1.0]
            strata += [s, s]
    odds_ratio, stat, p_value = mantel_haenszel(
        np.array(correct), np.array(is_focal), np.array(strata)
    )
    assert stat > 10
    assert p_value < 0.01
    # Separation must still yield a finite OR pointing at the focal group.
    assert np.isfinite(odds_ratio)
    assert -2.35 * np.log(odds_ratio) > 0


def test_detects_injected_dif():
    X, groups, truth = _dataset()
    report = detect_dif(X, groups, n_strata=5)
    flagged = {it.index for it in report.flagged}
    assert len(flagged & truth) >= len(truth) // 2  # majority recovered
    assert len(flagged - truth) <= 5  # few false alarms


def test_dif_sign_points_at_the_favoured_group():
    """Injected items help the focal group, so delta must be positive."""
    X, groups, truth = _dataset()
    report = detect_dif(X, groups, n_strata=5)
    for item in report.flagged:
        if item.index in truth:
            assert item.delta > 0
            assert item.favors == "focal"


def test_ability_gap_alone_does_not_create_dif():
    """The focal group is much stronger but no item is biased: expect no flags."""
    rng = np.random.default_rng(11)
    n_models, n_items = 100, 60
    groups = np.array([0] * 50 + [1] * 50)
    theta = np.where(groups == 1, rng.normal(1.2, 1, n_models), rng.normal(-0.6, 1, n_models))
    a = rng.lognormal(0, 0.3, n_items)
    b = rng.normal(0, 1, n_items)
    c = np.full(n_items, 0.25)
    P = p_correct(theta[:, None], a[None, :], b[None, :], c[None, :])
    X = (rng.random(P.shape) < P).astype(float)
    report = detect_dif(X, groups, n_strata=5)
    assert report.flag_rate < 0.15  # ability matching absorbs the gap


def test_report_serialization_and_worst_ordering():
    X, groups, _ = _dataset(n_models=60, n_items=40, seed=7)
    report = detect_dif(X, groups, n_strata=4)
    payload = report.to_dict()
    assert payload["n_items"] == 40
    assert payload["n_reference_models"] == 30
    assert len(payload["worst_items"]) <= 10
    deltas = [abs(it.delta) for it in report.worst(10)]
    assert deltas == sorted(deltas, reverse=True)


def test_input_validation():
    X = np.zeros((4, 5))
    for bad_groups in (np.array([0, 1, 0]), np.array([0, 1, 2, 0])):
        try:
            detect_dif(X, bad_groups)
        except ValueError:
            continue
        raise AssertionError("expected ValueError for malformed groups")
