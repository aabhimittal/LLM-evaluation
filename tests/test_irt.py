import numpy as np

from caliper.irt import estimate_ability, fit_items, item_information, p_correct
from caliper.irt.adaptive import AdaptiveSession, run_adaptive
from caliper.adapters import SimulatedSubject


def test_p_correct_monotonic_in_theta():
    thetas = np.linspace(-4, 4, 50)
    probs = p_correct(thetas, a=1.2, b=0.3, c=0.25)
    assert np.all(np.diff(probs) > 0)
    assert probs[0] >= 0.25 - 1e-9  # guessing floor
    assert probs[-1] < 1.0


def test_information_peaks_near_difficulty():
    thetas = np.linspace(-4, 4, 400)
    info = item_information(thetas, a=1.5, b=0.8, c=0.0)
    peak = thetas[np.argmax(info)]
    assert abs(peak - 0.8) < 0.1  # with c=0 the 2PL peak is exactly at b


def test_estimate_ability_recovers_truth():
    rng = np.random.default_rng(0)
    n = 200
    a = rng.lognormal(0, 0.3, n)
    b = rng.normal(0, 1, n)
    c = np.full(n, 0.25)
    for true_theta in (-1.0, 0.0, 1.2):
        y = (rng.random(n) < p_correct(true_theta, a, b, c)).astype(float)
        est = estimate_ability(a, b, c, y)
        assert abs(est.theta - true_theta) < 3 * est.se
        assert est.se < 0.35


def test_estimate_ability_empty_returns_prior():
    est = estimate_ability(np.array([]), np.array([]), np.array([]), np.array([]))
    assert est.theta == 0.0
    assert est.se == 1.0


def test_se_shrinks_with_more_items():
    rng = np.random.default_rng(1)
    a = rng.lognormal(0, 0.3, 100)
    b = rng.normal(0, 1, 100)
    c = np.full(100, 0.25)
    y = (rng.random(100) < p_correct(0.5, a, b, c)).astype(float)
    se_small = estimate_ability(a[:10], b[:10], c[:10], y[:10]).se
    se_large = estimate_ability(a, b, c, y).se
    assert se_large < se_small


def test_fit_items_parameter_recovery():
    rng = np.random.default_rng(3)
    n_items, n_resp = 60, 80
    true_a = rng.lognormal(0, 0.3, n_items)
    true_b = rng.normal(0, 1.0, n_items)
    thetas = rng.normal(size=n_resp)
    c = np.full(n_items, 0.25)
    P = p_correct(thetas[:, None], true_a[None, :], true_b[None, :], c[None, :])
    X = (rng.random(P.shape) < P).astype(float)
    result = fit_items(X, n_choices=4)
    corr = np.corrcoef(true_b, result.b)[0, 1]
    assert corr > 0.75  # difficulty ordering is recovered


def test_adaptive_session_converges(small_bank):
    subject = SimulatedSubject(theta=1.0, bank=small_bank, seed=5)
    last = None
    for state in run_adaptive(subject, small_bank, max_items=35, seed=2):
        last = state
    assert last is not None
    lo, hi = last.estimate.ci95
    assert lo < 1.0 < hi or abs(last.estimate.theta - 1.0) < 0.75
    # SE must shrink over the session
    session_history = last  # last state's estimate
    assert session_history.estimate.se < 1.0


def test_adaptive_selects_informative_items(small_bank):
    session = AdaptiveSession(bank=small_bank, seed=0, exposure_k=1)
    item = session.next_item()
    infos = [item_information(0.0, it.a, it.b, it.c) for it in small_bank]
    assert item_information(0.0, item.a, item.b, item.c) == max(infos)
