import json

import numpy as np

from caliper.adapters import SimulatedSubject
from caliper.cli import main
from caliper.report import render_html, run_fingerprint


def test_fingerprint_and_html(small_bank):
    subject = SimulatedSubject(theta=0.8, bank=small_bank, seed=2)
    fp = run_fingerprint(
        subject, small_bank,
        adaptive_max_items=12, robustness_items=4,
        calibration_items=10, contamination_items=5, seed=0,
    )
    dims = fp.dimensions()
    assert set(dims) == {"Ability", "Robustness", "Calibration", "Selective risk",
                         "Cleanliness"}
    assert all(0.0 <= v <= 1.0 for v in dims.values())
    payload = json.loads(fp.to_json())
    assert payload["ability"]["n_items"] == 12

    html = render_html(fp)
    assert "<svg" in html and "Caliper fingerprint" in html
    assert fp.model_name in html


def test_cli_adaptive_suite(tmp_path, capsys):
    rc = main([
        "run", "--adapter", "simulated", "--theta", "0.5", "--suite", "adaptive",
        "--max-items", "10", "--out", str(tmp_path), "--seed", "1",
    ])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_items"] == 10
    assert "theta" in payload and "ci95" in payload


def test_cli_calibrate(tmp_path, small_bank, capsys):
    bank_path = tmp_path / "bank.json"
    small_bank.save(bank_path)
    rng = np.random.default_rng(0)
    from caliper.irt import p_correct

    thetas = rng.normal(size=30)
    a = np.array([it.a for it in small_bank])
    b = np.array([it.b for it in small_bank])
    c = np.array([it.c for it in small_bank])
    P = p_correct(thetas[:, None], a[None, :], b[None, :], c[None, :])
    X = (rng.random(P.shape) < P).astype(int)
    matrix_path = tmp_path / "matrix.csv"
    np.savetxt(matrix_path, X, delimiter=",", fmt="%d")

    rc = main([
        "calibrate", "--matrix", str(matrix_path), "--bank", str(bank_path),
        "--out", str(tmp_path / "recal.json"), "--label", "test-recal",
    ])
    assert rc == 0
    from caliper.types import ItemBank

    recal = ItemBank.load(tmp_path / "recal.json")
    assert recal.calibration == "test-recal"
    fitted_b = np.array([it.b for it in recal])
    assert np.corrcoef(b, fitted_b)[0, 1] > 0.5
