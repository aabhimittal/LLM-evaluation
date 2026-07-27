import numpy as np

from caliper.adapters import SimulatedJudge
from caliper.judge import PairwiseJudge, cohen_kappa, judge_report_card


def _comparisons(n=60, seed=0):
    rng = np.random.default_rng(seed)
    prompts, resp_a, resp_b, gold = [], [], [], []
    for i in range(n):
        prompt = f"Question {i} about energy transfer in physical systems"
        a_better = rng.random() < 0.5
        good = prompt + " — a correct, detailed answer covering energy transfer."
        bad = "Short vague reply."
        prompts.append(prompt)
        resp_a.append(good if a_better else bad)
        resp_b.append(bad if a_better else good)
        gold.append("A" if a_better else "B")
    return prompts, resp_a, resp_b, gold


def _run(judge_model, n=60, seed=0):
    prompts, resp_a, resp_b, gold = _comparisons(n, seed)
    judge = PairwiseJudge(judge_model, n_samples=3)
    verdicts = [judge.compare(p, a, b) for p, a, b in zip(prompts, resp_a, resp_b)]
    return judge_report_card(verdicts, gold)


def test_cohen_kappa_edges():
    assert cohen_kappa(["A", "B", "A"], ["A", "B", "A"]) == 1.0
    # A constant labeler that matches the majority still earns kappa 0.
    assert abs(cohen_kappa(["A"] * 10, ["A"] * 7 + ["B"] * 3)) < 1e-9
    assert cohen_kappa([], []) == 0.0


def test_good_judge_scores_well():
    card = _run(SimulatedJudge(accuracy=0.95, seed=1))
    assert card.kappa > 0.7
    assert card.accuracy > 0.8
    assert card.grade.startswith("A")


def test_random_judge_is_graded_down():
    card = _run(SimulatedJudge(accuracy=0.5, seed=1))
    assert card.kappa < 0.4
    assert card.grade.startswith(("C", "D"))


def test_unstable_judge_loses_a_grade():
    """Debiasing may rescue the ranking, but flipping verdicts costs a notch."""
    card = _run(SimulatedJudge(accuracy=0.92, position_bias=0.45, seed=1))
    assert card.position_flip_rate > 0.25
    assert not card.grade.startswith("A")
    assert "flip" in card.grade


def test_confidence_predicts_correctness_for_a_good_judge():
    card = _run(SimulatedJudge(accuracy=0.9, seed=2))
    assert card.confidence_auc >= 0.5
    assert card.mean_confidence_correct >= card.mean_confidence_wrong


def test_report_card_serializes_and_lists_disagreements():
    card = _run(SimulatedJudge(accuracy=0.7, seed=3))
    payload = card.to_dict()
    assert payload["n_comparisons"] == 60
    assert "cohens_kappa" in payload
    assert set(payload["position_bias"]) == {
        "judge_picks_slot_a", "humans_pick_slot_a", "excess"
    }
    assert len(card.disagreements) <= 15
    for row in card.disagreements:
        assert row["judge"] != row["gold"]


def test_length_mismatch_raises():
    card_inputs = _comparisons(4)
    judge = PairwiseJudge(SimulatedJudge(seed=0), n_samples=1)
    verdicts = [
        judge.compare(p, a, b)
        for p, a, b in zip(card_inputs[0], card_inputs[1], card_inputs[2])
    ]
    try:
        judge_report_card(verdicts, ["A"])
    except ValueError:
        return
    raise AssertionError("expected ValueError on mismatched lengths")


def test_empty_input_is_safe():
    card = judge_report_card([], [])
    assert card.n == 0
    assert card.grade.startswith("D")


def _write_jsonl(path, n=10):
    import json

    prompts, resp_a, resp_b, gold = _comparisons(n, seed=5)
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(
            json.dumps({"prompt": p, "response_a": a, "response_b": b, "gold": g}) + "\n"
            for p, a, b, g in zip(prompts, resp_a, resp_b, gold)
        )


def test_cli_judge_card(tmp_path, capsys):
    import json

    from caliper.cli import main

    data = tmp_path / "gold.jsonl"
    _write_jsonl(data)
    rc = main([
        "judge-card", "--adapter", "simulated", "--data", str(data),
        "--judge-accuracy", "0.95", "--seed", "1",
    ])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_comparisons"] == 10
    assert payload["cohens_kappa"] > 0.5
    assert payload["grade"].startswith("A")


def test_cli_judge_card_rejects_bad_schema(tmp_path, capsys):
    from caliper.cli import main

    data = tmp_path / "bad.jsonl"
    data.write_text('{"prompt": "hi", "gold": "A"}\n', encoding="utf-8")
    assert main(["judge-card", "--adapter", "simulated", "--data", str(data)]) == 2


def test_bundled_example_dataset_is_well_formed():
    """The shipped example must stay loadable and correctly labeled."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "examples" / "gold_preferences.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(rows) >= 10
    for row in rows:
        assert set(row) == {"prompt", "response_a", "response_b", "gold"}
        assert row["gold"] in ("A", "B", "tie")
    # Labels must not be degenerate (all A or all B).
    golds = {row["gold"] for row in rows}
    assert len(golds) > 1
