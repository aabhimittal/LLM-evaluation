from caliper.parsing import parse_confidence, parse_judge_verdict, parse_mcq_answer


def test_parse_mcq_shapes():
    assert parse_mcq_answer("B", 4) == 1
    assert parse_mcq_answer("(C)", 4) == 2
    assert parse_mcq_answer("Answer: D", 4) == 3
    assert parse_mcq_answer("The answer is A.", 4) == 0
    assert parse_mcq_answer("**B**. Because…", 4) == 1
    assert parse_mcq_answer("b\nConfidence: 80", 4) == 1
    assert parse_mcq_answer("I don't know", 4) is None
    assert parse_mcq_answer("E", 4) is None  # out of range for 4 choices
    assert parse_mcq_answer("", 4) is None


def test_parse_confidence():
    assert parse_confidence("B\nConfidence: 85") == 0.85
    assert parse_confidence("Confidence: 0.4") == 0.4
    assert parse_confidence("confidence = 110") == 1.0  # clipped
    assert parse_confidence("no number here") is None


def test_parse_judge_verdict():
    assert parse_judge_verdict('{"winner": "A", "reason": "x"}') == "A"
    assert parse_judge_verdict('some text {"winner": "tie"} more') == "tie"
    assert parse_judge_verdict("The winner is B") == "B"
    assert parse_judge_verdict("no verdict at all") is None
