from caliper.adapters import SimulatedSubject
from caliper.contamination import evaluate_contamination, ngram_overlap, token_f1


def test_token_f1():
    assert token_f1("the quick brown fox", "the quick brown fox") == 1.0
    assert token_f1("alpha beta", "gamma delta") == 0.0
    assert 0.0 < token_f1("the quick fox", "the slow fox") < 1.0


def test_ngram_overlap():
    text = "one two three four five six seven eight nine ten"
    assert ngram_overlap(text, [text], n=8) == 1.0
    assert ngram_overlap(text, ["completely different words here for the corpus test"], n=8) == 0.0
    assert ngram_overlap("short", ["short"], n=8) == 0.0  # too short to score


def test_contaminated_model_scores_higher(small_bank):
    clean = SimulatedSubject(theta=0.5, bank=small_bank, contaminated=False, seed=8)
    dirty = SimulatedSubject(theta=0.5, bank=small_bank, contaminated=True, seed=8)
    r_clean = evaluate_contamination(clean, small_bank, n_items=10, seed=0)
    r_dirty = evaluate_contamination(dirty, small_bank, n_items=10, seed=0)
    assert r_dirty.risk > r_clean.risk
    assert r_dirty.continuation_gap > r_clean.continuation_gap
    assert r_dirty.option_recall_rate > r_clean.option_recall_rate
