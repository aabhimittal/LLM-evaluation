import numpy as np
import pytest

from caliper.adapters import ReplayAdapter, SimulatedSubject
from caliper.runner import administer_item


def test_simulated_subject_ability_ordering(small_bank):
    strong = SimulatedSubject(theta=2.5, bank=small_bank, seed=1)
    weak = SimulatedSubject(theta=-2.5, bank=small_bank, seed=1)
    strong_correct = sum(administer_item(strong, it).correct for it in small_bank)
    weak_correct = sum(administer_item(weak, it).correct for it in small_bank)
    assert strong_correct > weak_correct


def test_simulated_subject_deterministic(small_bank):
    s1 = SimulatedSubject(theta=0.3, bank=small_bank, seed=9)
    s2 = SimulatedSubject(theta=0.3, bank=small_bank, seed=9)
    item = small_bank.items[0]
    assert administer_item(s1, item).raw_response == administer_item(s2, item).raw_response


def test_replay_records_and_replays(tmp_path, small_bank):
    cache = tmp_path / "cache.json"
    subject = SimulatedSubject(theta=0.3, bank=small_bank, seed=9)
    recorder = ReplayAdapter(cache_path=cache, record=subject)
    item = small_bank.items[3]
    first = administer_item(recorder, item)
    assert cache.exists()

    replayer = ReplayAdapter(cache_path=cache)  # no backend attached
    second = administer_item(replayer, item)
    assert second.raw_response == first.raw_response


def test_replay_miss_raises(tmp_path):
    replayer = ReplayAdapter(cache_path=tmp_path / "empty.json")
    with pytest.raises(KeyError):
        replayer.ask("never recorded")


def test_replay_embeddings_roundtrip(tmp_path, small_bank):
    cache = tmp_path / "cache.json"
    subject = SimulatedSubject(theta=0.3, bank=small_bank, seed=9)
    recorder = ReplayAdapter(cache_path=cache, record=subject)
    v1 = recorder.embed(["hello world", "another text"])
    replayer = ReplayAdapter(cache_path=cache)
    v2 = replayer.embed(["hello world", "another text"])
    assert np.allclose(v1, v2)
