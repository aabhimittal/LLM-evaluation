import numpy as np
import pytest

from caliper.types import Item, ItemBank


@pytest.fixture(scope="session")
def small_bank() -> ItemBank:
    """A deterministic 40-item bank with known IRT parameters."""
    rng = np.random.default_rng(11)
    items = []
    subjects = ["photosynthesis", "gravity", "electricity", "evolution", "plate tectonics",
                "magnetism", "the water cycle", "cellular respiration"]
    for i in range(40):
        subject = subjects[i % len(subjects)]
        items.append(
            Item(
                id=f"test/{i}",
                question=(
                    f"Question {i}: which statement about {subject} is correct according "
                    f"to standard scientific understanding of process number {i}?"
                ),
                choices=[
                    f"The accurate description of {subject} variant {i}",
                    f"An incorrect claim about {subject} alpha {i}",
                    f"An incorrect claim about {subject} beta {i}",
                    f"An incorrect claim about {subject} gamma {i}",
                ],
                answer_index=int(rng.integers(0, 4)),
                a=float(rng.lognormal(0.0, 0.3)),
                b=float(rng.normal(0.0, 1.0)),
            )
        )
    return ItemBank(items=items, name="test-bank", calibration="known-true")
