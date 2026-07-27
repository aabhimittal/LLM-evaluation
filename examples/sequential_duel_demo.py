"""Why anytime-valid testing matters, in one script.

Runs many *null* comparisons (two identical models) while peeking after every
observation, and counts how often each method wrongly declares a winner.

Run: python examples/sequential_duel_demo.py
"""

import numpy as np

from caliper.adapters import SimulatedSubject
from caliper.sequential import SequentialDuel, run_item_duel
from caliper.types import ItemBank

ALPHA, N_SIM, MAX_STEPS = 0.05, 300, 300


def null_outcome(rng: np.random.Generator) -> float:
    """Two equal models: discordant pairs split evenly, the rest tie."""
    u = rng.random()
    return 1.0 if u < 0.25 else (0.0 if u < 0.5 else 0.5)


def false_positive_rates() -> tuple[float, float]:
    evalue_fires = naive_fires = 0
    for s in range(N_SIM):
        rng = np.random.default_rng(1000 + s)
        duel = SequentialDuel(alpha=ALPHA)
        observations: list[float] = []
        naive_fired = False
        for _ in range(MAX_STEPS):
            x = null_outcome(rng)
            observations.append(x)
            duel.update(x)
            if len(observations) >= 10 and not naive_fired:
                arr = np.asarray(observations)
                se = arr.std(ddof=1) / np.sqrt(len(arr))
                if se > 0 and abs(arr.mean() - 0.5) / se > 1.96:
                    naive_fired = True
            if duel.decided:
                break
        evalue_fires += duel.decided
        naive_fires += naive_fired
    return evalue_fires / N_SIM, naive_fires / N_SIM


def main() -> None:
    e_rate, naive_rate = false_positive_rates()
    print("Two IDENTICAL models, peeking after every observation:")
    print(f"  naive repeated z-test : {naive_rate:.1%} false positives")
    print(f"  Caliper e-process     : {e_rate:.1%} false positives "
          f"(guaranteed <= {ALPHA:.0%})")

    print("\nNow a real difference — how early can we stop?")
    bank = ItemBank.bundled()
    strong = SimulatedSubject(theta=0.6, bank=bank, seed=0, name="strong")
    weak = SimulatedSubject(theta=-1.2, bank=bank, seed=1, name="weak")
    last = None
    for state in run_item_duel(strong, weak, bank, max_items=200, seed=0):
        last = state
    if last and last.decided:
        winner = strong.name if last.winner == "A" else weak.name
        print(f"  decided after {last.step} of 200 items -> {winner} "
              f"({100 * (1 - last.step / 200):.0f}% of the budget saved)")
    else:
        print("  inconclusive within the budget — the models are too close")


if __name__ == "__main__":
    main()
