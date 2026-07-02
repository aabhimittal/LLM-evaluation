"""Offline demo: fingerprint two simulated models and compare.

Run: python examples/fingerprint_demo.py
"""

from caliper.adapters import SimulatedSubject
from caliper.report import render_html, run_fingerprint
from caliper.types import ItemBank


def main() -> None:
    bank = ItemBank.bundled()
    subjects = {
        "well-behaved": SimulatedSubject(theta=1.0, bank=bank, seed=1),
        "overconfident-and-contaminated": SimulatedSubject(
            theta=1.0, bank=bank, seed=1, calibration_skew=0.3,
            robustness=0.7, contaminated=True,
        ),
    }
    for name, subject in subjects.items():
        fp = run_fingerprint(subject, bank, seed=0, progress=print)
        print(f"\n== {name} ==")
        for dim, value in fp.dimensions().items():
            print(f"  {dim:<15} {value:.2f}")
        out = f"{name}.fingerprint.html"
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(render_html(fp))
        print(f"  report -> {out}")
    print(
        "\nSame ability, radically different fingerprints — the point estimate "
        "would never tell you."
    )


if __name__ == "__main__":
    main()
