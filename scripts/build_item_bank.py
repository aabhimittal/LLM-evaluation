"""Build the bundled item bank.

Fetches real ARC-Challenge questions (AI2 Reasoning Challenge, CC BY-SA 4.0)
from the Hugging Face datasets-server REST API, then produces demo IRT
parameters by simulating a diverse respondent population and running the
package's own `fit_items` calibration on the simulated correctness matrix.

The bundled parameters are therefore labeled ``synthetic-demo-v1``: the
*questions* are real, the *difficulty/discrimination* values demonstrate the
calibration pipeline but are not derived from real model responses. To
calibrate against real models, collect a correctness matrix and run
``caliper calibrate`` (see METHODOLOGY.md).

Usage: python scripts/build_item_bank.py [--n-items 250] [--out PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from caliper.irt.model import fit_items, p_correct  # noqa: E402
from caliper.types import Item, ItemBank  # noqa: E402

API = "https://datasets-server.huggingface.co/rows"
DATASET = "allenai/ai2_arc"
CONFIG = "ARC-Challenge"
SPLIT = "test"


def fetch_arc_items(n: int) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while len(rows) < n:
        resp = requests.get(
            API,
            params={
                "dataset": DATASET,
                "config": CONFIG,
                "split": SPLIT,
                "offset": offset,
                "length": min(100, n - len(rows)),
            },
            timeout=60,
        )
        resp.raise_for_status()
        batch = resp.json()["rows"]
        if not batch:
            break
        rows.extend(r["row"] for r in batch)
        offset += len(batch)
    return rows[:n]


def to_items(rows: list[dict]) -> list[Item]:
    items = []
    for row in rows:
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        key = row["answerKey"]
        if key not in labels or len(texts) < 3:
            continue
        items.append(
            Item(
                id=f"arc-c/{row['id']}",
                question=row["question"].strip(),
                choices=[t.strip() for t in texts],
                answer_index=labels.index(key),
                source="allenai/ai2_arc ARC-Challenge test (CC BY-SA 4.0)",
                tags=["science", "reasoning"],
            )
        )
    return items


def demo_calibrate(items: list[Item], n_respondents: int = 48, seed: int = 7) -> None:
    """Simulate a respondent population and fit item parameters in-place."""
    rng = np.random.default_rng(seed)
    n = len(items)
    true_a = rng.lognormal(mean=0.0, sigma=0.35, size=n)
    true_b = rng.normal(loc=0.0, scale=1.1, size=n)
    thetas = rng.normal(size=n_respondents)
    c = np.array([it.c for it in items])
    P = p_correct(thetas[:, None], true_a[None, :], true_b[None, :], c[None, :])
    X = (rng.random(P.shape) < P).astype(float)

    result = fit_items(X, n_choices=np.array([len(it.choices) for it in items]))
    corr_b = float(np.corrcoef(true_b, result.b)[0, 1])
    print(f"calibration: {result.n_iter} iterations, converged={result.converged}, "
          f"difficulty recovery r={corr_b:.3f}")
    for item, a, b in zip(items, result.a, result.b):
        item.a = round(float(a), 4)
        item.b = round(float(b), 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-items", type=int, default=250)
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[1] / "src/caliper/data/item_bank.json"),
    )
    args = parser.parse_args()

    print(f"fetching {args.n_items} ARC-Challenge items…")
    rows = fetch_arc_items(args.n_items)
    items = to_items(rows)
    print(f"kept {len(items)} items; running demo calibration…")
    demo_calibrate(items)

    bank = ItemBank(items=items, name="arc-challenge-demo",
                    calibration="synthetic-demo-v1")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bank.save(out)
    print(f"wrote {len(items)} items to {out}")


if __name__ == "__main__":
    main()
