"""Build a RAG evaluation bank from an open QA-with-context dataset.

Fetches (question, context, answer) rows from the Hugging Face datasets-server
REST API (default: SQuAD v2, CC BY-SA 4.0 — the same license family as the ARC
item bank) and writes a :class:`caliper.rag.RagBank`. To make the retrieval
realistic (a mix of relevant and distractor passages), each sample's context
is the gold passage plus a couple of passages sampled from *other* questions.

The small bank that ships in ``src/caliper/rag/data/rag_bank.json`` is instead
hand-authored original text (labeled a demo set), so tests and the offline
demo need no download. Use this script to build a larger, real bank.

Usage: python scripts/build_rag_bank.py [--n-samples 100] [--distractors 2] [--out PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from caliper.rag.types import RagBank, RagSample  # noqa: E402

API = "https://datasets-server.huggingface.co/rows"
DATASET = "rajpurkar/squad_v2"
CONFIG = "squad_v2"
SPLIT = "validation"


def fetch_rows(n: int) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while len(rows) < n * 3:  # over-fetch: many rows share a context/are unanswerable
        resp = requests.get(
            API,
            params={"dataset": DATASET, "config": CONFIG, "split": SPLIT,
                    "offset": offset, "length": min(100, n * 3 - len(rows))},
            timeout=60,
        )
        resp.raise_for_status()
        batch = resp.json()["rows"]
        if not batch:
            break
        rows.extend(r["row"] for r in batch)
        offset += len(batch)
    return rows


def to_samples(rows: list[dict], n: int, distractors: int, seed: int = 0) -> list[RagSample]:
    rng = np.random.default_rng(seed)
    usable = [
        r for r in rows
        if r.get("answers", {}).get("text") and r.get("context")
    ]
    contexts_pool = [r["context"] for r in usable]
    samples: list[RagSample] = []
    for i, row in enumerate(usable[:n]):
        gold = row["context"]
        others = [c for c in contexts_pool if c != gold]
        picks = list(rng.choice(len(others), size=min(distractors, len(others)),
                                replace=False)) if others else []
        contexts = [gold] + [others[int(j)] for j in picks]
        rng.shuffle(contexts)
        samples.append(
            RagSample(
                id=f"squad2/{row.get('id', i)}",
                question=row["question"].strip(),
                contexts=[c.strip() for c in contexts],
                reference_answer=row["answers"]["text"][0].strip(),
                tags=["squad", "reading-comprehension"],
            )
        )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--distractors", type=int, default=2)
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[1] / "rag_bank.built.json"),
    )
    args = parser.parse_args()

    print(f"fetching rows for {args.n_samples} samples…")
    rows = fetch_rows(args.n_samples)
    samples = to_samples(rows, args.n_samples, args.distractors)
    bank = RagBank(samples=samples, name="squad2-rag",
                   source="rajpurkar/squad_v2 validation (CC BY-SA 4.0)")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bank.save(out)
    print(f"wrote {len(samples)} samples to {out}")


if __name__ == "__main__":
    main()
