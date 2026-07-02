"""Deterministic record/replay backend.

Wrap any adapter with ``record=`` to capture real interactions into a JSON
cache; load the cache later to replay them with zero network access (used
for tests and the token-free Space demo).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from caliper.adapters.base import ChatMessage, ModelAdapter


def _key(messages: list[ChatMessage], temperature: float, seed: int | None) -> str:
    blob = json.dumps({"m": messages, "t": round(temperature, 4), "s": seed}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


class ReplayAdapter(ModelAdapter):
    def __init__(
        self,
        cache_path: str | Path | None = None,
        record: ModelAdapter | None = None,
        name: str | None = None,
    ):
        self.cache_path = Path(cache_path) if cache_path else None
        self.record = record
        self.name = name or (f"replay({record.name})" if record else "replay")
        self._cache: dict[str, str] = {}
        self._embed_cache: dict[str, list[float]] = {}
        if self.cache_path and self.cache_path.exists():
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            self._cache = payload.get("chat", {})
            self._embed_cache = payload.get("embed", {})

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
    ) -> str:
        key = _key(messages, temperature, seed)
        if key in self._cache:
            return self._cache[key]
        if self.record is None:
            raise KeyError(
                f"ReplayAdapter cache miss (key {key}); no recording backend attached"
            )
        response = self.record.chat(
            messages, temperature=temperature, max_tokens=max_tokens, seed=seed
        )
        self._cache[key] = response
        self._flush()
        return response

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        missing = [t for t in texts if _text_key(t) not in self._embed_cache]
        if missing:
            if self.record is None:
                raise KeyError("ReplayAdapter embed cache miss; no recording backend attached")
            fresh = self.record.embed(missing)
            for text, vec in zip(missing, fresh):
                self._embed_cache[_text_key(text)] = [float(x) for x in vec]
            self._flush()
        for text in texts:
            vectors.append(self._embed_cache[_text_key(text)])
        return np.asarray(vectors, dtype=float)

    def _flush(self) -> None:
        if self.cache_path:
            self.cache_path.write_text(
                json.dumps({"chat": self._cache, "embed": self._embed_cache}, indent=0),
                encoding="utf-8",
            )


def _text_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:24]
