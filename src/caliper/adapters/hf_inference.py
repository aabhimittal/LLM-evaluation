"""Hugging Face Inference Providers backend.

Uses two HF tasks:
- ``chat-completion`` for the subject / judge models
- ``feature-extraction`` for sentence embeddings (semantic consistency scoring)
"""

from __future__ import annotations

import numpy as np

from caliper.adapters.base import ChatMessage, ModelAdapter

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class HFInferenceAdapter(ModelAdapter):
    def __init__(
        self,
        model: str,
        token: str | None = None,
        provider: str = "auto",
        embed_model: str = DEFAULT_EMBED_MODEL,
        timeout: float = 90.0,
    ):
        try:
            from huggingface_hub import InferenceClient
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "HFInferenceAdapter requires huggingface_hub; "
                "install with `pip install llm-caliper[hf]`"
            ) from e
        self.name = model
        self.model = model
        self.embed_model = embed_model
        self._client = InferenceClient(api_key=token, provider=provider, timeout=timeout)

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
    ) -> str:
        out = self._client.chat_completion(
            messages=messages,
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
        return out.choices[0].message.content or ""

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = [
            np.asarray(self._client.feature_extraction(t, model=self.embed_model), dtype=float)
            for t in texts
        ]
        # Some backends return (tokens, dim): mean-pool to a sentence vector.
        pooled = [v.mean(axis=0) if v.ndim > 1 else v for v in vecs]
        return np.vstack(pooled)
