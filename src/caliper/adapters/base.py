"""Adapter protocol: the only interface evaluation modules depend on."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypedDict


class ChatMessage(TypedDict):
    role: str  # "system" | "user" | "assistant"
    content: str


class ModelAdapter(ABC):
    """A chat-capable model. Embeddings are optional."""

    name: str = "adapter"

    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
    ) -> str:
        """Return the assistant reply for a conversation."""

    def embed(self, texts: list[str]):  # -> np.ndarray, shape (n, d)
        """Sentence embeddings (HF feature-extraction task). Optional."""
        raise NotImplementedError(f"{self.name} does not support embeddings")

    def ask(self, prompt: str, *, system: str | None = None, **kwargs) -> str:
        """Convenience wrapper for single-turn prompts."""
        messages: list[ChatMessage] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **kwargs)
