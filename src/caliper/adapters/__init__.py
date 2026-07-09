"""Model backends. All evaluation code talks to a `ModelAdapter`."""

from caliper.adapters.base import ChatMessage, ModelAdapter
from caliper.adapters.replay import ReplayAdapter
from caliper.adapters.simulate import SimulatedJudge, SimulatedRAGSubject, SimulatedSubject

__all__ = [
    "ChatMessage",
    "ModelAdapter",
    "ReplayAdapter",
    "SimulatedJudge",
    "SimulatedRAGSubject",
    "SimulatedSubject",
    "make_adapter",
]


def make_adapter(kind: str, model: str = "", **kwargs) -> ModelAdapter:
    """Factory used by the CLI and the Space.

    ``kind`` is one of ``hf``, ``openai``, ``replay`` or ``simulated``.
    Imports are lazy so the core package works without optional deps.
    """
    if kind == "hf":
        from caliper.adapters.hf_inference import HFInferenceAdapter

        return HFInferenceAdapter(model=model, **kwargs)
    if kind == "openai":
        from caliper.adapters.openai_compat import OpenAICompatAdapter

        return OpenAICompatAdapter(model=model, **kwargs)
    if kind == "replay":
        return ReplayAdapter(**kwargs)
    if kind == "simulated":
        return SimulatedSubject(**kwargs)
    raise ValueError(f"Unknown adapter kind: {kind!r}")
