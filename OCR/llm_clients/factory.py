from __future__ import annotations

from typing import Mapping, Sequence

from llm_clients.gemini_ai import ask_gemini_text
from llm_clients.together_ai import ask_together_text


def ask_model_text(
    provider: str,
    api_key: str,
    model_id: str,
    messages: Sequence[Mapping[str, str]],
    max_tokens: int = 2048,
    temperature: float = 0.2,
) -> str:
    normalized_provider = provider.strip().lower()

    if normalized_provider == "together ai":
        return ask_together_text(
            api_key=api_key,
            model_id=model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    if normalized_provider == "google gemini":
        return ask_gemini_text(
            api_key=api_key,
            model_id=model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    raise ValueError(f"Unsupported provider: {provider}")
