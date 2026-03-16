from __future__ import annotations

from typing import Mapping, Sequence

import requests

TOGETHER_CHAT_URL = "https://api.together.xyz/v1/chat/completions"


def ask_together_text(
    api_key: str,
    model_id: str,
    messages: Sequence[Mapping[str, str]],
    max_tokens: int = 2048,
    temperature: float = 0.2,
) -> str:
    if not api_key.strip():
        raise ValueError("Together AI API key is empty.")

    payload = {
        "model": model_id,
        "messages": list(messages),
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            TOGETHER_CHAT_URL,
            json=payload,
            headers=headers,
            timeout=60,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Together AI request failed: {exc}") from exc

    try:
        result = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Together AI returned a non-JSON response (status: {response.status_code})."
        ) from exc

    if response.status_code != 200:
        if isinstance(result, dict):
            error_message = result.get("error", {}).get("message")
        else:
            error_message = None
        raise RuntimeError(
            f"Together AI API error ({response.status_code}): {error_message or 'Unknown error'}"
        )

    try:
        return result["choices"][0]["message"]["content"].strip()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Together AI response format was unexpected.") from exc
