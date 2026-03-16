from __future__ import annotations

from typing import Mapping, Sequence

import requests

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


def _to_model_path(model_id: str) -> str:
    return model_id if model_id.startswith("models/") else f"models/{model_id}"


def _build_payload(
    messages: Sequence[Mapping[str, str]],
    max_tokens: int,
    temperature: float,
) -> dict:
    system_parts: list[str] = []
    contents: list[dict] = []

    for message in messages:
        role = str(message.get("role", "user")).lower().strip()
        content = str(message.get("content", "")).strip()

        if not content:
            continue

        if role == "system":
            system_parts.append(content)
            continue

        gemini_role = "model" if role == "assistant" else "user"
        contents.append({"role": gemini_role, "parts": [{"text": content}]})

    if not contents:
        contents.append({"role": "user", "parts": [{"text": "請總結文件內容。"}]})

    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }

    if system_parts:
        payload["systemInstruction"] = {
            "parts": [{"text": "\n\n".join(system_parts)}],
        }

    return payload


def ask_gemini_text(
    api_key: str,
    model_id: str,
    messages: Sequence[Mapping[str, str]],
    max_tokens: int = 2048,
    temperature: float = 0.2,
) -> str:
    if not api_key.strip():
        raise ValueError("Gemini API key is empty.")

    model_path = _to_model_path(model_id.strip())
    payload = _build_payload(messages=messages, max_tokens=max_tokens, temperature=temperature)
    endpoint = f"{GEMINI_API_BASE}/{model_path}:generateContent"

    try:
        response = requests.post(endpoint, params={"key": api_key.strip()}, json=payload, timeout=60)
    except requests.RequestException as exc:
        raise RuntimeError(f"Gemini request failed: {exc}") from exc

    try:
        result = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Gemini returned a non-JSON response (status: {response.status_code})."
        ) from exc

    if response.status_code != 200:
        if isinstance(result, dict):
            error_message = result.get("error", {}).get("message")
        else:
            error_message = None
        raise RuntimeError(
            f"Gemini API error ({response.status_code}): {error_message or 'Unknown error'}"
        )

    candidates = result.get("candidates", []) if isinstance(result, dict) else []
    if not candidates:
        raise RuntimeError("Gemini response did not include candidates.")

    parts = candidates[0].get("content", {}).get("parts", [])
    text_chunks = [part.get("text", "") for part in parts if isinstance(part, dict)]
    text = "".join(text_chunks).strip()

    if not text:
        raise RuntimeError("Gemini response did not include text content.")

    return text
