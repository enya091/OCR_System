from __future__ import annotations

from llm_clients.factory import ask_model_text
from prompt.rules import ROUTER_PROMPT_TEMPLATE


def classify_document(provider: str, api_key: str, model_id: str, text: str) -> str:
    """
    Read the first 500 chars and classify document type.
    """
    sample_text = text[:500]
    prompt_content = ROUTER_PROMPT_TEMPLATE.format(sample_text=sample_text)

    category = ask_model_text(
        provider=provider,
        api_key=api_key,
        model_id=model_id,
        messages=[{"role": "user", "content": prompt_content}],
        max_tokens=10,
        temperature=0.0,
    )

    normalized = category.upper()
    if "FINANCIAL" in normalized:
        return "FINANCIAL"
    if "HARDWARE" in normalized:
        return "HARDWARE"
    return "GENERAL"
