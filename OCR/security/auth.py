from __future__ import annotations


def validate_api_key(api_key: str) -> bool:
    return bool(api_key and api_key.strip())
