from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"

PROVIDER_MODELS = {
    "Together AI": [
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    ],
    "Google Gemini": [
        "gemini-2.0-flash",
        "gemini-1.5-pro",
    ],
}

DEFAULT_PROVIDER = "Together AI"
ENV_KEY_BY_PROVIDER = {
    "Together AI": "TOGETHER_API_KEY",
    "Google Gemini": "GEMINI_API_KEY",
}


def load_env_file(env_path: Path = ENV_FILE) -> None:
    """
    Load key-value pairs from .env into os.environ.
    Existing environment variables are not overwritten.
    """
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        clean_key = key.strip()
        if not clean_key:
            continue

        clean_value = value.strip().strip('"').strip("'")
        os.environ.setdefault(clean_key, clean_value)


def get_default_provider() -> str:
    configured = os.getenv("DEFAULT_PROVIDER", DEFAULT_PROVIDER).strip()
    if configured in PROVIDER_MODELS:
        return configured
    return DEFAULT_PROVIDER


def get_provider_api_key(provider: str) -> str:
    env_key = ENV_KEY_BY_PROVIDER.get(provider, "")
    if not env_key:
        return ""
    return os.getenv(env_key, "").strip()
