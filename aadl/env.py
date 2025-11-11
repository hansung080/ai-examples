from __future__ import annotations

import os

from dotenv import load_dotenv


def getenv_or_raise(key: str) -> str:
    if (value := os.getenv(key)) is None:
        raise EnvironmentError(f"environment variable '{key}' required")
    return value


load_dotenv()

GPT_MODEL: str = os.getenv("AIEX_GPT_MODEL", default="gpt-4o-mini")
OPENAI_API_KEY: str = getenv_or_raise("AIEX_OPENAI_API_KEY")
