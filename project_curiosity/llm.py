"""OpenAI helper for project_curiosity."""
from __future__ import annotations

import os
from functools import lru_cache

from openai import OpenAI

SYSTEM_PROMPT = "You are a knowledgeable assistant that answers questions about relationships between concepts."

@lru_cache(maxsize=1)
def _init_key():
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise EnvironmentError("OPENAI_API_KEY env var required.")
    return OpenAI(api_key=key)


def ask(prompt: str, *, max_tokens: int = 16, temperature: float = 0.0) -> str:
    client = _init_key()
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()
