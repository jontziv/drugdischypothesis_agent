"""
Shared GROQ LLM client utilities.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential

import config

logger = logging.getLogger(__name__)

_groq_client: Groq | None = None


def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        if not config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set. Please add it to your .env file.")
        _groq_client = Groq(api_key=config.GROQ_API_KEY)
    return _groq_client


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
def chat_completion(
    messages: list[dict[str, str]],
    temperature: float = config.GROQ_TEMPERATURE,
    max_tokens: int = config.GROQ_MAX_TOKENS,
    model: str = config.GROQ_MODEL,
) -> str:
    """Call GROQ chat completion and return content string."""
    client = get_groq_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def extract_json_block(text: str) -> Any:
    """
    Attempt to extract and parse a JSON block from LLM output.
    Handles ```json ... ``` fences and bare JSON objects/arrays.
    """
    # Try fenced block first
    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find a JSON array or object
    for pattern in [r"(\[[\s\S]+\])", r"(\{[\s\S]+\})"]:
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not extract JSON from LLM output:\n{text[:500]}")
