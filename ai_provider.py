"""AI provider helpers used by the Streamlit dashboard."""

from typing import Any

import requests


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_OPENAI_MODEL = "gpt-5.6-terra"


class OpenAIAPIError(RuntimeError):
    """Raised when the OpenAI Responses API returns an unusable response."""


def extract_openai_output_text(payload: dict[str, Any]) -> str:
    """Extract assistant text from a raw Responses API payload."""
    direct_text = payload.get("output_text")
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text.strip()

    chunks: list[str] = []
    for item in payload.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict) or content.get("type") != "output_text":
                continue
            text = content.get("text")
            if isinstance(text, str) and text:
                chunks.append(text)
    return "\n".join(chunks).strip()


def generate_openai_text(
    api_key: str,
    prompt: str,
    *,
    model: str = DEFAULT_OPENAI_MODEL,
    max_output_tokens: int = 3000,
    verbosity: str = "medium",
    timeout: int = 180,
) -> str:
    """Generate text through the OpenAI Responses API."""
    if not api_key.strip():
        raise OpenAIAPIError("OPENAI_API_KEY is empty.")

    response = requests.post(
        OPENAI_RESPONSES_URL,
        headers={
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "input": prompt,
            "max_output_tokens": max_output_tokens,
            "reasoning": {"effort": "low"},
            "text": {"verbosity": verbosity},
        },
        timeout=timeout,
    )

    try:
        payload = response.json()
    except ValueError as exc:
        raise OpenAIAPIError(
            f"OpenAI API returned HTTP {response.status_code} with invalid JSON."
        ) from exc

    if not response.ok:
        error = payload.get("error", {}) if isinstance(payload, dict) else {}
        message = error.get("message") if isinstance(error, dict) else None
        raise OpenAIAPIError(
            f"OpenAI API error {response.status_code}: {message or 'Unknown error'}"
        )

    text = extract_openai_output_text(payload)
    if not text:
        raise OpenAIAPIError("OpenAI API returned no output text.")
    return text
