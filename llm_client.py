"""Single-call DeepSeek extraction for text or one Telegram image."""

import base64
import json
import logging
import os
from typing import Any

from prompts import BEAM_EXTRACTION_SYSTEM_PROMPT
from schema import ExtractedProblem, parse_extraction

logger = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    pass


class LLMExtractionError(RuntimeError):
    pass


def _image_data_url(image_bytes: bytes, mime_type: str) -> str:
    allowed = {"image/jpeg", "image/png", "image/gif", "image/webp"}
    if mime_type not in allowed:
        raise LLMExtractionError("Use a JPEG, PNG, GIF or WebP image.")
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _parse_json_content(content: str) -> ExtractedProblem:
    if not content or not content.strip():
        raise LLMExtractionError("The model returned an empty extraction.")
    try:
        return parse_extraction(content)
    except Exception as exc:
        # No automatic repair call: avoid hidden extra token usage.
        raise LLMExtractionError("The model response was not valid extraction JSON.") from exc


def _request_once(client, *, model: str, messages: list[dict[str, Any]], max_tokens: int) -> ExtractedProblem:
    """Make exactly one model request; separated for offline mocked tests."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        max_tokens=max_tokens,
        extra_body={"thinking": {"type": "disabled"}},
    )
    content = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)
    if usage:
        logger.info(
            "DeepSeek extraction complete; prompt_tokens=%s completion_tokens=%s",
            getattr(usage, "prompt_tokens", "?"),
            getattr(usage, "completion_tokens", "?"),
        )
    return _parse_json_content(content)


def extract_problem(
    user_text: str,
    *,
    image_bytes: bytes | None = None,
    image_mime_type: str | None = None,
) -> ExtractedProblem:
    """Call DeepSeek once and return validated structured beam information."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise LLMUnavailable(
            "DEEPSEEK_API_KEY is not configured. You can still submit a JSON beam config."
        )

    # Import lazily so offline schema tests do not require the API SDK.
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=60.0,
        max_retries=0,
    )

    question = (user_text or "").strip()
    if not question:
        question = "Extract the beam dimensions, supports, material/section data, loads and requested results from the attached image."

    if image_bytes is None:
        user_content: Any = question
    else:
        if not image_mime_type:
            raise LLMExtractionError("Image MIME type is missing.")
        user_content = [
            {"type": "text", "text": question},
            {
                "type": "image_url",
                "image_url": {
                    "url": _image_data_url(image_bytes, image_mime_type),
                    "detail": "high",
                },
            },
        ]

    try:
        return _request_once(
            client,
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
            messages=[
                {"role": "system", "content": BEAM_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=int(os.getenv("DEEPSEEK_MAX_OUTPUT_TOKENS", "1600")),
        )
    except LLMExtractionError:
        raise
    except Exception as exc:
        logger.warning("DeepSeek request failed (%s)", type(exc).__name__)
        raise LLMExtractionError(
            "DeepSeek could not read this problem right now. Please retry or use /json."
        ) from exc


__all__ = ["extract_problem", "LLMUnavailable", "LLMExtractionError", "_parse_json_content", "_request_once"]
