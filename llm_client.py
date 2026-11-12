"""Single-call DeepSeek extraction for text or one Telegram image."""
import base64
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
        raise LLMExtractionError("DeepSeek returned an empty JSON response. Please retry once; no calculation was run.")
    try:
        return parse_extraction(content.strip())
    except Exception as exc:
        logger.warning("DeepSeek JSON did not match extraction schema (%s)", type(exc).__name__)
        raise LLMExtractionError(
            "DeepSeek returned JSON, but it did not contain the required beam-case fields. Please retry with a clearer question."
        ) from exc


def _request_once(client, *, model: str, messages: list[dict[str, Any]], max_tokens: int) -> ExtractedProblem:
    """Make exactly one model request; separated for offline mocked tests."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        max_tokens=max_tokens,
        extra_body={"thinking": {"type": "disabled"}},
    )
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise LLMExtractionError("DeepSeek returned no completion choice.")
    choice = choices[0]
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None) or ""
    finish_reason = getattr(choice, "finish_reason", None)
    usage = getattr(response, "usage", None)
    if usage:
        logger.info(
            "DeepSeek extraction complete; prompt_tokens=%s completion_tokens=%s finish_reason=%s",
            getattr(usage, "prompt_tokens", "?"),
            getattr(usage, "completion_tokens", "?"),
            finish_reason,
        )
    if not content.strip():
        logger.warning("DeepSeek returned blank content; finish_reason=%s", finish_reason)
        if finish_reason == "length":
            raise LLMExtractionError("DeepSeek reached its output-token limit. Increase DEEPSEEK_MAX_OUTPUT_TOKENS or shorten the question.")
        raise LLMExtractionError("DeepSeek returned an empty JSON response. Please retry; no calculation was run.")
    if finish_reason == "length":
        raise LLMExtractionError("DeepSeek truncated the extracted cases. Increase DEEPSEEK_MAX_OUTPUT_TOKENS and retry.")
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
        raise LLMUnavailable("DEEPSEEK_API_KEY is not configured.")

    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=60.0,
        max_retries=0,
    )

    question = (user_text or "").strip()
    if not question:
        question = "Read the attached beam question and diagram. Extract all requested beam cases as JSON."
    if image_bytes is None:
        user_content: Any = question
    else:
        if not image_mime_type:
            raise LLMExtractionError("Image MIME type is missing.")
        user_content = [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {
                "url": _image_data_url(image_bytes, image_mime_type),
                "detail": "high",
            }},
        ]

    try:
        return _request_once(
            client,
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
            messages=[
                {"role": "system", "content": BEAM_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=int(os.getenv("DEEPSEEK_MAX_OUTPUT_TOKENS", "2400")),
        )
    except LLMExtractionError:
        raise
    except Exception as exc:
        logger.exception("DeepSeek API request failed (%s)", type(exc).__name__)
        raise LLMExtractionError(
            "DeepSeek API request failed. Check Railway logs for the exact API error and retry."
        ) from exc


__all__ = ["extract_problem", "LLMUnavailable", "LLMExtractionError", "_parse_json_content", "_request_once"]
