"""Single-call Gemini Interactions API extraction for text and beam images."""
import base64
import logging
import os

from google import genai
from google.genai import types

from prompts import BEAM_EXTRACTION_SYSTEM_PROMPT
from schema import ExtractedProblem, parse_extraction

logger = logging.getLogger(__name__)
MODEL_DEFAULT = "gemini-3.8-flash"


class LLMUnavailable(RuntimeError):
    pass


class LLMExtractionError(RuntimeError):
    pass


def _parse_json_content(content: str) -> ExtractedProblem:
    if not content or not content.strip():
        raise LLMExtractionError("Gemini returned an empty response. Please retry; no calculation was run.")
    try:
        return parse_extraction(content.strip())
    except Exception as exc:
        logger.warning("Gemini JSON did not match extraction schema (%s)", type(exc).__name__)
        raise LLMExtractionError(
            "Gemini returned a response that did not match the required beam-case format. Please retry with a clearer question."
        ) from exc


def _image_input(image_bytes: bytes, mime_type: str) -> dict[str, str]:
    allowed = {"image/jpeg", "image/png", "image/gif", "image/webp"}
    if mime_type not in allowed:
        raise LLMExtractionError("Use a JPEG, PNG, GIF or WebP image.")
    return {
        "type": "image",
        "data": base64.b64encode(image_bytes).decode("ascii"),
        "mime_type": mime_type,
    }


def _request_once(client, *, model: str, user_text: str, image_bytes: bytes | None,
                  image_mime_type: str | None, max_output_tokens: int) -> ExtractedProblem:
    """Make exactly one Gemini Interactions API request."""
    input_parts = [{
        "type": "text",
        "text": BEAM_EXTRACTION_SYSTEM_PROMPT + "\n\nUSER QUESTION:\n" + user_text,
    }]
    if image_bytes is not None:
        if not image_mime_type:
            raise LLMExtractionError("Image MIME type is missing.")
        input_parts.append(_image_input(image_bytes, image_mime_type))

    response = client.interactions.create(
        model=model,
        input=input_parts,
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": ExtractedProblem.model_json_schema(),
        },
        generation_config={
            "max_output_tokens": max_output_tokens,
            "thinking_level": os.getenv("GEMINI_THINKING_LEVEL", "low").lower(),
        },
        timeout=60,
    )
    content = getattr(response, "output_text", None) or ""
    usage = getattr(response, "usage", None)
    if usage:
        logger.info(
            "Gemini extraction complete; total_tokens=%s input_tokens=%s output_tokens=%s",
            getattr(usage, "total_tokens", "?"),
            getattr(usage, "total_input_tokens", "?"),
            getattr(usage, "total_output_tokens", "?"),
        )
    if not content.strip():
        logger.warning("Gemini returned no output_text; interaction_id=%s", getattr(response, "id", "unknown"))
        raise LLMExtractionError("Gemini completed without a text response. Check Railway logs.")
    return _parse_json_content(content)


def extract_problem(user_text: str, *, image_bytes: bytes | None = None,
                    image_mime_type: str | None = None) -> ExtractedProblem:
    """Call Gemini once; retry transient service failures once with a short bounded delay."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise LLMUnavailable("GEMINI_API_KEY is not configured in Railway variables.")

    question = (user_text or "").strip()
    if not question:
        question = "Read the attached beam question and diagram. Extract every requested case."

    model = os.getenv("GEMINI_MODEL", MODEL_DEFAULT)
    max_output_tokens = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "2400"))
    thinking_level = os.getenv("GEMINI_THINKING_LEVEL", "low").lower()
    if thinking_level not in {"low", "medium", "high"}:
        raise LLMExtractionError("GEMINI_THINKING_LEVEL must be low, medium, or high.")

    try:
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=30000,
                retry_options=types.HttpRetryOptions(
                    attempts=2,
                    initial_delay=1.0,
                    max_delay=2.0,
                    exp_base=2.0,
                    jitter=0.2,
                    http_status_codes=[408, 429, 500, 502, 503, 504],
                ),
            ),
        )
        return _request_once(
            client, model=model, user_text=question, image_bytes=image_bytes,
            image_mime_type=image_mime_type, max_output_tokens=max_output_tokens,
        )
    except LLMExtractionError:
        raise
    except Exception as exc:
        code = getattr(exc, "code", None)
        message = str(exc).replace(api_key, "[REDACTED]")
        logger.exception("Gemini Interactions API failed; code=%s error_type=%s message=%s",
                         code, type(exc).__name__, message[:500])
        if code == 503:
            raise LLMExtractionError(
                "Gemini is temporarily overloaded (HTTP 503). I tried once more automatically, but it is still unavailable. Please retry in 30–60 seconds; no calculation was run."
            ) from exc
        raise LLMExtractionError(
            f"Gemini API request failed ({type(exc).__name__}, HTTP {code or 'unknown'}). Check Railway logs. No calculation was run."
        ) from exc


__all__ = ["extract_problem", "LLMUnavailable", "LLMExtractionError", "_parse_json_content", "_request_once"]
