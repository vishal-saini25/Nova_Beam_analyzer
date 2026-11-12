"""Telegram interface for the Beam Analysis Assistant (text-only results)."""

import asyncio
import json
import logging
import os
import time
from collections import defaultdict, deque
from typing import Any

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

from beam_analyser import examples
from llm_client import extract_problem, LLMUnavailable, LLMExtractionError
from schema import ExtractedProblem, parse_extraction, validate_config_shape
from solver_service import analyse, BeamInputError


# -------------------- Runtime configuration --------------------

TOKEN = os.getenv("TELEGRAM_TOKEN_ID")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
MAX_TEXT_CHARS = 10_000
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_REQUESTS_PER_MINUTE = 5
TELEGRAM_MESSAGE_LIMIT = 3900

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("beam_bot")

# In-memory conversation state. A single Railway worker is intentional.
pending: dict[int, dict[str, Any]] = {}
editing_chats: set[int] = set()
json_mode_chats: set[int] = set()
chat_locks: set[int] = set()
request_times: dict[int, deque[float]] = defaultdict(deque)


# -------------------- Small helpers --------------------

def split_message(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    """Split long text at paragraph boundaries without dropping characters."""
    text = text.strip()
    if not text:
        return ["No text result was generated."]
    chunks: list[str] = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        chunks.append(text)
    return chunks


def allow_request(user_id: int) -> bool:
    now = time.monotonic()
    events = request_times[user_id]
    while events and now - events[0] > 60:
        events.popleft()
    if len(events) >= MAX_REQUESTS_PER_MINUTE:
        return False
    events.append(now)
    return True


def is_solver_config(data: dict[str, Any]) -> bool:
    """Accept either a direct solver config or {config: ..., unknowns_requested: ...}."""
    return "L" in data and "section" in data


def extraction_from_json(raw: str) -> ExtractedProblem:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("JSON input must be an object.")

    if "config" in data:
        data.setdefault("problem_type", "beam_analysis")
        data.setdefault("unknowns_requested", ["complete beam analysis"])
        data.setdefault("missing_info", [])
        data.setdefault("assumptions", ["Configuration supplied directly as JSON; no model call used."])
        data.setdefault("confidence", 1.0)
        data.setdefault("notes", "")
        return ExtractedProblem.model_validate(data)

    if is_solver_config(data):
        return ExtractedProblem(
            problem_type="beam_analysis",
            config=data,
            unknowns_requested=["complete beam analysis"],
            assumptions=["Configuration supplied directly as JSON; no model call used."],
            confidence=1.0,
        )
    raise ValueError("JSON must contain a beam config with L, section and supports.")


def short_config_summary(config: dict[str, Any] | None) -> list[str]:
    if not config:
        return ["Beam configuration: not fully extracted"]

    lines: list[str] = []
    lines.append(f"Length: {config.get('L', 'missing')} m")
    sec = config.get("section") or {}
    sec_type = str(sec.get("type", "")).lower()
    if sec_type in ("ei", "4"):
        stiffness = next((f"{key}={value}" for key, value in sec.items() if key.startswith("EI_")), "EI missing")
        lines.append(f"Stiffness: {stiffness}")
    else:
        lines.append(f"Section: {sec}")
        lines.append(f"E: {config.get('E_GPa', 'missing')} GPa")

    supports = config.get("supports") or []
    lines.append("Supports: " + (", ".join(
        f"{item.get('type')} at {item.get('x')} m" for item in supports
    ) or "missing"))

    points = config.get("point_loads") or []
    if points:
        lines.append("Point loads: " + "; ".join(
            f"{p.get('P_kN')} kN at {p.get('a')} m" for p in points
        ))

    distributed = config.get("distributed_loads", config.get("udls", [])) or []
    if distributed:
        lines.append("Distributed loads: " + "; ".join(
            f"{d.get('w1_kNm')}→{d.get('w2_kNm', d.get('w1_kNm'))} kN/m, "
            f"{d.get('start')}–{d.get('end')} m" for d in distributed
        ))

    moments = config.get("moments") or []
    if moments:
        lines.append("Applied couples: " + "; ".join(
            f"{m.get('M_kNm')} kN.m at {m.get('a')} m" for m in moments
        ))

    queries = config.get("query_points") or []
    if queries:
        lines.append("Query positions: " + ", ".join(f"{x} m" for x in queries))
    return lines


def confirmation_text(extraction: ExtractedProblem) -> str:
    lines = ["BEAM DATA EXTRACTED", ""]
    lines.extend(short_config_summary(extraction.config))
    lines.extend(["", "Requested: " + ("; ".join(extraction.unknowns_requested) or "complete analysis")])
    lines.append(f"Extraction confidence: {extraction.confidence:.0%}")

    if extraction.assumptions:
        lines.append("\nAssumptions:")
        lines.extend(f"• {item}" for item in extraction.assumptions[:8])

    if extraction.notes:
        lines.append("\nNotes: " + extraction.notes)

    missing = list(extraction.missing_info)
    missing.extend(validate_config_shape(extraction.config))
    if missing:
        lines.append("\nMISSING / UNCLEAR — I will not guess:")
        # Preserve order while removing exact duplicates.
        seen = set()
        for item in missing:
            if item not in seen:
                lines.append(f"• {item}")
                seen.add(item)
        lines.append("\nSend a corrected or clearer question/image. No calculation has been run yet.")
    else:
        lines.append("\nPlease check the extracted values before calculation.")
    return "\n".join(lines)


async def reply_long(message, text: str) -> None:
    for part in split_message(text):
        await message.reply_text(part)


async def run_analysis_and_reply(chat_id: int, status_message, context: ContextTypes.DEFAULT_TYPE, config: dict[str, Any]) -> None:
    """Run solver off the event loop and send only text results."""
    stages = [
        "1/3 Validating beam configuration...",
        "2/3 Computing reactions, shear, moment, slope and deflection...",
        "3/3 Preparing the text solution...",
    ]
    try:
        await status_message.edit_text(stages[0])
        await status_message.edit_text(stages[1])
        result = await asyncio.wait_for(
            asyncio.to_thread(analyse, config),
            timeout=30,
        )
        await status_message.edit_text(stages[2])
        if not result["equilibrium_ok"]:
            await status_message.edit_text(
                "WARNING: The solver's equilibrium check did not pass. "
                "The following result is marked unverified."
            )
        await reply_long(status_message, result["report"])
    except asyncio.TimeoutError:
        await status_message.edit_text("The calculation exceeded the 30-second limit. Check the beam configuration and try again.")
    except BeamInputError as exc:
        await status_message.edit_text(f"The solver could not validate this beam: {exc}")
    except Exception:
        logger.exception("Unexpected solver failure for chat %s", chat_id)
        await status_message.edit_text("The calculation failed unexpectedly. The error was logged for debugging; please check your inputs and try again.")


# -------------------- Commands --------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Beam Analysis Assistant\n\n"
        "Send a beam question as text, or upload a clear photo/screenshot of the question and beam diagram.\n\n"
        "I extract the data, show you a confirmation summary, then the numerical solver calculates the answer.\n"
        "All results are returned as text (no plot images).\n\n"
        "Example: A 5 m simply supported beam has a 20 kN downward point load at 2 m. E=200 GPa, I=8.3e-5 m^4. Find reactions and deflection at midspan.\n\n"
        "/help  /example  /json  /new  /cancel"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "HOW TO SEND A BEAM PROBLEM\n\n"
        "You may type the full textbook question or send one clear image with its diagram. Include beam length, supports, material/section stiffness, loads and requested positions when available.\n\n"
        "Units: length m (or mm/cm in text); loads kN; distributed loads kN/m; couples kN.m; E GPa; I m^4 or cm^4.\n\n"
        "Sign convention: downward loads positive; upward loads negative; clockwise couples positive. Positions are measured from the left end.\n\n"
        "The solver supports constant EI beams with fixed/pin supports, point loads, UDL, UVL and applied couples. Variable EI, springs, support settlement, frames and inverse problems are not supported by this solver.\n\n"
        "Text-only output: support reactions, shear/moment/slope/deflection results and requested locations.\n\n"
        "For token-saving, each natural-language or image problem uses one DeepSeek extraction call. JSON input bypasses DeepSeek entirely."
    )


async def new_problem(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    pending.pop(chat_id, None)
    editing_chats.discard(chat_id)
    json_mode_chats.discard(chat_id)
    await update.message.reply_text("Ready for a new beam problem. Send text, an image, or /json followed by a config.")


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    pending.pop(chat_id, None)
    editing_chats.discard(chat_id)
    json_mode_chats.discard(chat_id)
    await update.message.reply_text("Current beam problem cancelled.")


async def json_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    json_mode_chats.add(update.effective_chat.id)
    await update.message.reply_text(
        "Send the solver configuration as JSON in your next message. This path uses no DeepSeek tokens.\n\n"
        'Example: {"L":5,"E_GPa":200,"section":{"type":"custom","I_m4":0.000083},"supports":[{"type":"pin","x":0},{"type":"pin","x":5}],"point_loads":[{"P_kN":20,"a":2}],"query_points":[2.5]}'
    )


async def example_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    name = (context.args[0] if context.args else "5.2").lower()
    available = examples()
    aliases = {"5.1(a)": "5.1a", "5.1(a) ": "5.1a"}
    name = aliases.get(name, name)
    if name not in available:
        await update.message.reply_text("Available built-in examples: " + ", ".join(available.keys()) + "\nUse /example 5.2")
        return
    status = await update.message.reply_text(f"Running built-in example {name}...")
    await run_analysis_and_reply(update.effective_chat.id, status, context, available[name])


# -------------------- Input handling --------------------

async def process_problem(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    text: str,
    image_bytes: bytes | None = None,
    image_mime_type: str | None = None,
) -> None:
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id

    if chat_id in chat_locks:
        await update.effective_message.reply_text("A problem is already being processed in this chat. Please wait for it to finish.")
        return
    if not allow_request(user_id):
        await update.effective_message.reply_text("Rate limit reached: please wait a minute before sending another problem.")
        return
    if len(text) > MAX_TEXT_CHARS:
        await update.effective_message.reply_text(f"Please keep the text under {MAX_TEXT_CHARS} characters.")
        return

    chat_locks.add(chat_id)
    status = await update.effective_message.reply_text("Got it — reading the beam problem and extracting the values...")
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    try:
        # An edit message supplements the original text and costs one new extraction call.
        previous = pending.get(chat_id, {})
        if chat_id in editing_chats and previous.get("raw_text"):
            text = previous["raw_text"] + "\n\nUser correction: " + text
            editing_chats.discard(chat_id)

        stripped = text.strip()
        if stripped.startswith("{") or chat_id in json_mode_chats:
            extraction = extraction_from_json(stripped)
            json_mode_chats.discard(chat_id)
        else:
            extraction = await asyncio.wait_for(
                asyncio.to_thread(
                    extract_problem,
                    text,
                    image_bytes=image_bytes,
                    image_mime_type=image_mime_type,
                ),
                timeout=65,
            )

        if extraction.problem_type == "unsupported":
            await status.edit_text("This does not appear to be a beam-analysis problem supported by this assistant. No calculation was run.")
            return
        if extraction.problem_type == "needs_extension":
            details = extraction.notes or "; ".join(extraction.missing_info) or "The requested feature is outside the current solver's scope."
            await status.edit_text("This problem needs a solver extension and was not calculated.\n\n" + details)
            return

        # Catch obvious physical/configuration errors before showing Confirm.
        missing = list(extraction.missing_info) + validate_config_shape(extraction.config)
        if not missing and extraction.config:
            try:
                from beam_analyser import build_model
                build_model(extraction.config)
            except Exception as exc:
                missing.append("Configuration validation: " + str(exc))
        # Keep validation findings on the extraction so the confirmation card
        # displays them and the confirm callback cannot bypass them.
        extraction.missing_info = list(dict.fromkeys(missing))

        pending[chat_id] = {
            "user_id": user_id,
            "extraction": extraction,
            "raw_text": text,
        }

        card = confirmation_text(extraction)
        if missing:
            editing_chats.add(chat_id)
            await status.edit_text(card)
            return

        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("Confirm & solve", callback_data="beam:confirm"),
                InlineKeyboardButton("Edit", callback_data="beam:edit"),
                InlineKeyboardButton("Cancel", callback_data="beam:cancel"),
            ]
        ])
        await status.edit_text(card, reply_markup=keyboard)

    except LLMUnavailable as exc:
        await status.edit_text(
            str(exc) + "\n\nUse /json to provide a structured solver configuration without an LLM."
        )
    except (LLMExtractionError, ValueError, json.JSONDecodeError) as exc:
        logger.info("Problem extraction rejected for chat %s (%s)", chat_id, type(exc).__name__)
        await status.edit_text(
            f"I could not safely extract a complete beam configuration: {exc}\n\n"
            "Please clarify the missing values or use /json. No calculation was run."
        )
    except asyncio.TimeoutError:
        await status.edit_text("Problem extraction timed out. Please retry with a clearer, shorter question or use /json.")
    except Exception:
        logger.exception("Unexpected input-processing failure for chat %s", chat_id)
        await status.edit_text("I could not process that message. Please try again with a clearer question or image.")
    finally:
        chat_locks.discard(chat_id)


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.effective_message.text or ""
    await process_problem(update, context, text=text)


async def image_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    mime_type = "image/jpeg"
    file_size = None

    if message.photo:
        photo = message.photo[-1]
        telegram_file = await context.bot.get_file(photo.file_id)
        file_size = getattr(photo, "file_size", None)
    elif message.document and message.document.mime_type in {
        "image/jpeg", "image/png", "image/gif", "image/webp"
    }:
        mime_type = message.document.mime_type
        telegram_file = await context.bot.get_file(message.document.file_id)
        file_size = message.document.file_size
    else:
        await message.reply_text("Please upload a JPEG, PNG, GIF or WebP image.")
        return

    if file_size and file_size > MAX_IMAGE_BYTES:
        await message.reply_text("Image is too large. Please send an image under 12 MB.")
        return

    try:
        image_data = bytes(await telegram_file.download_as_bytearray())
        if len(image_data) > MAX_IMAGE_BYTES:
            await message.reply_text("Image is too large. Please send an image under 12 MB.")
            return
        caption = message.caption or "Read the beam question and diagram in this image. Extract all values accurately."
        await process_problem(
            update,
            context,
            text=caption,
            image_bytes=image_data,
            image_mime_type=mime_type,
        )
    except Exception:
        logger.exception("Failed to download image sent to bot")
        await message.reply_text("I could not read that image. Please resend it as a photo or supported image document.")


# -------------------- Confirmation callbacks --------------------

async def confirmation_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    state = pending.get(chat_id)

    if not state or state.get("user_id") != query.from_user.id:
        await query.edit_message_text("This confirmation has expired. Send the problem again.")
        return

    action = query.data.split(":", 1)[1]
    if action == "cancel":
        pending.pop(chat_id, None)
        await query.edit_message_text("Beam problem cancelled.")
        return

    if action == "edit":
        editing_chats.add(chat_id)
        await query.edit_message_text(
            "Send your correction as text. I will combine it with the original question and re-extract the data.\n"
            "This correction requires one new model call unless you send a JSON config."
        )
        return

    if action == "confirm":
        extraction: ExtractedProblem = state["extraction"]
        if extraction.missing_info or validate_config_shape(extraction.config):
            await query.edit_message_text("Required information is still missing. Send the missing values before solving.")
            return
        pending.pop(chat_id, None)
        if chat_id in chat_locks:
            await query.edit_message_text("A calculation is already running in this chat.")
            return
        chat_locks.add(chat_id)
        try:
            await run_analysis_and_reply(chat_id, query.message, context, extraction.config)
        finally:
            chat_locks.discard(chat_id)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Unhandled Telegram handler error", exc_info=context.error)


def main() -> None:
    if not TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN_ID environment variable is missing.")

    app = Application.builder().token(TOKEN).concurrent_updates(4).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("example", example_command))
    app.add_handler(CommandHandler("json", json_command))
    app.add_handler(CommandHandler(["new"], new_problem))
    app.add_handler(CommandHandler("cancel", cancel))
    app.add_handler(CallbackQueryHandler(confirmation_callback, pattern=r"^beam:(confirm|edit|cancel)$"))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, image_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    app.add_error_handler(error_handler)

    logger.info("Beam Analysis Assistant starting in long-polling mode (text-only results).")
    app.run_polling(drop_pending_updates=False, allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
