# Beam Analysis Assistant (Telegram + DeepSeek + verified solver)

A Telegram long-polling bot that reads beam questions in text or from one image, extracts one or more independent beam-case configurations with DeepSeek Flash in one call, validates each in Python, and passes each case to the supplied `beam_analyser.py` solver. Final results are **text only**: no diagrams, plot images, or second LLM explanation call.

## Architecture

1. User sends a text problem or one JPEG/PNG/GIF/WebP image.
2. `llm_client.py` makes one DeepSeek extraction call with JSON output and thinking disabled. Image input is sent in the same call as the text/caption.
3. `schema.py` validates the extraction shape; `beam_analyser.build_model()` checks engineering inputs.
4. The bot shows every extracted case in one confirmation summary and waits for Confirm / Edit / Cancel. Multi-part questions (a), (b), etc. are kept as separate cases.
5. On confirmation, `solver_service.py` runs the original numerical solver and captures its text report.
6. Telegram receives the report as text. No model call is used to explain calculated values.

JSON input bypasses DeepSeek entirely. This is useful when the API key is missing or when the user wants to avoid token charges.

## Files

- `bot.py`: Telegram commands, text/image inputs, confirmation flow, rate limiting.
- `llm_client.py`: one-call DeepSeek parser for text and images.
- `prompts.py`: compact, static extraction prompt.
- `schema.py`: Pydantic extraction schema and required-field checks for one or multiple cases.
- `solver_service.py`: wrapper around the supplied solver; returns a deterministic text report.
- `beam_analyser.py`: original uploaded solver, kept intact.
- `tests/`: offline tests; no DeepSeek request is made during tests.

## Supported solver scope

Constant-EI straight prismatic beams with fixed and pin/roller supports, point loads, UDLs, UVL/trapezoidal loads, and applied couples. It can solve cantilevers, simply supported beams, overhangs, propped cantilevers, fixed-fixed and continuous beams when the support arrangement is stable.

Not implemented by the numerical solver: variable EI, spring supports, support settlement, temperature loads, inclined/axial loads, 2-D frames, shear deformation, inverse problems, or a separate finite-difference method. The bot reports these as unsupported rather than inventing a verified solution.

## Local setup (Windows PowerShell)

Python 3.12 is recommended.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
$env:TELEGRAM_TOKEN_ID = "YOUR_BOT_TOKEN"
$env:DEEPSEEK_API_KEY = "YOUR_DEEPSEEK_API_KEY"
python bot.py
```

Do not commit real tokens. `.env.example` is a variable-name template only; the code reads environment variables directly.

## Commands

- `/start` — introduction and example input
- `/help` — units, sign convention and supported scope
- `/example` or `/example 5.2` — run a built-in solver example without an LLM call
- `/json` — next message should be solver configuration JSON
- `/new` — clear pending conversation state
- `/cancel` — cancel pending state

You can also paste a raw solver config JSON object directly; it bypasses DeepSeek.

## Railway deployment (separate worker service)

1. Push this repository to a **new GitHub repository**, e.g. `beam-analysis-bot`. Keep the media-downloader repository and service untouched.
2. In Railway, create a new project/service from this GitHub repository.
3. Use the repository root as the service root. Railway should detect `Procfile` and start the worker with `python bot.py`.
4. Add these variables in Railway:
   - `TELEGRAM_TOKEN_ID` = token from BotFather for this separate beam-analysis bot.
   - `DEEPSEEK_API_KEY` = your DeepSeek API key (needed for natural-language and image extraction).
   - `DEEPSEEK_MODEL` = `deepseek-flash` (optional; this is the default).
   - `DEEPSEEK_MAX_OUTPUT_TOKENS` = `2400` (optional; useful for questions with multiple subparts).
   - `LOG_LEVEL` = `INFO` (optional).
5. Deploy and inspect logs for `Beam Analysis Assistant starting in long-polling mode`.
6. Test `/start`, `/example`, a text problem, then an image problem.

This is a worker using Telegram long polling. It does not need a public HTTP domain or inbound web port. Run exactly one active worker for a bot token; two polling instances using the same token can cause Telegram polling conflicts.

## Token-saving choices

- One extraction call per text/image problem, including all subparts and alternative loading cases.
- Thinking mode disabled for extraction.
- JSON output and a capped response size.
- No duplicate extraction for consistency checking.
- No LLM repair loop; ask the user to clarify if extraction/validation fails.
- No LLM explanation call; result text comes from the numerical solver.
- JSON input and `/example` require zero DeepSeek calls.

Image analysis consumes image tokens too. DeepSeek Flash supports JPEG, PNG, GIF and WebP. It processes image inputs as part of the same extraction call; very small labels or blurry diagrams may still need user clarification.

## Tests

```powershell
pytest -q
```

The uploaded solver's own tests can also be run directly:

```powershell
python beam_analyser.py --test
```

The solver was executed during preparation and all 30 built-in self-checks passed.

## Safety / correctness notes

- The LLM is a parser, not the calculator. Do not treat extracted values as final until the confirmation summary is checked.
- The bot refuses to solve when required dimensions, stiffness, or supports are missing.
- Engineering calculations should be independently checked before design or construction decisions.
- Conversation state is in memory and is lost if Railway restarts the worker.
- Inverse problems and solver extensions are not silently approximated.
