# Beam Analysis Assistant

Telegram beam-analysis bot with two independent modes.

## Modes

### 1. Enter values manually
- Collects beam values step by step.
- Uses the existing deterministic Python beam solver.
- Supports the numerical solver's implemented beam cases and inputs.
- Uses no external language-model API.

### 2. Direct Solution
- Sends a typed problem statement to Groq chat completions.
- Uses `openai/gpt-oss-120b` by default.
- Produces a worked engineering solution using the detailed cantilever-focused prompt in `ai_solver.py`.
- Does not call the local Python beam solver.
- Text input only with this model integration. Image uploads are not sent to Groq; type the diagram's dimensions, load values, directions and positions into the message.

## Environment variables

Set these in Railway:

- `TELEGRAM_TOKEN_ID` — Telegram bot token.
- `GROQ_API_KEY` — Groq API key for Direct Solution.
- `GROQ_MODEL` — optional; defaults to `openai/gpt-oss-120b`.
- `LOG_LEVEL` — optional; defaults to `INFO`.

Remove the old `GEMINI_API_KEY` and `GEMINI_MODEL` variables from Railway after the Groq deployment is confirmed.

## Runtime

Railway installs dependencies from `requirements.txt` and starts the Telegram polling worker through the repository's `Procfile`.

## Important engineering note

Direct Solution is an independent model-generated calculation and can make arithmetic or interpretation errors. Cross-check important results against the manual numerical solver and established engineering references. Do not use unverified outputs for real structural design.
