# jarvis-core — Python Backend

FastAPI server that handles AI routing, tool execution, guardrails, and project memory.

## Stack

- Python 3.11
- FastAPI + uvicorn
- Anthropic SDK (`claude-haiku-4-5`, `claude-sonnet-4-6`)
- httpx (Ollama calls)
- python-telegram-bot (Telegram integration)
- APScheduler (scheduled tasks)
- openai-whisper + imageio-ffmpeg (voice transcription)
- pytest (261 tests)

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn server:app --host 127.0.0.1 --port 8765 --reload
```

## Test

```bash
pytest                          # all tests
pytest tests/test_agent.py -v  # single file
pytest -k "test_name"          # single test
```

## Key Files

| File | Purpose |
|---|---|
| `server.py` | FastAPI endpoints |
| `agent.py` | Claude agent with tool-use loop |
| `ollama_agent.py` | Ollama local agent |
| `router.py` | Pre-flight classifier + routing + conversation history |
| `command_pipeline.py` | Command lifecycle + single-command lock |
| `guardrails.py` | Approval engine — one-time trust per approval |
| `config.py` | Config load/save (~/.jarvis/config.json) |
| `memory.py` | Per-project build/test memory |
| `logger.py` | Rotating logs to ~/.jarvis/logs/ |
| `scheduler.py` | APScheduler — cron + one-time tasks via voice/Telegram |
| `telegram_bot.py` | Telegram bot — text + voice commands, away mode, approvals |
| `transcriber.py` | Whisper transcription — base model, bundled ffmpeg |
| `tools/` | shell, web, code, macOS, file tools |

## Config

Stored at `~/.jarvis/config.json`. Key fields:

```json
{
  "anthropic_api_key": "sk-ant-...",
  "brave_api_key": "",
  "ollama": {
    "host": "http://localhost:11434",
    "model": "llama3.1:8b",
    "routing_mode": "haiku_first"
  },
  "models": {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6"
  },
  "telegram": {
    "bot_token": "",
    "allowed_user_id": 0
  }
}
```

## Routing Modes

- `haiku_first` (default) — Ollama classifies intent, routes to Haiku or Sonnet
- `ollama_only` — local only, no Claude API calls
- `claude_only` — always uses Claude, skips Ollama classifier

## Features

- **Conversation history** — 5-turn rolling context; tool names appended so follow-up questions work
- **Scheduled tasks** — cron or one-time tasks created by voice or Telegram; persistent across restarts
- **Telegram** — send commands, get responses, approve actions, set away mode; voice messages transcribed via Whisper
- **Guardrails** — destructive/scheduled actions require per-command approval; approving one action never auto-trusts future ones
- **Real-time step feedback** — SSE stream pushes tool steps to HUD as they happen
