# jarvis-core — Python Backend

FastAPI server that handles AI routing, tool execution, guardrails, and project memory.

## Stack

- Python 3.11
- FastAPI + uvicorn
- Anthropic SDK (`claude-haiku-4-5`, `claude-sonnet-4-6`)
- httpx (Ollama calls)
- pytest (155 tests)

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
| `router.py` | Pre-flight classifier + routing |
| `command_pipeline.py` | Command lifecycle + single-command lock |
| `guardrails.py` | Approval engine for destructive actions |
| `config.py` | Config load/save (~/.jarvis/config.json) |
| `memory.py` | Per-project build/test memory |
| `logger.py` | Rotating logs to ~/.jarvis/logs/ |
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
  }
}
```

## Routing Modes

- `haiku_first` (default) — Ollama classifies intent, routes to Haiku or Sonnet
- `ollama_only` — local only, no Claude API calls
- `claude_only` — always uses Claude, skips Ollama classifier
