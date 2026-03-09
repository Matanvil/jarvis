# Jarvis — macOS AI Voice Assistant

> Talk to your Mac. Jarvis handles the rest.

Jarvis is a local-first AI operating layer for macOS. Activate it with ⌃Space or say "Hey Jarvis" — it takes voice commands and executes them: file operations, shell commands, web search, code execution, app control, and developer workflows. A floating HUD shows responses; voice narrates summaries.

**Hybrid AI engine:** Routes simple tasks to Claude Haiku (~1-2s), complex reasoning to Claude Sonnet, with Ollama available for fully offline operation.

---

## Demo

_Screenshots / GIF coming soon._

---

## Features

- **Voice-activated** — hotkey (⌃Space) or always-on wake word
- **Floating HUD** — non-intrusive overlay, keyboard-dismissable
- **File & shell operations** — with guardrails and approval prompts for destructive actions
- **Web search** — Brave Search API or DuckDuckGo fallback
- **Code execution** — multi-language, project-aware
- **App control** — open apps, AppleScript, system notifications
- **Project memory** — remembers your build/test commands per project
- **Haiku-first routing** — fast by default, escalates to Sonnet only when needed
- **Two-way delegation** — Claude can hand off sub-tasks to local Ollama

---

## Architecture

Two-process system:

```
Swift App (menu bar + HUD + hotkey + STT + TTS)
        ↕ HTTP localhost:8765
Python Core (FastAPI + Claude API + Ollama + tools)
```

- **Swift app** (`jarvis-swift/`) — menu bar icon, floating HUD, global hotkey, wake word, STT via SFSpeechRecognizer, TTS via `say`
- **Python core** (`jarvis-core/`) — FastAPI server, hybrid AI routing, tools, guardrails, project memory

---

## Requirements

| Dependency | Version |
|---|---|
| macOS | 13+ |
| Python | 3.11+ |
| Ollama | latest |
| Xcode | 15+ |
| Node.js | 18+ (for xcodegen) |

**API Keys required:**
- Anthropic API key (for Claude Haiku / Sonnet)
- Brave Search API key (optional — falls back to DuckDuckGo)

---

## Setup

### 1. Clone

```bash
git clone https://github.com/YOUR_USERNAME/jarvis.git
cd jarvis
```

### 2. Python core

```bash
cd jarvis-core
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure

```bash
cp jarvis-core/config.example.json ~/.jarvis/config.json
# Edit ~/.jarvis/config.json — add your Anthropic API key
```

### 4. Start the Python server

```bash
cd jarvis-core
source .venv/bin/activate
uvicorn server:app --host 127.0.0.1 --port 8765
```

### 5. Build the Swift app

```bash
cd jarvis-swift
npm install -g xcodegen
xcodegen generate
open Jarvis.xcodeproj
# Press ⌘B to build, ⌘R to run
```

---

## Development

See [`jarvis-core/README.md`](jarvis-core/README.md) and [`jarvis-swift/README.md`](jarvis-swift/README.md) for subsystem-specific guides.

Run tests:

```bash
cd jarvis-core
source .venv/bin/activate
pytest
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All contributions welcome.

---

## License

MIT — see [LICENSE](LICENSE).
