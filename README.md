# Jarvis — macOS AI Voice Assistant

> Talk to your Mac. Jarvis handles the rest.

Jarvis is a local-first AI operating layer for macOS. Activate it with ⌃⌥ or say "Hey Jarvis" — it takes voice commands and executes them: file operations, shell commands, web search, code execution, app control, and developer workflows. A floating streaming HUD shows responses building up live; voice narrates summaries.

**Hybrid AI engine:** A fine-tuned Qwen classifier routes intent before execution. Local Ollama models handle most tasks offline; Claude Haiku handles cloud-dependent tasks; Claude Sonnet handles deep reasoning.

---

## Demo

_Screenshots / GIF coming soon._

---

## Features

- **Voice-activated** — hotkey (⌃⌥) or always-on wake word
- **Live streaming HUD** — responses build up character-by-character in a floating overlay; keyboard-dismissable
- **Conversational sessions** — full multi-turn history with auto-compaction (Haiku summarises at 5K tokens)
- **File & shell operations** — guardrails and approval prompts for destructive actions
- **Web search** — Brave Search API or DuckDuckGo fallback
- **Code execution** — multi-language, project-aware
- **App control** — open apps, AppleScript, system notifications
- **Project memory** — remembers your build/test commands per project
- **Local-first routing** — fine-tuned Qwen classifier gates every command; Ollama handles most tasks without a cloud call
- **Two-way delegation** — Claude can hand off sub-tasks to local Ollama
- **MCP client** — connect GitHub and other MCP servers for rich tool access

---

## Architecture

Two-process system:

```
Swift App (menu bar + HUD + hotkey + STT + TTS)
        ↕ HTTP localhost:8765
Python Core (FastAPI + Ollama + Claude API + tools)
```

```
POST /command → CommandPipeline → Router (fine-tuned Qwen classifier via Ollama)
                      ↓                    ↓
               ProjectMemory       Local Ollama agent  (most tasks)
                                   Claude Haiku        (cloud/tool tasks)
                                   Claude Sonnet       (complex_reasoning)
                                         ↕ delegate_to_local
                                   OllamaAgent (sub-tasks)
```

- **Swift app** (`jarvis-swift/`) — menu bar icon, floating streaming HUD, global hotkey ⌃⌥, wake word, STT via SFSpeechRecognizer, TTS via AVSpeechSynthesizer
- **Python core** (`jarvis-core/`) — FastAPI server, local-first routing, tools, guardrails, conversational history with auto-compaction

---

## Requirements

| Dependency | Version |
|---|---|
| macOS | 26.0+ |
| Python | 3.11+ |
| Ollama | latest |
| Xcode | 16+ |
| Node.js | 18+ (for xcodegen) |

**API Keys (optional for local-only use):**
- Anthropic API key (for Claude Haiku / Sonnet — web search and complex reasoning)
- Brave Search API key (optional — falls back to DuckDuckGo)

---

## Setup

### 1. Clone

```bash
git clone https://github.com/Matanvil/jarvis.git
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
# Edit ~/.jarvis/config.json — add your Anthropic API key if you want cloud AI
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

## Building from source

**Requirements:** macOS 26.0+, Xcode 16+, Python 3.11+

### Swift app

1. Copy `jarvis-swift/Local.xcconfig.example` → `jarvis-swift/Local.xcconfig`
2. Open Xcode → Settings → Accounts, sign in with your Apple ID (free is fine)
3. Find your Team ID: click your Apple ID row in the accounts list — the Team ID column shows the 10-character ID. Paste it into `Local.xcconfig`
4. Open `jarvis-swift/Jarvis.xcodeproj` in Xcode and build with ⌘B

On first launch, macOS will prompt for Accessibility and Speech Recognition permissions — both are required.

### Python core

```bash
cd jarvis-core
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Local-first operation

Jarvis is designed to work without cloud AI for most tasks. The pre-flight classifier (a fine-tuned Qwen model served via Ollama) gates every command before execution:

| Intent class | Handler |
|---|---|
| `read_only`, `prepare` | Local Ollama agent |
| `destructive` | Local Ollama agent + approval prompt |
| `complex_reasoning` | Claude Sonnet (web search, deep analysis) |

Cloud models are only invoked when the command explicitly requires live web data or the classifier routes to `complex_reasoning`.

---

## Development

See [`jarvis-core/README.md`](jarvis-core/README.md) and [`jarvis-swift/README.md`](jarvis-swift/README.md) for subsystem-specific guides.

Run tests:

```bash
cd jarvis-core
source .venv/bin/activate
pytest
```

440+ tests covering the Python core, routing, agent loops, guardrails, tools, and SSE streaming.

View logs:

```bash
tail -f ~/.jarvis/logs/errors.log
tail -f ~/.jarvis/logs/commands.log
log show --predicate 'process == "Jarvis"' --last 5m   # Swift app logs
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All contributions welcome.

---

## License

MIT — see [LICENSE](LICENSE).
