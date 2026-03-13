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
- **Telegram integration** — send commands from your phone, receive proactive notifications, approve destructive actions remotely (optional)

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

## Building from source

**Requirements:** macOS 13+, Xcode 15+, Python 3.11+

### Swift app

1. Copy `jarvis-swift/Local.xcconfig.example` → `jarvis-swift/Local.xcconfig`
2. Open Xcode → Settings → Accounts, sign in with your Apple ID (free is fine)
3. Find your Team ID: click your Apple ID row in the accounts list — the Team ID column shows the 10-character ID. Paste it into `Local.xcconfig`
4. Open `jarvis-swift/Jarvis.xcodeproj` in Xcode and build with ⌘B

On first launch, macOS will prompt for notification permission — click **Allow**.

> **Without `Local.xcconfig`:** The app builds and runs but notifications fall back to the deprecated `NSUserNotificationCenter`. If you denied the permission prompt, re-enable in **System Settings → Notifications → Jarvis**.

### Python core

```bash
cd jarvis-core
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Optional Features

### Telegram Integration

Control Jarvis remotely from your phone and receive proactive notifications when you're away from your Mac.

**What it enables:**
- Send voice commands to Jarvis via Telegram while away
- Receive notifications when long tasks complete
- Approve or deny destructive actions remotely with `/approve` / `/deny`
- Toggle away mode from the menu bar, via Telegram (`/away` / `/back`), or by voice

**Setup:**

1. Create a bot via [@BotFather](https://t.me/BotFather) on Telegram — copy the token it gives you
2. Find your user ID by messaging [@userinfobot](https://t.me/userinfobot)
3. Add both to `~/.jarvis/config.json`:

```json
"telegram": {
  "bot_token": "YOUR_BOT_TOKEN",
  "allowed_user_id": 123456789
}
```

4. Restart Jarvis — the bot starts automatically if configured

The feature is fully opt-in. If `bot_token` is empty or `allowed_user_id` is `0`, the entire subsystem is disabled with zero overhead.

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
