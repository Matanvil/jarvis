# Jarvis — macOS AI Voice Assistant

## Project Overview

Jarvis is a local-first AI operating layer for macOS. Activated via hotkey (⌃Space) or wake word ("Hey Jarvis"), it takes voice commands and executes them: file operations, shell commands, web search, code execution, app control, developer workflows. A floating HUD shows responses; voice narrates summaries.

Uses a hybrid AI engine — Ollama classifies intent before execution, Claude Haiku handles most tasks (~1-2s), Claude Sonnet handles complex reasoning, with two-way delegation between Claude and Ollama.

## Before Starting Any Work

1. Run `git log --oneline -10` to see recent changes
2. Run `cd jarvis-core && source .venv/bin/activate && pytest` to verify tests pass
3. Read relevant source files before modifying them
4. Write tests before implementation (TDD)

## Architecture

Two-process system:

- **Swift app** (`jarvis-swift/`) — menu bar icon, floating HUD overlay, global hotkey (⌃Space), always-on wake word, STT via SFSpeechRecognizer, TTS via macOS `say`
- **Python core** (`jarvis-core/`) — FastAPI server on localhost:8765, command pipeline, hybrid AI routing, tools, guardrails

They communicate over local HTTP. Swift launches and monitors the Python process.

### Python Core Internal Flow

```
POST /command → CommandPipeline → Router (pre-flight Ollama classifier)
                      ↓                    ↓
               ProjectMemory       haiku_first routing:
                                   - non-complex → Agent(Haiku)
                                   - complex_reasoning → Agent(Sonnet)
                                         ↕ delegate_to_local
                                   OllamaAgent (sub-tasks only)
```

- **CommandPipeline** — `JarvisCommand` lifecycle, single-command lock, cancel/abort
- **Router** — pre-flight Ollama classifier decides routing before any execution
- **Agent(Haiku)** — primary executor for most tasks; fast, reliable (~1-2s)
- **Agent(Sonnet)** — complex reasoning (web search, deep analysis); escalated by router
- **OllamaAgent** — handles sub-tasks delegated from Claude via `delegate_to_local()`
- **ProjectMemory** — per-project build/test commands at `~/.jarvis/projects/<hash>/memory.json`

## Tech Stack

| Layer          | Technology                                                   |
| -------------- | ------------------------------------------------------------ |
| macOS UI       | Swift / SwiftUI                                              |
| STT            | Apple SFSpeechRecognizer                                     |
| TTS            | macOS `say` command                                          |
| Local AI       | Ollama (`llama3.1:8b` default, configurable)                 |
| Cloud AI       | Claude Haiku (primary) + Sonnet (complex) via Anthropic SDK  |
| Python server  | FastAPI + uvicorn                                            |
| HTTP client    | httpx (Ollama calls)                                         |
| Web search     | Brave Search API (or DuckDuckGo fallback)                    |
| IPC            | Local HTTP (localhost:8765)                                  |
| Config         | `~/.jarvis/config.json`                                      |
| Logs           | `~/.jarvis/logs/` (errors, commands, analytics, guardrails)  |
| Project memory | `~/.jarvis/projects/<md5(cwd)>/memory.json`                  |

## Key Decisions

- **Haiku-first routing** — pre-flight Ollama classifier runs before execution. Non-complex → Haiku. `complex_reasoning` → Sonnet. No mid-execution escalation.
- **Two-way delegation** — Claude can call `delegate_to_local()` to offload simple sub-tasks to Ollama.
- **Guardrails-first** — destructive actions require explicit approval. Pre-flight classifier also gates by intent class.
- **Response heuristic** — < 150 chars: speak + show in HUD. ≥ 150 chars or code: speak summary only, show full in HUD.
- **Shared dispatch** — `tools/_dispatch.py` has `execute_tool()` + `format_response()` used by both agents.
- **No Portkey/API gateway** — direct Anthropic SDK. Low volume, unique voice commands don't benefit from caching.
- **No always-on by default** — hotkey (⌃Space) is primary trigger; always-on is opt-in toggle in menu bar.

## Key Files

| File | Purpose |
|---|---|
| `jarvis-core/server.py` | FastAPI endpoints |
| `jarvis-core/agent.py` | Claude agent (Haiku/Sonnet) with tool-use loop |
| `jarvis-core/ollama_agent.py` | Ollama local agent |
| `jarvis-core/router.py` | Pre-flight classifier + haiku_first routing |
| `jarvis-core/command_pipeline.py` | Command lifecycle + single-command lock |
| `jarvis-core/guardrails.py` | Approval engine for destructive actions |
| `jarvis-core/config.py` | Config load/save (`~/.jarvis/config.json`) |
| `jarvis-core/memory.py` | Per-project build/test memory |
| `jarvis-core/logger.py` | Rotating logs to `~/.jarvis/logs/` |
| `jarvis-core/tools/` | shell, web, code, macOS, file tools |
| `jarvis-swift/Jarvis/AppDelegate.swift` | App lifecycle, Python core launch |
| `jarvis-swift/Jarvis/AudioController.swift` | Hotkey, STT, TTS, approval prompts |

## Rules

- Run all Python tests before committing: `cd jarvis-core && source .venv/bin/activate && pytest`
- Build Swift app (⌘B) after every Swift file change to catch compile errors early
- Write tests before implementation (TDD)
- Commit frequently with descriptive messages
- No hardcoded secrets — all config goes through `~/.jarvis/config.json`
- All tools return consistent dict shape: result keys + `"error": None` or `str`
