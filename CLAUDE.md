# Jarvis — macOS AI Voice Assistant

## Project Overview

Jarvis is a local-first AI operating layer for macOS. Activated via hotkey (⌃⌥) or text input, it takes voice and text commands and executes them: file operations, shell commands, web search, code execution, app control, developer workflows. A floating HUD shows responses; voice narrates summaries.

Primary executor is a local Qwen3.6 model served via Ollama/MLX. Claude Haiku handles tasks that need cloud reasoning, Sonnet handles complex analysis. A custom fine-tuned classifier routes commands before execution.

## Before Starting Any Work

1. Run `git log --oneline -10` to see recent changes
2. Run `cd jarvis-core && source .venv/bin/activate && pytest` to verify tests pass (440+ tests)
3. Read relevant source files before modifying them
4. Write tests before implementation (TDD)

## Architecture

Two-process system:

- **Swift app** (`jarvis-swift/`) — menu bar icon, floating HUD overlay, global hotkey (⌃⌥), STT via SFSpeechRecognizer, TTS via AVSpeechSynthesizer, 15fps streaming display timer
- **Python core** (`jarvis-core/`) — FastAPI server on localhost:8765, command pipeline, hybrid AI routing, tools, guardrails

They communicate over local HTTP. Swift launches and monitors the Python process.

### Python Core Internal Flow

```
POST /command → CommandPipeline → Router (pre-flight MLX classifier)
                      ↓                    ↓
               ProjectMemory       local_first routing:
                                   - all intents → OllamaAgent (Qwen3.6)
                                   - EscalateToCloud → Agent(Haiku/Sonnet)
                                         ↕ delegate_to_local
                                   OllamaAgent (sub-tasks)
```

- **CommandPipeline** — `JarvisCommand` lifecycle, single-command lock, cancel/abort
- **Router** — pre-flight MLX classifier decides routing; manages conversational history + compaction
- **OllamaAgent** — primary executor (local Qwen3.6); streams finalize() answers as SSE tokens; raises `EscalateToCloud` when it can't handle a task
- **Agent(Haiku)** — cloud fallback for escalated tasks
- **Agent(Sonnet)** — complex reasoning (web search, deep analysis)
- **ProjectMemory** — per-project build/test commands at `~/.jarvis/projects/<hash>/memory.json`

### Conversational Session (Phase E)
- Router holds `_history` (compressed turns) across commands in a session
- Auto-compaction via Haiku when history exceeds ~5K tokens; deferred to next command start so it doesn't block current SSE response
- Classifier receives last 2 history turns for context on follow-ups
- `reset_conversation()` clears history and resets compaction state

### Streaming HUD (Phase E)
- OllamaAgent streams `finalize(answer=...)` argument chunks as SSE `token` events in real-time using an escape-aware JSON string decoder
- Swift queues tokens in `HUDViewModel.tokenQueue`; a 15fps `Timer` drains 20 chars/tick into `streamingBuffer`
- `TurnRowView` renders `streamingBuffer` live on the active turn; transitions to `turn.response` on completion
- `pendingCompleteEvent` defers SSE "complete" finalization until the queue is drained; cleared on error

## Tech Stack

| Layer          | Technology                                                         |
| -------------- | ------------------------------------------------------------------ |
| macOS UI       | Swift / SwiftUI                                                    |
| STT            | Apple SFSpeechRecognizer (en-US, contextual vocabulary biasing)    |
| TTS            | AVSpeechSynthesizer (Daniel premium en-GB voice)                   |
| Local AI       | Qwen3.6-35B-A3B via Ollama + MLX Rapid (3.4x speedup)             |
| Classifier     | Custom fine-tuned Qwen on MLX (jarvis-classifier adapter, port 8090)|
| Cloud AI       | Claude Haiku (fallback) + Sonnet (complex) via Anthropic SDK       |
| Python server  | FastAPI + uvicorn                                                   |
| HTTP client    | httpx (Ollama/MLX calls)                                           |
| Web search     | Brave Search API (or DuckDuckGo fallback)                          |
| IPC            | Local HTTP (localhost:8765)                                        |
| Config         | `~/.jarvis/config.json`                                            |
| Logs           | `~/.jarvis/logs/` (errors, commands, analytics, guardrails)        |
| Project memory | `~/.jarvis/projects/<md5(cwd)>/memory.json`                        |

## Key Decisions

- **Local-first routing** — Qwen3.6 via Ollama/MLX is the primary executor. Cloud (Haiku/Sonnet) is a fallback via `EscalateToCloud`. Goal: eliminate API dependency for most commands.
- **Fine-tuned classifier** — Custom Qwen model trained on 500+ examples classifies intent (read_only/prepare/destructive/complex_reasoning) and routes before execution.
- **Streaming finalize** — OllamaAgent streams the `finalize()` answer content as SSE tokens using an escape-aware JSON string decoder. No tail buffer. Metrics (TTFT, tok/s) tracked for both text and finalize paths.
- **Conversational history** — Router compresses each turn (tool calls + result) and carries it across commands. Compaction is deferred, circuit-breakered on failure.
- **Guardrails-first** — destructive actions require explicit approval. Pre-flight classifier also gates by intent class.
- **Shared dispatch** — `tools/_dispatch.py` has `execute_tool()` + `format_response()` used by all agents.
- **No always-on by default** — hotkey (⌃⌥) is primary trigger.

## Key Files

| File | Purpose |
|---|---|
| `jarvis-core/server.py` | FastAPI endpoints, SSE event pipeline, analytics logging |
| `jarvis-core/agent.py` | Claude agent (Haiku/Sonnet) with tool-use loop |
| `jarvis-core/ollama_agent.py` | Local Ollama/MLX agent; streaming finalize; planning-text detection |
| `jarvis-core/router.py` | Pre-flight classifier, routing, conversational history, compaction |
| `jarvis-core/command_pipeline.py` | Command lifecycle + single-command lock |
| `jarvis-core/guardrails.py` | Approval engine for destructive actions |
| `jarvis-core/config.py` | Config load/save (`~/.jarvis/config.json`) |
| `jarvis-core/memory.py` | Per-project build/test memory |
| `jarvis-core/logger.py` | Rotating logs to `~/.jarvis/logs/` |
| `jarvis-core/tools/` | shell, web, code, macOS, file tools |
| `jarvis-swift/Jarvis/AppDelegate.swift` | App lifecycle, Python core launch |
| `jarvis-swift/Jarvis/AudioController.swift` | Hotkey, STT, TTS, SSE stream, 15fps display timer |
| `jarvis-swift/Jarvis/HUDView.swift` | Conversation thread, streaming turn display |
| `jarvis-swift/Jarvis/HUDViewModel.swift` | Token queue, streamingBuffer, drain logic |

## Logs

Always check logs to diagnose issues — don't guess:

```bash
tail -20 ~/.jarvis/logs/commands.log     # full request/response with steps
tail -20 ~/.jarvis/logs/analytics.log    # duration, ttft_ms, tok_s per command
tail -20 ~/.jarvis/logs/errors.log       # exceptions, classifier failures
```

Analytics fields: `duration_ms`, `agent`, `model`, `ttft_ms`, `gen_tokens`, `tok_s`

## Rules

- Run all Python tests before committing: `cd jarvis-core && source .venv/bin/activate && pytest`
- Build Swift app after every Swift change: `cd jarvis-swift && xcodebuild -scheme Jarvis -configuration Debug build`
- Write tests before implementation (TDD)
- Commit frequently with descriptive messages
- No hardcoded secrets — all config goes through `~/.jarvis/config.json`
- All tools return consistent dict shape: result keys + `"error": None` or `str`
- The SourceKit "line 1:1 Internal error" diagnostic is spurious — ignore it if the build succeeds
