import asyncio
import httpx
import json
import logging
import os
import shutil
import subprocess
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel, field_validator

import config as cfg_module
import logger as logger_module
from command_pipeline import CommandPipeline
from guardrails import Guardrails
from router import Router
from telegram_bot import start_bot, stop_bot
from notifications import notify
import telegram_state

_pipeline = None
_loggers = None
_guardrails = None


def load_dependencies():
    config = cfg_module.load()
    loggers = logger_module.setup()
    guardrails = Guardrails(config)
    router = Router(config=config, guardrails=guardrails)
    from memory import ProjectMemory
    pipeline = CommandPipeline(router=router, memory=ProjectMemory())
    return pipeline, loggers, guardrails


def _ensure_ollama_running() -> None:
    """Start ollama serve in the background if it isn't already running."""
    # When launched from a macOS app the PATH is minimal — probe Homebrew locations directly.
    ollama_bin = (
        shutil.which("ollama")
        or (p if (p := "/opt/homebrew/bin/ollama") and os.path.exists(p) else None)
        or (p if (p := "/usr/local/bin/ollama") and os.path.exists(p) else None)
    )
    if not ollama_bin:
        logging.warning("[Jarvis] ollama binary not found — skipping auto-start")
        return
    try:
        r = httpx.get("http://localhost:11434", timeout=2)
        if r.status_code < 500:
            return  # already running
    except Exception:
        pass  # not running — start it
    subprocess.Popen(
        [ollama_bin, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    logging.info("[Jarvis] ollama serve started via %s", ollama_bin)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline, _loggers, _guardrails
    _ensure_ollama_running()
    _pipeline, _loggers, _guardrails = load_dependencies()
    # Defer bot start until the server is accepting connections.
    # start_bot() before yield risks a message arriving before the HTTP server binds.
    bot_task = asyncio.create_task(_deferred_bot_start())
    yield
    bot_task.cancel()
    try:
        await bot_task
    except asyncio.CancelledError:
        pass
    await stop_bot()


async def _deferred_bot_start():
    # uvicorn typically binds in <100ms; 500ms gives comfortable headroom.
    await asyncio.sleep(0.5)
    await start_bot()


app = FastAPI(lifespan=lifespan)


class CommandRequest(BaseModel):
    text: str
    cwd: str | None = None
    source: str = "hotkey"

    @field_validator("cwd")
    @classmethod
    def normalize_cwd(cls, v: str | None) -> str | None:
        return None if v == "" else v


class ApprovalRequest(BaseModel):
    tool_use_id: str
    approved: bool
    trust_session: bool = False
    category: str = ""


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset_conversation():
    _pipeline.reset_conversation()
    return {"reset": True}


@app.post("/command")
async def command(req: CommandRequest):
    import anthropic as _anthropic
    start = time.time()
    try:
        result = _pipeline.submit(req.text, cwd=req.cwd, source=req.source)
    except _anthropic.RateLimitError:
        msg = "I've hit the API rate limit. Please wait a moment and try again."
        logging.getLogger("jarvis.errors").warning("Rate limit hit on /command")
        return {"speak": msg, "display": msg, "steps": []}
    except Exception as exc:
        logging.getLogger("jarvis.errors").exception("Unhandled error in /command")
        msg = "I'm experiencing an error. Please try again or restart Jarvis."
        return {"speak": msg, "display": msg, "steps": []}
    # Auto-disable away mode if command came from the Mac (not Telegram)
    if req.source != "telegram":
        state = telegram_state.get_state()
        if state.away:
            await notify("🟢 Jarvis back at the Mac — away mode off")
            state.away = False
    duration_ms = int((time.time() - start) * 1000)
    if _loggers:
        _loggers["commands"].info(
            f"cmd={req.text!r} cwd={req.cwd!r} duration_ms={duration_ms} result={result}"
        )
        logger_module.log_analytics(_loggers["analytics"], "command", {
            "duration_ms": duration_ms,
            "cwd": req.cwd,
            "has_approval_required": "approval_required" in result,
            "agent": result.get("_agent"),
            "model": result.get("_model"),
            "escalated": result.get("_escalated"),
            "escalation_reason": result.get("_escalation_reason"),
            "agent_response_ms": result.get("_response_ms"),
        })
        logger_module.log_training(
            _loggers["training"],
            request=req.text,
            cwd=req.cwd,
            intent_class=result.get("_intent_class"),
            model=result.get("_model"),
            steps=result.get("steps", []),
            speak=result.get("speak"),
            display=result.get("display"),
            duration_ms=duration_ms,
        )
    return result


@app.get("/commands")
def list_commands():
    return _pipeline.list_recent()


@app.get("/commands/{command_id}")
def get_command(command_id: str):
    from fastapi import HTTPException
    cmd = _pipeline.get(command_id)
    if cmd is None:
        raise HTTPException(status_code=404, detail="Command not found")
    return {"id": cmd.id, "source": cmd.source, "raw_input": cmd.raw_input,
            "status": cmd.status, "created_at": cmd.created_at, "completed_at": cmd.completed_at}


@app.post("/commands/abort")
def abort():
    return _pipeline.abort()


@app.post("/commands/{command_id}/cancel")
def cancel_command(command_id: str):
    return _pipeline.cancel(command_id)


@app.post("/approve")
def approve(req: ApprovalRequest):
    if req.trust_session and req.category and _guardrails:
        _guardrails.trust_for_session(req.category)
    # Approval is stateless — the client must re-issue the original /command request.
    # next_action tells the client what to do after this response.
    return {
        "acknowledged": True,
        "next_action": "reissue_command" if req.approved else "cancelled",
    }


class ClassifyRequest(BaseModel):
    text: str


def _classify_approval(text: str) -> bool | None:
    """Ask Ollama to classify text as approve/deny/unclear. Returns True/False/None."""
    config = cfg_module.load()
    host = config.get("ollama", {}).get("host", "http://localhost:11434")
    model = config.get("ollama", {}).get("model", "mistral:latest")
    system = (
        'You are a binary classifier. Reply with JSON only — no explanation.\n'
        'If the text means YES / APPROVE / ALLOW, reply: {"approved": true}\n'
        'If the text means NO / DENY / CANCEL, reply: {"approved": false}\n'
        'If unclear, reply: {"approved": null}'
    )
    try:
        r = httpx.post(
            f"{host}/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": text},
                ],
                "stream": False,
            },
            timeout=5.0,
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"].strip()
        return json.loads(content).get("approved")
    except Exception as e:
        logging.getLogger("jarvis.errors").warning("[classify_approval] failed: %s", e)
        return None


@app.post("/approve/classify")
def approve_classify(req: ClassifyRequest):
    result = _classify_approval(req.text)
    return {"approved": result}


class TelegramAwayRequest(BaseModel):
    away: bool


@app.get("/telegram/away")
def telegram_away_get():
    from telegram_state import get_state
    return {"away": get_state().away}


@app.post("/telegram/away")
def telegram_away(req: TelegramAwayRequest):
    from telegram_state import get_state
    get_state().away = req.away
    return {"away": req.away}


def _redact_sensitive(obj):
    """Recursively redact values whose keys end in _key, _secret, or _token."""
    if not isinstance(obj, dict):
        return obj
    return {
        k: "***" if isinstance(v, str) and (
            k.endswith("_key") or k.endswith("_secret") or k.endswith("_token")
        ) else _redact_sensitive(v)
        for k, v in obj.items()
    }


@app.get("/config")
def get_config():
    config = cfg_module.load()
    return _redact_sensitive(config)


def _deep_merge(base: dict, updates: dict) -> dict:
    """Recursively merge updates into base, preserving nested dicts."""
    result = dict(base)
    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@app.post("/config")
def update_config(updates: dict):
    current = cfg_module.load()
    merged = _deep_merge(current, updates)
    cfg_module.save(merged)
    return {"saved": True}


if __name__ == "__main__":
    import uvicorn
    config = cfg_module.load()
    port = config.get("server_port", 8765)
    uvicorn.run("server:app", host="127.0.0.1", port=port, reload=False)
