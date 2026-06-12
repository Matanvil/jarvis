import asyncio
import httpx
import json
import logging
import os
import shutil
import subprocess
import time
import uuid as _uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

import alert_bus
import config as cfg_module
import logger as logger_module
from command_pipeline import CommandPipeline
from guardrails import Guardrails
from router import Router
from telegram_bot import start_bot, stop_bot
from notifications import notify
import telegram_state
import scheduler as sched_module
from scheduler import Scheduler, get_scheduler
from schedule_store import ScheduleStore
from step_dispatcher import StepDispatcher

_pipeline = None
_loggers = None
_guardrails = None
_dispatchers: dict[str, StepDispatcher] = {}


def load_dependencies():
    config = cfg_module.load()
    loggers = logger_module.setup()
    guardrails = Guardrails(config)
    router = Router(config=config, guardrails=guardrails)
    from memory import ProjectMemory
    pipeline = CommandPipeline(router=router, memory=ProjectMemory())
    store = ScheduleStore()
    return pipeline, loggers, guardrails, store


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


def _ensure_mlx_server_running(config: dict) -> None:
    """Start mlx_lm.server in the background if local_first mode is active and it isn't already running."""
    ollama_cfg = config.get("ollama", {})
    if ollama_cfg.get("routing_mode") != "local_first":
        return
    base_host = ollama_cfg.get("host", "http://localhost:11434")
    executor_host = ollama_cfg.get("executor_host", "")
    executor_model = ollama_cfg.get("executor_model", "")
    if not executor_host or not executor_model:
        return
    if executor_host == base_host:
        return

    try:
        r = httpx.get(f"{executor_host}/v1/models", timeout=2)
        if r.status_code < 500:
            return  # already running
    except Exception:
        pass  # not running — start it

    mlx_bin = (
        shutil.which("mlx_lm.server")
        or (p if (p := "/opt/homebrew/bin/mlx_lm.server") and os.path.exists(p) else None)
    )
    if not mlx_bin:
        logging.warning("[Jarvis] mlx_lm.server binary not found — skipping auto-start")
        return

    from urllib.parse import urlparse
    parsed = urlparse(executor_host)
    port = parsed.port or 8090

    chat_template_kwargs = ollama_cfg.get("executor_chat_template_kwargs", {})
    chat_template_args = json.dumps(chat_template_kwargs) if chat_template_kwargs else None

    cmd = [mlx_bin, "--model", executor_model, "--port", str(port)]
    if chat_template_args:
        cmd += ["--chat-template-args", chat_template_args]

    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    logging.info("[Jarvis] mlx_lm.server started: model=%s port=%d", executor_model, port)


def _ensure_classifier_server_running(config: dict) -> None:
    """Start mlx_lm.server for the classifier if it runs on a separate host and isn't already up."""
    ollama_cfg = config.get("ollama", {})
    base_host = ollama_cfg.get("host", "http://localhost:11434")
    classifier_host = ollama_cfg.get("classifier_host", "")
    classifier_model = ollama_cfg.get("classifier_model", "")
    if not classifier_host or not classifier_model:
        return
    if classifier_host == base_host:
        return  # classifier is on Ollama, no MLX server needed

    try:
        r = httpx.get(f"{classifier_host}/v1/models", timeout=2)
        if r.status_code < 500:
            return  # already running
    except Exception:
        pass  # not running — start it

    mlx_bin = (
        shutil.which("mlx_lm.server")
        or (p if (p := "/opt/homebrew/bin/mlx_lm.server") and os.path.exists(p) else None)
    )
    if not mlx_bin:
        logging.warning("[Jarvis] mlx_lm.server binary not found — skipping classifier auto-start")
        return

    from urllib.parse import urlparse
    parsed = urlparse(classifier_host)
    port = parsed.port or 8090

    cmd = [mlx_bin, "--model", classifier_model, "--port", str(port)]

    adapter_path = ollama_cfg.get("classifier_adapter_path", "")
    if adapter_path and os.path.exists(adapter_path):
        cmd += ["--adapter-path", adapter_path]

    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    logging.info("[Jarvis] mlx_lm.server (classifier) started: model=%s port=%d adapter=%s",
                 classifier_model, port, adapter_path or "none")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline, _loggers, _guardrails
    config = cfg_module.load()
    _ensure_ollama_running()
    _ensure_mlx_server_running(config)
    _ensure_classifier_server_running(config)
    alert_bus.set_loop(asyncio.get_running_loop())
    _pipeline, _loggers, _guardrails, store = load_dependencies()
    scheduler = Scheduler(store=store, pipeline=_pipeline)
    sched_module.set_scheduler(scheduler)
    scheduler.start()
    bot_task = asyncio.create_task(_deferred_bot_start())
    yield
    bot_task.cancel()
    try:
        await bot_task
    except asyncio.CancelledError:
        pass
    try:
        await stop_bot()
    finally:
        scheduler.stop()


async def _deferred_bot_start():
    # uvicorn typically binds in <100ms; 500ms gives comfortable headroom.
    await asyncio.sleep(0.5)
    await start_bot()


app = FastAPI(lifespan=lifespan)

# Shared-secret auth between the Swift app and this local server. The Swift app
# generates a token at launch and passes it to this process via JARVIS_AUTH_TOKEN,
# then sends it as the X-Jarvis-Token header. Without this, any local process could
# POST /command and get arbitrary shell execution. Enforced only when the token is
# set — a manually-run dev server (or the test client) stays open.
_AUTH_TOKEN = os.environ.get("JARVIS_AUTH_TOKEN", "")
_AUTH_EXEMPT_PATHS = {"/health"}


@app.middleware("http")
async def _auth_middleware(request, call_next):
    if _AUTH_TOKEN and request.url.path not in _AUTH_EXEMPT_PATHS:
        if request.headers.get("X-Jarvis-Token") != _AUTH_TOKEN:
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=401, content={"detail": "unauthorized"})
    return await call_next(request)


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
    command_id: str = ""


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset_conversation():
    _pipeline.reset_conversation()
    return {"reset": True}


def _log_command(req: "CommandRequest", start: float, result: dict) -> None:
    """Log command analytics and training data."""
    _log_command_result(req, start, result, resumed=False)


def _log_command_result(req: "CommandRequest", start: float, result: dict, *, resumed: bool) -> None:
    """Log command analytics and training data for both fresh and resumed runs."""
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
            "resumed": resumed,
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


async def _handle_away_cancel(req: "CommandRequest") -> None:
    """Auto-disable away mode if command came from the Mac (not Telegram)."""
    if req.source != "telegram":
        state = telegram_state.get_state()
        if state.away:
            await notify("🟢 Jarvis back at the Mac — away mode off")
            state.away = False


@app.post("/command")
async def command(req: CommandRequest):
    import anthropic as _anthropic
    start = time.time()
    command_id = str(_uuid.uuid4())
    loop = asyncio.get_running_loop()

    if req.source == "telegram":
        # Blocking path — Telegram bot reads result from HTTP response body
        dispatcher = StepDispatcher(command_id, req.source, loop)
        try:
            result = await loop.run_in_executor(
                None,
                lambda: _pipeline.submit(
                    req.text, cwd=req.cwd, source=req.source,
                    step_callback=dispatcher.on_step, command_id=command_id
                )
            )
        except _anthropic.RateLimitError:
            msg = "I've hit the API rate limit. Please wait a moment and try again."
            logging.getLogger("jarvis.errors").warning("Rate limit hit on /command")
            duration_ms = int((time.time() - start) * 1000)
            if _loggers:
                _loggers["commands"].warning(
                    f"cmd={req.text!r} source={req.source!r} duration_ms={duration_ms} error='rate_limit'"
                )
            return {"speak": msg, "display": msg, "steps": []}
        except Exception as exc:
            logging.getLogger("jarvis.errors").exception("Unhandled error in /command")
            duration_ms = int((time.time() - start) * 1000)
            if _loggers:
                _loggers["commands"].error(
                    f"cmd={req.text!r} source={req.source!r} duration_ms={duration_ms} error={exc!r}"
                )
            msg = "I'm experiencing an error. Please try again or restart Jarvis."
            return {"speak": msg, "display": msg, "steps": []}
        finally:
            if _guardrails:
                _guardrails.clear_session_trusts()
        _log_command(req, start, result)
        await _handle_away_cancel(req)
        return {**result, "command_id": command_id}

    else:
        # Non-blocking path — Swift reads result from SSE stream
        dispatcher = StepDispatcher(command_id, req.source, loop)
        _dispatchers[command_id] = dispatcher

        async def _run_bg():
            try:
                result = await loop.run_in_executor(
                    None,
                    lambda: _pipeline.submit(
                        req.text, cwd=req.cwd, source=req.source,
                        step_callback=dispatcher.on_step, command_id=command_id
                    )
                )
                _log_command(req, start, result)
                await _handle_away_cancel(req)
                dispatcher.complete(result)
            except _anthropic.RateLimitError:
                msg = "I've hit the API rate limit. Please wait a moment and try again."
                logging.getLogger("jarvis.errors").warning("Rate limit hit on /command bg")
                dispatcher.error(msg)
            except Exception as exc:
                logging.getLogger("jarvis.errors").exception("Unhandled error in /command bg")
                dispatcher.error("I'm experiencing an error. Please try again or restart Jarvis.")
            finally:
                if _guardrails:
                    _guardrails.clear_session_trusts()
                # TTL cleanup: if SSE client never connects, remove dispatcher after 30s
                loop.call_later(30, lambda: _dispatchers.pop(command_id, None))

        asyncio.create_task(_run_bg())
        return {"command_id": command_id}


@app.get("/events/{command_id}")
async def events(command_id: str):
    dispatcher = _dispatchers.get(command_id)
    if dispatcher is None:
        return {"error": "command_id not found or already complete"}

    async def stream():
        try:
            while True:
                event = await asyncio.wait_for(dispatcher.queue.get(), timeout=300.0)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") in ("complete", "error"):
                    break
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'type': 'error', 'message': 'timeout'})}\n\n"
        finally:
            _dispatchers.pop(command_id, None)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/alerts")
async def alert_stream():
    """SSE stream of push notifications for Swift clients."""
    q: asyncio.Queue = asyncio.Queue(maxsize=50)
    alert_bus.subscribe(q)

    async def stream():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    # Keepalive ping so the connection doesn't idle-timeout
                    yield f"data: {json.dumps({'type': 'ping'})}\n\n"
                    continue
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            alert_bus.unsubscribe(q)

    return StreamingResponse(stream(), media_type="text/event-stream")


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
async def approve(req: ApprovalRequest):
    import approval_store
    if not req.approved:
        approval_store.discard(req.command_id)
        return {"acknowledged": True, "next_action": "cancelled"}

    if req.trust_session and req.category and _guardrails:
        _guardrails.trust_for_session(req.category)

    # If we still hold the paused run, resume it in place (execute the approved tool
    # and continue) instead of having the client replay the whole command. Falls back
    # to reissue_command when there is no paused state (e.g. after a server restart).
    if req.command_id and approval_store.has(req.command_id):
        paused_entry = approval_store.get(req.command_id) or {}
        paused_meta = paused_entry.get("meta", {})
        start = time.time()
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(None, lambda: _pipeline.resume(req.command_id))
        except Exception:
            logging.getLogger("jarvis.errors").exception("Resume after approval failed")
            approval_store.discard(req.command_id)
            return {"acknowledged": True, "next_action": "reissue_command"}
        finally:
            if _guardrails:
                _guardrails.clear_session_trusts()
        if result is None or result.get("busy"):
            return {"acknowledged": True, "next_action": "reissue_command"}
        resume_req = CommandRequest(
            text=paused_meta.get("user_text", ""),
            cwd=paused_meta.get("cwd"),
            source=paused_meta.get("source", "approval"),
        )
        _log_command_result(resume_req, start, result, resumed=True)
        return {"acknowledged": True, "next_action": "resumed", **result}

    # No paused state — client re-issues the original command (category now trusted).
    return {"acknowledged": True, "next_action": "reissue_command"}


class ClassifyRequest(BaseModel):
    text: str


def _classify_approval(text: str) -> bool | None:
    """Ask Ollama to classify text as approve/deny/unclear. Returns True/False/None."""
    config = cfg_module.load()
    host, model = cfg_module.classifier_backend(config)
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
    global _guardrails
    current = cfg_module.load()
    merged = _deep_merge(current, updates)
    cfg_module.save(merged)
    # Hot-reload guardrail settings so they take effect without a server restart.
    if _guardrails and "guardrails" in merged:
        for category, setting in merged["guardrails"].items():
            try:
                _guardrails.update_config(category, setting)
            except ValueError:
                pass
    return {"saved": True}


class CreateScheduleRequest(BaseModel):
    command: str
    label: str
    schedule_type: str
    cron: str | None = None
    run_at_iso: str | None = None


class PatchScheduleRequest(BaseModel):
    enabled: bool


@app.get("/schedules")
def list_schedules():
    s = get_scheduler()
    if s is None:
        return {"schedules": []}
    from dataclasses import asdict
    return {"schedules": [asdict(x) for x in s.list()]}


@app.post("/schedules")
def create_schedule(req: CreateScheduleRequest):
    s = get_scheduler()
    if s is None:
        raise HTTPException(status_code=503, detail="Scheduler not running")
    from dataclasses import asdict
    schedule = s.create(req.command, req.label, req.schedule_type, req.cron, req.run_at_iso)
    return asdict(schedule)


@app.delete("/schedules/{schedule_id}")
def delete_schedule(schedule_id: str):
    s = get_scheduler()
    if s is None:
        raise HTTPException(status_code=503, detail="Scheduler not running")
    if not s.delete(schedule_id):
        raise HTTPException(status_code=404, detail="Schedule not found")
    return {"ok": True}


@app.patch("/schedules/{schedule_id}")
def update_schedule(schedule_id: str, req: PatchScheduleRequest):
    s = get_scheduler()
    if s is None:
        raise HTTPException(status_code=503, detail="Scheduler not running")
    from dataclasses import asdict
    result = s.resume(schedule_id) if req.enabled else s.pause(schedule_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Schedule not found")
    return asdict(result)


if __name__ == "__main__":
    import uvicorn
    config = cfg_module.load()
    port = config.get("server_port", 8765)
    uvicorn.run("server:app", host="127.0.0.1", port=port, reload=False)
