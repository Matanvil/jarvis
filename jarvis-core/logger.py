import json
import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGS_DIR = Path.home() / ".jarvis" / "logs"
MAX_BYTES = 10 * 1024 * 1024  # 10MB
BACKUP_COUNT = 3


def _make_handler(path: Path) -> RotatingFileHandler:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    return handler


def setup() -> dict:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    loggers = {}
    for name in ["errors", "commands", "analytics", "guardrails", "training"]:
        lg = logging.getLogger(f"jarvis.{name}")
        lg.setLevel(logging.DEBUG)
        for h in lg.handlers[:]:
            h.close()
            lg.removeHandler(h)
        handler = _make_handler(LOGS_DIR / f"{name}.log")
        if name in ("analytics", "training"):
            handler.setFormatter(logging.Formatter("%(message)s"))
        lg.propagate = False
        lg.addHandler(handler)
        loggers[name] = lg
    return loggers


def log_analytics(lg: logging.Logger, event: str, data: dict) -> None:
    record = json.dumps({"ts": time.time(), "event": event, "data": data})
    lg.info(record)


def log_training(lg: logging.Logger, request: str, cwd: str | None,
                 intent_class: str | None, model: str | None,
                 steps: list, speak: str | None, display: str | None,
                 duration_ms: int) -> None:
    record = json.dumps({
        "ts": time.time(),
        "request": request,
        "cwd": cwd,
        "intent_class": intent_class,
        "model": model,
        "steps": steps,
        "speak": speak,
        "display": display[:2000] if display else None,
        "duration_ms": duration_ms,
    }, ensure_ascii=False)
    lg.info(record)
