import json
import logging
from pathlib import Path
from unittest.mock import patch
import logger


def test_loggers_created(tmp_path):
    logs_dir = tmp_path / "logs"
    with patch("logger.LOGS_DIR", logs_dir):
        loggers = logger.setup()
    assert "errors" in loggers
    assert "commands" in loggers
    assert "analytics" in loggers
    assert "guardrails" in loggers
    assert logs_dir.exists()


def test_loggers_do_not_propagate_to_root(tmp_path):
    """Each logger must have propagate=False to avoid double-printing under uvicorn."""
    logs_dir = tmp_path / "logs"
    with patch("logger.LOGS_DIR", logs_dir):
        loggers = logger.setup()
    for name, lg in loggers.items():
        assert lg.propagate is False, f"logger {name} still propagates to root"


def test_analytics_log_writes_json_lines(tmp_path):
    logs_dir = tmp_path / "logs"
    with patch("logger.LOGS_DIR", logs_dir):
        loggers = logger.setup()
        logger.log_analytics(loggers["analytics"], "command", {"tool": "shell", "duration_ms": 123})
    log_file = logs_dir / "analytics.log"
    assert log_file.exists()
    line = json.loads(log_file.read_text().strip().split("\n")[-1])
    assert line["event"] == "command"
    assert line["data"]["tool"] == "shell"
