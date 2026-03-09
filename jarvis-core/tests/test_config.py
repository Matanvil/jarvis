import json
import pytest
from pathlib import Path
from unittest.mock import patch
import config


def test_config_creates_default_file(tmp_path):
    config_path = tmp_path / "config.json"
    with patch("config.CONFIG_PATH", config_path):
        cfg = config.load()
    assert config_path.exists()
    assert cfg["hotkey"] == "ctrl+space"
    assert cfg["voice"] == "Daniel"
    assert cfg["always_on"] is False
    assert "guardrails" in cfg
    assert cfg["guardrails"]["delete_files"] == "require_approval"


def test_config_loads_existing_file(tmp_path):
    config_path = tmp_path / "config.json"
    existing = {"hotkey": "cmd+j", "voice": "Alex", "always_on": True,
                 "anthropic_api_key": "sk-test", "guardrails": {}}
    config_path.write_text(json.dumps(existing))
    with patch("config.CONFIG_PATH", config_path):
        cfg = config.load()
    assert cfg["hotkey"] == "cmd+j"
    assert cfg["anthropic_api_key"] == "sk-test"


def test_config_save_and_reload(tmp_path):
    config_path = tmp_path / "config.json"
    with patch("config.CONFIG_PATH", config_path):
        cfg = config.load()
        cfg["voice"] = "Victoria"
        config.save(cfg)
        reloaded = config.load()
    assert reloaded["voice"] == "Victoria"


def test_defaults_include_ollama_block(tmp_path, monkeypatch):
    monkeypatch.setattr("config.CONFIG_PATH", tmp_path / "config.json")
    cfg = config.load()
    assert "ollama" in cfg
    assert cfg["ollama"]["host"] == "http://localhost:11434"
    assert cfg["ollama"]["model"] == "llama3.1:8b"
    assert cfg["ollama"]["routing_mode"] == "haiku_first"
    assert cfg["ollama"]["timeout_seconds"] == 30


def test_load_deep_merges_ollama(tmp_path, monkeypatch):
    monkeypatch.setattr("config.CONFIG_PATH", tmp_path / "config.json")
    (tmp_path / "config.json").write_text(json.dumps({
        "ollama": {"model": "llama3.1:8b"}
    }))
    cfg = config.load()
    # custom model preserved, other keys filled from defaults
    assert cfg["ollama"]["model"] == "llama3.1:8b"
    assert cfg["ollama"]["host"] == "http://localhost:11434"
    assert cfg["ollama"]["routing_mode"] == "haiku_first"


def test_defaults_include_reasoning_block(tmp_path, monkeypatch):
    monkeypatch.setattr("config.CONFIG_PATH", tmp_path / "config.json")
    cfg = config.load()
    assert cfg["reasoning"]["max_steps_claude"] == 10
    assert cfg["reasoning"]["max_steps_ollama"] == 5
    assert cfg["reasoning"]["max_total_steps"] == 20
    assert cfg["reasoning"]["stall_detection"] is True


def test_defaults_include_narration_block(tmp_path, monkeypatch):
    monkeypatch.setattr("config.CONFIG_PATH", tmp_path / "config.json")
    cfg = config.load()
    assert cfg["narration"]["mode"] == "milestones"


def test_load_deep_merges_reasoning(tmp_path, monkeypatch):
    monkeypatch.setattr("config.CONFIG_PATH", tmp_path / "config.json")
    (tmp_path / "config.json").write_text(json.dumps({
        "reasoning": {"max_steps_claude": 10}
    }))
    cfg = config.load()
    assert cfg["reasoning"]["max_steps_claude"] == 10
    assert cfg["reasoning"]["max_total_steps"] == 20   # default preserved


def test_config_has_models_section(tmp_path, monkeypatch):
    monkeypatch.setattr("config.CONFIG_PATH", tmp_path / "config.json")
    cfg = config.load()
    assert cfg["models"]["haiku"] == "claude-haiku-4-5-20251001"
    assert cfg["models"]["sonnet"] == "claude-sonnet-4-6"
