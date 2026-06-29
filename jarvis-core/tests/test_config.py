import json
import pytest
from pathlib import Path
from unittest.mock import patch
import config


def test_executor_backend_prefers_executor_keys():
    cfg = {"local": {"host": "http://base:11434", "model": "base",
                     "executor_host": "http://exec:8090", "executor_model": "exec-model"}}
    assert config.executor_backend(cfg) == ("http://exec:8090", "exec-model")


def test_executor_backend_falls_back_to_base():
    cfg = {"local": {"host": "http://base:11434", "model": "base"}}
    assert config.executor_backend(cfg) == ("http://base:11434", "base")


def test_classifier_backend_prefers_classifier_keys():
    cfg = {"local": {"host": "http://base:11434", "model": "base",
                     "classifier_host": "http://cls:8090", "classifier_model": "cls-model"}}
    assert config.classifier_backend(cfg) == ("http://cls:8090", "cls-model")


def test_classifier_backend_falls_back_to_base():
    cfg = {"local": {"host": "http://base:11434", "model": "base"}}
    assert config.classifier_backend(cfg) == ("http://base:11434", "base")


def test_embedder_backend_defaults_to_nomic():
    cfg = {"local": {"host": "http://base:11434"}}
    assert config.embedder_backend(cfg) == ("http://base:11434", "nomic-embed-text")


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
    assert "local" in cfg
    assert cfg["local"]["host"] == "http://localhost:11434"
    assert cfg["local"]["model"] == "qwen3.6:35b-a3b"
    assert cfg["local"]["executor_host"] == "http://localhost:8000"
    assert cfg["local"]["executor_model"] == "mlx-community/Qwen3.6-35B-A3B-4bit"
    assert cfg["local"]["executor_rapid_mlx"] is True
    assert cfg["local"]["classifier_model"] == "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    assert cfg["local"]["routing_mode"] == "automatic"
    assert cfg["local"]["timeout_seconds"] == 300


def test_load_deep_merges_ollama(tmp_path, monkeypatch):
    monkeypatch.setattr("config.CONFIG_PATH", tmp_path / "config.json")
    (tmp_path / "config.json").write_text(json.dumps({
        "ollama": {"model": "custom-model"}
    }))
    cfg = config.load()
    # legacy "ollama" key migrated to "local"; custom model preserved, other keys filled from defaults
    assert cfg["local"]["model"] == "custom-model"
    assert cfg["local"]["host"] == "http://localhost:11434"
    assert cfg["local"]["routing_mode"] == "automatic"
    assert cfg["local"]["classifier_model"] == "mlx-community/Qwen3-4B-Instruct-2507-4bit"


def test_defaults_include_reasoning_block(tmp_path, monkeypatch):
    monkeypatch.setattr("config.CONFIG_PATH", tmp_path / "config.json")
    cfg = config.load()
    assert cfg["reasoning"]["max_steps_claude"] == 15
    assert cfg["reasoning"]["max_steps_local"] == 15
    assert "max_total_steps" not in cfg["reasoning"]
    assert "step_budgets" not in cfg["reasoning"]
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
    assert cfg["reasoning"]["max_steps_local"] == 15  # default preserved


def test_config_has_models_section(tmp_path, monkeypatch):
    monkeypatch.setattr("config.CONFIG_PATH", tmp_path / "config.json")
    cfg = config.load()
    assert cfg["models"]["haiku"] == "claude-haiku-4-5-20251001"
    assert cfg["models"]["sonnet"] == "claude-sonnet-4-6"


@pytest.fixture
def tmp_config(tmp_path, monkeypatch):
    monkeypatch.setattr("config.CONFIG_PATH", tmp_path / "config.json")


def test_telegram_defaults_present(tmp_config):
    cfg = config.load()
    assert "telegram" in cfg
    assert cfg["telegram"]["bot_token"] == ""
    assert cfg["telegram"]["allowed_user_id"] == 0


def test_telegram_configured_false_when_missing(tmp_config):
    assert config.telegram_configured() is False


def test_telegram_configured_false_when_no_user_id(tmp_config):
    data = config.load()
    data["telegram"] = {"bot_token": "abc", "allowed_user_id": 0}
    config.save(data)
    assert config.telegram_configured() is False


def test_telegram_configured_false_when_no_token(tmp_config):
    data = config.load()
    data["telegram"] = {"bot_token": "", "allowed_user_id": 12345}
    config.save(data)
    assert config.telegram_configured() is False


def test_telegram_configured_true_when_both_set(tmp_config):
    data = config.load()
    data["telegram"] = {"bot_token": "abc123", "allowed_user_id": 99999}
    config.save(data)
    assert config.telegram_configured() is True


def test_load_deep_merges_telegram(tmp_config):
    config.save({**config.load(), "telegram": {"bot_token": "mytoken"}})
    cfg = config.load()
    assert cfg["telegram"]["bot_token"] == "mytoken"
    assert cfg["telegram"]["allowed_user_id"] == 0


def test_step_voice_default_is_false():
    """narration.step_voice defaults to False."""
    from config import DEFAULTS
    assert DEFAULTS["narration"]["step_voice"] is False


def test_classifier_adapter_path_default_is_empty():
    from config import DEFAULTS
    assert DEFAULTS["local"]["classifier_adapter_path"] == ""


def test_mcp_servers_default_is_empty_list():
    from config import DEFAULTS
    assert DEFAULTS["mcp_servers"] == []


def test_mcp_servers_loads_from_file(tmp_path, monkeypatch):
    monkeypatch.setattr("config.CONFIG_PATH", tmp_path / "config.json")
    (tmp_path / "config.json").write_text(json.dumps({
        "mcp_servers": [{"name": "fs", "command": "npx", "args": []}]
    }))
    cfg = config.load()
    assert len(cfg["mcp_servers"]) == 1
    assert cfg["mcp_servers"][0]["name"] == "fs"


def test_classifier_adapter_path_loads_from_file(tmp_path, monkeypatch):
    monkeypatch.setattr("config.CONFIG_PATH", tmp_path / "config.json")
    (tmp_path / "config.json").write_text(json.dumps({
        "ollama": {"classifier_adapter_path": "/some/adapter"}
    }))
    cfg = config.load()
    # legacy "ollama" key migrated to "local"
    assert cfg["local"]["classifier_adapter_path"] == "/some/adapter"
    assert cfg["local"]["classifier_host"] == "http://127.0.0.1:8090"  # default preserved


def test_dev_config_overlay_is_removed():
    """config.load() must not read config.dev.json — ~/.jarvis/config.json is the only source."""
    import inspect
    import config as cfg_module
    src = inspect.getsource(cfg_module.load)
    assert "DEV_CONFIG_PATH" not in src
    assert "config.dev" not in src


def test_config_local_key_is_used():
    """Config must use 'local' key not 'ollama'."""
    cfg = config.load()
    assert "local" in cfg
    assert "ollama" not in cfg
    assert cfg["local"]["routing_mode"] in ("automatic", "local", "cloud")


def test_config_migrates_ollama_key_to_local(tmp_path, monkeypatch):
    """Loading a config with legacy 'ollama' key must migrate it to 'local'."""
    monkeypatch.setattr("config.CONFIG_PATH", tmp_path / "config.json")
    import json
    (tmp_path / "config.json").write_text(json.dumps({
        "ollama": {"model": "oldmodel", "routing_mode": "local_first"}
    }))
    cfg = config.load()
    assert "local" in cfg
    assert cfg["local"]["model"] == "oldmodel"
    assert cfg["local"]["routing_mode"] == "automatic"


def test_config_migrates_ollama_mode_values(tmp_path, monkeypatch):
    """Legacy mode values must be migrated: ollama_only→local, claude_only→cloud."""
    monkeypatch.setattr("config.CONFIG_PATH", tmp_path / "config.json")
    import json
    (tmp_path / "config.json").write_text(json.dumps({
        "local": {"routing_mode": "ollama_only"}
    }))
    cfg = config.load()
    assert cfg["local"]["routing_mode"] == "local"


def test_config_max_steps_local():
    """reasoning.max_steps_local must exist (renamed from max_steps_ollama)."""
    cfg = config.load()
    assert "max_steps_local" in cfg["reasoning"]
    assert "max_steps_ollama" not in cfg["reasoning"]
