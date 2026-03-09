import copy
import json
from pathlib import Path

CONFIG_PATH = Path.home() / ".jarvis" / "config.json"
DEV_CONFIG_PATH = Path(__file__).parent / "config.dev.json"

DEFAULTS = {
    "anthropic_api_key": "",
    "brave_api_key": "",
    "hotkey": "ctrl+space",
    "wake_word": "hey jarvis",
    "voice": "Daniel",
    "always_on": False,
    "server_port": 8765,
    "guardrails": {
        "read_files": "auto_allow",
        "create_files": "auto_allow",
        "edit_files": "auto_allow",
        "run_shell": "auto_allow",
        "modify_filesystem": "require_approval",
        "web_search": "auto_allow",
        "open_apps": "auto_allow",
        "delete_files": "require_approval",
        "send_messages": "require_approval",
        "modify_system": "require_approval",
        "run_code_with_effects": "auto_allow",
    },
    "ollama": {
        "host": "http://localhost:11434",
        "model": "llama3.1:8b",
        "routing_mode": "haiku_first",   # haiku_first | ollama_first | claude_only | ollama_only
        "timeout_seconds": 30,
    },
    "reasoning": {
        "max_steps_claude": 10,
        "max_steps_ollama": 5,
        "max_total_steps": 20,
        "stall_detection": True,
    },
    "narration": {
        "mode": "milestones",   # milestones | all | silent
    },
    "models": {
        "haiku": "claude-haiku-4-5-20251001",
        "sonnet": "claude-sonnet-4-6",
    },
    "telegram": {
        "bot_token": "",
        "allowed_user_id": 0,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = {**base}
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load() -> dict:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        defaults = copy.deepcopy(DEFAULTS)
        save(defaults)
        cfg = defaults
    else:
        with open(CONFIG_PATH) as f:
            data = json.load(f)
        cfg = {**DEFAULTS, **data}
        cfg["guardrails"] = {**DEFAULTS["guardrails"], **data.get("guardrails", {})}
        cfg["ollama"] = {**DEFAULTS["ollama"], **data.get("ollama", {})}
        cfg["reasoning"] = {**DEFAULTS["reasoning"], **data.get("reasoning", {})}
        cfg["narration"] = {**DEFAULTS["narration"], **data.get("narration", {})}
        cfg["models"] = {**DEFAULTS["models"], **data.get("models", {})}
        cfg["telegram"] = {**DEFAULTS["telegram"], **data.get("telegram", {})}

    if DEV_CONFIG_PATH.exists():
        with open(DEV_CONFIG_PATH) as f:
            dev = json.load(f)
        cfg = _deep_merge(cfg, dev)

    return cfg


def telegram_configured() -> bool:
    """Return True only if both bot_token and allowed_user_id are set."""
    t = load().get("telegram", {})
    return bool(t.get("bot_token")) and int(t.get("allowed_user_id", 0)) != 0


def save(cfg: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
