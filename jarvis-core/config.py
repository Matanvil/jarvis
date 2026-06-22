import copy
import json
from pathlib import Path

CONFIG_PATH = Path.home() / ".jarvis" / "config.json"

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
    "local": {
        "host": "http://localhost:11434",
        "model": "qwen3.6:35b-a3b",
        "routing_mode": "automatic",   # automatic | local | cloud
        "timeout_seconds": 300,
        "executor_host": "http://localhost:8000",
        "executor_model": "mlx-community/Qwen3.6-35B-A3B-4bit",
        "executor_rapid_mlx": True,
        "executor_chat_template_kwargs": {"enable_thinking": False},
        "classifier_host": "http://127.0.0.1:8090",
        "classifier_model": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
        "classifier_adapter_path": "",
    },
    "reasoning": {
        "max_steps_claude": 15,
        "max_steps_local": 15,
        "stall_detection": True,
    },
    "narration": {
        "mode": "milestones",   # milestones | all | silent
        "step_voice": False,
    },
    "models": {
        "haiku": "claude-haiku-4-5-20251001",
        "sonnet": "claude-sonnet-4-6",
    },
    "telegram": {
        "bot_token": "",
        "allowed_user_id": 0,
    },
    "local_model": "",
    "mcp_servers": [],
}


def _migrate(data: dict) -> dict:
    """Migrate legacy config keys/values in-place."""
    # Move "ollama" → "local"
    if "ollama" in data and "local" not in data:
        data["local"] = data.pop("ollama")
    # Migrate routing mode values
    _MODE_MAP = {
        "local_first": "automatic",
        "ollama_only": "local",
        "claude_only": "cloud",
        "ollama_first": "automatic",
        "haiku_first": "automatic",
    }
    local = data.get("local", {})
    if local.get("routing_mode") in _MODE_MAP:
        local["routing_mode"] = _MODE_MAP[local["routing_mode"]]
    # Rename max_steps_ollama → max_steps_local
    r = data.get("reasoning", {})
    if "max_steps_ollama" in r and "max_steps_local" not in r:
        r["max_steps_local"] = r.pop("max_steps_ollama")
    return data


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
        data = _migrate(data)
        cfg = {**DEFAULTS, **data}
        cfg["guardrails"] = {**DEFAULTS["guardrails"], **data.get("guardrails", {})}
        cfg["local"] = {**DEFAULTS["local"], **data.get("local", {})}
        cfg["reasoning"] = {**DEFAULTS["reasoning"], **data.get("reasoning", {})}
        cfg["narration"] = {**DEFAULTS["narration"], **data.get("narration", {})}
        cfg["models"] = {**DEFAULTS["models"], **data.get("models", {})}
        cfg["telegram"] = {**DEFAULTS["telegram"], **data.get("telegram", {})}

    return cfg


def telegram_configured() -> bool:
    """Return True only if both bot_token and allowed_user_id are set."""
    t = load().get("telegram", {})
    return bool(t.get("bot_token")) and int(t.get("allowed_user_id", 0)) != 0


def save(cfg: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


# ── Backend resolution ────────────────────────────────────────────────────────
# Single source of truth for which (host, model) each local role talks to. Before
# this, the executor used ollama.executor_*, the router classifier used
# ollama.classifier_*, but approval-classify and project-memory discovery read the
# base ollama.host/model — so they hit the wrong server when the executor/classifier
# ran on a separate MLX endpoint. All call sites now resolve through these helpers.

def executor_backend(config: dict) -> tuple[str, str]:
    """(host, model) for the local executor model. Prefers explicit executor_* keys,
    falls back to the base local host/model."""
    o = config.get("local", {})
    host = o.get("executor_host") or o.get("host") or "http://localhost:11434"
    model = o.get("executor_model") or o.get("model") or "mistral:latest"
    return host, model


def classifier_backend(config: dict) -> tuple[str, str]:
    """(host, model) for the intent / approval classifier."""
    o = config.get("local", {})
    host = o.get("classifier_host") or o.get("host") or "http://localhost:11434"
    model = o.get("classifier_model") or o.get("model") or "mistral:latest"
    return host, model


def embedder_backend(config: dict) -> tuple[str, str]:
    """(host, model) for embeddings (coding-agent semantic index)."""
    o = config.get("local", {})
    host = o.get("embedder_host") or o.get("host") or "http://localhost:11434"
    model = o.get("embedder_model") or "nomic-embed-text"
    return host, model
