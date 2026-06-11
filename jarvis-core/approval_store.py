"""In-memory registry of paused agent runs awaiting approval.

When an agent hits a guardrail it stops mid-run and registers a resume callable
keyed by the command_id. On approval the server pops and invokes it, so the run
continues from where it stopped (executing the approved tool, then onward) instead
of replaying the whole command from the original text.

In-memory only — paused runs do not survive a server restart, which is acceptable:
approval happens seconds after the prompt, and a lost entry falls back to replay.
"""

from typing import Callable

# command_id → {"resume": resume(step_callback) -> result dict, "meta": {...}}
# meta carries what the router needs to annotate the resumed result and update
# history: user_text, agent label, model name, intent_class.
_PAUSED: dict[str, dict] = {}


def register(command_id: str, resume_fn: Callable, meta: dict | None = None) -> None:
    _PAUSED[command_id] = {"resume": resume_fn, "meta": meta or {}}


def has(command_id: str) -> bool:
    return command_id in _PAUSED


def pop(command_id: str) -> dict | None:
    return _PAUSED.pop(command_id, None)


def discard(command_id: str) -> None:
    """Drop a paused run without resuming (e.g. on denial)."""
    _PAUSED.pop(command_id, None)


def clear() -> None:
    _PAUSED.clear()
