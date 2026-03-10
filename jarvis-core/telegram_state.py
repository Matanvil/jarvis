from dataclasses import dataclass


@dataclass
class TelegramState:
    away: bool = False
    chat_id: int | None = None
    pending_command: str | None = None
    pending_tool_use_id: str | None = None
    pending_category: str | None = None


_state = TelegramState()


def get_state() -> TelegramState:
    return _state


def reset_state() -> None:
    global _state
    _state = TelegramState()
