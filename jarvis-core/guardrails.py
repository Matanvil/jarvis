from dataclasses import dataclass
from enum import Enum


class Decision(Enum):
    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"


@dataclass
class Action:
    category: str
    description: str


class Guardrails:
    def __init__(self, config: dict):
        self._config = config.get("guardrails", {})
        self._session_trusts: set[str] = set()

    def classify(self, action: Action) -> Decision:
        if action.category in self._session_trusts:
            return Decision.ALLOW
        setting = self._config.get(action.category, "require_approval")
        if setting == "auto_allow":
            return Decision.ALLOW
        return Decision.REQUIRE_APPROVAL

    def trust_for_session(self, category: str) -> None:
        """Grant one-time trust for category. Cleared after the next command completes."""
        self._session_trusts.add(category)

    def clear_session_trusts(self) -> None:
        """Revoke all one-time trusts. Called after each command completes."""
        self._session_trusts.clear()

    def revoke_session_trust(self, category: str) -> None:
        self._session_trusts.discard(category)

    _VALID_SETTINGS = {"auto_allow", "require_approval"}

    def update_config(self, category: str, setting: str) -> None:
        if setting not in self._VALID_SETTINGS:
            raise ValueError(f"Invalid setting {setting!r}; must be one of {self._VALID_SETTINGS}")
        self._config[category] = setting
