import pytest
from guardrails import Guardrails, Action, Decision


def make_guardrails(overrides=None):
    cfg = {
        "guardrails": {
            "read_files": "auto_allow",
            "create_files": "auto_allow",
            "edit_files": "auto_allow",
            "run_shell": "auto_allow",
            "web_search": "auto_allow",
            "open_apps": "auto_allow",
            "delete_files": "require_approval",
            "send_messages": "require_approval",
            "modify_system": "require_approval",
            "run_code_with_effects": "require_approval",
        }
    }
    if overrides:
        cfg["guardrails"].update(overrides)
    return Guardrails(cfg)


def test_safe_action_auto_allowed():
    g = make_guardrails()
    action = Action(category="read_files", description="read /tmp/foo.txt")
    assert g.classify(action) == Decision.ALLOW


def test_destructive_action_requires_approval():
    g = make_guardrails()
    action = Action(category="delete_files", description="rm ~/Downloads/logs")
    assert g.classify(action) == Decision.REQUIRE_APPROVAL


def test_trust_session_overrides_approval():
    g = make_guardrails()
    g.trust_for_session("delete_files")
    action = Action(category="delete_files", description="rm ~/Downloads/logs")
    assert g.classify(action) == Decision.ALLOW


def test_toggle_to_strict_blocks_safe_action():
    g = make_guardrails({"run_shell": "require_approval"})
    action = Action(category="run_shell", description="ls /tmp")
    assert g.classify(action) == Decision.REQUIRE_APPROVAL


def test_unknown_category_requires_approval():
    g = make_guardrails()
    action = Action(category="unknown_thing", description="do something weird")
    assert g.classify(action) == Decision.REQUIRE_APPROVAL


def test_revoke_session_trust_restores_approval():
    g = make_guardrails()
    g.trust_for_session("delete_files")
    g.revoke_session_trust("delete_files")
    action = Action(category="delete_files", description="rm ~/Downloads/logs")
    assert g.classify(action) == Decision.REQUIRE_APPROVAL


def test_update_config_invalid_setting_raises():
    g = make_guardrails()
    with pytest.raises(ValueError, match="Invalid setting"):
        g.update_config("delete_files", "always_allow")
