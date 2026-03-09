import pytest
import telegram_state


@pytest.fixture(autouse=True)
def clean_state():
    telegram_state.reset_state()
    yield
    telegram_state.reset_state()


def test_default_state():
    s = telegram_state.get_state()
    assert s.away is False
    assert s.chat_id is None
    assert s.pending_command is None
    assert s.pending_tool_use_id is None


def test_set_away():
    s = telegram_state.get_state()
    s.away = True
    assert telegram_state.get_state().away is True


def test_set_chat_id():
    s = telegram_state.get_state()
    s.chat_id = 12345
    assert telegram_state.get_state().chat_id == 12345


def test_set_pending():
    s = telegram_state.get_state()
    s.pending_command = "delete my downloads folder"
    s.pending_tool_use_id = "tool_abc123"
    st = telegram_state.get_state()
    assert st.pending_command == "delete my downloads folder"
    assert st.pending_tool_use_id == "tool_abc123"


def test_reset_state():
    s = telegram_state.get_state()
    s.away = True
    s.chat_id = 999
    telegram_state.reset_state()
    s2 = telegram_state.get_state()
    assert s2.away is False
    assert s2.chat_id is None
