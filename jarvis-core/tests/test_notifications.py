import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import telegram_state


@pytest.fixture(autouse=True)
def clean_state():
    telegram_state.reset_state()
    yield
    telegram_state.reset_state()


async def test_notify_noop_when_not_away():
    import notifications
    with patch("notifications.get_bot") as mock_get_bot:
        await notifications.notify("hello")
        mock_get_bot.assert_not_called()


async def test_notify_noop_when_no_chat_id():
    import notifications
    telegram_state.get_state().away = True
    with patch("notifications.get_bot") as mock_get_bot:
        await notifications.notify("hello")
        mock_get_bot.assert_not_called()


async def test_notify_noop_when_bot_none():
    import notifications
    s = telegram_state.get_state()
    s.away = True
    s.chat_id = 12345
    with patch("notifications.get_bot", return_value=None):
        await notifications.notify("hello")  # should not raise


async def test_notify_sends_when_away_with_chat_id():
    import notifications
    s = telegram_state.get_state()
    s.away = True
    s.chat_id = 12345
    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock()
    with patch("notifications.get_bot", return_value=mock_bot):
        await notifications.notify("build complete!")
    mock_bot.send_message.assert_awaited_once_with(chat_id=12345, text="build complete!")


async def test_notify_swallows_exceptions():
    import notifications
    s = telegram_state.get_state()
    s.away = True
    s.chat_id = 12345
    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock(side_effect=Exception("network error"))
    with patch("notifications.get_bot", return_value=mock_bot):
        await notifications.notify("message")  # should not raise
