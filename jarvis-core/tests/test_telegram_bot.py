import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import telegram_state


@pytest.fixture(autouse=True)
def clean_state():
    telegram_state.reset_state()
    yield
    telegram_state.reset_state()


def _make_update(user_id: int, text: str, chat_id: int = 999):
    update = MagicMock()
    update.effective_user.id = user_id
    update.effective_chat.id = chat_id
    update.message.text = text
    update.message.reply_text = AsyncMock()
    return update


def _make_context():
    ctx = MagicMock()
    ctx.bot.send_chat_action = AsyncMock()
    return ctx


async def test_invalid_user_silently_ignored():
    import telegram_bot
    with patch("telegram_bot.cfg_module.load", return_value={"telegram": {"bot_token": "x", "allowed_user_id": 111}}):
        update = _make_update(user_id=999, text="hello")
        await telegram_bot._handle_message(update, _make_context())
    update.message.reply_text.assert_not_called()


async def test_valid_user_sets_chat_id():
    import telegram_bot
    with patch("telegram_bot.cfg_module.load", return_value={"telegram": {"bot_token": "x", "allowed_user_id": 111}}):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"speak": "done", "display": "done"}
        with patch("telegram_bot.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_client
            update = _make_update(user_id=111, text="list files", chat_id=555)
            await telegram_bot._handle_message(update, _make_context())
    assert telegram_state.get_state().chat_id == 555


async def test_valid_message_sends_reply():
    import telegram_bot
    with patch("telegram_bot.cfg_module.load", return_value={"telegram": {"bot_token": "x", "allowed_user_id": 111}}):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"speak": "here are your files", "display": "here are your files"}
        with patch("telegram_bot.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_client
            update = _make_update(user_id=111, text="list files")
            await telegram_bot._handle_message(update, _make_context())
    update.message.reply_text.assert_awaited_once_with("here are your files")


async def test_approval_required_stores_pending():
    import telegram_bot
    with patch("telegram_bot.cfg_module.load", return_value={"telegram": {"bot_token": "x", "allowed_user_id": 111}}):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "approval_required": {
                "tool_use_id": "tool_abc",
                "description": "delete files",
            }
        }
        with patch("telegram_bot.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_client
            update = _make_update(user_id=111, text="delete all logs")
            await telegram_bot._handle_message(update, _make_context())
    state = telegram_state.get_state()
    assert state.pending_command == "delete all logs"
    assert state.pending_tool_use_id == "tool_abc"
    assert "/approve" in update.message.reply_text.call_args[0][0]


async def test_away_command_sets_away():
    import telegram_bot
    with patch("telegram_bot.cfg_module.load", return_value={"telegram": {"bot_token": "x", "allowed_user_id": 111}}):
        update = _make_update(user_id=111, text="/away")
        await telegram_bot._handle_away(update, _make_context())
    assert telegram_state.get_state().away is True
    update.message.reply_text.assert_awaited_once()


async def test_back_command_sets_not_away():
    import telegram_bot
    telegram_state.get_state().away = True
    with patch("telegram_bot.cfg_module.load", return_value={"telegram": {"bot_token": "x", "allowed_user_id": 111}}):
        update = _make_update(user_id=111, text="/back")
        await telegram_bot._handle_back(update, _make_context())
    assert telegram_state.get_state().away is False
    update.message.reply_text.assert_awaited_once()


async def test_away_invalid_user_ignored():
    import telegram_bot
    with patch("telegram_bot.cfg_module.load", return_value={"telegram": {"bot_token": "x", "allowed_user_id": 111}}):
        update = _make_update(user_id=999, text="/away")
        await telegram_bot._handle_away(update, _make_context())
    assert telegram_state.get_state().away is False
    update.message.reply_text.assert_not_called()


async def test_approve_with_pending():
    import telegram_bot
    state = telegram_state.get_state()
    state.chat_id = 999
    state.pending_command = "delete logs"
    state.pending_tool_use_id = "tool_123"
    with patch("telegram_bot.cfg_module.load", return_value={"telegram": {"bot_token": "x", "allowed_user_id": 111}}):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"speak": "done", "display": "done"}
        with patch("telegram_bot.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_client
            update = _make_update(user_id=111, text="/approve")
            await telegram_bot._handle_approve(update, _make_context())
    assert telegram_state.get_state().pending_command is None
    assert telegram_state.get_state().pending_tool_use_id is None
    update.message.reply_text.assert_awaited_once_with("done")


async def test_approve_no_pending():
    import telegram_bot
    with patch("telegram_bot.cfg_module.load", return_value={"telegram": {"bot_token": "x", "allowed_user_id": 111}}):
        update = _make_update(user_id=111, text="/approve")
        await telegram_bot._handle_approve(update, _make_context())
    update.message.reply_text.assert_awaited_once_with("Nothing pending approval.")


async def test_deny_with_pending():
    import telegram_bot
    state = telegram_state.get_state()
    state.pending_command = "delete logs"
    state.pending_tool_use_id = "tool_123"
    with patch("telegram_bot.cfg_module.load", return_value={"telegram": {"bot_token": "x", "allowed_user_id": 111}}):
        with patch("telegram_bot.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=MagicMock())
            mock_cls.return_value = mock_client
            update = _make_update(user_id=111, text="/deny")
            await telegram_bot._handle_deny(update, _make_context())
    assert telegram_state.get_state().pending_command is None
    update.message.reply_text.assert_awaited_once_with("Action denied.")
