import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from step_dispatcher import StepDispatcher


@pytest.fixture
def loop():
    lp = asyncio.new_event_loop()
    yield lp
    lp.close()


def test_step_pushed_to_queue(loop):
    """step event is pushed to the SSE queue."""
    dispatcher = StepDispatcher(command_id="cmd1", source="hotkey", loop=loop)
    event = {"type": "step", "label": "Running command", "tool": "shell_run", "milestone": True}
    dispatcher.on_step(event)
    loop.run_until_complete(asyncio.sleep(0))  # drain callbacks
    assert not dispatcher.queue.empty()
    queued = dispatcher.queue.get_nowait()
    assert queued == event


def test_no_telegram_for_hotkey_not_away(loop):
    """Telegram NOT notified for hotkey source when not in away mode."""
    mock_bot = MagicMock()
    with patch("step_dispatcher.get_state") as mock_state:
        mock_state.return_value.away = False
        mock_state.return_value.chat_id = 123
        with patch("step_dispatcher.get_bot", return_value=mock_bot):
            with patch("step_dispatcher.asyncio.run_coroutine_threadsafe") as mock_rcts:
                dispatcher = StepDispatcher(command_id="cmd1", source="hotkey", loop=loop)
                loop.run_until_complete(asyncio.sleep(0))  # drain callbacks
                dispatcher.on_step({"type": "step", "label": "Running command", "tool": "shell_run", "milestone": True})
                loop.run_until_complete(asyncio.sleep(0))
    mock_rcts.assert_not_called()


def test_telegram_notified_for_telegram_source(loop):
    """Telegram IS notified when source == 'telegram'."""
    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock()
    with patch("step_dispatcher.get_state") as mock_state:
        mock_state.return_value.away = False
        mock_state.return_value.chat_id = 123
        with patch("step_dispatcher.get_bot", return_value=mock_bot):
            with patch("step_dispatcher.asyncio.run_coroutine_threadsafe") as mock_rcts:
                dispatcher = StepDispatcher(command_id="cmd1", source="telegram", loop=loop)
                dispatcher.on_step({"type": "step", "label": "Searching the web", "tool": "web_search", "milestone": True})
    mock_rcts.assert_called_once()


def test_telegram_notified_when_away(loop):
    """Telegram IS notified for hotkey source when away mode is on."""
    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock()
    with patch("step_dispatcher.get_state") as mock_state:
        mock_state.return_value.away = True
        mock_state.return_value.chat_id = 456
        with patch("step_dispatcher.get_bot", return_value=mock_bot):
            with patch("step_dispatcher.asyncio.run_coroutine_threadsafe") as mock_rcts:
                dispatcher = StepDispatcher(command_id="cmd1", source="hotkey", loop=loop)
                dispatcher.on_step({"type": "step", "label": "Running command", "tool": "shell_run", "milestone": True})
    mock_rcts.assert_called_once()


def test_complete_pushes_to_queue(loop):
    """complete() pushes complete event to queue."""
    dispatcher = StepDispatcher(command_id="cmd1", source="hotkey", loop=loop)
    result = {"speak": "done", "display": "done", "steps": []}
    dispatcher.complete(result)
    loop.run_until_complete(asyncio.sleep(0))  # drain callbacks
    assert not dispatcher.queue.empty()
    queued = dispatcher.queue.get_nowait()
    assert queued["type"] == "complete"
    assert queued["speak"] == "done"
