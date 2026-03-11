import asyncio
import logging
from telegram_bot import get_bot
from telegram_state import get_state

log = logging.getLogger("jarvis.steps")


class StepDispatcher:
    """Routes step events to the SSE queue and optionally to Telegram."""

    def __init__(self, command_id: str, source: str, loop: asyncio.AbstractEventLoop):
        self.command_id = command_id
        self.source = source
        self.loop = loop
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=100)

    def _enqueue(self, event: dict) -> None:
        """Thread-safe enqueue: schedules put_nowait via call_soon_threadsafe."""
        def _put():
            try:
                self.queue.put_nowait(event)
            except asyncio.QueueFull:
                log.warning("Step queue full for command %s — dropping event", self.command_id)

        self.loop.call_soon_threadsafe(_put)

    def on_step(self, event: dict) -> None:
        """Called from background thread at each milestone step."""
        # Push to SSE queue only for non-telegram sources (Telegram path has no SSE consumer)
        if self.source != "telegram":
            self._enqueue(event)

        # Conditionally notify Telegram
        state = get_state()
        notify_telegram = (self.source == "telegram") or state.away
        bot = get_bot()
        chat_id = state.chat_id
        if notify_telegram and bot and chat_id:
                label = event.get("label", "Working…")

                async def _send():
                    try:
                        await bot.send_message(chat_id=chat_id, text=f"⏳ {label}…")
                    except Exception as e:
                        log.warning("Failed to send step to Telegram: %s", e)

                try:
                    asyncio.run_coroutine_threadsafe(_send(), self.loop)
                except Exception as e:
                    log.warning("Could not schedule Telegram step send: %s", e)

    def complete(self, result: dict) -> None:
        """Push the final result as a complete event to the SSE queue."""
        event = {"type": "complete", **result}
        self._enqueue(event)

    def error(self, message: str) -> None:
        """Push an error event to the SSE queue."""
        self._enqueue({"type": "error", "message": message})
