import logging
from telegram_state import get_state

log = logging.getLogger("jarvis.notifications")


def get_bot():
    """Lazy import to avoid circular imports at module load time."""
    try:
        from telegram_bot import get_bot as _get_bot
        return _get_bot()
    except ImportError:
        return None


async def notify(message: str) -> None:
    """Send message to Telegram if away mode is on and chat_id is known."""
    state = get_state()
    if not state.away or state.chat_id is None:
        return
    bot = get_bot()
    if bot is None:
        return
    try:
        await bot.send_message(chat_id=state.chat_id, text=message)
    except Exception as e:
        log.warning("Telegram notify failed: %s", e)
