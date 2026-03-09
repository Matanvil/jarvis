import logging
import httpx
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
import config as cfg_module
from telegram_state import get_state

log = logging.getLogger("jarvis.telegram")

_app: Application | None = None
_SERVER_URL = "http://127.0.0.1:8765"


def get_bot():
    return _app.bot if _app else None


async def _validate(update: Update) -> bool:
    allowed = cfg_module.load()["telegram"]["allowed_user_id"]
    return update.effective_user.id == int(allowed)


async def _handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _validate(update):
        return
    state = get_state()
    state.chat_id = update.effective_chat.id
    await context.bot.send_chat_action(chat_id=state.chat_id, action="typing")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{_SERVER_URL}/command",
                json={"text": update.message.text, "source": "telegram"},
                timeout=60.0,
            )
            data = resp.json()
    except (httpx.RequestError, httpx.TimeoutException):
        await update.message.reply_text("Server unavailable — is Jarvis running?")
        return
    if data.get("approval_required"):
        state.pending_command = update.message.text
        state.pending_tool_use_id = data.get("tool_use_id")
        action = data.get("action", "this action")
        await update.message.reply_text(
            f"Approval required: {action}\nReply /approve or /deny"
        )
    else:
        reply = data.get("display") or data.get("speak") or data.get("error") or "Done."
        await update.message.reply_text(reply)


async def _handle_away(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _validate(update):
        return
    state = get_state()
    state.chat_id = update.effective_chat.id
    state.away = True
    await update.message.reply_text("Away mode on — I'll notify you here.")


async def _handle_back(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _validate(update):
        return
    state = get_state()
    state.chat_id = update.effective_chat.id
    state.away = False
    await update.message.reply_text("Away mode off — see you at the Mac.")


async def _handle_approve(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _validate(update):
        return
    state = get_state()
    if not state.pending_tool_use_id:
        await update.message.reply_text("Nothing pending approval.")
        return
    tool_use_id = state.pending_tool_use_id
    pending_cmd = state.pending_command
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{_SERVER_URL}/approve",
                json={
                    "tool_use_id": tool_use_id,
                    "approved": True,
                    "trust_session": False,
                    "category": "",
                },
            )
            resp = await client.post(
                f"{_SERVER_URL}/command",
                json={"text": pending_cmd, "source": "telegram"},
                timeout=60.0,
            )
            data = resp.json()
        reply = data.get("display") or data.get("speak") or data.get("error") or "Done."
        await update.message.reply_text(reply)
    except (httpx.RequestError, httpx.TimeoutException):
        await update.message.reply_text("Server unavailable — is Jarvis running?")
    finally:
        state.pending_command = None
        state.pending_tool_use_id = None


async def _handle_deny(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _validate(update):
        return
    state = get_state()
    if not state.pending_tool_use_id:
        await update.message.reply_text("Nothing pending approval.")
        return
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{_SERVER_URL}/approve",
                json={
                    "tool_use_id": state.pending_tool_use_id,
                    "approved": False,
                    "trust_session": False,
                    "category": "",
                },
            )
    except (httpx.RequestError, httpx.TimeoutException):
        await update.message.reply_text("Server unavailable — is Jarvis running?")
        return
    finally:
        state.pending_command = None
        state.pending_tool_use_id = None
    await update.message.reply_text("Action denied.")


def create_app() -> Application | None:
    conf = cfg_module.load()
    t = conf.get("telegram", {})
    token = t.get("bot_token", "")
    allowed_id = t.get("allowed_user_id", 0)
    if not token or not int(allowed_id):
        return None
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("away", _handle_away))
    app.add_handler(CommandHandler("back", _handle_back))
    app.add_handler(CommandHandler("approve", _handle_approve))
    app.add_handler(CommandHandler("deny", _handle_deny))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_message))
    return app


async def start_bot() -> None:
    global _app
    _app = create_app()
    if _app is None:
        log.info("Telegram not configured — skipping")
        return
    await _app.initialize()
    await _app.start()
    await _app.updater.start_polling(drop_pending_updates=True)
    log.info("Telegram bot started (polling)")


async def stop_bot() -> None:
    global _app
    if _app is None:
        return
    await _app.updater.stop()
    await _app.stop()
    await _app.shutdown()
    _app = None
    log.info("Telegram bot stopped")
