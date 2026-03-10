import asyncio
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
    try:
        allowed = int(cfg_module.load()["telegram"]["allowed_user_id"])
    except (ValueError, TypeError):
        return False
    return update.effective_user.id == allowed


async def _handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _validate(update):
        return
    state = get_state()
    state.chat_id = update.effective_chat.id
    await context.bot.send_chat_action(chat_id=state.chat_id, action="typing")
    for attempt in range(2):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{_SERVER_URL}/command",
                    json={"text": update.message.text, "source": "telegram"},
                    timeout=120.0,
                )
                data = resp.json()
            break
        except httpx.TimeoutException:
            await update.message.reply_text("Jarvis is taking too long — command may still be running.")
            return
        except httpx.RequestError:
            if attempt == 0:
                await asyncio.sleep(2)
                continue
            await update.message.reply_text("Server unavailable — is Jarvis running?")
            return
    ar = data.get("approval_required")
    if ar:
        state.pending_command = update.message.text
        state.pending_tool_use_id = ar.get("tool_use_id")
        action = ar.get("description", "this action")
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
        for attempt in range(2):
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
                        timeout=120.0,
                    )
                    data = resp.json()
                reply = data.get("display") or data.get("speak") or data.get("error") or "Done."
                await update.message.reply_text(reply)
                break
            except httpx.TimeoutException:
                await update.message.reply_text("Jarvis is taking too long — command may still be running.")
                break
            except httpx.RequestError:
                if attempt == 0:
                    await asyncio.sleep(2)
                    continue
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


async def _handle_schedule(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /schedule <command> — forwards to command pipeline."""
    if not await _validate(update):
        return
    state = get_state()
    state.chat_id = update.effective_chat.id
    text = update.message.text.removeprefix("/schedule").strip()
    if not text:
        await update.message.reply_text(
            "Usage: /schedule <what and when>\nExample: /schedule every morning at 9, summarise my calendar"
        )
        return
    await context.bot.send_chat_action(chat_id=state.chat_id, action="typing")
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{_SERVER_URL}/command",
                json={"text": text, "source": "telegram"},
            )
        result = resp.json()
        response_text = result.get("display") or result.get("speak") or result.get("error") or "Done."
    except (httpx.TimeoutException, httpx.RequestError) as e:
        log.error("Schedule command failed: %s", e)
        response_text = "Sorry, couldn't reach the Jarvis server. Try again."
    await update.message.reply_text(response_text)


async def _handle_schedules(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /schedules — list all scheduled tasks."""
    if not await _validate(update):
        return
    state = get_state()
    state.chat_id = update.effective_chat.id
    import scheduler as sched_module
    s = sched_module.get_scheduler()
    if s is None:
        await update.message.reply_text("Scheduler not running.")
        return
    schedules = s.list()
    if not schedules:
        await update.message.reply_text("No scheduled tasks.")
        return
    lines = []
    for i, sched in enumerate(schedules, 1):
        status = "✓" if sched.enabled else "⏸"
        timing = f"cron: {sched.cron}" if sched.cron else f"at: {sched.run_at_iso}"
        lines.append(f"{i}. {status} {sched.label} ({timing}) [id: {sched.id}]")
    await update.message.reply_text("\n".join(lines))


def create_app() -> Application | None:
    conf = cfg_module.load()
    t = conf.get("telegram", {})
    token = t.get("bot_token", "")
    allowed_id = t.get("allowed_user_id", 0)
    try:
        allowed_id_int = int(allowed_id)
    except (ValueError, TypeError):
        return None
    if not token or not allowed_id_int:
        return None
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("away", _handle_away))
    app.add_handler(CommandHandler("back", _handle_back))
    app.add_handler(CommandHandler("approve", _handle_approve))
    app.add_handler(CommandHandler("deny", _handle_deny))
    app.add_handler(CommandHandler("schedule", _handle_schedule))
    app.add_handler(CommandHandler("schedules", _handle_schedules))
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
