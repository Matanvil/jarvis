import asyncio
import logging
import os
import tempfile
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
import transcriber

log = logging.getLogger("jarvis.telegram")
err_log = logging.getLogger("jarvis.errors")

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


async def _handle_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Global error handler — logs full traceback and notifies the user."""
    err_log.error("Unhandled exception in Telegram handler", exc_info=context.error)
    # Best-effort reply so the user is never left in silence.
    msg = "Something went wrong — try again."
    try:
        if isinstance(update, Update) and update.effective_message:
            await update.effective_message.reply_text(msg)
        else:
            state = get_state()
            if state.chat_id and _app:
                await _app.bot.send_message(chat_id=state.chat_id, text=msg)
    except Exception:
        pass  # If even the fallback fails, at least it's logged above.


async def _run_command(update: Update, text: str) -> None:
    """POST text to /command and handle the response.

    Used by both _handle_message (text) and _handle_voice (voice transcription).
    state.pending_command is always set to the `text` parameter — never to
    update.message.text — so /approve re-submits the right command in both cases.
    """
    state = get_state()
    for attempt in range(2):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{_SERVER_URL}/command",
                    json={"text": text, "source": "telegram"},
                    timeout=120.0,
                )
                try:
                    data = resp.json()
                except ValueError:
                    err_log.error(
                        "Non-JSON response from server: status=%d body=%r",
                        resp.status_code, resp.text[:200],
                    )
                    await update.message.reply_text("Server returned an unexpected response — try again.")
                    return
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
    if data.get("busy"):
        await update.message.reply_text(
            "I'm busy with another command right now. Please try again in a moment."
        )
        return
    ar = data.get("approval_required")
    if ar:
        state.pending_command = text  # NOTE: `text` param, not update.message.text
        state.pending_tool_use_id = ar.get("tool_use_id")
        state.pending_category = ar.get("category", "")
        action = ar.get("description", "this action")
        await update.message.reply_text(
            f"Approval required: {action}\nReply /approve or /deny"
        )
    else:
        reply = data.get("display") or data.get("speak") or data.get("error") or "Done."
        await update.message.reply_text(reply)


async def _handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _validate(update):
        return
    state = get_state()
    state.chat_id = update.effective_chat.id
    await update.message.reply_text("⏳ On it...")
    await _run_command(update, update.message.text)


async def _handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _validate(update):
        return
    state = get_state()
    state.chat_id = update.effective_chat.id

    await update.message.reply_text("⏳ Transcribing...")

    voice = update.message.voice
    tg_file = await context.bot.get_file(voice.file_id)
    audio_bytes = await tg_file.download_as_bytearray()

    tmp = tempfile.NamedTemporaryFile(suffix=".ogg", delete=False)
    try:
        tmp.write(audio_bytes)
        tmp.close()
        text = transcriber.transcribe(tmp.name)
    except RuntimeError as e:
        # RuntimeError means Whisper is unavailable (model not loaded).
        # Split from bare Exception so the user sees "not available" vs "failed".
        # Note: spec pseudocode uses a single bare except; this plan intentionally
        # differentiates the two error types for a clearer user-facing message.
        await update.message.reply_text(str(e))
        return
    except Exception:
        await update.message.reply_text("Failed to transcribe — try again.")
        return
    finally:
        os.unlink(tmp.name)

    if not text:
        await update.message.reply_text("Couldn't make that out — try again.")
        return

    await update.message.reply_text(f'🎙 "{text}"\n⏳ On it...')
    await _run_command(update, text)


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
    await update.message.reply_text("⏳ On it...")
    try:
        for attempt in range(2):
            try:
                async with httpx.AsyncClient() as client:
                    await client.post(
                        f"{_SERVER_URL}/approve",
                        json={
                            "tool_use_id": tool_use_id,
                            "approved": True,
                            "trust_session": True,
                            "category": state.pending_category or "",
                        },
                    )
                    resp = await client.post(
                        f"{_SERVER_URL}/command",
                        json={"text": pending_cmd, "source": "telegram"},
                        timeout=120.0,
                    )
                    try:
                        data = resp.json()
                    except ValueError:
                        err_log.error("Non-JSON response from server on approve: status=%d body=%r", resp.status_code, resp.text[:200])
                        await update.message.reply_text("Server returned an unexpected response — try again.")
                        return
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
    await update.message.reply_text("⏳ On it...")
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{_SERVER_URL}/command",
                json={"text": text, "source": "telegram"},
            )
        try:
            result = resp.json()
        except ValueError:
            err_log.error("Non-JSON response from server on /schedule: status=%d body=%r", resp.status_code, resp.text[:200])
            await update.message.reply_text("Server returned an unexpected response — try again.")
            return
    except (httpx.TimeoutException, httpx.RequestError) as e:
        log.error("Schedule command failed: %s", e)
        await update.message.reply_text("Sorry, couldn't reach the Jarvis server. Try again.")
        return
    ar = result.get("approval_required")
    if ar:
        state.pending_command = text
        state.pending_tool_use_id = ar.get("tool_use_id")
        state.pending_category = ar.get("category", "")
        action = ar.get("description", "this scheduled task")
        await update.message.reply_text(f"Approval required: {action}\nReply /approve or /deny")
    else:
        response_text = result.get("display") or result.get("speak") or result.get("error") or "Done."
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
    app.add_error_handler(_handle_error)
    app.add_handler(CommandHandler("away", _handle_away))
    app.add_handler(CommandHandler("back", _handle_back))
    app.add_handler(CommandHandler("approve", _handle_approve))
    app.add_handler(CommandHandler("deny", _handle_deny))
    app.add_handler(CommandHandler("schedule", _handle_schedule))
    app.add_handler(CommandHandler("schedules", _handle_schedules))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_message))
    app.add_handler(MessageHandler(filters.VOICE, _handle_voice))
    return app


async def start_bot() -> None:
    global _app
    try:
        transcriber.load()
    except Exception:
        log.warning("openai-whisper not available — voice transcription disabled")
    _app = create_app()
    if _app is None:
        log.info("Telegram not configured — skipping")
        return
    await _app.initialize()
    await _app.start()
    await _app.updater.start_polling(drop_pending_updates=False)
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
