import asyncio
import logging
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

from schedule_store import Schedule, ScheduleStore

log = logging.getLogger("jarvis.scheduler")


def _notify_mac(label: str, body: str) -> None:
    """Push a native notification to the Swift app via the alert bus."""
    try:
        import alert_bus
        alert_bus.push(title=f"Jarvis: {label}", body=body[:200])
    except Exception as e:
        log.warning("alert_bus push failed for '%s': %s", label, e)


def send_telegram_result(label: str, response: str, error: str | None, loop=None) -> None:
    """Send scheduled task result to Telegram."""
    from telegram_bot import get_bot
    from telegram_state import get_state
    import asyncio

    bot = get_bot()
    state = get_state()
    if bot is None or state.chat_id is None:
        log.warning("Cannot send scheduled result — bot or chat_id not available")
        return

    if error:
        text = f"Scheduled task '{label}' failed: {error}"
    else:
        text = f"[Scheduled: {label}]\n{response}"

    async def _send():
        await bot.send_message(chat_id=state.chat_id, text=text)

    try:
        if loop is not None and loop.is_running():
            asyncio.run_coroutine_threadsafe(_send(), loop)
        else:
            # Fallback for tests / non-async contexts
            import asyncio as _asyncio
            _asyncio.run(_send())
    except Exception as e:
        log.error("Failed to send scheduled result to Telegram: %s", e)


class Scheduler:
    def __init__(self, store: ScheduleStore, pipeline):
        self.store = store
        self._pipeline = pipeline
        self._apscheduler = BackgroundScheduler()
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self):
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None
        self._apscheduler.start()
        for s in self.store.list_all():
            if s.enabled:
                self._register(s)
        log.info("Scheduler started with %d jobs", len(self.list()))

    def stop(self):
        self._apscheduler.shutdown(wait=False)

    def list(self) -> list[Schedule]:
        return self.store.list_all()

    def create(
        self,
        command: str,
        label: str,
        schedule_type: str,
        cron: str | None,
        run_at_iso: str | None,
    ) -> Schedule:
        s = self.store.add(command, label, schedule_type, cron, run_at_iso)
        self._register(s)
        return s

    def delete(self, schedule_id: str) -> bool:
        job = self._apscheduler.get_job(schedule_id)
        if job:
            job.remove()
        return self.store.remove(schedule_id)

    def pause(self, schedule_id: str) -> Schedule | None:
        job = self._apscheduler.get_job(schedule_id)
        if job:
            job.remove()
        return self.store.update(schedule_id, enabled=False)

    def resume(self, schedule_id: str) -> Schedule | None:
        s = self.store.update(schedule_id, enabled=True)
        if s:
            self._register(s)
        return s

    def run_job(self, schedule_id: str) -> None:
        """Trigger a job immediately (used in tests and for manual runs)."""
        s = self.store.get(schedule_id)
        if s is None:
            log.warning("run_job: schedule %s not found", schedule_id)
            return
        self._execute(schedule_id)

    def _register(self, s: Schedule) -> None:
        try:
            if s.schedule_type == "recurring" and s.cron:
                trigger = CronTrigger.from_crontab(s.cron)
            elif s.schedule_type == "one_time" and s.run_at_iso:
                trigger = DateTrigger(run_date=datetime.fromisoformat(s.run_at_iso))
            else:
                log.warning("Cannot register schedule %s — missing cron/run_at_iso", s.id)
                return
        except (ValueError, KeyError) as e:
            log.error("Invalid trigger for schedule %s: %s", s.id, e)
            return

        self._apscheduler.add_job(
            self._execute,
            trigger=trigger,
            id=s.id,
            args=[s.id],
            replace_existing=True,
        )

    def _execute(self, schedule_id: str) -> None:
        import time
        s = self.store.get(schedule_id)
        if s is None:
            return
        log.info("Running scheduled task '%s': %s", s.label, s.command)
        cmd_log = logging.getLogger("jarvis.commands")
        start = time.time()
        try:
            result = self._pipeline.submit(s.command, source="scheduled")
            duration_ms = int((time.time() - start) * 1000)
            cmd_log.info(
                f"cmd={s.command!r} source='scheduled' label={s.label!r} duration_ms={duration_ms} result={result}"
            )
            # Always fire a macOS notification so the user sees the result when on the Mac,
            # regardless of whether Telegram is connected.
            if not result.get("approval_required"):
                _notify_mac(s.label, result.get("speak") or result.get("display") or "Done.")
            send_telegram_result(s.label, result.get("display") or result.get("speak", ""), result.get("error"), loop=self._loop)
        except Exception as e:
            duration_ms = int((time.time() - start) * 1000)
            cmd_log.error(
                f"cmd={s.command!r} source='scheduled' label={s.label!r} duration_ms={duration_ms} error={e!r}"
            )
            log.error("Scheduled task '%s' failed: %s", s.label, e)
            send_telegram_result(s.label, "", str(e), loop=self._loop)

        if s.schedule_type == "one_time":
            self.store.update(schedule_id, enabled=False)


# Module-level singleton — set by server.py on startup
_scheduler: Scheduler | None = None


def get_scheduler() -> Scheduler | None:
    return _scheduler


def set_scheduler(s: Scheduler) -> None:
    global _scheduler
    _scheduler = s
