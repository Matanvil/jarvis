import logging
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

from schedule_store import Schedule, ScheduleStore

log = logging.getLogger("jarvis.scheduler")


def send_telegram_result(label: str, response: str, error: str | None) -> None:
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
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(_send())
        else:
            loop.run_until_complete(_send())
    except Exception as e:
        log.error("Failed to send scheduled result to Telegram: %s", e)


class Scheduler:
    def __init__(self, store: ScheduleStore, pipeline):
        self.store = store
        self._pipeline = pipeline
        self._apscheduler = BackgroundScheduler()

    def start(self):
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
        if s.schedule_type == "recurring" and s.cron:
            parts = s.cron.split()
            trigger = CronTrigger(
                minute=parts[0],
                hour=parts[1],
                day=parts[2],
                month=parts[3],
                day_of_week=parts[4],
            )
        elif s.schedule_type == "one_time" and s.run_at_iso:
            trigger = DateTrigger(run_date=datetime.fromisoformat(s.run_at_iso))
        else:
            log.warning("Cannot register schedule %s — missing cron/run_at_iso", s.id)
            return

        self._apscheduler.add_job(
            self._execute,
            trigger=trigger,
            id=s.id,
            args=[s.id],
            replace_existing=True,
        )

    def _execute(self, schedule_id: str) -> None:
        s = self.store.get(schedule_id)
        if s is None:
            return
        log.info("Running scheduled task '%s': %s", s.label, s.command)
        try:
            result = self._pipeline.submit(s.command, source="scheduled")
            send_telegram_result(s.label, result.get("response", ""), result.get("error"))
        except Exception as e:
            log.error("Scheduled task '%s' failed: %s", s.label, e)
            send_telegram_result(s.label, "", str(e))

        if s.schedule_type == "one_time":
            self.store.update(schedule_id, enabled=False)


# Module-level singleton — set by server.py on startup
_scheduler: Scheduler | None = None


def get_scheduler() -> Scheduler | None:
    return _scheduler


def set_scheduler(s: Scheduler) -> None:
    global _scheduler
    _scheduler = s
