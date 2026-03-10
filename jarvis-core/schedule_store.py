import json
import uuid
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("jarvis.schedule_store")

SCHEDULES_PATH = str(Path.home() / ".jarvis" / "schedules.json")


@dataclass
class Schedule:
    id: str
    label: str
    command: str
    schedule_type: str      # "recurring" | "one_time"
    cron: str | None        # "0 9 * * *" for recurring, None for one_time
    run_at_iso: str | None  # ISO datetime string for one_time, None for recurring
    enabled: bool
    created_at: str
    output: str             # "telegram" for now


class ScheduleStore:
    def __init__(self):
        self._path = Path(SCHEDULES_PATH)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._schedules: dict[str, Schedule] = {}
        self._load()

    def _load(self):
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            for item in data.get("schedules", []):
                s = Schedule(**item)
                self._schedules[s.id] = s
        except Exception as e:
            log.error("Failed to load schedules from %s: %s", self._path, e)

    def _save(self):
        data = {"schedules": [asdict(s) for s in self._schedules.values()]}
        try:
            self._path.write_text(json.dumps(data, indent=2))
        except OSError as e:
            log.error("Failed to save schedules to %s: %s", self._path, e)

    def list_all(self) -> list[Schedule]:
        return list(self._schedules.values())

    def get(self, schedule_id: str) -> Schedule | None:
        return self._schedules.get(schedule_id)

    def add(
        self,
        command: str,
        label: str,
        schedule_type: str,
        cron: str | None,
        run_at_iso: str | None,
        output: str = "telegram",
    ) -> Schedule:
        s = Schedule(
            id=uuid.uuid4().hex[:8],
            label=label,
            command=command,
            schedule_type=schedule_type,
            cron=cron,
            run_at_iso=run_at_iso,
            enabled=True,
            created_at=datetime.now(timezone.utc).isoformat(),
            output=output,
        )
        self._schedules[s.id] = s
        self._save()
        return s

    def remove(self, schedule_id: str) -> bool:
        if schedule_id not in self._schedules:
            return False
        del self._schedules[schedule_id]
        self._save()
        return True

    def update(self, schedule_id: str, **kwargs) -> Schedule | None:
        s = self._schedules.get(schedule_id)
        if s is None:
            return None
        for k, v in kwargs.items():
            if not hasattr(s, k):
                raise ValueError(f"Unknown schedule field: {k!r}")
            setattr(s, k, v)
        self._save()
        return s
