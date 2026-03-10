import pytest
from unittest.mock import MagicMock, patch
from schedule_store import ScheduleStore


@pytest.fixture
def mock_store(tmp_path):
    with patch("schedule_store.SCHEDULES_PATH", str(tmp_path / "s.json")):
        yield ScheduleStore()


@pytest.fixture
def mock_pipeline():
    p = MagicMock()
    p.submit.return_value = {"response": "done", "error": None}
    return p


@pytest.fixture
def sched(mock_store, mock_pipeline):
    from scheduler import Scheduler
    s = Scheduler(store=mock_store, pipeline=mock_pipeline)
    s.start()
    yield s
    s.stop()


def test_create_recurring_schedule(sched):
    s = sched.create(
        command="summarise my calendar",
        label="morning summary",
        schedule_type="recurring",
        cron="0 9 * * *",
        run_at_iso=None,
    )
    assert s.id
    assert s.schedule_type == "recurring"
    assert len(sched.list()) == 1


def test_create_onetime_schedule(sched):
    s = sched.create(
        command="check weather",
        label="weather check",
        schedule_type="one_time",
        cron=None,
        run_at_iso="2030-03-11T09:00:00",
    )
    assert s.schedule_type == "one_time"


def test_delete_schedule(sched):
    s = sched.create("cmd", "label", "recurring", "0 9 * * *", None)
    assert sched.delete(s.id) is True
    assert sched.list() == []


def test_delete_nonexistent_returns_false(sched):
    assert sched.delete("bad") is False


def test_pause_schedule(sched):
    s = sched.create("cmd", "label", "recurring", "0 9 * * *", None)
    updated = sched.pause(s.id)
    assert updated.enabled is False


def test_resume_schedule(sched):
    s = sched.create("cmd", "label", "recurring", "0 9 * * *", None)
    sched.pause(s.id)
    updated = sched.resume(s.id)
    assert updated.enabled is True


def test_run_job_calls_pipeline(sched, mock_pipeline):
    s = sched.create("summarise my calendar", "morning summary", "recurring", "0 9 * * *", None)
    sched.run_job(s.id)
    mock_pipeline.submit.assert_called_once_with(
        "summarise my calendar", source="scheduled"
    )


def test_run_job_sends_to_telegram(sched, mock_pipeline):
    with patch("scheduler.send_telegram_result") as mock_send:
        s = sched.create("cmd", "label", "recurring", "0 9 * * *", None)
        sched.run_job(s.id)
        mock_send.assert_called_once()


def test_run_job_disables_onetime_after_firing(sched, mock_pipeline):
    with patch("scheduler.send_telegram_result"):
        s = sched.create("cmd", "label", "one_time", None, "2030-03-11T09:00:00")
        sched.run_job(s.id)
        assert sched.store.get(s.id).enabled is False


def test_start_reregisters_enabled_schedules(mock_store, mock_pipeline):
    mock_store.add("cmd", "label", "recurring", "0 9 * * *", None)
    from scheduler import Scheduler
    s = Scheduler(store=mock_store, pipeline=mock_pipeline)
    s.start()
    assert len(s.list()) == 1
    s.stop()


def test_start_skips_disabled_schedules(mock_store, mock_pipeline):
    added = mock_store.add("cmd", "label", "recurring", "0 9 * * *", None)
    mock_store.update(added.id, enabled=False)
    from scheduler import Scheduler
    s = Scheduler(store=mock_store, pipeline=mock_pipeline)
    s.start()
    assert s._apscheduler.get_job(added.id) is None
    s.stop()
