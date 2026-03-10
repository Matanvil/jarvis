import json
import pytest
from unittest.mock import patch
from schedule_store import ScheduleStore, Schedule


@pytest.fixture
def tmp_store(tmp_path):
    store_path = tmp_path / "schedules.json"
    with patch("schedule_store.SCHEDULES_PATH", str(store_path)):
        yield ScheduleStore()


def test_empty_store_returns_empty_list(tmp_store):
    assert tmp_store.list_all() == []


def test_add_recurring_schedule(tmp_store):
    s = tmp_store.add(
        command="summarise my calendar",
        label="morning calendar summary",
        schedule_type="recurring",
        cron="0 9 * * *",
        run_at_iso=None,
    )
    assert s.id
    assert s.label == "morning calendar summary"
    assert s.cron == "0 9 * * *"
    assert s.enabled is True
    assert s.output == "telegram"


def test_add_onetime_schedule(tmp_store):
    s = tmp_store.add(
        command="check the weather",
        label="one-time weather check",
        schedule_type="one_time",
        cron=None,
        run_at_iso="2026-03-11T09:00:00",
    )
    assert s.schedule_type == "one_time"
    assert s.run_at_iso == "2026-03-11T09:00:00"
    assert s.cron is None


def test_persists_to_disk(tmp_path):
    store_path = tmp_path / "schedules.json"
    with patch("schedule_store.SCHEDULES_PATH", str(store_path)):
        s1 = ScheduleStore()
        s1.add("cmd", "label", "recurring", "0 9 * * *", None)
        s2 = ScheduleStore()  # reload from disk
        assert len(s2.list_all()) == 1
        assert s2.list_all()[0].label == "label"


def test_remove_schedule(tmp_store):
    s = tmp_store.add("cmd", "label", "recurring", "0 9 * * *", None)
    assert tmp_store.remove(s.id) is True
    assert tmp_store.list_all() == []


def test_remove_nonexistent_returns_false(tmp_store):
    assert tmp_store.remove("nonexistent") is False


def test_update_enabled_false(tmp_store):
    s = tmp_store.add("cmd", "label", "recurring", "0 9 * * *", None)
    updated = tmp_store.update(s.id, enabled=False)
    assert updated.enabled is False
    assert tmp_store.get(s.id).enabled is False


def test_get_by_id(tmp_store):
    s = tmp_store.add("cmd", "label", "recurring", "0 9 * * *", None)
    assert tmp_store.get(s.id).id == s.id


def test_get_nonexistent_returns_none(tmp_store):
    assert tmp_store.get("bad") is None
