"""RecentProjectsStore: the one MRU list shared by MainWindow and Welcome.

The legacy JSON format, newest first, a folder once (``same_file``), at most ``max_entries``
read and written, garbage skipped, and a headless window keeps its list in memory, never the
user's (C5-04).
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import pytest

from app.gui.recent_projects import RecentProject, RecentProjectsStore
from app.settings import MemorySettings
from tests.qt_teardown import delete_widgets


class _FakeSettings:
    """QSettings stand-in (``value`` / ``setValue``)."""

    def __init__(self, raw=None):
        self.data = {} if raw is None else {"recent_projects": raw}

    def value(self, key, default=None):
        return self.data.get(key, default)

    def setValue(self, key, value):  # noqa: N802 - Qt API name
        self.data[key] = value


def test_empty_store_has_no_entries():
    assert RecentProjectsStore(_FakeSettings()).entries() == []


def test_add_writes_the_legacy_json_format(tmp_path):
    settings = _FakeSettings()
    store = RecentProjectsStore(settings)
    store.add(str(tmp_path), 12, now=datetime(2026, 9, 28, 10, 0, 0))

    raw = json.loads(settings.data["recent_projects"])
    assert raw == [
        {
            "name": os.path.basename(str(tmp_path)),
            "path": str(tmp_path),
            "count": 12,
            "ts": "2026-09-28T10:00:00",
        }
    ]


def test_add_dedupes_moves_to_front_and_caps_at_8(tmp_path):
    store = RecentProjectsStore(_FakeSettings())
    paths = [str(tmp_path / f"p{i}") for i in range(10)]
    for p in paths:
        store.add(p, 1)
    store.add(paths[3], 5)
    entries = store.entries()
    assert len(entries) == 8
    assert entries[0].path == paths[3] and entries[0].count == 5
    assert [e.path for e in entries].count(paths[3]) == 1


def test_reads_existing_settings_and_skips_garbage():
    raw = json.dumps([{"path": "D:/a", "name": "a", "count": 3, "ts": "x"}, {"nope": 1}, "str"])
    entries = RecentProjectsStore(_FakeSettings(raw)).entries()
    assert entries == [RecentProject(path="D:/a", name="a", count=3, ts="x")]
    assert RecentProjectsStore(_FakeSettings("{not json")).entries() == []


def test_remove_and_clear(tmp_path):
    store = RecentProjectsStore(_FakeSettings())
    store.add(str(tmp_path / "a"), 1)
    store.add(str(tmp_path / "b"), 1)
    store.remove(str(tmp_path / "a"))
    assert [e.name for e in store.entries()] == ["b"]
    store.clear()
    assert store.entries() == []


def test_exists_reports_missing_folders(tmp_path):
    assert RecentProject(str(tmp_path), "x").exists() is True
    assert RecentProject(str(tmp_path / "gone"), "x").exists() is False


def test_default_store_uses_the_app_settings_file(monkeypatch, tmp_path):
    """Without settings the store is the app's (``app.settings.app_settings``): the user's
    settings file on a screen, memory when headless (C5-04). The user's settings are never
    opened here."""
    import app.settings as factory

    assert isinstance(RecentProjectsStore().settings, MemorySettings)  # the suite is headless
    made = []
    monkeypatch.setenv("SPROUTS_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(factory, "headless", lambda: False)
    monkeypatch.setattr(factory, "ini_settings", lambda path: made.append(path) or "the user's")
    assert RecentProjectsStore().settings == "the user's"
    assert made == [os.path.join(str(tmp_path), "settings.ini")]


def test_memory_settings_keep_the_list_for_this_object_only(tmp_path):
    """What a headless window without settings records (C5-04): never the user's list."""
    settings = MemorySettings()
    store = RecentProjectsStore(settings)
    assert store.settings is settings and store.entries() == []
    store.add(str(tmp_path / "a"), 3)
    store.add(str(tmp_path / "b"), 4)
    assert [e.name for e in store.entries()] == ["b", "a"]
    assert RecentProjectsStore(settings).entries() == store.entries()  # the same object
    assert RecentProjectsStore(MemorySettings()).entries() == []  # nothing shared or saved
    store.remove(str(tmp_path / "b"))
    assert [e.name for e in store.entries()] == ["a"]
    store.clear()
    assert store.entries() == [] and settings.value("recent_projects") == "[]"
    assert settings.value("missing", "fallback") == "fallback"


class _Settings:
    def __init__(self, entries=()):
        self.data = {"recent_projects": json.dumps(list(entries))}

    def value(self, key, default=None):
        return self.data.get(key, default)

    def setValue(self, key, value):  # noqa: N802
        self.data[key] = value


def _entries(n: int, prefix: str = "p") -> list[dict]:
    return [
        {"name": f"{prefix}{i}", "path": f"D:/x/{prefix}{i}", "count": 5, "ts": "2026-09-20T10:00"}
        for i in range(n)
    ]


def test_reading_keeps_the_newest_max_entries():
    store = RecentProjectsStore(_Settings(_entries(500)))
    entries = store.entries()
    assert [e.name for e in entries] == [f"p{i}" for i in range(store.max_entries)]


def test_reading_shows_a_path_stored_twice_once():
    stored = _entries(3)
    stored.insert(2, dict(stored[0], path="D:/x/p0/", name="again"))  # same folder, other spelling
    names = [e.name for e in RecentProjectsStore(_Settings(stored)).entries()]
    assert names == ["p0", "p1", "p2"]


def test_headless_window_keeps_recent_projects_in_memory(app, tmp_path):
    from app.gui.main_window import MainWindow

    window = MainWindow()
    try:
        # Checked first: with the old code this fails before anything is recorded, so the
        # test never writes the user's list.
        assert isinstance(window.recent_projects.settings, MemorySettings)
        window._last_loaded_dir = str(tmp_path)
        window._record_recent_project(2)
        assert [entry.path for entry in window.recent_projects.entries()] == [str(tmp_path)]
    finally:
        delete_widgets([window])


@pytest.mark.parametrize("raw", ['{"path": "D:/a"}', "5", '"D:/a"', "null"])
def test_a_stored_value_that_is_not_a_list_gives_no_entries(raw):
    assert RecentProjectsStore(_FakeSettings(raw)).entries() == []


def test_a_list_written_to_the_settings_file_is_read_back_by_the_next_start(tmp_path):
    from PyQt6.QtCore import QSettings

    path = str(tmp_path / "settings.ini")
    store = RecentProjectsStore(QSettings(path, QSettings.Format.IniFormat))
    store.add(str(tmp_path / "a"), 3, now=datetime(2026, 9, 28, 10, 0, 0))
    store.add(str(tmp_path / "b,c"), 46178, now=datetime(2026, 9, 29, 10, 0, 0))
    store.settings.sync()
    again = RecentProjectsStore(QSettings(path, QSettings.Format.IniFormat))
    assert again.entries() == store.entries()
    assert [(e.name, e.count) for e in again.entries()] == [("b,c", 46178), ("a", 3)]
