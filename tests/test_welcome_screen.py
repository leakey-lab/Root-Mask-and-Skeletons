"""The Welcome screen: a dropped folder opens, recent rows open / dim / go, the keyboard keeps
its place, the stage line and the footer say what is true, and a malformed recent list never
stops SPROUTS from starting.

Folders that no longer exist are checked off the GUI thread (network paths can block) and
dimmed "Folder not found". Removing a row from the keyboard moves focus to the × of the row
that took its place (Browse when none is left). A timestamp with a UTC offset or a NaN count
in the stored list used to raise while the window was built (found 2026-09-30).
"""

from __future__ import annotations

import json
import math
import os
from datetime import UTC, datetime, timedelta

import pytest
from PyQt6.QtCore import QEvent, QMimeData, QPoint, QPointF, QSettings, Qt, QUrl
from PyQt6.QtGui import QDropEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QLabel, QScrollArea

from app.gui.recent_projects import SETTINGS_KEY, RecentProjectsStore
from app.gui.welcome_screen import WelcomeWidget
from tests.qt_teardown import delete_widgets
from tests.qt_wait import wait_until
from tests.window_support import build_window


class _FakeSettings:
    def __init__(self, entries=()):
        self.data = {"recent_projects": json.dumps(list(entries))}

    def value(self, key, default=None):
        return self.data.get(key, default)

    def setValue(self, key, value):  # noqa: N802
        self.data[key] = value


def _store(paths):
    entries = [
        {"name": os.path.basename(p), "path": p, "count": 10, "ts": "2026-09-01T10:00:00"}
        for p in paths
    ]
    return RecentProjectsStore(_FakeSettings(entries))


class _Spy:
    def __init__(self):
        self.started = 0
        self.opened: list[str] = []

    def start(self):
        self.started += 1

    def open(self, path):
        self.opened.append(path)


def _welcome(spy, store=None, **kw):
    return WelcomeWidget(
        spy.start,
        spy.open,
        recent_store=store or _store([]),
        probe_async=False,
        **kw,
    )


def _drop(widget, urls):
    mime = QMimeData()
    mime.setUrls(urls)
    event = QDropEvent(
        QPointF(QPoint(5, 5)),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    widget.dropEvent(event)
    return event


def test_dropping_a_folder_opens_that_folder(app, tmp_path):
    """UI-13 / PG-11: the drop used to open the Browse dialog and ignore the folder."""
    spy = _Spy()
    w = _welcome(spy)
    _drop(w.dropzone, [QUrl.fromLocalFile(str(tmp_path))])
    assert [os.path.normcase(p) for p in spy.opened] == [os.path.normcase(str(tmp_path))]
    assert spy.started == 0


def test_dropping_a_file_opens_its_folder(app, tmp_path):
    f = tmp_path / "a.jpg"
    f.write_bytes(b"x")
    spy = _Spy()
    _drop(_welcome(spy).dropzone, [QUrl.fromLocalFile(str(f))])
    assert [os.path.normcase(p) for p in spy.opened] == [os.path.normcase(str(tmp_path))]


def test_dropping_a_remote_url_is_ignored(app):
    spy = _Spy()
    _drop(_welcome(spy).dropzone, [QUrl("https://example.org/x")])
    assert spy.opened == [] and spy.started == 0


def test_eight_recents_do_not_raise_the_minimum_height(app, tmp_path):
    """UXS-14: 8 rows pushed the Welcome minimum height to 823 px."""
    spy = _Spy()
    w = _welcome(spy, _store([str(tmp_path / f"p{i}") for i in range(8)]))
    assert w.findChild(QScrollArea, "recentScroll") is not None
    assert w.minimumSizeHint().height() <= 560


def test_recent_rows_are_accessible_and_open(app, tmp_path):
    spy = _Spy()
    w = _welcome(spy, _store([str(tmp_path)]))
    rows = w.recent_rows()
    assert len(rows) == 1
    assert rows[0].open_button.accessibleName()
    assert str(tmp_path) in rows[0].open_button.toolTip()
    rows[0].open_button.click()
    assert spy.opened == [str(tmp_path)]


def test_missing_recent_is_flagged_and_removable(app, tmp_path):
    gone = str(tmp_path / "gone")
    store = _store([gone, str(tmp_path)])
    w = _welcome(_Spy(), store)
    missing, present = w.recent_rows()
    assert missing.is_missing() is True and not missing.open_button.isEnabled()
    assert present.is_missing() is False and present.open_button.isEnabled()
    missing.remove_button.click()
    assert [e.path for e in store.entries()] == [str(tmp_path)]
    assert len(w.recent_rows()) == 1


def test_existence_probe_runs_off_the_gui_thread(app, tmp_path):
    import threading
    import time

    seen_threads = []

    def exists(path):
        seen_threads.append(threading.current_thread())
        return False

    w = WelcomeWidget(
        lambda: None,
        lambda p: None,
        recent_store=_store([str(tmp_path)]),
        path_exists=exists,
    )
    deadline = time.monotonic() + 5
    while not w.recent_rows()[0].is_missing() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert w.recent_rows()[0].is_missing() is True
    assert seen_threads and seen_threads[0] is not threading.main_thread()


def test_the_stage_line_names_the_top_bars_stages_in_order(app, tmp_path):
    """The stage line is plain text in the top bar's order (W15): read from the stages, never
    a list of its own."""
    from PyQt6.QtWidgets import QLabel

    from tests.window_support import build_window, quiet_window

    text = _welcome(_Spy()).findChild(QLabel, "stageLine").text()
    assert text == "Library  ·  Generate  ·  Correct  ·  Measure  ·  Visualize"
    window = build_window(tmp_path / "s.ini", show=False)
    try:
        assert text == "  ·  ".join(b.text() for b in window.ribbon_buttons.values())
        assert window.welcome.findChild(QLabel, "stageLine").text() == text
    finally:
        delete_widgets([window], prepare=quiet_window)


def test_footer_shows_the_real_version_and_no_hardcoded_gpu(app):
    from app import __version__

    w = _welcome(_Spy())
    footer = w.findChild(QLabel, "welcomeFooter").text()
    assert f"v{__version__}" in footer
    assert "v2.0" not in footer and "CUDA 12.8" not in footer


def test_deleting_the_welcome_while_probing_is_safe(app, tmp_path, monkeypatch):
    """The probe outlives a deleted Welcome screen: it ends without an exception, and its
    late answer reaches nothing (a thread's exception would not fail the test by itself)."""
    import threading

    from PyQt6 import sip

    raised = []
    monkeypatch.setattr(threading, "excepthook", raised.append)
    release = threading.Event()

    def slow_exists(path):
        release.wait(5)
        return True

    before = set(threading.enumerate())
    w = WelcomeWidget(
        lambda: None, lambda p: None, recent_store=_store([str(tmp_path)]), path_exists=slow_exists
    )
    (probe,) = [t for t in set(threading.enumerate()) - before if t.name == "recent-projects-probe"]
    sip.delete(w)
    release.set()
    assert wait_until(lambda: not probe.is_alive())
    app.processEvents()  # the queued answer, to a deleted widget
    assert raised == []


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


@pytest.fixture
def welcome(app):
    made = []

    def make(entries):
        widget = WelcomeWidget(
            lambda: None,
            lambda path: None,
            recent_store=RecentProjectsStore(_Settings(entries)),
            probe_async=False,
            on_about=lambda: None,
        )
        made.append(widget)
        widget.resize(1280, 800)
        widget.show()
        widget.activateWindow()
        QApplication.processEvents()
        return widget

    yield make
    delete_widgets(made)


def _remove_with_keyboard(widget: WelcomeWidget, index: int) -> None:
    button = widget.recent_rows()[index].remove_button
    button.setFocus(Qt.FocusReason.TabFocusReason)
    QApplication.processEvents()
    assert QApplication.focusWidget() is button
    QTest.keyClick(button, Qt.Key.Key_Space)
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QApplication.processEvents()


def test_focus_moves_to_the_row_that_took_the_removed_rows_place(welcome):
    widget = welcome(_entries(3))
    _remove_with_keyboard(widget, 1)
    rows = widget.recent_rows()
    assert [r.entry.name for r in rows] == ["p0", "p2"]
    assert QApplication.focusWidget() is rows[1].remove_button


def test_removing_the_last_row_focuses_the_new_last_row(welcome):
    widget = welcome(_entries(3))
    _remove_with_keyboard(widget, 2)
    rows = widget.recent_rows()
    assert QApplication.focusWidget() is rows[-1].remove_button


def test_removing_the_only_row_focuses_browse(welcome):
    widget = welcome(_entries(1))
    _remove_with_keyboard(widget, 0)
    assert widget.recent_rows() == []
    assert QApplication.focusWidget() is widget.browse_button


def test_a_mouse_removal_does_not_steal_focus(welcome):
    widget = welcome(_entries(3))
    widget.browse_button.setFocus(Qt.FocusReason.OtherFocusReason)
    QApplication.processEvents()
    widget.recent_rows()[1].remove_button.click()  # a click: the × never had focus
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QApplication.processEvents()
    assert QApplication.focusWidget() is widget.browse_button


def test_a_huge_stored_list_builds_at_most_max_entries_rows(welcome):
    widget = welcome(_entries(500))
    assert len(widget.recent_rows()) == RecentProjectsStore(_Settings()).max_entries


class Settings(dict):
    def value(self, key, default=None):
        return self.get(key, default)

    def setValue(self, key, value):  # noqa: N802 - Qt API name
        self[key] = value


def _raw_store(raw: str) -> RecentProjectsStore:
    return RecentProjectsStore(Settings({SETTINGS_KEY: raw}))


def _plain_welcome(store: RecentProjectsStore) -> WelcomeWidget:
    return WelcomeWidget(
        on_get_started=lambda: None,
        recent_store=store,
        probe_async=False,
        path_exists=lambda _path: True,
    )


def _info(widget: WelcomeWidget) -> list[str]:
    return [row._info_label.text() for row in widget.recent_rows()]


def test_a_timestamp_with_an_offset_shows_a_relative_date(app):
    stamp = (datetime.now(UTC) - timedelta(hours=3)).isoformat()
    raw = json.dumps([{"path": "C:/data", "name": "data", "count": 3, "ts": stamp}])
    widget = _plain_welcome(_raw_store(raw))
    assert _info(widget) == ["3 images  ·  3h ago"]
    widget.deleteLater()


@pytest.mark.parametrize("ts", ["0001-01-01T00:00:00+05:00", "9999-12-31T23:59:59-05:00", "junk"])
def test_extreme_or_unreadable_timestamps_are_left_out(app, ts):
    raw = json.dumps([{"path": "C:/data", "count": 2, "ts": ts}])
    widget = _plain_welcome(_raw_store(raw))
    assert len(widget.recent_rows()) == 1
    assert _info(widget)[0].startswith("2 images")
    widget.deleteLater()


@pytest.mark.parametrize("count", ["NaN", "Infinity", "-Infinity"])
def test_a_non_finite_count_is_dropped_not_raised(app, count):
    raw = f'[{{"path": "C:/data", "count": {count}}}]'
    entries = _raw_store(raw).entries()
    assert [(e.path, e.count) for e in entries] == [("C:/data", None)]
    widget = _plain_welcome(_raw_store(raw))
    assert _info(widget) == [""]
    widget.deleteLater()


def test_rows_count_images_like_the_library(app):
    raw = json.dumps(
        [
            {"path": "C:/one", "count": 1, "ts": ""},
            {"path": "C:/many", "count": 46178, "ts": ""},
            {"path": "C:/half", "count": 2.0, "ts": ""},
        ]
    )
    widget = _plain_welcome(_raw_store(raw))
    assert _info(widget) == ["1 image", "46,178 images", "2 images"]
    widget.deleteLater()


def test_the_window_starts_with_such_a_list(app, tmp_path):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    settings.setValue(
        SETTINGS_KEY,
        '[{"path": "C:/a", "count": NaN, "ts": "2026-09-30T10:00:00+00:00"},'
        ' {"path": "C:/b", "count": 5, "ts": "2026-09-30T10:00:00+02:00"}]',
    )
    window = build_window(settings=settings, size=None, show=False)
    try:
        assert len(window.welcome.recent_rows()) == 2
        assert not any(math.isnan(e.count) for e in window.recent_projects.entries() if e.count)
    finally:
        delete_widgets([window])


def _tab_walk(start, steps: int) -> list:
    start.setFocus(Qt.FocusReason.TabFocusReason)
    QApplication.processEvents()
    seen = [QApplication.focusWidget()]
    for _ in range(steps):
        QTest.keyClick(QApplication.focusWidget(), Qt.Key.Key_Tab)
        QApplication.processEvents()
        seen.append(QApplication.focusWidget())
    return seen


def test_welcome_tab_order_follows_the_screen(app):
    import json

    from app.gui.recent_projects import RecentProjectsStore
    from app.gui.welcome_screen import WelcomeWidget

    class Settings(dict):
        def value(self, key, default=None):
            return self.get(key, default)

        def setValue(self, key, value):  # noqa: N802 - Qt API name
            self[key] = value

    raw = json.dumps([{"path": "C:/a", "count": 1}, {"path": "C:/b", "count": 2}])
    store = RecentProjectsStore(Settings(recent_projects=raw))
    welcome = WelcomeWidget(
        on_get_started=lambda: None,
        recent_store=store,
        probe_async=False,
        path_exists=lambda _p: True,
        on_about=lambda: None,
    )
    welcome.resize(1280, 800)
    welcome.show()
    welcome.activateWindow()
    assert QTest.qWaitForWindowActive(welcome, 3000)
    rows = welcome.recent_rows()
    walk = _tab_walk(welcome.browse_button, 5)
    assert walk[:5] == [
        welcome.browse_button,
        rows[0].open_button,
        rows[0].remove_button,
        rows[1].open_button,
        rows[1].remove_button,
    ]
    assert walk[5].text() == "About"  # the footer comes last

    welcome.refresh_recents()  # rebuilt rows keep the order
    rows = welcome.recent_rows()
    assert _tab_walk(welcome.browse_button, 1)[1] is rows[0].open_button
    welcome.close()
    welcome.deleteLater()


@pytest.mark.parametrize(
    ("url", "accepted"),
    [("folder", True), ("file", True), ("https://example.org/x", False)],
)
def test_a_drag_over_the_dropzone_is_accepted_only_for_a_local_path(app, tmp_path, url, accepted):
    """Qt delivers a drop only after the drag-enter was accepted: the drop tests above call
    ``dropEvent`` directly and cannot see this."""
    from PyQt6.QtGui import QDragEnterEvent

    (tmp_path / "a.jpg").write_bytes(b"x")
    local = {"folder": tmp_path, "file": tmp_path / "a.jpg"}
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(local[url])) if url in local else QUrl(url)])
    event = QDragEnterEvent(
        QPoint(5, 5),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    event.ignore()
    _welcome(_Spy()).dropzone.dragEnterEvent(event)
    assert event.isAccepted() is accepted


@pytest.mark.parametrize(
    ("seconds", "label"),
    [
        (59, "just now"),
        (60, "1m ago"),
        (3599, "59m ago"),
        (3600, "1h ago"),
        (86399, "23h ago"),
        (86400, "1d ago"),
        (6 * 86400, "6d ago"),
    ],
)
def test_a_recent_rows_age_reads_in_the_largest_whole_unit(seconds, label):
    stamp = (datetime.now() - timedelta(seconds=seconds)).isoformat()
    assert WelcomeWidget._relative_date(stamp) == label


def test_a_recent_row_a_week_old_or_more_shows_its_date():
    stamp = datetime.now() - timedelta(days=7, seconds=1)
    assert WelcomeWidget._relative_date(stamp.isoformat()) == stamp.strftime("%Y-%m-%d")
