"""viz-embed: the dashboards and the canvases' OpenGL viewports, the release of the dashboards
when another project opens, and their idle release after Visualize was left
(app/gui/visualization_manager.py).

Where a dashboard is a plain child widget the canvases use raster viewports while one lives
and get OpenGL back when the last one closes, only if they had it; a released dashboard
leaves the window before OpenGL returns and is destroyed by the deferred delete. Dashboards
close ``IDLE_RELEASE_MS`` after Visualize was left, never under a stroke or a dialog.
Stand-ins: ``tests/viz_manager_support.py``.
"""

from __future__ import annotations

import gc
import os

import pytest
from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication, QEvent, QObject, Qt
from PyQt6.QtWidgets import QApplication

from app.gui import visualization_manager as vm
from tests.qt_teardown import flush_deleted
from tests.qt_wait import wait_until as _wait
from tests.viz_manager_support import (
    _PANELS,
    FakeImageManager,
    StubLength,
    _environment,
    _touch_newer,
)


@pytest.fixture
def project(tmp_path):
    (tmp_path / "skeletons").mkdir()
    (tmp_path / "mask").mkdir()
    (tmp_path / "skeletons" / "root_lengths.csv").write_text(
        "Image,Length (mm)\nimg_001,1.5\n", encoding="utf-8"
    )
    (tmp_path / "mask" / "root_areas.csv").write_text(
        "Image,Area (mm²)\nimg_001,0.5\n", encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def env(qapp, monkeypatch, project):
    ctx = _environment(qapp, monkeypatch, project, opengl=False)
    yield ctx
    ctx.mw.deleteLater()
    del ctx
    flush_deleted()
    QCoreApplication.processEvents()


@pytest.fixture
def env_gl(qapp, monkeypatch, project):
    ctx = _environment(qapp, monkeypatch, project, opengl=True)
    yield ctx
    ctx.mw.deleteLater()
    del ctx
    flush_deleted()
    QCoreApplication.processEvents()


# --------------------------------------------------------------------------- #
#  OpenGL viewports vs. WebEngine (PV-12, UI-16)
# --------------------------------------------------------------------------- #
def test_opengl_stays_off_until_the_last_dashboard_closes(env_gl):
    mw = env_gl.mw
    vm.show_root_length_visualization(mw)
    vm.show_root_area_visualization(mw)

    vm.close_root_length_visualization(mw)
    assert mw.gl_calls == [False]  # the area dashboard still embeds a browser

    vm.close_root_area_visualization(mw)
    assert mw.gl_calls == [False, True]


def test_opengl_is_not_forced_on_when_it_was_off_before(env):
    mw = env.mw  # canvases on raster viewports (opt-out or no GL support)
    vm.show_root_length_visualization(mw)

    vm.close_root_length_visualization(mw)

    assert True not in mw.gl_calls


def test_native_dashboards_leave_the_opengl_canvases_on(env_gl, monkeypatch):
    # Route (b) of the dashboard blink (KNOWN_ISSUES D8): on Windows a dashboard is a
    # native child window with its own Direct3D surface, which composites next to the
    # OpenGL canvases, so the canvases are never switched to raster viewports.
    monkeypatch.setattr(vm, "dashboards_embed_natively", lambda: True)
    mw = env_gl.mw

    vm.show_root_length_visualization(mw)
    vm.show_root_area_visualization(mw)
    vm.close_root_length_visualization(mw)
    vm.release_all_visualizations(mw)
    vm.show_root_length_visualization(mw)

    assert mw.gl_calls == []
    assert mw._gl_viewport is not None


def test_a_dashboard_that_cannot_be_built_gives_the_canvases_opengl_back(env_gl, monkeypatch):
    class Broken:
        def __init__(self, *_args):
            raise RuntimeError("no WebEngine")

    monkeypatch.setattr(vm, "RootLengthVisualization", Broken)
    mw = env_gl.mw

    vm.show_root_length_visualization(mw)

    assert mw.root_length_viz is None
    assert mw.gl_calls == [False, True]  # suspended for the browser, then given back
    assert env_gl.boxes == [("critical", "Error", "Failed to create visualization: no WebEngine")]


# --------------------------------------------------------------------------- #
#  A released dashboard leaves the window before OpenGL returns (flicker, Part A).
#  Qt picks the window's surface from its children: a dying dashboard still in the
#  window made it Direct3D once more, so the release recreated the window twice.
# --------------------------------------------------------------------------- #
def _where_at_gl_return(mw, dashboards):
    """Spy on the OpenGL return: for each dashboard, (hidden, parentless, in the panel,
    deleted) when ``set_opengl_viewports_enabled(True)`` runs."""
    seen = []
    real = mw.set_opengl_viewports_enabled

    def spy(enabled):
        if enabled:
            seen.extend(
                (
                    board.isHidden(),
                    board.parent() is None,
                    mw.right_panel.indexOf(board) != -1,
                    sip.isdeleted(board),
                )
                for board in dashboards
            )
        real(enabled)

    mw.set_opengl_viewports_enabled = spy
    return seen


def test_a_closed_dashboard_is_out_of_the_window_when_opengl_returns(env_gl):
    mw = env_gl.mw
    vm.show_root_length_visualization(mw)
    dashboard = mw.root_length_viz
    seen = _where_at_gl_return(mw, [dashboard])

    vm.close_root_length_visualization(mw)

    assert seen == [(True, True, False, False)]  # hidden, no parent, not in the panel, alive
    assert mw.gl_calls == [False, True]


def test_the_release_takes_every_dashboard_out_before_opengl_returns(env_gl):
    mw = env_gl.mw
    vm.show_root_length_visualization(mw)
    vm.show_root_area_visualization(mw)
    boards = [mw.root_length_viz, mw.root_area_viz]
    seen = _where_at_gl_return(mw, boards)

    vm.release_all_visualizations(mw)

    assert seen == [(True, True, False, False)] * 2


def test_a_replaced_dashboard_leaves_the_window_too(env):
    mw = env.mw
    vm.show_root_length_visualization(mw)
    old = mw.root_length_viz
    _touch_newer(env.project / "skeletons" / "root_lengths.csv", "Image,Length (mm)\n")

    vm.show_root_length_visualization(mw)  # rebuilt: the stale one is discarded

    assert old.isHidden() and old.parent() is None
    assert mw.right_panel.indexOf(old) == -1


def _alive(name: str) -> bool:
    """True while a widget of that object name exists in C++ (no Python reference needed)."""
    return any(widget.objectName() == name for widget in QApplication.allWidgets())


def test_a_released_dashboard_is_destroyed_by_the_deferred_delete_not_earlier(env_gl):
    mw = env_gl.mw
    vm.show_root_length_visualization(mw)
    mw.root_length_viz.setObjectName("released-dashboard")
    StubLength.made = []  # no reference left but the window's, as in the app
    alive_at_gl_return = []
    real = mw.set_opengl_viewports_enabled
    mw.set_opengl_viewports_enabled = lambda on: (
        alive_at_gl_return.append(_alive("released-dashboard")),
        real(on),
    )

    vm.release_all_visualizations(mw)
    gc.collect()  # a collection before the deferred delete (any thread may run one)

    assert alive_at_gl_return == [True]
    assert _alive("released-dashboard")  # its Python wrapper may go; the widget stays
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not _alive("released-dashboard")


class _ShowCounter(QObject):
    def __init__(self):
        super().__init__()
        self.shows = 0

    def eventFilter(self, obj, event):  # noqa: N802 - Qt's name
        if event.type() == QEvent.Type.Show:
            self.shows += 1
        return False


@pytest.mark.parametrize("release", ["close", "release_all"])
def test_a_released_dashboard_is_never_shown_as_a_window_of_its_own(env, release):
    mw = env.mw
    mw.show()
    vm.show_root_length_visualization(mw)  # on screen
    dashboard = mw.root_length_viz
    counter = _ShowCounter()
    dashboard.installEventFilter(counter)
    try:
        if release == "close":
            vm.close_root_length_visualization(mw)
        else:
            vm.release_all_visualizations(mw)

        assert not dashboard.isVisible()
        assert dashboard.windowHandle() is None  # no native window was made for it
        QCoreApplication.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        assert sip.isdeleted(dashboard)
        assert counter.shows == 0
    finally:
        mw.hide()


# --------------------------------------------------------------------------- #
#  Another project was opened (PKG-VIZ, V3)
# --------------------------------------------------------------------------- #
def test_opening_another_project_releases_both_dashboards(env_gl):
    # Before, both dashboards kept charting the previous project's CSVs (and their
    # servers, pages and the OpenGL hold) until they were shown again; the one on
    # screen stayed on screen under the new project's library.
    mw = env_gl.mw
    vm.show_root_length_visualization(mw)
    vm.show_root_area_visualization(mw)  # on screen
    length, area = mw.root_length_viz, mw.root_area_viz

    shown = vm.release_all_visualizations(mw)

    assert shown is True
    assert mw.root_length_viz is None and mw.root_area_viz is None
    assert (length.cleanups, area.cleanups) == (1, 1)
    assert mw.right_panel.indexOf(length) == -1 and mw.right_panel.indexOf(area) == -1
    assert mw.right_panel.currentIndex() == _PANELS["display"]
    assert mw.gl_calls == [False, True]  # the canvases get OpenGL back
    assert mw.refreshed == []  # the old project's image is not shown again
    assert "Close" not in mw.visualize_root_length_button.text()


def test_releasing_hidden_dashboards_keeps_the_current_page(env):
    mw = env.mw
    vm.show_root_length_visualization(mw)
    mw.switch_right_panel("display")

    assert vm.release_all_visualizations(mw) is False
    assert mw.root_length_viz is None
    assert mw.right_panel.currentIndex() == _PANELS["display"]


def test_releasing_without_any_dashboard_does_nothing(env):
    assert vm.release_all_visualizations(env.mw) is False
    assert env.mw.gl_calls == []


def test_a_dashboard_opened_after_the_release_reads_the_new_project(env, tmp_path_factory):
    mw = env.mw
    vm.show_root_length_visualization(mw)
    other = tmp_path_factory.mktemp("other_project")
    (other / "skeletons").mkdir()
    (other / "skeletons" / "root_lengths.csv").write_text(
        "Image,Length (mm)\nimg_002,2.5\n", encoding="utf-8"
    )
    mw.image_manager = FakeImageManager(other)

    vm.release_all_visualizations(mw)
    vm.show_root_length_visualization(mw)

    assert len(StubLength.made) == 2
    expected = other / "skeletons" / "root_lengths.csv"
    assert os.path.normcase(mw.root_length_viz.csv_path) == os.path.normcase(str(expected))


# --------------------------------------------------------------------------- #
#  Idle release: dashboards close a while after the Visualize stage is left
#  (R-VIZ X1). Revisits within it stay instant; after it the canvases get
#  OpenGL back and the browser processes are freed.
# --------------------------------------------------------------------------- #
def _visit_and_leave(mw, stage="Correct"):
    vm.stage_changed(mw, "Visualize")
    vm.show_root_length_visualization(mw)
    mw.switch_right_panel("display")
    vm.stage_changed(mw, stage)
    return vm.controller_for(mw)


def test_leaving_visualize_counts_down_to_closing_the_dashboards(env_gl):
    mw = env_gl.mw
    controller = _visit_and_leave(mw)
    dashboard = mw.root_length_viz

    assert controller.idle_release_pending()
    assert controller._idle_timer.interval() == vm.IDLE_RELEASE_MS == 5 * 60 * 1000
    assert mw.gl_calls == [False]  # raster canvases while the dashboard lives

    controller._idle_timer.stop()
    controller._on_idle()  # the countdown ended

    assert mw.root_length_viz is None and dashboard.cleanups == 1
    assert mw.gl_calls == [False, True]  # OpenGL canvases again
    assert mw.right_panel.currentIndex() == _PANELS["display"]  # the page shown stays
    assert mw.refreshed == []


def test_the_countdown_really_fires(env, monkeypatch):
    monkeypatch.setattr(vm, "IDLE_RELEASE_MS", 30)
    mw = env.mw
    _visit_and_leave(mw)

    assert _wait(lambda: mw.root_length_viz is None, 3.0)


def test_coming_back_within_the_countdown_keeps_the_live_dashboards(env):
    mw = env.mw
    controller = _visit_and_leave(mw)
    dashboard = mw.root_length_viz

    vm.stage_changed(mw, "Visualize")
    vm.show_root_length_visualization(mw)

    assert not controller.idle_release_pending()
    assert mw.root_length_viz is dashboard and dashboard.cleanups == 0


def test_moving_between_other_stages_keeps_the_original_deadline(env, monkeypatch):
    mw = env.mw
    controller = _visit_and_leave(mw, "Correct")
    started = []
    monkeypatch.setattr(controller._idle_timer, "start", lambda *a: started.append(a))

    vm.stage_changed(mw, "Library")
    vm.stage_changed(mw, "Generate")

    assert started == []  # still counting from when Visualize was left
    assert controller.idle_release_pending()


def test_a_dashboard_back_on_screen_is_never_closed_by_the_countdown(env):
    mw = env.mw
    controller = _visit_and_leave(mw)
    vm.show_root_length_visualization(mw)  # e.g. a legacy button, no stage report

    controller._idle_timer.stop()
    controller._on_idle()

    assert mw.root_length_viz is not None and mw.root_length_viz.cleanups == 0


@pytest.mark.parametrize("busy", ["mouse", "popup", "modal"])
def test_a_release_during_a_stroke_or_dialog_waits(env, monkeypatch, busy):
    # The release swaps the canvases' viewports (raster -> OpenGL): never under a
    # brush stroke, a drag or an open dialog.
    mw = env.mw
    controller = _visit_and_leave(mw)
    if busy == "mouse":
        monkeypatch.setattr(
            vm.QApplication, "mouseButtons", staticmethod(lambda: Qt.MouseButton.LeftButton)
        )
    elif busy == "popup":
        monkeypatch.setattr(vm.QApplication, "activePopupWidget", staticmethod(lambda: mw))
    else:
        monkeypatch.setattr(vm.QApplication, "activeModalWidget", staticmethod(lambda: mw))

    controller._idle_timer.stop()
    controller._on_idle()

    assert mw.root_length_viz is not None  # kept for now
    assert controller.idle_release_pending()
    assert controller._idle_timer.interval() == vm.IDLE_RELEASE_RETRY_MS
    controller._idle_timer.stop()


def test_stage_changes_before_any_dashboard_create_nothing(env):
    vm.stage_changed(env.mw, "Correct")
    vm.stage_changed(env.mw, "Visualize")

    assert env.mw not in vm._controllers


def test_leaving_visualize_without_a_live_dashboard_starts_no_countdown(env):
    mw = env.mw
    controller = _visit_and_leave(mw)
    vm.close_root_length_visualization(mw)
    controller._idle_timer.stop()

    vm.stage_changed(mw, "Visualize")
    vm.stage_changed(mw, "Library")

    assert not controller.idle_release_pending()


def test_opening_another_project_ends_the_countdown(env):
    mw = env.mw
    controller = _visit_and_leave(mw)

    vm.release_all_visualizations(mw)

    assert not controller.idle_release_pending()


def test_the_countdown_does_not_keep_its_controller_alive(env):
    import gc
    import weakref as _weakref

    controller = vm.DashboardController(env.mw)  # not registered for the window
    controller._idle_timer.start(60_000)
    ref = _weakref.ref(controller)
    del controller
    gc.collect()

    assert ref() is None  # the timer's slot holds the controller weakly
