"""viz-embed: dashboard management for the main window (app/gui/visualization_manager.py):
show, bring forward, rebuild, close, the guards before a build, and the export notices.

``show`` is idempotent: it builds a dashboard when none is alive, when its CSV changed on disk
or when it failed, else brings the live one forward; showing one dashboard hides the other;
``close`` removes one exactly once. The OpenGL suspension, the release and the idle countdown
are in ``test_viz_embed_manager_release.py``. Stand-ins: ``tests/viz_manager_support.py``.
"""

from __future__ import annotations

import os

import pytest
from PyQt6.QtCore import QCoreApplication

from app.gui import visualization_manager as vm
from tests.qt_teardown import flush_deleted
from tests.viz_manager_support import (
    _PANELS,
    StubArea,
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


# --------------------------------------------------------------------------- #
#  Show is idempotent (UI-05, PV-04, RB-14)
# --------------------------------------------------------------------------- #
def test_show_opens_a_dashboard_once_then_only_brings_it_forward(env):
    mw = env.mw
    vm.show_root_length_visualization(mw)
    first = mw.root_length_viz
    mw.switch_right_panel("display")

    vm.show_root_length_visualization(mw)

    assert StubLength.made == [first]
    assert mw.right_panel.currentWidget() is first
    assert first.cleanups == 0


def test_length_dashboard_reads_the_skeletons_csv_with_the_image_manager(env):
    vm.show_root_length_visualization(env.mw)

    (dashboard,) = StubLength.made
    expected = env.project / "skeletons" / "root_lengths.csv"
    assert os.path.normcase(dashboard.csv_path) == os.path.normcase(str(expected))
    assert dashboard.image_manager is env.mw.image_manager


def test_area_dashboard_reads_the_mask_csv(env):
    vm.show_root_area_visualization(env.mw)

    (dashboard,) = StubArea.made
    expected = env.project / "mask" / "root_areas.csv"
    assert os.path.normcase(dashboard.csv_path) == os.path.normcase(str(expected))


def test_area_dashboard_gets_the_image_manager(env):
    vm.show_root_area_visualization(env.mw)

    assert StubArea.made[0].image_manager is env.mw.image_manager


def test_showing_one_dashboard_hides_the_other_without_closing_it(env):
    mw = env.mw
    vm.show_root_length_visualization(mw)
    vm.show_root_area_visualization(mw)

    vm.show_root_length_visualization(mw)

    assert len(StubLength.made) == 1 and len(StubArea.made) == 1
    assert mw.right_panel.currentWidget() is mw.root_length_viz
    assert mw.root_area_viz is StubArea.made[0]
    assert mw.root_area_viz.cleanups == 0
    assert vm.visible_kind(mw) == "length"


def test_visible_kind_follows_the_panel(env):
    mw = env.mw
    assert vm.visible_kind(mw) is None

    vm.show_root_area_visualization(mw)
    assert vm.visible_kind(mw) == "area"

    mw.switch_right_panel("display")
    assert vm.visible_kind(mw) is None


def test_dashboard_is_rebuilt_when_its_csv_changed_on_disk(env):
    mw = env.mw
    vm.show_root_length_visualization(mw)
    old = mw.root_length_viz
    _touch_newer(env.project / "skeletons" / "root_lengths.csv", "Image,Length (mm)\n")

    vm.show_root_length_visualization(mw)

    assert old.cleanups == 1
    assert mw.right_panel.indexOf(old) == -1
    assert len(StubLength.made) == 2
    assert mw.root_length_viz is StubLength.made[1]
    assert mw.right_panel.currentWidget() is StubLength.made[1]


def test_a_csv_rewritten_at_the_same_size_is_rebuilt_too(env):
    # Measure can rewrite a CSV with other values of the same length: only its time tells.
    mw = env.mw
    vm.show_root_length_visualization(mw)
    old = mw.root_length_viz
    csv = env.project / "skeletons" / "root_lengths.csv"
    size = csv.stat().st_size
    _touch_newer(csv, "Image,Length (mm)\nimg_001,2.5\n")
    assert csv.stat().st_size == size

    vm.show_root_length_visualization(mw)

    assert old.cleanups == 1
    assert len(StubLength.made) == 2
    assert mw.right_panel.currentWidget() is StubLength.made[1]


def test_a_live_dashboard_is_kept_while_measure_rewrites_its_csv(env):
    mw = env.mw
    vm.show_root_length_visualization(mw)
    live = mw.root_length_viz
    mw.switch_right_panel("display")
    mw.skeleton_handler.busy_kinds = {"length"}
    _touch_newer(env.project / "skeletons" / "root_lengths.csv", "Image,Length (mm)\n")

    vm.show_root_length_visualization(mw)

    assert StubLength.made == [live] and live.cleanups == 0  # not rebuilt from a half-written CSV
    assert mw.right_panel.currentWidget() is live


def test_failed_dashboard_is_rebuilt_on_show(env):
    mw = env.mw
    vm.show_root_area_visualization(mw)
    broken = mw.root_area_viz
    broken.failed = True

    vm.show_root_area_visualization(mw)

    assert broken.cleanups == 1
    assert len(StubArea.made) == 2
    assert mw.right_panel.currentWidget() is mw.root_area_viz


def test_generic_api_takes_the_dashboard_kind(env):
    mw = env.mw
    vm.show_visualization(mw, "area")
    assert mw.root_area_viz is not None

    vm.close_visualization(mw, "area")
    assert mw.root_area_viz is None

    with pytest.raises(ValueError):
        vm.show_visualization(mw, "volume")


# --------------------------------------------------------------------------- #
#  Guards before anything is built
# --------------------------------------------------------------------------- #
def test_missing_data_leaves_the_panels_and_opengl_alone(env):
    mw = env.mw
    os.remove(env.project / "skeletons" / "root_lengths.csv")

    vm.show_root_length_visualization(mw)

    assert StubLength.made == []
    assert mw.gl_calls == []
    assert mw.right_panel.currentIndex() == 0
    assert [box[0] for box in env.boxes] == ["information"]


def test_show_waits_while_the_data_is_being_measured(env):
    mw = env.mw
    mw.skeleton_handler.busy_kinds = {"area"}

    vm.show_root_area_visualization(mw)

    assert StubArea.made == []
    assert mw.skeleton_handler.asked == ["area"]
    assert [box[0] for box in env.boxes] == ["information"]


def test_an_editor_that_refuses_to_close_keeps_its_panel(env):
    mw = env.mw
    mw.right_panel.setCurrentIndex(_PANELS["correct"])
    mw.leave_allowed = False

    vm.show_root_length_visualization(mw)

    assert StubLength.made == []
    assert mw.gl_calls == []
    assert mw.right_panel.currentIndex() == _PANELS["correct"]


def test_an_editor_that_keeps_its_edits_stays_in_front_of_a_live_dashboard(env):
    mw = env.mw
    vm.show_root_length_visualization(mw)
    live = mw.root_length_viz
    mw.switch_right_panel("correct")
    mw.leave_allowed = False

    vm.show_root_length_visualization(mw)

    assert mw.right_panel.currentIndex() == _PANELS["correct"]
    assert StubLength.made == [live] and live.cleanups == 0


def test_no_images_loaded_warns_and_opens_nothing(env):
    env.mw.file_list.clear()

    vm.show_root_length_visualization(env.mw)

    assert StubLength.made == []
    assert [box[0] for box in env.boxes] == ["warning"]


# --------------------------------------------------------------------------- #
#  Toggle and close (RB-12)
# --------------------------------------------------------------------------- #
def test_toggle_keeps_its_open_then_close_meaning(env):
    mw = env.mw
    vm.toggle_root_area_visualization(mw)
    dashboard = mw.root_area_viz
    assert dashboard is not None
    assert "Close" in mw.visualize_root_area_button.text()

    vm.toggle_root_area_visualization(mw)

    assert mw.root_area_viz is None
    assert dashboard.cleanups == 1
    assert mw.right_panel.indexOf(dashboard) == -1
    assert mw.right_panel.currentIndex() == _PANELS["display"]
    assert "Close" not in mw.visualize_root_area_button.text()


def test_close_cleans_up_once_although_server_closed_re_enters(env):
    mw = env.mw
    vm.show_root_length_visualization(mw)
    dashboard = mw.root_length_viz

    vm.close_root_length_visualization(mw)

    assert dashboard.cleanups == 1
    assert len(mw.refreshed) == 1  # the current image is re-displayed once
    assert mw.root_length_viz is None


def test_closing_a_hidden_dashboard_keeps_the_current_panel(env):
    mw = env.mw
    vm.show_root_length_visualization(mw)
    vm.show_root_area_visualization(mw)

    vm.close_root_length_visualization(mw)

    assert mw.right_panel.currentWidget() is mw.root_area_viz
    assert mw.refreshed == []


def test_exports_are_announced_in_the_main_window(env):
    mw = env.mw
    vm.show_root_length_visualization(mw)

    mw.root_length_viz.export_saved.emit(str(env.project / "skeletons" / "plot.svg"))

    assert mw.notices and "plot.svg" in mw.notices[-1][0]


def test_failed_exports_are_announced_as_warnings(env):
    mw = env.mw
    vm.show_root_area_visualization(mw)
    message = f"Could not save plot.svg in {env.project / 'mask'}: Insufficient Permissions"

    mw.root_area_viz.export_failed.emit(message)

    assert mw.notices[-1] == (message, "warn")


# --------------------------------------------------------------------------- #
#  Where dashboards embed natively (KNOWN_ISSUES D8)
# --------------------------------------------------------------------------- #
def test_x11_dashboards_embed_natively_and_skip_the_canvas_swap(monkeypatch):
    import app.gui.visualization_manager as vm
    import app.visualization.viz_window as vw

    real = vw.native_child_surface
    monkeypatch.setattr(
        vw, "native_child_surface", lambda platform=None, environ=None: real("xcb", {})
    )
    assert vm.dashboards_embed_natively() is True
    monkeypatch.setattr(
        vw, "native_child_surface", lambda platform=None, environ=None: real("wayland", {})
    )
    assert vm.dashboards_embed_natively() is False


# --------------------------------------------------------------------------- #
#  The dashboards' import warm-up (the warm-up queue's DASHBOARDS job)
# --------------------------------------------------------------------------- #
def test_the_import_warmup_imports_the_dashboard_stack(monkeypatch):
    import app.visualization.viz_window as vw

    calls = []
    monkeypatch.setattr(vw, "prewarm_imports", lambda: calls.append(1))

    assert vm.warm_dashboard_imports() == ""
    assert calls == [1]


def test_a_failing_import_prewarm_is_only_logged(monkeypatch, caplog):
    import app.visualization.viz_window as vw

    def broken():
        raise ImportError("dash is not installed")

    monkeypatch.setattr(vw, "prewarm_imports", broken)

    assert vm.warm_dashboard_imports() == "dash is not installed"  # never raises
    assert "dash is not installed" in caplog.text
