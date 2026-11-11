"""The window's composition and its fixed texts.

Promises: five stages in order, each with its Alt key in its tooltip; the right panel's
pages in ``Panel`` order with Correct's page the editor and its GT actions on its action
page; the metrics bar on the display page only, with the Library's zoom in it; the job
registry and the OpenGL switch reach every canvas; the buttons say what they do (Measure
Length / Area name their CSV and the GT file read first; Generate makes masks and skeletons
from the photo, neither from the other).
"""

from __future__ import annotations

import pytest

from app.gui import shell_chrome
from app.gui.panels import Panel
from tests.metrics_bar_support import shown, text
from tests.qt_teardown import delete_widgets
from tests.window_project_support import enter, new_project, open_project, select
from tests.window_support import build_window, quiet_window
from tests.window_support import make_window as make_window  # fixture
from tests.window_support import mw as mw  # fixture
from tests.window_support import no_delayed_temp_sweep as no_delayed_temp_sweep  # fixture
from tests.window_support import settings as settings  # fixture

_CONNECT_MAP_ATTRS = [
    "load_images_button",
    "generate_mask_button",
    "generate_button",
    "calculate_length_button",
    "calculate_area_button",
    "visualize_root_length_button",
    "visualize_root_area_button",
    "view_mode_combo",
    "expand_all_button",
    "collapse_all_button",
    "file_list",
]


def test_main_window_contract(app):
    from app.gui.main_window import MainWindow

    mw = MainWindow()
    try:
        for attr in _CONNECT_MAP_ATTRS:
            assert hasattr(mw, attr), f"missing {attr}"
        assert mw.view_mode_combo.count() == 3
        assert mw.right_panel.count() == len(Panel)  # display, Correct, viz
        # Five-stage pipeline, in order: mask+skeleton generation is one stage, and so
        # is their correction with the class labels (one canvas, three layers).
        assert list(mw.ribbon_buttons.keys()) == [
            "Library",
            "Generate",
            "Correct",
            "Measure",
            "Visualize",
        ]
        assert mw.right_panel.widget(Panel.CORRECT) is mw.correct_editor
        assert mw.gt_classes_button.text() == "Classes…"
        assert mw.gt_export_button.text() == "Export dataset…"
        assert mw.gt_remove_button.text() == "Remove GT…"
        assert mw.gt_empty_button.text() == "Mark empty"
        # WG-T5: the GT actions are on the Correct page.
        correct_page = mw.action_bar_stack.widget(mw._action_bar_pages["Correct"])
        for button in (
            mw.gt_classes_button,
            mw.gt_export_button,
            mw.gt_empty_button,
            mw.gt_remove_button,
        ):
            assert button.parentWidget() is correct_page
        assert mw.app_stack.count() >= 2
        # FIX2: the dialog-free loader entry point exists for recent rows.
        assert callable(getattr(mw, "open_path", None))
    finally:
        delete_widgets([mw])


@pytest.mark.usefixtures("window_guards")
def test_the_metrics_bar_is_on_the_library_page_only(app, tmp_path):
    """The bar sits under the display page: on screen in the Library (and Generate /
    Measure), gone in Correct and on the Visualize page (the toasts keep clear of it only
    while it is visible)."""
    window = build_window(tmp_path / "s.ini")
    try:
        open_project(window, new_project(tmp_path))
        select(window, 0)
        assert window.metrics_bar.isVisible()

        enter(window, "Correct")
        assert window.right_panel.currentIndex() == Panel.CORRECT
        assert not window.metrics_bar.isVisible()

        enter(window, "Visualize")  # nothing measured: the stage page names the next step
        assert window.right_panel.currentIndex() == Panel.VISUALIZATION
        assert not window.metrics_bar.isVisible()

        enter(window, "Measure")
        assert window.metrics_bar.isVisible()
    finally:
        delete_widgets([window], prepare=quiet_window)


def test_stage_tooltips_name_their_alt_key(app):
    from unittest.mock import MagicMock

    from app.gui import shell_chrome, shortcuts
    from app.gui.key_text import platform_keys

    bar = shell_chrome.ShellTopbar(MagicMock())  # a stand-in window: only slots are wired
    for (key, *_), keys in zip(shell_chrome.STAGES, shortcuts.stage_keys(), strict=True):
        assert bar.stage_buttons[key].toolTip().endswith(f"({platform_keys(' / '.join(keys))})")


def test_the_generate_stage_says_masks_and_skeletons_are_independent():
    tips = dict((k, tip) for k, _, _, tip in shell_chrome.STAGES)
    assert "neither follows the other" in tips["Generate"]


def test_the_measure_buttons_say_what_they_do(mw):  # noqa: F811
    assert mw.calculate_length_button.text() == "Measure Length"
    assert mw.calculate_area_button.text() == "Measure Area"
    length_tip = mw.calculate_length_button.toolTip()
    area_tip = mw.calculate_area_button.toolTip()
    assert "root_lengths.csv" in length_tip and "GT skeleton" in length_tip
    assert "root_areas.csv" in area_tip and "GT mask" in area_tip


def test_skeleton_tooltips_do_not_claim_masks_are_the_input(mw):
    # generate_button's tooltip is re-set by shell_chrome (see notes/library.md).
    tip = mw.generate_all_button.toolTip().lower()
    assert "from the masks" not in tip and "original images" in tip


def test_window_exposes_job_registry_and_toggles_every_view(mw, monkeypatch):
    from app.handlers.job_registry import JobRegistry

    assert isinstance(mw.jobs, JobRegistry)
    seen = []
    views = [
        mw.display_controller.magnifying_view,
        mw.correct_editor.graphics_view,
    ]
    for v in views:
        monkeypatch.setattr(v, "set_opengl_viewport_enabled", lambda e, v=v: seen.append((v, e)))
    mw.set_opengl_viewports_enabled(False)
    assert [v for v, _ in seen] == views and all(e is False for _, e in seen)


@pytest.mark.usefixtures("window_guards")
def test_the_library_zoom_is_on_the_metrics_bar(app, tmp_path):
    window = build_window(tmp_path / "s.ini")
    try:
        open_project(window, new_project(tmp_path))
        select(window, 1)
        view = window.display_controller.magnifying_view
        bar = window.metrics_bar

        assert shown(bar, "zoom")
        assert text(bar, "zoom") == f"{view.zoom_factor * 100:.0f} %"
        view.set_zoom(1.0)
        assert text(bar, "zoom") == "100 %"
    finally:
        delete_widgets([window])
