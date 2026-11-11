"""LayerPanel (``app/gui/correct/layer_panel.py``, plan contract C20): the Layers panel of
the Correct page, one row per layer with the mask on top, its key, colour, visibility and
save state; clicks never take the keyboard, Tab reaches each row and its eye (W18); and the
side card (``side_panels.py``) takes the rows' new width at once (S02-01).
A pure view: it asks for a layer or a visibility, the editor answers.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QColor, QImage
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
)

from app.gui.correct.layer import CLASSES, MASK, SKELETON, LayerStatus
from app.gui.correct.layer_panel import LayerPanel
from app.gui.widgets import tokens
from tests.correct_controls_support import (
    M,
    opaque_colours,
    pixel,
    record,
    right,
    settle,
    text_colours,
)
from tests.correct_controls_support import make_page as make_page  # fixture (the alias: no F811)
from tests.correct_fakes import FakeLayer
from tests.correct_support import activate, new_project
from tests.correct_support import settle as settle_editor
from tests.qt_teardown import delete_widgets
from tests.qt_wait import settle as run_events

#: The code under test: an exception it raises in a Qt slot fails the test (``slot_errors``,
#: which replaces the hook of ``correct_guards``, so the page's own module is named too).
SLOT_ERRORS_FROM = (
    os.path.join("correct", "layer_panel.py"),
    os.path.join("correct", "side_panels.py"),
    os.path.join("correct", "editor.py"),
    "test_correct_layer_panel.py",
)

pytestmark = pytest.mark.usefixtures("correct_guards", "slot_errors")


# ======================================================================= the Layers panel
def test_the_panel_lists_the_mask_above_the_skeleton_with_keys_names_and_colours(app, make_page):
    page = make_page((1400, 900))
    rows = page.panel.rows

    assert list(rows) == [MASK, SKELETON]
    assert rows[MASK].geometry().y() < rows[SKELETON].geometry().y()
    assert [rows[i].key_label.text() for i in (MASK, SKELETON)] == ["M", "L"]
    assert [rows[i].name_label.text() for i in (MASK, SKELETON)] == ["Mask", "Skeleton"]
    for layer_id, colour in ((MASK, tokens.MASK), (SKELETON, tokens.SKEL)):
        swatch = rows[layer_id].swatch.grab().toImage()
        assert pixel(swatch, QPoint(swatch.width() // 2, swatch.height() // 2)) == colour


@pytest.fixture
def standalone(app):
    """A LayerPanel over a plain host, placing itself (no OverlayLayout)."""
    hosts = []

    def make(layers) -> LayerPanel:
        host = QWidget()
        host.resize(600, 400)
        panel = LayerPanel(layers, host)
        host.show()
        panel.reposition()
        settle(app)
        hosts.append(host)
        return panel

    yield make
    delete_widgets(hosts)


def test_the_panel_is_never_narrower_than_155_px(standalone):
    """Spec 3.2: about 155 px, so the fit zoom barely changes; short names do not shrink it."""
    tiny = FakeLayer(MASK)
    tiny.noun = "m"

    panel = standalone([tiny])

    assert panel.sizeHint().width() < 155 and panel.width() == 155
    assert right(panel) == 600 - M.margin  # its own placement: the host's right edge


def test_the_mask_row_stays_on_top_and_keeps_its_key_whatever_the_order_given(standalone):
    rows = standalone([FakeLayer(SKELETON), FakeLayer(MASK)]).rows

    assert rows[MASK].geometry().y() < rows[SKELETON].geometry().y()
    assert [rows[i].key_label.text() for i in (MASK, SKELETON)] == ["M", "L"]


@pytest.mark.parametrize(
    "status, text, colour",
    [
        (LayerStatus("clean"), "Saved", tokens.OK),
        (LayerStatus("dirty"), "Unsaved", tokens.WARN),
        (LayerStatus("error"), "Not saved", tokens.DANGER),
        (LayerStatus("none", "No mask"), "No mask", tokens.TEXT_FAINT),
        (LayerStatus("none", "Prediction", "Not corrected yet"), "Prediction", tokens.TEXT_FAINT),
        (LayerStatus("error", "Not saved", "Disk full"), "Not saved", tokens.DANGER),
        (LayerStatus("clean", "GT"), "GT", tokens.OK),
    ],
    ids=["clean", "dirty", "error", "none-label", "tooltip", "error-reason", "label-wins"],
)
def test_set_status_shows_the_label_else_the_state_name_in_the_chip_colour(
    app, make_page, status, text, colour
):
    page = make_page((1400, 900))
    row = page.panel.rows[SKELETON]
    page.panel.set_status(SKELETON, LayerStatus("dirty", "something else", "old tip"))

    page.panel.set_status(SKELETON, status)
    settle(app, page)

    assert row.state_label.text() == text
    assert row.state_label.toolTip() == row.state_dot.toolTip() == status.tooltip
    assert text_colours(row.state_label.grab().toImage()) == {colour}
    assert opaque_colours(row.state_dot.grab().toImage()) == {colour}


def test_a_status_without_a_label_shows_nothing(app, make_page):
    page = make_page((1400, 900))
    row = page.panel.rows[MASK]
    page.panel.set_status(MASK, LayerStatus("clean"))

    page.panel.set_status(MASK, LayerStatus("none"))
    settle(app, page)

    assert row.state_label.text() == ""
    assert opaque_colours(row.state_label.grab().toImage()) == set()
    assert opaque_colours(row.state_dot.grab().toImage()) == set()


def test_a_long_status_label_is_shown_in_full(app, make_page):
    page = make_page((1400, 900))
    panel = page.panel
    label = panel.rows[SKELETON].state_label

    panel.set_status(SKELETON, LayerStatus("none", "Prediction · empty", "Not corrected yet"))
    for _ in range(3):  # no relayout from the test: the panel asks for its new width itself
        app.processEvents()

    assert label.width() >= label.fontMetrics().horizontalAdvance("Prediction · empty")
    left = label.mapTo(panel, QPoint(0, 0)).x()
    assert left + label.width() <= panel.width()


def test_set_active_gives_only_that_row_the_fill_and_the_bar_in_its_colour(app, make_page):
    page = make_page((1400, 900))
    panel = page.panel

    def bar(row) -> QPoint:
        return row.mapTo(panel, QPoint(0, row.height() // 2))

    def fill(row) -> QPoint:
        return row.mapTo(panel, QPoint(row.width() // 2, 2))

    for active, other, colour in ((SKELETON, MASK, tokens.SKEL), (MASK, SKELETON, tokens.MASK)):
        panel.set_active(active)
        settle(app, page)
        image = panel.grab().toImage()
        on, off = panel.rows[active], panel.rows[other]
        assert (pixel(image, bar(on)), pixel(image, fill(on))) == (colour, tokens.BG_2)
        width = [pixel(image, bar(on) + QPoint(dx, 0)) for dx in (1, 3)]
        assert width == [colour, tokens.BG_2]  # a 2 px bar
        assert (pixel(image, bar(off)), pixel(image, fill(off))) == (tokens.BG_1, tokens.BG_1)
        assert on.is_active() and not off.is_active()


def test_a_click_on_a_row_requests_that_layer_and_leaves_the_active_row_to_the_editor(
    app, make_page
):
    page = make_page((1400, 900))
    panel = page.panel
    panel.set_active(MASK)
    requests = record(panel.activate_requested)

    QTest.mouseClick(panel.rows[SKELETON].name_label, Qt.MouseButton.LeftButton)
    QTest.mouseClick(panel.rows[SKELETON], Qt.MouseButton.RightButton, pos=QPoint(4, 4))
    QTest.mouseClick(panel.rows[MASK], Qt.MouseButton.LeftButton, pos=QPoint(4, 4))

    assert requests == [(SKELETON,), (MASK,)]  # the right click asked for nothing
    assert panel.rows[MASK].is_active() and not panel.rows[SKELETON].is_active()


def test_an_eye_click_requests_the_other_visibility_and_does_not_activate_the_row(app, make_page):
    page = make_page((1400, 900))
    panel = page.panel
    toggles, activations = record(panel.visibility_toggled), record(panel.activate_requested)
    eye = panel.rows[SKELETON].eye_button

    QTest.mouseClick(eye, Qt.MouseButton.LeftButton)
    panel.set_visible_state(SKELETON, False)  # the editor hid it
    QTest.mouseClick(eye, Qt.MouseButton.LeftButton)
    panel.set_visible_state(SKELETON, False)  # the editor kept it hidden
    QTest.mouseClick(eye, Qt.MouseButton.LeftButton)

    assert toggles == [(SKELETON, False), (SKELETON, True), (SKELETON, True)]
    assert activations == []


def test_set_visible_state_changes_only_the_eye_and_emits_nothing(app, make_page):
    page = make_page((1400, 900))
    panel = page.panel
    row = panel.rows[MASK]
    panel.set_active(MASK)
    panel.set_status(MASK, LayerStatus("dirty"))
    toggles, activations = record(panel.visibility_toggled), record(panel.activate_requested)

    def eye_image() -> QImage:
        return row.eye_button.icon().pixmap(row.eye_button.iconSize(), 1.0).toImage()

    shown_eye, shown_tip = eye_image(), row.eye_button.toolTip()
    button = row.eye_button.grab().toImage()
    assert QColor.fromRgba(button.pixel(1, button.height() // 2)).alpha() == 0  # no checked pill
    panel.set_visible_state(MASK, False)
    hidden_eye, hidden_tip = eye_image(), row.eye_button.toolTip()
    panel.set_visible_state(MASK, True)

    assert hidden_eye != shown_eye and eye_image() == shown_eye
    assert "Hide" in shown_tip and "Show" in hidden_tip and "mask" in hidden_tip
    assert all("H: every layer" in tip and "Shift+H" in tip for tip in (shown_tip, hidden_tip))
    assert toggles == [] and activations == []
    assert row.is_active() and row.state_label.text() == "Unsaved"


def test_flash_highlights_the_row_for_about_300_ms_then_restores_it(app, make_page):
    page = make_page((1400, 900))
    panel = page.panel
    panel.set_active(MASK)
    settle(app, page)
    row = panel.rows[SKELETON]
    before = row.grab().toImage()

    start = time.perf_counter()
    panel.flash(SKELETON)
    assert row.grab().toImage() != before
    while row.grab().toImage() != before:
        assert time.perf_counter() - start < 3.0, "the flash never ends"
        QTest.qWait(10)

    assert time.perf_counter() - start >= 0.25


def test_the_panel_takes_its_side_panel_place_and_fits_a_shorter_height(app, make_page):
    page = make_page((1400, 900))
    panel = page.panel
    natural = panel.preferred_height(900)

    assert right(panel) == 1400 - M.margin and panel.height() == natural
    panel.fit_height(natural - 20)
    assert panel.height() == natural - 20
    assert panel.preferred_height(900) == natural
    panel.fit_height(natural + 50)
    assert panel.height() == natural


def test_a_click_never_gives_the_panel_the_focus_but_tab_reaches_rows_and_eyes(make_page):
    """W18: the rows and their eyes are Tab stops (TabFocus); nothing takes click focus."""
    panel = make_page(show=False).panel
    tab_stops = {w for row in panel.rows.values() for w in (row, row.eye_button)}

    for widget in (panel, *panel.findChildren(QWidget)):
        wanted = Qt.FocusPolicy.TabFocus if widget in tab_stops else Qt.FocusPolicy.NoFocus
        assert widget.focusPolicy() == wanted, widget


Focus = Qt.FocusPolicy


@pytest.fixture
def page(make_editor, tmp_path):
    """The page on an image, active, the keys on its canvas."""
    project = new_project(tmp_path)
    editor = make_editor(project.a, project.folder)
    activate(editor)
    return editor


def tab_chain(editor, presses: int = 80) -> list[QWidget]:
    """The widgets Tab visits from the canvas, in order (until it comes back to the canvas)."""
    seen = []
    for _ in range(presses):
        QTest.keyClick(QApplication.focusWidget(), Qt.Key.Key_Tab)
        widget = QApplication.focusWidget()
        if widget is editor.graphics_view:
            break
        seen.append(widget)
    return seen


def pixels(widget) -> np.ndarray:
    settle_editor()
    image = widget.grab().toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    rows = np.frombuffer(image.constBits().asstring(image.sizeInBytes()), np.uint8)
    width = image.width()
    return rows.reshape(image.height(), image.bytesPerLine())[:, : width * 4].reshape(
        image.height(), width, 4
    )


def ring_changed(widget) -> float:
    """Share of the widget's outermost pixel frame that changed when it took the focus by Tab
    (a 1 px ring with rounded corners repaints most of it)."""
    before = pixels(widget)
    widget.setFocus(Qt.FocusReason.TabFocusReason)
    assert widget.hasFocus()
    after = pixels(widget)
    frame = np.ones(before.shape[:2], bool)
    frame[1:-1, 1:-1] = False
    return float((before != after).any(axis=2)[frame].mean())


def test_tab_reaches_each_row_then_its_eye_mask_first(page):
    rows = page.layer_panel.rows
    wanted = [rows[MASK], rows[MASK].eye_button, rows[SKELETON], rows[SKELETON].eye_button]

    chain = tab_chain(page)

    assert all(widget in chain for widget in wanted), chain
    start = chain.index(wanted[0])
    assert chain[start : start + 4] == wanted


@pytest.mark.parametrize("part", ["row", "eye"])
def test_the_focused_row_or_eye_draws_a_focus_ring(page, part):
    row = page.layer_panel.rows[SKELETON]
    widget = row if part == "row" else row.eye_button

    assert ring_changed(widget) > 0.5


@pytest.mark.parametrize("key", [Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter])
def test_space_or_enter_on_a_row_edits_its_layer(page, key):
    rows = page.layer_panel.rows
    rows[SKELETON].setFocus(Qt.FocusReason.TabFocusReason)

    QTest.keyClick(rows[SKELETON], key)

    assert page.active_layer_id == SKELETON and rows[SKELETON].is_active()
    rows[MASK].setFocus(Qt.FocusReason.TabFocusReason)
    QTest.keyClick(rows[MASK], key)
    assert page.active_layer_id == MASK


def test_space_on_an_eye_hides_then_shows_its_layer(page):
    eye = page.layer_panel.rows[MASK].eye_button
    eye.setFocus(Qt.FocusReason.TabFocusReason)

    QTest.keyClick(eye, Qt.Key.Key_Space)
    assert not page.layers[MASK].is_visible() and not eye.isChecked()
    QTest.keyClick(eye, Qt.Key.Key_Space)
    assert page.layers[MASK].is_visible() and eye.isChecked()


def test_the_page_keys_still_work_from_a_focused_row(page):
    row = page.layer_panel.rows[MASK]
    row.setFocus(Qt.FocusReason.TabFocusReason)

    QTest.keyClick(row, Qt.Key.Key_L)
    assert page.active_layer_id == SKELETON
    QTest.keyClick(row, Qt.Key.Key_H)
    assert not any(layer.is_visible() for layer in page.layers.values())


#: Realistic row labels, long and short, in the order the probe of S02-01 used.
LABELS = (
    (CLASSES, "error", "Class labels not shown"),
    (CLASSES, "none", "No class labels"),
    (SKELETON, "error", "Skeleton not shown"),
    (SKELETON, "none", "Prediction"),
    (CLASSES, "dirty", "Unsaved · remapped"),
    (CLASSES, "clean", "GT"),
    (CLASSES, "none", "No class labels"),
)


def _card_after_each_label(make_editor, project) -> list[tuple[str, int, int, list[str]]]:
    """For each label: ``(label, card width, card size hint, cut row labels)`` once the
    events have run."""
    editor = make_editor(project.a, project.folder, size=(1400, 900))
    settle_editor(editor)
    card, panel = editor.side_panels, editor.layer_panel
    seen = []
    for layer_id, state, text in LABELS:
        panel.set_status(layer_id, LayerStatus(state, text))
        run_events()  # what the page does: no extra relayout after the events
        cut = [
            row.state_label.text()
            for row in panel.rows.values()
            if row.state_label.isVisible()
            and row.state_label.width() < row.state_label.sizeHint().width()
        ]
        seen.append((text, card.width(), card.sizeHint().width(), cut))
    return seen


def test_the_card_takes_its_rows_width_after_every_label_change(
    make_editor, project, correct_guards
):
    seen = _card_after_each_label(make_editor, project)
    assert [(text, width, cut) for text, width, _hint, cut in seen] == [
        (text, hint, []) for text, _width, hint, _cut in seen
    ]
