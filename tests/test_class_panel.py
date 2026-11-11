"""``ClassPanel`` (``app/gui/class_panel.py``), the floating panel of the class labels.

One row per active class in registry order with its key hint (1-9, 0 = Ignore); the one tick
and the highlight are the active class alone, set by ``set_active`` (silent) or by a click on a
row or a box (``active_changed``); a box never changes itself and a toggle is put back (a toggle
on picks the class); ``picked_by_key`` tells Space from a click; the rows are rebuilt at once on
a new registry; long names elide at the right with the full name as the tooltip, and the panel
widens for them; the height is capped at 60 % of the host and the rows scroll; the key hints keep
WCAG AA contrast on the panel's own fill; the gear asks for the class editor.
"""

from __future__ import annotations

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QScrollArea,
    QWidget,
)

from app.gt.registry import default_registry
from app.gui.class_panel import ClassPanel
from app.gui.widgets import tokens
from tests.qt_teardown import delete_widgets

_MADE: list[QWidget] = []  # the hosts and dialogs a test made: deleted after it


# ClassPanel has no public accessor for a class's row, box or labels, nor for the highlighted row
# (recorded for the owner: ``box(key)``, ``keys()``, ``highlighted_keys()``). These helpers are
# the one place that reads its private parts: the day the accessors exist, only they change.
def rows_of(panel) -> dict:
    """The rows shown, by class key, in display order."""
    return panel._rows


def row_of(panel, key):
    return panel._rows[key]


def box_of(panel, key) -> QCheckBox:
    """The box of class ``key`` (ticked: the active class)."""
    return panel._rows[key].checkbox


def is_highlighted(panel, key) -> bool:
    return panel._rows[key]._active


def row_highlighted(row) -> bool:
    return row._active


def name_label(panel, key):
    return panel._rows[key]._name_label


def row_name_label(row):
    return row._name_label


def hint_label(panel, key):
    return panel._rows[key]._number_label


def gear(panel):
    return panel._gear


def scroll_area(panel) -> QScrollArea:
    return panel._scroll


def rows_container(panel) -> QWidget:
    return panel._rows_container


@pytest.fixture(autouse=True)
def _delete_test_widgets(app):
    """Close and delete the test's hosts and dialogs, and drop every dead wrapper, before
    the next test runs (see tests/qt_teardown.py)."""
    yield
    delete_widgets(_MADE)


def _host(w=400, h=600):
    host = QWidget()
    _MADE.append(host)
    host.resize(w, h)
    return host


def _registry_with_many_classes(n_labelled: int = 12):
    reg = default_registry()
    for i in range(n_labelled - len(reg.labelled())):
        reg = reg.add(key=f"extra_{i}", name=f"Extra {i}", color=f"#{100 + i:06x}")
    return reg


@pytest.fixture
def made(app):
    """Widgets closed and deleted deterministically after the test."""
    widgets: list = []
    yield widgets
    delete_widgets(widgets)


def test_panel_shows_one_row_per_active_class_in_order(app):
    host = _host()
    panel = ClassPanel(host)
    panel.set_registry(default_registry())

    assert list(rows_of(panel).keys()) == ["root", "nodule", "root_tip", "ignore"]
    # Every row has a box, Ignore too (it can be the active class: key 0).
    tips = {key: row.checkbox.toolTip() for key, row in rows_of(panel).items()}
    assert tips == {
        "root": "Paint with this class (key 1)",
        "nodule": "Paint with this class (key 2)",
        "root_tip": "Paint with this class (key 3)",
        "ignore": "Paint with this class (key 0)",
    }


def test_clicking_a_row_emits_active_changed_and_highlights(app):
    host = _host()
    panel = ClassPanel(host)
    panel.set_registry(default_registry())

    got = []
    panel.active_changed.connect(got.append)
    QTest.mouseClick(row_of(panel, "nodule"), Qt.MouseButton.LeftButton)

    assert got == ["nodule"]
    assert panel.active_key() == "nodule"
    assert is_highlighted(panel, "nodule") is True

    # set_active() must not emit.
    got.clear()
    panel.set_active("root")
    assert got == []
    assert panel.active_key() == "root"
    assert is_highlighted(panel, "root") is True
    assert is_highlighted(panel, "nodule") is False


def _highlighted(panel) -> list[str]:
    return [key for key, row in rows_of(panel).items() if row_highlighted(row)]


def test_the_tick_and_the_highlight_are_the_active_class_alone(app):
    """Owner bug 2026-11-01: Root nodule was ticked while Tip was highlighted (and painted).
    The tick and the highlight now come from the active key alone."""
    host = _host()
    panel = ClassPanel(host)
    panel.set_registry(default_registry())
    assert panel.ticked_keys() == [] == _highlighted(panel)  # no class set yet

    panel.set_active("nodule")
    assert panel.ticked_keys() == ["nodule"] == _highlighted(panel)

    got = []
    panel.active_changed.connect(got.append)
    QTest.mouseClick(row_of(panel, "root_tip"), Qt.MouseButton.LeftButton)  # Tip's row

    assert got == ["root_tip"] and panel.active_key() == "root_tip"
    assert panel.ticked_keys() == ["root_tip"] == _highlighted(panel)


def test_a_click_on_a_box_picks_its_class_and_is_the_only_tick(app):
    host = _host()
    panel = ClassPanel(host)
    panel.set_registry(default_registry())
    panel.set_active("root")
    got = []
    panel.active_changed.connect(got.append)

    QTest.mouseClick(box_of(panel, "nodule"), Qt.MouseButton.LeftButton)

    assert got == ["nodule"] and panel.active_key() == "nodule"
    assert panel.ticked_keys() == ["nodule"] == _highlighted(panel)
    assert box_of(panel, "nodule").isEnabled() is True  # a box is never locked


def test_a_click_on_the_ticked_box_keeps_it_ticked(app):
    host = _host()
    panel = ClassPanel(host)
    panel.set_registry(default_registry())
    panel.set_active("nodule")
    got = []
    panel.active_changed.connect(got.append)

    for _ in range(2):  # not an off switch: a second click changes nothing either
        QTest.mouseClick(box_of(panel, "nodule"), Qt.MouseButton.LeftButton)
        assert panel.ticked_keys() == ["nodule"] == _highlighted(panel)
    assert got == ["nodule", "nodule"]


def test_the_ignore_box_can_be_the_tick(app):
    host = _host()
    panel = ClassPanel(host)
    panel.set_registry(default_registry())
    panel.set_active("root")

    QTest.mouseClick(box_of(panel, "ignore"), Qt.MouseButton.LeftButton)

    assert panel.active_key() == "ignore"
    assert panel.ticked_keys() == ["ignore"] == _highlighted(panel)


def test_set_active_moves_the_one_tick_silently(app):
    host = _host()
    panel = ClassPanel(host)
    panel.set_registry(default_registry())
    got = []
    panel.active_changed.connect(got.append)

    for key in ("root", "ignore", "root_tip", "root_tip"):
        panel.set_active(key)
        assert panel.ticked_keys() == [key] == _highlighted(panel)
    assert got == []


def test_set_active_of_the_shown_class_changes_nothing(app):
    """Every stroke and undo puts the active class on the panel again: the box it shows
    must not untick and tick again (two toggles, two restyles each time)."""
    host = _host()
    panel = ClassPanel(host)
    panel.set_registry(default_registry())
    panel.set_active("nodule")
    toggles = []
    for row in rows_of(panel).values():
        row.checkbox.toggled.connect(lambda on, k=row.key: toggles.append((k, on)))

    panel.set_active("nodule")
    assert toggles == []

    panel.set_active("root")
    assert sorted(toggles) == [("nodule", False), ("root", True)]


def test_a_toggle_never_changes_a_box_and_a_toggle_on_picks_its_class(app):
    """Qt's accessible Toggle action (Narrator, NVDA) calls toggle(), which skips the
    click path: the active box stays ticked, and another box picks its class."""
    host = _host()
    panel = ClassPanel(host)
    panel.set_registry(default_registry())
    panel.set_active("root")
    got = []
    panel.active_changed.connect(lambda k: got.append((k, panel.picked_by_key())))

    box_of(panel, "root").toggle()  # off: put back, nothing asked
    assert panel.ticked_keys() == ["root"] == _highlighted(panel) and got == []
    box_of(panel, "root").setChecked(False)
    assert panel.ticked_keys() == ["root"] and got == []

    box_of(panel, "nodule").toggle()  # on: asks for the class, which ticks it alone

    assert got == [("nodule", True)] and panel.active_key() == "nodule"
    assert panel.ticked_keys() == ["nodule"] == _highlighted(panel)
    assert panel.picked_by_key() is False  # only while the pick is handled


def test_picked_by_key_tells_space_from_a_click(app):
    host = _host()
    panel = ClassPanel(host)
    panel.set_registry(default_registry())
    panel.set_active("root")
    got = []
    panel.active_changed.connect(lambda k: got.append((k, panel.picked_by_key())))
    nodule, root_tip = box_of(panel, "nodule"), box_of(panel, "root_tip")

    QTest.mouseClick(nodule, Qt.MouseButton.LeftButton)
    QTest.keyClick(root_tip, Qt.Key.Key_Space)
    QTest.mouseClick(row_of(panel, "root"), Qt.MouseButton.LeftButton)

    assert got == [("nodule", False), ("root_tip", True), ("root", False)]
    assert panel.ticked_keys() == ["root"]


def test_one_tick_after_the_classes_change(app):
    """A rebuild keeps the active class ticked; a retired active class leaves no stale tick
    (the page then sets the class it falls back to)."""
    host = _host()
    panel = ClassPanel(host)
    registry = default_registry()
    panel.set_registry(registry)
    panel.set_active("nodule")

    registry = registry.add(key="hypha", name="Hypha", color="#3fa7ff")
    panel.set_registry(registry)
    assert panel.ticked_keys() == ["nodule"] == _highlighted(panel)

    panel.set_registry(registry.retire("nodule"))
    assert panel.ticked_keys() == [] == _highlighted(panel)
    panel.set_active("root")
    assert panel.ticked_keys() == ["root"] == _highlighted(panel)


def test_key_for_shortcut_defaults(app):
    host = _host()
    panel = ClassPanel(host)
    panel.set_registry(default_registry())

    assert panel.key_for_shortcut(1) == "root"
    assert panel.key_for_shortcut(3) == "root_tip"
    assert panel.key_for_shortcut(0) == "ignore"
    assert panel.key_for_shortcut(9) is None


def test_set_registry_drops_old_rows_at_once(app):
    """Rebuilt rows replace the old ones immediately (no stale highlighted row)."""
    host = _host()
    panel = ClassPanel(host)
    host.show()
    try:
        panel.set_registry(default_registry())
        panel.set_active("root")
        QApplication.processEvents()
        old_rows = list(rows_of(panel).values())

        panel.set_registry(default_registry())
        rows = panel.findChildren(QFrame, "classRow")
        assert len(rows) == 4, len(rows)
        assert all(r.isHidden() and r.parent() is None for r in old_rows)

        QApplication.processEvents()
        visible = [r for r in panel.findChildren(QFrame, "classRow") if r.isVisible()]
        assert len(visible) == 4
        assert [r for r in visible if row_highlighted(r)] == [row_of(panel, "root")]
    finally:
        host.close()


def test_gear_button_emits_edit_requested(app):
    host = _host()
    panel = ClassPanel(host)
    panel.set_registry(default_registry())

    got = []
    panel.edit_requested.connect(lambda: got.append(1))
    gear(panel).click()
    assert got == [1]


def test_panel_many_classes_scrolls(app):
    host = _host(400, 600)
    panel = ClassPanel(host)
    # Enough rows that the natural (unscrolled) content height genuinely
    # exceeds the 60%-of-600 = 360px cap, so this exercises real clipping,
    # not just an assertion that happens to hold for a short list.
    reg = _registry_with_many_classes(25)
    panel.set_registry(reg)

    assert isinstance(scroll_area(panel), QScrollArea)
    assert scroll_area(panel).widget() is rows_container(panel)
    assert len(rows_of(panel)) == 26  # 25 labelled + Ignore

    panel.reposition()
    cap = int(host.height() * 0.6)
    assert panel.height() <= cap
    assert rows_container(panel).sizeHint().height() > scroll_area(panel).height(), (
        "content must be taller than the visible scroll viewport for this to be a real test "
        "of scrolling/clipping, not a no-op cap"
    )
    assert panel.key_for_shortcut(9) == reg.labelled()[8].key


def test_panel_shrinks_back_after_fewer_classes(app):
    """Regression: QScrollArea.sizeHint() caches the contained widget's size
    and does not reliably invalidate without a real show/resize event cycle,
    so a naive adjustSize()-based reposition() can get stuck at the capped
    height even after the registry shrinks back down (e.g. retiring classes
    via the editor). reposition() must measure the rows container directly.
    """
    host = _host(400, 600)
    panel = ClassPanel(host)
    panel.set_registry(_registry_with_many_classes(25))
    panel.reposition()
    grown_height = panel.height()

    panel.set_registry(default_registry())
    panel.reposition()
    assert panel.height() < grown_height
    assert panel.height() <= int(host.height() * 0.6)


def test_class_panel_prefers_all_rows_up_to_60_percent_of_the_host(app):
    """``preferred_height`` is the height the standalone placement uses."""
    host = _host(400, 600)
    panel = ClassPanel(host)
    panel.set_registry(default_registry())
    short = panel.preferred_height(600)
    panel.reposition()
    assert panel.height() == short < 360

    panel.set_registry(_registry_with_many_classes(25))
    assert panel.preferred_height(600) == 360
    assert panel.preferred_height(400) == 240


def test_class_panel_fits_a_smaller_height_and_scrolls(app):
    """An open Enhance popover leaves the panel less room; its rows scroll (UXE-09)."""
    host = _host(400, 600)
    panel = ClassPanel(host)
    panel.set_registry(_registry_with_many_classes(25))
    panel.fit_height(200)
    assert panel.height() == 200
    assert rows_container(panel).sizeHint().height() > scroll_area(panel).height()


def test_a_long_class_name_is_elided_at_its_end(app):
    """The start of the name stays (it is what tells the classes apart); the full name is
    the tooltip."""
    host = _host(900, 600)
    name = "Mycorrhizal hyphae and fine roots"
    panel = ClassPanel(host)
    panel.set_registry(default_registry().add(key="long", name=name, color="#123456"))
    host.show()
    panel.reposition()
    QApplication.processEvents()
    label = name_label(panel, "long")
    shown = label.elided_text()
    assert shown.endswith("…") and name.startswith(shown[:-1])
    assert label.is_elided() and label.toolTip() == name and label.text() == name


def _relative_luminance(color: QColor) -> float:
    def channel(c: float) -> float:
        c /= 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    return (
        0.2126 * channel(color.red())
        + 0.7152 * channel(color.green())
        + 0.0722 * channel(color.blue())
    )


def _contrast(a: QColor, b: QColor) -> float:
    la, lb = sorted((_relative_luminance(a), _relative_luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def test_contrast_helper_matches_wcag_reference_values():
    assert round(_contrast(QColor("#000000"), QColor("#ffffff")), 1) == 21.0
    assert round(_contrast(QColor("#777777"), QColor("#ffffff")), 2) == 4.48


def test_class_panel_paints_its_own_background(app):
    """UXE-04: the rows used to be drawn straight on the soil image (no panel fill)."""
    host = _host(900, 600)
    panel = ClassPanel(host)
    panel.set_registry(default_registry())
    host.show()
    try:
        QTest.qWaitForWindowExposed(host)
        panel.reposition()
        # A pixel inside the 6 px content margin, clear of the rounded corners.
        pixel = panel.grab().toImage().pixelColor(3, panel.height() // 2)
        assert pixel.alpha() == 255
        assert pixel.name() == QColor(tokens.BG_1).name()
    finally:
        host.close()


def test_class_key_hints_are_readable_on_the_panel(app):
    """UXE-04: the 1-9/0 hints need WCAG AA contrast (4.5:1) on the panel fill."""
    host = _host()
    panel = ClassPanel(host)
    panel.set_registry(default_registry())
    hint = hint_label(panel, "root")
    hint.ensurePolished()
    color = hint.palette().color(QPalette.ColorRole.WindowText)
    assert _contrast(color, QColor(tokens.BG_1)) >= 4.5


def _long_named_registry(n=12):
    reg = default_registry()
    for i in range(n - len(reg.labelled())):
        reg = reg.add(
            key=f"extra_{i}", name=f"Extra class number {i}", color=f"#{0x40A0C0 + 7 * i:06x}"
        )
    return reg


@pytest.mark.parametrize("n", [4, 12, 25])
def test_long_class_names_elide_and_every_box_shows(made, n):
    host = QWidget()
    made.append(host)
    host.resize(1000, 700)
    panel = ClassPanel(host)
    reg = _long_named_registry(n)
    panel.set_registry(reg)
    host.show()
    QTest.qWaitForWindowExposed(host)
    panel.reposition()
    QApplication.processEvents()

    scroll = scroll_area(panel)
    assert not scroll.horizontalScrollBar().isVisible()
    viewport = scroll.viewport()
    for key, class_row in rows_of(panel).items():
        assert row_name_label(class_row).text() == reg.by_key(key).name
        if class_row.checkbox is None:
            continue
        right = class_row.mapTo(viewport, class_row.checkbox.geometry().topRight()).x()
        assert right < viewport.width(), f"{key}: box at x={right}, viewport {viewport.width()}"
    assert panel.width() <= 280
    for class_row in rows_of(panel).values():
        label = row_name_label(class_row)
        assert label.toolTip() == (label.text() if label.is_elided() else "")


def test_a_forty_character_name_elides_with_the_full_name_as_tooltip(made):
    host = QWidget()
    made.append(host)
    host.resize(1000, 700)
    panel = ClassPanel(host)
    name = "A very long class name that goes on, and"  # 40 characters, the most allowed
    assert len(name) == 40
    panel.set_registry(default_registry().add(key="long_one", name=name, color="#123456"))
    host.show()
    QTest.qWaitForWindowExposed(host)
    panel.reposition()
    QApplication.processEvents()
    label = name_label(panel, "long_one")
    assert label.is_elided() and label.toolTip() == name and label.text() == name
    checkbox = box_of(panel, "long_one")
    viewport = scroll_area(panel).viewport()
    right = row_of(panel, "long_one").mapTo(viewport, checkbox.geometry().topRight()).x()
    assert right < viewport.width()


def test_the_panel_widens_for_longer_names_after_a_registry_change(made):
    host = QWidget()
    made.append(host)
    host.resize(1000, 700)
    panel = ClassPanel(host)
    panel.set_registry(default_registry())
    host.show()
    QTest.qWaitForWindowExposed(host)
    panel.reposition()
    QApplication.processEvents()
    narrow = panel.width()

    def swap(registry):
        panel.set_registry(registry)
        panel.reposition()
        QApplication.processEvents()
        panel.reposition()
        QApplication.processEvents()

    swap(_long_named_registry(12))
    assert panel.width() > narrow
    label = name_label(panel, "extra_0")
    assert label.width() == label.sizeHint().width()

    swap(default_registry())
    assert panel.width() == narrow
