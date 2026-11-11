"""The metrics bar (``app/gui/metrics_bar.py``): it shows the values it is handed, read-only.

Promises: a value reads ``x.xx mm`` / ``x.xx mm²``; a cell with nothing to show is hidden
(never a dash, never an earlier value); a failed row shows ``failed`` with its message; a
value that is not a number shows a red dash (a CSV edited to hold ``nan`` showed "nan mm");
an outdated value (its input changed after the Measure run, or it was measured with another
field of view or length method) is shown in the warning colour with the reason in its
tooltip; each value's tooltip names what it was measured from; the FOV cell shows the shown
CSV's field of view, else the app's; separators sit only between the shown cells; the name,
STATUS and ZOOM cells never widen the window (a long name falls back to its tube, depth,
date and time); a value's style sheet is set only when its colour changes (a restyle
re-polishes the label on every Library click).

The cells are read through ``tests/metrics_bar_support.py`` (the bar has no accessor yet).
"""

from __future__ import annotations

import math

import pytest
from PyQt6.QtGui import QFontMetricsF
from PyQt6.QtWidgets import QApplication, QLabel

from app.gui import metrics_bar
from app.gui.key_text import platform_keys
from app.gui.library_index import stem_forms
from app.gui.metrics_bar import MetricsBar, _ElidedLabel
from app.gui.widgets import tokens
from tests.metrics_bar_support import cell, shown, stale, style, text, tooltip, value
from tests.qt_teardown import delete_widgets, flush_deleted

pytestmark = pytest.mark.usefixtures("app")


@pytest.fixture(autouse=True)
def _flush():
    """The bars a test deleteLater()s go before the next test."""
    yield
    flush_deleted()


@pytest.fixture
def bar(app):
    made = [MetricsBar()]
    yield made[0]
    delete_widgets(made, close=False)


def test_set_metrics_with_values(app):
    bar = MetricsBar()
    bar.set_metrics(length=12.5, area=6.7, status="Skeleton ready")
    assert text(bar, "length") == "12.50 mm"
    assert text(bar, "area") == "6.70 mm²"
    assert text(bar, "status") == "Skeleton ready"


def test_cells_with_nothing_to_show_are_hidden_not_left_with_an_earlier_value(bar):
    bar.set_metrics(length=4.0, area=2.0, status="x", name="img")
    assert all(shown(bar, name) for name in ("name", "length", "area", "status"))

    bar.set_metrics()  # another image, nothing measured, nothing selected

    assert not any(shown(bar, name) for name in ("name", "length", "area", "status"))
    assert "4.00" not in text(bar, "length") and "2.00" not in text(bar, "area")
    assert tooltip(bar, "length") == tooltip(bar, "area") == ""


def test_a_measured_zero_is_shown_not_taken_for_no_value(bar):
    """An image without roots measures 0: that is a value, not a missing one."""
    bar.set_metrics(length=0.0, area=0.0, name="img")

    assert shown(bar, "length") and text(bar, "length") == "0.00 mm"
    assert shown(bar, "area") and text(bar, "area") == "0.00 mm²"
    assert stale(bar, "length") is False and stale(bar, "area") is False


def test_measurement_error_is_shown_instead_of_a_value(app):
    """A failed Measure row shows 'failed' with the message, never a fabricated 0."""
    bar = MetricsBar()
    bar.set_metrics(length=None, area=None, error="Length: cannot read image")
    assert shown(bar, "length")
    assert text(bar, "length") == "failed"
    assert "cannot read image" in tooltip(bar, "length")


def test_a_failed_area_beside_a_measured_length_shows_failed_in_area(bar):
    bar.set_metrics(length=18.77, area=None, error="Area: cannot read the mask")

    assert text(bar, "length") == "18.77 mm"
    assert shown(bar, "area") and text(bar, "area") == "failed"
    assert tooltip(bar, "area") == "Measurement failed — Area: cannot read the mask"
    assert style(bar, "area") == metrics_bar._value_qss(tokens.DANGER)
    assert "failed" not in tooltip(bar, "length")


LENGTHS = "D:/p/skeletons/root_lengths.csv"


AREAS = "D:/p/mask/root_areas.csv"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_length_that_is_not_a_number_shows_a_dash(bar, bad):
    bar.set_metrics(length=bad, area=16.41, source=LENGTHS, area_source=AREAS)

    assert text(bar, "length") == "—"
    assert shown(bar, "length")
    assert tokens.DANGER in style(bar, "length")
    tip = tooltip(bar, "length")
    assert "root_lengths.csv" in tip and "no valid number" in tip
    assert text(bar, "area") == "16.41 mm²"  # the other value is untouched


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_an_area_that_is_not_a_number_shows_a_dash(bar, bad):
    bar.set_metrics(length=18.77, area=bad, source=LENGTHS, area_source=AREAS)

    assert text(bar, "area") == "—"
    assert shown(bar, "area")
    assert "nan" not in text(bar, "area") and "inf" not in text(bar, "area")
    tip = tooltip(bar, "area")
    assert "root_areas.csv" in tip and "no valid number" in tip
    assert text(bar, "length") == "18.77 mm"


def test_a_value_that_is_not_a_number_is_never_flagged_outdated(bar):
    bar.set_metrics(length=float("nan"), stale=True, source=LENGTHS)

    assert stale(bar, "length") is False
    assert "Measure again" not in tooltip(bar, "length")


def test_a_valid_value_afterwards_shows_normally(bar):
    bar.set_metrics(length=float("nan"), area=float("nan"), source=LENGTHS)
    bar.set_metrics(length=18.77, area=16.41, source=LENGTHS)

    assert text(bar, "length") == "18.77 mm"
    assert tokens.DANGER not in style(bar, "length")
    assert "no valid number" not in tooltip(bar, "length")
    assert text(bar, "area") == "16.41 mm²"


def test_each_value_has_its_own_flag_and_source(app):
    from app.gui.metrics_bar import MetricsBar

    bar = MetricsBar()
    bar.set_metrics(
        length=18.77,
        area=16.41,
        stale=False,
        source="D:/p/skeletons/root_lengths.csv",
        area_stale=True,
        area_source="D:/p/mask/root_areas.csv",
    )

    assert stale(bar, "length") is False
    assert stale(bar, "area") is True
    assert "root_lengths.csv" in tooltip(bar, "length")
    assert "Measure again" not in tooltip(bar, "length")
    assert "root_areas.csv" in tooltip(bar, "area")
    assert "root_lengths.csv" not in tooltip(bar, "area")
    assert "Measure again" in tooltip(bar, "area")
    bar.deleteLater()


def test_each_tooltip_names_a_gt_file_that_was_not_used(app):
    from app.gui.metrics_bar import MetricsBar

    bar = MetricsBar()
    bar.set_metrics(length=18.77, area=16.41, length_note="A GT skeleton exists")
    assert "A GT skeleton exists" in tooltip(bar, "length")
    assert "GT" not in tooltip(bar, "area")
    bar.set_metrics(length=18.77, area=16.41, area_note="A GT mask exists")
    assert "A GT mask exists" in tooltip(bar, "area")
    assert "GT" not in tooltip(bar, "length")
    bar.set_metrics(length=18.77, area=16.41)
    assert "GT" not in tooltip(bar, "length") + tooltip(bar, "area")
    bar.deleteLater()


def test_one_stale_flag_still_flags_both_values(app):
    """Callers that pass one flag (and one source) keep today's meaning."""
    from app.gui.metrics_bar import MetricsBar

    bar = MetricsBar()
    bar.set_metrics(length=1.0, area=2.0, stale=True, source="D:/p/x.csv")
    assert stale(bar, "length") is True
    assert stale(bar, "area") is True
    assert "x.csv" in tooltip(bar, "area")
    bar.deleteLater()


def test_a_method_note_marks_the_value_outdated(app):
    """O3: a value measured with another field of view (or length method) is flagged, not
    hidden; its tooltip gets the note as a line of its own."""
    note = (
        "Measured with field of view 18×13 mm; this project uses 18×13.5 mm. "
        "Measure again before comparing with new values."
    )
    bar = MetricsBar()
    bar.set_metrics(length=18.77, area=16.41, method_notes={"length": "", "area": note})
    assert text(bar, "area") == "16.41 mm²"
    assert stale(bar, "area") is True
    assert note in tooltip(bar, "area").split("\n")
    assert "changed after this measurement" not in tooltip(bar, "area")
    assert stale(bar, "length") is False
    assert note not in tooltip(bar, "length")
    bar.set_metrics(length=18.77, area=16.41)
    assert stale(bar, "area") is False


def test_fov_cell_is_derived_from_calibration(app, monkeypatch):
    """PG-12 / AR-06: the FOV text comes from app.config.CALIBRATION (single source)."""
    from app import config
    from app.gui import metrics_bar

    assert text(MetricsBar(), "fov") == "18×13.5 mm"
    monkeypatch.setattr(config, "CALIBRATION", config.Calibration(20.5, 13.0))
    assert metrics_bar.fov_text() == "20.5×13 mm"
    assert text(MetricsBar(), "fov") == "20.5×13 mm"


def test_fov_text_formats_a_given_calibration(app):
    """M2: a CSV's own recorded calibration is shown with the bar's text, not the app's."""
    from app.gui import metrics_bar

    cal = {"fov_width_mm": 18.0, "fov_height_mm": 13.0}
    assert metrics_bar.fov_text(cal) == "18×13 mm"
    assert metrics_bar.fov_text() == "18×13.5 mm"
    bar = MetricsBar()
    bar.set_fov("18×13 mm", "x")
    assert text(bar, "fov") == "18×13 mm"
    assert tooltip(bar, "fov") == "x"
    bar.set_fov("18×13.5 mm")
    assert tooltip(bar, "fov") == ""


def _separated(bar: MetricsBar) -> list[str]:
    """Names of the shown cells, each with ``|`` in front when it draws a separator."""
    cells = [
        ("name", cell(bar, "name")),
        ("length", cell(bar, "length")),
        ("area", cell(bar, "area")),
        ("fov", cell(bar, "fov")),
        ("status", cell(bar, "status")),
    ]
    return [
        ("|" if "border-left: 1px" in cell.styleSheet() else "") + name
        for name, cell in cells
        if not cell.isHidden()
    ]


def test_without_a_selection_the_first_cell_has_no_separator(app):
    bar = MetricsBar()
    bar.set_metrics()
    assert _separated(bar) == ["fov"]
    bar.deleteLater()


def test_separators_sit_between_the_shown_cells(app):
    bar = MetricsBar()
    bar.set_metrics(length=1.5, area=2.5, status="Mask ready", name="img")
    assert _separated(bar) == ["name", "|length", "|area", "|fov", "|status"]
    bar.set_metrics(status="No mask", name="img")
    assert _separated(bar) == ["name", "|fov", "|status"]
    bar.set_metrics()
    assert _separated(bar) == ["fov"]
    bar.deleteLater()


@pytest.fixture
def restyled(monkeypatch):
    """The value labels given a style sheet, in order."""
    seen: list[QLabel] = []
    real = QLabel.setStyleSheet

    def spy(label, qss):
        seen.append(label)
        real(label, qss)

    monkeypatch.setattr(QLabel, "setStyleSheet", spy)
    return seen


def test_the_same_colour_is_not_set_again(bar, restyled):
    bar.set_metrics(length=12.5, area=3.25, status="Mask · Skeleton", name="img")
    first = list(restyled)
    restyled.clear()
    bar.set_metrics(length=13.0, area=4.0, status="Mask · Skeleton", name="other")

    assert first == [] or set(first) <= {value(bar, "length"), value(bar, "area")}
    assert restyled == []  # values change, colours do not
    assert text(bar, "length") == "13.00 mm"
    assert style(bar, "length") == metrics_bar._value_qss(tokens.ACCENT)
    assert style(bar, "area") == metrics_bar._value_qss(tokens.TEXT)


def test_a_colour_change_restyles_the_value(bar, restyled):
    bar.set_metrics(length=12.5, area=3.25, name="img")
    restyled.clear()

    bar.set_metrics(length=12.5, area=3.25, name="img", stale=True, area_stale=False)
    assert restyled == [value(bar, "length")]
    assert style(bar, "length") == metrics_bar._value_qss(tokens.WARN)
    assert stale(bar, "length") is True

    restyled.clear()
    bar.set_metrics(length=12.5, area=3.25, name="img")
    assert restyled == [value(bar, "length")]
    assert style(bar, "length") == metrics_bar._value_qss(tokens.ACCENT)
    assert stale(bar, "length") is False


def test_a_failed_value_and_back_restyles_both_ways(bar, restyled):
    bar.set_metrics(length=12.5, name="img")
    restyled.clear()
    bar.set_metrics(length=None, name="img", error="unreadable")
    assert text(bar, "length") == "failed"
    assert style(bar, "length") == metrics_bar._value_qss(tokens.DANGER)
    bar.set_metrics(length=12.5, name="img")
    assert style(bar, "length") == metrics_bar._value_qss(tokens.ACCENT)
    assert restyled == [value(bar, "length"), value(bar, "length")]


LONG_NAME = "24EFCAM1_T001_L011_2024.07.18_111618_001_SMM_with_a_very_long_suffix_rep2"


def _bar_with_everything() -> MetricsBar:
    bar = MetricsBar()
    bar.set_metrics(length=123.45, area=67.89, status="Skeleton ready", name=LONG_NAME)
    return bar


def test_long_name_does_not_widen_the_bar_minimum():
    bar = _bar_with_everything()
    short = MetricsBar()
    short.set_metrics(length=123.45, area=67.89, status="Skeleton ready", name="img")
    # The long name may cost its elided stub (the label's MIN_WIDTH in place of the short
    # name's own width, whatever the font) but never its full width.
    stub = max(0, _ElidedLabel.MIN_WIDTH - value(short, "name").minimumSizeHint().width())
    assert bar.minimumSizeHint().width() <= short.minimumSizeHint().width() + stub


def test_name_keeps_full_text_and_tooltip_when_elided():
    bar = _bar_with_everything()
    bar.resize(520, MetricsBar.HEIGHT)
    bar.show()
    QApplication.processEvents()
    assert text(bar, "name") == LONG_NAME
    assert value(bar, "name").toolTip() == LONG_NAME
    bar.close()


STEM = "24EFCAM1_T001_L044_2024.07.18_111530_001_SMM"


def _name_label(app, name: str, width: int):
    """The metrics bar's name label showing ``name``, ``width`` px of contents wide."""
    bar = MetricsBar()
    bar.set_metrics(name=name)
    label = value(bar, "name")
    label.setParent(None)
    label.setFixedWidth(width + label.contentsMargins().left() + label.contentsMargins().right())
    label.show()
    QApplication.processEvents()
    bar.deleteLater()
    return label


def test_the_name_cell_falls_back_to_tube_depth_date_and_time(app):
    forms = stem_forms(STEM)
    probe = _name_label(app, STEM, 2000)
    # The label's own font; the fractional advance rounded up, as a font with fractional
    # advances (DejaVu on Linux) elides a text that exactly fills its rounded advance when the
    # fraction is under .5.
    metrics = QFontMetricsF(probe.font())

    def advance(text: str) -> int:
        return math.ceil(metrics.horizontalAdvance(text))

    assert probe._elided() == STEM and probe.toolTip() == ""  # room for the whole name
    probe.deleteLater()
    for i, form in enumerate(forms[1:], start=1):
        label = _name_label(app, STEM, advance(form))
        assert label.text() == STEM and label.toolTip() == STEM
        assert label._elided() == form, (i, label._elided())
        label.deleteLater()
    tiny = _name_label(app, STEM, advance(forms[-1]) - 4)
    assert tiny._elided().startswith("T1") and tiny._elided().endswith("…")
    tiny.deleteLater()


def test_an_unparsed_name_still_elides_in_the_middle(app):
    name = "an_image_name_without_tube_or_depth_" + "x" * 80
    probe = _name_label(app, name, 2000)
    half = probe.fontMetrics().horizontalAdvance(name) // 2
    probe.deleteLater()
    label = _name_label(app, name, half)
    painted = label._elided()
    assert "…" in painted and painted.startswith("an_") and painted.endswith("x")
    assert label.toolTip() == name
    label.deleteLater()


def test_the_zoom_cell_formats_the_factor(app):
    bar = MetricsBar()
    bar.set_metrics(name="img", status="No mask")
    bar.set_zoom(1.0)
    assert shown(bar, "zoom") and text(bar, "zoom") == "100 %"
    bar.set_zoom(4.3467)
    assert text(bar, "zoom") == "435 %"
    bar.set_zoom(0.084)
    assert text(bar, "zoom") == "8.4 %"
    assert platform_keys("Ctrl+0") in tooltip(bar, "zoom")
    bar.set_metrics()  # nothing selected: no zoom to report
    assert not shown(bar, "zoom")
    bar.deleteLater()


def test_the_zoom_cell_never_widens_the_window_and_goes_first_when_narrow(app):
    """The window fits 900 px (UXS-04): the ZOOM cell adds nothing to the bar's minimum and
    hides while the other cells need its room."""
    bar = MetricsBar()
    bar.set_metrics(
        length=1234.56,
        area=987.65,
        status="Skeleton corrected",
        name="24EFCAM1_T001_L044_2024.07.18_111530_001_SMM",
    )
    bar.show()  # a hidden widget gets its resize events only when it is shown
    without = bar.minimumSizeHint().width()
    bar.resize(without + 400, bar.height())
    bar.set_zoom(4.35)
    assert shown(bar, "zoom")
    assert bar.minimumSizeHint().width() == without

    bar.resize(without, bar.height())  # no room for it
    assert not shown(bar, "zoom")
    bar.resize(without + 400, bar.height())
    assert shown(bar, "zoom")
    bar.close()
    bar.deleteLater()
