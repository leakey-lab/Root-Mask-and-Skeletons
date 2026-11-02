"""The length dashboard's CSV lookup on the GUI thread: where Measure writes it, a few stats.

Visualize looks for ``root_lengths.csv`` on the GUI thread: on entry (the stage button,
Alt+5, the Length segment, the retry after a job end) in ``stages.missing_dashboard_data``,
and in the dashboard controller's ``_show`` and ``_is_up_to_date``. It looks in
``skeletons/``, then the project folder (S08-01 kept a walk of the folders for the older
layout off this thread; that layout is retired, #38, so there is no walk). A CSV in the
older layout's folders (``[output/]skeletonizer/test_latest/images``, a ``test_latest``
folder nearby) opens no dashboard.
"""

from __future__ import annotations

import os
import time

import pytest
from PyQt6.QtCore import QThreadPool, pyqtSignal
from PyQt6.QtWidgets import QApplication, QMainWindow

from app.config import LENGTH_CSV_HEADERS, ROOT_LENGTHS_CSV
from app.gui import visualization_manager as vm
from app.inference.metrics import write_metric_csv
from app.project_layout import ProjectLayout
from tests.qt_teardown import delete_widgets
from tests.qt_wait import settle as _settle
from tests.window_support import build_window, write_photos

STEM = "24EFCAM1_T001_L044_2024.07.18_111530_001_SMM"


def _lengths(folder) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    row = {
        "Image": STEM,
        "Tube": 1,
        "Position": 44,
        "Date": "2024.07.18",
        "Time": "11:15:30",
        LENGTH_CSV_HEADERS[5]: 9.25,
    }
    path = str(folder / ROOT_LENGTHS_CSV)
    write_metric_csv([row], path, LENGTH_CSV_HEADERS)
    return path


# ------------------------------------------------------------------------ the lookup
def test_the_lookup_reads_skeletons_then_the_project_folder(tmp_path):
    layout = ProjectLayout(str(tmp_path))
    missing = vm.LENGTH.find_csv(layout)
    assert missing.path is None and missing.title == "No root length data"
    assert os.path.join(layout.skeletons_dir, ROOT_LENGTHS_CSV) in missing.message

    base = _lengths(tmp_path)
    assert vm.LENGTH.find_csv(layout).path == base
    primary = _lengths(tmp_path / "skeletons")
    assert vm.LENGTH.find_csv(layout).path == primary


def test_the_area_lookup_reads_mask(tmp_path):
    layout = ProjectLayout(str(tmp_path))
    assert vm.AREA.find_csv(layout).title == "No mask data"
    (tmp_path / "output" / "mask").mkdir(parents=True)  # the older layout's folder: not read
    (tmp_path / "output" / "mask" / "root_areas.csv").write_text("Image\n", encoding="utf-8")
    assert vm.AREA.find_csv(layout).title == "No mask data"
    (tmp_path / "mask").mkdir()
    assert vm.AREA.find_csv(layout).title == "No root area data"
    (tmp_path / "mask" / "root_areas.csv").write_text("Image\n", encoding="utf-8")
    assert vm.AREA.find_csv(layout).path == str(tmp_path / "mask" / "root_areas.csv")


# ------------------------------------------------------------------------ the window
class StubLength(QMainWindow):
    """A length dashboard without Dash or a browser: the real one's signals and clean-up."""

    server_closed = pyqtSignal()
    export_saved = pyqtSignal(str)
    made: list = []

    def __init__(self, csv_path, *_args):
        super().__init__()
        self.csv_path = csv_path
        self.failed = False
        type(self).made.append(self)

    def cleanup_server(self) -> None:
        pass


_UNSET = object()


@pytest.fixture
def window(app, tmp_path, no_prompts, no_temp_sweep):
    """A shown window whose length dashboard is :class:`StubLength` (set on the module itself:
    a monkeypatch would import the Dash stack to read the old value)."""
    saved = vm.__dict__.get("RootLengthVisualization", _UNSET)
    vm.RootLengthVisualization = StubLength
    StubLength.made = []
    made = build_window(tmp_path / "s.ini", show=False)
    made.show()
    try:
        yield made
    finally:
        QThreadPool.globalInstance().waitForDone(5000)
        delete_widgets([made])
        if saved is _UNSET:
            vm.__dict__.pop("RootLengthVisualization", None)
        else:
            vm.RootLengthVisualization = saved


def _until(condition, timeout_s: float = 5.0) -> bool:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if condition():
            return True
        QApplication.processEvents()
    return condition()


def _idle() -> None:
    QThreadPool.globalInstance().waitForDone(5000)
    _settle()


def _project(root, *, skeletons: bool):
    write_photos(root, [STEM])
    if skeletons:
        write_photos(root / "skeletons", [f"{STEM}_fake"], suffix=".png")
    return root


def _open(window, root) -> None:
    """Open ``root`` and wait until the pool's search of its CSVs has reported."""
    window.open_path(str(root))
    assert _until(lambda: window._image_leaves())
    assert _until(lambda: not window.metrics.loading)
    _idle()


def _enter(window, stage: str) -> None:
    window.ribbon_buttons[stage].click()
    _settle()


def _blocked_texts(window) -> tuple[str, str, str]:
    page = window.visualization_widget
    assert page.is_blocked()
    return page.title_label.text(), page.body_label.text(), page.detail_label.text()


def test_visualize_without_a_length_csv_says_why(window, tmp_path):
    root = _project(tmp_path / "proj", skeletons=True)
    _open(window, root)

    _enter(window, "Visualize")
    window.action_bar_viz_seg.button("length").click()
    _settle()
    texts = _blocked_texts(window)
    window._on_jobs_busy_changed(False)  # a job ended while the page waits: it looks again
    _idle()

    assert (
        texts
        == _blocked_texts(window)
        == (
            "No root length measurements yet",
            "Run Measure → Measure Length to chart the root length of this trial.",
            f"Looked for {os.path.join(str(root), 'skeletons', ROOT_LENGTHS_CSV)}",
        )
    )
    assert StubLength.made == []


@pytest.mark.parametrize(
    "folder",
    [
        ("results", "run1", "test_latest", "images"),
        ("results", "run1", "test_latest"),
        ("output", "skeletonizer", "test_latest", "images"),
        ("skeletonizer", "test_latest", "images"),
    ],
    ids=["walked images", "walked", "output folder", "skeletonizer folder"],
)
def test_a_csv_of_the_older_layout_opens_no_dashboard(window, tmp_path, folder):
    root = _project(tmp_path / "proj", skeletons=False)
    _lengths(root.joinpath(*folder))
    _open(window, root)

    _enter(window, "Visualize")
    window.action_bar_viz_seg.button("length").click()
    _settle()

    assert _blocked_texts(window)[0] == "No root length measurements yet"
    assert StubLength.made == []


def test_a_csv_measure_writes_later_opens_the_dashboard_at_once(window, tmp_path):
    root = _project(tmp_path / "proj", skeletons=True)
    _open(window, root)
    _enter(window, "Visualize")
    assert StubLength.made == []
    _enter(window, "Measure")
    written = _lengths(root / "skeletons")  # the Measure run wrote it

    _enter(window, "Visualize")

    assert [os.path.normcase(d.csv_path) for d in StubLength.made] == [os.path.normcase(written)]
    assert window.right_panel.currentWidget() is StubLength.made[0]
