"""The metrics controller (``app/gui/window/metrics.py``): the bar shows the selected image's
row of the Measure CSVs, and nothing the CSVs do not hold.

Promises: the values are the CSV's (never computed); an image without a row shows no value;
a CSV not parsed yet shows nothing until a thread-pool parse reports, and the GUI thread
never parses nor waits for a parse; one parse at a time, and a parse of a replaced project
refreshes nothing; each value is judged against its own inputs (a saved mask outdates AREA,
never LENGTH); the tooltips name the measured file and its CSV, and a GT file the run did
not use; the FOV cell shows the field of view recorded with the shown value's CSV.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from PyQt6 import sip
from PyQt6.QtCore import QEvent, QThreadPool
from PyQt6.QtWidgets import QApplication

from app.config import AREA_CSV_HEADERS, LENGTH_CSV_HEADERS
from app.data_processing.measurement_index import Measurement
from app.gui.metrics_bar import MetricsBar
from app.gui.window import metrics
from app.gui.window.metrics import MetricsController
from app.inference.metrics import write_metric_csv
from tests.metrics_bar_support import shown, stale, text, tooltip
from tests.qt_teardown import delete_widgets, flush_deleted
from tests.qt_wait import wait_until as _wait
from tests.window_support import build_window, open_folder, quiet_window, write_photos

pytestmark = pytest.mark.usefixtures("no_temp_sweep")


class Bar:
    def __init__(self) -> None:
        self.shown: list[dict] = []

    def set_metrics(self, **values) -> None:
        self.shown.append(values)

    def set_fov(self, text: str, tooltip: str = "") -> None:
        pass


class Index:
    """A measurement index stand-in: one measured image, counters for what was asked."""

    root = None  # no project file is read: the default field of view (O3)

    def __init__(self, *, needs_load: bool = False) -> None:
        self._needs_load = needs_load
        self.asked = self.prewarmed = 0

    def lookup(self, _name, load=False):
        return Measurement(12.5, 3.25, None)

    def needs_load(self) -> bool:
        self.asked += 1
        return self._needs_load

    def outdated(self, _name) -> dict:
        return {"length": False, "area": True}

    def sources(self) -> dict:
        return {"length": "root_lengths.csv", "area": "root_areas.csv"}

    def by_content(self) -> dict:
        return {"length": True, "area": False}

    def calibrations(self) -> dict:
        return {"length": None, "area": None}

    def method_notes(self, _current) -> dict:
        return {"length": "", "area": ""}

    def gt_skip_reason(self, _name) -> dict:
        return {}

    def measured_from(self, _name) -> dict:
        return {"length": ("skeletons/a_fake.png", "prediction"), "area": ("gt/mask/a.png", "gt")}

    def prewarm(self) -> None:
        self.prewarmed += 1


def metrics_window(*, bar=None, stage=None, checks=None) -> SimpleNamespace:
    checks = [] if checks is None else checks
    window = SimpleNamespace(
        image_manager=SimpleNamespace(
            has_mask=lambda name: name == "a",
            has_skeleton=lambda _name: False,
            has_gt_skeleton=lambda _name: False,
            mask_state=lambda name: "prediction" if name == "a" else "none",
            skeleton_state=lambda _name: "none",
            has_gt_labels=lambda _name: False,
            original_folder=None,
            images={},
        ),
        display_controller=SimpleNamespace(current_name="a"),
        stage_status=SimpleNamespace(check=lambda *a, **k: checks.append((a, k))),
        _current_stage=lambda: stage,
    )
    if bar is not None:
        window.metrics_bar = bar
    return window


class Pool:
    """The thread pool, recording the jobs it is given (run by hand)."""

    def __init__(self) -> None:
        self.jobs: list = []

    def globalInstance(self):  # noqa: N802 - Qt API name
        return self

    def start(self, job) -> None:
        self.jobs.append(job)


def test_the_bar_needs_a_metrics_bar():
    controller = MetricsController(metrics_window())
    controller.index = Index()
    controller.refresh("a")  # no metrics bar yet (before init_ui): nothing to update
    assert controller.index.asked == 0


@pytest.fixture
def real_bar(app):
    made = [MetricsBar()]
    yield made[0]
    delete_widgets(made, close=False)


def test_the_bar_shows_the_status_and_the_measured_values(real_bar):
    bar = real_bar
    controller = MetricsController(metrics_window(bar=bar))
    controller.index = Index()  # LENGTH by content and current, AREA by file time, outdated
    controller.refresh("a")

    assert text(bar, "name") == "a"
    assert text(bar, "status") == "Mask: prediction · No skeleton · No classes"
    assert text(bar, "length") == "12.50 mm" and stale(bar, "length") is False
    assert tooltip(bar, "length") == (
        "Measured value from skeletons/a_fake.png (prediction), recorded in root_lengths.csv"
    )
    assert text(bar, "area") == "3.25 mm²" and stale(bar, "area") is True
    assert tooltip(bar, "area").split("\n") == [
        "Measured value from gt/mask/a.png (GT), recorded in root_areas.csv",
        "A mask of this image (GT or prediction) changed after this measurement — run Measure"
        " again for a current value.",
        "Compared by file time: this CSV has no record of its inputs.",
    ]

    controller.index.outdated = lambda _name: {"length": True, "area": True}
    controller.refresh("a")  # LENGTH outdated too: its CSV has a record, so by content
    assert stale(bar, "length") is True
    assert "Compared by file time" not in tooltip(bar, "length")
    assert "Compared by file time" in tooltip(bar, "area")


class BrokenIndex(Index):
    """An index whose CSVs can no longer be read (a network folder gone)."""

    def lookup(self, _name, load=False):
        raise OSError("the project folder is gone")


def test_an_index_that_cannot_be_read_leaves_the_bar_without_values(real_bar, caplog):
    bar = real_bar
    controller = MetricsController(metrics_window(bar=bar))
    controller.index = Index()
    controller.refresh("a")
    assert shown(bar, "length") and shown(bar, "area")

    controller.index = BrokenIndex()
    controller.refresh("a")  # logged, never raised

    assert "Could not read the measured values of a" in caplog.text
    assert text(bar, "name") == "a" and shown(bar, "status")
    assert not shown(bar, "length") and not shown(bar, "area")  # no value left from before


def test_a_save_reported_for_another_image_never_repaints_the_bar(real_bar):
    """The editors report saves by image; the bar follows only the selected one's."""
    bar = real_bar
    window = metrics_window(bar=bar)
    selected = SimpleNamespace(data=lambda _column, _role: "a")
    window.file_list = SimpleNamespace(currentItem=lambda: selected)
    controller = MetricsController(window)
    controller.index = Index()
    controller.refresh("a")

    controller.refresh_if_current("b")  # a save of image b

    assert text(bar, "name") == "a"
    assert text(bar, "status") == "Mask: prediction · No skeleton · No classes"


def _gt_window(bar, *, outdated: bool) -> MetricsController:
    """A controller on an image with a GT skeleton that the Measure run skipped (another
    size) and measured from its prediction; ``outdated``: a file changed since."""
    window = metrics_window(bar=bar)
    window.image_manager.skeleton_state = lambda _name: "gt"
    index = Index()
    index.lookup = lambda _name, load=False: Measurement(12.5, 3.25, None, "prediction", None)
    index.gt_skip_reason = lambda _name: {"length": "size"}
    index.outdated = lambda _name: {"length": outdated, "area": False}
    controller = MetricsController(window)
    controller.index = index
    return controller


def test_a_skipped_gt_skeleton_is_named_unless_the_value_is_outdated(real_bar):
    bar = real_bar
    _gt_window(bar, outdated=False).refresh("a")
    note = metrics.GT_NOT_USED.format(
        noun="skeleton", folder="skeletons", why="it is not the photo's size"
    )
    assert note in tooltip(bar, "length").split("\n")

    _gt_window(bar, outdated=True).refresh("a")  # a GT file saved after the CSV is new
    assert "Measure again" in tooltip(bar, "length")
    assert note not in tooltip(bar, "length")


def test_the_bar_never_asks_the_index_while_a_parse_runs():
    controller = MetricsController(metrics_window(bar=Bar()))
    controller.index, controller.loading = Index(needs_load=True), True
    controller.refresh("a")
    assert controller.index.asked == 0  # needs_load() waits for the parse's CSV lock


def test_one_parse_at_a_time_in_the_pool(monkeypatch):
    pool = Pool()
    monkeypatch.setattr(metrics, "QThreadPool", pool)
    controller = MetricsController(metrics_window(bar=Bar()))
    controller.index = index = Index(needs_load=True)
    controller.prewarm()
    controller.prewarm()
    assert controller.loading and len(pool.jobs) == 1
    reported = []
    controller.signals.ready.connect(reported.append)
    pool.jobs[0]()  # the pool runs the job: it parses, then reports the index
    assert index.prewarmed == 1 and reported == [index]


def test_the_parse_runs_as_a_marked_pool_job(monkeypatch):
    """Started by app.workers.start_pool_job: marked as a pool job (torch refuses there)."""
    from app.workers import in_pool_job

    pool = Pool()
    monkeypatch.setattr(metrics, "QThreadPool", pool)
    controller = MetricsController(metrics_window(bar=Bar()))
    controller.index = index = Index(needs_load=True)
    marks = []
    index.prewarm = lambda: marks.append(in_pool_job())
    controller.prewarm()
    pool.jobs[0]()
    assert marks == [True] and in_pool_job() is False


def test_a_parse_of_a_replaced_project_refreshes_nothing():
    bar, checks = Bar(), []
    controller = MetricsController(metrics_window(bar=bar, stage="Measure", checks=checks))
    controller.index, controller.loading = Index(), True
    controller.on_ready(Index())  # the parse of the previous project's index
    assert controller.loading is False and bar.shown == [] and checks == []
    controller.on_ready(controller.index)
    assert len(bar.shown) == 1 and len(checks) == 1  # Measure's counts follow a new CSV


def test_the_bar_shows_the_field_of_view_recorded_with_the_shown_value(app):
    """M2: the FOV cell follows the CSV of the shown value (length's, else area's)."""
    from app.data_processing.measurement_index import Measurement
    from app.gui.window.metrics import _shown_fov

    recorded = {"fov_width_mm": 18.0, "fov_height_mm": 13.0}
    same = {"fov_width_mm": 18.0, "fov_height_mm": 13.5}
    csvs = {"length": "D:/p/skeletons/root_lengths.csv", "area": "D:/p/mask/root_areas.csv"}
    both = Measurement(1.0, 2.0, None)

    assert _shown_fov(both, {"length": recorded, "area": same}, csvs) == (
        "18×13 mm",
        "Field of view recorded with root_lengths.csv\nThis app now uses 18×13.5 mm",
    )
    # Equal to the running one: the CSV is named, no second line.
    assert _shown_fov(both, {"length": same, "area": None}, csvs) == (
        "18×13.5 mm",
        "Field of view recorded with root_lengths.csv",
    )
    # Only the area is shown: the area's CSV; no value, no CSV or no record: the running one.
    assert (
        _shown_fov(Measurement(None, 2.0, None), {"length": same, "area": recorded}, csvs)[0]
        == "18×13 mm"
    )
    running = ("18×13.5 mm", "This app's field of view")
    assert _shown_fov(Measurement(None, None, None), {"length": recorded}, csvs) == running
    assert _shown_fov(both, {"length": None, "area": None}, csvs) == running
    assert _shown_fov(both, {}, {}) == running


STEM = "24EFCAM1_T001_L044_2024.07.18_111530_001_SMM"


def _row(stem: str, column: str, value) -> dict:
    return {
        "Image": stem,
        "Tube": 1,
        "Position": 44,
        "Date": "2024.07.18",
        "Time": "11:15:30",
        column: value,
    }


def _shift(path, seconds: float) -> None:
    """Move ``path``'s modification time by ``seconds``."""
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + int(seconds * 1e9)))


def _measured_project(root):
    """A project whose image has a mask, a generated skeleton and both CSVs, the inputs
    older than the CSVs (nothing outdated)."""
    (root / "mask").mkdir(parents=True)
    (root / "skeletons").mkdir()
    mask = root / "mask" / f"{STEM}.png"
    skeleton = root / "skeletons" / f"{STEM}_fake.png"
    mask.write_bytes(b"m")
    skeleton.write_bytes(b"s")
    lengths = str(root / "skeletons" / "root_lengths.csv")
    areas = str(root / "mask" / "root_areas.csv")
    write_metric_csv([_row(STEM, LENGTH_CSV_HEADERS[5], 18.77)], lengths, LENGTH_CSV_HEADERS)
    write_metric_csv([_row(STEM, AREA_CSV_HEADERS[5], 16.41)], areas, AREA_CSV_HEADERS)
    for path in (mask, skeleton):
        _shift(path, -60)
    return mask, skeleton, lengths, areas


@pytest.fixture
def window(app, tmp_path):
    made = [build_window(tmp_path / "s.ini", size=None, show=False)]
    yield made[0]
    delete_widgets(made, prepare=quiet_window)
    QApplication.processEvents()


def _open(window, folder) -> None:
    import time

    open_folder(window, folder, 1)
    item = window.library_index.item_for(STEM)
    window.file_list.setCurrentItem(item)
    window.on_image_selected(item)
    end = time.monotonic() + 5
    bar = window.metrics_bar
    # Both CSVs parsed (in the pool, one after the other): both values are on show.
    while time.monotonic() < end and (not shown(bar, "length") or not shown(bar, "area")):
        QApplication.processEvents()


def test_a_saved_mask_flags_area_not_length_in_the_window(window, tmp_path):
    folder = tmp_path / "proj"
    folder.mkdir()
    mask, _skeleton, _lengths, _areas = _measured_project(folder)
    write_photos(folder, [STEM])
    _shift(mask, 120)  # Ctrl+S in Correct rewrote the mask after the Measure run

    _open(window, folder)
    bar = window.metrics_bar

    assert text(bar, "length") == "18.77 mm"
    assert stale(bar, "length") is False
    assert stale(bar, "area") is True
    assert "root_lengths.csv" in tooltip(bar, "length")
    assert "root_areas.csv" in tooltip(bar, "area")


def test_a_gt_skeleton_saved_after_the_measurement_outdates_the_length(window, tmp_path):
    """O1: Measure reads the GT skeleton first, so one saved after the CSV makes LENGTH
    outdated (measure again), never AREA."""
    folder = tmp_path / "proj"
    folder.mkdir()
    _measured_project(folder)
    write_photos(folder, [STEM])
    gt_skeletons = folder / "gt" / "skeletons"
    gt_skeletons.mkdir(parents=True)
    (gt_skeletons / f"{STEM}.png").write_bytes(b"c")

    _open(window, folder)
    bar = window.metrics_bar
    tip = tooltip(bar, "length")

    assert stale(bar, "length") is True
    assert "Measure again" in tip and "not used" not in tip  # outdated: no skip note
    assert stale(bar, "area") is False


# ----------------------------------------------- end to end: the CSVs on disk, the real window
#: Four images: both values; no row at all; an area row only; a failed length row.
MEASURED, UNMEASURED, AREA_ONLY, FAILED = STEMS = [
    f"24EFCAM1_T001_L0{d}_2024.07.18_1115{d}_001_SMM" for d in (11, 12, 13, 14)
]


def _write_lengths(folder, rows) -> str:
    path = str(folder / "skeletons" / "root_lengths.csv")
    write_metric_csv(rows, path, LENGTH_CSV_HEADERS)
    return path


def _measured_folder(root, length: float = 18.77):
    """A project of :data:`STEMS` and both CSVs, written with the app's own writer."""
    folder = write_photos(root / "proj", STEMS)
    (folder / "skeletons").mkdir()
    (folder / "mask").mkdir()
    failed = {**_row(FAILED, LENGTH_CSV_HEADERS[5], ""), "Error": "cannot read the skeleton"}
    _write_lengths(folder, [_row(MEASURED, LENGTH_CSV_HEADERS[5], length), failed])
    areas = [
        _row(MEASURED, AREA_CSV_HEADERS[5], 16.41),
        _row(AREA_ONLY, AREA_CSV_HEADERS[5], 2.0),
        _row(FAILED, AREA_CSV_HEADERS[5], 3.5),
    ]
    write_metric_csv(areas, str(folder / "mask" / "root_areas.csv"), AREA_CSV_HEADERS)
    return folder


def _select(window, stem: str) -> None:
    item = window.library_index.item_for(stem)
    window.file_list.setCurrentItem(item)
    window.on_image_selected(item)


def _parsed(window) -> bool:
    """No CSV parse is running or pending: the bar shows what the CSVs hold."""
    return not window.metrics.loading and not window.metrics.index.needs_load()


def test_the_bar_shows_the_selected_images_values_from_the_csvs(window, tmp_path):
    open_folder(window, _measured_folder(tmp_path), len(STEMS))
    bar = window.metrics_bar

    _select(window, MEASURED)
    assert _wait(lambda: _parsed(window) and shown(bar, "length"))

    assert text(bar, "name") == MEASURED
    assert text(bar, "length") == "18.77 mm" and text(bar, "area") == "16.41 mm²"
    assert tooltip(bar, "length").endswith("root_lengths.csv")
    assert tooltip(bar, "area").endswith("root_areas.csv")


def test_an_image_without_a_row_shows_no_value_and_a_failed_row_says_so(window, tmp_path):
    open_folder(window, _measured_folder(tmp_path), len(STEMS))
    bar = window.metrics_bar
    _select(window, MEASURED)
    assert _wait(lambda: _parsed(window) and shown(bar, "length"))

    _select(window, UNMEASURED)
    assert text(bar, "name") == UNMEASURED
    assert not shown(bar, "length") and not shown(bar, "area")  # nothing carried over

    _select(window, AREA_ONLY)
    assert not shown(bar, "length")
    assert text(bar, "area") == "2.00 mm²"

    _select(window, FAILED)
    assert text(bar, "length") == "failed"
    assert tooltip(bar, "length") == "Measurement failed — Length: cannot read the skeleton"
    assert text(bar, "area") == "3.50 mm²"


def test_a_rewritten_csv_never_shows_its_old_value(window, tmp_path, monkeypatch):
    folder = _measured_folder(tmp_path)
    open_folder(window, folder, len(STEMS))
    bar = window.metrics_bar
    _select(window, MEASURED)
    assert _wait(lambda: _parsed(window) and text(bar, "length") == "18.77 mm")
    pool = Pool()  # from here on the re-parse runs when the test says so
    monkeypatch.setattr(metrics, "QThreadPool", pool)

    _write_lengths(folder, [_row(MEASURED, LENGTH_CSV_HEADERS[5], 20.5)])  # Measure ran again
    _select(window, UNMEASURED)  # the bar sees a changed CSV: a parse is queued
    _select(window, MEASURED)

    assert len(pool.jobs) == 1
    assert not shown(bar, "length")  # not parsed yet: nothing, never the old 18.77 mm
    pool.jobs.pop(0)()  # the re-parse runs and reports
    assert _wait(lambda: _parsed(window))
    assert text(bar, "length") == "20.50 mm"


def test_a_mask_saved_in_correct_reaches_the_bar_at_once(window, tmp_path):
    folder = _measured_folder(tmp_path)
    open_folder(window, folder, len(STEMS))
    bar = window.metrics_bar
    _select(window, MEASURED)
    assert _wait(lambda: _parsed(window) and shown(bar, "area"))
    assert not text(bar, "status").startswith("Mask: GT")
    assert stale(bar, "area") is False

    gt_mask = folder / "gt" / "mask" / f"{MEASURED}.png"  # what Correct's save wrote
    gt_mask.parent.mkdir(parents=True)
    gt_mask.write_bytes(b"m")
    _shift(gt_mask, 120)
    window.correct_editor.mask_saved.emit(str(folder / f"{MEASURED}.jpg"))

    assert text(bar, "status").startswith("Mask: GT")
    assert stale(bar, "area") is True and stale(bar, "length") is False


@pytest.fixture
def make_window(app, tmp_path):
    made: list = []

    def make():
        made.append(build_window(tmp_path / f"s{len(made)}.ini", size=None, show=False))
        return made[-1]

    yield make
    QThreadPool.globalInstance().waitForDone(10_000)
    delete_widgets(made, prepare=quiet_window)


def wait(predicate, timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        QApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_the_parse_reports_to_the_window_on_the_gui_thread(make_window, monkeypatch):
    window = make_window()
    ran_on = []
    monkeypatch.setattr(
        window.metrics, "on_ready", lambda index: ran_on.append(threading.get_ident())
    )
    signals = window.metrics.signals
    QThreadPool.globalInstance().start(lambda: signals.ready.emit(object()))
    assert wait(lambda: ran_on)
    assert ran_on == [threading.main_thread().ident]


def test_a_parse_that_ends_after_the_window_is_gone_reaches_nothing(app, tmp_path, monkeypatch):
    """The window, not the controller, receives the pool's report: once the window's C++
    side is deleted (its Python wrapper and the controller may live on until the next
    garbage collection), a parse that ends later reaches nothing."""
    reached, hooked = [], []
    monkeypatch.setattr(MetricsController, "on_ready", lambda self, index: reached.append(index))
    monkeypatch.setattr(sys, "excepthook", lambda *exc: hooked.append(exc))
    window = build_window(tmp_path / "s.ini", size=None, show=False)
    signals = window.metrics.signals  # the worker's closure keeps it, as in prewarm()
    window.close()
    window.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    try:
        assert sip.isdeleted(window)  # the wrapper (and its controller) are still here
        done = threading.Event()
        QThreadPool.globalInstance().start(lambda: (signals.ready.emit(object()), done.set()))
        assert done.wait(5)
        for _ in range(5):
            QApplication.processEvents()
        assert reached == [] and hooked == []
    finally:
        del window
        flush_deleted()
