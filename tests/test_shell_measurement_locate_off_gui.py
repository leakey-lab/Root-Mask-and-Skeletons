"""The metrics bar finds and parses the measurement CSVs off the GUI thread.

``MeasurementIndex.lookup(load=False)`` / ``needs_load()`` are the GUI thread's calls (every
library selection): they look for a CSV only where Measure writes it (a few stats) and never
parse; :meth:`MeasurementIndex.prewarm` parses in the pool, when a project opens and when a
selection finds a new or changed CSV (after a Measure run too). A located prewarm asked for
while another parse runs (a project opened during the previous project's parse) starts when
that parse reports. The older layout's folders (``[output/]skeletonizer/test_latest/images``,
a ``test_latest`` folder nearby) are not searched since #38 (2026-10-31): the walk of the
folders that searched for them, once kept off the GUI thread here, is gone.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest
from PyQt6.QtCore import QThread, QThreadPool
from PyQt6.QtWidgets import QApplication

from app.config import LENGTH_CSV_HEADERS
from app.data_processing.measurement_index import MeasurementIndex
from app.inference.metrics import write_metric_csv
from tests.qt_teardown import delete_widgets
from tests.window_support import build_window, open_folder, write_photos

STEM = "24EFCAM1_T001_L044_2024.07.18_111530_001_SMM"
#: Where Measure writes root_lengths.csv.
SKELETONS = ("skeletons",)

pytestmark = pytest.mark.usefixtures("no_temp_sweep")


def _lengths(root, parts: tuple[str, ...] = SKELETONS) -> str:
    folder = root.joinpath(*parts)
    folder.mkdir(parents=True, exist_ok=True)
    row = {
        "Image": STEM,
        "Tube": 1,
        "Position": 44,
        "Date": "2024.07.18",
        "Time": "11:15:30",
        LENGTH_CSV_HEADERS[5]: 9.25,
    }
    path = str(folder / "root_lengths.csv")
    write_metric_csv([row], path, LENGTH_CSV_HEADERS)
    return path


# ------------------------------------------------------------------------ the window
def test_the_bar_never_waits_for_a_csv_the_pool_is_parsing(tmp_path):
    """needs_load() takes the CSV lock, which the pool's parse holds (about 50 ms at 46k
    rows here): while a parse runs, a selection must not ask it (the parse's ready signal
    refreshes the bar)."""
    from app.gui.window.metrics import MetricsController

    shown: list[dict] = []
    window = SimpleNamespace(
        image_manager=SimpleNamespace(
            mask_state=lambda _n: "prediction",
            skeleton_state=lambda _n: "prediction",
            has_gt_labels=lambda _n: False,
        ),
        display_controller=SimpleNamespace(current_name=STEM),
        metrics_bar=SimpleNamespace(
            set_metrics=lambda **k: shown.append(k), set_fov=lambda *a: None
        ),
    )
    controller = MetricsController(window)
    index = MeasurementIndex(str(tmp_path))
    asked = []
    index.needs_load = lambda: asked.append(1) or False
    controller.index = index

    controller.loading = True  # a pool job is parsing
    controller.refresh(STEM)
    assert asked == [] and len(shown) == 1  # the bar was refreshed, the lock never asked

    controller.loading = False  # no parse: a selection asks whether the CSV changed
    controller.refresh(STEM)
    assert asked == [1]


def test_a_csv_reaches_the_metrics_bar(app, tmp_path):
    root = write_photos(tmp_path / "proj", [STEM])
    _lengths(root)
    window = build_window(tmp_path / "s.ini", size=None, show=False)
    try:
        open_folder(window, root, 1)
        item = window.library_index.item_for(STEM)
        window.file_list.setCurrentItem(item)
        window.on_image_selected(item)
        end = time.monotonic() + 5
        while time.monotonic() < end and window.metrics_bar._length_cell.isHidden():
            QApplication.processEvents()
        assert window.metrics_bar._length_value.text() == "9.25 mm"
    finally:
        delete_widgets([window])


# ------------------------------------------------------------------ after a batch job
class _Job(QThread):
    """A batch job that ends at once."""

    def run(self) -> None:
        pass


def _until(condition, timeout_s: float = 5.0) -> bool:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if condition():
            return True
        QApplication.processEvents()
    return condition()


def _end_job(window, name: str) -> None:
    """Run job ``name`` through the window's registry until it has ended."""
    ended = []
    window.jobs.busy_changed.connect(lambda busy: ended.append(1) if not busy else None)
    thread = _Job()
    assert window.jobs.try_start(name, thread)
    assert _until(lambda: ended)
    thread.wait()


def _window(made: list, tmp_path, *, csv: bool = True) -> None:
    """A window on a project whose length CSV is (or will be) in ``skeletons/``, its image
    selected; with ``csv`` it waits until the bar shows the CSV's value."""
    root = write_photos(tmp_path / "proj", [STEM])
    if csv:
        _lengths(root)
    window = build_window(tmp_path / "s.ini", size=None, show=False)
    made.append(window)
    open_folder(window, root, 1)
    item = window.library_index.item_for(STEM)
    window.file_list.setCurrentItem(item)
    window.on_image_selected(item)
    assert _until(lambda: not window.metrics.loading)
    if csv:
        assert _until(lambda: window.metrics_bar._length_value.text() == "9.25 mm")


@pytest.fixture
def made():
    widgets: list = []
    yield widgets
    QThreadPool.globalInstance().waitForDone(5000)
    delete_widgets(widgets)


def _length_text(window) -> str:
    bar = window.metrics_bar
    return "" if bar._length_cell.isHidden() else bar._length_value.text()


@pytest.mark.parametrize("job", ["Measuring root length", None])
def test_a_csv_stays_on_the_bar_after_a_measure_run(app, tmp_path, made, job):
    _window(made, tmp_path)
    window = made[0]

    if job is None:
        window.jobs.busy_changed.emit(False)  # a job end of no known name
    else:
        _end_job(window, job)

    assert _until(lambda: _length_text(window) == "9.25 mm" and not window.metrics.loading)
    assert window.metrics.index.sources()["length"].endswith("root_lengths.csv")


def test_a_csv_stays_on_the_bar_after_a_generation_pass(app, tmp_path, made):
    _window(made, tmp_path)
    window = made[0]

    _end_job(window, window.mask_generation_handler.job_label)

    assert _length_text(window) == "9.25 mm"  # never hidden: nothing was re-read


def test_a_measure_run_reaches_the_bar(app, tmp_path, made):
    _window(made, tmp_path, csv=False)
    window = made[0]
    assert _length_text(window) == ""  # nothing measured yet

    _lengths(tmp_path / "proj")  # what the Measure Length run wrote
    _end_job(window, "Measuring root length")

    assert _until(lambda: _length_text(window) == "9.25 mm")


def test_a_measure_run_that_ends_during_an_older_parse_reaches_the_bar(
    app, tmp_path, made, monkeypatch
):
    """A parse in flight at a Measure end searched for root_lengths.csv before the run wrote
    it: the bar's refresh when that parse reports finds the CSV and parses it (S06-01)."""
    _window(made, tmp_path, csv=False)
    window = made[0]
    searched, gate, calls = threading.Event(), threading.Event(), []
    real_prewarm = MeasurementIndex.prewarm

    def prewarm(index):
        calls.append(1)
        if len(calls) > 1:
            return real_prewarm(index)
        index._length.load()  # its search predates the run's CSV
        searched.set()
        gate.wait(5)
        index._area.load()

    monkeypatch.setattr(MeasurementIndex, "prewarm", prewarm)
    window.metrics.prewarm(locate=True)  # the older parse
    assert searched.wait(5)
    _lengths(tmp_path / "proj")  # where the run writes

    _end_job(window, "Measuring root length")
    gate.set()

    assert _until(lambda: _length_text(window) == "9.25 mm")
    assert len(calls) == 2


def test_a_project_opened_during_the_previous_parse_gets_its_own_search(app, tmp_path):
    """One parse at a time: the located prewarm of a project opened while the previous
    project's parse runs is refused, and starts when that parse reports."""
    from app.gui.window.metrics import MetricsController

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _lengths(tmp_path / "b")
    first, second = MeasurementIndex(str(tmp_path / "a")), MeasurementIndex(str(tmp_path / "b"))
    gate = threading.Event()
    real = first.prewarm
    first.prewarm = lambda: (gate.wait(5), real())  # the previous project's parse in flight
    window = SimpleNamespace(
        image_manager=None,
        display_controller=SimpleNamespace(current_name=None),
        stage_status=SimpleNamespace(check=lambda *a, **k: None),
        _current_stage=lambda: None,
    )
    controller = MetricsController(window)
    controller.signals.ready.connect(controller.on_ready)
    controller.index = first
    controller.prewarm(locate=True)  # project A opens
    controller.index = second
    controller.prewarm(locate=True)  # project B opens while A's parse runs: refused

    gate.set()

    assert _until(lambda: second.sources()["length"] is not None and not controller.loading)
    assert second.lookup(STEM, load=False).length_mm == 9.25


class _Pool:
    """The thread pool, recording the jobs it is given (run by hand)."""

    def __init__(self) -> None:
        self.jobs: list = []

    def globalInstance(self):  # noqa: N802 - Qt API name
        return self

    def start(self, job) -> None:
        self.jobs.append(job)


def _controller(monkeypatch, refreshed: list, checks: list):
    """A metrics controller on the Measure stage whose bar refreshes and Measure counts are
    recorded, over a stub pool (its jobs run by hand) and a stand-in index."""
    from app.gui.window import metrics

    pool = _Pool()
    monkeypatch.setattr(metrics, "QThreadPool", pool)
    window = SimpleNamespace(
        image_manager=None,
        display_controller=SimpleNamespace(current_name=None),
        stage_status=SimpleNamespace(check=lambda *a, **k: checks.append(1)),
        _current_stage=lambda: "Measure",
    )
    controller = metrics.MetricsController(window)
    controller.refresh = lambda name=None: refreshed.append(name)
    controller.index = SimpleNamespace(prewarm=lambda: None, needs_load=lambda: False)
    return controller, pool


def test_a_located_prewarm_asked_during_a_parse_runs_after_it(monkeypatch):
    """A Measure run ended while an older parse ran (its search may predate the run's CSV):
    the located prewarm runs once that parse reports, and the bar and Measure's counts
    follow its own report."""
    refreshed, checks = [], []
    controller, pool = _controller(monkeypatch, refreshed, checks)
    controller.prewarm()  # nothing to parse: refused
    controller.loading = True  # an older parse in flight
    controller.prewarm(locate=True)  # refused for now
    assert pool.jobs == []

    controller.on_ready(controller.index)  # the older parse reports

    assert len(pool.jobs) == 1 and controller.loading
    assert refreshed == [] and checks == []  # the located parse reports next
    controller.on_ready(controller.index)
    assert refreshed == [None] and checks == [1] and not controller.loading


def test_a_plain_prewarm_refused_during_a_parse_waits_for_nothing(monkeypatch):
    refreshed, checks = [], []
    controller, pool = _controller(monkeypatch, refreshed, checks)
    controller.loading = True
    controller.prewarm()  # a selection's parse: the report's refresh asks again

    controller.on_ready(controller.index)

    assert pool.jobs == [] and refreshed == [None] and checks == [1]
