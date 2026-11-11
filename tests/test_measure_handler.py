"""SkeletonHandler Measure flow: jobs, feedback, Retry, cancel, thread release, is_busy.

Covers RB-03/FR-06 (locked CSV -> Retry), RB-07 (one job at a time, Cancel),
RB-08 (thread released only after it finished), AR-07 (named feedback,
``is_busy(kind)``). Uses a fake main window: no MainWindow internals needed.
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PyQt6 import sip
from PyQt6.QtWidgets import QMessageBox

from app import provenance
from app.inference import measure_run
from app.io_utils import FileLockedError
from tests.measure_support import _handler, _project, _wait_idle, _warnings, _write_project_file


def test_measure_length_runs_as_named_job_with_feedback(tmp_path, qapp):
    _project(tmp_path)
    handler, mw = _handler(tmp_path, qapp)
    handler.calculate_root_length()
    thread = handler.calculator_thread
    assert thread is not None
    assert mw.jobs.running_name() == "Measuring root length"
    _wait_idle(handler, qapp)

    assert handler.calculator_thread is None
    # RB-08: released (app.workers.release_qthread): deleted once Qt says it finished;
    # deleting a still-running QThread would abort the process ("QThread: Destroyed while
    # thread is still running").
    assert sip.isdeleted(thread)
    names = mw.feedback.names()
    assert names[0] == "task_started" and "task_finished" in names
    label = mw.feedback.calls[0].args[0]
    assert "root length" in label.lower() and "4" in label
    assert mw.feedback.calls[0].kwargs["cancelable"] is True
    assert os.path.isfile(tmp_path / "skeletons" / "root_lengths.csv")
    assert os.path.isfile(tmp_path / "skeletons" / "root_lengths.csv.provenance.json")


def test_measure_area_reads_png_masks(tmp_path, qapp):
    _project(tmp_path, skeletons=False)
    handler, mw = _handler(tmp_path, qapp)
    handler.calculate_root_area()
    assert mw.jobs.running_name() == "Measuring root area"
    _wait_idle(handler, qapp)
    assert handler.area_calculator_thread is None
    assert os.path.isfile(tmp_path / "mask" / "root_areas.csv")


def test_is_busy_reports_each_kind(tmp_path, qapp):
    handler, _ = _handler(tmp_path, qapp)
    assert handler.is_busy() is False
    handler.calculator_thread = SimpleNamespace(isRunning=lambda: True)
    assert handler.is_busy("length") and handler.is_busy()
    assert not handler.is_busy("area")


def test_skeleton_generation_makes_length_busy(tmp_path, qapp):
    handler, _ = _handler(tmp_path, qapp)
    handler.skeleton_handler = SimpleNamespace(is_busy=lambda: True)
    assert handler.is_busy("length")
    assert not handler.is_busy("area")


def test_refused_while_another_job_runs(tmp_path, qapp):
    _project(tmp_path)
    jobs = SimpleNamespace(
        try_start=lambda name, thread: False,
        running_name=lambda: "Generating skeletons",
        is_busy=lambda: True,
    )
    handler, mw = _handler(tmp_path, qapp, jobs=jobs)
    handler.calculate_root_length()
    assert handler.calculator_thread is None
    assert mw.feedback.names() == ["notify"]
    message, kind = mw.feedback.calls[0].args
    assert "Generating skeletons" in message and kind == "warn"


def test_missing_skeletons_is_a_warning_not_a_job(tmp_path, qapp):
    _project(tmp_path, skeletons=False)
    handler, mw = _handler(tmp_path, qapp)
    handler.calculate_root_length()
    assert handler.calculator_thread is None
    assert mw.feedback.calls[-1].args[1] == "warn"
    assert not mw.jobs.is_busy()


def test_locked_csv_fails_softly_and_retry_saves_kept_rows(tmp_path, qapp, monkeypatch):
    _project(tmp_path)
    handler, mw = _handler(tmp_path, qapp)
    prompts = []
    monkeypatch.setattr(handler, "_offer_retry", lambda kind, msg: prompts.append((kind, msg)))

    def locked(results, filename, headers):
        raise FileLockedError(filename)

    monkeypatch.setattr(measure_run, "write_metric_csv", locked)
    handler.calculate_root_length()
    _wait_idle(handler, qapp)
    assert "task_failed" in mw.feedback.names()
    assert prompts and prompts[0][0] == "length"
    assert "root_lengths.csv is open in another program" in prompts[0][1]
    assert handler.calculator_thread is None

    monkeypatch.undo()  # the user closed Excel
    monkeypatch.setattr(measure_run, "run_metric_pool", pytest.fail)  # no re-measuring
    assert handler.retry_measure("length") is True
    _wait_idle(handler, qapp)
    assert mw.feedback.names()[-2:] == ["task_finished", "notify"]
    assert os.path.isfile(tmp_path / "skeletons" / "root_lengths.csv")
    assert handler.retry_measure("length") is False  # nothing left to retry


def test_cancel_stops_the_run_without_writing(tmp_path, qapp):
    _project(tmp_path, n=12)
    handler, mw = _handler(tmp_path, qapp)
    handler.calculate_root_length()
    mw.feedback.cancel()  # the Cancel button
    _wait_idle(handler, qapp)
    assert mw.feedback.names()[-1] == "task_finished"
    assert "cancel" in mw.feedback.last_message.lower()
    assert not os.path.exists(tmp_path / "skeletons" / "root_lengths.csv")


def test_retry_prompt_buttons_drive_retry(tmp_path, qapp, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    handler, _ = _handler(tmp_path, qapp)
    retried = []
    monkeypatch.setattr(handler, "retry_measure", retried.append)
    handler._retry_jobs["area"] = object()
    handler._offer_retry("length", "root_lengths.csv is open in another program")
    handler._retry_box.button(QMessageBox.StandardButton.Retry).click()
    handler._offer_retry("area", "root_areas.csv is open in another program")
    handler._retry_box.button(QMessageBox.StandardButton.Cancel).click()
    qapp.processEvents()
    assert retried == ["length"]
    assert "area" not in handler._retry_jobs and handler._retry_box is None


# --------------------------------------------------------------------------- O3: project FOV


def test_measure_reads_the_project_file(tmp_path, qapp):
    import csv

    from app.config import AREA_VALUE_COLUMN, Calibration
    from app.inference import metrics

    _project(tmp_path, n=2, skeletons=False)
    _write_project_file(
        # Another field of view than the default (18 x 13.5), so the file visibly decides.
        tmp_path,
        '{"schema": 1, "calibration": {"fov_width_mm": 18.0, "fov_height_mm": 13.0}}',
    )
    handler, mw = _handler(tmp_path, qapp)
    handler.calculate_root_area()
    _wait_idle(handler, qapp)

    csv_path = tmp_path / "mask" / "root_areas.csv"
    record = provenance.read_sidecar(csv_path)
    assert record["calibration"]["source"] == "project"
    assert record["calibration"]["fov_height_mm"] == 13.0
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    for row in rows:
        mask = metrics.load_mask(str(tmp_path / "mask" / f"{row['Image']}.png"))
        expected = metrics.root_area_mm2(mask, calibration=Calibration(18.0, 13.0))
        assert float(row[AREA_VALUE_COLUMN]) == round(expected, 2)
    assert not any("sprouts_project.json" in w for w in _warnings(mw))


def test_an_invalid_project_file_warns_once_and_measures_with_the_default(tmp_path, qapp):
    _project(tmp_path, n=2, skeletons=False)
    _write_project_file(tmp_path, "{")
    handler, mw = _handler(tmp_path, qapp)
    handler.calculate_root_area()
    _wait_idle(handler, qapp)

    project_warnings = [w for w in _warnings(mw) if "sprouts_project.json" in w]
    assert project_warnings == [
        "sprouts_project.json was not used (it is not valid JSON): "
        "measured with the default field of view 18×13.5 mm."
    ]
    record = provenance.read_sidecar(tmp_path / "mask" / "root_areas.csv")
    assert record["calibration"]["source"] == "default"


def test_a_refused_run_says_nothing_about_the_project_file(tmp_path, qapp):
    _project(tmp_path, n=2, skeletons=False)
    _write_project_file(tmp_path, "{")
    jobs = SimpleNamespace(
        try_start=lambda name, thread: False,
        running_name=lambda: "Generating skeletons",
        is_busy=lambda: True,
    )
    handler, mw = _handler(tmp_path, qapp, jobs=jobs)
    handler.calculate_root_area()
    assert handler.area_calculator_thread is None
    assert [w for w in _warnings(mw) if "sprouts_project.json" in w] == []
    assert any("Generating skeletons" in w for w in _warnings(mw))


# ----------------------------------------------------------------- "Measure again?" (R8, O17)
AGAIN_TEXT = "Nothing changed since the last measurement. Measure again?"


def _measure_area_twice(tmp_path, qapp, between=None, before=None):
    """Measure area once (after ``before``), run ``between``, ask again: ``(handler, mw, csv, mtime)``."""
    _project(tmp_path, skeletons=False)
    if before is not None:
        before()
    handler, mw = _handler(tmp_path, qapp)
    handler.calculate_root_area()
    _wait_idle(handler, qapp)
    csv = tmp_path / "mask" / "root_areas.csv"
    mtime = csv.stat().st_mtime_ns
    if between is not None:
        between()
    handler.calculate_root_area()
    _wait_idle(handler, qapp)
    return handler, mw, csv, mtime


def test_measuring_again_with_nothing_changed_asks_first(tmp_path, qapp):
    handler, mw, csv, mtime = _measure_area_twice(tmp_path, qapp)
    box = handler._again_box
    assert box is not None and box.isVisible()
    assert box.text() == AGAIN_TEXT
    assert box.informativeText() == "root_areas.csv already holds the values of these 4 masks."
    assert box.defaultButton() is box.button(QMessageBox.StandardButton.Cancel)
    assert csv.stat().st_mtime_ns == mtime
    time.sleep(0.02)
    box.done(QMessageBox.StandardButton.Yes.value)
    assert handler._again_box is None
    _wait_idle(handler, qapp)
    assert csv.stat().st_mtime_ns > mtime


def test_a_changed_input_measures_without_asking(tmp_path, qapp):
    def repaint():
        mask = next((tmp_path / "mask").glob("*.png"))
        cv2.imwrite(str(mask), np.full((64, 80), 255, np.uint8))

    handler, mw, csv, mtime = _measure_area_twice(tmp_path, qapp, between=repaint)
    assert handler._again_box is None
    assert csv.stat().st_mtime_ns > mtime


def test_a_csv_without_a_record_measures_without_asking(tmp_path, qapp):
    def forget():
        os.remove(tmp_path / "mask" / "root_areas.csv.inputs.json")

    handler, mw, csv, mtime = _measure_area_twice(tmp_path, qapp, between=forget)
    assert handler._again_box is None
    assert csv.stat().st_mtime_ns > mtime


def test_cancel_keeps_the_csv(tmp_path, qapp):
    handler, mw, csv, mtime = _measure_area_twice(tmp_path, qapp)
    box = handler._again_box
    box.done(QMessageBox.StandardButton.Cancel.value)
    _wait_idle(handler, qapp)
    assert handler.area_calculator_thread is None
    assert csv.stat().st_mtime_ns == mtime
    finished = [c.args[0] for c in mw.feedback.calls if c.name == "task_finished"]
    assert finished[-1] == "Nothing changed since the last measurement: root_areas.csv kept"


def _write_gt_mask(tmp_path, offset=0):
    stem = next((tmp_path / "mask").glob("*.png")).stem
    gt = np.zeros((60, 80), np.uint8)  # the photo's size (_project): a GT file that is used
    gt[10 + offset : 30 + offset, 10:30] = 255
    (tmp_path / "gt" / "mask").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(tmp_path / "gt" / "mask" / f"{stem}.png"), gt)


def test_measuring_again_with_a_gt_file_asks_first(tmp_path, qapp):
    handler, mw, csv, mtime = _measure_area_twice(
        tmp_path, qapp, before=lambda: _write_gt_mask(tmp_path)
    )
    box = handler._again_box
    assert box is not None and box.text() == AGAIN_TEXT
    assert box.informativeText() == "root_areas.csv already holds the values of these 4 masks."
    assert csv.stat().st_mtime_ns == mtime
    box.done(QMessageBox.StandardButton.Cancel.value)


def test_editing_the_gt_mask_measures_without_asking(tmp_path, qapp):
    handler, mw, csv, mtime = _measure_area_twice(
        tmp_path,
        qapp,
        before=lambda: _write_gt_mask(tmp_path),
        between=lambda: _write_gt_mask(tmp_path, offset=7),
    )
    assert handler._again_box is None
    assert csv.stat().st_mtime_ns > mtime


# ------------------------------------------------------------------ the older layout (#38)
def test_an_older_mask_folder_next_to_mask_is_ignored(tmp_path, qapp):
    _project(tmp_path, skeletons=False)  # mask/: where Generate and Correct save
    older = tmp_path / "output" / "mask"
    older.mkdir(parents=True)
    cv2.imwrite(str(older / "x_T1_L2_2024.01.02_010200_001.png"), np.zeros((64, 80), np.uint8))
    handler, mw = _handler(tmp_path, qapp)

    handler.calculate_root_area()
    _wait_idle(handler, qapp)

    assert _warnings(mw) == []
    assert (tmp_path / "mask" / "root_areas.csv").is_file()
    assert sorted(p.name for p in older.iterdir()) == ["x_T1_L2_2024.01.02_010200_001.png"]


def test_an_older_layout_alone_has_nothing_to_measure(tmp_path, qapp):
    """Skeletons only in ``skeletonizer/test_latest/images`` and masks only in
    ``output/mask``: Measure says there is nothing to measure and writes no CSV."""
    _project(tmp_path, skeletons=False, masks=False)
    plane = np.zeros((64, 80), np.uint8)
    for folder, suffix in (
        (tmp_path / "skeletonizer" / "test_latest" / "images", "_fake.png"),
        (tmp_path / "output" / "mask", ".png"),
    ):
        folder.mkdir(parents=True)
        cv2.imwrite(str(folder / f"x_T1_L2_2024.01.02_010200_001{suffix}"), plane)
    handler, mw = _handler(tmp_path, qapp)

    handler.calculate_root_length()
    handler.calculate_root_area()
    _wait_idle(handler, qapp)

    assert _warnings(mw) == [
        "No skeleton images found. Generate skeletons first.",
        "No mask images found. Generate masks first.",
    ]
    assert "task_started" not in mw.feedback.names()
    assert sorted(p.name for p in tmp_path.rglob("*.csv")) == []


# ------------------------------------------------------------------ cancel and failure
def test_a_cancelled_remeasure_leaves_the_previous_results_untouched(tmp_path, qapp):
    """Cancel during a second run: the CSV, its sidecar and its inputs record of the first run
    stay byte for byte, and no temp file is left beside them or in the project folder."""
    _project(tmp_path, n=12, skeletons=False)
    handler, mw = _handler(tmp_path, qapp)
    handler.calculate_root_area()
    _wait_idle(handler, qapp)
    mask_dir = tmp_path / "mask"
    files = [
        mask_dir / "root_areas.csv",
        mask_dir / "root_areas.csv.provenance.json",
        mask_dir / "root_areas.csv.inputs.json",
    ]
    before = {path.name: path.read_bytes() for path in files}
    listing = {folder: sorted(os.listdir(folder)) for folder in (tmp_path, mask_dir)}
    # A changed input, so the second run measures instead of asking "Measure again?".
    mask = next(mask_dir.glob("x_*.png"))
    cv2.imwrite(str(mask), np.full((64, 80), 255, np.uint8))

    handler.calculate_root_area()
    mw.feedback.cancel()  # the Cancel button
    _wait_idle(handler, qapp)

    assert "cancel" in mw.feedback.last_message.lower()
    assert {path.name: path.read_bytes() for path in files} == before
    assert {folder: sorted(os.listdir(folder)) for folder in (tmp_path, mask_dir)} == listing


def test_a_run_whose_measuring_fails_says_so_and_keeps_nothing_to_retry(
    tmp_path, qapp, monkeypatch
):
    """Not the write but the measuring fails (a worker raises): a "Measure failed" box with
    the error, a failed task, no CSV, and no rows kept for a Retry."""
    from app.handlers import skeleton_handler

    _project(tmp_path, skeletons=False)
    shown = []
    monkeypatch.setattr(
        skeleton_handler.QMessageBox,
        "warning",
        lambda parent, title, text: shown.append((title, text)),
    )

    def broken(*_args, **_kwargs):
        raise RuntimeError("simulated measuring failure")

    monkeypatch.setattr(measure_run, "run_metric_pool", broken)
    handler, mw = _handler(tmp_path, qapp)
    handler.calculate_root_area()
    _wait_idle(handler, qapp)

    assert len(shown) == 1 and shown[0][0] == "Measure failed"
    assert "simulated measuring failure" in shown[0][1]
    assert "task_failed" in mw.feedback.names()
    assert handler.retry_measure("area") is False
    assert not os.path.exists(tmp_path / "mask" / "root_areas.csv")
