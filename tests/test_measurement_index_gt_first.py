"""MeasurementIndex on a GT-first project (O1): freshness follows the file each row was
measured from.

A ``gt`` row is outdated by its GT file only (a regenerated prediction under it is not); a
``prediction`` row by its prediction and by a GT file saved since; a row whose GT file is gone
is outdated; a CSV written before the ``Source`` column is outdated beside a GT file. Without
the record of the inputs, file times decide, with the same files. The GUI thread finds GT
files with a stat, never a folder listing, and the file named like the image wins over another
of its case key.
"""

from __future__ import annotations

import os
import time

import cv2
import numpy as np
import pytest

from app.config import AREA_CSV_HEADERS, LENGTH_CSV_HEADERS
from app.data_processing.measurement_index import MeasurementIndex
from app.gui.stage_summary import Freshness, input_freshness
from app.inference import metrics
from app.project_layout import ProjectLayout
from tests.measure_sources_support import (
    PHOTO_SIZE,
    _write_png,
    make_gt_project,
    prediction_skeleton,
    write_gt_mask,
    write_gt_skeleton,
    write_photo,
)
from tests.measure_support import _csv_path, _run

LENGTH_COL, AREA_COL = LENGTH_CSV_HEADERS[5], AREA_CSV_HEADERS[5]


pytestmark = pytest.mark.usefixtures("no_prompts")


@pytest.fixture
def gt_project(tmp_path) -> tuple[ProjectLayout, dict[str, str]]:
    return make_gt_project(str(tmp_path))


def other_skeleton() -> np.ndarray:
    """Another hand-traced 1 px skeleton at the photo's size."""
    w, h = PHOTO_SIZE
    gray = np.zeros((h, w), np.uint8)
    cv2.line(gray, (30, 30), (w - 30, h // 2), 255, 1)
    return gray


def test_outdated_follows_the_measured_file(gt_project, qapp):
    layout, _ = gt_project

    def fresh() -> MeasurementIndex:
        _run(layout, qapp, "length")
        _run(layout, qapp, "area")
        made = MeasurementIndex(layout.root)
        made.prewarm()
        return made

    def again(index: MeasurementIndex) -> MeasurementIndex:
        index.invalidate()
        index.prewarm()
        return index

    index = fresh()
    assert index.outdated("g") == {"length": False, "area": False}
    assert index.outdated("s") == {"length": False, "area": False}  # its unused GT is recorded

    write_gt_skeleton(layout, "g", other_skeleton())
    assert again(index).outdated("g")["length"] is True  # the measured GT file changed
    write_gt_skeleton(layout, "p")
    assert again(index).outdated("p")["length"] is True  # a GT saved after the CSV
    os.remove(layout.gt_skeleton_path("o"))
    assert again(index).outdated("o")["length"] is True  # the measured GT file is gone

    index = fresh()
    _write_png(layout.skeleton_path("g"), np.roll(prediction_skeleton(), 9, axis=1))
    # a regenerated prediction under a gt row does not outdate it
    assert again(index).outdated("g") == {"length": False, "area": False}
    write_gt_mask(layout, "p")
    assert again(index).outdated("p") == {"length": False, "area": True}
    assert index.gt_skip_reason("s") == {"length": "size"}
    assert index.gt_skip_reason("x") == {"length": "unreadable", "area": "unreadable"}
    assert index.gt_skip_reason("g") == {}


def test_a_browse_step_does_not_list_the_gt_folders(gt_project, qapp, monkeypatch):
    """A case-sensitive disk (Linux) simulated: the GUI thread finds a GT file by its exact
    name, else in the listing read with the CSVs on the worker; it never lists gt/."""
    from app import project_layout

    layout, _ = gt_project
    os.remove(layout.gt_skeleton_path("o"))  # o's GT skeleton named O.png: a case variant
    _write_png(os.path.join(layout.gt_skeletons_dir, "O.png"), other_skeleton())
    _run(layout, qapp, "length")
    _run(layout, qapp, "area")
    index = MeasurementIndex(layout.root)
    index.prewarm()  # the worker's load lists gt/ (once)

    listed: list[str] = []
    real_scandir = os.scandir

    def scandir(path="."):
        listed.append(os.path.normcase(os.path.abspath(os.fspath(path))))
        return real_scandir(path)

    monkeypatch.setattr(project_layout, "_ignores_case", lambda _directory: False)
    monkeypatch.setattr(os, "scandir", scandir)
    for stem in ("p", "g", "s", "o", "x", "nothing"):
        index.outdated(stem)
        index.measured_from(stem)
        index.gt_skip_reason(stem)
    monkeypatch.setattr(os, "scandir", real_scandir)

    gt_dir = os.path.normcase(os.path.abspath(layout.gt_dir))
    assert [path for path in listed if path.startswith(gt_dir)] == []
    # the case variant is still found, by the name it has on disk
    assert index.measured_from("o")["length"] == ("gt/skeletons/O.png", "gt")
    assert index.outdated("o") == {"length": False, "area": False}


def _age(*folders: str, seconds: float = -60.0) -> None:
    """Move every file of ``folders`` back in time by ``seconds``."""
    for folder in folders:
        for entry in os.scandir(folder):
            st = entry.stat()
            os.utime(entry.path, ns=(st.st_atime_ns, st.st_mtime_ns + int(seconds * 1e9)))


def _by_time(layout: ProjectLayout, images: dict[str, str], csv_path: str) -> Freshness:
    """The status line's count of length without a record (file times)."""
    return input_freshness(
        layout.skeletons_dir,
        "_fake.png",
        csv_path,
        record=None,
        project_root=layout.root,
        images=images,
        gt_folder=layout.gt_skeletons_dir,
    )


def test_without_a_record_the_bar_and_the_status_line_compare_the_gt_file_only(gt_project, qapp):
    layout, images = gt_project
    _run(layout, qapp, "length")
    csv_path = _csv_path(layout, "length")
    os.remove(csv_path + ".inputs.json")  # no record: file times
    _age(layout.skeletons_dir, layout.gt_skeletons_dir)
    measured_at = time.time_ns()
    os.utime(csv_path, ns=(measured_at, measured_at))  # the CSV is the newest file
    later = (measured_at + 60_000_000_000,) * 2  # explicit: file-time ticks are coarse

    def outdated(stem: str) -> bool:
        index = MeasurementIndex(layout.root)
        index.prewarm()
        return index.outdated(stem)["length"]

    assert _by_time(layout, images, csv_path).changed == 0
    assert not outdated("g") and not outdated("s")
    # a regenerated prediction of an image with a GT file: the GT file is what Measure reads
    _write_png(layout.skeleton_path("g"), np.roll(prediction_skeleton(), 9, axis=1))
    os.utime(layout.skeleton_path("g"), ns=later)
    assert _by_time(layout, images, csv_path).changed == 0 and not outdated("g")
    write_gt_skeleton(layout, "g", other_skeleton())  # the GT file changed after the CSV
    os.utime(layout.gt_skeleton_path("g"), ns=later)
    assert _by_time(layout, images, csv_path).changed == 1 and outdated("g")


def test_a_csv_without_source_is_outdated_beside_a_gt_file(tmp_path):
    """A CSV written before WG-T8 measured predictions only: where a GT file exists now,
    Measure would read it, whatever the file times."""
    layout = ProjectLayout(str(tmp_path))
    images = {stem: write_photo(layout, stem) for stem in ("a", "b")}
    for stem in ("a", "b"):
        _write_png(layout.skeleton_path(stem), prediction_skeleton())
    write_gt_skeleton(layout, "a")
    rows = [
        {"Image": stem, "Tube": None, "Position": None, "Date": None, "Time": None, LENGTH_COL: 1.0}
        for stem in ("a", "b")
    ]
    csv_path = _csv_path(layout, "length")
    metrics.write_metric_csv(rows, csv_path, LENGTH_CSV_HEADERS[:7])  # no Source column
    _age(layout.skeletons_dir, layout.gt_skeletons_dir)
    os.utime(csv_path)

    index = MeasurementIndex(layout.root)
    index.prewarm()
    assert index.outdated("a")["length"] is True and index.outdated("b")["length"] is False
    assert _by_time(layout, images, csv_path).changed == 1


def test_without_a_record_a_prediction_row_compares_its_prediction_too(gt_project, qapp):
    """s was measured from its prediction (its GT file is not the photo's size): a
    regenerated prediction outdates it, by file time too; a gt row ignores its prediction."""
    layout, _ = gt_project
    _run(layout, qapp, "length")
    csv_path = _csv_path(layout, "length")
    os.remove(csv_path + ".inputs.json")  # no record: file times
    _age(layout.skeletons_dir, layout.gt_skeletons_dir)
    measured_at = time.time_ns()
    os.utime(csv_path, ns=(measured_at, measured_at))
    for stem in ("s", "g"):  # both predictions regenerated after the CSV
        _write_png(layout.skeleton_path(stem), np.roll(prediction_skeleton(), 9, axis=1))
        os.utime(layout.skeleton_path(stem), ns=(measured_at + 60_000_000_000,) * 2)

    index = MeasurementIndex(layout.root)
    index.prewarm()

    assert index.outdated("s")["length"] is True  # the file its row was measured from
    assert index.outdated("g")["length"] is False  # a gt row: its GT file only


def _case_sensitive(folder: str) -> bool:
    probe = os.path.join(folder, "Probe")
    with open(probe, "wb"):
        pass
    try:
        return not os.path.exists(os.path.join(folder, "probe"))
    finally:
        os.remove(probe)


def test_with_both_names_the_bar_compares_the_file_measure_read(tmp_path, qapp):
    """A case-sensitive disk with gt/skeletons/o.png and O.png: Measure reads o.png (named
    like the image o.jpg, as split_inputs keeps it); the bar compares that file, not the
    listing's pick (O.png sorts first)."""
    layout = ProjectLayout(str(tmp_path))
    os.makedirs(layout.gt_skeletons_dir)
    if not _case_sensitive(layout.gt_skeletons_dir):
        pytest.skip("needs a case-sensitive file system (runs on the ubuntu CI)")
    write_photo(layout, "o")
    write_gt_skeleton(layout, "o")
    _write_png(os.path.join(layout.gt_skeletons_dir, "O.png"), other_skeleton())

    _run(layout, qapp, "length")
    index = MeasurementIndex(layout.root)
    index.prewarm()

    assert index.measured_from("o")["length"] == ("gt/skeletons/o.png", "gt")
    assert index.outdated("o")["length"] is False


def test_the_image_named_gt_file_wins_over_another_listed_one(gt_project, qapp):
    """The same rule on any disk: the listing names another file for o's stem key (what a
    case-sensitive disk lists for O.png beside o.png); the bar still compares o.png."""
    layout, _ = gt_project
    _run(layout, qapp, "length")
    index = MeasurementIndex(layout.root)
    index.prewarm()
    other = os.path.join(layout.gt_skeletons_dir, "o_listed.png")
    _write_png(other, other_skeleton())
    index._gt_listing["length"]["o"] = other  # as listed beside o.png

    assert index.measured_from("o")["length"] == ("gt/skeletons/o.png", "gt")
    assert index.outdated("o")["length"] is False
    os.remove(layout.gt_skeleton_path("o"))  # only the listed file left: it is the GT file
    assert index.measured_from("o")["length"] == ("gt/skeletons/o_listed.png", "gt")
