"""MeasurementIndex: the read-only per-image lookup of the Measure CSVs (KNOWN_ISSUES #3).

The index never computes a metric: it reads ``root_lengths.csv`` / ``root_areas.csv`` where
Measure writes them (never the older layout's folders, #38), verbatim; an ``Error`` row gives
``None`` and the message, never its 0. A CSV is parsed once per change, by a worker
(``prewarm``); the GUI thread's ``lookup(load=False)`` never parses. A stem is looked up
exactly, else without case; a case key two rows share finds nothing (O7).
"""

from __future__ import annotations

import os
import time

import pytest

from app.config import AREA_CSV_HEADERS, LENGTH_CSV_HEADERS
from app.data_processing.measurement_index import Measurement, MeasurementIndex
from app.inference.metrics import write_metric_csv
from tests.measure_support import _bump_mtime, _row, _write_areas, _write_lengths

STEM = "24EFCAM1_T001_L044_2024.07.18_111530_001_SMM"
OTHER = "24EFCAM1_T001_L045_2024.07.18_111612_001_SMM"


def test_values_are_read_verbatim_from_both_csvs(tmp_path):
    _write_lengths(tmp_path, [_row(STEM, LENGTH_CSV_HEADERS[5], 18.77)])
    _write_areas(tmp_path, [_row(STEM, AREA_CSV_HEADERS[5], 16.41)])

    m = MeasurementIndex(str(tmp_path)).lookup(STEM)

    assert m == Measurement(18.77, 16.41, None)
    length, area, error, length_source, area_source = m  # tuple-compatible
    assert (length, area, error) == (18.77, 16.41, None)
    assert (length_source, area_source) == (None, None)  # an empty Source cell


def test_lookup_accepts_an_image_path_or_a_stem_as_it_is(tmp_path):
    # A bare name is a stem, never stripped again: the stem of "X.png.jpg" is "X.png", whose
    # row is not image X's (final review D1).
    rows = [_row(STEM, LENGTH_CSV_HEADERS[5], 3.5), _row(STEM + ".png", LENGTH_CSV_HEADERS[5], 1.5)]
    _write_lengths(tmp_path, rows)
    index = MeasurementIndex(str(tmp_path))
    assert index.lookup(os.path.join(str(tmp_path), STEM + ".jpg")).length_mm == 3.5
    assert index.lookup(STEM).length_mm == 3.5
    assert index.lookup(STEM + ".png").length_mm == 1.5
    assert index.lookup(os.path.join(str(tmp_path), STEM + ".png.jpg")).length_mm == 1.5
    assert index.lookup(OTHER + ".png").length_mm is None  # no row of that stem


def test_missing_csvs_give_all_none(tmp_path):
    assert MeasurementIndex(str(tmp_path)).lookup(STEM) == Measurement(None, None, None)


def test_unknown_stem_gives_none(tmp_path):
    _write_lengths(tmp_path, [_row(STEM, LENGTH_CSV_HEADERS[5], 1.0)])
    assert MeasurementIndex(str(tmp_path)).lookup(OTHER) == Measurement(None, None, None)


def test_error_row_gives_none_value_and_message_never_the_zero(tmp_path):
    _write_lengths(tmp_path, [_row(STEM, LENGTH_CSV_HEADERS[5], 0, error="cannot read image")])
    _write_areas(tmp_path, [_row(STEM, AREA_CSV_HEADERS[5], 2.25)])

    m = MeasurementIndex(str(tmp_path)).lookup(STEM)

    assert m.length_mm is None
    assert m.area_mm2 == 2.25
    assert "cannot read image" in m.error
    assert "length" in m.error.lower()


def test_rewritten_csv_is_reparsed(tmp_path):
    path = _write_lengths(tmp_path, [_row(STEM, LENGTH_CSV_HEADERS[5], 1.0)])
    index = MeasurementIndex(str(tmp_path))
    assert index.lookup(STEM).length_mm == 1.0

    write_metric_csv([_row(STEM, LENGTH_CSV_HEADERS[5], 2.5)], path, LENGTH_CSV_HEADERS)
    _bump_mtime(path)

    assert index.lookup(STEM).length_mm == 2.5


def test_unchanged_csv_is_parsed_once(tmp_path, monkeypatch):
    _write_lengths(tmp_path, [_row(STEM, LENGTH_CSV_HEADERS[5], 1.0)])
    index = MeasurementIndex(str(tmp_path))
    calls = []
    from app.data_processing import measurement_index as mod

    real = mod._read_metric_csv
    monkeypatch.setattr(mod, "_read_metric_csv", lambda *a: calls.append(a) or real(*a))
    for _ in range(5):
        index.lookup(STEM)
    assert len(calls) == 1


def test_csv_created_after_first_lookup_is_found(tmp_path):
    index = MeasurementIndex(str(tmp_path), relocate_after_s=0.0)
    assert index.lookup(STEM).length_mm is None
    _write_lengths(tmp_path, [_row(STEM, LENGTH_CSV_HEADERS[5], 7.0)])
    assert index.lookup(STEM).length_mm == 7.0


def test_invalidate_forces_relocation(tmp_path):
    index = MeasurementIndex(str(tmp_path), relocate_after_s=3600.0)
    assert index.lookup(STEM).area_mm2 is None
    _write_areas(tmp_path, [_row(STEM, AREA_CSV_HEADERS[5], 4.0)])
    index.invalidate()
    assert index.lookup(STEM).area_mm2 == 4.0


def test_peek_never_parses_and_reports_needs_load(tmp_path):
    _write_lengths(tmp_path, [_row(STEM, LENGTH_CSV_HEADERS[5], 5.0)])
    index = MeasurementIndex(str(tmp_path))
    assert index.needs_load() is True
    assert index.lookup(STEM, load=False) == Measurement(None, None, None)
    index.prewarm()
    assert index.needs_load() is False
    assert index.lookup(STEM, load=False).length_mm == 5.0


def test_malformed_csv_is_ignored_not_raised(tmp_path):
    folder = tmp_path / "skeletons"
    folder.mkdir()
    (folder / "root_lengths.csv").write_bytes(b"\xff\xfe not,a,metric\x00csv\n1,2")
    assert MeasurementIndex(str(tmp_path)).lookup(STEM) == Measurement(None, None, None)


def test_outdated_when_skeleton_is_newer_than_length_csv(tmp_path):
    path = _write_lengths(tmp_path, [_row(STEM, LENGTH_CSV_HEADERS[5], 1.0)])
    skel = tmp_path / "skeletons" / f"{STEM}_fake.png"
    skel.write_bytes(b"x")
    index = MeasurementIndex(str(tmp_path))
    assert index.lookup(STEM).length_mm == 1.0  # the bar looks up first, then asks
    st = os.stat(path)
    os.utime(str(skel), ns=(st.st_atime_ns, st.st_mtime_ns - 10**9))
    assert index.is_outdated(STEM) is False
    os.utime(str(skel), ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert index.is_outdated(STEM) is True


def test_sources_name_the_csv_files(tmp_path):
    path = _write_lengths(tmp_path, [_row(STEM, LENGTH_CSV_HEADERS[5], 1.0)])
    index = MeasurementIndex(str(tmp_path))
    index.lookup(STEM)
    assert os.path.normcase(index.sources()["length"]) == os.path.normcase(os.path.normpath(path))
    assert index.sources()["area"] is None


@pytest.mark.wallclock
def test_warm_lookup_is_fast(tmp_path):
    rows = [_row(f"IMG_{i:06d}", LENGTH_CSV_HEADERS[5], i / 100) for i in range(46_000)]
    _write_lengths(tmp_path, rows)
    index = MeasurementIndex(str(tmp_path))
    index.prewarm()
    t0 = time.perf_counter()
    for i in range(200):
        index.lookup(f"IMG_{i:06d}")
    per = (time.perf_counter() - t0) / 200
    assert per < 2e-3  # measured ~0.02 ms; generous for CI


SKELETONS = ("skeletons",)


OLDER = ("skeletonizer", "test_latest", "images")


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


def test_the_gui_thread_never_parses_the_worker_prewarm_does(tmp_path):
    written = _lengths(tmp_path)
    index = MeasurementIndex(str(tmp_path))
    assert index.lookup(STEM, load=False).length_mm is None  # never parsed on the GUI thread
    assert index.needs_load() is True and index.sources()["length"] == written

    index.prewarm()  # a worker's job

    assert index.lookup(STEM, load=False).length_mm == 9.25


@pytest.mark.parametrize(
    "parts",
    [OLDER, ("output", *OLDER), ("results", "run1", "test_latest", "images")],
    ids=["skeletonizer", "output", "walked"],
)
def test_a_csv_of_the_older_layout_is_not_found(tmp_path, parts):
    _lengths(tmp_path, parts)
    index = MeasurementIndex(str(tmp_path))
    index.prewarm()

    assert index.lookup(STEM, load=False).length_mm is None
    assert index.sources()["length"] is None and index.needs_load() is False


def test_a_csv_is_found_again_after_a_job_end(tmp_path):
    """invalidate() (every Measure job end) forgets the located CSV: the GUI thread's search
    finds it again (a stat), so the bar asks for its parse."""
    written = _lengths(tmp_path)
    now = [0.0]
    index = MeasurementIndex(str(tmp_path), clock=lambda: now[0])
    index.prewarm()
    assert index.lookup(STEM, load=False).length_mm == 9.25

    index.invalidate()
    now[0] += 5.0

    assert index.needs_load() is True
    assert index.sources()["length"] == written
    index.prewarm()
    assert index.lookup(STEM, load=False).length_mm == 9.25


def test_a_csv_measure_writes_later_is_found(tmp_path):
    now = [0.0]
    index = MeasurementIndex(str(tmp_path), clock=lambda: now[0])
    index.prewarm()
    folder = tmp_path / "skeletons"
    folder.mkdir()
    row = {"Image": STEM, "Tube": 1, "Position": 44, "Date": "d", "Time": "t"}
    row[LENGTH_CSV_HEADERS[5]] = 3.5
    write_metric_csv([row], str(folder / "root_lengths.csv"), LENGTH_CSV_HEADERS)
    now[0] += 5.0

    assert index.needs_load() is True
    index.prewarm()
    assert index.lookup(STEM, load=False).length_mm == 3.5


def test_lookup_finds_a_row_whose_stem_differs_in_case(tmp_path):
    _write_lengths(tmp_path, [_row("a", LENGTH_CSV_HEADERS[5], 18.77)])
    assert MeasurementIndex(str(tmp_path)).lookup("A") == Measurement(18.77, None, None)


def test_a_lookup_prefers_the_row_of_the_exact_stem(tmp_path):
    column = LENGTH_CSV_HEADERS[5]
    _write_lengths(tmp_path, [_row("A", column, 1.5), _row("a", column, 2.5)])
    index = MeasurementIndex(str(tmp_path))
    assert index.lookup("A").length_mm == 1.5 and index.lookup("a").length_mm == 2.5


def test_a_lookup_never_guesses_between_case_variant_rows(tmp_path):
    column = LENGTH_CSV_HEADERS[5]
    _write_lengths(tmp_path, [_row("Ab", column, 1.5), _row("aB", column, 2.5)])
    assert MeasurementIndex(str(tmp_path)).lookup("AB") == Measurement(None, None, None)
