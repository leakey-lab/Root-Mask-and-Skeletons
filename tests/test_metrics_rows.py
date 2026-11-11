"""The per-image rows of ``app.inference.metrics``: ``length_row`` / ``area_row``.

One row per image: the value rounded to 2 dp and the fields of the image name
(``metrics.parse_image_name``, the dict shape of :func:`app.naming.parse_image_name`); on any
failure a 0 row whose ``Error`` names the file only, never its folder (O38). The private keys
(``_size_px``, ``_non_binary``) are never CSV columns. The run's memo (small files that
measured 0) changes no row and keeps no failure; the first failures of a run are logged one by
one, the rest counted (C6-02).
"""

from __future__ import annotations

import json
import logging

import cv2
import numpy as np
import pytest

from app import provenance
from app.inference import measure_run, metrics
from app.inference.measure_run import AREA, LENGTH, MeasureRun
from tests.measure_support import _NAMES, _legacy_parse, _make_dataset


@pytest.mark.parametrize("name", _NAMES + ["scan_T005_L030_2024.03.15_143022_fake.png"])
def test_parse_image_name_matches_legacy_for_canonical_names(name):
    assert metrics.parse_image_name(name) == _legacy_parse(name)


@pytest.mark.parametrize("spec", [LENGTH, AREA], ids=["length", "area"])
def test_non_ascii_folder_measures_same_values(tmp_path, spec):
    ascii_images = _make_dataset(tmp_path / "plain", spec.kind, _NAMES[:2])
    uni_images = _make_dataset(tmp_path / "Données_根系", spec.kind, _NAMES[:2])
    for name in _NAMES[:2]:
        a = spec.row(name, ascii_images[name])
        u = spec.row(name, uni_images[name])
        assert "Error" not in u
        assert u == a
        assert u[spec.value_column] > 0


def _fake_png(tmp_path):
    p = tmp_path / "deep" / "dir" / "x_T1_L1_fake.png"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"not a png")
    return p


def test_an_unreadable_file_names_only_the_file(tmp_path):
    from app.inference import metrics

    p = _fake_png(tmp_path)
    assert metrics.length_row("x", str(p))["Error"] == "Could not read image: x_T1_L1_fake.png"
    assert metrics.area_row("x", str(p))["Error"] == "Could not read mask: x_T1_L1_fake.png"


def test_an_os_error_loses_its_folder(tmp_path, monkeypatch):
    from app.inference import metrics

    p = _fake_png(tmp_path)

    def boom(path, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", str(p))

    monkeypatch.setattr(metrics, "preprocess_image", boom)
    error = metrics.length_row("x", str(p))["Error"]
    assert "x_T1_L1_fake.png" in error
    assert str(tmp_path) not in error
    assert repr(str(tmp_path))[1:-1] not in error


def test_an_error_row_has_no_size(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not a png")
    row = metrics.area_row("bad", str(bad))
    assert "Error" in row and "_size_px" not in row
    assert measure_run.sizes_px([row]) == {}


def _same_files(folder, names, data, suffix):
    """One file per name, every one holding ``data``: ``name -> path``."""
    folder.mkdir(parents=True, exist_ok=True)
    images = {}
    for name in names:
        path = folder / f"{name}{suffix}"
        path.write_bytes(data)
        images[name] = str(path)
    return images


def _serial_pool(monkeypatch):
    """Measure the run's images one after another (pool threads that miss the memo at the
    same moment each decode: harmless, but it makes a call count depend on the timing)."""

    def serial(images, worker, progress_cb=None, should_stop=None):
        rows = [worker(name, path) for name, path in images.items()]
        return sorted(rows, key=metrics._sort_key)

    monkeypatch.setattr(measure_run, "run_metric_pool", serial)


def _counting(monkeypatch, name):
    """Count the calls of ``metrics.<name>`` (the decode-and-measure step of a row)."""
    calls = []
    real = getattr(metrics, name)

    def counted(*args, **kwargs):
        calls.append(args[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(metrics, name, counted)
    return calls


def test_byte_identical_empty_files_are_measured_once_per_run(tmp_path, monkeypatch):
    """The empty predictions of a project are one file content: a run decodes it once, and
    every row is the row of its own file measured alone."""
    empty = cv2.imencode(".png", np.zeros((256, 341), np.uint8))[1].tobytes()
    assert len(empty) <= metrics._MEMO_MAX_BYTES
    images = _same_files(tmp_path / "skeletons", _NAMES[:4], empty, "_fake.png")
    images.update(_make_dataset(tmp_path / "skeletons", "length", _NAMES[4:]))
    expected = sorted((LENGTH.row(n, p) for n, p in images.items()), key=metrics._sort_key)
    _serial_pool(monkeypatch)
    calls = _counting(monkeypatch, "preprocess_image")

    run = MeasureRun(LENGTH, images, str(tmp_path / "skeletons"), project_root=str(tmp_path))
    run.compute()

    assert len(calls) == 1 + len(_NAMES[4:])  # the empty content once, each other file once
    assert [{k: v for k, v in row.items() if k != "Source"} for row in run.rows] == expected
    assert run._memo is None  # nothing kept on the run (a failed write keeps it for Retry)


def test_the_memo_keeps_grey_levels_and_skips_large_files(tmp_path, monkeypatch):
    """A grey-level mask that measures 0 keeps its private ``_non_binary`` key on every row;
    a file above the size limit is decoded each time."""
    grey = cv2.imencode(".png", np.full((48, 64), 100, np.uint8))[1].tobytes()
    big = cv2.imencode(".png", np.zeros((256, 341), np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 0])[
        1
    ].tobytes()
    assert len(grey) <= metrics._MEMO_MAX_BYTES < len(big)
    images = _same_files(tmp_path / "mask", _NAMES[:3], grey, ".png")
    images.update(_same_files(tmp_path / "mask", _NAMES[3:], big, ".png"))
    _serial_pool(monkeypatch)
    calls = _counting(monkeypatch, "_load_mask_and_grey_flag")

    run = MeasureRun(AREA, images, str(tmp_path / "mask"), project_root=str(tmp_path))
    run.compute()
    csv_path = run.write()

    assert len(calls) == 1 + len(_NAMES[3:])
    assert sorted(r["Image"] for r in run.rows if r.get("_non_binary")) == sorted(_NAMES[:3])
    record = json.loads(open(provenance.sidecar_path(csv_path), encoding="utf-8").read())
    assert record["outputs"]["n_non_binary"] == 3
    assert record["outputs"]["n_zero"] == len(_NAMES)


def test_the_pool_threads_share_the_memo_and_give_the_same_rows(tmp_path, monkeypatch):
    empty = cv2.imencode(".png", np.zeros((256, 341), np.uint8))[1].tobytes()
    names = [f"img{i:03d}" for i in range(64)]
    images = _same_files(tmp_path / "skeletons", names, empty, "_fake.png")
    expected = sorted((LENGTH.row(n, p) for n, p in images.items()), key=metrics._sort_key)
    calls = _counting(monkeypatch, "preprocess_image")
    run = MeasureRun(LENGTH, images, str(tmp_path / "skeletons"))
    run.compute()
    assert [{k: v for k, v in row.items() if k != "Source"} for row in run.rows] == expected
    assert 1 <= len(calls) < len(names)


def test_failures_are_not_kept_so_each_error_names_its_own_file(tmp_path):
    images = _same_files(tmp_path / "skeletons", _NAMES[:3], b"not a png", "_fake.png")
    run = MeasureRun(LENGTH, images, str(tmp_path / "skeletons"), project_root=str(tmp_path))
    run.compute()
    errors = {row["Image"]: row["Error"] for row in run.rows}
    assert errors == {name: f"Could not read image: {name}_fake.png" for name in _NAMES[:3]}


def test_a_stopped_run_keeps_no_memo(tmp_path):
    empty = cv2.imencode(".png", np.zeros((48, 64), np.uint8))[1].tobytes()
    images = _same_files(tmp_path / "skeletons", _NAMES, empty, "_fake.png")
    run = MeasureRun(LENGTH, images, str(tmp_path / "skeletons"))
    assert run.compute(should_stop=lambda: True) is None
    assert run._memo is None and run.rows is None


def test_a_run_logs_its_first_rows_with_an_error_and_counts_the_rest(tmp_path, caplog):
    """C6-02: every unreadable input of a run is an Error row, as before; the log names the
    first ones (each with its own line in ``_metric_row``) and one line counts the rest."""
    from app.inference.batch_pipeline import FAILURE_LOG_LIMIT

    names = [f"bad{i:03d}" for i in range(FAILURE_LOG_LIMIT + 30)]
    images = _same_files(tmp_path / "skeletons", names, b"not a png", "_fake.png")
    run = MeasureRun(LENGTH, images, str(tmp_path / "skeletons"), project_root=str(tmp_path))

    with caplog.at_level(logging.DEBUG, logger=metrics.logger.name):
        run.compute()

    errors = {row["Image"]: row["Error"] for row in run.rows}
    assert errors == {name: f"Could not read image: {name}_fake.png" for name in names}
    records = [r for r in caplog.records if r.name == metrics.logger.name]
    failed = [r for r in records if r.levelno == logging.ERROR]
    assert len(failed) == FAILURE_LOG_LIMIT
    assert {r.funcName for r in failed} == {"_metric_row"}
    assert all(r.getMessage().startswith("Length metric failed for 'bad") for r in failed)
    assert [r.getMessage() for r in records if r.levelno != logging.ERROR] == [
        "30 more rows with an Error not logged one by one"
    ]


def test_a_row_measured_outside_a_run_logs_its_error(tmp_path, caplog):
    bad = tmp_path / "x_fake.png"
    bad.write_bytes(b"not a png")

    with caplog.at_level(logging.DEBUG, logger=metrics.logger.name):
        row = metrics.length_row("x", str(bad))

    assert row["Error"] == "Could not read image: x_fake.png"
    assert [(r.levelno, r.getMessage()) for r in caplog.records] == [
        (logging.ERROR, "Length metric failed for 'x': Could not read image: x_fake.png")
    ]


def test_rows_without_a_memo_are_the_same(tmp_path):
    """``memo`` changes no row: the same bytes measured with and without it."""
    empty = cv2.imencode(".png", np.zeros((48, 64), np.uint8))[1].tobytes()
    for spec, suffix in ((LENGTH, "_fake.png"), (AREA, ".png")):
        path = tmp_path / f"{spec.kind}{suffix}"
        path.write_bytes(empty)
        memo: dict = {}
        plain = spec.row(_NAMES[0], str(path), data=empty)
        assert spec.row(_NAMES[0], str(path), data=empty, memo=memo) == plain
        assert len(memo) == 1
        assert spec.row(_NAMES[1], str(path), data=empty, memo=memo) == spec.row(
            _NAMES[1], str(path), data=empty
        )
