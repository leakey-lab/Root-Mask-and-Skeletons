"""One Measure run (``MeasureRun``): the CSV it writes, byte for byte.

``test_measure_csv_is_byte_identical_to_legacy_pipeline`` is a differential test: the CSV
must equal, byte for byte, the one a copy of the pre-refactor pipeline writes
(``tests/measure_support._legacy_*``: ``cv2.imread``, the legacy name regexes, a plain
``csv.DictWriter``), whose length comes from the independent pure-Python reference
``tests/length_reference.py`` (O12) and whose last column is ``Source``. The real-sample
variant runs it on the owner's data when present. These pin the values of the paper: their
expected values never change.

Also: the CSV is replaced atomically (RB-03 / FR-06), a stopped ``compute()`` keeps no rows, and
kept rows are not measured again (the Retry of a failed write).
"""

from __future__ import annotations

import os

import pytest

from app.config import AREA_CSV_HEADERS, LENGTH_CSV_HEADERS
from app.inference import measure_run, metrics
from app.inference.measure_run import AREA, LENGTH, MeasureRun
from tests.measure_support import _NAMES, _legacy_csv, _make_dataset
from tests.real_data import real_data_dir


@pytest.mark.parametrize("spec", [LENGTH, AREA], ids=["length", "area"])
def test_measure_csv_is_byte_identical_to_legacy_pipeline(tmp_path, spec):
    images = _make_dataset(tmp_path / "in", spec.kind)
    # plus a corrupt file and a name with no metadata (Error row / None fields)
    bad = tmp_path / "in" / "corrupt_T001_L001.png"
    bad.write_bytes(b"not a png")
    images["corrupt_T001_L001"] = str(bad)
    images["IMG_0001"] = images[_NAMES[0]]

    legacy = _legacy_csv(images, spec.kind, str(tmp_path / "legacy.csv"))
    out = tmp_path / "out"
    out.mkdir()
    run = MeasureRun(spec, images, str(out))
    run.compute()
    csv_path = run.write()

    assert os.path.basename(csv_path) == spec.csv_name
    with open(legacy, "rb") as a, open(csv_path, "rb") as b:
        assert b.read() == a.read()


REAL = real_data_dir()


@pytest.mark.skipif(REAL is None, reason="real dataset not present")
@pytest.mark.parametrize("spec", [LENGTH, AREA], ids=["length", "area"])
def test_real_sample_is_byte_identical_to_legacy_pipeline(tmp_path, spec):
    sub, suffix = ("skeletons", "_fake.png") if spec.kind == "length" else ("mask", ".png")
    folder = os.path.join(REAL, sub)
    names = sorted(e.name for e in os.scandir(folder) if e.name.endswith(suffix))
    sample = names[:: max(1, len(names) // 60)][:60]
    images = {n[: -len(suffix)]: os.path.join(folder, n) for n in sample}
    legacy = _legacy_csv(images, spec.kind, str(tmp_path / "legacy.csv"), source=False)
    run = MeasureRun(spec, images, str(tmp_path), gt_files={})  # GT-first; no gt/ there
    run.compute()
    with open(legacy, "rb") as a, open(run.write(), "rb") as b:
        legacy_lines, lines = a.read().split(b"\r\n"), b.read().split(b"\r\n")
    # WG-T8: the first seven columns byte for byte, and the eighth is prediction everywhere
    assert [line.rsplit(b",", 1)[0] for line in lines] == legacy_lines
    assert [line.rsplit(b",", 1)[-1] for line in lines] == (
        [b"Source"] + [b"prediction"] * len(images) + [b""]
    )


@pytest.mark.parametrize("spec", [measure_run.LENGTH, measure_run.AREA], ids=["length", "area"])
def test_prediction_rows_are_byte_identical_to_the_legacy_writer_in_the_first_seven_columns(
    tmp_path, spec
):
    images = _make_dataset(tmp_path / "in", spec.kind)
    bad = tmp_path / "in" / "corrupt_T001_L001.png"
    bad.write_bytes(b"not a png")
    images["corrupt_T001_L001"] = str(bad)
    legacy = _legacy_csv(images, spec.kind, str(tmp_path / "legacy.csv"), source=False)
    out = tmp_path / "out"
    out.mkdir()
    run = measure_run.MeasureRun(spec, images, str(out), gt_files={}, photos={})  # GT-first, no GT
    run.compute()

    with open(run.write(), "rb") as fh:
        lines = fh.read().split(b"\r\n")
    with open(legacy, "rb") as fh:
        legacy_lines = fh.read().split(b"\r\n")
    assert [line.rsplit(b",", 1)[0] for line in lines if line] == [
        line for line in legacy_lines if line
    ]
    assert [line.rsplit(b",", 1)[1] for line in lines if line] == [b"Source"] + [b"prediction"] * (
        len(images)
    )


def test_the_csvs_gain_a_last_source_column():
    assert LENGTH_CSV_HEADERS == [
        "Image",
        "Tube",
        "Position",
        "Date",
        "Time",
        "Length (mm)",
        "Error",
        "Source",
    ]
    assert AREA_CSV_HEADERS == [
        "Image",
        "Tube",
        "Position",
        "Date",
        "Time",
        "Area (mm²)",
        "Error",
        "Source",
    ]


def test_failed_csv_replace_leaves_previous_csv_and_no_temp(tmp_path, monkeypatch):
    target = tmp_path / "root_lengths.csv"
    target.write_bytes(b"previous,complete\r\n1,2\r\n")

    def boom(src, dst):
        raise PermissionError(13, "Permission denied", dst)

    monkeypatch.setattr("app.io_utils.os.replace", boom)
    rows = [{"Image": "a", "Tube": 1, "Length (mm)": 1.0}]
    with pytest.raises(OSError):
        metrics.write_metric_csv(rows, str(target), LENGTH_CSV_HEADERS)
    assert target.read_bytes() == b"previous,complete\r\n1,2\r\n"
    assert sorted(os.listdir(tmp_path)) == ["root_lengths.csv"]


def test_cancelled_compute_keeps_no_rows(tmp_path):
    images = _make_dataset(tmp_path / "in", "length", _NAMES[:3])
    run = MeasureRun(LENGTH, images, str(tmp_path))
    run.compute(should_stop=lambda: True)
    assert run.rows is None


def test_compute_is_skipped_when_rows_are_already_known(tmp_path, monkeypatch):
    images = _make_dataset(tmp_path / "in", "length", _NAMES[:2])
    run = MeasureRun(LENGTH, images, str(tmp_path))
    run.compute()
    monkeypatch.setattr(measure_run, "run_metric_pool", pytest.fail)
    run.compute()  # retry path: rows reused, nothing recomputed
    assert len(run.rows) == 2


# --------------------------------------------------------------------------- inputs that vanish
def test_an_input_that_cannot_be_opened_still_gets_its_error_row(tmp_path):
    """A file listed by the scan but gone (or locked) when the pool reaches it: its image keeps
    its row, an Error row naming the file only, and no fingerprint is recorded for it."""
    images = _make_dataset(tmp_path / "skeletons", "length", _NAMES[:3])
    gone = tmp_path / "skeletons" / f"{_NAMES[3]}_fake.png"
    images[_NAMES[3]] = str(gone)  # listed, never written: it cannot be opened
    run = MeasureRun(LENGTH, images, str(tmp_path / "skeletons"), project_root=str(tmp_path))

    run.compute()
    csv_path = run.write()

    rows = {row["Image"]: row for row in run.rows}
    assert sorted(rows) == sorted(_NAMES[:4])
    assert rows[_NAMES[3]]["Error"] == f"Could not read image: {gone.name}"
    assert rows[_NAMES[3]][LENGTH.value_column] == 0
    assert not any(rows[name].get("Error") for name in _NAMES[:3])
    assert set(run.fingerprints) == {f"skeletons/{name}_fake.png" for name in _NAMES[:3]}
    with open(csv_path, encoding="utf-8") as fh:
        assert sum(1 for line in fh if line.startswith(_NAMES[3] + ",")) == 1


def test_write_before_compute_raises_and_writes_nothing(tmp_path):
    run = MeasureRun(AREA, _make_dataset(tmp_path / "mask", "area", _NAMES[:1]), str(tmp_path))
    with pytest.raises(RuntimeError, match="before compute"):
        run.write()
    assert not os.path.exists(run.csv_path)
