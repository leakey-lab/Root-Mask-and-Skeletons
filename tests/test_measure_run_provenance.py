"""The provenance sidecar of a Measure CSV (``<csv>.provenance.json``, FR-02) and the record of
its inputs (``<csv>.inputs.json``, R5), as ``MeasureRun.write()`` writes them.

The sidecar names the run: the inputs (folder, counts, names digest, the decoded sizes), the
parameters (the source rule, the binarization), the outputs (columns and counts) and the field
of view. Writing it never changes a CSV byte, and its failure fails nothing; a failed sidecar
or record removes the earlier one, so a CSV is never paired with another run's record.
"""

from __future__ import annotations

import csv
import json
import os

import cv2
import numpy as np
import pytest

from app import provenance
from app.config import LENGTH_CSV_HEADERS
from app.data_processing.measurement_index import MeasurementIndex
from app.inference import measure_inputs, measure_run, metrics
from app.inference.measure_run import AREA, LENGTH, MeasureRun
from app.io_utils import imread_unicode
from app.project_layout import ProjectLayout
from tests.measure_sources_support import (
    PHOTO_SIZE,
    _write,
    make_gt_project,
    write_gt_mask,
    write_photo,
)
from tests.measure_sources_support import _write_png as _write_png_bytes
from tests.measure_support import _NAMES, _csv_path, _make_dataset, _run, _write_png


def test_write_adds_provenance_sidecar_without_touching_csv(tmp_path):
    images = _make_dataset(tmp_path / "skeletons", "length", _NAMES[:3])
    bad = tmp_path / "skeletons" / "x_fake.png"
    bad.write_bytes(b"")
    images["x"] = str(bad)
    run = MeasureRun(LENGTH, images, str(tmp_path / "skeletons"), project_root=str(tmp_path))
    run.compute()
    csv_path = run.write()
    first = open(csv_path, "rb").read()

    sidecar = provenance.sidecar_path(csv_path)
    assert os.path.basename(sidecar) == "root_lengths.csv.provenance.json"
    record = json.loads(open(sidecar, encoding="utf-8").read())
    assert record["kind"] == "root_length"
    assert record["calibration"] == {
        "fov_width_mm": 18.0,
        "fov_height_mm": 13.5,
        "source": "default",
    }
    assert record["inputs"]["folder"] == "skeletons"
    assert record["inputs"]["n_images"] == 4
    assert record["inputs"]["names_sha256"] == provenance.names_digest(images)
    assert record["outputs"]["csv"] == "root_lengths.csv"
    assert record["outputs"]["columns"] == LENGTH_CSV_HEADERS
    assert record["outputs"]["n_rows"] == 4
    assert record["outputs"]["n_errors"] == 1
    assert "D:" not in json.dumps(record["inputs"]) and str(tmp_path) not in json.dumps(record)

    os.remove(sidecar)
    run.write()
    assert open(csv_path, "rb").read() == first


def test_sidecar_failure_does_not_fail_the_measurement(tmp_path, monkeypatch):
    images = _make_dataset(tmp_path / "in", "area", _NAMES[:1])

    def broken(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(measure_run.provenance, "write_sidecar", broken)
    run = MeasureRun(AREA, images, str(tmp_path))
    run.compute()
    assert os.path.isfile(run.write())


def test_the_record_does_not_touch_the_csv(tmp_path, monkeypatch):
    """The ``<csv>.inputs.json`` step (R5) changes no CSV byte, and its failure fails
    nothing."""
    images = _make_dataset(tmp_path / "in", "area", _NAMES[:3])
    recorded = MeasureRun(AREA, images, str(tmp_path / "recorded"), project_root=str(tmp_path))
    os.makedirs(recorded.output_dir)
    recorded.compute()
    with_record = open(recorded.write(), "rb").read()
    assert os.path.isfile(measure_inputs.inputs_record_path(recorded.csv_path))

    def broken(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(measure_inputs, "write_inputs_record", broken)
    plain = MeasureRun(AREA, images, str(tmp_path / "plain"), project_root=str(tmp_path))
    os.makedirs(plain.output_dir)
    plain.compute()
    without_record = open(plain.write(), "rb").read()
    assert not os.path.exists(measure_inputs.inputs_record_path(plain.csv_path))
    assert without_record == with_record
    # Measured again with the same bytes: the earlier run's record goes (it may differ).
    assert open(recorded.write(), "rb").read() == with_record
    assert not os.path.exists(measure_inputs.inputs_record_path(recorded.csv_path))


def _masks_of_sizes(folder, sizes):
    """One mask per ``(name, (rows, cols))``: a white block in a black image."""
    folder.mkdir(parents=True, exist_ok=True)
    images = {}
    for name, shape in sizes:
        img = np.zeros(shape, np.uint8)
        img[4:20, 4:30] = 255
        _write_png(folder / f"{name}.png", img)
        images[name] = str(folder / f"{name}.png")
    return images


def test_the_sidecar_records_the_measured_sizes(tmp_path):
    images = _masks_of_sizes(
        tmp_path / "mask", [(_NAMES[0], (64, 80)), (_NAMES[1], (64, 80)), (_NAMES[2], (48, 96))]
    )
    run = MeasureRun(AREA, images, str(tmp_path / "mask"), project_root=str(tmp_path))
    run.compute()
    assert all("_size_px" in row for row in run.rows)
    csv_path = run.write()

    record = json.loads(open(provenance.sidecar_path(csv_path), encoding="utf-8").read())
    assert record["inputs"]["sizes_px"] == {"80x64": 2, "96x48": 1}
    assert list(record["inputs"]["sizes_px"]) == ["80x64", "96x48"]  # most common first

    # The private key is never a column: the CSV is the one of rows without it.
    stripped = tmp_path / "stripped.csv"
    metrics.write_metric_csv(
        [{k: v for k, v in row.items() if k != "_size_px"} for row in run.rows],
        str(stripped),
        list(AREA.headers),
    )
    assert open(csv_path, "rb").read() == stripped.read_bytes()
    assert "_size_px" not in open(csv_path, encoding="utf-8").readline()


def test_sizes_px_counts_the_rows_most_common_first():
    rows = [{"_size_px": "10x5"}, {"_size_px": "20x5"}, {"_size_px": "20x5"}, {"Error": "x"}]
    assert measure_run.sizes_px(rows) == {"20x5": 2, "10x5": 1}
    assert list(measure_run.sizes_px(rows)) == ["20x5", "10x5"]


def test_mixed_sizes_are_looked_for_within_a_source_group():
    """Photo-size GT files beside model-size predictions are not mixed (every file spans its
    photo's frame); two sizes among the GT files, or among the predictions, are."""
    gt = lambda size: {"Source": "gt", "_size_px": size}  # noqa: E731
    pred = lambda size: {"Source": "prediction", "_size_px": size}  # noqa: E731
    assert measure_run.mixed_sizes([gt("640x480"), gt("640x480"), pred("341x256")]) is None
    assert measure_run.mixed_sizes([gt("640x480"), gt("320x240"), gt("320x240")]) == {
        "320x240": 2,
        "640x480": 1,
    }
    assert measure_run.mixed_sizes([pred("341x256"), pred("100x100")]) is not None
    # A row without a Source is a prediction; GT is looked at first.
    assert measure_run.mixed_sizes([{"_size_px": "1x1"}, {"_size_px": "2x2"}]) is not None
    both = [gt("2x2"), gt("3x3"), pred("1x1"), pred("5x5"), pred("5x5")]
    assert set(measure_run.mixed_sizes(both)) == {"2x2", "3x3"}
    assert measure_run.mixed_sizes([]) is None


def test_the_non_binary_count_is_recorded(tmp_path):
    folder = tmp_path / "mask"
    images = _masks_of_sizes(folder, [(_NAMES[0], (64, 80)), (_NAMES[1], (64, 80))])
    grey = cv2.imread(images[_NAMES[1]], cv2.IMREAD_GRAYSCALE)
    grey[0, 0] = 100
    _write_png(folder / f"{_NAMES[1]}.png", grey)

    run = MeasureRun(AREA, images, str(folder), project_root=str(tmp_path))
    run.compute()
    assert sum(1 for row in run.rows if row.get("_non_binary")) == 1
    csv_path = run.write()

    record = json.loads(open(provenance.sidecar_path(csv_path), encoding="utf-8").read())
    assert record["outputs"]["n_non_binary"] == 1
    assert record["parameters"]["binarization"] == "gray > 127"

    stripped = tmp_path / "stripped.csv"
    metrics.write_metric_csv(
        [{k: v for k, v in row.items() if k != "_non_binary"} for row in run.rows],
        str(stripped),
        list(AREA.headers),
    )
    assert open(csv_path, "rb").read() == stripped.read_bytes()


def test_length_records_carry_no_non_binary_count(tmp_path):
    images = _make_dataset(tmp_path / "skeletons", "length")
    run = MeasureRun(LENGTH, images, str(tmp_path / "skeletons"), project_root=str(tmp_path))
    run.compute()
    csv_path = run.write()
    record = json.loads(open(provenance.sidecar_path(csv_path), encoding="utf-8").read())
    assert "n_non_binary" not in record["outputs"]
    assert not any("_non_binary" in row for row in run.rows)


@pytest.fixture
def gt_project(tmp_path) -> tuple[ProjectLayout, dict[str, str]]:
    return make_gt_project(str(tmp_path))


def _sidecar(csv_path: str) -> dict:
    with open(csv_path + ".provenance.json", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.mark.usefixtures("no_prompts")
def test_a_failed_provenance_write_removes_the_earlier_sidecar(gt_project, qapp, monkeypatch):
    layout, _ = gt_project
    _run(layout, qapp, "length")
    csv_path = _csv_path(layout, "length")
    assert _sidecar(csv_path)["inputs"]["gt"]["skipped"]

    def broken(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(measure_run.provenance, "write_sidecar", broken)
    # Force a re-measure: nothing changed, so the run would stop at "Measure again?".
    os.remove(csv_path + ".inputs.json")
    _run(layout, qapp, "length")

    assert os.path.isfile(csv_path)
    # an earlier run's inputs.gt.skipped can never pair with the new CSV
    assert not os.path.exists(csv_path + ".provenance.json")
    index = MeasurementIndex(layout.root)
    index.prewarm()
    assert index.gt_skip_reason("s") == {}


@pytest.mark.parametrize(
    "spec, words",
    [(measure_run.LENGTH, "generated skeletons (*_fake.png)"), (measure_run.AREA, "masks (*.png)")],
    ids=["length", "area"],
)
def test_a_run_without_gt_inputs_records_the_prediction_rule(tmp_path, spec, words):
    """A run built without gt_files (scripts, tests) measures predictions only: its
    provenance says so."""
    images = _make_dataset(tmp_path / "in", spec.kind)
    run = measure_run.MeasureRun(spec, images, str(tmp_path / "out"))
    run.compute()
    rec = _sidecar(run.write())
    assert (rec["parameters"]["source_rule"], rec["parameters"]["input"]) == ("prediction", words)
    assert "gt" not in rec["inputs"]

    gt_first = measure_run.MeasureRun(spec, images, str(tmp_path / "gt"), gt_files={})
    gt_first.compute()
    assert _sidecar(gt_first.write())["parameters"]["source_rule"] == "gt-first"


@pytest.mark.usefixtures("no_prompts")
def test_the_sidecar_agrees_with_the_csv_beside_it(tmp_path, qapp):
    """Every count of the sidecar, recomputed from the CSV read back from disk (and, for the
    grey levels, from the files its rows name), on a project with GT, prediction, error, zero
    and grey-level rows."""
    layout, _ = make_gt_project(str(tmp_path))
    write_photo(layout, "e")
    _write(layout.mask_path("e"), b"not a png")  # an Error row
    write_gt_mask(layout, "p", np.zeros(PHOTO_SIZE[::-1], bool))  # an emptied GT mask: 0 mm²
    grey = imread_unicode(layout.mask_path("g"), cv2.IMREAD_GRAYSCALE)
    grey[0, :10] = 100
    _write_png_bytes(layout.mask_path("g"), grey)  # a prediction with grey levels
    _run(layout, qapp, "area")
    csv_path = _csv_path(layout, "area")

    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    record = provenance.read_sidecar(csv_path)
    value = AREA.value_column

    def non_binary(row) -> bool:
        stem = row["Image"]
        path = layout.gt_mask_path(stem) if row["Source"] == "gt" else layout.mask_path(stem)
        pixels = imread_unicode(path, cv2.IMREAD_GRAYSCALE)
        return bool(np.any((pixels != 0) & (pixels != 255)))

    expected = {
        "n_rows": len(rows),
        "n_errors": sum(1 for row in rows if row["Error"]),
        "n_zero": sum(1 for row in rows if float(row[value]) == 0),
        "n_gt": sum(1 for row in rows if row["Source"] == "gt"),
        "n_prediction": sum(1 for row in rows if row["Source"] == "prediction"),
        "n_non_binary": sum(1 for row in rows if not row["Error"] and non_binary(row)),
    }
    assert {key: record["outputs"][key] for key in expected} == expected
    assert record["outputs"]["columns"] == list(rows[0])
    assert record["inputs"]["n_images"] == len(rows)
    assert record["inputs"]["names_sha256"] == provenance.names_digest(r["Image"] for r in rows)
    # the project holds every kind of row, so no count above is trivially 0
    assert expected["n_errors"] == 1 and expected["n_non_binary"] == 1
    assert expected["n_zero"] == 2 and expected["n_gt"] >= 2 and expected["n_prediction"] >= 2
