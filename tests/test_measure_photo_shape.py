"""Measure checks the shape of the project's photos against its field of view's frame.

The field of view is mapped onto each measured file's whole frame, and a prediction is
always written at the model's size: only the photos tell whether the field of view's frame
(640 x 480 for the default, the project file's ``reference_size_px`` else) fits. A sample
of the photos of the measured images is checked from their headers on the job's worker,
recorded as ``inputs.photos`` in the provenance, and Measure warns once when photos do not
fit. The check changes no row, no CSV byte and not ``unchanged()``.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from app import calibration, config, provenance
from app.calibration import ProjectCalibration, load_project_calibration, reference_frame
from app.config import Calibration
from app.feedback import RecordingFeedback
from app.handlers.job_registry import JobRegistry
from app.handlers.skeleton_handler import SkeletonHandler
from app.inference import measure_run, measure_sources
from app.inference.measure_run import AREA, PHOTO_SHAPE_SAMPLE, MeasureRun

# --------------------------------------------------------------------------- reference_frame


def test_reference_frame_of_the_default_is_the_cameras_frame():
    assert config.CALIBRATION_REFERENCE_SIZE_PX == (640, 480)
    assert reference_frame(calibration.default_calibration()) == (640, 480)


def test_reference_frame_of_a_project_file(tmp_path):
    with_ref = tmp_path / "a"
    without = tmp_path / "b"
    with_ref.mkdir()
    without.mkdir()
    (with_ref / "sprouts_project.json").write_text(
        '{"schema": 1, "calibration": {"fov_width_mm": 20, "fov_height_mm": 13.6, '
        '"reference_size_px": [754, 510]}}',
        encoding="utf-8",
    )
    (without / "sprouts_project.json").write_text(
        '{"schema": 1, "calibration": {"fov_width_mm": 20, "fov_height_mm": 13.6}}',
        encoding="utf-8",
    )
    assert reference_frame(load_project_calibration(with_ref)) == (754, 510)
    assert reference_frame(load_project_calibration(without)) is None
    assert reference_frame(ProjectCalibration(Calibration(1, 1), "project")) is None


def test_reference_frame_of_an_invalid_project_file_is_the_defaults(tmp_path):
    (tmp_path / "sprouts_project.json").write_text("{", encoding="utf-8")
    pc = load_project_calibration(tmp_path)
    assert pc.problem and pc.source == "default"
    assert reference_frame(pc) == (640, 480)


# --------------------------------------------------------------------------- the sample


def _masks(folder, names, shape=(6, 8)):
    """A small binary mask per name in ``folder``; ``name -> path``."""
    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    for i, name in enumerate(names):
        img = np.zeros(shape, np.uint8)
        img[1, : 1 + i % shape[1]] = 255
        path = folder / f"{name}.png"
        cv2.imwrite(str(path), img)
        out[name] = str(path)
    return out


def _photo(folder, name, shape, ext=".jpg"):
    """A photo of ``shape`` (rows, cols); its path."""
    path = folder / f"{name}{ext}"
    cv2.imwrite(str(path), np.zeros(shape, np.uint8))
    return str(path)


class _SizeSpy:
    """Stands in for ``measure_run.image_size``: every photo is ``size``; records paths."""

    def __init__(self, size=(640, 480)):
        self.size = size
        self.paths: list[str] = []

    def __call__(self, path):
        self.paths.append(path)
        return self.size


def _run(tmp_path, images, photos, **kwargs):
    run = MeasureRun(AREA, images, str(tmp_path / "mask"), photos=photos, **kwargs)
    assert run.compute() is not None
    return run


def test_fewer_photos_than_the_limit_are_all_checked(tmp_path):
    names = [f"p{i}" for i in range(3)]
    images = _masks(tmp_path / "mask", names)
    photos = {n: _photo(tmp_path, n, (480, 640)) for n in names}
    run = _run(tmp_path, images, photos)
    assert run.photo_shapes == {
        "n_photos": 3,
        "n_checked": 3,
        "n_unreadable": 0,
        "sizes_px": {"640x480": 3},
    }


def test_more_photos_than_the_limit_give_an_even_sample(tmp_path, monkeypatch):
    names = [f"p{i:03d}" for i in range(PHOTO_SHAPE_SAMPLE + 37)]
    images = _masks(tmp_path / "mask", names)
    photos = {n: str(tmp_path / f"{n}.jpg") for n in names}  # never opened: the spy reads
    spy = _SizeSpy()
    monkeypatch.setattr(measure_run, "image_size", spy)
    first = _run(tmp_path, images, photos)
    sample = list(spy.paths)
    assert len(sample) == PHOTO_SHAPE_SAMPLE == 200
    assert len(set(sample)) == PHOTO_SHAPE_SAMPLE
    assert sample[0] == photos[names[0]] and sample[-1] == photos[names[-1]]
    assert first.photo_shapes["n_photos"] == len(names)
    assert first.photo_shapes["n_checked"] == PHOTO_SHAPE_SAMPLE

    spy.paths.clear()
    _run(tmp_path, dict(reversed(images.items())), photos)  # the same project, another order
    assert spy.paths == sample


def test_photos_of_images_without_a_row_are_left_out(tmp_path, monkeypatch):
    images = _masks(tmp_path / "mask", ["a", "b"])
    photos = {n: str(tmp_path / f"{n}.jpg") for n in ("a", "b", "c", "d")}
    spy = _SizeSpy()
    monkeypatch.setattr(measure_run, "image_size", spy)
    run = _run(tmp_path, images, photos)
    assert sorted(spy.paths) == [photos["a"], photos["b"]]
    assert run.photo_shapes["n_photos"] == 2


def test_stems_that_differ_in_case_are_paired(tmp_path):
    images = _masks(tmp_path / "mask", ["img_one"])
    photos = {"IMG_One": _photo(tmp_path, "IMG_One", (460, 640))}
    run = _run(tmp_path, images, photos)
    assert run.photo_shapes["n_photos"] == 1
    assert run.photo_shapes["sizes_px"] == {"640x460": 1}


def test_a_jpeg_and_a_png_are_read_and_an_unreadable_photo_is_counted(tmp_path):
    images = _masks(tmp_path / "mask", ["a", "b", "c", "d"])
    bad = tmp_path / "c.jpg"
    bad.write_bytes(b"not a photo")
    photos = {
        "a": _photo(tmp_path, "a", (480, 640), ".jpg"),
        "b": _photo(tmp_path, "b", (460, 640), ".png"),
        "c": str(bad),
        "d": str(tmp_path / "gone.jpg"),  # missing on disk
    }
    run = _run(tmp_path, images, photos)
    assert run.photo_shapes == {
        "n_photos": 4,
        "n_checked": 2,
        "n_unreadable": 2,
        "sizes_px": {"640x480": 1, "640x460": 1},
    }


def test_sizes_are_listed_most_common_first(tmp_path, monkeypatch):
    names = ["a", "b", "c"]
    images = _masks(tmp_path / "mask", names)
    photos = {n: str(tmp_path / f"{n}.jpg") for n in names}
    sizes = {photos["a"]: (640, 480), photos["b"]: (640, 460), photos["c"]: (640, 460)}
    monkeypatch.setattr(measure_run, "image_size", sizes.get)
    run = _run(tmp_path, images, photos)
    assert list(run.photo_shapes["sizes_px"].items()) == [("640x460", 2), ("640x480", 1)]


def test_a_second_compute_on_kept_rows_reads_no_header(tmp_path, monkeypatch):
    images = _masks(tmp_path / "mask", ["a", "b"])
    photos = {n: str(tmp_path / f"{n}.jpg") for n in images}
    spy = _SizeSpy()
    monkeypatch.setattr(measure_run, "image_size", spy)
    run = _run(tmp_path, images, photos)
    record = run.photo_shapes
    assert len(spy.paths) == 2
    assert run.compute() is run.rows
    assert len(spy.paths) == 2
    assert run.photo_shapes == record


def test_a_stopped_run_checks_no_photo(tmp_path, monkeypatch):
    images = _masks(tmp_path / "mask", ["a", "b"])
    photos = {n: str(tmp_path / f"{n}.jpg") for n in images}
    spy = _SizeSpy()
    monkeypatch.setattr(measure_run, "image_size", spy)
    run = MeasureRun(AREA, images, str(tmp_path / "mask"), photos=photos)
    assert run.compute(should_stop=lambda: True) is None
    assert spy.paths == [] and run.photo_shapes is None


def test_a_failed_check_is_logged_and_gives_none(tmp_path, monkeypatch, caplog):
    images = _masks(tmp_path / "mask", ["a"])
    photos = {"a": _photo(tmp_path, "a", (480, 640))}

    def broken(_images):
        raise RuntimeError("boom")

    monkeypatch.setattr(measure_sources, "_photo_finder", broken)
    with caplog.at_level("WARNING", logger=measure_run.logger.name):
        run = _run(tmp_path, images, photos)
    assert run.photo_shapes is None
    assert "shape of the photos" in caplog.text
    run.write()
    record = provenance.read_sidecar(run.csv_path)
    assert record["inputs"]["photos"] is None


def test_a_photo_header_that_raises_is_counted_unreadable(tmp_path, monkeypatch):
    images = _masks(tmp_path / "mask", ["a", "b"])
    photos = {n: str(tmp_path / f"{n}.jpg") for n in images}

    def size(path):
        if path == photos["a"]:
            raise SyntaxError("broken header")
        return (640, 480)

    monkeypatch.setattr(measure_run, "image_size", size)
    run = _run(tmp_path, images, photos)
    assert run.photo_shapes["n_unreadable"] == 1 and run.photo_shapes["n_checked"] == 1


# --------------------------------------------------------------------------- provenance


def test_the_provenance_records_the_photos(tmp_path):
    images = _masks(tmp_path / "mask", ["a", "b"])
    photos = {
        "a": _photo(tmp_path, "a", (480, 640)),
        "b": _photo(tmp_path, "b", (460, 640)),
    }
    run = _run(tmp_path, images, photos)
    run.write()
    record = provenance.read_sidecar(run.csv_path)
    assert record["inputs"]["photos"] == {
        "n_photos": 2,
        "n_checked": 2,
        "n_unreadable": 0,
        "sizes_px": {"640x480": 1, "640x460": 1},
    }


def test_a_run_without_photos_records_none(tmp_path, monkeypatch):
    images = _masks(tmp_path / "mask", ["a"])
    spy = _SizeSpy()
    monkeypatch.setattr(measure_run, "image_size", spy)
    run = MeasureRun(AREA, images, str(tmp_path / "mask"))
    run.compute()
    run.write()
    assert spy.paths == []
    assert provenance.read_sidecar(run.csv_path)["inputs"]["photos"] is None


def _measure_into(folder, *, check: bool, monkeypatch):
    """Measure the same masks and photos into ``folder``; the CSV bytes and sidecar."""
    images = _masks(folder / "mask", ["a", "b", "c"])
    photos = {
        "a": _photo(folder, "a", (480, 640)),
        "b": _photo(folder, "b", (460, 640)),
        "c": _photo(folder, "c", (460, 640), ".png"),
    }
    with monkeypatch.context() as patch:
        if not check:
            patch.setattr(MeasureRun, "_check_photo_shapes", lambda self, rows: None)
        run = MeasureRun(AREA, images, str(folder / "mask"), photos=photos, gt_files={})
        run.compute()
        rows = [dict(row) for row in run.rows]
        run.write()
        unchanged = run.unchanged()
    with open(run.csv_path, "rb") as fh:
        data = fh.read()
    return data, provenance.read_sidecar(run.csv_path), rows, unchanged


def test_the_check_changes_no_csv_byte_and_no_other_sidecar_key(tmp_path, monkeypatch):
    on = _measure_into(tmp_path / "on", check=True, monkeypatch=monkeypatch)
    off = _measure_into(tmp_path / "off", check=False, monkeypatch=monkeypatch)
    assert on[0] == off[0]  # the CSV bytes
    assert on[2] == off[2]  # every row, private keys too
    assert on[3] is off[3] is True  # unchanged()
    side_on, side_off = on[1], off[1]
    assert side_on["inputs"].pop("photos")["n_checked"] == 3
    assert side_off["inputs"].pop("photos") is None
    for side in (side_on, side_off):
        side.pop("created_utc")
    assert side_on == side_off
    # The record is JSON (a sidecar is read back as written).
    json.dumps(on[1])


# --------------------------------------------------------------------------- handler warnings


class _IdleGeneration:
    def is_busy(self) -> bool:
        return False


def _handler(tmp_path):
    mw = SimpleNamespace(
        image_manager=SimpleNamespace(original_folder=str(tmp_path), images={}),
        feedback=RecordingFeedback(),
        jobs=JobRegistry(),
    )
    return SkeletonHandler(mw, generation=_IdleGeneration()), mw


def _wait_idle(handler, qapp, timeout=20.0):
    end = time.monotonic() + timeout
    while handler.is_busy() or handler.main_window.jobs.is_busy():
        assert time.monotonic() < end, "measure did not finish"
        qapp.processEvents()
        time.sleep(0.005)
    qapp.processEvents()


def _project(tmp_path, photo_shapes, *, gt=False, predictions=None):
    """One photo per ``(rows, cols)`` of ``photo_shapes``, each with a 640 x 480 predicted
    mask (the model's size; ``predictions``: index -> another shape, a file the model did not
    write); ``gt``: also a GT mask at the photo's size (``True``: every image, else the
    indices of the images that have one)."""
    (tmp_path / "mask").mkdir(exist_ok=True)
    if gt:
        (tmp_path / "gt" / "mask").mkdir(parents=True, exist_ok=True)
    with_gt = range(len(photo_shapes)) if gt is True else (gt or ())
    for i, shape in enumerate(photo_shapes):
        name = f"s_T{i + 1}_L2_2024.01.02_01020{i}_001"
        cv2.imwrite(str(tmp_path / f"{name}.jpg"), np.zeros(shape, np.uint8))
        pred_shape = (predictions or {}).get(i, (480, 640))
        pred = np.zeros(pred_shape, np.uint8)
        cv2.line(pred, (0, 0), (pred_shape[1] - 1, 100 + i), 255, 3)
        cv2.imwrite(str(tmp_path / "mask" / f"{name}.png"), pred)
        if i in with_gt:
            mask = np.zeros(shape, np.uint8)
            cv2.line(mask, (0, 5), (shape[1] - 1, shape[0] // 2 + i), 255, 5)
            cv2.imwrite(str(tmp_path / "gt" / "mask" / f"{name}.png"), mask)
    return tmp_path


def _write_project_file(tmp_path, reference=None):
    cal = {"fov_width_mm": 18.0, "fov_height_mm": 13.5}
    if reference is not None:
        cal["reference_size_px"] = list(reference)
    (tmp_path / "sprouts_project.json").write_text(
        json.dumps({"schema": 1, "calibration": cal}), encoding="utf-8"
    )


def _measure_area(tmp_path, qapp):
    handler, mw = _handler(tmp_path)
    handler.calculate_root_area()
    _wait_idle(handler, qapp)
    return mw


def _size_warnings(mw):
    """The warnings about shapes and sizes."""
    return [
        c.args[0]
        for c in mw.feedback.calls
        if c.name == "notify"
        and c.args[1] == "warn"
        and ("another shape" in c.args[0] or "sizes (" in c.args[0])
    ]


_CROP = (460, 640)  # a cropped frame: 640 x 460
_FULL = (480, 640)


def test_default_calibration_photos_of_another_shape_warn_once(tmp_path, qapp):
    _project(tmp_path, [_CROP] * 3)
    mw = _measure_area(tmp_path, qapp)
    assert _size_warnings(mw) == [
        "3 of 3 photos checked have another shape than the 640×480 frame of the default "
        "18×13.5 mm field of view (640×460 ×3): it was applied to each whole image. State "
        "this camera's field of view in sprouts_project.json."
    ]
    record = provenance.read_sidecar(tmp_path / "mask" / "root_areas.csv")
    assert record["inputs"]["photos"] == {
        "n_photos": 3,
        "n_checked": 3,
        "n_unreadable": 0,
        "sizes_px": {"640x460": 3},
    }
    assert record["inputs"]["sizes_px"] == {"640x480": 3}  # the predictions


def test_a_mixed_project_counts_the_odd_photos(tmp_path, qapp):
    _project(tmp_path, [_FULL, _CROP, _FULL, _CROP])
    mw = _measure_area(tmp_path, qapp)
    assert _size_warnings(mw) == [
        "2 of 4 photos checked have another shape than the 640×480 frame of the default "
        "18×13.5 mm field of view (640×460 ×2): it was applied to each whole image. State "
        "this camera's field of view in sprouts_project.json."
    ]


def test_one_odd_photo_is_said_in_the_singular(tmp_path, qapp):
    _project(tmp_path, [_FULL, _CROP, _FULL])
    mw = _measure_area(tmp_path, qapp)
    assert _size_warnings(mw) == [
        "1 of 3 photos checked has another shape than the 640×480 frame of the default "
        "18×13.5 mm field of view (640×460 ×1): it was applied to each whole image. State "
        "this camera's field of view in sprouts_project.json."
    ]


def test_a_project_file_with_a_reference_names_it(tmp_path, qapp):
    _project(tmp_path, [_CROP, _CROP, _FULL])
    _write_project_file(tmp_path, reference=(640, 480))
    mw = _measure_area(tmp_path, qapp)
    assert _size_warnings(mw) == [
        "2 of 3 photos checked have another shape than the 640×480 reference in "
        "sprouts_project.json (640×460 ×2): the field of view was applied to each whole image."
    ]


def test_a_project_file_without_a_reference_is_not_checked_against(tmp_path, qapp):
    _project(tmp_path, [_CROP, _CROP])
    _write_project_file(tmp_path)
    mw = _measure_area(tmp_path, qapp)
    assert _size_warnings(mw) == []
    record = provenance.read_sidecar(tmp_path / "mask" / "root_areas.csv")
    assert record["inputs"]["photos"]["sizes_px"] == {"640x460": 2}  # still recorded


def test_photos_that_fit_do_not_warn(tmp_path, qapp):
    _project(tmp_path, [_FULL, _FULL, (240, 320)])  # 320 x 240 is 4:3 too
    mw = _measure_area(tmp_path, qapp)
    assert _size_warnings(mw) == []


def test_gt_files_of_the_photos_shape_give_one_shape_warning(tmp_path, qapp):
    """GT files are photo-sized: their shape is the photos' shape, one cause, one warning."""
    _project(tmp_path, [_CROP, _CROP], gt=True)
    mw = _measure_area(tmp_path, qapp)
    record = provenance.read_sidecar(tmp_path / "mask" / "root_areas.csv")
    assert record["outputs"]["n_gt"] == 2
    assert record["inputs"]["sizes_px"] == {"640x460": 2}
    assert _size_warnings(mw) == [
        "2 of 2 photos checked have another shape than the 640×480 frame of the default "
        "18×13.5 mm field of view (640×460 ×2): it was applied to each whole image. State "
        "this camera's field of view in sprouts_project.json."
    ]


def test_gt_files_of_another_shape_still_warn_without_a_photo_sample(tmp_path, qapp, monkeypatch):
    """No photo sample (the check gave nothing): the measured files are checked against
    the default frame, as they are against a project file's reference."""
    _project(tmp_path, [_CROP, _CROP], gt=True)
    monkeypatch.setattr(MeasureRun, "_check_photo_shapes", lambda self, rows: None)
    mw = _measure_area(tmp_path, qapp)
    assert _size_warnings(mw) == [
        "2 measured files have another shape than the 640×480 frame of the default "
        "18×13.5 mm field of view: the field of view was applied to each whole image."
    ]


def test_gt_files_of_two_sizes_with_the_photo_warning_warn_once(tmp_path, qapp):
    """GT files are photo-sized: their two sizes are the photos' two shapes, which the photo
    warning already says (one cause, one warning)."""
    _project(tmp_path, [_CROP, _CROP, (400, 640)], gt=True)
    mw = _measure_area(tmp_path, qapp)
    assert _size_warnings(mw) == [
        "3 of 3 photos checked have another shape than the 640×480 frame of the default "
        "18×13.5 mm field of view (640×460 ×2, 640×400 ×1): it was applied to each whole "
        "image. State this camera's field of view in sprouts_project.json."
    ]


def test_one_odd_photo_with_gt_masks_on_both_shapes_warns_once(tmp_path, qapp):
    """Three photos 640 x 480 and one 640 x 460, a GT mask on one of each: the GT masks
    differ in size because the photos do."""
    _project(tmp_path, [_FULL, _FULL, _FULL, _CROP], gt=(0, 3))
    mw = _measure_area(tmp_path, qapp)
    assert _size_warnings(mw) == [
        "1 of 4 photos checked has another shape than the 640×480 frame of the default "
        "18×13.5 mm field of view (640×460 ×1): it was applied to each whole image. State "
        "this camera's field of view in sprouts_project.json."
    ]


def test_predictions_of_two_sizes_still_warn_beside_the_photo_warning(tmp_path, qapp):
    """Predictions of two sizes are files the model did not write: another cause."""
    _project(tmp_path, [_FULL, _FULL, _CROP], predictions={0: (460, 640)})
    mw = _measure_area(tmp_path, qapp)
    assert _size_warnings(mw) == [
        "1 of 3 photos checked has another shape than the 640×480 frame of the default "
        "18×13.5 mm field of view (640×460 ×1): it was applied to each whole image. State "
        "this camera's field of view in sprouts_project.json.",
        "The measured files have 2 sizes (640x480 ×2, 640x460 ×1): one 18×13.5 mm field of "
        "view was applied to each.",
    ]


def test_gt_files_of_two_sizes_with_fitting_photos_still_warn(tmp_path, qapp):
    """Photos 640 x 480 and 1280 x 960 both fit the frame: no photo warning, so the GT
    files' two sizes (two cameras) are said."""
    _project(tmp_path, [_FULL, (960, 1280)], gt=True)
    mw = _measure_area(tmp_path, qapp)
    assert _size_warnings(mw) == [
        "The measured files have 2 sizes (640x480 ×1, 1280x960 ×1): one 18×13.5 mm field of "
        "view was applied to each."
    ]


# A camera of another frame, stated in the project file (754 x 510 photos).
_WIDE = (510, 754)


def test_a_reference_with_fitting_photos_and_model_size_predictions_does_not_warn(tmp_path, qapp):
    """A prediction is at the model's size whatever the photo: its shape says nothing about
    the frame, so it is never compared with the reference."""
    _project(tmp_path, [_WIDE] * 3)
    _write_project_file(tmp_path, reference=(754, 510))
    mw = _measure_area(tmp_path, qapp)
    record = provenance.read_sidecar(tmp_path / "mask" / "root_areas.csv")
    assert record["inputs"]["sizes_px"] == {"640x480": 3}
    assert _size_warnings(mw) == []


def test_a_reference_with_fitting_photos_and_one_gt_mask_does_not_warn(tmp_path, qapp):
    _project(tmp_path, [_WIDE] * 3, gt=(0,))
    _write_project_file(tmp_path, reference=(754, 510))
    mw = _measure_area(tmp_path, qapp)
    record = provenance.read_sidecar(tmp_path / "mask" / "root_areas.csv")
    assert record["inputs"]["sizes_px"] == {"640x480": 2, "754x510": 1}
    assert _size_warnings(mw) == []


def test_without_a_photo_sample_only_gt_rows_are_compared_with_the_reference(
    tmp_path, qapp, monkeypatch
):
    """No photo checked: the GT files (photo-sized) speak for the frame, the predictions
    never do. Two GT masks of another shape than the reference, one prediction-only image."""
    _project(tmp_path, [_CROP, _CROP, _WIDE], gt=(0, 1))
    _write_project_file(tmp_path, reference=(754, 510))
    monkeypatch.setattr(MeasureRun, "_check_photo_shapes", lambda self, rows: None)
    mw = _measure_area(tmp_path, qapp)
    assert _size_warnings(mw) == [
        "2 measured files have another shape than the 754×510 reference in "
        "sprouts_project.json: the field of view was applied to each whole image."
    ]


def test_without_a_photo_sample_predictions_only_give_no_shape_warning(tmp_path, qapp, monkeypatch):
    _project(tmp_path, [_WIDE] * 2)
    _write_project_file(tmp_path, reference=(754, 510))
    monkeypatch.setattr(MeasureRun, "_check_photo_shapes", lambda self, rows: None)
    mw = _measure_area(tmp_path, qapp)
    assert _size_warnings(mw) == []


def test_a_sample_with_no_photo_read_leaves_the_check_to_the_gt_rows(tmp_path, qapp, monkeypatch):
    """Photos sampled but none read (``n_checked`` 0): the GT rows are checked."""
    _project(tmp_path, [_CROP, _CROP], gt=(0,))
    none_read = {"n_photos": 2, "n_checked": 0, "n_unreadable": 2, "sizes_px": {}}
    monkeypatch.setattr(MeasureRun, "_check_photo_shapes", lambda self, rows: none_read)
    mw = _measure_area(tmp_path, qapp)
    assert _size_warnings(mw) == [
        "1 measured file has another shape than the 640×480 frame of the default "
        "18×13.5 mm field of view: the field of view was applied to each whole image."
    ]


def test_gt_files_of_two_sizes_and_another_shape_give_one_warning(tmp_path, qapp, monkeypatch):
    """No photo sample: the GT files' two sizes and their shape are said in one warning."""
    _project(tmp_path, [_CROP, _CROP, _FULL], gt=True)
    monkeypatch.setattr(MeasureRun, "_check_photo_shapes", lambda self, rows: None)
    mw = _measure_area(tmp_path, qapp)
    assert _size_warnings(mw) == [
        "The measured files have 2 sizes (640x460 ×2, 640x480 ×1), and 2 have another shape "
        "than the 640×480 frame of the default 18×13.5 mm field of view: one 18×13.5 mm field "
        "of view was applied to each whole image."
    ]


@pytest.mark.parametrize("n_sizes", [3, 4])
def test_at_most_three_odd_photo_sizes_are_listed(tmp_path, qapp, n_sizes):
    shapes = [(400 + 10 * i, 640) for i in range(n_sizes)]
    _project(tmp_path, [shapes[0], *shapes])  # the first size twice: most common first
    mw = _measure_area(tmp_path, qapp)
    (photo, *_rest) = _size_warnings(mw)
    assert photo.startswith(f"{n_sizes + 1} of {n_sizes + 1} photos checked have ")
    listed = photo[photo.index("(") + 1 : photo.index(")")]
    assert listed.startswith("640×400 ×2, ")
    assert listed.count(" ×") == 3
    assert listed.endswith(", …") is (n_sizes > 3)
