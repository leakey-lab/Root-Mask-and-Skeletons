"""The warnings at the end of a Measure run (M2, M5, O3): measured files of several sizes, masks
with grey levels, files of another shape than the project file's reference.

Each cause is said once, in one warning; the provenance holds the details.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from app import provenance
from tests.measure_support import _handler, _project, _wait_idle, _warnings, _write_project_file

_MIXED = "The measured files have"


def _mixed_warnings(mw):
    return [
        call.args[0]
        for call in mw.feedback.calls
        if call.name == "notify" and call.args[1] == "warn" and call.args[0].startswith(_MIXED)
    ]


def _extra_image(tmp_path, name, shape, folder, *, suffix=".png", photo_shape=None):
    """One more image of the project (``photo_shape``, default ``shape``) with its file in
    ``folder`` at ``shape`` (rows, cols)."""
    cv2.imwrite(str(tmp_path / f"{name}.jpg"), np.zeros(photo_shape or shape, np.uint8))
    img = np.zeros(shape, np.uint8)
    cv2.line(img, (0, 0), (shape[1] - 1, shape[0] - 1), 255, 2)
    (tmp_path / folder).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(tmp_path / folder / f"{name}{suffix}"), img)


def test_mixed_sizes_warn_once(tmp_path, qapp):
    _project(tmp_path, n=2, skeletons=False)
    _extra_image(tmp_path, "y_T9_L2_2024.01.02_010209_001", (48, 96), "mask")
    handler, mw = _handler(tmp_path, qapp)
    handler.calculate_root_area()
    _wait_idle(handler, qapp)

    warnings = _mixed_warnings(mw)
    assert len(warnings) == 1
    assert warnings[0].startswith("The measured files have 2 sizes (80x64 ×2, 96x48 ×1): ")
    assert warnings[0].endswith("one 18×13.5 mm field of view was applied to each.")


def test_same_size_files_do_not_warn(tmp_path, qapp):
    _project(tmp_path, n=3)
    handler, mw = _handler(tmp_path, qapp)
    handler.calculate_root_length()
    _wait_idle(handler, qapp)
    assert os.path.isfile(tmp_path / "skeletons" / "root_lengths.csv")
    assert _mixed_warnings(mw) == []


def test_at_most_three_sizes_are_listed(tmp_path, qapp):
    _project(tmp_path, n=2, skeletons=False)
    for i, shape in enumerate([(48, 96), (50, 96), (52, 96)]):
        _extra_image(tmp_path, f"z_T{i + 5}_L2_2024.01.02_01020{i}_001", shape, "mask")
    handler, mw = _handler(tmp_path, qapp)
    handler.calculate_root_area()
    _wait_idle(handler, qapp)
    (message,) = _mixed_warnings(mw)
    assert message.startswith("The measured files have 4 sizes (80x64 ×2, ")
    assert message.count(" ×") == 3 and "…" in message


def _photo_size_project(tmp_path, photo_shapes):
    """Photos of ``photo_shapes`` (rows, cols), each with a model-size prediction (256 x 341)
    and a GT skeleton at the photo's size (``None``: no GT file for that image)."""
    (tmp_path / "skeletons").mkdir()
    (tmp_path / "gt" / "skeletons").mkdir(parents=True)
    for i, (photo, has_gt) in enumerate(photo_shapes):
        name = f"w_T{i + 1}_L2_2024.01.02_01020{i}_001"
        cv2.imwrite(str(tmp_path / f"{name}.jpg"), np.zeros(photo, np.uint8))
        pred = np.zeros((256, 341), np.uint8)
        cv2.line(pred, (0, 0), (340, 100 + i), 255, 2)
        cv2.imwrite(str(tmp_path / "skeletons" / f"{name}_fake.png"), pred)
        if has_gt:
            gt = np.zeros(photo, np.uint8)
            cv2.line(gt, (0, 5), (photo[1] - 1, photo[0] // 2), 255, 3)
            cv2.imwrite(str(tmp_path / "gt" / "skeletons" / f"{name}.png"), gt)
    return tmp_path


def test_photo_size_gt_beside_model_size_predictions_is_not_mixed(tmp_path, qapp):
    """Every measured file spans its photo's frame, so one field of view is right for each:
    GT files (photo size) and predictions (model size) are not compared with each other."""
    _photo_size_project(
        tmp_path, [((480, 640), True), ((480, 640), True), ((480, 640), False), ((480, 640), False)]
    )
    handler, mw = _handler(tmp_path, qapp)
    handler.calculate_root_length()
    _wait_idle(handler, qapp)

    rows = open(tmp_path / "skeletons" / "root_lengths.csv", encoding="utf-8").read()
    assert rows.count(",gt") == 2 and rows.count(",prediction") == 2
    record = provenance.read_sidecar(tmp_path / "skeletons" / "root_lengths.csv")
    assert record["inputs"]["sizes_px"] == {"640x480": 2, "341x256": 2}
    assert _mixed_warnings(mw) == []


def test_gt_files_of_two_sizes_warn_once(tmp_path, qapp):
    """GT files of two sizes mean photos of two sizes (two cameras): that is the signal."""
    _photo_size_project(
        tmp_path, [((480, 640), True), ((240, 320), True), ((480, 640), False), ((480, 640), False)]
    )
    handler, mw = _handler(tmp_path, qapp)
    handler.calculate_root_length()
    _wait_idle(handler, qapp)

    (message,) = _mixed_warnings(mw)
    assert message.startswith("The measured files have 2 sizes (640x480 ×1, 320x240 ×1): ")


_GREY = "grey levels: pixels above 127 were counted as root."


def _grey_warnings(mw):
    return [
        call.args[0]
        for call in mw.feedback.calls
        if call.name == "notify" and call.args[1] == "warn" and _GREY in call.args[0]
    ]


def test_grey_masks_warn_once(tmp_path, qapp):
    _project(tmp_path, n=3, skeletons=False)
    mask = next((tmp_path / "mask").iterdir())
    grey = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
    grey[0, 0] = 100
    cv2.imwrite(str(mask), grey)
    handler, mw = _handler(tmp_path, qapp)
    handler.calculate_root_area()
    _wait_idle(handler, qapp)
    assert _grey_warnings(mw) == ["1 mask has grey levels: pixels above 127 were counted as root."]


def test_binary_masks_and_length_runs_do_not_warn_about_grey(tmp_path, qapp):
    _project(tmp_path, n=2)
    handler, mw = _handler(tmp_path, qapp)
    handler.calculate_root_area()
    _wait_idle(handler, qapp)
    handler.calculate_root_length()
    _wait_idle(handler, qapp)
    assert _grey_warnings(mw) == []


def test_files_of_another_shape_warn_once(tmp_path, qapp):
    _write_project_file(
        tmp_path,
        '{"schema": 1, "calibration": {"fov_width_mm": 18.0, "fov_height_mm": 13.5, '
        '"reference_size_px": [640, 480]}}',
    )
    # Model-size predictions (80 x 64 here) and a GT mask at each photo's size.
    for i in range(2):
        name = f"v_T{i + 1}_L2_2024.01.02_01020{i}_001"
        _extra_image(tmp_path, name, (64, 80), "mask", photo_shape=(60, 80))
        _extra_image(tmp_path, name, (60, 80), "gt/mask")
    name = "v_T9_L2_2024.01.02_010209_001"
    _extra_image(tmp_path, name, (64, 80), "mask", photo_shape=(48, 96))
    _extra_image(tmp_path, name, (48, 96), "gt/mask")
    handler, mw = _handler(tmp_path, qapp)
    handler.calculate_root_area()
    _wait_idle(handler, qapp)

    # One odd photo is one cause: the photos say it, and the GT files' two sizes (they are
    # photo-sized) are not said again (review M2, FR-5).
    record = provenance.read_sidecar(tmp_path / "mask" / "root_areas.csv")
    assert record["inputs"]["sizes_px"] == {"80x60": 2, "96x48": 1}
    sizes = [w for w in _warnings(mw) if "another shape" in w or w.startswith(_MIXED)]
    assert sizes == [
        "1 of 3 photos checked has another shape than the 640×480 reference in "
        "sprouts_project.json (96×48 ×1): the field of view was applied to each whole image."
    ]


def test_files_all_of_another_shape_warn_once(tmp_path, qapp):
    _write_project_file(
        tmp_path,
        '{"schema": 1, "calibration": {"fov_width_mm": 18.0, "fov_height_mm": 13.5, '
        '"reference_size_px": [640, 480]}}',
    )
    for i in range(2):
        _extra_image(tmp_path, f"v_T{i + 1}_L2_2024.01.02_01020{i}_001", (48, 96), "mask")
    handler, mw = _handler(tmp_path, qapp)
    handler.calculate_root_area()
    _wait_idle(handler, qapp)

    sizes = [w for w in _warnings(mw) if "another shape" in w or w.startswith(_MIXED)]
    assert sizes == [  # the photos say it (T3): the measured files' clause is left out
        "2 of 2 photos checked have another shape than the 640×480 reference in "
        "sprouts_project.json (96×48 ×2): the field of view was applied to each whole image."
    ]
