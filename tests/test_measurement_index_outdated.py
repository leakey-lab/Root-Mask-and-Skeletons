"""``MeasurementIndex.outdated``: each measured value against its own input (W4, W3).

LENGTH was measured from the generated skeleton (``skeletons/<stem>_fake.png``), AREA from
the mask (``mask/<stem>.png``): a newer mask outdates AREA only, a newer skeleton LENGTH
only, and an image without a row is never outdated.

(Kept apart from A10's metrics-bar files so the measurement-index home can take it whole.)
"""

from __future__ import annotations

import os

from app.config import AREA_CSV_HEADERS, LENGTH_CSV_HEADERS
from app.data_processing.measurement_index import MeasurementIndex
from app.inference.metrics import write_metric_csv

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


def test_a_newer_mask_outdates_the_area_only(tmp_path):
    mask, _skeleton, _lengths, _areas = _measured_project(tmp_path)
    index = MeasurementIndex(str(tmp_path))
    index.prewarm()
    assert index.outdated(STEM) == {"length": False, "area": False}

    _shift(mask, 120)  # saved after the Measure run

    assert index.outdated(STEM) == {"length": False, "area": True}
    assert index.is_outdated(STEM) is True  # the old any-metric answer stays


def test_a_newer_generated_skeleton_outdates_the_length_only(tmp_path):
    _mask, skeleton, _lengths, _areas = _measured_project(tmp_path)
    index = MeasurementIndex(str(tmp_path))
    index.prewarm()

    _shift(skeleton, 120)

    assert index.outdated(STEM) == {"length": True, "area": False}


def test_a_metric_without_a_row_is_never_outdated(tmp_path):
    mask, skeleton, _lengths, _areas = _measured_project(tmp_path)
    index = MeasurementIndex(str(tmp_path))
    index.prewarm()
    for path in (mask, skeleton):
        _shift(path, 120)

    assert index.outdated("24EFCAM1_T001_L045_2024.07.18_111612_001_SMM") == {
        "length": False,
        "area": False,
    }
