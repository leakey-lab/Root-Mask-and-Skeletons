"""Helpers of the Measure tests: the legacy CSV reference and its dataset, the fake-window
handler, a GT-first run through it, and the Measure-CSV writers.

Moved out of ``test_measure_core.py`` (the legacy reference and the dataset),
``test_measure_handler.py`` (the fake main window), ``test_measure_gt_first.py``
(``run_length`` / ``run_area``) and ``test_shell_info_measurement_index.py`` (the CSV
writers), so that no test module imports another.
"""

from __future__ import annotations

import csv
import os
import re
import time
from types import SimpleNamespace

import cv2
import numpy as np

from app.config import AREA_CSV_HEADERS, CALIBRATION, LENGTH_CSV_HEADERS
from app.feedback import RecordingFeedback
from app.handlers.job_registry import JobRegistry
from app.handlers.skeleton_handler import SkeletonHandler
from app.inference.metrics import write_metric_csv
from app.project_layout import ProjectLayout
from tests import length_reference

# --------------------------------------------------------------------------- legacy reference
# Verbatim logic of the pre-refactor pipeline (metrics.py @ a1599e6): cv2.imread,
# first-match T/L regexes, open(..., "w") + csv.DictWriter.


def _legacy_parse(name):
    info = {"original_name": name, "tube_number": None, "length_position": None}
    info.update(date=None, time=None)
    if m := re.search(r"T(\d+)", name):
        info["tube_number"] = int(m.group(1))
    if m := re.search(r"L(\d+)", name):
        info["length_position"] = int(m.group(1))
    if m := re.search(r"(\d{4})\.(\d{2})\.(\d{2})", name):
        info["date"] = ".".join(m.groups())
    if m := re.search(r"_(\d{6})(?:_|$)", name):
        t = m.group(1)
        info["time"] = f"{t[:2]}:{t[2:4]}:{t[4:]}"
    return info


def _legacy_row(name, path, kind):
    from skimage.morphology import skeletonize

    value_col = LENGTH_CSV_HEADERS[5] if kind == "length" else AREA_CSV_HEADERS[5]
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            what = "image" if kind == "length" else "mask"
            # O38 (W3-T7): the file name only
            raise FileNotFoundError(f"Could not read {what}: {os.path.basename(path)}")
        h, w = img.shape[:2]
        if kind == "length":
            _, b = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if np.mean(b) > 127:
                b = 255 - b
            s = skeletonize(b / 255) > 0
            mx, my = CALIBRATION.mm_per_px(w, h)
            # O12 (W3-T8): the corner-count estimator, from the independent reference
            value = length_reference.vs_length_mm(s, mx, my)
        else:
            _, b = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            value = int(np.count_nonzero(b == 255)) * CALIBRATION.area_per_px_mm2(w, h)
        info = _legacy_parse(name)
        return {
            "Image": info["original_name"],
            "Tube": info["tube_number"],
            "Position": info["length_position"],
            "Date": info["date"],
            "Time": info["time"],
            value_col: round(value, 2),
        }
    except Exception as e:  # noqa: BLE001 - mirrors the legacy worker
        row = {"Image": name, "Tube": None, "Position": None, "Date": None, "Time": None}
        return row | {value_col: 0, "Error": str(e)}


def _legacy_csv(images, kind, out_path, *, source=True):
    """The legacy CSV of ``images``. Every input is a prediction: ``source`` adds the last
    column ``Source`` = ``prediction`` (WG-T8, O1); without it, the seven columns written
    before."""
    headers = LENGTH_CSV_HEADERS if kind == "length" else AREA_CSV_HEADERS
    if not source:
        headers = headers[:7]
    rows = sorted(
        ({**_legacy_row(n, p, kind), "Source": "prediction"} for n, p in images.items()),
        key=lambda r: (
            r.get("Tube") or float("inf"),
            r.get("Date") or "",
            r.get("Time") or "",
            r.get("Position") or float("inf"),
            r.get("Image") or "",  # tie order by image name (WX-T9)
        ),
    )
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow({h: r.get(h, "") for h in headers})
    return out_path


# --------------------------------------------------------------------------- fixtures

_NAMES = [
    "24EFCAM1_T001_L044_2024.07.18_111530_001_SMM",
    "24EFCAM1_T001_L045_2024.07.18_111529_001_SMM",
    "24EFCAM1_T012_L043_2024.07.18_101914_001_SMM",
    "24EFCAM2_T005_L056_2024.08.26_085019_003_SMM",
    "24EFCAM3_T100_L049_2024.08.07_103103_002_SMM",
    "24EFCAM1_T150_L038_2024.08.27_084641_003_SMM",
]


def _skeleton_image(seed: int, shape=(256, 341)) -> np.ndarray:
    """A deterministic grey image with diagonal + orthogonal strokes."""
    rng = np.random.default_rng(seed)
    img = np.zeros(shape, np.uint8)
    for _ in range(6):
        p1 = tuple(int(v) for v in rng.integers(0, shape[1], 2) % (shape[1], shape[0]))
        p2 = tuple(int(v) for v in rng.integers(0, shape[1], 2) % (shape[1], shape[0]))
        cv2.line(img, p1, p2, int(rng.integers(160, 256)), int(rng.integers(1, 4)))
    return img


def _write_png(path, img):
    ok, buf = cv2.imencode(".png", img)
    assert ok
    buf.tofile(str(path))  # works for non-ASCII paths


def _make_dataset(folder, kind, names=_NAMES):
    folder.mkdir(parents=True, exist_ok=True)
    images = {}
    for i, name in enumerate(names):
        img = _skeleton_image(i)
        if kind == "area":
            img = cv2.dilate(img, np.ones((5, 5), np.uint8))
            p = folder / f"{name}.png"
        else:
            p = folder / f"{name}_fake.png"
        _write_png(p, img)
        images[name] = str(p)
    return images


# --------------------------------------------------------------------------- fake main window


class _IdleGeneration:
    def is_busy(self) -> bool:
        return False


def _project(tmp_path, n=4, skeletons=True, masks=True):
    """``n`` photos of the default field of view's shape (80 x 60, 4:3, so Measure has no
    photo shape to warn about), each with an 80 x 64 skeleton and mask."""
    for i in range(n):
        img = np.zeros((64, 80), np.uint8)
        cv2.line(img, (0, i), (79, 40 + i), 255, 2)
        name = f"x_T{i + 1}_L2_2024.01.02_01020{i}_001"
        cv2.imwrite(str(tmp_path / f"{name}.jpg"), np.zeros((60, 80), np.uint8))
        if skeletons:
            (tmp_path / "skeletons").mkdir(exist_ok=True)
            cv2.imwrite(str(tmp_path / "skeletons" / f"{name}_fake.png"), img)
        if masks:
            (tmp_path / "mask").mkdir(exist_ok=True)
            cv2.imwrite(str(tmp_path / "mask" / f"{name}.png"), img)
    return tmp_path


def _handler(tmp_path, qapp, jobs=None):
    mw = SimpleNamespace(
        image_manager=SimpleNamespace(original_folder=str(tmp_path), images={}),
        feedback=RecordingFeedback(),
        jobs=jobs if jobs is not None else JobRegistry(),
    )
    return SkeletonHandler(mw, generation=_IdleGeneration()), mw


def _wait_idle(handler, qapp, timeout=20.0):
    end = time.monotonic() + timeout
    while handler.is_busy() or handler.main_window.jobs.is_busy():
        assert time.monotonic() < end, "measure did not finish"
        qapp.processEvents()
        time.sleep(0.005)
    qapp.processEvents()


# --------------------------------------------------------------------------- GT-first runs


def _csv_path(layout: ProjectLayout, kind: str) -> str:
    if kind == "length":
        return os.path.join(layout.skeletons_dir, "root_lengths.csv")
    return os.path.join(layout.mask_dir, "root_areas.csv")


def _rows(path: str) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return {row["Image"]: row for row in csv.DictReader(fh)}


def _run(layout: ProjectLayout, qapp, kind: str):
    """Measure ``kind`` through the handler, as the Measure button does; the fake window."""
    handler, mw = _handler(layout.root, qapp)
    (handler.calculate_root_length if kind == "length" else handler.calculate_root_area)()
    _wait_idle(handler, qapp)
    return mw


def run_length(layout: ProjectLayout, qapp) -> dict[str, dict]:
    _run(layout, qapp, "length")
    return _rows(_csv_path(layout, "length"))


def run_area(layout: ProjectLayout, qapp) -> dict[str, dict]:
    _run(layout, qapp, "area")
    return _rows(_csv_path(layout, "area"))


# --------------------------------------------------------------------------- Measure CSVs


def _row(stem: str, column: str, value, error: str = "") -> dict:
    row = {"Image": stem, "Tube": 1, "Position": 44, "Date": "2024.07.18", "Time": "11:15:30"}
    row[column] = value
    if error:
        row["Error"] = error
    return row


def _write_lengths(root, rows) -> str:
    folder = root / "skeletons"
    folder.mkdir(exist_ok=True)
    path = str(folder / "root_lengths.csv")
    write_metric_csv(rows, path, LENGTH_CSV_HEADERS)
    return path


def _write_areas(root, rows) -> str:
    folder = root / "mask"
    folder.mkdir(exist_ok=True)
    path = str(folder / "root_areas.csv")
    write_metric_csv(rows, path, AREA_CSV_HEADERS)
    return path


def _bump_mtime(path: str, seconds: float = 5.0) -> None:
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + int(seconds * 1e9)))


# --------------------------------------------------------------------------- handler notices


def _write_project_file(tmp_path, text):
    (tmp_path / "sprouts_project.json").write_text(text, encoding="utf-8")


def _warnings(mw):
    return [c.args[0] for c in mw.feedback.calls if c.name == "notify" and c.args[1] == "warn"]
