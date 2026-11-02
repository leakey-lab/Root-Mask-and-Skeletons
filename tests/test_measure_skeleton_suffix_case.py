"""Measure Length reads a skeleton whose ``_fake.png`` suffix is in another case (#3, C1-04).

Correct and the Library display showed ``skeletons/B_fake.PNG`` on Windows (an exact-name
lookup there ignores case), but Measure Length matched the suffix with its exact case and
gave that image no row. Measure, its status count and the Library badge now fold the
suffix's case (ASCII letters only); where a case-sensitive disk holds several skeletons of
one image, a file with the exact suffix is measured before any case variant of the suffix
(``A_fake.png`` before ``a_FAKE.png`` for image ``a``), the others are duplicates, and
Correct's lookup finds the measured file.

The owner's answer: #3 adds rows only, and every other row stays identical. So a suffix
variant is read only for an image without a GT skeleton file (``gt/skeletons/<stem>.png``,
valid or not): such an image gets a new ``prediction`` row. An image with a GT skeleton file
is measured as before #3, which did not see the variant: the same ``Image`` name,
``Source`` and value, or no row; the variant is not even a duplicate there.

The metrics bar names the file a row was measured from as it is on disk
(``skeletons/B_fake.PNG``, ``mask/A.PNG``) and calls it current right after Measure.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from app import provenance
from app.calibration import load_project_calibration
from app.data_processing.measurement_index import MeasurementIndex
from app.editing.thinning import thin_raster
from app.feedback import RecordingFeedback
from app.gui.image_manager import ImageManager
from app.gui.stage_summary import Freshness, input_freshness
from app.handlers.job_registry import JobRegistry
from app.handlers.skeleton_handler import SkeletonHandler
from app.inference.input_split import split_inputs
from app.inference.measure_inputs import load_inputs_record
from app.inference.measure_run import AREA, LENGTH, MeasureRun, scan_png_files
from app.project_layout import (
    SKELETON_SUFFIX,
    ProjectLayout,
    group_skeletons,
    skeleton_stem,
    skip_gt_variants,
)


def _skeleton() -> np.ndarray:
    plane = np.zeros((48, 64), np.uint8)
    plane[20, 5:60] = 255
    return plane


def _write(path: str, plane: np.ndarray) -> None:
    """``cv2.imwrite`` under any name (it picks the codec from a lower-case extension)."""
    temp = os.path.join(os.path.dirname(path), "_write.png")
    cv2.imwrite(temp, plane)
    os.replace(temp, path)


def _project(root: str, skeletons: dict[str, str], masks: dict[str, str] | None = None):
    """Photos ``A.jpg`` / ``B.jpg``; ``skeletons`` and ``masks`` map the file names to write
    to the photo each belongs to (for the reader)."""
    layout = ProjectLayout(root)
    os.makedirs(layout.skeletons_dir)
    os.makedirs(layout.mask_dir)
    for stem in ("A", "B"):
        cv2.imwrite(os.path.join(root, stem + ".jpg"), np.full((48, 64, 3), 128, np.uint8))
    for name in skeletons:
        _write(os.path.join(layout.skeletons_dir, name), _skeleton())
    mask = np.zeros((48, 64), np.uint8)
    mask[10:30, 10:40] = 255
    for name in masks or {}:
        _write(os.path.join(layout.mask_dir, name), mask)
    old = time.time_ns() - 300 * 10**9  # well before the CSVs
    for folder in (layout.skeletons_dir, layout.mask_dir):
        for entry in os.scandir(folder):
            os.utime(entry.path, ns=(old, old))
    return layout


def _measure(layout: ProjectLayout, spec, folder: str, scan, gt_folder: str) -> MeasureRun:
    """One metric measured as the Measure handler builds its run."""
    listed, shadowed = scan(folder)
    gt_files, gt_shadowed = scan_png_files(gt_folder)
    photos = layout.scan_images()
    gt_kept = split_inputs(gt_files, photos, gt_shadowed).kept
    if spec is LENGTH:
        listed = dict(skip_gt_variants(listed.items(), gt_kept))
        shadowed = list(skip_gt_variants(shadowed, gt_kept))
    split = split_inputs(listed, photos, shadowed)
    run = MeasureRun(
        spec,
        split.kept,
        folder,
        project_root=layout.root,
        duplicates=split.duplicates,
        gt_files=gt_kept,
        photos=photos,
        calibration=load_project_calibration(layout.root),
    )
    run.compute()
    run.write()
    return run


def _measure_length(layout: ProjectLayout) -> MeasureRun:
    return _measure(
        layout, LENGTH, layout.skeletons_dir, layout.scan_skeleton_files, layout.gt_skeletons_dir
    )


# --------------------------------------------------------------------------- the name rule
def test_the_suffix_matches_in_any_case_of_its_ascii_letters():
    assert skeleton_stem("A_fake.png") == "A"
    assert skeleton_stem("B_fake.PNG") == "B"
    assert skeleton_stem("C_FAKE.Png") == "C"
    assert skeleton_stem("_FAKE.PNG") == ""  # as "_fake.png": a stem of its own
    assert skeleton_stem("._D_Fake.png") == "._D"  # hidden names count, as before
    for name in ("a.png", "a_fake.png.tmp", "a_fake.jpg", "a_faKe.png", "fake.png"):
        assert skeleton_stem(name) is None, name  # U+212A KELVIN SIGN lowers to "k"


@pytest.mark.parametrize("order", [1, -1], ids=["exact first", "exact last"])
def test_the_exact_suffix_wins_over_its_case_variants(order):
    files = [
        ("a", os.path.join("s", "a_fake.PNG")),
        ("a", os.path.join("s", "a_fake.png")),
        ("a", os.path.join("s", "a_FAKE.png")),
        ("b", os.path.join("s", "b_fake.Png")),
        ("b", os.path.join("s", "b_FAKE.PNG")),
    ][::order]

    listed, shadowed = group_skeletons(files)

    # the exact name, else the name that sorts first
    assert listed == {"a": os.path.join("s", "a_fake.png"), "b": os.path.join("s", "b_FAKE.PNG")}
    assert sorted(shadowed) == [
        ("a", os.path.join("s", "a_FAKE.png")),
        ("a", os.path.join("s", "a_fake.PNG")),
        ("b", os.path.join("s", "b_fake.Png")),
    ]


# --------------------------------------------------------------------------- Measure
def test_measure_length_gives_a_case_variant_suffix_its_row(tmp_path):
    layout = _project(str(tmp_path / "p"), {"A_fake.png": "A", "B_fake.PNG": "B"})
    before = _project(str(tmp_path / "q"), {"A_fake.png": "A"})

    rows = {row["Image"]: row for row in _measure_length(layout).rows}
    alone = {row["Image"]: row for row in _measure_length(before).rows}

    assert sorted(rows) == ["A", "B"]
    assert not rows["B"].get("Error") and rows["B"]["Source"] == "prediction"
    assert rows["B"][LENGTH.value_column] == rows["A"][LENGTH.value_column] > 0
    assert rows["A"] == alone["A"]  # the other rows stay as they were
    assert layout.scan_skeleton_stems() == {"A", "B"}  # the Library badge counts it too
    assert layout.find_skeleton("B") is not None


def test_the_status_count_and_the_metrics_bar_follow_the_measured_file(tmp_path):
    layout = _project(
        str(tmp_path / "p"),
        {"A_fake.png": "A", "B_fake.PNG": "B"},
        masks={"A.PNG": "A", "B.png": "B"},
    )
    _measure_length(layout)
    _measure(layout, AREA, layout.mask_dir, scan_png_files, layout.gt_mask_dir)
    index = MeasurementIndex(layout.root)
    index.prewarm()
    images = layout.scan_images()

    for folder, suffix, sources, gt_folder in (
        (layout.skeletons_dir, SKELETON_SUFFIX, "length", layout.gt_skeletons_dir),
        (layout.mask_dir, ".png", "area", layout.gt_mask_dir),
    ):
        csv_path = index.sources()[sources]
        fresh = input_freshness(
            folder,
            suffix,
            csv_path,
            record=load_inputs_record(csv_path),
            project_root=layout.root,
            images=images,
            gt_folder=gt_folder,
        )
        assert fresh == Freshness(2, True, by_content=True), sources
    for stem in ("A", "B"):
        assert index.outdated(stem) == {"length": False, "area": False}, stem
    assert index.measured_from("B")["length"] == ("skeletons/B_fake.PNG", "prediction")
    assert index.measured_from("A")["area"] == ("mask/A.PNG", "prediction")
    assert index.measured_from("A")["length"] == ("skeletons/A_fake.png", "prediction")

    _write(os.path.join(layout.skeletons_dir, "B_fake.PNG"), np.zeros((48, 64), np.uint8))
    assert index.outdated("B") == {"length": True, "area": False}  # its own file changed


def _press_measure_length(layout: ProjectLayout, qapp) -> list[dict[str, str]]:
    """Measure Length as the button runs it (the handler's listing and its job); the CSV's
    rows."""
    window = SimpleNamespace(
        image_manager=SimpleNamespace(original_folder=layout.root, images={}),
        feedback=RecordingFeedback(),
        jobs=JobRegistry(),
        _loading_project=False,
    )
    handler = SkeletonHandler(window, generation=SimpleNamespace(is_busy=lambda: False))

    handler.calculate_root_length()
    end = time.monotonic() + 20
    while handler.is_busy() or window.jobs.is_busy():
        assert time.monotonic() < end, "measure did not finish"
        qapp.processEvents()
        time.sleep(0.005)
    qapp.processEvents()

    with open(os.path.join(layout.skeletons_dir, "root_lengths.csv"), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_the_measure_button_measures_it(tmp_path, qapp):
    layout = _project(str(tmp_path), {"A_fake.png": "A", "B_fake.PNG": "B"})

    rows = _press_measure_length(layout, qapp)

    assert sorted(row["Image"] for row in rows) == ["A", "B"]


# --------------------------------------------------------------------------- beside GT
def _curve() -> np.ndarray:
    """A prediction whose thinned, upscaled copy measures another length than it does."""
    plane = np.zeros((48, 64), np.uint8)
    cv2.line(plane, (2, 3), (60, 44), 255, 1)
    cv2.circle(plane, (32, 24), 15, 255, 1)
    return plane


_PHOTO = (128, 96)  # (w, h): twice the prediction's size, as a generated skeleton is smaller


def _gt_project(root: str, prediction: str | None, gt: str | None) -> ProjectLayout:
    """Photos ``A.jpg`` / ``B.jpg``; predictions ``A_fake.png`` and ``prediction`` (image
    ``B``'s, None: none); ``gt/skeletons/B.png`` (``gt``): ``"unedited"`` (the prediction as
    Correct opens it), ``"edited"``, ``"size"`` (another size than the photo) or None."""
    layout = ProjectLayout(root)
    os.makedirs(layout.skeletons_dir)
    for stem in ("A", "B"):
        photo = np.full((_PHOTO[1], _PHOTO[0], 3), 128, np.uint8)
        cv2.imwrite(os.path.join(root, stem + ".jpg"), photo)
    _write(os.path.join(layout.skeletons_dir, "A_fake.png"), _curve())
    if prediction is not None:
        _write(os.path.join(layout.skeletons_dir, prediction), _curve())
    opened = (thin_raster(_curve(), _PHOTO) > 0).astype(np.uint8) * 255
    planes = {"unedited": opened, "size": np.zeros((10, 10), np.uint8)}
    planes["edited"] = opened.copy()
    planes["edited"][40, 10:40] = 0
    if gt is not None:
        os.makedirs(layout.gt_skeletons_dir)
        _write(os.path.join(layout.gt_skeletons_dir, "B.png"), planes[gt])
    return layout


def _what_measure_wrote(layout: ProjectLayout) -> tuple[dict, list[str]]:
    """The ``inputs`` of the length CSV's provenance and the files its record names."""
    csv_path = os.path.join(layout.skeletons_dir, "root_lengths.csv")
    inputs = provenance.read_sidecar(csv_path)["inputs"]
    return inputs, sorted(load_inputs_record(csv_path).inputs)


def _status_count(layout: ProjectLayout) -> Freshness:
    csv_path = os.path.join(layout.skeletons_dir, "root_lengths.csv")
    return input_freshness(
        layout.skeletons_dir,
        SKELETON_SUFFIX,
        csv_path,
        record=load_inputs_record(csv_path),
        project_root=layout.root,
        images=layout.scan_images(),
        gt_folder=layout.gt_skeletons_dir,
    )


def test_without_a_gt_skeleton_a_case_variant_gets_a_prediction_row(tmp_path, qapp):
    layout = _gt_project(str(tmp_path / "p"), "B_fake.PNG", None)
    before = _gt_project(str(tmp_path / "q"), None, None)

    rows = {row["Image"]: row for row in _press_measure_length(layout, qapp)}
    alone = {row["Image"]: row for row in _press_measure_length(before, qapp)}

    assert sorted(rows) == ["A", "B"] and sorted(alone) == ["A"]
    assert rows["B"]["Source"] == "prediction" and not rows["B"]["Error"]
    assert rows["A"] == alone["A"]
    assert _status_count(layout) == Freshness(2, True, by_content=True)


@pytest.mark.parametrize(
    ("prediction", "gt"),
    [("B_fake.PNG", "unedited"), ("b_fake.PNG", "edited"), ("B_fake.PNG", "size")],
    ids=["unedited GT copy", "edited GT", "GT of another size"],
)
def test_beside_a_gt_skeleton_a_case_variant_changes_nothing(tmp_path, qapp, prediction, gt):
    """An image with a GT skeleton file (valid or not) is measured as before #3, which did
    not see a prediction whose suffix is in another case: the same rows (``Image`` name,
    ``Source``, value, or no row), provenance and record as a project without that file."""
    layout = _gt_project(str(tmp_path / "p"), prediction, gt)
    before = _gt_project(str(tmp_path / "q"), None, gt)

    rows = _press_measure_length(layout, qapp)
    alone = _press_measure_length(before, qapp)

    assert rows == alone
    assert [row["Image"] for row in rows] == (["A"] if gt == "size" else ["A", "B"])
    if gt != "size":
        assert rows[1]["Source"] == "gt"
    assert _what_measure_wrote(layout) == _what_measure_wrote(before)
    assert _status_count(layout) == _status_count(before)
    assert _status_count(layout) == Freshness(2, True, by_content=True, gt=1)


def test_the_library_badge_says_what_measure_measures(tmp_path):
    """GT first, as Measure: image ``B`` (a GT skeleton beside ``B_fake.PNG``) shows its GT,
    image ``C`` (``C_fake.PNG`` alone) its prediction."""
    layout = _gt_project(str(tmp_path / "p"), "B_fake.PNG", "edited")
    cv2.imwrite(os.path.join(layout.root, "C.jpg"), np.full((96, 128, 3), 128, np.uint8))
    _write(os.path.join(layout.skeletons_dir, "C_fake.PNG"), _curve())
    manager = ImageManager()
    manager.original_folder = layout.root
    manager.images = {stem: os.path.join(layout.root, stem + ".jpg") for stem in "ABC"}

    manager.refresh_outputs()

    states = {stem: manager.skeleton_state(stem) for stem in "ABC"}
    assert states == {"A": "prediction", "B": "gt", "C": "prediction"}


# --------------------------------------------------------------------------- both names
def _case_sensitive(folder: str) -> bool:
    probe = os.path.join(folder, "Probe")
    with open(probe, "wb"):
        pass
    try:
        return not os.path.exists(os.path.join(folder, "probe"))
    finally:
        os.remove(probe)


@pytest.fixture
def case_sensitive_root(tmp_path) -> str:
    root = str(tmp_path / "cs")
    os.makedirs(root)
    if sys.platform == "win32":
        subprocess.run(
            ["fsutil", "file", "setCaseSensitiveInfo", root, "enable"],
            capture_output=True,
            check=False,
        )
    if not _case_sensitive(root):
        pytest.skip("needs a case-sensitive folder (fsutil on Windows; the ubuntu CI)")
    return root


def test_where_both_names_exist_the_exact_one_is_measured(case_sensitive_root):
    layout = _project(case_sensitive_root, {"B_fake.PNG": "B"})
    if sys.platform == "win32":  # a new sub-folder inherits the flag; make sure
        subprocess.run(
            ["fsutil", "file", "setCaseSensitiveInfo", layout.skeletons_dir, "enable"],
            capture_output=True,
            check=False,
        )
    if not _case_sensitive(layout.skeletons_dir):
        pytest.skip("needs a case-sensitive skeletons folder")
    _write(os.path.join(layout.skeletons_dir, "B_fake.png"), _skeleton())

    listed, shadowed = layout.scan_skeleton_files()
    run = _measure_length(layout)

    assert listed == {"B": os.path.join(layout.skeletons_dir, "B_fake.png")}
    assert run.duplicates == ("B_fake.PNG",)
    assert [row["Image"] for row in run.rows] == ["B"]
    assert layout.find_skeleton("B") == os.path.join(layout.skeletons_dir, "B_fake.png")
    # The status count compares the file Measure read with the record of its inputs.
    csv_path = os.path.join(layout.skeletons_dir, "root_lengths.csv")
    fresh = input_freshness(
        layout.skeletons_dir,
        SKELETON_SUFFIX,
        csv_path,
        record=load_inputs_record(csv_path),
        project_root=layout.root,
        images=layout.scan_images(),
        gt_folder=layout.gt_skeletons_dir,
    )
    assert fresh == Freshness(1, True, by_content=True, duplicates=1)
    index = MeasurementIndex(layout.root)
    index.prewarm()
    assert index.outdated("B") == {"length": False, "area": False}
    assert index.measured_from("B")["length"] == ("skeletons/B_fake.png", "prediction")


@pytest.mark.parametrize(
    "names",
    [("B_fake.png", "B_fake.PNG"), ("b_fake.PNG", "B_fake.PNG")],
    ids=["exact and variant", "two variants"],
)
def test_beside_a_gt_skeleton_a_case_variant_is_not_even_a_duplicate(
    case_sensitive_root, tmp_path, qapp, names
):
    """Image ``B`` with a GT skeleton: its predictions whose suffix is in another case are
    not read at all (not a duplicate either), as before #3; the status count agrees."""
    layout = _gt_project(case_sensitive_root, names[0], "edited")
    _case_sensitive_skeletons(layout)
    _write(os.path.join(layout.skeletons_dir, names[1]), _curve())
    exact = [name for name in names if name.endswith(SKELETON_SUFFIX)]
    before = _gt_project(str(tmp_path / "q"), exact[0] if exact else None, "edited")

    rows = _press_measure_length(layout, qapp)

    assert rows == _press_measure_length(before, qapp)
    inputs, recorded = _what_measure_wrote(layout)
    assert (inputs, recorded) == _what_measure_wrote(before)
    assert inputs["duplicates"] == [] and inputs["n_duplicate"] == 0
    assert _status_count(layout) == _status_count(before)
    assert _status_count(layout).duplicates == 0


def test_a_case_variant_alone_is_found_on_a_case_sensitive_disk(case_sensitive_root):
    """The lookups of Correct and the Library display fold the suffix's case there too."""
    layout = _project(case_sensitive_root, {"B_fake.PNG": "B"})

    assert layout.find_skeleton("B") == os.path.join(layout.skeletons_dir, "B_fake.PNG")
    assert layout.find_skeleton("b") == os.path.join(layout.skeletons_dir, "B_fake.PNG")


def _case_sensitive_skeletons(layout: ProjectLayout) -> None:
    """Make sure the new ``skeletons/`` keeps case too (a new sub-folder inherits the flag)."""
    if sys.platform == "win32":
        subprocess.run(
            ["fsutil", "file", "setCaseSensitiveInfo", layout.skeletons_dir, "enable"],
            capture_output=True,
            check=False,
        )
    if not _case_sensitive(layout.skeletons_dir):
        pytest.skip("needs a case-sensitive skeletons folder")


def test_a_case_variant_never_takes_the_row_of_an_exact_suffix(case_sensitive_root, tmp_path):
    """Image ``B`` with ``b_fake.png`` (measured before #3) and ``B_FAKE.png`` (named like the
    image, the suffix in another case): the row stays the one of ``b_fake.png``, the variant
    is a duplicate, and Correct and the display show the file Measure measured."""
    layout = _project(case_sensitive_root, {"A_fake.png": "A"})
    _case_sensitive_skeletons(layout)
    _write(os.path.join(layout.skeletons_dir, "b_fake.png"), _skeleton())
    short = np.zeros((48, 64), np.uint8)
    short[20, 5:20] = 255  # another length: the row shows which file it was measured from
    _write(os.path.join(layout.skeletons_dir, "B_FAKE.png"), short)
    before = _project(str(tmp_path / "q"), {"A_fake.png": "A", "b_fake.png": "B"})

    listed, shadowed = layout.scan_skeleton_files()
    run = _measure_length(layout)
    rows = {row["Image"]: row for row in run.rows}
    alone = {row["Image"]: row for row in _measure_length(before).rows}

    exact = os.path.join(layout.skeletons_dir, "b_fake.png")
    assert listed == {"A": os.path.join(layout.skeletons_dir, "A_fake.png"), "b": exact}
    assert shadowed == [("B", os.path.join(layout.skeletons_dir, "B_FAKE.png"))]
    assert rows == alone  # the rows a project without the variant gets
    assert run.duplicates == ("B_FAKE.png",)
    assert layout.find_skeleton("B") == exact
    assert layout.scan_skeleton_stems() == {"A", "b"}  # the Library badge
    csv_path = os.path.join(layout.skeletons_dir, "root_lengths.csv")
    fresh = input_freshness(
        layout.skeletons_dir,
        SKELETON_SUFFIX,
        csv_path,
        record=load_inputs_record(csv_path),
        project_root=layout.root,
        images=layout.scan_images(),
        gt_folder=layout.gt_skeletons_dir,
    )
    assert fresh == Freshness(2, True, by_content=True, duplicates=1)
    index = MeasurementIndex(layout.root)
    index.prewarm()
    assert index.outdated("B") == {"length": False, "area": False}
    assert index.measured_from("B")["length"] == ("skeletons/b_fake.png", "prediction")


_EXACT = ("Ab_fake.png", "AB_fake.png", "ab_fake.png")
_VARIANTS = ("Ab_FAKE.png", "ab_fake.PNG", "AB_Fake.png", "Ab_fake.PNG")


def test_correct_measure_and_the_status_count_pick_the_same_file(case_sensitive_root):
    """Every set of skeletons of image ``Ab``: Correct's lookup finds the file Measure keeps,
    the status count has one input and the others as duplicates, and while one file has the
    exact suffix the kept file is the one Measure kept before #3 (the exact name, else the
    name that sorts first)."""
    layout = ProjectLayout(case_sensitive_root)
    os.makedirs(layout.skeletons_dir)
    _case_sensitive_skeletons(layout)
    names = (*_EXACT, *_VARIANTS)
    held: set[str] = set()
    for mask in range(1, 2 ** len(names)):
        chosen = {name for i, name in enumerate(names) if mask >> i & 1}
        for name in held - chosen:
            os.remove(os.path.join(layout.skeletons_dir, name))
        for name in chosen - held:
            open(os.path.join(layout.skeletons_dir, name), "wb").close()
        held = chosen

        listed, shadowed = layout.scan_skeleton_files()
        split = split_inputs(listed, ["Ab"], shadowed)
        fresh = input_freshness(
            layout.skeletons_dir, SKELETON_SUFFIX, None, project_root=layout.root, images=["Ab"]
        )

        assert len(split.kept) == 1 and not split.without_image, chosen
        kept = next(iter(split.kept.values()))
        assert layout.find_skeleton("Ab") == kept, chosen
        assert len(split.duplicates) == len(chosen) - 1, chosen
        assert (fresh.inputs, fresh.duplicates) == (1, len(chosen) - 1), chosen
        exact = chosen & set(_EXACT)
        if exact:
            before = "Ab_fake.png" if "Ab_fake.png" in exact else min(exact)
            assert os.path.basename(kept) == before, chosen


def test_correct_skips_a_suffix_with_a_kelvin_sign(case_sensitive_root):
    """``str.lower()`` maps U+212A KELVIN SIGN to "k": Measure does not read ``A_fa\u212ae.png``
    (:func:`skeleton_stem`), so Correct and the display do not show it either."""
    layout = ProjectLayout(case_sensitive_root)
    os.makedirs(layout.skeletons_dir)
    _case_sensitive_skeletons(layout)
    open(os.path.join(layout.skeletons_dir, "A_fa\u212ae.png"), "wb").close()

    assert layout.scan_skeleton_files() == ({}, [])
    assert layout.find_skeleton("A") is None
