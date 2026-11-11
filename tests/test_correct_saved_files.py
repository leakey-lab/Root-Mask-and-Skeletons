"""What a Correct save writes and reads back: the GT files of the mask and the skeleton.

Measure reads ``gt/mask/<stem>.png`` and ``gt/skeletons/<stem>.png`` first (O1), so these
files are the paper's numbers. The documents (Qt-free) promise:

- a save changes exactly one file of the project: an edit writes its GT file, an edit back to
  the prediction removes it, a save without an edit changes nothing (the whole folder is
  compared by bytes, files that appear or vanish included);
- a fresh open of what was saved gives back exactly the edited plane / skeleton, clean;
- the skeleton is compared with the prediction on disk at save time: a prediction rewritten
  or deleted since Load prediction (a Generate run) never makes the user's GT vanish;
- all of it also in a folder whose name has a space and non-ASCII letters.

Seeded random edit sequences (``numpy.random.default_rng(seed)``) against oracles that share no
code with the documents: the plane / mask as last saved, kept by the test, and the files'
bytes. A failure message names the seed and the step, so it can be replayed.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest
from skimage.morphology import skeletonize

from app.gui.correct.mask_document import MaskDocument
from app.gui.correct.skeleton_document import SkeletonDocument
from app.io_utils import imread_unicode, imwrite_unicode

STEM = "plot_T001_L001_2024.07.18_111530_001_SMM"
W, H = 96, 72
SEEDS = range(20)
STEPS = 6


def tree(folder: str) -> dict[str, bytes]:
    """``relative path -> bytes`` of every file under ``folder``."""
    files = {}
    for dirpath, _dirs, names in os.walk(folder):
        for name in names:
            path = os.path.join(dirpath, name)
            with open(path, "rb") as fh:
                files[os.path.relpath(path, folder).replace(os.sep, "/")] = fh.read()
    return files


def changed(before: dict[str, bytes], after: dict[str, bytes]) -> list[str]:
    """The files that appeared, vanished or changed bytes."""
    return sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p))


def write(path: str, pixels: np.ndarray) -> None:
    """Write an image file, its folder first (unicode-safe, as the generator writes)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imwrite_unicode(path, pixels, durable=False)


def photo(root: str) -> str:
    """The photo (the documents never decode it: the page passes its size)."""
    path = os.path.join(root, f"{STEM}.jpg")
    write(path, np.full((H, W, 3), 90, np.uint8))
    return path


# ===================================================================================== mask
MASK_GT = f"gt/mask/{STEM}.png"


def mask_project(root: str) -> tuple[str, np.ndarray]:
    """A photo and its prediction ``mask/<stem>.png`` (8-bit grey, 0 / 255); returns the photo
    and the prediction's 0/1 plane."""
    image = photo(root)
    prediction = np.zeros((H, W), np.uint8)
    cv2.ellipse(prediction, (40, 30), (25, 12), 20, 0, 360, 1, -1)
    write(os.path.join(root, "mask", f"{STEM}.png"), prediction * 255)
    return image, prediction


def random_mask_edit(doc: MaskDocument, rng, prediction: np.ndarray) -> None:
    """One user action on the plane, as the layer commits it (``history`` then ``touch``)."""
    kind = rng.choice(["box", "box", "erase", "clear", "prediction", "none"])
    if kind == "none":
        return
    if kind == "clear":
        doc.history.commit((0, 0, W, H), np.zeros((H, W), np.uint8))
    elif kind == "prediction":
        doc.history.commit((0, 0, W, H), prediction.copy())
    else:
        x, y = int(rng.integers(0, W - 1)), int(rng.integers(0, H - 1))
        w, h = int(rng.integers(1, W - x + 1)), int(rng.integers(1, H - y + 1))
        value = 1 if kind == "box" else 0
        doc.history.commit((x, y, w, h), np.full((h, w), value, np.uint8))
    doc.touch()


def test_each_mask_save_changes_only_its_gt_file_and_reopens_as_edited(tmp_path):
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        root = str(tmp_path / f"seed{seed}")
        image, prediction = mask_project(root)
        doc = MaskDocument()
        doc.open(image, (W, H))
        saved = prediction.copy()  # the oracle: what the files hold, kept by the test
        for step in range(STEPS):
            where = f"seed {seed} step {step}"
            random_mask_edit(doc, rng, prediction)
            plane = doc.history.plane.copy()
            gt_there = os.path.isfile(os.path.join(root, *MASK_GT.split("/")))
            before = tree(root)

            assert doc.save().ok, where

            if np.array_equal(plane, saved):
                expected = []  # nothing unsaved: nothing written (O2)
            elif np.array_equal(plane, prediction):
                expected = [MASK_GT] if gt_there else []  # back to the prediction: GT removed
            else:
                expected = [MASK_GT]
            assert changed(before, tree(root)) == expected, where
            saved = plane
            fresh = MaskDocument()
            fresh.open(image, (W, H))
            assert np.array_equal(fresh.history.plane, plane), where
            assert not fresh.is_dirty() and doc.matches_disk(), where


# ================================================================================= skeleton
SKELETON_GT = f"gt/skeletons/{STEM}.png"


def thin(mask: np.ndarray) -> np.ndarray:
    """scikit-image's thinning as uint8 0/255 (independent of the document)."""
    return skeletonize(mask > 0).astype(np.uint8) * 255


def skeleton_project(root: str) -> tuple[str, np.ndarray]:
    """A photo and its prediction ``skeletons/<stem>_fake.png``: 1 px lines at the photo's
    size that are their own thinning, so the prediction as Correct opens it is the file's set
    pixels. Returns the photo and those pixels (0/255)."""
    image = photo(root)
    raster = np.zeros((H, W), np.uint8)
    cv2.polylines(raster, [np.array([[5, 10], [40, 20], [90, 15]], np.int32)], False, 255, 1)
    cv2.line(raster, (20, 60), (70, 40), 255, 1)
    prediction = thin(raster)
    write(os.path.join(root, "skeletons", f"{STEM}_fake.png"), prediction)
    return image, prediction


def random_skeleton_edit(doc: SkeletonDocument, rng) -> None:
    """One user action on the skeleton, as the tools make it (``push_undo`` first)."""
    model = doc.model
    kind = rng.choice(["line", "line", "erase", "clear", "prediction", "none"])
    if kind == "none":
        return
    if kind == "prediction":
        plane, _notices = doc.read_prediction()
        if plane is not None:
            doc.load_prediction(plane)
        return
    model.push_undo()
    if kind == "clear":
        model.clear()
        return
    points = [(int(rng.integers(0, W)), int(rng.integers(0, H))) for _ in range(rng.integers(2, 5))]
    if kind == "line":
        model.draw_polyline(points, thickness=3)
    else:
        model.erase_segment(points[0], points[1], int(rng.integers(1, 6)))
    model.skeletonize()


def test_each_skeleton_save_changes_only_its_gt_file_and_reopens_as_edited(tmp_path):
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        root = str(tmp_path / f"seed{seed}")
        image, prediction = skeleton_project(root)
        doc = SkeletonDocument()
        doc.open(image, (W, H), root)
        assert np.array_equal(doc.model.mask > 0, prediction > 0), f"seed {seed}: not thin"
        saved = prediction > 0
        for step in range(STEPS):
            where = f"seed {seed} step {step}"
            random_skeleton_edit(doc, rng)
            mask = doc.model.mask > 0
            ends = sorted(doc.model.endpoints())
            gt_there = os.path.isfile(os.path.join(root, *SKELETON_GT.split("/")))
            before = tree(root)

            assert doc.save().ok, where

            if np.array_equal(mask, saved):
                expected = []
            elif np.array_equal(mask, prediction > 0):
                expected = [SKELETON_GT] if gt_there else []
            else:
                expected = [SKELETON_GT]
            assert changed(before, tree(root)) == expected, where
            saved = mask
            fresh = SkeletonDocument()
            fresh.open(image, (W, H), root)
            assert np.array_equal(fresh.model.mask > 0, mask), where
            assert sorted(fresh.model.endpoints()) == ends, where
            assert not fresh.is_dirty(), where


@pytest.mark.parametrize("meanwhile", ["rewritten", "deleted"])
def test_a_prediction_changed_after_load_prediction_never_removes_the_skeleton_gt(
    tmp_path, meanwhile
):
    """The mask's twin is ``test_correct_mask_gt.py::test_a_prediction_rewritten_while_editing_
    is_not_taken_for_the_gt``: Save compares with the prediction on disk now."""
    root = str(tmp_path)
    image, prediction = skeleton_project(root)
    gt_path = os.path.join(root, *SKELETON_GT.split("/"))
    drawn = np.zeros((H, W), np.uint8)
    drawn[50, 10:80] = 255
    write(gt_path, drawn)
    doc = SkeletonDocument()
    doc.open(image, (W, H), root)
    plane, _notices = doc.read_prediction()
    doc.load_prediction(plane)  # the user sees the prediction P
    shown = doc.model.mask > 0
    generated = os.path.join(root, "skeletons", f"{STEM}_fake.png")
    if meanwhile == "rewritten":  # a Generate run writes P'
        newer = np.zeros((H, W), np.uint8)
        newer[30, 10:80] = 255
        write(generated, newer)
        stat = os.stat(generated)
        os.utime(generated, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    else:
        os.remove(generated)
    before = tree(root)

    result = doc.save()

    assert result.ok and not result.removed and os.path.isfile(gt_path)
    assert changed(before, tree(root)) == [SKELETON_GT]  # the prediction is left as it is
    saved = imread_unicode(gt_path, cv2.IMREAD_GRAYSCALE) > 0
    assert np.array_equal(saved, shown)  # P's pixels, as the user saw them


# ============================================================== a folder with a strange name
@pytest.mark.parametrize("layer", ["mask", "skeleton"])
def test_a_folder_with_a_space_and_non_ascii_letters_saves_and_reopens(tmp_path, layer):
    root = str(tmp_path / "Wurzel Bilder äöü 根")
    if layer == "mask":
        image, _prediction = mask_project(root)
        doc = MaskDocument()
        doc.open(image, (W, H))
        doc.history.commit((10, 50, 30, 10), np.ones((10, 30), np.uint8))
        doc.touch()
        edited, gt = doc.history.plane.copy(), MASK_GT
    else:
        image, _prediction = skeleton_project(root)
        doc = SkeletonDocument()
        doc.open(image, (W, H), root)
        assert doc.model.mask.any()  # the prediction was read from that folder
        doc.model.push_undo()
        doc.model.draw_polyline([(10, 65), (85, 65)], thickness=3)
        doc.model.skeletonize()
        edited, gt = doc.model.mask > 0, SKELETON_GT
    before = tree(root)

    assert doc.save().ok

    assert changed(before, tree(root)) == [gt]
    if layer == "mask":
        fresh = MaskDocument()
        fresh.open(image, (W, H))
        assert fresh.source == "gt" and np.array_equal(fresh.history.plane, edited)
    else:
        fresh = SkeletonDocument()
        fresh.open(image, (W, H), root)
        assert fresh.source == "gt" and np.array_equal(fresh.model.mask > 0, edited)
