"""The class labels' paint rules (``app/gui/correct/classes_paint.py``, Qt-free).

``class_lut`` shows each class (retired included) in its colour, background transparent and
ids no class owns grey; ``canvas_ids`` reads a painted canvas box back to ids by exact colour
match, keeping the committed id under a colour no class owns. ``_reference_ids_from_argb`` is
Trace GT's plain matching before its fast path, kept as the reference: random canvases (class
colours, transparent, unowned colours, noise alpha) give the same ids, also with a class in the
unowned grey.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.gt import (
    default_registry,
)
from app.gui.correct.classes_document import ClassesDocument
from app.gui.correct.classes_paint import canvas_ids, class_lut
from app.project_layout import ProjectLayout
from tests.gt_goldens_support import GT_STEM, make_gt_project

ROOT = Path(__file__).resolve().parents[1]
W, H = 640, 480
LABELS = f"gt/{GT_STEM}.png"


def _project(tmp_path, case: str) -> SimpleNamespace:
    image, folder = make_gt_project(str(tmp_path), case)
    layout = ProjectLayout(folder)
    return SimpleNamespace(image=image, folder=folder, labels=layout.gt_path(GT_STEM))


@pytest.fixture
def gt_project(tmp_path):
    """A 640 x 480 photo, nothing under ``gt/``."""
    return _project(tmp_path, "two_classes_crossing")


def opened(project, size=(W, H)) -> ClassesDocument:
    doc = ClassesDocument()
    doc.open(project.image, size)
    return doc


def commit(doc: ClassesDocument, rect, value) -> None:
    """One action as the layer records it: a patch, its counts, a new revision."""
    x, y, w, h = rect
    after = np.full((h, w), value, np.uint8) if np.isscalar(value) else value
    patch = doc.history.commit(rect, after)
    doc.count_patch(patch, True)
    doc.touch()


def test_the_lut_shows_each_class_in_its_colour_and_background_transparent(gt_project):
    doc = opened(gt_project)
    lut = doc.lut()

    assert lut.shape == (256, 4) and lut.dtype == np.uint8
    assert tuple(lut[0]) == (0, 0, 0, 0)
    assert tuple(lut[2]) == (0xB6, 0x7E, 0xFF, 255)  # nodule #ff7eb6 as B, G, R, A
    assert tuple(lut[7]) == (0x80, 0x80, 0x80, 255)  # an id no class owns: grey
    assert np.array_equal(lut, class_lut(doc.registry))


def test_ids_from_argb_keeps_the_committed_id_under_an_unmatched_colour(gt_project):
    doc = opened(gt_project)
    row = np.array([[0, 0, 1, 2, 7, 255, 0, 0]], np.uint8)  # 7: no class owns it (grey)
    commit(doc, (8, 10, 8, 1), row)
    lut = doc.lut()
    argb = lut[row].copy()  # what the committed plane shows
    argb[0, 0] = lut[2]  # nodule painted over background
    argb[0, 2] = (9, 9, 9, 255)  # a colour no class owns over root: root stays
    argb[0, 5] = 0  # erased Ignore: background
    argb[0, 6] = (9, 9, 9, 255)  # a colour no class owns over background: background

    ids = doc.ids_from_argb(argb, (8, 10, 8, 1))

    assert ids.tolist() == [[2, 0, 1, 2, 7, 0, 0, 0]]
    assert ids.dtype == np.uint8 and not np.shares_memory(ids, doc.history.plane)


def _reference_ids_from_argb(arr, registry, fallback=None):
    """Trace GT's implementation before its fast path (G3), verbatim."""
    arr = np.ascontiguousarray(arr)
    packed = arr.view("<u4")[:, :, 0]
    ids = np.zeros(packed.shape, np.uint8)
    known = arr[:, :, 3] == 0
    for c in registry.classes:
        b, g, r, a = class_lut(registry)[c.id]
        hit = packed == (int(b) | int(g) << 8 | int(r) << 16 | int(a) << 24)
        ids[hit] = c.id
        known |= hit
    other = ~known
    if fallback is not None and other.any():
        ids[other] = fallback[other]
    return ids


@pytest.mark.parametrize("seed", range(6))
def test_ids_from_argb_matches_the_reference(seed):
    """Random canvases: class colours, transparent, colours no class owns, noise alpha."""
    rng = np.random.default_rng(seed)
    registry = default_registry().add(key="hypha", name="Hypha", color="#3fa7ff").retire("nodule")
    lut = class_lut(registry)
    h, w = 37, 53
    plane_ids = np.array([0, 1, 2, 3, 4, 255, 7, 200], np.uint8)  # 7, 200: no class owns them
    fallback = rng.choice(plane_ids, size=(h, w)).astype(np.uint8)
    canvas = lut[fallback].copy()
    painted = rng.random((h, w)) < 0.3
    colours = np.vstack(
        [lut[[0, 1, 2, 3, 4, 255]], [[1, 2, 3, 255], [10, 20, 30, 128], [0, 0, 0, 255]]]
    ).astype(np.uint8)
    canvas[painted] = colours[rng.integers(0, len(colours), painted.sum())]
    for fb in (fallback, None, fallback[3:30, 5:40]):
        arr = canvas if fb is None or fb is fallback else canvas[3:30, 5:40]
        expected = _reference_ids_from_argb(arr, registry, fb)
        got = canvas_ids(arr, registry, fb)
        assert got.dtype == np.uint8 and got.shape == expected.shape
        assert np.array_equal(got, expected)
        if fb is not None:
            assert not np.shares_memory(got, fb)  # a new array: the history compares it


def test_ids_from_argb_matches_the_reference_with_a_grey_class():
    """A class in the grey shown for ids no class owns: those pixels still match like before."""
    registry = default_registry().update("nodule", color="#808080")
    lut = class_lut(registry)
    fallback = np.array([[0, 1, 7, 7], [2, 200, 255, 3]], np.uint8)
    canvas = lut[fallback].copy()
    canvas[0, 0] = lut[1]
    expected = _reference_ids_from_argb(canvas, registry, fallback)
    assert expected[0, 2] == 2 and expected[1, 1] == 2  # the grey is the nodule's colour now
    assert np.array_equal(canvas_ids(canvas, registry, fallback), expected)
