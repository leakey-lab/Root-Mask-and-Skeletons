"""What a class-label save writes and reads back: ``gt/<stem>.png``, the training labels.

The labels are exported as a training dataset, so the saved file must hold exactly the ids the
user painted. ``ClassesDocument`` (Qt-free) promises:

- a save changes exactly the image's labels file, and ``gt/classes.json`` when it is missing
  (the whole project folder is compared by bytes, files that appear or vanish included); a save
  of the same ids again changes no byte;
- ``read_tracing`` of the saved file gives the plane as edited, and a fresh open gives it back
  clean, as on disk (``matches_disk``);
- the ids at the edges survive: a class with id 254, Ignore (255), a retired class's id, 0;
- all of it also in a folder whose name has a space and non-ASCII letters.

Seeded random edit sequences (``numpy.random.default_rng(seed)``: boxes of ids, Clear, undo,
redo, a class picked) against oracles that share no code with the document: the plane the
test keeps and the folder's bytes. A failure message names the seed and the step.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from app.gt import Registry, default_registry, read_tracing, registry_path, save_registry
from app.gui.correct.classes_document import ClassesDocument
from app.io_utils import imwrite_unicode

STEM = "plot_T001_L001_2024.07.18_111530_001_SMM"
W, H = 96, 72
SEEDS = range(20)
STEPS = 6
LABELS = f"gt/{STEM}.png"
REGISTRY = "gt/classes.json"


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


def edge_registry() -> Registry:
    """The defaults, a class with the highest id (254) and a retired class (id 4)."""
    data = default_registry().to_dict()
    data["classes"].insert(-1, {"id": 254, "key": "edge", "name": "Edge", "color": "#3fa7ff"})
    data["classes"].insert(-1, {"id": 4, "key": "old", "name": "Old", "color": "#7f5af0"})
    data["classes"][-2]["retired"] = True
    return Registry.from_dict(data)


def project(root: str, *, with_registry: bool) -> str:
    """A photo (the document never decodes it: the page passes its size), its ``mask/`` and
    ``skeletons/`` predictions, and the edge registry when ``with_registry``; the photo."""
    os.makedirs(root, exist_ok=True)
    image = os.path.join(root, f"{STEM}.jpg")
    imwrite_unicode(image, np.full((H, W, 3), 90, np.uint8), durable=False)
    for folder, name in (("mask", f"{STEM}.png"), ("skeletons", f"{STEM}_fake.png")):
        os.makedirs(os.path.join(root, folder), exist_ok=True)
        imwrite_unicode(os.path.join(root, folder, name), np.full((H, W), 255, np.uint8))
    if with_registry:
        save_registry(root, edge_registry())
    return image


def commit(doc: ClassesDocument, rect, after: np.ndarray) -> None:
    """One action as the layer records it: a patch, its counts, a new revision."""
    patch = doc.history.commit(rect, after)
    doc.count_patch(patch, True)
    doc.touch()


def step(doc: ClassesDocument, rng, values: np.ndarray) -> str:
    """One random action on the open plane; its name (for the failure message)."""
    op = int(rng.integers(0, 6))
    if op == 0:
        patch = doc.history.undo()
        if patch is not None:
            doc.count_patch(patch, False)
            doc.touch()
        return "undo"
    if op == 1:
        patch = doc.history.redo()
        if patch is not None:
            doc.count_patch(patch, True)
            doc.touch()
        return "redo"
    if op == 2:
        commit(doc, (0, 0, W, H), np.zeros((H, W), np.uint8))
        return "clear"
    if op == 3:
        doc.set_active_class(str(rng.choice(["root", "nodule", "ignore"])))
        return "pick"
    x, y = int(rng.integers(0, W - 1)), int(rng.integers(0, H - 1))
    w, h = int(rng.integers(1, W - x + 1)), int(rng.integers(1, H - y + 1))
    commit(doc, (x, y, w, h), rng.choice(values, size=(h, w)).astype(np.uint8))
    return f"box {x},{y} {w}x{h}"


def run_session(root: str, seed: int) -> None:
    """``STEPS`` saves of a seeded session in ``root``, each checked against the folder."""
    rng = np.random.default_rng(seed)
    with_registry = seed % 2 == 0  # else the defaults: the first save writes classes.json
    image = project(root, with_registry=with_registry)
    values = np.array([0, 1, 2, 3, 255] + ([254, 4] if with_registry else []), np.uint8)
    doc = ClassesDocument()
    assert doc.open(image, (W, H)) == []
    saved: np.ndarray | None = None
    for n in range(STEPS):
        where = f"seed {seed}, step {n}"
        actions = [step(doc, rng, values) for _ in range(int(rng.integers(1, 4)))]
        where += f" ({', '.join(actions)})"
        plane = doc.history.plane.copy()
        before = tree(root)

        assert doc.save().ok, where

        after = tree(root)
        expected = [] if saved is not None and np.array_equal(plane, saved) else [LABELS]
        if saved is None and not with_registry:
            expected = [REGISTRY, LABELS]
        assert changed(before, after) == expected, where
        on_disk = read_tracing(os.path.join(root, LABELS))
        assert np.array_equal(on_disk.ids, plane), where
        assert not doc.is_dirty() and doc.matches_disk(), where
        fresh = ClassesDocument()
        assert fresh.open(image, (W, H)) == [], where
        assert np.array_equal(fresh.history.plane, plane), where
        assert not fresh.is_dirty() and fresh.status().state == "clean", where
        assert doc.counts() == fresh.counts(), where  # the pixels per class, kept from patches
        saved = plane
    assert os.path.isfile(registry_path(root))


@pytest.mark.parametrize("seed", SEEDS)
def test_each_labels_save_changes_only_its_file_and_reopens_as_painted(tmp_path, seed):
    run_session(str(tmp_path / "project"), seed)


def test_a_folder_with_a_space_and_non_ascii_letters_saves_and_reopens(tmp_path):
    run_session(str(tmp_path / "Wurzeln ü ő 根"), 2)


def test_an_edge_registrys_ids_survive_a_save_and_a_reopen(tmp_path):
    """0, the retired id 4, the highest class id 254 and Ignore 255, side by side."""
    root = str(tmp_path / "project")
    image = project(root, with_registry=True)
    doc = ClassesDocument()
    doc.open(image, (W, H))
    row = np.array([[0, 4, 254, 255, 1, 2, 3, 0]], np.uint8)
    commit(doc, (10, 20, 8, 1), row)

    assert doc.save().ok

    ids = read_tracing(os.path.join(root, LABELS)).ids
    assert ids[20, 10:18].tolist() == row[0].tolist()
    fresh = ClassesDocument()
    assert fresh.open(image, (W, H)) == [] and not fresh.is_dirty()
    assert fresh.history.plane[20, 10:18].tolist() == row[0].tolist()
    assert fresh.present_keys() == {"old", "edge", "root", "nodule", "root_tip"}
