"""ClassesDocument: the class labels of one image, without Qt (WG-T4).

The document is the Trace GT editor's storage half (``GtTracingInterface``) with the Qt
removed: the class-id plane and its patch history, the pixel count of every id kept from
each patch (no "checked" ticks since 2026-11-01: the class panel's box is the active class), ids read back from the canvas by exact colour match, Save
(``gt/classes.json`` first when it is missing, then ``gt/<stem>.png``, always written) and
the open of a remapped, unreadable or read-only file. It never reads ``mask/`` or
``skeletons/``.

Ported here (the old files were deleted in WG-T11): ``tests/test_gt_editor_save_order.py`` (the
registry first) and ``tests/test_gt_editor_speed_paths.py`` (``ids_from_argb``'s fallback,
the counts from patches, the exact histogram, Save's checks).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from app.config import GT_META_CHUNK
from app.gt import (
    Tracing,
    default_registry,
    present_keys,
    read_tracing,
    registry_path,
    save_registry,
    write_tracing,
)
from app.gt import store as gt_store
from app.gt.store import gt_path
from app.gui.correct import classes_document
from app.gui.correct.classes_document import ClassesDocument, ClassesRead, ClassesSaveResult
from app.gui.correct.layer import LayerStatus
from app.io_utils import imwrite_unicode
from app.project_layout import ProjectLayout
from tests.gt_goldens_support import GT_STEM, make_gt_project, write_legacy_tracing
from tests.gt_support import break_a_chunk, write_header_bomb

ROOT = Path(__file__).resolve().parents[1]
W, H = 640, 480
LABELS = f"gt/{GT_STEM}.png"


# ------------------------------------------------------------------------------ fixtures
def _project(tmp_path, case: str) -> SimpleNamespace:
    image, folder = make_gt_project(str(tmp_path), case)
    layout = ProjectLayout(folder)
    return SimpleNamespace(image=image, folder=folder, labels=layout.gt_path(GT_STEM))


@pytest.fixture
def gt_project(tmp_path):
    """A 640 x 480 photo, nothing under ``gt/``."""
    return _project(tmp_path, "two_classes_crossing")


@pytest.fixture
def gt_project_lossy(tmp_path):
    """``gt/<stem>.png`` whose id 2 has a key the registry does not know (one warning)."""
    return _project(tmp_path, "lossy_remap_on_load")


@pytest.fixture
def gt_project_bad_registry(tmp_path):
    """An invalid ``gt/classes.json``."""
    project = _project(tmp_path, "empty_save")
    os.makedirs(os.path.join(project.folder, "gt"))
    with open(registry_path(project.folder), "w", encoding="utf-8") as fh:
        fh.write("{")
    return project


@pytest.fixture
def recorder(monkeypatch):
    """The order the registry and the labels file are written in (the real writes run)."""
    order: list[str] = []
    store = classes_document.gt  # the document writes through app.gt
    real_registry, real_tracing = store.save_registry, store.write_tracing

    def registry(project_dir, registry):
        order.append("classes.json")
        return real_registry(project_dir, registry)

    def tracing(path, tracing, registry):
        order.append(os.path.basename(path))
        return real_tracing(path, tracing, registry)

    monkeypatch.setattr(store, "save_registry", registry)
    monkeypatch.setattr(store, "write_tracing", tracing)
    return SimpleNamespace(order=order)


# ------------------------------------------------------------------------------ helpers
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


def undo(doc: ClassesDocument) -> None:
    patch = doc.history.undo()
    if patch is not None:
        doc.count_patch(patch, False)
        doc.touch()


def redo(doc: ClassesDocument) -> None:
    patch = doc.history.redo()
    if patch is not None:
        doc.count_patch(patch, True)
        doc.touch()


def expected_counts(doc: ClassesDocument) -> dict[str, int]:
    """``doc.counts()`` computed from scratch: pixels per id, named by the registry."""
    names = {c.id: c.key for c in doc.registry.classes}
    names[0] = "background"
    counts = np.bincount(doc.history.plane.ravel(), minlength=256)
    return {names.get(i, str(i)): int(n) for i, n in enumerate(counts) if n}


def file_bytes(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


# ------------------------------------------------------------------------------ layering
#: What a class-label module must never load: Qt, torch, and the mask's storage (Design 3.7:
#: one layer, one file; the document reaches ``gt/<stem>.png`` through ``app.gt`` only).
_PROBE = (
    "import importlib, sys; importlib.import_module(sys.argv[1]); "
    "mask = ('app.masks', 'app.gui.correct.mask_document', 'app.gui.correct.mask_files'); "
    "print(sorted(m for m in sys.modules if m.split('.')[0] in ('PyQt6', 'torch') "
    "or m in mask or m.startswith('app.masks.')))"
)


@pytest.mark.parametrize(
    "module",
    [
        "app.gui.correct.classes_document",
        "app.gui.correct.classes_files",
        "app.gui.correct.classes_paint",
    ],
)
def test_the_document_is_qt_free(module):
    """Qt-free, and it never imports the mask's storage (a static guard: a module that
    imported ``app.masks`` would fail here even if no test ever reached the call)."""
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, module],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


# ------------------------------------------------------------------------------ open
def test_opens_empty_and_never_reads_mask_or_skeletons(gt_project, monkeypatch):
    full = np.full((H, W), 255, np.uint8)
    for folder, name in (("mask", f"{GT_STEM}.png"), ("skeletons", f"{GT_STEM}_fake.png")):
        os.makedirs(os.path.join(gt_project.folder, folder))
        imwrite_unicode(os.path.join(gt_project.folder, folder, name), full, durable=False)

    def boom(*_args, **_kwargs):
        raise AssertionError("the class labels read a model output")

    monkeypatch.setattr("app.masks.store.MaskStore.read", boom)
    monkeypatch.setattr("app.io_utils.imread_unicode", boom)
    doc = ClassesDocument()

    assert doc.open(gt_project.image, (W, H)) == []
    assert doc.counts() == {"background": W * H}
    assert not doc.is_dirty() and not doc.is_read_only() and not doc.lossy_remap()
    assert doc.status() == LayerStatus("none", "No class labels")
    assert doc.active_key == "root"


def test_a_lossy_remap_opens_dirty_with_warnings_and_needs_a_prompt(gt_project_lossy):
    doc = ClassesDocument()

    notices = doc.open(gt_project_lossy.image, (W, H))

    assert [n.kind for n in notices] == ["warn"] and doc.is_dirty() and doc.lossy_remap()
    assert "unknown_x" in notices[0].message
    assert doc.status() == LayerStatus(
        "dirty",
        "Unsaved · remapped",
        "Unknown classes were mapped to Ignore:\n" + notices[0].message.split(": ", 1)[1],
    )
    assert (doc.history.plane[300:400, 350:500] == 255).all()  # unknown_x became Ignore
    assert doc.counts() == {"background": 272200, "root": 20000, "ignore": 15000}


def test_a_discarded_remap_is_clean_and_only_a_save_ends_the_prompt(gt_project_lossy):
    doc = opened(gt_project_lossy)
    doc.revert()  # Discard: the remapped ids as opened, nothing written

    assert not doc.is_dirty() and doc.lossy_remap()  # a later save still writes the remap
    assert doc.save().ok
    assert not doc.lossy_remap() and doc.warnings == ()
    assert (read_tracing(gt_project_lossy.labels).ids[300:400, 350:500] == 255).all()


def test_an_invalid_registry_makes_the_document_read_only(gt_project_bad_registry):
    doc = ClassesDocument()

    notices = doc.open(gt_project_bad_registry.image, (W, H))

    # The open sends no notice: the status says it, and the layer's first refused edit
    # (LOCKED_NOTICE, once per image; review WG-T5 r1).
    assert doc.is_read_only() and doc.status().label == "Read-only" and notices == []
    assert doc.status() == LayerStatus(
        "error", "Read-only", f"gt/classes.json is invalid: {doc.registry_error}", local=True
    )
    assert doc.registry == default_registry()
    assert doc.save() == ClassesSaveResult(False, None, doc.registry_error)
    assert not os.path.exists(gt_project_bad_registry.labels)
    assert not doc.is_dirty()
    assert ClassesDocument.LOCKED_NOTICE.format(err=doc.registry_error) == (
        f"Editing is off: gt/classes.json is invalid ({doc.registry_error}). "
        "Fix the file, then reopen the image."
    )


NOT_SHOWN = {
    "unreadable": (
        "Saved class labels could not be read: they are not shown and are kept unless you "
        "save over them"
    ),
    "size": (
        "Saved class labels do not match the image size: they are not shown and are kept "
        "unless you save over them"
    ),
}


@pytest.mark.parametrize("kind", ["unreadable", "size"])
def test_a_file_that_cannot_be_used_opens_empty_not_shown_and_kept(gt_project, kind):
    os.makedirs(os.path.dirname(gt_project.labels))
    if kind == "unreadable":
        with open(gt_project.labels, "wb") as fh:
            fh.write(b"not a png")
    else:
        write_tracing(gt_project.labels, Tracing(np.ones((10, 10), np.uint8)), default_registry())
    before = file_bytes(gt_project.labels)
    doc = ClassesDocument()

    notices = doc.open(gt_project.image, (W, H))

    assert [(n.message, n.kind) for n in notices] == [(NOT_SHOWN[kind], "warn")]
    assert not doc.history.plane.any() and not doc.is_dirty() and doc.unloaded_file()
    assert doc.status() == LayerStatus("error", "Class labels not shown", NOT_SHOWN[kind])
    assert file_bytes(gt_project.labels) == before
    commit(doc, (10, 10, 5, 5), 1)
    assert doc.save().ok and not doc.unloaded_file()  # saved over: shown from now on
    assert read_tracing(gt_project.labels).ids.shape == (H, W)


# ------------------------------------------------------------------------------ no ticks
def test_the_document_has_no_checked_state(gt_project):
    """Owner decision 2026-11-01: the class panel's tick is the active class, so the
    document keeps no "checked" / reviewed set and picking a class is no edit."""
    doc = opened(gt_project)
    for name in ("ticked", "reviewed", "set_ticked"):
        assert not hasattr(doc, name)

    for key in ("nodule", "ignore", "root_tip"):
        assert doc.set_active_class(key) and doc.active_key == key
        assert not doc.is_dirty()


def test_an_older_file_with_a_reviewed_list_opens_clean(gt_project):
    """A file written before 2026-11-01 lists its "checked" classes: the list is ignored,
    the file opens clean, shows as GT, and a save writes it without the list."""
    ids = np.zeros((H, W), np.uint8)
    ids[5:10, 5:10] = 2
    save_registry(gt_project.folder, default_registry())
    write_legacy_tracing(gt_project.labels, Tracing(ids), default_registry(), {"root", "nodule"})
    doc = ClassesDocument()

    assert doc.open(gt_project.image, (W, H)) == []

    assert not doc.is_dirty() and np.array_equal(doc.history.plane, ids)
    assert doc.status() == LayerStatus("clean", "GT", LABELS)
    assert doc.matches_disk() and doc.read_back(doc.file_stamps()) is not None
    commit(doc, (40, 30, 1, 1), 1)
    assert doc.save().ok
    with Image.open(gt_project.labels) as image:
        assert "reviewed" not in json.loads(image.text[GT_META_CHUNK])


# ------------------------------------------------------------------------------ classes
def test_the_active_class_and_its_brush_size(gt_project):
    doc = opened(gt_project)
    retired = default_registry().add(key="dead_root", name="Dead root", color="#7f5af0")
    doc.apply_registry(retired.retire("dead_root"))

    assert doc.set_active_class("nodule") is True and doc.active_key == "nodule"
    assert doc.set_active_class("dead_root") is False  # retired: not paintable
    assert doc.set_active_class("no_such_class") is False and doc.active_key == "nodule"
    assert doc.brush_size("nodule") == 8  # the default brush
    doc.set_brush_size("nodule", 25)
    assert (doc.brush_size("nodule"), doc.brush_size("root")) == (25, 8)


def test_fill_source_is_the_active_class_only(gt_project):
    doc = opened(gt_project)
    commit(doc, (10, 10, 3, 3), 1)
    commit(doc, (20, 20, 3, 3), 2)
    doc.set_active_class("nodule")

    outline, value = doc.fill_source(doc.history.plane)

    assert value == 2 and outline.dtype == np.uint8
    assert np.array_equal(outline > 0, doc.history.plane == 2)
    assert set(np.unique(outline).tolist()) == {0, 255}


def test_apply_registry_recolours_and_keeps_the_ids(gt_project):
    doc = opened(gt_project)
    commit(doc, (10, 10, 3, 3), 1)
    assert doc.save().ok
    ids = doc.history.plane.copy()
    doc.set_active_class("nodule")
    recoloured = doc.registry.update("root", color="#123456").retire("nodule")

    doc.apply_registry(recoloured)

    assert doc.registry == recoloured and np.array_equal(doc.history.plane, ids)
    assert tuple(doc.lut()[1]) == (0x56, 0x34, 0x12, 255)
    assert doc.active_key == "root"  # the retired class is no longer paintable
    assert not doc.is_dirty()


# ------------------------------------------------------------------------------ save
def test_save_writes_the_registry_first_then_the_file(gt_project, recorder):
    doc = opened(gt_project)
    commit(doc, (100, 100, 20, 20), 1)

    result = doc.save()

    assert recorder.order == ["classes.json", f"{GT_STEM}.png"]
    assert result == ClassesSaveResult(True, doc.gt_path)
    assert os.path.samefile(doc.gt_path, gt_project.labels)
    commit(doc, (10, 10, 2, 2), 2)
    assert doc.save().ok
    assert recorder.order[2:] == [f"{GT_STEM}.png"]  # the registry exists now


def test_an_empty_save_writes_a_file(gt_project):
    doc = opened(gt_project)

    assert doc.save() == ClassesSaveResult(True, doc.gt_path)

    assert os.path.isfile(gt_project.labels)
    tracing = read_tracing(gt_project.labels)
    assert tracing.ids.shape == (H, W) and tracing.ids.max() == 0
    assert doc.status() == LayerStatus("clean", "GT", LABELS)  # as the mask and the skeleton


def test_an_empty_save_is_offered_only_for_untouched_labels_without_a_file(gt_project):
    """Mark empty (2026-11-02): a reason while it would write anything but an empty
    file over no file, "" otherwise (the plane is all background then)."""
    doc = ClassesDocument()
    assert doc.empty_save_refusal().startswith("Open an image")
    doc.open(gt_project.image, (W, H))

    assert doc.empty_save_refusal() == "" and not doc.history.plane.any()
    commit(doc, (10, 10, 5, 5), 1)
    assert doc.empty_save_refusal() == "Save or discard your painted class labels first"
    undo(doc)
    assert doc.empty_save_refusal() == ""
    commit(doc, (10, 10, 5, 5), 0)  # background over background: nothing to save
    assert doc.empty_save_refusal() == ""

    assert doc.save().ok
    assert doc.empty_save_refusal() == "This image already has class labels"


def test_an_empty_save_is_refused_while_the_registry_is_invalid(gt_project_bad_registry):
    doc = opened(gt_project_bad_registry)

    assert doc.empty_save_refusal().startswith("gt/classes.json is invalid")


@pytest.mark.parametrize("kind", ["unreadable", "size"])
def test_an_empty_save_is_refused_over_a_kept_file(gt_project, kind):
    os.makedirs(os.path.dirname(gt_project.labels))
    if kind == "unreadable":
        with open(gt_project.labels, "wb") as fh:
            fh.write(b"not a png")
    else:
        write_tracing(gt_project.labels, Tracing(np.ones((10, 10), np.uint8)), default_registry())

    doc = opened(gt_project)

    assert "could not be shown" in doc.empty_save_refusal()


def test_save_writes_no_reviewed_list(gt_project):
    doc = opened(gt_project)
    commit(doc, (100, 100, 20, 20), 1)  # root pixels
    doc.set_active_class("root_tip")

    assert doc.save().ok

    with Image.open(gt_project.labels) as image:
        meta = json.loads(image.text[GT_META_CHUNK])
    assert set(meta) == {"v", "classes"} and not doc.is_dirty()
    undo(doc)  # root's pixels go
    assert doc.is_dirty() and doc.status() == LayerStatus("dirty")


def test_a_failed_registry_write_leaves_the_file_alone(gt_project, monkeypatch):
    ids = np.zeros((H, W), np.uint8)
    ids[5:10, 5:10] = 2
    write_tracing(gt_project.labels, Tracing(ids), default_registry())
    before = file_bytes(gt_project.labels)  # saved by another copy: no classes.json yet
    doc = opened(gt_project)
    commit(doc, (40, 30, 1, 1), 1)

    def locked(*_args, **_kwargs):
        raise OSError("classes.json is locked")

    monkeypatch.setattr(classes_document.gt, "save_registry", locked)

    result = doc.save()

    assert not result.ok and result.error == "classes.json is locked"
    assert doc.is_dirty()
    assert doc.status() == LayerStatus("error", "Not saved", "classes.json is locked")
    assert file_bytes(gt_project.labels) == before  # the "not saved" edit is not in the file
    commit(doc, (41, 30, 1, 1), 1)
    assert doc.status() == LayerStatus("dirty")  # until the next edit


def test_a_first_save_without_a_file_writes_nothing_when_the_registry_fails(
    gt_project, monkeypatch
):
    doc = opened(gt_project)
    commit(doc, (40, 30, 1, 1), 1)

    def denied(*_args, **_kwargs):
        raise OSError("denied")

    monkeypatch.setattr(classes_document.gt, "save_registry", denied)
    assert doc.save().ok is False
    assert not os.path.exists(gt_project.labels)

    monkeypatch.undo()  # the registry can be written again: the next save writes both

    assert doc.save().ok is True
    assert read_tracing(gt_project.labels).ids[30, 40] == 1
    assert os.path.isfile(registry_path(gt_project.folder))


def test_save_writes_the_plane_of_a_random_session(gt_project):
    doc = opened(gt_project, size=(120, 90))
    rng = np.random.default_rng(4)
    for i, _ in enumerate(_random_session(doc, rng, 30)):
        if i % 5 == 4:
            plane = doc.history.plane.copy()
            assert doc.save().ok
            assert np.array_equal(read_tracing(gt_project.labels).ids, plane)
            assert not doc.is_dirty()


# ------------------------------------------------------------------------------ revert
def test_revert_puts_back_the_saved_ids_with_an_empty_history(gt_project):
    doc = opened(gt_project)
    commit(doc, (100, 100, 20, 20), 1)
    assert doc.save().ok
    saved = doc.history.plane.copy()
    commit(doc, (10, 10, 30, 30), 2)
    doc.set_active_class("root_tip")

    doc.revert()

    assert np.array_equal(doc.history.plane, saved) and doc.active_key == "root_tip"
    assert not doc.is_dirty() and not doc.history.can_undo
    assert doc.counts() == expected_counts(doc)


# ------------------------------------------------------------------------------ counts
def _random_session(doc: ClassesDocument, rng, steps: int):
    """Random actions on the plane (boxes of random ids, Clear, undo, redo, a class
    picked)."""
    height, width = doc.history.plane.shape
    ids = np.array([0, 1, 2, 3, 255, 7], np.uint8)  # 7: no class owns it
    keys = ["root", "nodule", "root_tip"]
    for _ in range(steps):
        op = rng.integers(0, 6)
        if op == 0:
            undo(doc)
        elif op == 1:
            redo(doc)
        elif op == 2:
            commit(doc, (0, 0, width, height), 0)  # Clear
        elif op == 3:
            doc.set_active_class(keys[int(rng.integers(0, 3))])
        else:
            x, y = int(rng.integers(0, width - 1)), int(rng.integers(0, height - 1))
            w = int(rng.integers(1, width - x + 1))
            h = int(rng.integers(1, height - y + 1))
            commit(doc, (x, y, w, h), rng.choice(ids, size=(h, w)).astype(np.uint8))
        yield


@pytest.mark.parametrize("seed", range(3))
def test_counts_follow_the_patches(gt_project, seed):
    doc = opened(gt_project, size=(120, 90))
    rng = np.random.default_rng(seed)
    for _ in _random_session(doc, rng, 80):
        assert doc.counts() == expected_counts(doc)
        assert doc.present_keys() == present_keys(Tracing(doc.history.plane), doc.registry)


def test_counts_stay_exact_on_the_bincount_path(gt_project, monkeypatch):
    monkeypatch.setattr(gt_store, "_EXACT_HIST_MAX", 50)
    doc = opened(gt_project, size=(120, 90))
    rng = np.random.default_rng(9)
    for _ in _random_session(doc, rng, 40):
        assert doc.counts() == expected_counts(doc)


# ------------------------------------------------------------------------------ ids_from_argb


# ------------------------------------------------------------------------------ files
def test_file_stamps_cover_the_file_and_the_registry_and_write_stamps_the_file(gt_project):
    doc = ClassesDocument()
    assert doc.file_stamps() == () and doc.write_stamps() == ()
    doc.open(gt_project.image, (W, H))

    stamps = doc.file_stamps()

    assert len(stamps) == 2 and doc.write_stamps() == (stamps[0],)
    assert os.path.normcase(stamps[0][0]) == os.path.normcase(gt_project.labels)
    assert os.path.normcase(stamps[1][0]) == os.path.normcase(registry_path(gt_project.folder))
    assert [stamp[1:] for stamp in stamps] == [(None,), (None,)]  # neither exists yet
    assert doc.save().ok
    assert all(len(stamp) == 3 for stamp in doc.file_stamps())


def test_read_matches_open(gt_project_lossy):
    read = ClassesDocument.read(gt_project_lossy.image, (W, H))
    doc = ClassesDocument()

    notices = doc.open(gt_project_lossy.image, (W, H), read)

    assert isinstance(read, ClassesRead) and read.nbytes == W * H
    assert doc.history.plane is read.tracing.ids  # adopted, not read again
    assert (doc.history.plane == read.tracing.ids).all()
    assert notices == ClassesDocument().open(gt_project_lossy.image, (W, H))
    assert read.warnings and read.changed and read.error == "" and read.registry_error == ""


def test_a_read_of_files_that_changed_since_is_not_used(gt_project):
    read = ClassesDocument.read(gt_project.image, (W, H))
    ids = np.zeros((H, W), np.uint8)
    ids[:10, :10] = 1
    write_tracing(gt_project.labels, Tracing(ids), default_registry())
    doc = ClassesDocument()

    doc.open(gt_project.image, (W, H), read)

    assert doc.history.plane[:10, :10].all() and doc.has_saved_file


def test_matches_disk_reads_the_files_again(gt_project):
    doc = opened(gt_project)
    commit(doc, (100, 100, 20, 20), 1)
    assert doc.save().ok
    assert doc.matches_disk()

    save_registry(gt_project.folder, default_registry())
    write_tracing(gt_project.labels, Tracing(doc.history.plane), doc.registry)
    assert doc.matches_disk()  # rewritten as it was
    write_legacy_tracing(gt_project.labels, Tracing(doc.history.plane), doc.registry, {"root"})
    assert doc.matches_disk()  # an old reviewed list is no difference

    other = doc.history.plane.copy()
    other[0, 0] = 2
    write_tracing(gt_project.labels, Tracing(other), doc.registry)
    assert not doc.matches_disk()


def test_read_back_hands_the_clean_document_back(gt_project):
    doc = opened(gt_project)
    commit(doc, (100, 100, 20, 20), 1)
    assert doc.read_back(doc.file_stamps()) is None  # unsaved
    assert doc.save().ok
    stamps = doc.file_stamps()

    read = doc.read_back(stamps)

    assert isinstance(read, ClassesRead) and read.stamps == stamps
    assert read.tracing.ids is doc.history.plane
    other = ClassesDocument()
    assert other.open(gt_project.image, (W, H), read) == []
    assert other.history.plane is read.tracing.ids and not other.is_dirty()


def test_read_back_is_none_while_a_remap_is_only_in_memory(gt_project_lossy):
    doc = opened(gt_project_lossy)
    doc.revert()  # clean, but the file still holds the old ids

    assert doc.read_back(doc.file_stamps()) is None


@pytest.fixture
def image(tmp_path):
    """A photo with an empty ``gt/`` next to it."""
    path = str(tmp_path / "plot_T1_L1_2024.01.01_120000.png")
    Image.new("RGB", (W, H), (10, 20, 30)).save(path)
    os.makedirs(os.path.dirname(gt_path(path)))
    return path


def _too_large(path: str) -> None:
    write_header_bomb(path)


def _broken(path: str) -> None:
    Image.new("P", (W, H)).save(path)
    break_a_chunk(path)


@pytest.mark.parametrize("make", [_too_large, _broken], ids=["too_large", "broken_chunk"])
def test_correct_opens_refused_labels_empty_with_the_notice(image, make):
    labels = gt_path(image)
    make(labels)
    before = open(labels, "rb").read()
    doc = ClassesDocument()

    assert ClassesDocument.read(image, (W, H)).error == "unreadable"  # the read-ahead's read
    notices = doc.open(image, (W, H))

    assert [(n.message, n.kind) for n in notices] == [(NOT_SHOWN["unreadable"], "warn")]
    assert doc.image_path == image and doc.unloaded_file() and not doc.history.plane.any()
    assert doc.status() == LayerStatus("error", "Class labels not shown", NOT_SHOWN["unreadable"])
    assert open(labels, "rb").read() == before  # kept unless saved over
