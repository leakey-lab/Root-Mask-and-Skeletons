"""``ImageManager`` (app/gui/image_manager.py): the Library's state for a project folder.

- The library hierarchy is built once per load (lazily, or taken from the loader thread),
  what the window is told when a scan is adopted, and ``load_images`` without a folder.
- Which images have GT (mask, skeleton, class labels: WG-T1), GT first in the source states,
  and ``set_gt`` / ``refresh_outputs`` following the files.
- The output sets hold the Library's own stem strings, not a second copy: a folder scan
  makes a new string for every file name, and ``masks`` / ``skeleton_stems`` are keyed by
  the very strings of ``images`` (about 8 MiB for the masks and skeletons of 46k images).
"""

from __future__ import annotations

import os
import time

import pytest
from PyQt6.QtCore import QCoreApplication, QEvent
from PyQt6.QtWidgets import QApplication, QFileDialog

from app.gui import image_manager as im
from app.gui.library_index import index_names
from app.project_layout import SKELETON_SUFFIX, ProjectLayout
from tests.window_support import write_photos

pytestmark = pytest.mark.usefixtures("no_temp_sweep")

STEMS = ["plot_T1_L1_2024.01.01_120000", "plot_T1_L2_2024.01.01_120000"]


class _Window:
    """The one main-window slot :meth:`ImageManager.on_loading_finished` calls."""

    def __init__(self) -> None:
        self.loaded: list[tuple] = []

    def on_loading_finished(self, *args) -> None:
        self.loaded.append(args)


def _images(root) -> dict[str, str]:
    return {stem: os.path.join(str(root), f"{stem}.png") for stem in STEMS}


def _count_index_names(monkeypatch) -> list[int]:
    calls: list[int] = []

    def counted(names):
        calls.append(1)
        return index_names(names)

    monkeypatch.setattr(im, "index_names", counted)
    return calls


def test_the_hierarchy_is_built_once_until_another_load(monkeypatch, tmp_path):
    calls = _count_index_names(monkeypatch)
    manager = im.ImageManager(None)
    manager.on_loading_finished(_images(tmp_path), {}, {}, False)  # no folder: no scans
    assert calls == []  # lazily, on the first use

    first = manager.get_hierarchical_structure()
    assert manager.get_hierarchical_structure() is first
    assert set(manager.get_leaf_labels()) == set(STEMS)
    assert len(calls) == 1

    manager.on_loading_finished(_images(tmp_path), {}, {}, False)
    second = manager.get_hierarchical_structure()
    assert second is not first and second == first
    assert len(calls) == 2


def test_a_precomputed_hierarchy_is_kept_even_when_empty(monkeypatch, tmp_path):
    """The loader thread's hierarchy is used as it is: only None means "not built"."""
    hierarchy, labels = index_names(_images(tmp_path))
    calls = _count_index_names(monkeypatch)
    manager = im.ImageManager(None)

    manager.on_loading_finished(
        _images(tmp_path), {}, {}, False, hierarchy=hierarchy, labels=labels
    )
    assert manager.get_hierarchical_structure() is hierarchy
    assert manager.get_leaf_labels() is labels

    manager.on_loading_finished(_images(tmp_path), {}, {}, False, hierarchy={}, labels={})
    assert manager.get_hierarchical_structure() == {}
    assert manager.get_leaf_labels() == {}
    assert calls == []


def test_clearing_the_images_drops_the_cached_hierarchy(monkeypatch, tmp_path):
    calls = _count_index_names(monkeypatch)
    manager = im.ImageManager(None)
    manager.on_loading_finished(_images(tmp_path), {}, {}, False)
    assert manager.get_hierarchical_structure()

    manager._clear_images()
    assert manager.get_hierarchical_structure() == {}  # rebuilt, from no images
    assert len(calls) == 2


def test_the_window_is_told_no_fake_real_pairs_even_after_an_overlay_was_found(tmp_path):
    """The window's 4th argument is False on every load, also after an overlay skeleton was
    found (that flag was reset by the load itself before it was read)."""
    (tmp_path / "skeletons").mkdir()
    skeleton = tmp_path / "skeletons" / f"{STEMS[0]}{SKELETON_SUFFIX}"
    skeleton.write_bytes(b"x")
    window = _Window()
    manager = im.ImageManager(window)
    manager.original_folder = str(tmp_path)

    manager.on_loading_finished(_images(tmp_path), {}, {}, False)
    manager.reload_for_view_mode("Overlay")
    assert manager.get_fake_image_path(STEMS[0]) == str(skeleton)
    manager.on_loading_finished(_images(tmp_path), {}, {}, True)

    assert [args[3] for args in window.loaded] == [False, False]
    assert all(args[1] is manager.fake_images for args in window.loaded)
    assert manager.fake_images == {}  # a load forgets the overlay paths


@pytest.mark.parametrize("folder", [None, ""])
def test_load_images_without_a_folder_starts_nothing(monkeypatch, folder):
    """The window's own load action owns the folder dialog; the manager never opens one."""
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", lambda *a, **k: pytest.fail("a dialog opened")
    )
    manager = im.ImageManager(None)
    manager.load_images(folder)
    assert manager.loader_thread is None and manager._generation == 0


# ------------------------------------------------------------------ GT (WG-T1)

GT_STEMS = ["a", "b", "c", "d"]


def _touch(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(b"x")


@pytest.fixture
def gt_project(tmp_path):
    """Images a..d; gt/mask/a.png, gt/skeletons/b.png, gt/c.png; mask/d.png (prediction)."""
    write_photos(tmp_path, GT_STEMS, size=(8, 6), suffix=".png")
    layout = ProjectLayout(str(tmp_path))
    _touch(layout.gt_mask_path("a"))
    _touch(layout.gt_skeleton_path("b"))
    _touch(layout.gt_path("c"))
    _touch(layout.mask_path("d"))
    return layout


def _load(layout: ProjectLayout) -> im.ImageManager:
    manager = im.ImageManager(None)
    manager.load_images(layout.root)
    end = time.monotonic() + 10
    while time.monotonic() < end and not manager.images:
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        QApplication.processEvents()
        time.sleep(0.005)
    assert manager.images
    return manager


def test_the_loader_scans_gt_after_the_predictions(gt_project, app):
    manager = _load(gt_project)
    assert manager.gt_mask_stems == {"a"}
    assert manager.gt_skeleton_stems == {"b"}
    assert manager.gt_label_stems == {"c"}
    assert manager.has_gt_mask("a") and not manager.has_gt_mask("b")
    assert manager.has_gt_skeleton("b") and not manager.has_gt_skeleton("a")
    assert manager.has_gt_labels("c") and not manager.has_gt_labels("a")
    assert manager.mask_state("a") == "gt"
    assert manager.mask_state("d") == "prediction"
    assert manager.mask_state("b") == "none"
    assert manager.skeleton_state("a") == "none"
    assert manager.skeleton_state("b") == "gt"


def test_a_gt_file_named_in_another_case_belongs_to_the_librarys_stem(tmp_path, app):
    write_photos(tmp_path, ["Plot"], size=(8, 6), color=(1, 2, 3), suffix=".png")
    layout = ProjectLayout(str(tmp_path))
    _touch(layout.gt_mask_path("plot"))
    manager = _load(layout)
    assert manager.gt_mask_stems == {"Plot"} and manager.has_gt_mask("Plot")


def test_set_gt_and_refresh_outputs_follow_the_files(gt_project, app):
    manager = _load(gt_project)

    manager.set_gt("d", "mask", True)
    assert manager.has_gt_mask("d")
    manager.set_gt(gt_project.gt_mask_path("d"), "mask", False)  # a path names the stem too
    assert not manager.has_gt_mask("d")
    manager.set_gt("d", "skeleton", True)
    manager.set_gt("d", "labels", True)
    assert manager.has_gt_skeleton("d") and manager.has_gt_labels("d")
    assert manager.skeleton_state("d") == "gt"
    with pytest.raises(ValueError):
        manager.set_gt("d", "classes", True)

    os.remove(gt_project.gt_mask_path("a"))
    _touch(gt_project.gt_skeleton_path("c"))
    manager.refresh_outputs()

    assert not manager.has_gt_mask("a")
    assert manager.gt_skeleton_stems == {"b", "c"}  # d's set_gt was a guess; the files rule
    assert manager.gt_label_stems == {"c"}
    assert manager.mask_state("a") == "none"


def test_a_new_load_forgets_the_gt_of_the_old_one(gt_project, tmp_path_factory, app):
    manager = _load(gt_project)
    other = tmp_path_factory.mktemp("other")
    write_photos(other, ["z"], size=(8, 6), color=(1, 2, 3), suffix=".png")
    manager.load_images(str(other))
    end = time.monotonic() + 10
    while time.monotonic() < end and "z" not in manager.images:
        QApplication.processEvents()
        time.sleep(0.005)
    assert manager.gt_mask_stems == set() == manager.gt_skeleton_stems == manager.gt_label_stems


# ------------------------------------------------------------------ the Library's strings

# Long names: CPython shares one object for each one-letter string, so those prove nothing.
LONG_STEMS = [
    f"24EFCAM1_T{tube:03d}_L{depth:03d}_2024.06.04_100001_001_SMM"
    for tube in (1, 2)
    for depth in range(1, 13)
]
ORPHAN = "24EFCAM1_T009_L001_2024.06.04_100001_001_SMM"  # an output of a deleted photo


def _project(root, *, skeletons=tuple(LONG_STEMS)) -> ProjectLayout:
    """``LONG_STEMS`` as photos with a mask each; a skeleton for each of ``skeletons``."""
    layout = ProjectLayout(str(root))
    for stem in LONG_STEMS:
        _touch(os.path.join(layout.root, stem + ".jpg"))
        _touch(os.path.join(layout.mask_dir, stem + ".png"))
    for stem in skeletons:
        _touch(os.path.join(layout.skeletons_dir, stem + SKELETON_SUFFIX))
    return layout


def _ids(strings) -> set[int]:
    return {id(s) for s in strings}


def test_a_load_keys_masks_and_skeletons_by_the_librarys_strings(app, tmp_path):
    _project(tmp_path, skeletons=LONG_STEMS[:-3])
    thread = im.ImageLoaderThread(str(tmp_path))
    delivered = []
    thread.finished.connect(lambda *args: delivered.append(args))

    thread.run()  # the loader's body, on this thread

    [(images, _fake_images, masks, _pairs)] = delivered
    assert sorted(images) == sorted(LONG_STEMS)
    assert set(masks) == set(LONG_STEMS)
    assert thread.skeleton_stems == set(LONG_STEMS[:-3])
    library = _ids(images)
    assert _ids(masks) <= library
    assert _ids(thread.skeleton_stems) <= library


def test_refresh_outputs_keys_them_by_the_librarys_strings(app, tmp_path):
    # A skeleton named in another case belongs to its photo (O7); an orphan keeps its name.
    layout = _project(tmp_path, skeletons=[LONG_STEMS[0].lower(), *LONG_STEMS[1:], ORPHAN])
    manager = im.ImageManager(None)
    manager.original_folder = layout.root
    manager.images = layout.scan_images()

    manager.refresh_outputs()

    library = _ids(manager.images)
    assert set(manager.masks) == set(LONG_STEMS) and _ids(manager.masks) <= library
    assert manager.skeleton_stems == {*LONG_STEMS, ORPHAN}
    assert {s for s in manager.skeleton_stems if id(s) not in library} == {ORPHAN}


# ------------------------------------------------------------------ the overlay skeleton's path
def test_the_overlay_finds_a_skeleton_named_in_another_case_and_remembers_a_miss(app, tmp_path):
    """On a case-sensitive disk ``skeletons/plot_fake.png`` is the overlay of image ``Plot``
    (O7); an image without one is not looked up again on every click until the outputs are
    read again (a generation run, a save)."""
    from tests.correct_support import case_sensitive_folder

    root = case_sensitive_folder(str(tmp_path / "project"))
    skeletons = os.path.join(root, "skeletons")
    os.makedirs(skeletons)
    _touch(os.path.join(skeletons, "plot" + SKELETON_SUFFIX))
    images = {stem: os.path.join(root, stem + ".png") for stem in ("Plot", "Other")}
    manager = im.ImageManager(None)
    manager.original_folder = root
    manager.on_loading_finished(images, {}, {}, False)
    manager.reload_for_view_mode("Overlay")

    found = manager.get_fake_image_path("Plot")
    assert found is not None and os.path.basename(found) == "plot" + SKELETON_SUFFIX

    assert manager.get_fake_image_path("Other") is None
    _touch(os.path.join(skeletons, "other" + SKELETON_SUFFIX))  # made outside the app
    assert manager.get_fake_image_path("Other") is None  # the miss is remembered
    manager.refresh_outputs()
    found = manager.get_fake_image_path("Other")
    assert found is not None and os.path.basename(found) == "other" + SKELETON_SUFFIX
