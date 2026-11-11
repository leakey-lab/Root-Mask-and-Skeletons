"""The Library's per-image status in the window: the badge, the overlay and the metrics bar
follow Generate, a Correct save, a removal back to the prediction and a rescan.

The badge reads ``ImageManager`` state at paint time, so a save or a removal costs one row
repaint and no rescan. ``gt/skeletons/<stem>.png`` (what Correct saves; Measure reads it
first, O1) is drawn by the Library's Overlay and Side by side instead of the generated
skeleton, and the metrics bar says "Skeleton: GT" (W17).

Offscreen; projects are tmp_path folders of tiny photos.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QApplication, QMessageBox, QStyleOptionViewItem

from app.gui.canvas_layers.skeleton import SkeletonLayerItem
from app.gui.correct.layer import MASK, SKELETON
from app.gui.library_index import status_text
from app.project_layout import ProjectLayout
from tests.correct_support import H, W, draw_line, paint, write_mask, write_skeleton
from tests.qt_teardown import delete_widgets
from tests.qt_wait import settle
from tests.window_project_support import STEMS, enter, leaf, new_project, open_project, select
from tests.window_project_support import mw as mw  # fixture
from tests.window_support import build_window, open_folder, quiet_window, write_photos

pytestmark = pytest.mark.usefixtures("window_guards")

#: The five photos of the bare-window (``mw``) tests.
FIVE_STEMS = [f"24EFCAM1_T001_L0{d}_2024.07.18_1115{d}_001_SMM" for d in (11, 12, 13, 14, 15)]


@pytest.fixture
def window(app, tmp_path):
    made = [build_window(tmp_path / "s.ini")]
    yield made[0]
    delete_widgets(made, prepare=quiet_window)
    QApplication.processEvents()


def icon_key(window, stem: str) -> int:
    """The cache key of the badge the Library paints for ``stem``."""
    delegate = window.file_list.itemDelegate()
    option = QStyleOptionViewItem()
    delegate.initStyleOption(
        option, window.file_list.indexFromItem(window.library_index.item_for(stem))
    )
    assert not option.icon.isNull()
    return option.icon.cacheKey()


def badge_words(window, stem: str) -> str:
    return status_text(*window.library_index.status_of(stem))


def test_the_badge_follows_saves_removals_and_a_rescan(window, tmp_path, monkeypatch):
    project = new_project(tmp_path)  # image 0: a model mask and a generated skeleton
    open_project(window, project)
    stem, manager, index = STEMS[0], window.image_manager, window.library_index
    assert index.status_of(stem) == ("prediction", "prediction", False)
    key = icon_key(window, stem)

    repaints = []
    real = index.refresh_status
    monkeypatch.setattr(
        index, "refresh_status", lambda name=None: repaints.append(name) or real(name)
    )
    select(window, 0)
    enter(window, "Correct")
    editor = window.correct_editor

    paint(editor, (20, 100), (60, 100))  # a mask save (WG-T2)
    assert editor.save()
    settle()
    assert index.status_of(stem) == ("gt", "prediction", False)
    assert stem in repaints  # the save itself repainted the row
    after_mask = icon_key(window, stem)
    assert after_mask != key

    repaints.clear()
    editor.set_active_layer(SKELETON)
    draw_line(editor)  # a skeleton save (WG-T3)
    assert editor.save()
    settle()
    assert index.status_of(stem) == ("gt", "gt", False)
    assert stem in repaints
    assert icon_key(window, stem) != after_mask
    assert badge_words(window, stem) == "Mask: GT · Skeleton: GT · No classes"

    editor.set_active_layer(MASK)  # back to the prediction: the GT mask goes
    editor.load_prediction()
    assert editor.save()
    settle()
    assert not manager.has_gt_mask(stem) and manager.has_gt_skeleton(stem)
    assert index.status_of(stem) == ("prediction", "gt", False)

    # class labels come from the scan of gt/<stem>.png: a rescan finds them
    labels = ProjectLayout(project.folder).gt_path(stem)
    write_mask(labels, np.ones((H, W), np.uint8))
    assert index.status_of(stem)[2] is False
    manager.refresh_outputs()
    index.refresh_status()
    assert index.status_of(stem) == ("prediction", "gt", True)
    assert badge_words(window, stem) == "Mask: prediction · Skeleton: GT · Classes: traced"


def test_a_gt_mask_on_disk_reaches_the_badge_and_correct_through_the_load(window, tmp_path):
    project = new_project(tmp_path)  # image 0 has a model mask in mask/
    write_mask(project.gt_mask(0), np.ones((H, W), np.uint8))  # and a saved GT mask in gt/mask/
    assert os.path.isfile(project.mask(0))

    open_project(window, project)  # the normal load path: no set_gt by hand

    stem = STEMS[0]
    assert window.image_manager.has_mask(stem)  # the prediction is still there
    assert window.image_manager.mask_state(stem) == "gt"
    assert window.library_index.status_of(stem)[0] == "gt"
    assert badge_words(window, stem).startswith("Mask: GT")
    assert badge_words(window, STEMS[1]).startswith("No mask")
    select(window, 0)
    enter(window, "Correct")
    assert window.correct_editor.layers[MASK].status().label == "GT"


def _corrected_project(root):
    """Image 0: a generated skeleton on row 110 and a GT skeleton on row 40."""
    project = new_project(root)
    write_skeleton(project.correction(0), row=40)
    return project


def _overlay_rows(window) -> set[int]:
    """Rows of the image that the Library's skeleton overlay draws."""
    items = [
        item for item in window.display_area.scene().items() if isinstance(item, SkeletonLayerItem)
    ]
    assert len(items) == 1
    mask = items[0]._mask
    assert mask is not None and mask.shape == (H, W)
    return set(np.nonzero(mask.any(axis=1))[0].tolist())


def _overlay_mode(window) -> None:
    window.view_mode_combo.setCurrentIndex(1)  # Overlay
    settle()


def test_the_overlay_draws_the_gt_skeleton_when_one_exists(window, tmp_path):
    project = _corrected_project(tmp_path)
    open_project(window, project)
    _overlay_mode(window)

    select(window, 0)

    assert window.display_controller.current_fake_image is not None
    assert os.path.normcase(window.display_controller.current_fake_image) == os.path.normcase(
        os.path.normpath(project.correction(0))
    )
    assert _overlay_rows(window) == {40}


def test_without_a_gt_skeleton_the_overlay_draws_the_generated_skeleton(window, tmp_path):
    project = new_project(tmp_path)
    open_project(window, project)
    _overlay_mode(window)

    select(window, 0)

    assert _overlay_rows(window) == {110}


def test_side_by_side_shows_the_gt_skeleton_too(window, tmp_path):
    project = _corrected_project(tmp_path)
    open_project(window, project)
    window.view_mode_combo.setCurrentIndex(2)  # Side by side
    settle()

    select(window, 0)

    assert os.path.normcase(window.display_controller.current_fake_image) == os.path.normcase(
        os.path.normpath(project.correction(0))
    )


def test_the_metrics_bar_says_skeleton_gt_and_what_is_drawn(window, tmp_path):
    open_project(window, _corrected_project(tmp_path))

    select(window, 0)

    bar = window.metrics_bar
    assert bar._status_value.text() == "Mask: prediction · Skeleton: GT · No classes"
    tip = bar._status_cell.toolTip()
    assert "gt/skeletons/" in tip and "prediction (skeletons/)" in tip
    select(window, 1)
    assert bar._status_value.text() == "No mask · No skeleton · No classes"
    assert bar._status_cell.toolTip() == ""


def test_the_library_row_marks_a_gt_skeleton(window, tmp_path):
    open_project(window, _corrected_project(tmp_path))
    index = window.library_index

    assert window.image_manager.has_gt_skeleton(STEMS[0])
    assert not window.image_manager.has_gt_skeleton(STEMS[1])
    assert index.status_of(STEMS[0]) == ("prediction", "gt", False)
    assert index.status_of(STEMS[1]) == ("none", "none", False)
    assert "Skeleton: GT" in index.tooltip_for(STEMS[0])
    assert "Skeleton: GT" not in index.tooltip_for(STEMS[1])
    assert leaf(window, 0) is not None


def test_a_skeleton_saved_in_correct_reaches_the_library(window, tmp_path):
    project = new_project(tmp_path)
    open_project(window, project)
    select(window, 1)
    enter(window, "Correct")
    editor = window.correct_editor
    assert not window.image_manager.has_gt_skeleton(STEMS[1])

    draw_line(editor)
    assert editor.save()
    settle()

    assert os.path.isfile(project.correction(1))
    assert window.image_manager.has_gt_skeleton(STEMS[1])
    assert window.library_index.status_of(STEMS[1])[1] == "gt"
    # The window hears skeleton_saved once (WG-T11), and it is not a generated skeleton.
    assert editor.receivers(editor.skeleton_saved) == 1
    assert not window.image_manager.has_skeleton(STEMS[1])

    enter(window, "Library")
    _overlay_mode(window)
    assert os.path.normcase(window.display_controller.current_fake_image) == os.path.normcase(
        os.path.normpath(project.correction(1))
    )
    assert window.metrics_bar._status_value.text() == "No mask · Skeleton: GT · No classes"


def test_a_gt_skeleton_edited_back_to_the_prediction_leaves_the_library(window, tmp_path):
    project = _corrected_project(tmp_path)
    open_project(window, project)
    select(window, 0)
    enter(window, "Correct")
    editor = window.correct_editor
    assert window.library_index.status_of(STEMS[0])[1] == "gt"

    editor.layers[SKELETON].load_file(project.generated(0))
    editor.session.forget(SKELETON)  # Load clears the layer's own history
    assert editor.save()
    settle()

    assert not os.path.isfile(project.correction(0))
    assert not window.image_manager.has_gt_skeleton(STEMS[0])
    assert window.library_index.status_of(STEMS[0])[1] == "prediction"


def _load(mw, folder):
    """Open ``folder`` (all of FIVE_STEMS) and wait until the load has ended."""
    open_folder(mw, folder, len(FIVE_STEMS))
    return folder


def _select(mw, stem):
    leaf = mw.find_tree_item_by_image_name(stem)
    mw.file_list.setCurrentItem(leaf)
    mw.on_image_selected(leaf)
    return leaf


def test_load_results_accepts_the_skeletons_folder(mw, tmp_path, monkeypatch):
    folder = _load(mw, write_photos(tmp_path / "p", FIVE_STEMS))
    _select(mw, FIVE_STEMS[0])
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a))
    skel = folder / "skeletons"
    skel.mkdir()
    QImage(8, 8, QImage.Format.Format_Grayscale8).save(str(skel / f"{FIVE_STEMS[0]}_fake.png"))

    mw.load_results(str(skel))

    assert errors == []
    assert os.path.normcase(mw.image_manager.processed_base_path) == os.path.normcase(str(skel))
    assert mw.image_manager.has_skeleton(FIVE_STEMS[0])
    assert mw.metrics_bar._status_value.text() == "No mask · Skeleton: prediction · No classes"


def test_load_results_keeps_the_legacy_images_layout(mw, tmp_path, monkeypatch):
    _load(mw, write_photos(tmp_path / "p", FIVE_STEMS))
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a))
    legacy = tmp_path / "p" / "out" / "images"
    legacy.mkdir(parents=True)
    mw.load_results(str(legacy.parent))
    assert errors == []


def test_load_results_reports_a_missing_folder(mw, tmp_path, monkeypatch):
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a))
    mw.load_results(str(tmp_path / "nope"))
    assert len(errors) == 1


def test_on_outputs_changed_picks_up_generated_masks(mw, tmp_path):
    folder = _load(mw, write_photos(tmp_path / "p", FIVE_STEMS))
    _select(mw, FIVE_STEMS[1])
    assert mw.metrics_bar._status_value.text() == "No mask · No skeleton · No classes"
    (folder / "mask").mkdir()
    QImage(8, 8, QImage.Format.Format_ARGB32).save(str(folder / "mask" / f"{FIVE_STEMS[1]}.png"))
    mw.on_outputs_changed()
    assert mw.image_manager.has_mask(FIVE_STEMS[1])
    assert mw.metrics_bar._status_value.text() == "Mask: prediction · No skeleton · No classes"


def test_saved_mask_counts_and_survives_stage_switches_without_stats(mw, tmp_path, monkeypatch):
    folder = _load(mw, write_photos(tmp_path / "p", FIVE_STEMS))
    _select(mw, FIVE_STEMS[2])
    image_path = str(folder / f"{FIVE_STEMS[2]}.jpg")
    mw.toggle_correct()
    mw.on_mask_saved(image_path)  # an editor save is GT (WG-T2), never a prediction
    assert mw.image_manager.has_gt_mask(FIVE_STEMS[2]) and not mw.mask_exists(FIVE_STEMS[2])

    calls = []
    real_exists = os.path.exists
    monkeypatch.setattr(os.path, "exists", lambda p: calls.append(p) or real_exists(p))
    mw.toggle_correct()  # leave
    mw.toggle_correct()  # enter again
    mw.toggle_correct()  # leave
    monkeypatch.setattr(os.path, "exists", real_exists)
    # Only the editor looks up the open image's own mask: no per-leaf status stats.
    mask_stats = [p for p in calls if os.path.basename(os.path.dirname(str(p))) == "mask"]
    assert all(FIVE_STEMS[2] in str(p) for p in mask_stats)
    assert mw.image_manager.has_gt_mask(FIVE_STEMS[2])

    mw.on_mask_cleared(image_path)
    assert not mw.image_manager.has_gt_mask(FIVE_STEMS[2])
