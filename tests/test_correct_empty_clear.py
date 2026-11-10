"""Clear on an empty Mask or Skeleton layer changes nothing and says so (review 2026-11-10).

Clear on a layer with no set pixel added an undo step that undid nothing, and the mask's
notice promised "Save writes an empty GT mask" while nothing was unsaved, so Save wrote
nothing (O2). Now a layer with no set pixel adds no step and toasts ``No mask to clear`` /
``No skeleton to clear``. A layer with a set pixel clears as before, with its old notice.
An unsaved plane already emptied by the eraser stays unsaved: Clear adds nothing, and Save
writes it. (The Classes layer words its own two notices: ``test_correct_classes_layer``.)
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from app.gui.correct.layer import MASK, SKELETON
from app.gui.correct.mask_layer import MaskLayer
from app.gui.correct.skeleton_layer import SkeletonLayer
from tests.correct_support import H, W, drag, plane, raster, record, settle, write_mask

pytestmark = pytest.mark.usefixtures("correct_guards")

NO_MASK = ("No mask to clear", "info")
NO_SKELETON = ("No skeleton to clear", "info")


def clear(page, layer_id: str) -> list:
    """Clear ``layer_id`` with the page's Clear; the notices it sent."""
    page.set_active_layer(layer_id)
    notices = record(page.notice)
    page.clear_active_layer()
    settle(page)
    return notices


def no_step(page, layer_id: str) -> bool:
    return not page.layers[layer_id].can_undo() and not page.session.can_undo()


def write_empty(path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, np.zeros((H, W), np.uint8))
    return path


# ==================================================================================== mask
def test_mask_clear_without_a_prediction_changes_nothing(make_editor, project):
    page = make_editor(project.b, project.folder)

    notices = clear(page, MASK)

    assert notices == [NO_MASK] and no_step(page, MASK)
    assert not page.layers[MASK].is_dirty()


def test_mask_clear_on_an_empty_prediction_changes_nothing(make_editor, project):
    write_mask(project.mask("a"), np.zeros((H, W), np.uint8))
    page = make_editor(project.a, project.folder)

    notices = clear(page, MASK)

    assert notices == [NO_MASK] and no_step(page, MASK)


def test_mask_clear_on_an_empty_gt_mask_changes_nothing(make_editor, project):
    write_empty(project.gt_mask("b"))
    page = make_editor(project.b, project.folder)
    assert page.layers[MASK].status().label == "GT"

    notices = clear(page, MASK)

    assert notices == [NO_MASK] and no_step(page, MASK)
    assert not page.layers[MASK].is_dirty()


def test_mask_clear_on_a_plane_the_eraser_emptied_adds_no_step(make_editor, project):
    page = make_editor(project.a, project.folder)
    page.set_active_layer(MASK)
    layer = page.layers[MASK]
    layer.set_tool("eraser")
    layer.set_size(100)
    drag(page, (40, 50), (110, 50))
    assert not plane(page).any() and layer.is_dirty()
    steps = len(layer.document.history.undo_stack)

    notices = clear(page, MASK)

    assert notices == [NO_MASK] and len(layer.document.history.undo_stack) == steps
    assert layer.is_dirty()  # still unsaved: Save writes the empty GT mask
    assert page.save()
    assert os.path.isfile(project.gt_mask("a"))


def test_mask_clear_on_a_plane_with_pixels_clears_and_undoes(make_editor, project):
    page = make_editor(project.a, project.folder)
    before = plane(page).copy()
    assert before.any()

    notices = clear(page, MASK)

    assert notices == [(MaskLayer.CLEARED_NOTICE, "info")]
    assert not plane(page).any() and page.layers[MASK].is_dirty()
    page.undo()
    assert np.array_equal(plane(page), before)


# ================================================================================ skeleton
def test_skeleton_clear_without_a_skeleton_changes_nothing(make_editor, project):
    page = make_editor(project.b, project.folder)

    notices = clear(page, SKELETON)

    assert notices == [NO_SKELETON] and no_step(page, SKELETON)
    assert not page.layers[SKELETON].is_dirty()


def test_skeleton_clear_on_an_empty_prediction_changes_nothing(make_editor, project):
    write_empty(project.generated("a"))
    page = make_editor(project.a, project.folder)

    notices = clear(page, SKELETON)

    assert notices == [NO_SKELETON] and no_step(page, SKELETON)


def test_skeleton_clear_on_an_empty_gt_skeleton_changes_nothing(make_editor, project):
    write_empty(project.correction("a"))
    page = make_editor(project.a, project.folder)
    assert page.layers[SKELETON].status().label == "GT"

    notices = clear(page, SKELETON)

    assert notices == [NO_SKELETON] and no_step(page, SKELETON)
    assert not page.layers[SKELETON].is_dirty()


def test_skeleton_clear_on_a_skeleton_the_eraser_emptied_adds_no_step(make_editor, project):
    page = make_editor(project.a, project.folder)
    page.request_tool(SKELETON, "eraser")
    layer = page.layers[SKELETON]
    layer.set_size(80)
    drag(page, (5, 110), (100, 110), (195, 110))
    assert not raster(page).any() and layer.is_dirty()
    steps = len(layer.model._history.undo_stack)

    notices = clear(page, SKELETON)

    assert notices == [NO_SKELETON] and len(layer.model._history.undo_stack) == steps
    assert layer.is_dirty()  # still unsaved: Save writes the emptied skeleton
    assert page.save()
    assert os.path.isfile(project.correction("a"))


def test_skeleton_clear_on_a_skeleton_with_pixels_clears_and_undoes(make_editor, project):
    page = make_editor(project.a, project.folder)
    before = raster(page).copy()
    assert before.any()

    notices = clear(page, SKELETON)

    assert notices == [(SkeletonLayer.CLEARED_NOTICE, "info")]
    assert not raster(page).any() and page.layers[SKELETON].is_dirty()
    page.undo()
    assert np.array_equal(raster(page), before)
