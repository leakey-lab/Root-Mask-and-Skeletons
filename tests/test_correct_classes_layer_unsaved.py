"""When the class labels are unsaved (``ClassesLayer`` over ``ClassesDocument``, WG-T4).

Unsaved means the ids differ from the saved ones, whatever the history says: undoing every
stroke, erasing a stroke by hand or clearing back to the opened labels leaves nothing to save,
the same class counts elsewhere are still a change, and a class picked is never an edit. A
remap on open stays unsaved until it is saved or discarded; a lossy one (unknown classes sent to
Ignore) asks first, a lossless one (ids only) saves silently. Clear on all-background labels adds
no step and says so; saved all-background labels are "marked empty" (the page's Unmark), never
while painted, read-only or not shown. The check is cheap: counts first, one pixel pass when they
match. Ported from ``tests/test_gt_unsaved_state.py`` (deleted in WG-T11).
"""

from __future__ import annotations

import os
import statistics
import time

import numpy as np
import pytest
from PyQt6.QtCore import QPoint

from app.editing import planes
from app.gt import (
    Registry,
    Tracing,
    default_registry,
    read_tracing,
    registry_path,
    write_tracing,
)
from app.gui.correct.layer import (
    LayerStatus,
)
from app.gui.correct.session import SaveReport
from tests.correct_classes_layer_support import (
    NO_MODIFIER,
    H,
    W,
    file_bytes,
    ids,
    notices_of,
    open_layer,
    open_session,
    stroke,
    write_with_unknown_key,
)
from tests.correct_classes_layer_support import gt_project as gt_project  # fixture
from tests.correct_classes_layer_support import (
    gt_project_bad_registry as gt_project_bad_registry,  # fixture
)
from tests.correct_classes_layer_support import project as project  # fixture
from tests.correct_classes_layer_support import scene as scene  # fixture
from tests.correct_fakes import FakePrompts

STRICT = os.environ.get("SPROUTS_PERF_TESTS") == "1"
strict = pytest.mark.skipif(not STRICT, reason="wall-clock budget: set SPROUTS_PERF_TESTS=1")
NOTHING_TO_CLEAR = "No class labels to clear"
NOTHING_TO_CLEAR_UNSAVED = "No class labels to clear · Mark empty saves an empty file"

pytestmark = pytest.mark.usefixtures("no_prompts")


def test_a_lossy_remap_asks_before_it_is_saved_and_discard_keeps_the_file(scene, project):
    before = write_with_unknown_key(project.image)

    opened = open_session(scene, project.image, FakePrompts(unsaved=["discard"]))

    layer = opened.layer
    assert any(kind == "warn" and "hypha" in message for message, kind in opened.notices)
    assert layer.is_dirty() and layer.save_needs_prompt()
    assert layer.status().label == "Unsaved · remapped"
    assert opened.session.release() is True  # the normal prompt; Discard
    assert opened.prompts.asked == [("unsaved", None, "class labels", "plot_001.png")]
    assert file_bytes(project.labels) == before and not layer.is_dirty()


def test_saving_a_lossy_remap_writes_ignore_and_ends_the_prompt(scene, project):
    write_with_unknown_key(project.image)
    layer, _, _ = open_layer(scene, project.image)

    assert layer.save().ok

    assert (read_tracing(project.labels).ids[5:15, 5:15] == 255).all()
    assert not layer.save_needs_prompt() and layer.status().label != "Unsaved · remapped"


def test_a_lossless_remap_is_saved_without_a_prompt(scene, project):
    """Ids that differ only by number (same keys) are remapped and saved silently."""
    swapped = default_registry().to_dict()
    swapped["classes"][0]["id"], swapped["classes"][1]["id"] = 2, 1  # root=2, nodule=1
    plane = np.zeros((H, W), np.uint8)
    plane[5:15, 5:15] = 2  # root in the other project's ids
    write_tracing(project.labels, Tracing(plane), Registry.from_dict(swapped))

    opened = open_session(scene, project.image)

    assert not [n for n in opened.notices if n[1] == "warn"]
    assert opened.layer.is_dirty() and not opened.layer.save_needs_prompt()
    assert opened.layer.status() == LayerStatus("dirty")
    assert opened.session.save_dirty().ok and opened.prompts.asked == []
    assert (read_tracing(project.labels).ids[5:15, 5:15] == 1).all()


def test_undoing_every_stroke_leaves_nothing_to_save(scene, project):
    opened = open_session(scene, project.image)
    layer = opened.layer
    layer.set_active_class("root")
    stroke(layer, (10, 30), (70, 30), size=6)
    layer.set_active_class("nodule")
    stroke(layer, (40, 5), (40, 55), size=6)

    layer.undo()
    assert layer.is_dirty()  # the root stroke is still there
    layer.undo()

    assert not layer.is_dirty() and opened.dirty == [True, False]
    assert layer.status() == LayerStatus("none", "No class labels")  # as opened
    assert opened.session.release() is True and opened.prompts.asked == []
    layer.redo()
    assert layer.is_dirty() and opened.dirty == [True, False, True]


def test_undo_back_to_the_saved_labels_is_saved_again(scene, project):
    opened = open_session(scene, project.image)
    layer = opened.layer
    layer.set_active_class("root")
    stroke(layer, (10, 30), (70, 30), size=6)
    assert layer.save().ok
    layer.set_active_class("nodule")
    stroke(layer, (40, 5), (40, 55), size=6)
    opened.dirty.clear()

    layer.undo()
    assert not layer.is_dirty()
    assert layer.status() == LayerStatus("clean", "GT", "gt/plot_001.png")
    layer.redo()
    assert layer.is_dirty() and layer.status() == LayerStatus("dirty")
    layer.undo()
    assert not layer.is_dirty() and opened.dirty == [False, True, False]


def test_leaving_after_undoing_everything_asks_nothing_and_writes_nothing(scene, project):
    opened = open_session(scene, project.image)
    stroke(opened.layer, (10, 30), (70, 30))
    opened.layer.undo()

    assert opened.session.release() is True  # leaving: no prompt

    assert opened.prompts.asked == [] and not os.path.exists(project.labels)


def test_pgdn_after_undoing_everything_writes_nothing(scene, project):
    opened = open_session(scene, project.image)
    stroke(opened.layer, (10, 30), (70, 30))
    opened.layer.undo()

    assert opened.session.save_dirty() == SaveReport()  # PgDn's save: nothing is dirty

    assert not os.path.exists(project.labels)  # unchanged labels are not written


def test_a_stroke_that_changed_nothing_is_no_change(scene, project):
    layer, _, _ = open_layer(scene, project.image)
    layer.set_tool("eraser")  # erasing empty labels changes no pixel
    layer.set_size(6)
    layer.press(QPoint(10, 30), NO_MODIFIER)
    layer.move(QPoint(70, 30), True)
    assert layer.gesture_active()  # the stroke is not committed yet

    layer.release(QPoint(70, 30))

    assert not layer.is_dirty() and layer.status() == LayerStatus("none", "No class labels")


def test_a_click_outside_the_image_changes_nothing(scene, project):
    layer, _, _ = open_layer(scene, project.image)
    stroke(layer, (W + 40, H + 40))  # a dab entirely off the canvas
    assert not layer.is_dirty() and not layer.can_undo()


def test_clearing_back_to_the_opened_labels_is_no_change(scene, project):
    """Pixels decide, not the history: two actions later the labels are as opened."""
    layer, _, _ = open_layer(scene, project.image)
    stroke(layer, (10, 30), (70, 30))
    layer.clear()

    assert not layer.is_dirty() and len(layer.document.history.undo_stack) == 2
    layer.undo()  # the stroke is back
    assert layer.is_dirty()


def test_erasing_a_stroke_by_hand_is_saved_again(scene, project):
    layer, _, _ = open_layer(scene, project.image)
    assert layer.save().ok  # empty labels on disk
    layer.set_active_class("root")
    stroke(layer, (10, 30), (70, 30), size=6)
    assert layer.is_dirty()

    layer.set_tool("eraser")
    stroke(layer, (5, 30), (75, 30), size=20)

    assert not ids(layer).any() and not layer.is_dirty()
    assert layer.status() == LayerStatus("clean", "GT", "gt/plot_001.png")


def test_clearing_empty_labels_changes_nothing(scene, project):
    """Clear on an all-background plane: no patch, no undo step, and a notice that promises
    no file (Save writes nothing then: O2)."""
    opened = open_session(scene, project.image)
    assert opened.layer.clear() is True
    assert not opened.layer.is_dirty() and opened.dirty == []
    assert not opened.layer.can_undo() and not opened.session.can_undo()
    assert notices_of(opened.host) == [(NOTHING_TO_CLEAR_UNSAVED, "info")]


def test_clearing_saved_empty_labels_says_there_is_nothing_to_clear(scene, project):
    write_tracing(project.labels, Tracing(np.zeros((H, W), np.uint8)), default_registry())
    opened = open_session(scene, project.image)

    assert opened.layer.clear() is True

    assert not opened.layer.is_dirty() and not opened.layer.can_undo()
    assert notices_of(opened.host) == [(NOTHING_TO_CLEAR, "info")]


def test_clearing_labels_erased_by_hand_adds_no_step(scene, project):
    """Painted, then erased back to background: the plane is empty, Clear adds nothing."""
    layer, host, _ = open_layer(scene, project.image)
    stroke(layer, (10, 30), (70, 30), size=6)
    layer.set_tool("eraser")
    stroke(layer, (5, 30), (75, 30), size=20)
    steps = len(layer.document.history.undo_stack)

    assert layer.clear() is True

    assert len(layer.document.history.undo_stack) == steps
    assert notices_of(host)[-1] == (NOTHING_TO_CLEAR_UNSAVED, "info")


def test_clearing_labelled_pixels_clears_with_the_cleared_notice(scene, project):
    layer, host, _ = open_layer(scene, project.image)
    stroke(layer, (10, 30), (70, 30), size=6)

    assert layer.clear() is True

    assert not ids(layer).any() and len(layer.document.history.undo_stack) == 2
    assert notices_of(host)[-1] == (layer.CLEARED_NOTICE, "info")


def test_saved_empty_labels_are_marked_empty(scene, project):
    layer, _, _ = open_layer(scene, project.image)
    assert not layer.document.marked_empty()  # no file
    assert layer.save().ok  # an empty file (Mark empty's save)

    assert layer.document.marked_empty()
    layer.open(project.image, (W, H), project.folder)
    assert layer.document.marked_empty()


def test_labels_with_a_labelled_pixel_are_not_marked_empty(scene, project):
    plane = np.zeros((H, W), np.uint8)
    plane[5, 5] = 1
    write_tracing(project.labels, Tracing(plane), default_registry())

    layer, _, _ = open_layer(scene, project.image)

    assert not layer.document.marked_empty()


def test_unsaved_paint_over_empty_labels_is_not_marked_empty(scene, project):
    write_tracing(project.labels, Tracing(np.zeros((H, W), np.uint8)), default_registry())
    layer, _, _ = open_layer(scene, project.image)
    stroke(layer, (10, 30), (70, 30), size=6)

    assert not layer.document.marked_empty()
    layer.undo()
    assert layer.document.marked_empty()


def test_empty_labels_under_an_invalid_registry_are_not_marked_empty(scene, project):
    write_tracing(project.labels, Tracing(np.zeros((H, W), np.uint8)), default_registry())
    with open(registry_path(project.folder), "w", encoding="utf-8") as fh:
        fh.write("{")

    layer, _, _ = open_layer(scene, project.image)

    assert not layer.document.marked_empty()


@pytest.mark.parametrize("kind", ["unreadable", "size"])
def test_a_labels_file_not_shown_is_not_marked_empty(scene, project, kind):
    os.makedirs(os.path.dirname(project.labels), exist_ok=True)
    if kind == "unreadable":
        with open(project.labels, "wb") as fh:
            fh.write(b"not a png")
    else:
        write_tracing(project.labels, Tracing(np.zeros((10, 10), np.uint8)), default_registry())

    layer, _, _ = open_layer(scene, project.image)

    assert not layer.document.marked_empty()


def test_the_document_tells_a_labels_file_changed_since_it_was_read(scene, project):
    write_tracing(project.labels, Tracing(np.zeros((H, W), np.uint8)), default_registry())
    layer, _, _ = open_layer(scene, project.image)
    assert not layer.document.labels_changed_on_disk()

    plane = np.zeros((H, W), np.uint8)
    plane[0:20, 0:20] = 1  # another size on disk: the stamp changes
    write_tracing(project.labels, Tracing(plane), default_registry())

    assert layer.document.labels_changed_on_disk()
    layer.open(project.image, (W, H), project.folder)
    assert not layer.document.labels_changed_on_disk()
    os.remove(project.labels)
    assert layer.document.labels_changed_on_disk()


def test_picking_classes_changes_nothing(scene, project):
    opened = open_session(scene, project.image)

    for key in ("nodule", "ignore", "nodule"):
        opened.layer.set_active_class(key)

    assert not opened.layer.is_dirty() and opened.dirty == []


def test_a_remapped_file_stays_unsaved_after_undoing_every_stroke(scene, project):
    write_with_unknown_key(project.image)
    opened = open_session(scene, project.image, FakePrompts(unsaved=["discard"]))
    layer = opened.layer
    assert layer.is_dirty() and layer.status().label == "Unsaved · remapped"

    stroke(layer, (10, 40), (70, 40))
    layer.undo()

    assert layer.is_dirty() and layer.status().label == "Unsaved · remapped"
    assert opened.session.release() is True  # Discard: the file stays as it is
    assert not layer.is_dirty()
    stroke(layer, (10, 40), (70, 40))
    layer.undo()
    assert not layer.is_dirty()  # back at the labels as Discard left them


def test_a_class_count_change_needs_no_pixel_compare(scene, project, monkeypatch):
    """The document keeps pixel counts per class: while they differ from the saved ones the
    labels are unsaved without a full pass; the pixels are compared only when they match."""
    calls = []
    real = planes.planes_equal
    monkeypatch.setattr(planes, "planes_equal", lambda a, b: calls.append(1) or real(a, b))
    layer, _, _ = open_layer(scene, project.image)
    stroke(layer, (10, 30), (70, 30))
    stroke(layer, (10, 40), (70, 40))
    layer.undo()
    assert layer.is_dirty() and calls == []

    layer.undo()  # every class count is back: one pass decides

    assert not layer.is_dirty() and calls == [1]


def test_the_same_class_counts_elsewhere_are_still_a_change(scene, project):
    layer, _, _ = open_layer(scene, project.image)
    layer.set_active_class("root")
    stroke(layer, (15, 15))  # one dab
    assert layer.save().ok
    saved = layer.document.counts()
    layer.set_tool("eraser")
    stroke(layer, (15, 15), size=30)  # erased
    layer.set_tool("brush")
    layer.set_size(8)

    stroke(layer, (50, 40))  # the same dab elsewhere: the same pixel counts

    assert layer.document.counts() == saved and saved["root"] > 0
    assert layer.is_dirty()


def _big_layer(scene, project):
    layer, _, _ = open_layer(scene, project.image, (2048, 1536))
    layer.set_active_class("root")
    stroke(layer, (100, 100), (1900, 1400), size=40)
    layer.undo()  # equal planes: the comparison runs to the last pixel
    return layer


def _median_check_ms(layer, count=30) -> float:
    times = []
    for _ in range(count):
        layer.document.touch()  # a new revision: the next check compares again
        t0 = time.perf_counter()
        layer.is_dirty()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


@pytest.mark.wallclock
def test_the_check_is_cheap_on_a_2048_plane(scene, project):
    """About 0.05 ms (cv2.norm over 3.1 MB); 2 ms leaves 40x headroom on a busy machine."""
    layer = _big_layer(scene, project)
    assert not layer.is_dirty()
    assert _median_check_ms(layer) < 2.0


@pytest.mark.wallclock
@strict
def test_the_check_is_within_0_25_ms_on_a_2048_plane(scene, project):
    assert _median_check_ms(_big_layer(scene, project), 200) <= 0.25
